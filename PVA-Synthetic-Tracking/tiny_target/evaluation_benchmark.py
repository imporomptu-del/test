"""Phase 11 controlled end-to-end accuracy and execution-mode benchmark."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import gc
import hashlib
import json
import math
from pathlib import Path
import resource
import statistics
import sys
import time
from typing import Any, Callable, Sequence

import numpy as np

from .detection import (
    CandidateExtractionConfig,
    CandidateExtractor,
    MatchedFilterConfig,
    PsfMatchedFilter,
    ReferenceShiftAndStack,
    ReferenceSyntheticWindow,
    SyntheticTrackingConfig,
)
from .evaluation import (
    SyntheticInjectionSpec,
    SyntheticInjector,
    SyntheticTarget,
    ThresholdAccumulator,
    latency_summary,
    load_dataset_inventory,
    match_candidates,
)
from .preprocessing import BackgroundConfig, NoiseConfig, RobustPreprocessor
from .stabilization import StabilizedFrame
from .telemetry import run_identity, write_json_exclusive
from .tracking import KalmanTrackManager, KalmanTrackingConfig
from .types import Frame, TimestampSource


REPOSITORY = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "seaqr.tiny-target.end-to-end-evaluation.v1"
DEFAULT_INVENTORY = REPOSITORY / "configs" / "evaluation" / "phase11_dataset_inventory.json"
THRESHOLDS = (4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0)


@dataclass(frozen=True, slots=True)
class EvaluationModeConfig:
    mode: str = "correctness"
    input_rate_hz: float = 10.0
    queue_capacity_frames: int = 0
    soak_repetitions: int = 1

    def __post_init__(self) -> None:
        if self.mode not in {"correctness", "throughput", "real_time", "soak"}:
            raise ValueError("mode must be correctness, throughput, real_time, or soak")
        if not math.isfinite(self.input_rate_hz) or self.input_rate_hz <= 0:
            raise ValueError("input_rate_hz must be finite and positive")
        if self.queue_capacity_frames != 0:
            raise ValueError(
                "the current synchronous pull runner has an application queue capacity of zero"
            )
        if self.soak_repetitions <= 0:
            raise ValueError("soak_repetitions must be positive")
        if self.mode != "soak" and self.soak_repetitions != 1:
            raise ValueError("soak_repetitions may exceed one only in soak mode")


def _identity() -> dict[str, str]:
    files = (
        Path(__file__),
        Path(__file__).parent / "evaluation" / "core.py",
        Path(__file__).parent / "preprocessing" / "model.py",
        Path(__file__).parent / "detection" / "matched_filter.py",
        Path(__file__).parent / "detection" / "synthetic_reference.py",
        Path(__file__).parent / "detection" / "candidates.py",
        Path(__file__).parent / "tracking" / "kalman.py",
    )
    return {
        str(path.relative_to(REPOSITORY)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in files
    }


def _tracking_config() -> KalmanTrackingConfig:
    return KalmanTrackingConfig(
        position_measurement_sigma_px=1.0,
        velocity_measurement_sigma_px_s=1.0,
        acceleration_process_sigma_px_s2=1.0,
        initial_position_sigma_px=2.0,
        initial_velocity_sigma_px_s=2.0,
        mahalanobis_gate_squared=16.0,
        maximum_position_residual_px=5.0,
        maximum_velocity_residual_px_s=3.0,
        confirmation_independent_hits=2,
        max_missed_windows=1,
        maximum_timestamp_gap_s=2.0,
        measurement_noise_source="synthetic_characterization",
        max_active_tracks=128,
    )


def _candidate_config(threshold: float, window_frames: int) -> CandidateExtractionConfig:
    return CandidateExtractionConfig(
        score_threshold_snr=threshold,
        minimum_support_frames=window_frames,
        local_maximum_radius_px=1,
        spatial_nms_radius_px=2.5,
        velocity_nms_radius_px_s=1.5,
        border_margin_px=4,
        invalid_margin_px=1,
        diagnostic_distance_limit_px=16,
        pre_nms_candidate_limit=1024,
        max_candidates_per_window=128,
    )


def _frame_image(seed: int, frame_index: int, shape: tuple[int, int]) -> np.ndarray:
    rng = np.random.default_rng(seed + frame_index * 1_000_003)
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    background = 1000 + 18 * np.sin(xx / 5.3) + 13 * np.cos(yy / 4.1)
    background += 0.1 * frame_index
    return np.clip(np.rint(background + rng.normal(0, 4, shape)), 0, 65535).astype(
        np.uint16
    )


def _pipeline_components(window_frames: int) -> tuple[Any, Any, Any]:
    preprocessor = RobustPreprocessor(
        BackgroundConfig(
            method="robust_running",
            warmup_frames=4,
            history_frames=8,
            minimum_history_samples=4,
            update_rate=0.12,
            outlier_clip_sigma=4,
            update_exclusion_sigma=3,
            global_change_median_sigma=4,
            global_change_robust_scale=2,
        ),
        NoiseConfig(
            method="robust_ewma",
            sigma_floor=2.0,
            mad_scale=1.4826,
            mask_saturated=True,
            saturation_value=65535,
        ),
    )
    matched_filter = PsfMatchedFilter(
        MatchedFilterConfig(
            source="gaussian",
            gaussian_sigma_px=0.8,
            radius_px=3,
            phases_per_axis=2,
            normalize="snr_l2",
            polarity="bright",
            backend="numpy_reference",
            minimum_valid_fraction=1,
        )
    )
    synthetic_window = ReferenceSyntheticWindow(
        ReferenceShiftAndStack(
            SyntheticTrackingConfig(
                window_frames=window_frames,
                window_stride_frames=window_frames,
                vx_min_px_s=-2,
                vx_max_px_s=2,
                vy_min_px_s=-2,
                vy_max_px_s=2,
                velocity_step_px_s=1,
                min_valid_fraction=1,
                tile_rows=16,
            )
        )
    )
    return preprocessor, matched_filter, synthetic_window


def _process_sequence(
    *,
    seed: int,
    window_frames: int,
    target: SyntheticTarget | None,
    mode: EvaluationModeConfig,
    window_observer: Callable[[Any], None] | None = None,
) -> dict[str, Any]:
    shape = (48, 64)
    frame_count = 24
    preprocessor, matched_filter, synthetic_window = _pipeline_components(window_frames)
    injector = (
        SyntheticInjector(SyntheticInjectionSpec(75, 0.8, 3, (target,)))
        if target is not None
        else None
    )
    windows = []
    frame_latencies = []
    stage_times: dict[str, list[float]] = {
        "source_and_injection": [],
        "preprocessing": [],
        "matched_filter": [],
        "synthetic_tracking": [],
        "candidate_and_tracking": [],
    }
    deadline_misses = 0
    run_started = time.perf_counter()
    for index in range(frame_count):
        if mode.mode == "real_time":
            deadline = run_started + index / mode.input_rate_hz
            remaining = deadline - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)
            else:
                deadline_misses += int(index > 0)
        frame_started = time.perf_counter_ns()
        stage_started = time.perf_counter_ns()
        raw = Frame(
            image=_frame_image(seed, index, shape),
            timestamp_ns=index * 100_000_000,
            frame_index=index,
            source_id=f"controlled:{seed}",
            bit_depth=16,
            timestamp_source=TimestampSource.MANIFEST,
        )
        if injector is not None:
            raw = injector.inject(raw)
        stage_times["source_and_injection"].append(
            (time.perf_counter_ns() - stage_started) / 1e6
        )
        valid = np.ones(shape, bool)
        stabilized_frame = Frame(
            image=raw.image.astype(np.float32),
            timestamp_ns=raw.timestamp_ns,
            frame_index=raw.frame_index,
            source_id=raw.source_id,
            bit_depth=raw.bit_depth,
            timestamp_source=raw.timestamp_source,
            valid_mask=valid,
        )
        stabilized = StabilizedFrame(
            frame=stabilized_frame,
            reference_frame_index=0,
            segment_index=0,
            source_to_reference_matrix=np.eye(3),
            interpolation="identity",
            backend="controlled_identity",
            resampling_count=0,
            metrics={},
            timings_ms={},
        )
        stage_started = time.perf_counter_ns()
        residual = preprocessor.process(stabilized)
        stage_times["preprocessing"].append(
            (time.perf_counter_ns() - stage_started) / 1e6
        )
        stage_started = time.perf_counter_ns()
        matched = matched_filter.process(residual)
        stage_times["matched_filter"].append(
            (time.perf_counter_ns() - stage_started) / 1e6
        )
        stage_started = time.perf_counter_ns()
        window = synthetic_window.update(matched)
        synthetic_ms = (time.perf_counter_ns() - stage_started) / 1e6
        stage_times["synthetic_tracking"].append(synthetic_ms)
        if window is not None:
            windows.append(window)
            observer_started = time.perf_counter_ns()
            if window_observer is not None:
                window_observer(window)
            stage_times["candidate_and_tracking"].append(
                (time.perf_counter_ns() - observer_started) / 1e6
            )
        frame_latencies.append((time.perf_counter_ns() - frame_started) / 1e6)
    return {
        "shape": shape,
        "frame_count": frame_count,
        "duration_s": (frame_count - 1) * 0.1,
        "windows": windows,
        "injection_records": injector.records if injector is not None else [],
        "frame_latencies_ms": frame_latencies,
        "stage_times_ms": stage_times,
        "deadline_misses": deadline_misses,
        "wall_time_s": time.perf_counter() - run_started,
    }


def _truth(target: SyntheticTarget, timestamp_ns: int) -> dict[str, Any]:
    return {
        "target_id": target.target_id,
        "position_xy_px": list(target.position_at(timestamp_ns)),
        "velocity_xy_px_s": list(target.velocity_xy_px_s),
        "flux_dn": target.flux_dn,
    }


def _seconds_summary(values: Sequence[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    array = np.asarray(values, np.float64)
    return {
        "count": int(array.size),
        "median_s": float(np.median(array)),
        "p90_s": float(np.percentile(array, 90)),
        "p95_s": float(np.percentile(array, 95)),
        "p99_s": float(np.percentile(array, 99)),
        "maximum_s": float(np.max(array)),
    }


def _target_cases() -> list[SyntheticTarget]:
    values = (
        (60, (15.25, 15.25), (0, 0)),
        (100, (47.75, 15.25), (1, 0)),
        (160, (16.25, 32.75), (1, 1)),
        (260, (46.75, 32.75), (-2, 1)),
    )
    return [
        SyntheticTarget(
            target_id=f"flux_{flux:g}",
            flux_dn=flux,
            reference_timestamp_ns=0,
            reference_position_xy_px=position,
            velocity_xy_px_s=velocity,
            first_frame_index=4,
            last_frame_index=23,
        )
        for flux, position, velocity in values
    ]


def _grouped_detection(
    cases: Sequence[dict[str, Any]],
    *,
    integration_length: int,
    threshold: float,
) -> dict[str, dict[str, float]]:
    grouped: dict[str, dict[str, list[int]]] = {
        "subpixel_phase_xy": {},
        "speed_px_s": {},
        "direction_deg": {},
        "field_quadrant": {},
    }
    for case in cases:
        if case["integration_window_frames"] != integration_length:
            continue
        target = case["target"]
        position = target["reference_position_xy_px"]
        velocity = target["velocity_xy_px_s"]
        phase = (position[0] - round(position[0]), position[1] - round(position[1]))
        speed = math.hypot(*velocity)
        direction = math.degrees(math.atan2(velocity[1], velocity[0])) if speed else 0.0
        quadrant = (
            ("right" if position[0] >= 32 else "left")
            + "_"
            + ("bottom" if position[1] >= 24 else "top")
        )
        result = case["threshold_results"][f"{threshold:g}"]
        detected = int(result["detected_window_count"])
        total = int(result["window_count"])
        keys = {
            "subpixel_phase_xy": f"{phase[0]:g},{phase[1]:g}",
            "speed_px_s": f"{speed:g}",
            "direction_deg": f"{direction:g}",
            "field_quadrant": quadrant,
        }
        for group_name, key in keys.items():
            counts = grouped[group_name].setdefault(key, [0, 0])
            counts[0] += detected
            counts[1] += total
    return {
        group_name: {
            key: detected / total
            for key, (detected, total) in sorted(values.items())
        }
        for group_name, values in grouped.items()
    }


def _evaluate_once(mode: EvaluationModeConfig, seed: int) -> dict[str, Any]:
    evaluation_started = time.perf_counter()
    accumulators = {
        (threshold, length): ThresholdAccumulator(threshold, unmatched_are_false=True)
        for threshold in THRESHOLDS
        for length in (2, 4, 8)
    }
    noise_accumulators = {
        (threshold, length): ThresholdAccumulator(threshold, unmatched_are_false=True)
        for threshold in THRESHOLDS
        for length in (2, 4, 8)
    }
    confirmed_trials = {(threshold, length): 0 for threshold in THRESHOLDS for length in (2, 4, 8)}
    confirmation_latencies: dict[tuple[float, int], list[float]] = {
        key: [] for key in confirmed_trials
    }
    false_confirmed = {(threshold, length): 0 for threshold in THRESHOLDS for length in (2, 4, 8)}
    frame_latencies: list[float] = []
    stage_times: dict[str, list[float]] = {}
    candidate_times: list[float] = []
    tracking_times: list[float] = []
    total_frames = 0
    total_wall_s = 0.0
    deadline_misses = 0
    injection_record_count = 0
    target_case_summaries = []
    for length in (2, 4, 8):
        for case_index, target in enumerate(_target_cases()):
            trackers = {
                threshold: KalmanTrackManager(_tracking_config())
                for threshold in THRESHOLDS
            }
            ever_confirmed = {threshold: False for threshold in THRESHOLDS}
            first_confirmation = {threshold: None for threshold in THRESHOLDS}
            detected_windows = {threshold: 0 for threshold in THRESHOLDS}

            def observe_target(window: Any) -> None:
                truth = [_truth(target, window.reference_timestamp_ns)]
                for threshold in THRESHOLDS:
                    started = time.perf_counter_ns()
                    candidates = CandidateExtractor(
                        _candidate_config(threshold, length)
                    ).extract(window)
                    candidate_times.append((time.perf_counter_ns() - started) / 1e6)
                    matched = match_candidates(
                        candidates.candidates,
                        truth,
                        maximum_position_error_px=2,
                        maximum_velocity_error_px_s=1.5,
                    )
                    accumulators[(threshold, length)].add(truth, matched)
                    detected_windows[threshold] += int(
                        matched["true_positive_count"] > 0
                    )
                    started = time.perf_counter_ns()
                    tracks = trackers[threshold].update(candidates)
                    tracking_times.append((time.perf_counter_ns() - started) / 1e6)
                    for track in tracks.tracks:
                        if track.lifecycle_state != "confirmed":
                            continue
                        state = track.state_xy_vx_vy
                        position_error = np.linalg.norm(
                            np.asarray(state[:2]) - truth[0]["position_xy_px"]
                        )
                        velocity_error = np.linalg.norm(
                            np.asarray(state[2:]) - truth[0]["velocity_xy_px_s"]
                        )
                        if position_error <= 2 and velocity_error <= 1.5:
                            ever_confirmed[threshold] = True
                            if first_confirmation[threshold] is None:
                                first_confirmation[threshold] = (
                                    track.confirmation_latency_s
                                )

            sequence = _process_sequence(
                seed=seed + case_index,
                window_frames=length,
                target=target,
                mode=mode,
                window_observer=observe_target,
            )
            frame_latencies.extend(sequence["frame_latencies_ms"])
            for name, values in sequence["stage_times_ms"].items():
                stage_times.setdefault(name, []).extend(values)
            total_frames += sequence["frame_count"]
            total_wall_s += sequence["wall_time_s"]
            deadline_misses += sequence["deadline_misses"]
            injection_record_count += len(sequence["injection_records"])
            per_threshold = {}
            for threshold in THRESHOLDS:
                confirmed_trials[(threshold, length)] += int(
                    ever_confirmed[threshold]
                )
                if first_confirmation[threshold] is not None:
                    confirmation_latencies[(threshold, length)].append(
                        first_confirmation[threshold]
                    )
                per_threshold[f"{threshold:g}"] = {
                    "detected_window_count": detected_windows[threshold],
                    "window_count": len(sequence["windows"]),
                    "confirmed": ever_confirmed[threshold],
                }
            target_case_summaries.append(
                {
                    "target": asdict(target),
                    "integration_window_frames": length,
                    "threshold_results": per_threshold,
                }
            )
        for noise_index in range(4):
            trackers = {
                threshold: KalmanTrackManager(_tracking_config())
                for threshold in THRESHOLDS
            }
            confirmed_ids = {threshold: set() for threshold in THRESHOLDS}

            def observe_noise(window: Any) -> None:
                for threshold in THRESHOLDS:
                    started = time.perf_counter_ns()
                    candidates = CandidateExtractor(
                        _candidate_config(threshold, length)
                    ).extract(window)
                    candidate_times.append((time.perf_counter_ns() - started) / 1e6)
                    matched = match_candidates(
                        candidates.candidates,
                        [],
                        maximum_position_error_px=2,
                        maximum_velocity_error_px_s=1.5,
                    )
                    noise_accumulators[(threshold, length)].add([], matched)
                    started = time.perf_counter_ns()
                    tracks = trackers[threshold].update(candidates)
                    tracking_times.append((time.perf_counter_ns() - started) / 1e6)
                    confirmed_ids[threshold].update(
                        track.track_id
                        for track in tracks.tracks
                        if track.lifecycle_state in {"confirmed", "coasted"}
                    )

            sequence = _process_sequence(
                seed=seed + 100 + noise_index,
                window_frames=length,
                target=None,
                mode=mode,
                window_observer=observe_noise,
            )
            frame_latencies.extend(sequence["frame_latencies_ms"])
            for name, values in sequence["stage_times_ms"].items():
                stage_times.setdefault(name, []).extend(values)
            total_frames += sequence["frame_count"]
            total_wall_s += sequence["wall_time_s"]
            deadline_misses += sequence["deadline_misses"]
            for threshold in THRESHOLDS:
                false_confirmed[(threshold, length)] += len(
                    confirmed_ids[threshold]
                )
    curves = []
    duration_per_sequence = 2.3
    for length in (2, 4, 8):
        for threshold in THRESHOLDS:
            detection = accumulators[(threshold, length)].summary(
                evaluated_frame_count=4 * 24,
                duration_s=4 * duration_per_sequence,
                frame_shape=(48, 64),
            )
            noise = noise_accumulators[(threshold, length)].summary(
                evaluated_frame_count=4 * 24,
                duration_s=4 * duration_per_sequence,
                frame_shape=(48, 64),
            )
            key = (threshold, length)
            curves.append(
                {
                    "integration_window_frames": length,
                    "threshold_snr": threshold,
                    "probability_of_detection": detection["probability_of_detection"],
                    "probability_of_detection_by_flux_dn": detection[
                        "probability_of_detection_by_flux_dn"
                    ],
                    "probability_of_detection_by_condition": _grouped_detection(
                        target_case_summaries,
                        integration_length=length,
                        threshold=threshold,
                    ),
                    "false_alarms": noise["false_alarms"],
                    "precision": (
                        detection["true_positive_count"]
                        / (
                            detection["true_positive_count"]
                            + noise["unmatched_candidate_count"]
                        )
                        if detection["true_positive_count"]
                        + noise["unmatched_candidate_count"]
                        else None
                    ),
                    "recall": detection["recall"],
                    "localization_error_px": detection["localization_error_px"],
                    "velocity_error_px_s": detection["velocity_error_px_s"],
                    "confirmed_track_probability": confirmed_trials[key] / 4,
                    "confirmation_latency_s": _seconds_summary(
                        confirmation_latencies[key]
                    ),
                    "false_confirmed_track_count": false_confirmed[key],
                }
            )
    evaluation_elapsed_s = time.perf_counter() - evaluation_started
    return {
        "curve_points": curves,
        "target_cases": target_case_summaries,
        "performance": {
            "input_frames": total_frames,
            "completed_output_frames": total_frames,
            "dropped_frames": 0,
            "maximum_application_queue_depth_frames": 0,
            "queue_policy": "synchronous_pull_backpressure_no_application_queue",
            "deadline_misses": deadline_misses,
            "wall_time_s": evaluation_elapsed_s,
            "sequence_processing_wall_time_s": total_wall_s,
            "input_frames_per_second": total_frames / evaluation_elapsed_s,
            "completed_output_frames_per_second": total_frames / evaluation_elapsed_s,
            "end_to_end_frame_latency": latency_summary(frame_latencies),
            "cold_start_first_frame_ms": frame_latencies[0],
            "steady_state_frame_latency": latency_summary(frame_latencies[1:]),
            "stages": {
                name: latency_summary(values) for name, values in stage_times.items()
            }
            | {
                "candidate_threshold_passes": latency_summary(candidate_times),
                "temporal_tracking": latency_summary(tracking_times),
            },
            "host_device_transfer_bytes": 0,
            "backend_note": "controlled CPU reference; hardware metrics require Jetson RAW16 report",
        },
        "stabilization_metrics": {
            "controlled_transform": "identity",
            "success_fraction": 1.0,
            "input_background_motion_px": 0.0,
            "residual_background_motion_px": 0.0,
        },
        "injection_record_count": injection_record_count,
    }


def run_benchmark(
    *,
    mode: str = "correctness",
    seed: int = 75,
    soak_repetitions: int = 1,
    input_rate_hz: float = 10.0,
    inventory_path: str | Path = DEFAULT_INVENTORY,
) -> dict[str, Any]:
    settings = EvaluationModeConfig(
        mode=mode,
        input_rate_hz=input_rate_hz,
        soak_repetitions=soak_repetitions,
    )
    inventory = load_dataset_inventory(inventory_path)
    before_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    started = time.perf_counter_ns()
    primary: dict[str, Any] | None = None
    completed_repetitions = 0
    bounded_queue_repetitions = 0
    repetition_peak_rss = []
    repetition_curve_sha256 = []
    for repetition in range(settings.soak_repetitions):
        result = _evaluate_once(settings, seed)
        repetition_curve_sha256.append(
            hashlib.sha256(
                json.dumps(
                    result["curve_points"],
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
        )
        completed_repetitions += 1
        bounded_queue_repetitions += int(
            result["performance"]["maximum_application_queue_depth_frames"] == 0
        )
        if primary is None:
            primary = result
        else:
            del result
        gc.collect()
        repetition_peak_rss.append(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        )
    total_ms = (time.perf_counter_ns() - started) / 1e6
    after_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_stability_tolerance = 1024 * 1024 if sys.platform == "darwin" else 1024
    if primary is None:  # guarded by EvaluationModeConfig, retained for type safety
        raise RuntimeError("evaluation completed without a repetition")
    report = {
        "schema_version": SCHEMA_VERSION,
        "run": run_identity(REPOSITORY),
        "implementation_sha256": _identity(),
        "dataset_inventory": inventory.to_dict(),
        "mode": asdict(settings),
        "configuration": {
            "random_seed": seed,
            "thresholds_snr": list(THRESHOLDS),
            "integration_window_frames": [2, 4, 8],
            "frame_shape": [48, 64],
            "frames_per_sequence": 24,
            "frame_rate_hz": 10,
            "target_flux_dn": [60, 100, 160, 260],
            "injection_stage": "decoded_source_before_motion_stabilization_preprocessing",
            "controlled_stabilization": "known_identity",
        },
        "accuracy_curve_points": primary["curve_points"],
        "target_cases": primary["target_cases"] if mode == "correctness" else None,
        "performance": primary["performance"],
        "soak": {
            "repetitions": completed_repetitions,
            "total_duration_ms": total_ms,
            "maximum_resident_set_size_before": before_rss,
            "maximum_resident_set_size_after": after_rss,
            "maximum_resident_set_size_unit": (
                "bytes_on_macos_kibibytes_on_linux_per_getrusage"
            ),
            "memory_growth_raw_ru_maxrss": after_rss - before_rss,
            "peak_rss_after_each_repetition": repetition_peak_rss,
            "peak_rss_stability_tolerance": rss_stability_tolerance,
            "curve_sha256_by_repetition": repetition_curve_sha256,
            "deterministic_curve_replay_all_repetitions": (
                len(set(repetition_curve_sha256)) == 1
            ),
            "all_repetitions_completed": (
                completed_repetitions == settings.soak_repetitions
            ),
            "bounded_queue_all_repetitions": (
                bounded_queue_repetitions == settings.soak_repetitions
            ),
            "peak_rss_stable_after_first_repetition": (
                max(repetition_peak_rss[1:], default=repetition_peak_rss[0])
                <= repetition_peak_rss[0] + rss_stability_tolerance
            ),
        },
        "limitations": [
            "Controlled accuracy uses known identity stabilization and CPU reference integration.",
            "Generated Gaussian noise cannot establish a real-camera false-alarm rate.",
            "The current inventory has no labeled real targets or stress-scene collection.",
            "Synchronous recorded-input pacing does not model live-camera driver drops.",
            "Power, clocks, temperatures, PVA/GPU utilization, and device memory require the Jetson hardware report.",
        ],
    }
    if mode == "correctness":
        repeat = _evaluate_once(settings, seed)
        report["correctness"] = {
            "deterministic_curve_replay_equal": (
                repeat["curve_points"] == primary["curve_points"]
            ),
            "debug_assertions_enabled": __debug__,
        }
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("correctness", "throughput", "real_time", "soak"),
        default="correctness",
    )
    parser.add_argument("--seed", type=int, default=75)
    parser.add_argument("--input-rate-hz", type=float, default=10)
    parser.add_argument("--soak-repetitions", type=int, default=1)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = run_benchmark(
        mode=args.mode,
        seed=args.seed,
        soak_repetitions=args.soak_repetitions,
        input_rate_hz=args.input_rate_hz,
        inventory_path=args.inventory,
    )
    if args.output is None:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        output = write_json_exclusive(args.output, report)
        print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
