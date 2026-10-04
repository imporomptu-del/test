"""Recorded PVA correspondences and robust global camera-motion estimation."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import statistics
import time
from typing import Any, Iterator, Sequence

import numpy as np

from .config import ConfigError, load_config
from .detection import (
    CandidateExtractionConfig,
    CandidateExtractionError,
    CandidateExtractor,
    CandidateRankingSurface,
    CudaShiftAndStack,
    MatchedFilterConfig,
    MatchedFilterError,
    PsfMatchedFilter,
    ReferenceShiftAndStack,
    ReferenceSyntheticWindow,
    SyntheticTrackingConfig,
    SyntheticTrackingError,
)
from .frame_source import FrameSourceError, source_from_config
from .evaluation import (
    EvaluationError,
    SyntheticInjector,
    load_injection_spec,
    transformed_target_truth,
)
from .motion import (
    ComposedMotionState,
    GlobalMotionConfig,
    GlobalMotionTracker,
    PvaMotionConfig,
    PvaMotionError,
    PvaPyrLkMotionEstimator,
    fit_global_motion,
)
from .preprocessing import (
    BackgroundConfig,
    NoiseConfig,
    PreprocessingError,
    RobustPreprocessor,
)
from .preprocessing.model import load_bad_pixel_map
from .stabilization import (
    FullResolutionStabilizer,
    StabilizationConfig,
    StabilizationError,
    ValidMaskWindow,
    alignment_improvement_metrics,
)
from .telemetry import file_identity, run_identity, write_json_exclusive
from .tracking import (
    KalmanTrackManager,
    KalmanTrackingConfig,
    TemporalTrackingError,
)
from .types import Discontinuity, Frame


REPOSITORY = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "seaqr.tiny-target.motion.v10"


def _implementation_identity() -> dict[str, str]:
    files = (
        Path(__file__),
        Path(__file__).with_name("frame_source.py"),
        Path(__file__).parent / "motion" / "geometry.py",
        Path(__file__).parent / "motion" / "global_motion.py",
        Path(__file__).parent / "motion" / "pva_pyrlk.py",
        Path(__file__).parent / "motion" / "types.py",
        Path(__file__).parent / "stabilization" / "types.py",
        Path(__file__).parent / "stabilization" / "warp.py",
        Path(__file__).parent / "preprocessing" / "types.py",
        Path(__file__).parent / "preprocessing" / "model.py",
        Path(__file__).parent / "detection" / "types.py",
        Path(__file__).parent / "detection" / "matched_filter.py",
        Path(__file__).parent / "detection" / "synthetic_types.py",
        Path(__file__).parent / "detection" / "synthetic_reference.py",
        Path(__file__).parent / "detection" / "synthetic_cuda.py",
        Path(__file__).parent / "detection" / "cuda" / "synthetic_tracking.cu",
        Path(__file__).parent / "detection" / "candidates.py",
        Path(__file__).parent / "tracking" / "kalman.py",
        Path(__file__).parent / "evaluation" / "core.py",
    )
    return {
        str(path.relative_to(REPOSITORY)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in files
    }


def _close_iterator(iterator: Iterator[Frame]) -> None:
    close = getattr(iterator, "close", None)
    if close is not None:
        close()


def estimate_config(
    config_path: str | Path,
    *,
    max_pairs: int | None = None,
    max_frames_override: int | None = None,
    input_path_override: str | Path | None = None,
    timestamp_csv_override: str | Path | None = None,
    minimum_grid_coverage_override: float | None = None,
    max_candidates_per_cell_override: int | None = None,
    track_reservation_position_radius_px_override: float | None = None,
    track_reservation_velocity_radius_px_s_override: float | None = None,
    track_reservation_minimum_mean_speed_px_s_override: float | None = None,
    max_track_reservations_per_window_override: int | None = None,
    velocity_grid_override: Sequence[float] | None = None,
    include_points: bool = True,
    injection_spec_path: str | Path | None = None,
    evaluation_thresholds_snr: Sequence[float] | None = None,
    evaluation_cfar_thresholds_sigma: Sequence[float] | None = None,
) -> dict[str, Any]:
    if max_pairs is not None and max_pairs <= 0:
        raise ValueError("max_pairs must be positive")
    if max_frames_override is not None and (
        isinstance(max_frames_override, bool)
        or not isinstance(max_frames_override, int)
        or max_frames_override <= 0
    ):
        raise ValueError("max_frames_override must be a positive integer")
    if (input_path_override is None) != (timestamp_csv_override is None):
        raise ValueError(
            "input_path_override and timestamp_csv_override must be supplied together"
        )
    if minimum_grid_coverage_override is not None and (
        isinstance(minimum_grid_coverage_override, bool)
        or not isinstance(minimum_grid_coverage_override, (int, float))
        or not np.isfinite(minimum_grid_coverage_override)
        or not 0 <= minimum_grid_coverage_override <= 1
    ):
        raise ValueError("minimum_grid_coverage_override must be finite and in [0, 1]")
    if max_candidates_per_cell_override is not None and (
        isinstance(max_candidates_per_cell_override, bool)
        or not isinstance(max_candidates_per_cell_override, int)
        or max_candidates_per_cell_override <= 0
    ):
        raise ValueError("max_candidates_per_cell_override must be a positive integer")
    reservation_overrides = (
        track_reservation_position_radius_px_override,
        track_reservation_velocity_radius_px_s_override,
        max_track_reservations_per_window_override,
    )
    if any(value is not None for value in reservation_overrides) and not all(
        value is not None for value in reservation_overrides
    ):
        raise ValueError("track-guided reservation overrides must be supplied together")
    if (
        track_reservation_minimum_mean_speed_px_s_override is not None
        and track_reservation_position_radius_px_override is None
    ):
        raise ValueError(
            "minimum reservation speed override requires the three "
            "track-guided reservation overrides"
        )
    for name, value in (
        (
            "track_reservation_position_radius_px_override",
            track_reservation_position_radius_px_override,
        ),
        (
            "track_reservation_velocity_radius_px_s_override",
            track_reservation_velocity_radius_px_s_override,
        ),
    ):
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{name} must be finite and positive")
    if max_track_reservations_per_window_override is not None and (
        isinstance(max_track_reservations_per_window_override, bool)
        or not isinstance(max_track_reservations_per_window_override, int)
        or max_track_reservations_per_window_override <= 0
    ):
        raise ValueError(
            "max_track_reservations_per_window_override must be a positive integer"
        )
    if track_reservation_minimum_mean_speed_px_s_override is not None and (
        isinstance(track_reservation_minimum_mean_speed_px_s_override, bool)
        or not isinstance(
            track_reservation_minimum_mean_speed_px_s_override, (int, float)
        )
        or not np.isfinite(track_reservation_minimum_mean_speed_px_s_override)
        or track_reservation_minimum_mean_speed_px_s_override < 0
    ):
        raise ValueError(
            "track_reservation_minimum_mean_speed_px_s_override must be finite "
            "and non-negative"
        )
    velocity_grid_values = (
        tuple(float(value) for value in velocity_grid_override)
        if velocity_grid_override is not None
        else None
    )
    if velocity_grid_values is not None:
        if len(velocity_grid_values) != 5:
            raise ValueError(
                "velocity_grid_override must contain vx_min,vx_max,vy_min,vy_max,step"
            )
        if any(not np.isfinite(value) for value in velocity_grid_values):
            raise ValueError("velocity_grid_override values must be finite")
        if velocity_grid_values[0] > velocity_grid_values[1] or velocity_grid_values[
            2
        ] > velocity_grid_values[3]:
            raise ValueError("velocity grid minimum cannot exceed maximum")
        if velocity_grid_values[4] <= 0:
            raise ValueError("velocity grid step must be positive")
    config = load_config(config_path)
    input_config = dict(config.input)
    runtime_overrides: dict[str, Any] = {}
    if input_path_override is not None and timestamp_csv_override is not None:
        input_config["path"] = str(Path(input_path_override).expanduser().resolve())
        input_config["timestamp_csv"] = str(
            Path(timestamp_csv_override).expanduser().resolve()
        )
        runtime_overrides["input.path"] = input_config["path"]
        runtime_overrides["input.timestamp_csv"] = input_config["timestamp_csv"]
    if max_frames_override is not None:
        input_config["max_frames"] = max_frames_override
        runtime_overrides["input.max_frames"] = max_frames_override
    effective_raw = dict(config.raw)
    effective_raw["input"] = input_config
    if minimum_grid_coverage_override is not None:
        coverage = float(minimum_grid_coverage_override)
        motion_override = dict(effective_raw.get("motion") or {})
        motion_override["minimum_grid_coverage"] = coverage
        effective_raw["motion"] = motion_override
        global_motion_override = dict(effective_raw.get("global_motion") or {})
        global_motion_override["minimum_inlier_grid_coverage"] = coverage
        effective_raw["global_motion"] = global_motion_override
        runtime_overrides["motion.minimum_grid_coverage"] = coverage
        runtime_overrides["global_motion.minimum_inlier_grid_coverage"] = coverage
    if max_candidates_per_cell_override is not None:
        candidate_override = dict(effective_raw.get("candidates") or {})
        candidate_override["max_candidates_per_cell"] = (
            max_candidates_per_cell_override
        )
        effective_raw["candidates"] = candidate_override
        runtime_overrides["candidates.max_candidates_per_cell"] = (
            max_candidates_per_cell_override
        )
    if track_reservation_position_radius_px_override is not None:
        assert track_reservation_velocity_radius_px_s_override is not None
        assert max_track_reservations_per_window_override is not None
        candidate_override = dict(effective_raw.get("candidates") or {})
        reservation_candidate_overrides = {
            "track_guided_reservation_position_radius_px": float(
                track_reservation_position_radius_px_override
            ),
            "track_guided_reservation_velocity_radius_px_s": float(
                track_reservation_velocity_radius_px_s_override
            ),
            "max_track_guided_reservations_per_window": (
                max_track_reservations_per_window_override
            ),
        }
        if track_reservation_minimum_mean_speed_px_s_override is not None:
            reservation_candidate_overrides[
                "track_guided_reservation_minimum_mean_speed_px_s"
            ] = float(track_reservation_minimum_mean_speed_px_s_override)
        candidate_override.update(reservation_candidate_overrides)
        effective_raw["candidates"] = candidate_override
        runtime_overrides.update(
            {
                f"candidates.{name}": value
                for name, value in reservation_candidate_overrides.items()
            }
        )
    if velocity_grid_values is not None:
        synthetic_override = dict(effective_raw.get("synthetic_tracking") or {})
        override_names = (
            "vx_min_px_s",
            "vx_max_px_s",
            "vy_min_px_s",
            "vy_max_px_s",
            "velocity_step_px_s",
        )
        synthetic_override.update(dict(zip(override_names, velocity_grid_values)))
        effective_raw["synthetic_tracking"] = synthetic_override
        runtime_overrides["synthetic_tracking.velocity_grid"] = {
            name: value for name, value in zip(override_names, velocity_grid_values)
        }
    effective_config_sha256 = hashlib.sha256(
        json.dumps(effective_raw, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    motion_value = effective_raw.get("motion")
    if motion_value is not None and not isinstance(motion_value, dict):
        raise ConfigError("motion must be a mapping")
    motion_config = PvaMotionConfig.from_mapping(motion_value)
    global_motion_value = effective_raw.get("global_motion")
    if global_motion_value is not None and not isinstance(global_motion_value, dict):
        raise ConfigError("global_motion must be a mapping")
    global_motion_config = GlobalMotionConfig.from_mapping(global_motion_value)
    stabilization_value = config.raw.get("stabilization")
    if stabilization_value is not None and not isinstance(stabilization_value, dict):
        raise ConfigError("stabilization must be a mapping")
    stabilization_config = StabilizationConfig.from_mapping(stabilization_value)
    background_value = config.raw.get("background")
    if background_value is not None and not isinstance(background_value, dict):
        raise ConfigError("background must be a mapping")
    background_config = BackgroundConfig.from_mapping(background_value)
    noise_value = config.raw.get("noise")
    if noise_value is not None and not isinstance(noise_value, dict):
        raise ConfigError("noise must be a mapping")
    noise_config = NoiseConfig.from_mapping(noise_value)
    psf_value = config.raw.get("psf")
    if psf_value is not None and not isinstance(psf_value, dict):
        raise ConfigError("psf must be a mapping")
    matched_filter_config = MatchedFilterConfig.from_mapping(psf_value)
    synthetic_value = effective_raw.get("synthetic_tracking")
    if synthetic_value is not None and not isinstance(synthetic_value, dict):
        raise ConfigError("synthetic_tracking must be a mapping")
    synthetic_config = SyntheticTrackingConfig.from_mapping(synthetic_value)
    candidate_value = effective_raw.get("candidates")
    if candidate_value is not None and not isinstance(candidate_value, dict):
        raise ConfigError("candidates must be a mapping")
    candidate_config = CandidateExtractionConfig.from_mapping(candidate_value)
    tracking_value = config.raw.get("tracking")
    if tracking_value is not None and not isinstance(tracking_value, dict):
        raise ConfigError("tracking must be a mapping")
    tracking_config = KalmanTrackingConfig.from_mapping(tracking_value)
    injection_spec = None
    injection_identity = None
    injector = None
    if injection_spec_path is not None:
        injection_spec, injection_identity = load_injection_spec(injection_spec_path)
        injector = SyntheticInjector(injection_spec)
    raw_sweep_requested = bool(
        evaluation_thresholds_snr is not None
        and len(evaluation_thresholds_snr) > 0
    )
    cfar_sweep_requested = bool(
        evaluation_cfar_thresholds_sigma is not None
        and len(evaluation_cfar_thresholds_sigma) > 0
    )
    if raw_sweep_requested and cfar_sweep_requested:
        raise ValueError(
            "choose either raw-SNR or CFAR evaluation thresholds, not both"
        )
    threshold_sweep_parameter = (
        "cfar_threshold_sigma"
        if cfar_sweep_requested
        else "score_threshold_snr"
    )
    thresholds = tuple(
        sorted(
            set(
                float(value)
                for value in (
                    evaluation_cfar_thresholds_sigma
                    if evaluation_cfar_thresholds_sigma is not None
                    else evaluation_thresholds_snr or ()
                )
            )
        )
    )
    if any(not np.isfinite(value) or value <= 0 for value in thresholds):
        raise ValueError("evaluation candidate thresholds must be finite and positive")
    if (
        cfar_sweep_requested
        and candidate_config.ranking_mode != "tile_robust_cfar"
    ):
        raise ValueError(
            "CFAR threshold evaluation requires candidates.ranking_mode="
            "tile_robust_cfar"
        )
    estimator = PvaPyrLkMotionEstimator(motion_config)
    stabilizer = FullResolutionStabilizer(stabilization_config)
    matched_filter = PsfMatchedFilter(
        matched_filter_config, base_path=config.path.parent
    )
    synthetic_tracker = (
        CudaShiftAndStack(synthetic_config, base_path=config.path.parent)
        if synthetic_config.backend == "cuda"
        else ReferenceShiftAndStack(synthetic_config)
    )
    synthetic_window = ReferenceSyntheticWindow(synthetic_tracker)
    candidate_extractor = CandidateExtractor(candidate_config)
    track_manager = KalmanTrackManager(tracking_config)
    evaluation_extractors = {
        threshold: CandidateExtractor(
            replace(
                candidate_config,
                **{threshold_sweep_parameter: threshold},
            )
        )
        for threshold in thresholds
    }
    evaluation_trackers = {
        threshold: KalmanTrackManager(tracking_config) for threshold in thresholds
    }
    support_window = ValidMaskWindow(stabilization_config.integration_window_frames)
    source = source_from_config(input_config)

    preprocessor: RobustPreprocessor | None = None

    def ensure_preprocessor(shape: tuple[int, int]) -> RobustPreprocessor:
        nonlocal preprocessor
        if preprocessor is None:
            bad_pixel_mask = None
            if noise_config.bad_pixel_map is not None:
                map_path = Path(noise_config.bad_pixel_map).expanduser()
                if not map_path.is_absolute():
                    map_path = (config.path.parent / map_path).resolve()
                bad_pixel_mask = load_bad_pixel_map(map_path, shape)
            preprocessor = RobustPreprocessor(
                background_config,
                noise_config,
                bad_pixel_mask=bad_pixel_mask,
            )
        return preprocessor

    def prepare_source(frame: Frame) -> Frame:
        return ensure_preprocessor(frame.shape).prepare_source(frame)

    def preprocess(stabilized: Any) -> Any:
        return ensure_preprocessor(stabilized.frame.shape).process(stabilized)

    pairs: list[dict[str, Any]] = []
    skipped_pairs: list[dict[str, Any]] = []
    stabilized_frames: list[dict[str, Any]] = []
    preprocessed_frames: list[dict[str, Any]] = []
    matched_filter_frames: list[dict[str, Any]] = []
    synthetic_windows: list[dict[str, Any]] = []
    candidate_batches: list[dict[str, Any]] = []
    track_batches: list[dict[str, Any]] = []
    injected_truth_score_probes: list[dict[str, Any]] = []
    evaluation_threshold_batches: dict[str, list[dict[str, Any]]] = {
        f"{threshold:g}": [] for threshold in thresholds
    }

    def probe_injected_truth(
        window: Any,
        ranking_surface: CandidateRankingSurface,
    ) -> None:
        if injection_spec is None:
            return
        timestamps = {
            int(item["frame_index"]): int(item["timestamp_ns"])
            for item in preprocessed_frames
        }
        frame_metadata = {
            int(item["frame_index"]): (
                timestamps[int(item["frame_index"])],
                np.asarray(item["source_to_reference_matrix"], np.float64),
            )
            for item in stabilized_frames
            if int(item["frame_index"]) in window.frame_indices
        }
        truths = [
            truth
            for target in injection_spec.targets
            if (
                truth := transformed_target_truth(
                    target,
                    window.frame_indices,
                    window.reference_timestamp_ns,
                    frame_metadata,
                )
            )
            is not None
        ]
        valid_scores = window.score[window.valid_mask]
        selection_valid = window.valid_mask & np.isfinite(ranking_surface.score)
        valid_selection_scores = ranking_surface.score[selection_valid]
        height, width = window.score.shape
        radius = 3
        target_probes = []
        for truth in truths:
            x, y = (float(value) for value in truth["position_xy_px"])
            nearest_x = int(round(x))
            nearest_y = int(round(y))
            x0 = max(0, nearest_x - radius)
            x1 = min(width, nearest_x + radius + 1)
            y0 = max(0, nearest_y - radius)
            y1 = min(height, nearest_y + radius + 1)
            local_valid = window.valid_mask[y0:y1, x0:x1]
            probe = dict(truth)
            probe["probe_radius_px"] = radius
            probe["diagnostic_only_not_candidate_output"] = True
            if local_valid.size == 0 or not np.any(local_valid):
                probe["valid_score_available"] = False
                probe["selection_score_available"] = False
                target_probes.append(probe)
                continue
            local_scores = np.where(
                local_valid,
                window.score[y0:y1, x0:x1],
                -np.inf,
            )
            local_y, local_x = np.unravel_index(
                int(np.argmax(local_scores)), local_scores.shape
            )
            peak_x = x0 + int(local_x)
            peak_y = y0 + int(local_y)
            peak_score = float(window.score[peak_y, peak_x])
            velocity_index = int(window.velocity_index[peak_y, peak_x])
            selected_velocity = window.velocity_grid_xy_px_s[velocity_index]
            probe.update(
                {
                    "valid_score_available": True,
                    "local_peak_position_xy_px": [peak_x, peak_y],
                    "local_peak_score_snr": peak_score,
                    "local_peak_selected_velocity_xy_px_s": (
                        selected_velocity.tolist()
                    ),
                    "local_peak_support_frames": int(
                        window.valid_support_count[peak_y, peak_x]
                    ),
                    "local_peak_position_error_px": float(
                        np.hypot(peak_x - x, peak_y - y)
                    ),
                    "local_peak_velocity_error_px_s": float(
                        np.linalg.norm(
                            selected_velocity
                            - np.asarray(truth["velocity_xy_px_s"], np.float64)
                        )
                    ),
                    "surface_rank_lower_bound": (
                        1 + int(np.count_nonzero(valid_scores > peak_score))
                    ),
                    "surface_rank_ties_not_resolved": True,
                }
            )
            local_selection_valid = selection_valid[y0:y1, x0:x1]
            if not np.any(local_selection_valid):
                probe["selection_score_available"] = False
                target_probes.append(probe)
                continue
            local_selection = np.where(
                local_selection_valid,
                ranking_surface.score[y0:y1, x0:x1],
                -np.inf,
            )
            selection_y, selection_x = np.unravel_index(
                int(np.argmax(local_selection)), local_selection.shape
            )
            selection_peak_x = x0 + int(selection_x)
            selection_peak_y = y0 + int(selection_y)
            selection_peak = float(
                ranking_surface.score[selection_peak_y, selection_peak_x]
            )
            selection_probe = {
                "selection_score_available": True,
                "local_peak_selection_position_xy_px": [
                    selection_peak_x,
                    selection_peak_y,
                ],
                "local_peak_selection_score": selection_peak,
                "selection_score_units": ranking_surface.units,
                "local_peak_selection_position_error_px": float(
                    np.hypot(selection_peak_x - x, selection_peak_y - y)
                ),
                "selection_surface_rank_lower_bound": (
                    1
                    + int(
                        np.count_nonzero(valid_selection_scores > selection_peak)
                    )
                ),
                "selection_surface_rank_ties_not_resolved": True,
            }
            if ranking_surface.tile_center_snr is not None:
                tile_y = selection_peak_y // int(ranking_surface.tile_height_px)
                tile_x = selection_peak_x // int(ranking_surface.tile_width_px)
                selection_probe.update(
                    {
                        "local_clutter_center_snr": float(
                            ranking_surface.tile_center_snr[tile_y, tile_x]
                        ),
                        "local_clutter_scale_snr": float(
                            ranking_surface.tile_scale_snr[tile_y, tile_x]
                        ),
                        "local_clutter_sample_count": int(
                            ranking_surface.tile_sample_count[tile_y, tile_x]
                        ),
                    }
                )
            probe.update(selection_probe)
            target_probes.append(probe)
        injected_truth_score_probes.append(
            {
                "frame_indices": list(window.frame_indices),
                "reference_timestamp_ns": window.reference_timestamp_ns,
                "targets": target_probes,
            }
        )

    def consume_synthetic_window(window: Any) -> None:
        synthetic_windows.append(window.to_dict())
        primary_ranking = candidate_extractor.ranking_surface(window)
        probe_injected_truth(window, primary_ranking)
        primary_hints = (
            track_manager.reservation_hints(
                reference_timestamp_ns=window.reference_timestamp_ns,
                segment_index=window.segment_index,
            )
            if candidate_config.max_track_guided_reservations_per_window
            else ()
        )
        primary_candidates = candidate_extractor.extract(
            window,
            ranking_surface=primary_ranking,
            track_prediction_hints=primary_hints,
        )
        candidate_batches.append(primary_candidates.to_dict())
        track_batches.append(track_manager.update(primary_candidates).to_dict())
        for threshold in thresholds:
            evaluated_hints = (
                evaluation_trackers[threshold].reservation_hints(
                    reference_timestamp_ns=window.reference_timestamp_ns,
                    segment_index=window.segment_index,
                )
                if candidate_config.max_track_guided_reservations_per_window
                else ()
            )
            evaluated_candidates = evaluation_extractors[threshold].extract(
                window,
                ranking_surface=primary_ranking,
                track_prediction_hints=evaluated_hints,
            )
            evaluated_tracks = evaluation_trackers[threshold].update(
                evaluated_candidates
            )
            evaluation_threshold_batches[f"{threshold:g}"].append(
                {
                    "candidate_batch": evaluated_candidates.to_dict(),
                    "track_metrics": evaluated_tracks.metrics,
                    "confirmed_or_coasted_tracks": [
                        track.to_dict()
                        for track in evaluated_tracks.tracks
                        if track.lifecycle_state in {"confirmed", "coasted"}
                    ],
                    "reset_reason": evaluated_tracks.reset_reason,
                    "tracking_timings_ms": evaluated_tracks.timings_ms,
                }
            )
    frame_count = 0
    attempted_pair_count = 0
    started = time.perf_counter_ns()
    iterator = iter(source)
    previous: Frame | None = None
    previous_stabilized = None
    tracker: GlobalMotionTracker | None = None
    try:
        for current in iterator:
            if injector is not None:
                current = injector.inject(current)
            current = prepare_source(current)
            frame_count += 1
            if previous is None:
                previous = current
                tracker = GlobalMotionTracker(
                    global_motion_config, initial_frame_index=current.frame_index
                )
                initial_state = ComposedMotionState(
                    reference_frame_index=current.frame_index,
                    current_frame_index=current.frame_index,
                    segment_index=0,
                    reference_from_current_matrix=np.eye(3),
                    status="initial_reference",
                    window_reset=False,
                    reused_pairs=0,
                    pair_parameter_delta=None,
                )
                previous_stabilized = stabilizer.stabilize(current, initial_state)
                initial_support = support_window.update(previous_stabilized)
                initial_record = previous_stabilized.to_dict()
                initial_record["valid_support_window"] = initial_support.metrics()
                stabilized_frames.append(initial_record)
                initial_residual = preprocess(previous_stabilized)
                preprocessed_frames.append(initial_residual.to_dict())
                initial_matched = matched_filter.process(initial_residual)
                matched_filter_frames.append(initial_matched.to_dict())
                initial_window = synthetic_window.update(initial_matched)
                if initial_window is not None:
                    consume_synthetic_window(initial_window)
                continue
            attempted_pair_count += 1
            assert tracker is not None
            assert previous_stabilized is not None
            pair_record: dict[str, Any] | None = None
            skipped_record: dict[str, Any] | None = None
            if current.timestamp_ns <= previous.timestamp_ns:
                chain = tracker.reset(current.frame_index)
                skipped_record = {
                    "previous_frame_index": previous.frame_index,
                    "current_frame_index": current.frame_index,
                    "previous_timestamp_ns": previous.timestamp_ns,
                    "current_timestamp_ns": current.timestamp_ns,
                    "reason": "timestamps_not_strictly_increasing",
                    "discontinuities": [
                        item.value for item in current.discontinuities
                    ],
                    "stabilization_chain": chain.to_dict(),
                }
            else:
                try:
                    correspondence = estimator.estimate(previous, current)
                except PvaMotionError as exc:
                    chain = tracker.reset(current.frame_index)
                    skipped_record = {
                        "previous_frame_index": previous.frame_index,
                        "current_frame_index": current.frame_index,
                        "previous_timestamp_ns": previous.timestamp_ns,
                        "current_timestamp_ns": current.timestamp_ns,
                        "reason": "pva_motion_failure",
                        "detail": str(exc),
                        "discontinuities": [
                            item.value for item in current.discontinuities
                        ],
                        "stabilization_chain": chain.to_dict(),
                    }
                else:
                    global_estimate = fit_global_motion(
                        correspondence, global_motion_config
                    )
                    force_reset = bool(current.discontinuities)
                    chain = tracker.update(global_estimate, force_reset=force_reset)
                    pair_record = correspondence.to_dict(include_points=include_points)
                    pair_record["input"] = {
                        "previous_pixel_sha256": previous.pixel_sha256(),
                        "current_pixel_sha256": current.pixel_sha256(),
                        "current_discontinuities": [
                            item.value for item in current.discontinuities
                        ],
                        "crosses_timestamp_gap": (
                            Discontinuity.TIMESTAMP_GAP in current.discontinuities
                        ),
                    }
                    pair_record["global_motion"] = global_estimate.to_dict(
                        include_inlier_indices=include_points
                    )
                    pair_record["stabilization_chain"] = chain.to_dict()

            current_stabilized = stabilizer.stabilize(current, chain)
            support = support_window.update(current_stabilized)
            stabilization_record = current_stabilized.to_dict()
            stabilization_record["valid_support_window"] = support.metrics()
            stabilized_frames.append(stabilization_record)
            current_residual = preprocess(current_stabilized)
            preprocessed_frames.append(current_residual.to_dict())
            current_matched = matched_filter.process(current_residual)
            matched_filter_frames.append(current_matched.to_dict())
            current_window = synthetic_window.update(current_matched)
            if current_window is not None:
                consume_synthetic_window(current_window)
            alignment = alignment_improvement_metrics(
                previous,
                current,
                previous_stabilized,
                current_stabilized,
                sample_stride=stabilization_config.alignment_sample_stride,
            )
            if pair_record is not None:
                pair_record["image_alignment"] = alignment
                pairs.append(pair_record)
            if skipped_record is not None:
                skipped_record["image_alignment"] = alignment
                skipped_pairs.append(skipped_record)
            previous = current
            previous_stabilized = current_stabilized
            if max_pairs is not None and attempted_pair_count >= max_pairs:
                break
    finally:
        _close_iterator(iterator)

    injection_coverage = injector.validate_coverage() if injector is not None else None
    if not pairs:
        raise PvaMotionError("No valid frame pairs produced motion correspondences")
    total_ms = (time.perf_counter_ns() - started) / 1_000_000
    counts = [int(pair["accepted_count"]) for pair in pairs]
    correspondence_usable_count = sum(
        bool(pair["metrics"]["usable_for_transform"]) for pair in pairs
    )
    accepted_transform_count = sum(
        pair["global_motion"]["quality_status"] == "accepted" for pair in pairs
    )
    reset_count = sum(
        pair["stabilization_chain"]["window_reset"] for pair in pairs
    ) + sum(
        item["stabilization_chain"]["window_reset"] for item in skipped_pairs
    )
    reused_count = sum(
        pair["stabilization_chain"]["status"] == "reused_previous"
        for pair in pairs
    )
    resampled_count = sum(
        int(item["resampling_count"] == 1) for item in stabilized_frames
    )
    comparable_alignments = [
        pair["image_alignment"]
        for pair in pairs
        if pair["image_alignment"]["median_absolute_difference_after"] is not None
    ]
    detection_ready_preprocessed = [
        item for item in preprocessed_frames if item["detection_ready"]
    ]
    residual_scales = [
        float(item["metrics"]["residual"]["robust_scale"])
        for item in detection_ready_preprocessed
        if item["metrics"]["residual"]["robust_scale"] is not None
    ]
    whitened_scales = [
        float(item["metrics"]["whitened"]["robust_scale"])
        for item in detection_ready_preprocessed
        if item["metrics"]["whitened"]["robust_scale"] is not None
    ]
    whitened_peaks = [
        float(item["metrics"]["whitened"]["maximum"])
        for item in detection_ready_preprocessed
        if item["metrics"]["whitened"]["maximum"] is not None
    ]
    ready_matched_filter_frames = [
        item for item in matched_filter_frames if item["detection_ready"]
    ]
    matched_scales = [
        float(item["metrics"]["response"]["robust_scale"])
        for item in ready_matched_filter_frames
        if item["metrics"]["response"]["robust_scale"] is not None
    ]
    matched_peaks = [
        float(item["metrics"]["score"]["maximum"])
        for item in ready_matched_filter_frames
        if item["metrics"]["score"]["maximum"] is not None
    ]
    matched_tail_fractions = [
        float(item["metrics"]["score"]["fraction_gt_5"])
        for item in ready_matched_filter_frames
        if item["metrics"]["score"]["fraction_gt_5"] is not None
    ]
    synthetic_scores = [
        float(item["metrics"]["score"]["maximum"])
        for item in synthetic_windows
        if item["metrics"]["score"]["maximum"] is not None
    ]
    synthetic_tail_fractions = [
        float(item["metrics"]["score"]["fraction_gt_5"])
        for item in synthetic_windows
        if item["metrics"]["score"]["fraction_gt_5"] is not None
    ]
    timestamp_semantics = input_config.get("timestamp_semantics", "unspecified")
    warnings: list[str] = []
    if timestamp_semantics != "sensor_exposure":
        warnings.append(
            "Timestamps are not identified as sensor-exposure timestamps; retain "
            "their acquisition semantics with later velocity measurements."
        )
    if skipped_pairs:
        warnings.append(
            "One or more pairs were skipped after a timestamp or PVA motion failure."
        )
    if correspondence_usable_count != len(pairs):
        warnings.append(
            f"{len(pairs) - correspondence_usable_count} motion pair(s) failed the configured "
            "feature-count or spatial-coverage gate and must not feed a transform."
        )
    if accepted_transform_count != len(pairs):
        warnings.append(
            f"{len(pairs) - accepted_transform_count} global transform(s) were "
            "rejected and triggered the configured failure policy."
        )
    if not detection_ready_preprocessed:
        warnings.append(
            "No preprocessing frame completed warm-up in a stable segment; later "
            "detection stages must remain suppressed for this run."
        )
    if matched_filter.bank.provisional:
        warnings.append(
            "The matched filter uses a provisional Gaussian PSF; detection "
            "sensitivity claims require a measured camera PSF."
        )
    if not synthetic_windows:
        warnings.append(
            "No complete detection-ready synthetic tracking window was formed."
        )
    if len(synthetic_tracker.velocity_grid) == 1 and np.array_equal(
        synthetic_tracker.velocity_grid[0], np.zeros(2, np.float32)
    ):
        warnings.append(
            "The recording run uses a zero-velocity diagnostic grid because "
            "physical target velocity bounds are not yet calibrated."
        )
    if candidate_batches and all(
        int(item["candidate_count"]) == candidate_config.max_candidates_per_window
        for item in candidate_batches
    ):
        warnings.append(
            "Every candidate window reached the configured output cap; this "
            "operating point has an unbounded false-alarm burden until calibrated."
        )
    if any(
        int(item["metrics"]["dropped_birth_count_at_active_track_cap"]) > 0
        for item in track_batches
    ):
        warnings.append(
            "The active-track cap rejected one or more candidate births; tracking "
            "counts are bounded diagnostics at this detection operating point."
        )
    confirmed_track_ids = sorted(
        {
            int(track["track_id"])
            for item in track_batches
            for track in item["tracks"]
            if track["lifecycle_state"] in {"confirmed", "coasted"}
        }
    )
    if track_batches and not confirmed_track_ids:
        warnings.append(
            "No track accumulated the configured number of independent, "
            "non-overlapping confirmation hits in this run."
        )
    if tracking_config.measurement_noise_source == "provisional":
        warnings.append(
            "Kalman measurement-noise sigmas are provisional; track covariance "
            "and association gates require empirical camera calibration."
        )
    stage_names = sorted(
        {name for pair in pairs for name in pair["timings_ms"]}
    )
    stage_medians = {
        name: statistics.median(
            float(pair["timings_ms"][name])
            for pair in pairs
            if name in pair["timings_ms"]
        )
        for name in stage_names
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "run": run_identity(REPOSITORY),
        "implementation_sha256": _implementation_identity(),
        "config": {
            "path": str(config.path),
            "sha256": config.sha256,
            "resolved": config.raw,
            "runtime_overrides": runtime_overrides,
            "effective_sha256": effective_config_sha256,
            "effective_resolved": effective_raw,
        },
        "source": {
            "identity": file_identity(input_config["path"]),
            "timestamp_sidecar_identity": (
                file_identity(input_config["timestamp_csv"])
                if input_config.get("timestamp_csv") is not None
                else None
            ),
            "timestamp_semantics": timestamp_semantics,
            "timestamp_gap_detection": {
                "expected_interval_ns": getattr(
                    source, "expected_interval_ns", None
                ),
                "expected_interval_source": getattr(
                    source, "expected_interval_source", None
                ),
                "gap_factor": input_config.get("timestamp_gap_factor", 1.5),
            },
        },
        "summary": {
            "frames_read": frame_count,
            "pairs_processed": len(pairs),
            "pairs_attempted": attempted_pair_count,
            "pairs_skipped": len(skipped_pairs),
            "correspondence_usable_pairs": correspondence_usable_count,
            "accepted_global_transforms": accepted_transform_count,
            "rejected_global_transforms": len(pairs) - accepted_transform_count,
            "reference_resets": reset_count,
            "reused_previous_transforms": reused_count,
            "accepted_features": {
                "minimum": min(counts),
                "median": statistics.median(counts),
                "maximum": max(counts),
            },
            "end_to_end_ms": total_ms,
            "stage_median_ms": stage_medians,
            "global_motion_fit_median_ms": statistics.median(
                float(pair["global_motion"]["timing_ms"]) for pair in pairs
            ),
            "stabilization": {
                "frames": len(stabilized_frames),
                "resampled_frames": resampled_count,
                "identity_frames_without_resampling": (
                    len(stabilized_frames) - resampled_count
                ),
                "median_total_ms": statistics.median(
                    float(item["timings_ms"]["total"])
                    for item in stabilized_frames
                ),
                "minimum_valid_fraction": min(
                    float(item["metrics"]["valid_fraction"])
                    for item in stabilized_frames
                ),
                "final_common_valid_fraction": float(
                    stabilized_frames[-1]["valid_support_window"][
                        "common_valid_fraction"
                    ]
                ),
                "failure_reset_rate": (
                    reset_count / attempted_pair_count
                    if attempted_pair_count
                    else 0.0
                ),
                "background_alignment": {
                    "comparable_pairs": len(comparable_alignments),
                    "median_absolute_difference_before": (
                        statistics.median(
                            float(item["median_absolute_difference_before"])
                            for item in comparable_alignments
                        )
                        if comparable_alignments
                        else None
                    ),
                    "median_absolute_difference_after": (
                        statistics.median(
                            float(item["median_absolute_difference_after"])
                            for item in comparable_alignments
                        )
                        if comparable_alignments
                        else None
                    ),
                },
            },
            "preprocessing": {
                "frames": len(preprocessed_frames),
                "detection_ready_frames": len(detection_ready_preprocessed),
                "detection_suppressed_frames": (
                    len(preprocessed_frames) - len(detection_ready_preprocessed)
                ),
                "warmup_suppressed_frames": sum(
                    not bool(item["metrics"]["warmup_complete"])
                    for item in preprocessed_frames
                ),
                "global_change_suppressed_frames": sum(
                    bool(item["metrics"]["global_change_suppressed"])
                    for item in preprocessed_frames
                ),
                "median_total_ms": statistics.median(
                    float(item["timings_ms"]["total"])
                    for item in preprocessed_frames
                ),
                "minimum_detection_valid_fraction": (
                    min(
                        float(item["metrics"]["detection_valid_fraction"])
                        for item in detection_ready_preprocessed
                    )
                    if detection_ready_preprocessed
                    else None
                ),
                "median_residual_robust_scale": (
                    statistics.median(residual_scales) if residual_scales else None
                ),
                "median_whitened_robust_scale": (
                    statistics.median(whitened_scales) if whitened_scales else None
                ),
                "maximum_spatial_whitened_peak": (
                    max(whitened_peaks) if whitened_peaks else None
                ),
            },
            "matched_filter": {
                "frames": len(matched_filter_frames),
                "detection_ready_frames": len(ready_matched_filter_frames),
                "suppressed_frames": (
                    len(matched_filter_frames) - len(ready_matched_filter_frames)
                ),
                "median_total_ms": statistics.median(
                    float(item["timings_ms"]["total"])
                    for item in matched_filter_frames
                ),
                "median_ready_total_ms": (
                    statistics.median(
                        float(item["timings_ms"]["total"])
                        for item in ready_matched_filter_frames
                    )
                    if ready_matched_filter_frames
                    else None
                ),
                "minimum_valid_fraction": (
                    min(
                        float(item["metrics"]["valid_fraction"])
                        for item in ready_matched_filter_frames
                    )
                    if ready_matched_filter_frames
                    else None
                ),
                "median_response_robust_scale": (
                    statistics.median(matched_scales) if matched_scales else None
                ),
                "maximum_score": max(matched_peaks) if matched_peaks else None,
                "median_fraction_score_gt_5": (
                    statistics.median(matched_tail_fractions)
                    if matched_tail_fractions
                    else None
                ),
                "kernel": matched_filter.bank.metadata(),
            },
            "synthetic_tracking": {
                "backend": synthetic_tracker.backend,
                "windows": len(synthetic_windows),
                "window_frames": synthetic_config.window_frames,
                "window_stride_frames": synthetic_config.window_stride_frames,
                "velocity_trial_count": len(synthetic_tracker.velocity_grid),
                "velocity_grid_xy_px_s": synthetic_tracker.velocity_grid.tolist(),
                "median_total_ms": (
                    statistics.median(
                        float(item["timings_ms"]["total"])
                        for item in synthetic_windows
                    )
                    if synthetic_windows
                    else None
                ),
                "median_cuda_kernel_ms": (
                    statistics.median(
                        float(item["timings_ms"]["cuda_kernel"])
                        for item in synthetic_windows
                    )
                    if synthetic_windows and synthetic_tracker.backend == "cuda"
                    else None
                ),
                "median_cuda_gpu_total_ms": (
                    statistics.median(
                        float(item["timings_ms"]["cuda_gpu_total"])
                        for item in synthetic_windows
                    )
                    if synthetic_windows and synthetic_tracker.backend == "cuda"
                    else None
                ),
                "cuda_library_sha256": (
                    synthetic_windows[0]["metrics"]["cuda"]["library_sha256"]
                    if synthetic_windows and synthetic_tracker.backend == "cuda"
                    else None
                ),
                "median_cuda_effective_pixel_samples_per_second": (
                    statistics.median(
                        float(
                            item["metrics"]["cuda"][
                                "effective_pixel_samples_per_second"
                            ]
                        )
                        for item in synthetic_windows
                    )
                    if synthetic_windows and synthetic_tracker.backend == "cuda"
                    else None
                ),
                "maximum_cuda_allocated_device_bytes": (
                    max(
                        int(item["metrics"]["cuda"]["allocated_device_bytes"])
                        for item in synthetic_windows
                    )
                    if synthetic_windows and synthetic_tracker.backend == "cuda"
                    else None
                ),
                "minimum_valid_fraction": (
                    min(
                        float(item["metrics"]["valid_fraction"])
                        for item in synthetic_windows
                    )
                    if synthetic_windows
                    else None
                ),
                "maximum_score": max(synthetic_scores) if synthetic_scores else None,
                "median_fraction_score_gt_5": (
                    statistics.median(synthetic_tail_fractions)
                    if synthetic_tail_fractions
                    else None
                ),
                "maximum_velocity_quantization_endpoint_error_px": (
                    max(
                        float(
                            item["metrics"][
                                "maximum_velocity_quantization_endpoint_error_px"
                            ]
                        )
                        for item in synthetic_windows
                    )
                    if synthetic_windows
                    else None
                ),
            },
            "candidate_extraction": {
                "windows": len(candidate_batches),
                "threshold_units": "normalized_shift_and_stack_snr",
                "score_threshold_snr": candidate_config.score_threshold_snr,
                "minimum_support_frames": candidate_config.minimum_support_frames,
                "total_output_candidates": sum(
                    int(item["candidate_count"]) for item in candidate_batches
                ),
                "maximum_output_candidates_per_window": (
                    max(int(item["candidate_count"]) for item in candidate_batches)
                    if candidate_batches
                    else 0
                ),
                "windows_with_pre_nms_truncation": sum(
                    bool(item["metrics"]["pre_nms_truncated"])
                    for item in candidate_batches
                ),
                "windows_with_output_truncation": sum(
                    bool(item["metrics"]["output_truncated"])
                    for item in candidate_batches
                ),
                "median_total_ms": (
                    statistics.median(
                        float(item["timings_ms"]["total"])
                        for item in candidate_batches
                    )
                    if candidate_batches
                    else None
                ),
                "score_surface": "best_score_over_velocity_hypotheses",
            },
            "temporal_tracking": {
                "windows": len(track_batches),
                "motion_model": "constant_velocity_xy_vx_vy",
                "measurement_noise_policy": "configured_fixed_diagonal_sigma",
                "measurement_noise_source": tracking_config.measurement_noise_source,
                "confirmation_evidence_policy": tracking_config.evidence_policy,
                "confirmation_independent_hits": (
                    tracking_config.confirmation_independent_hits
                ),
                "confirmed_track_ids": confirmed_track_ids,
                "confirmed_track_count": len(confirmed_track_ids),
                "total_births": sum(
                    int(item["metrics"]["birth_count"]) for item in track_batches
                ),
                "total_deletions": sum(
                    int(item["metrics"]["deleted_track_count"])
                    for item in track_batches
                ),
                "total_associations": sum(
                    int(item["metrics"]["associated_candidate_count"])
                    for item in track_batches
                ),
                "total_dropped_births_at_active_track_cap": sum(
                    int(item["metrics"]["dropped_birth_count_at_active_track_cap"])
                    for item in track_batches
                ),
                "maximum_active_tracks": (
                    max(
                        int(item["metrics"]["active_track_count"])
                        for item in track_batches
                    )
                    if track_batches
                    else 0
                ),
                "reset_count": sum(
                    item["reset_reason"] is not None for item in track_batches
                ),
                "median_total_ms": (
                    statistics.median(
                        float(item["timings_ms"]["total"])
                        for item in track_batches
                    )
                    if track_batches
                    else None
                ),
                "detector_score_used_as_track_confidence": False,
            },
        },
        "pairs": pairs,
        "skipped_pairs": skipped_pairs,
        "stabilized_frames": stabilized_frames,
        "preprocessed_frames": preprocessed_frames,
        "matched_filter_frames": matched_filter_frames,
        "synthetic_tracking_windows": synthetic_windows,
        "candidate_batches": candidate_batches,
        "track_batches": track_batches,
        "evaluation": (
            {
                "injection": (
                    {
                        "identity": injection_identity,
                        "specification": injection_spec.to_dict(),
                        "frame_records": injector.records,
                        "coverage": injection_coverage,
                    }
                    if injector is not None and injection_spec is not None
                    else None
                ),
                "candidate_threshold_sweep": evaluation_threshold_batches,
                "candidate_threshold_sweep_parameter": threshold_sweep_parameter,
                "injected_truth_score_probes": injected_truth_score_probes,
                "unmatched_candidates_are_false_alarms": False,
                "false_alarm_note": (
                    "The RAW16 source is unlabeled and not verified empty; only "
                    "injected-target misses can be labeled without new ground truth."
                ),
            }
            if injector is not None or thresholds
            else None
        ),
        "warnings": warnings,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate PVA motion, stabilize full-resolution frames, and produce "
            "noise-normalized, matched-filtered synthetic-tracking statistics"
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Override input.max_frames and record the effective configuration",
    )
    parser.add_argument(
        "--input-video",
        type=Path,
        help=(
            "Override input.path; requires --timestamp-csv and is recorded in "
            "the effective configuration"
        ),
    )
    parser.add_argument(
        "--timestamp-csv",
        type=Path,
        help=(
            "Override input.timestamp_csv; requires --input-video and is recorded "
            "in the effective configuration"
        ),
    )
    parser.add_argument(
        "--minimum-grid-coverage",
        type=float,
        help=(
            "Override both correspondence and inlier grid-coverage gates; "
            "recorded in the effective configuration"
        ),
    )
    parser.add_argument(
        "--max-candidates-per-cell",
        type=int,
        help=(
            "Override the final spatial-quota capacity per cell; recorded in "
            "the effective configuration"
        ),
    )
    parser.add_argument(
        "--track-reservation-position-radius-px",
        type=float,
        help="Enable track-guided reservation with this position gate radius",
    )
    parser.add_argument(
        "--track-reservation-velocity-radius-px-s",
        type=float,
        help="Track-guided reservation velocity gate radius; requires both companion options",
    )
    parser.add_argument(
        "--max-track-reservations-per-window",
        type=int,
        help="Strict per-window reservation cap; requires both radius options",
    )
    parser.add_argument(
        "--track-reservation-minimum-mean-speed-px-s",
        type=float,
        help=(
            "Minimum causal mean measured speed for reservation eligibility; "
            "recorded in the effective configuration"
        ),
    )
    parser.add_argument(
        "--velocity-grid",
        help=(
            "Override synthetic velocity grid as "
            "vx_min,vx_max,vy_min,vy_max,step; recorded in effective config"
        ),
    )
    parser.add_argument(
        "--injection-spec",
        type=Path,
        help="Inject a versioned synthetic PSF specification before motion processing",
    )
    parser.add_argument(
        "--evaluation-thresholds",
        help="Comma-separated normalized-SNR thresholds evaluated from each dense window",
    )
    parser.add_argument(
        "--evaluation-cfar-thresholds",
        help=(
            "Comma-separated tile-robust CFAR sigma thresholds evaluated from "
            "each dense window"
        ),
    )
    parser.add_argument(
        "--omit-points",
        action="store_true",
        help="Keep metrics but omit individual correspondences from the JSON report",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = estimate_config(
            args.config,
            max_pairs=args.max_pairs,
            max_frames_override=args.max_frames,
            input_path_override=args.input_video,
            timestamp_csv_override=args.timestamp_csv,
            minimum_grid_coverage_override=args.minimum_grid_coverage,
            max_candidates_per_cell_override=args.max_candidates_per_cell,
            track_reservation_position_radius_px_override=(
                args.track_reservation_position_radius_px
            ),
            track_reservation_velocity_radius_px_s_override=(
                args.track_reservation_velocity_radius_px_s
            ),
            track_reservation_minimum_mean_speed_px_s_override=(
                args.track_reservation_minimum_mean_speed_px_s
            ),
            max_track_reservations_per_window_override=(
                args.max_track_reservations_per_window
            ),
            velocity_grid_override=(
                tuple(float(value) for value in args.velocity_grid.split(","))
                if args.velocity_grid
                else None
            ),
            include_points=not args.omit_points,
            injection_spec_path=args.injection_spec,
            evaluation_thresholds_snr=(
                tuple(float(value) for value in args.evaluation_thresholds.split(","))
                if args.evaluation_thresholds
                else None
            ),
            evaluation_cfar_thresholds_sigma=(
                tuple(
                    float(value)
                    for value in args.evaluation_cfar_thresholds.split(",")
                )
                if args.evaluation_cfar_thresholds
                else None
            ),
        )
        if args.output is None:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            output = write_json_exclusive(args.output, report)
            print(f"Wrote {output}")
    except (
        ConfigError,
        FrameSourceError,
        PvaMotionError,
        StabilizationError,
        PreprocessingError,
        MatchedFilterError,
        SyntheticTrackingError,
        CandidateExtractionError,
        TemporalTrackingError,
        EvaluationError,
        OSError,
        ValueError,
    ) as exc:
        raise SystemExit(f"tiny-target motion failed: {exc}") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
