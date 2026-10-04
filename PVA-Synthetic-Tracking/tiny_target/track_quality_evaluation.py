"""Compare bounded track-quality evidence against injected moving-target truth."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .evaluation import EvaluationError
from .injected_raw_evaluation import analyze as analyze_injected_raw
from .telemetry import run_identity, write_json_exclusive


REPOSITORY = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "seaqr.tiny-target.track-quality-evaluation.v1"


@dataclass(frozen=True, slots=True)
class _Feature:
    name: str
    path: tuple[str, ...]
    gate_direction: str | None


@dataclass(frozen=True, slots=True)
class DiagnosticQualityPolicy:
    minimum_observations: int
    minimum_observation_fraction: float
    maximum_detector_score_standard_deviation: float
    minimum_peak_to_neighbor_ratio_mean: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.minimum_observations, bool)
            or not isinstance(self.minimum_observations, int)
            or self.minimum_observations < 2
        ):
            raise ValueError("minimum_observations must be an integer of at least 2")
        values = (
            self.minimum_observation_fraction,
            self.maximum_detector_score_standard_deviation,
            self.minimum_peak_to_neighbor_ratio_mean,
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("diagnostic quality policy values must be finite")
        if not 0 < self.minimum_observation_fraction <= 1:
            raise ValueError("minimum_observation_fraction must be in (0, 1]")
        if self.maximum_detector_score_standard_deviation < 0:
            raise ValueError(
                "maximum_detector_score_standard_deviation cannot be negative"
            )
        if self.minimum_peak_to_neighbor_ratio_mean <= 0:
            raise ValueError("minimum_peak_to_neighbor_ratio_mean must be positive")

    def to_dict(self) -> dict[str, float | int]:
        return {
            "minimum_observations": self.minimum_observations,
            "minimum_observation_fraction": self.minimum_observation_fraction,
            "maximum_detector_score_standard_deviation": (
                self.maximum_detector_score_standard_deviation
            ),
            "minimum_peak_to_neighbor_ratio_mean": (
                self.minimum_peak_to_neighbor_ratio_mean
            ),
        }


@dataclass(frozen=True, slots=True)
class DiagnosticMotionPersistencePolicy:
    minimum_observations: int
    minimum_observation_fraction: float
    minimum_measurement_speed_px_s: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.minimum_observations, bool)
            or not isinstance(self.minimum_observations, int)
            or self.minimum_observations < 2
        ):
            raise ValueError("minimum_observations must be an integer of at least 2")
        if (
            not math.isfinite(self.minimum_observation_fraction)
            or not 0 < self.minimum_observation_fraction <= 1
        ):
            raise ValueError("minimum_observation_fraction must be finite and in (0, 1]")
        if (
            not math.isfinite(self.minimum_measurement_speed_px_s)
            or self.minimum_measurement_speed_px_s < 0
        ):
            raise ValueError(
                "minimum_measurement_speed_px_s must be finite and non-negative"
            )

    def to_dict(self) -> dict[str, float | int]:
        return {
            "minimum_observations": self.minimum_observations,
            "minimum_observation_fraction": self.minimum_observation_fraction,
            "minimum_measurement_speed_px_s": self.minimum_measurement_speed_px_s,
        }


_FEATURES = (
    _Feature("age_windows", ("age_windows",), "minimum_required"),
    _Feature(
        "associated_update_count",
        ("associated_update_count",),
        "minimum_required",
    ),
    _Feature(
        "independent_confirmation_hits",
        ("confirmation", "independent_hits"),
        "minimum_required",
    ),
    _Feature(
        "observation_fraction_of_age_windows",
        ("quality_evidence", "observation_fraction_of_age_windows"),
        "minimum_required",
    ),
    _Feature(
        "detector_score_snr_minimum",
        ("quality_evidence", "detector_score_snr", "minimum"),
        "minimum_required",
    ),
    _Feature(
        "detector_score_snr_mean",
        ("quality_evidence", "detector_score_snr", "mean"),
        "minimum_required",
    ),
    _Feature(
        "detector_score_snr_standard_deviation",
        ("quality_evidence", "detector_score_snr", "standard_deviation"),
        "maximum_allowed",
    ),
    _Feature(
        "selection_score_minimum",
        ("quality_evidence", "selection_score", "minimum"),
        "minimum_required",
    ),
    _Feature(
        "selection_score_mean",
        ("quality_evidence", "selection_score", "mean"),
        "minimum_required",
    ),
    _Feature(
        "selection_score_standard_deviation",
        ("quality_evidence", "selection_score", "standard_deviation"),
        "maximum_allowed",
    ),
    _Feature(
        "peak_contrast_snr_mean",
        ("quality_evidence", "peak_contrast_snr", "mean"),
        "minimum_required",
    ),
    _Feature(
        "peak_to_neighbor_ratio_mean",
        ("quality_evidence", "peak_to_neighbor_ratio", "mean"),
        "minimum_required",
    ),
    _Feature(
        "measurement_speed_px_s_mean",
        ("quality_evidence", "measurement_speed_px_s", "mean"),
        None,
    ),
    _Feature(
        "measurement_velocity_step_px_s_rms",
        ("quality_evidence", "measurement_velocity_step_px_s", "rms"),
        "maximum_allowed",
    ),
    _Feature(
        "mahalanobis_distance_squared_rms",
        (
            "quality_evidence",
            "kalman_innovation",
            "mahalanobis_distance_squared",
            "rms",
        ),
        "maximum_allowed",
    ),
    _Feature(
        "position_residual_px_rms",
        (
            "quality_evidence",
            "kalman_innovation",
            "position_residual_px",
            "rms",
        ),
        "maximum_allowed",
    ),
    _Feature(
        "velocity_residual_px_s_rms",
        (
            "quality_evidence",
            "kalman_innovation",
            "velocity_residual_px_s",
            "rms",
        ),
        "maximum_allowed",
    ),
)


def _identity() -> dict[str, str]:
    files = (
        Path(__file__),
        Path(__file__).with_name("injected_raw_evaluation.py"),
        Path(__file__).parent / "tracking" / "kalman.py",
    )
    return {
        str(path.relative_to(REPOSITORY)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in files
    }


def _read_motion_report(path: str | Path) -> tuple[Path, dict[str, Any], str]:
    resolved = Path(path).expanduser().resolve()
    try:
        raw = resolved.read_bytes()
        report = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"Cannot read track-quality motion report: {exc}") from exc
    if not isinstance(report, dict) or report.get("schema_version") != (
        "seaqr.tiny-target.motion.v10"
    ):
        raise EvaluationError("track-quality evaluation requires a motion.v10 report")
    return resolved, report, hashlib.sha256(raw).hexdigest()


def _number_at(value: Mapping[str, Any], path: Sequence[str]) -> float | None:
    current: Any = value
    for name in path:
        if not isinstance(current, Mapping) or name not in current:
            return None
        current = current[name]
    if isinstance(current, bool) or not isinstance(current, (int, float)):
        return None
    number = float(current)
    return number if math.isfinite(number) else None


def _distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "minimum": None,
            "p25": None,
            "median": None,
            "p75": None,
            "maximum": None,
        }
    array = np.asarray(values, np.float64)
    return {
        "count": len(values),
        "minimum": float(np.min(array)),
        "p25": float(np.percentile(array, 25)),
        "median": float(np.median(array)),
        "p75": float(np.percentile(array, 75)),
        "maximum": float(np.max(array)),
    }


def _passes(value: float | None, direction: str, threshold: float) -> bool:
    if value is None:
        return False
    if direction == "minimum_required":
        return value >= threshold
    if direction == "maximum_allowed":
        return value <= threshold
    raise AssertionError(f"unsupported gate direction {direction}")


def _passes_diagnostic_policy(
    track: Mapping[str, Any], policy: DiagnosticQualityPolicy
) -> bool:
    quality = track["quality_evidence"]
    return bool(
        track["lifecycle_state"] in {"confirmed", "coasted"}
        and int(quality["observation_count"]) >= policy.minimum_observations
        and float(quality["observation_fraction_of_age_windows"])
        >= policy.minimum_observation_fraction
        and float(quality["detector_score_snr"]["standard_deviation"])
        <= policy.maximum_detector_score_standard_deviation
        and float(quality["peak_to_neighbor_ratio"]["mean"])
        >= policy.minimum_peak_to_neighbor_ratio_mean
    )


def _passes_motion_persistence_policy(
    track: Mapping[str, Any], policy: DiagnosticMotionPersistencePolicy
) -> bool:
    quality = track["quality_evidence"]
    return bool(
        track["lifecycle_state"] in {"confirmed", "coasted"}
        and int(quality["observation_count"]) >= policy.minimum_observations
        and float(quality["observation_fraction_of_age_windows"])
        >= policy.minimum_observation_fraction
        and float(quality["measurement_speed_px_s"]["mean"])
        >= policy.minimum_measurement_speed_px_s
    )


def _causal_policy_evaluation(
    latest_tracks: Mapping[int, Mapping[str, Any]],
    matched_targets: Mapping[int, set[str]],
    first_qualified_timestamp_ns: Mapping[int, int],
    policy: DiagnosticQualityPolicy | DiagnosticMotionPersistencePolicy,
    predicate: Any,
) -> dict[str, Any]:
    truth_track_ids = set(matched_targets)
    ever_qualified_ids = set(first_qualified_timestamp_ns)
    qualified_latest_ids = {
        track_id
        for track_id, track in latest_tracks.items()
        if predicate(track, policy)
    }
    qualified_truth_ids = ever_qualified_ids & truth_track_ids
    qualification_details = []
    for track_id in sorted(qualified_truth_ids):
        track = latest_tracks[track_id]
        timestamp_ns = first_qualified_timestamp_ns[track_id]
        confirmation_ns = int(track["confirmation"]["timestamp_ns"])
        birth_ns = int(track["birth_timestamp_ns"])
        qualification_details.append(
            {
                "track_id": track_id,
                "target_ids": sorted(matched_targets[track_id]),
                "first_qualification_timestamp_ns": timestamp_ns,
                "qualification_latency_from_birth_s": (timestamp_ns - birth_ns)
                / 1e9,
                "qualification_latency_after_confirmation_s": (
                    timestamp_ns - confirmation_ns
                )
                / 1e9,
            }
        )
    return {
        "policy": policy.to_dict(),
        "used_by_live_tracker": False,
        "ever_qualified_track_count": len(ever_qualified_ids),
        "ever_qualified_injected_track_count": len(qualified_truth_ids),
        "ever_qualified_injected_target_ids": sorted(
            {
                target_id
                for track_id in qualified_truth_ids
                for target_id in matched_targets[track_id]
            }
        ),
        "ever_qualified_unmatched_track_count": len(
            ever_qualified_ids - truth_track_ids
        ),
        "qualified_at_latest_observation_track_count": len(qualified_latest_ids),
        "qualified_at_latest_observation_unmatched_track_count": len(
            qualified_latest_ids - truth_track_ids
        ),
        "injected_track_first_qualification": qualification_details,
    }


def summarize_track_quality(
    batches: Sequence[Mapping[str, Any]],
    truth_windows: Sequence[Mapping[str, Any]],
    *,
    diagnostic_policy: DiagnosticQualityPolicy | None = None,
    motion_persistence_policy: DiagnosticMotionPersistencePolicy | None = None,
) -> dict[str, Any]:
    """Summarize latest evidence for every track that ever confirmed or coasted."""

    if len(batches) != len(truth_windows):
        raise EvaluationError("track batches and truth windows are not aligned")
    latest_tracks: dict[int, Mapping[str, Any]] = {}
    matched_targets: dict[int, set[str]] = {}
    first_qualified_timestamp_ns: dict[int, int] = {}
    first_motion_persistence_timestamp_ns: dict[int, int] = {}
    for batch, truth_window in zip(batches, truth_windows, strict=True):
        for match in truth_window["confirmed_track_matching"]["matches"]:
            matched_targets.setdefault(int(match["track_id"]), set()).add(
                str(match["target_id"])
            )
        for track in batch["confirmed_or_coasted_tracks"]:
            if not isinstance(track.get("quality_evidence"), Mapping):
                raise EvaluationError(
                    "track report lacks quality_evidence; rerun with instrumentation"
                )
            track_id = int(track["track_id"])
            if diagnostic_policy is not None and _passes_diagnostic_policy(
                track, diagnostic_policy
            ):
                first_qualified_timestamp_ns.setdefault(
                    track_id, int(track["state"]["timestamp_ns"])
                )
            if (
                motion_persistence_policy is not None
                and _passes_motion_persistence_policy(
                    track, motion_persistence_policy
                )
            ):
                first_motion_persistence_timestamp_ns.setdefault(
                    track_id, int(track["state"]["timestamp_ns"])
                )
            current = latest_tracks.get(track_id)
            if current is None or int(track["state"]["timestamp_ns"]) >= int(
                current["state"]["timestamp_ns"]
            ):
                latest_tracks[track_id] = track

    track_features = []
    truth_track_ids = set(matched_targets)
    unmatched_track_ids = set(latest_tracks) - truth_track_ids
    for track_id, track in sorted(latest_tracks.items()):
        values = {
            feature.name: _number_at(track, feature.path) for feature in _FEATURES
        }
        track_features.append(
            {
                "track_id": track_id,
                "classification": (
                    "injected_truth_matched"
                    if track_id in truth_track_ids
                    else "unmatched_to_injected_truth"
                ),
                "matched_target_ids": sorted(matched_targets.get(track_id, set())),
                "lifecycle_state_at_last_observation": track["lifecycle_state"],
                "state_timestamp_ns": track["state"]["timestamp_ns"],
                "state_xy_vx_vy": track["state"]["mean"],
                "birth_timestamp_ns": track["birth_timestamp_ns"],
                "confirmation_timestamp_ns": track["confirmation"]["timestamp_ns"],
                "feature_values": values,
            }
        )

    by_id = {int(item["track_id"]): item for item in track_features}
    feature_analysis = []
    for feature in _FEATURES:
        truth_values = [
            by_id[track_id]["feature_values"][feature.name]
            for track_id in sorted(truth_track_ids)
        ]
        unmatched_values = [
            by_id[track_id]["feature_values"][feature.name]
            for track_id in sorted(unmatched_track_ids)
        ]
        finite_truth = [float(value) for value in truth_values if value is not None]
        finite_unmatched = [
            float(value) for value in unmatched_values if value is not None
        ]
        threshold = None
        retained_unmatched = None
        retained_unmatched_ids = None
        if (
            feature.gate_direction is not None
            and finite_truth
            and len(finite_truth) == len(truth_values)
        ):
            threshold = (
                min(finite_truth)
                if feature.gate_direction == "minimum_required"
                else max(finite_truth)
            )
            retained_unmatched_ids = [
                track_id
                for track_id in sorted(unmatched_track_ids)
                if _passes(
                    by_id[track_id]["feature_values"][feature.name],
                    feature.gate_direction,
                    threshold,
                )
            ]
            retained_unmatched = len(retained_unmatched_ids)
        feature_analysis.append(
            {
                "feature": feature.name,
                "gate_direction": feature.gate_direction,
                "injected_truth_matched": _distribution(finite_truth),
                "unmatched_to_injected_truth": _distribution(finite_unmatched),
                "discovery_gate_retaining_all_injected_tracks": (
                    {
                        "threshold": threshold,
                        "retained_injected_track_count": len(truth_track_ids),
                        "retained_unmatched_track_count": retained_unmatched,
                        "retained_unmatched_track_ids": retained_unmatched_ids,
                        "rejected_unmatched_track_count": (
                            len(unmatched_track_ids) - int(retained_unmatched)
                        ),
                    }
                    if threshold is not None and retained_unmatched is not None
                    else None
                ),
            }
        )

    gates = {
        item["feature"]: (
            item["gate_direction"],
            item["discovery_gate_retaining_all_injected_tracks"]["threshold"],
        )
        for item in feature_analysis
        if item["discovery_gate_retaining_all_injected_tracks"] is not None
    }
    pair_envelopes = []
    for (first_name, first), (second_name, second) in combinations(gates.items(), 2):
        retained = [
            track_id
            for track_id in sorted(unmatched_track_ids)
            if _passes(
                by_id[track_id]["feature_values"][first_name], first[0], first[1]
            )
            and _passes(
                by_id[track_id]["feature_values"][second_name], second[0], second[1]
            )
        ]
        pair_envelopes.append(
            {
                "features": [first_name, second_name],
                "rules": [
                    {"direction": first[0], "threshold": first[1]},
                    {"direction": second[0], "threshold": second[1]},
                ],
                "retained_injected_track_count": len(truth_track_ids),
                "retained_unmatched_track_count": len(retained),
                "retained_unmatched_track_ids": retained,
                "rejected_unmatched_track_count": len(unmatched_track_ids) - len(retained),
            }
        )
    pair_envelopes.sort(
        key=lambda item: (
            item["retained_unmatched_track_count"],
            item["features"],
        )
    )
    diagnostic_policy_evaluation = None
    if diagnostic_policy is not None:
        diagnostic_policy_evaluation = _causal_policy_evaluation(
            latest_tracks,
            matched_targets,
            first_qualified_timestamp_ns,
            diagnostic_policy,
            _passes_diagnostic_policy,
        )
    motion_persistence_policy_evaluation = None
    if motion_persistence_policy is not None:
        motion_persistence_policy_evaluation = _causal_policy_evaluation(
            latest_tracks,
            matched_targets,
            first_motion_persistence_timestamp_ns,
            motion_persistence_policy,
            _passes_motion_persistence_policy,
        )
    return {
        "ever_confirmed_or_coasted_track_count": len(latest_tracks),
        "injected_truth_matched_track_count": len(truth_track_ids),
        "unmatched_to_injected_truth_track_count": len(unmatched_track_ids),
        "ambiguous_multi_truth_track_ids": sorted(
            track_id
            for track_id, target_ids in matched_targets.items()
            if len(target_ids) > 1
        ),
        "feature_analysis": feature_analysis,
        "best_discovery_only_two_feature_envelopes": pair_envelopes[:10],
        "diagnostic_causal_policy_evaluation": diagnostic_policy_evaluation,
        "diagnostic_motion_persistence_policy_evaluation": (
            motion_persistence_policy_evaluation
        ),
        "tracks": track_features,
    }


def analyze(
    report_path: str | Path,
    *,
    maximum_position_error_px: float = 3.0,
    maximum_velocity_error_px_s: float = 2.0,
    diagnostic_policy: DiagnosticQualityPolicy | None = None,
    motion_persistence_policy: DiagnosticMotionPersistencePolicy | None = None,
) -> dict[str, Any]:
    path, motion, digest = _read_motion_report(report_path)
    accuracy = analyze_injected_raw(
        path,
        maximum_position_error_px=maximum_position_error_px,
        maximum_velocity_error_px_s=maximum_velocity_error_px_s,
    )
    evaluation = motion.get("evaluation")
    if not isinstance(evaluation, Mapping):
        raise EvaluationError("motion report does not contain evaluation batches")
    sweep = evaluation.get("candidate_threshold_sweep")
    if not isinstance(sweep, Mapping):
        raise EvaluationError("motion report does not contain a candidate threshold sweep")
    threshold_reports = {}
    for threshold, batches in sorted(sweep.items(), key=lambda item: float(item[0])):
        truth_windows = accuracy["window_evidence"].get(str(threshold))
        if truth_windows is None:
            raise EvaluationError(f"accuracy evidence omits threshold {threshold}")
        threshold_reports[str(threshold)] = summarize_track_quality(
            batches,
            truth_windows,
            diagnostic_policy=diagnostic_policy,
            motion_persistence_policy=motion_persistence_policy,
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "run": run_identity(REPOSITORY),
        "implementation_sha256": _identity(),
        "input": {"path": str(path), "sha256": digest},
        "matching_gates": accuracy["matching_gates"],
        "threshold_parameter": accuracy["candidate_threshold_sweep_parameter"],
        "thresholds": threshold_reports,
        "interpretation": {
            "quality_evidence_used_by_live_tracker": False,
            "unmatched_tracks_are_false_tracks": False,
            "unmatched_note": (
                "Source content is unlabeled; unmatched means only that a track did "
                "not match injected truth. It may be structured clutter or a real object."
            ),
            "discovery_gate_status": (
                "Same-run diagnostic only. Thresholds are derived from injected-track "
                "extrema and require independent held-out validation before gating."
            ),
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-report", required=True, type=Path)
    parser.add_argument("--position-gate-px", type=float, default=3)
    parser.add_argument("--velocity-gate-px-s", type=float, default=2)
    parser.add_argument("--diagnostic-min-observations", type=int)
    parser.add_argument("--diagnostic-min-observation-fraction", type=float)
    parser.add_argument("--diagnostic-max-score-std", type=float)
    parser.add_argument("--diagnostic-min-peak-ratio", type=float)
    parser.add_argument("--motion-policy-min-observations", type=int)
    parser.add_argument("--motion-policy-min-observation-fraction", type=float)
    parser.add_argument("--motion-policy-min-speed-px-s", type=float)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    policy_values = (
        args.diagnostic_min_observations,
        args.diagnostic_min_observation_fraction,
        args.diagnostic_max_score_std,
        args.diagnostic_min_peak_ratio,
    )
    if any(value is not None for value in policy_values) and not all(
        value is not None for value in policy_values
    ):
        raise ValueError("all diagnostic quality-policy arguments must be supplied")
    diagnostic_policy = (
        DiagnosticQualityPolicy(
            minimum_observations=args.diagnostic_min_observations,
            minimum_observation_fraction=args.diagnostic_min_observation_fraction,
            maximum_detector_score_standard_deviation=(
                args.diagnostic_max_score_std
            ),
            minimum_peak_to_neighbor_ratio_mean=args.diagnostic_min_peak_ratio,
        )
        if all(value is not None for value in policy_values)
        else None
    )
    motion_policy_values = (
        args.motion_policy_min_observations,
        args.motion_policy_min_observation_fraction,
        args.motion_policy_min_speed_px_s,
    )
    if any(value is not None for value in motion_policy_values) and not all(
        value is not None for value in motion_policy_values
    ):
        raise ValueError("all motion-persistence policy arguments must be supplied")
    motion_persistence_policy = (
        DiagnosticMotionPersistencePolicy(
            minimum_observations=args.motion_policy_min_observations,
            minimum_observation_fraction=(
                args.motion_policy_min_observation_fraction
            ),
            minimum_measurement_speed_px_s=args.motion_policy_min_speed_px_s,
        )
        if all(value is not None for value in motion_policy_values)
        else None
    )
    report = analyze(
        args.motion_report,
        maximum_position_error_px=args.position_gate_px,
        maximum_velocity_error_px_s=args.velocity_gate_px_s,
        diagnostic_policy=diagnostic_policy,
        motion_persistence_policy=motion_persistence_policy,
    )
    if args.output is None:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        output = write_json_exclusive(args.output, report)
        print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
