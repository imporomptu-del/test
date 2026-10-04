"""Evaluate injected RAW targets in stabilized candidate coordinates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .evaluation import EvaluationError, SyntheticTarget, transformed_target_truth
from .telemetry import run_identity, write_json_exclusive


REPOSITORY = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "seaqr.tiny-target.injected-raw-evaluation.v1"


def _identity() -> dict[str, str]:
    files = (Path(__file__), Path(__file__).parent / "evaluation" / "core.py")
    return {
        str(path.relative_to(REPOSITORY)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in files
    }


def _read(path: str | Path) -> tuple[Path, dict[str, Any], str]:
    resolved = Path(path).expanduser().resolve()
    try:
        raw = resolved.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"Cannot read injected motion report: {exc}") from exc
    if not isinstance(value, dict) or value.get("schema_version") != "seaqr.tiny-target.motion.v10":
        raise EvaluationError("injected evaluation requires a motion.v10 report")
    return resolved, value, hashlib.sha256(raw).hexdigest()


def _match_serialized(
    candidates: Sequence[Mapping[str, Any]],
    truths: Sequence[Mapping[str, Any]],
    position_gate_px: float,
    velocity_gate_px_s: float,
) -> dict[str, Any]:
    options = []
    for truth_index, truth in enumerate(truths):
        for candidate_index, candidate in enumerate(candidates):
            position_error = float(
                np.linalg.norm(
                    np.asarray(candidate["discrete_position_xy_px"], np.float64)
                    - truth["position_xy_px"]
                )
            )
            velocity_error = float(
                np.linalg.norm(
                    np.asarray(candidate["discrete_velocity_xy_px_s"], np.float64)
                    - truth["velocity_xy_px_s"]
                )
            )
            if position_error <= position_gate_px and velocity_error <= velocity_gate_px_s:
                options.append(
                    (
                        (position_error / position_gate_px) ** 2
                        + (velocity_error / velocity_gate_px_s) ** 2,
                        str(truth["target_id"]),
                        int(candidate["candidate_index"]),
                        truth_index,
                        candidate_index,
                        position_error,
                        velocity_error,
                    )
                )
    options.sort(key=lambda item: item[:3])
    used_truths: set[int] = set()
    used_candidates: set[int] = set()
    matches = []
    for _, _, _, truth_index, candidate_index, position_error, velocity_error in options:
        if truth_index in used_truths or candidate_index in used_candidates:
            continue
        used_truths.add(truth_index)
        used_candidates.add(candidate_index)
        matches.append(
            {
                "target_id": truths[truth_index]["target_id"],
                "flux_dn": truths[truth_index]["flux_dn"],
                "candidate_index": candidates[candidate_index]["candidate_index"],
                "score_snr": candidates[candidate_index]["normalized_score_snr"],
                "selection_score": candidates[candidate_index]
                .get("selection", {})
                .get("ranking_score"),
                "selection_score_units": candidates[candidate_index]
                .get("selection", {})
                .get("ranking_score_units"),
                "position_error_px": position_error,
                "velocity_error_px_s": velocity_error,
            }
        )
    return {
        "matches": matches,
        "false_negative_target_ids": [
            truth["target_id"]
            for index, truth in enumerate(truths)
            if index not in used_truths
        ],
        "unmatched_candidate_count": len(candidates) - len(used_candidates),
    }


def _match_tracks_serialized(
    tracks: Sequence[Mapping[str, Any]],
    truths: Sequence[Mapping[str, Any]],
    position_gate_px: float,
    velocity_gate_px_s: float,
) -> dict[str, Any]:
    options = []
    for truth_index, truth in enumerate(truths):
        for track_index, track in enumerate(tracks):
            state = np.asarray(track["state"]["mean"], np.float64)
            position_error = float(
                np.linalg.norm(state[:2] - truth["position_xy_px"])
            )
            velocity_error = float(
                np.linalg.norm(state[2:] - truth["velocity_xy_px_s"])
            )
            if position_error <= position_gate_px and velocity_error <= velocity_gate_px_s:
                options.append(
                    (
                        (position_error / position_gate_px) ** 2
                        + (velocity_error / velocity_gate_px_s) ** 2,
                        str(truth["target_id"]),
                        int(track["track_id"]),
                        truth_index,
                        track_index,
                        position_error,
                        velocity_error,
                    )
                )
    options.sort(key=lambda item: item[:3])
    used_truths: set[int] = set()
    used_tracks: set[int] = set()
    matches = []
    for _, _, _, truth_index, track_index, position_error, velocity_error in options:
        if truth_index in used_truths or track_index in used_tracks:
            continue
        used_truths.add(truth_index)
        used_tracks.add(track_index)
        truth = truths[truth_index]
        track = tracks[track_index]
        confirmation = track["confirmation"]
        matches.append(
            {
                "target_id": truth["target_id"],
                "flux_dn": truth["flux_dn"],
                "track_id": track["track_id"],
                "lifecycle_state": track["lifecycle_state"],
                "position_error_px": position_error,
                "velocity_error_px_s": velocity_error,
                "confirmation_timestamp_ns": confirmation.get("timestamp_ns"),
                "confirmation_latency_s": confirmation.get("latency_s"),
                "independent_confirmation_hits": confirmation.get(
                    "independent_hits"
                ),
            }
        )
    return {
        "matches": matches,
        "unmatched_confirmed_or_coasted_track_ids": [
            int(track["track_id"])
            for index, track in enumerate(tracks)
            if index not in used_tracks
        ],
    }


def _error(values: Sequence[float]) -> dict[str, float] | None:
    if not values:
        return None
    array = np.asarray(values, np.float64)
    return {
        "rmse": float(np.sqrt(np.mean(array**2))),
        "median": float(np.median(array)),
        "maximum": float(np.max(array)),
    }


def _probe_summary(windows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for window in windows:
        for target in window.get("targets", []):
            grouped.setdefault(f"{float(target['flux_dn']):g}", []).append(target)
    result = {}
    for flux, probes in sorted(grouped.items(), key=lambda item: float(item[0])):
        valid = [probe for probe in probes if probe.get("valid_score_available")]
        scores = [float(probe["local_peak_score_snr"]) for probe in valid]
        ranks = [int(probe["surface_rank_lower_bound"]) for probe in valid]
        selection_valid = [
            probe for probe in probes if probe.get("selection_score_available")
        ]
        selection_scores = [
            float(probe["local_peak_selection_score"])
            for probe in selection_valid
        ]
        selection_ranks = [
            int(probe["selection_surface_rank_lower_bound"])
            for probe in selection_valid
        ]
        selection_units = sorted(
            {
                str(probe["selection_score_units"])
                for probe in selection_valid
            }
        )
        result[flux] = {
            "window_count": len(probes),
            "valid_probe_count": len(valid),
            "local_peak_score_snr": (
                {
                    "minimum": min(scores),
                    "median": float(np.median(scores)),
                    "maximum": max(scores),
                }
                if scores
                else None
            ),
            "surface_rank_lower_bound": (
                {
                    "best": min(ranks),
                    "median": float(np.median(ranks)),
                    "worst": max(ranks),
                }
                if ranks
                else None
            ),
            "selection_probe_count": len(selection_valid),
            "selection_score_units": (
                selection_units[0] if len(selection_units) == 1 else selection_units
            ),
            "local_peak_selection_score": (
                {
                    "minimum": min(selection_scores),
                    "median": float(np.median(selection_scores)),
                    "maximum": max(selection_scores),
                }
                if selection_scores
                else None
            ),
            "selection_surface_rank_lower_bound": (
                {
                    "best": min(selection_ranks),
                    "median": float(np.median(selection_ranks)),
                    "worst": max(selection_ranks),
                }
                if selection_ranks
                else None
            ),
        }
    return result


def analyze(
    report_path: str | Path,
    *,
    maximum_position_error_px: float = 3.0,
    maximum_velocity_error_px_s: float = 2.0,
) -> dict[str, Any]:
    path, report, digest = _read(report_path)
    evaluation = report.get("evaluation")
    if not isinstance(evaluation, Mapping) or not isinstance(
        evaluation.get("injection"), Mapping
    ):
        raise EvaluationError("motion report does not contain injection evidence")
    spec = evaluation["injection"]["specification"]
    targets = tuple(
        SyntheticTarget.from_mapping(item) for item in spec["targets"]
    )
    timestamps = {
        int(item["frame_index"]): int(item["timestamp_ns"])
        for item in report["preprocessed_frames"]
    }
    frame_metadata = {
        int(item["frame_index"]): (
            timestamps[int(item["frame_index"])],
            np.asarray(item["source_to_reference_matrix"], np.float64),
        )
        for item in report["stabilized_frames"]
    }
    curves = []
    detailed = {}
    truth_probes = evaluation.get("injected_truth_score_probes", [])
    probes_by_window = {
        (
            tuple(int(value) for value in window["frame_indices"]),
            int(window["reference_timestamp_ns"]),
        ): {
            str(target["target_id"]): bool(target.get("valid_score_available"))
            for target in window.get("targets", [])
        }
        for window in truth_probes
    }
    threshold_sweep_parameter = str(
        evaluation.get(
            "candidate_threshold_sweep_parameter",
            "score_threshold_snr",
        )
    )
    if threshold_sweep_parameter not in {
        "score_threshold_snr",
        "cfar_threshold_sigma",
    }:
        raise EvaluationError(
            "unsupported candidate threshold sweep parameter: "
            f"{threshold_sweep_parameter}"
        )
    for threshold_text, batches in sorted(
        evaluation["candidate_threshold_sweep"].items(),
        key=lambda item: float(item[0]),
    ):
        truth_count = 0
        valid_support_truth_count = 0
        valid_support_match_count = 0
        support_evidence_complete = True
        matches = []
        unmatched_candidates = 0
        windows = []
        by_flux: dict[str, list[int]] = {}
        by_flux_valid_support: dict[str, list[int]] = {}
        confirmed_ids: set[int] = set()
        truth_matched_confirmed_track_ids: set[int] = set()
        confirmed_target_ids: set[str] = set()
        first_target_confirmations: dict[str, dict[str, Any]] = {}
        for batch in batches:
            candidate_batch = batch["candidate_batch"]
            truths = [
                truth
                for target in targets
                if (
                    truth := transformed_target_truth(
                        target,
                        candidate_batch["frame_indices"],
                        candidate_batch["reference_timestamp_ns"],
                        frame_metadata,
                    )
                )
                is not None
            ]
            result = _match_serialized(
                candidate_batch["candidates"],
                truths,
                maximum_position_error_px,
                maximum_velocity_error_px_s,
            )
            truth_count += len(truths)
            matches.extend(result["matches"])
            window_support = probes_by_window.get(
                (
                    tuple(int(value) for value in candidate_batch["frame_indices"]),
                    int(candidate_batch["reference_timestamp_ns"]),
                )
            )
            support_evidence_complete &= window_support is not None and all(
                str(truth["target_id"]) in window_support for truth in truths
            )
            valid_support_ids = {
                target_id
                for target_id, is_valid in (window_support or {}).items()
                if is_valid
            }
            valid_support_truth_count += sum(
                str(truth["target_id"]) in valid_support_ids for truth in truths
            )
            valid_support_match_count += sum(
                str(item["target_id"]) in valid_support_ids
                for item in result["matches"]
            )
            unmatched_candidates += result["unmatched_candidate_count"]
            matched_ids = {item["target_id"] for item in result["matches"]}
            for truth in truths:
                flux = f"{truth['flux_dn']:g}"
                counts = by_flux.setdefault(flux, [0, 0])
                counts[0] += int(truth["target_id"] in matched_ids)
                counts[1] += 1
                support_counts = by_flux_valid_support.setdefault(flux, [0, 0])
                if str(truth["target_id"]) in valid_support_ids:
                    support_counts[0] += int(truth["target_id"] in matched_ids)
                    support_counts[1] += 1
            confirmed_ids.update(
                int(track["track_id"])
                for track in batch["confirmed_or_coasted_tracks"]
            )
            track_result = _match_tracks_serialized(
                batch["confirmed_or_coasted_tracks"],
                truths,
                maximum_position_error_px,
                maximum_velocity_error_px_s,
            )
            for match in track_result["matches"]:
                target_id = str(match["target_id"])
                confirmed_target_ids.add(target_id)
                truth_matched_confirmed_track_ids.add(int(match["track_id"]))
                current = first_target_confirmations.get(target_id)
                timestamp = match["confirmation_timestamp_ns"]
                if current is None or (
                    timestamp is not None
                    and (
                        current["confirmation_timestamp_ns"] is None
                        or int(timestamp) < int(current["confirmation_timestamp_ns"])
                    )
                ):
                    first_target_confirmations[target_id] = dict(match)
            windows.append(
                {
                    "frame_indices": candidate_batch["frame_indices"],
                    "truths": truths,
                    "matching": result,
                    "confirmed_track_matching": track_result,
                    "candidate_metrics": candidate_batch["metrics"],
                }
            )
        curve = {
            "threshold_parameter": threshold_sweep_parameter,
            "threshold_value": float(threshold_text),
            "truth_opportunity_count": truth_count,
            "matched_count": len(matches),
            "probability_of_detection": (
                len(matches) / truth_count if truth_count else None
            ),
            "probability_of_detection_by_flux_dn": {
                flux: detected / total
                for flux, (detected, total) in sorted(by_flux.items())
            },
            "valid_support_evidence_complete": support_evidence_complete,
            "valid_support_truth_opportunity_count": (
                valid_support_truth_count if support_evidence_complete else None
            ),
            "matched_valid_support_count": (
                valid_support_match_count if support_evidence_complete else None
            ),
            "probability_of_detection_given_valid_support": (
                valid_support_match_count / valid_support_truth_count
                if support_evidence_complete and valid_support_truth_count
                else None
            ),
            "probability_of_detection_by_flux_dn_given_valid_support": (
                {
                    flux: detected / total if total else None
                    for flux, (detected, total) in sorted(
                        by_flux_valid_support.items()
                    )
                }
                if support_evidence_complete
                else None
            ),
            "unmatched_candidate_burden": unmatched_candidates,
            "unmatched_candidates_are_false_alarms": False,
            "localization_error_px": _error(
                [item["position_error_px"] for item in matches]
            ),
            "velocity_error_px_s": _error(
                [item["velocity_error_px_s"] for item in matches]
            ),
            "confirmed_or_coasted_track_count": len(confirmed_ids),
            "truth_matched_confirmed_track_count": len(
                truth_matched_confirmed_track_ids
            ),
            "unmatched_confirmed_or_coasted_track_count": len(
                confirmed_ids - truth_matched_confirmed_track_ids
            ),
            "confirmed_injected_target_ids": sorted(confirmed_target_ids),
            "confirmed_injected_target_count": len(confirmed_target_ids),
            "valid_support_injected_target_count": len(
                {
                    target_id
                    for support in probes_by_window.values()
                    for target_id, is_valid in support.items()
                    if is_valid
                }
            ),
            "confirmed_injected_target_probability_given_any_valid_support": (
                len(confirmed_target_ids)
                / len(
                    {
                        target_id
                        for support in probes_by_window.values()
                        for target_id, is_valid in support.items()
                        if is_valid
                    }
                )
                if probes_by_window
                and any(
                    is_valid
                    for support in probes_by_window.values()
                    for is_valid in support.values()
                )
                else None
            ),
            "first_confirmation_by_injected_target": {
                target_id: first_target_confirmations[target_id]
                for target_id in sorted(first_target_confirmations)
            },
        }
        curve[threshold_sweep_parameter] = float(threshold_text)
        curves.append(curve)
        detailed[threshold_text] = windows
    frame_records = evaluation["injection"]["frame_records"]
    requested = sum(
        float(event["in_bounds_requested_flux_dn"])
        for frame in frame_records
        for event in frame["events"]
    )
    achieved = sum(
        float(frame["achieved_total_flux_dn_after_quantization_and_clipping"])
        for frame in frame_records
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "run": run_identity(REPOSITORY),
        "implementation_sha256": _identity(),
        "input": {"path": str(path), "sha256": digest},
        "injection_identity": evaluation["injection"]["identity"],
        "injection_stage": "decoded_source_before_motion_stabilization_preprocessing",
        "injection_flux": {
            "in_bounds_requested_total_dn": requested,
            "achieved_total_dn_after_quantization_and_clipping": achieved,
            "retention_fraction_at_source": achieved / requested if requested else None,
        },
        "matching_gates": {
            "maximum_position_error_px": maximum_position_error_px,
            "maximum_velocity_error_px_s": maximum_velocity_error_px_s,
        },
        "candidate_threshold_sweep_parameter": threshold_sweep_parameter,
        "accuracy_curve_points": curves,
        "injected_truth_score_probe_summary_by_flux_dn": _probe_summary(
            truth_probes
        ),
        "injected_truth_score_probes": truth_probes,
        "window_evidence": detailed,
        "false_alarm_metrics_valid": False,
        "false_alarm_note": (
            "Underlying RAW16 content is unlabeled; unmatched candidates are burden, "
            "not verified false alarms."
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-report", required=True, type=Path)
    parser.add_argument("--position-gate-px", type=float, default=3)
    parser.add_argument("--velocity-gate-px-s", type=float, default=2)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = analyze(
        args.motion_report,
        maximum_position_error_px=args.position_gate_px,
        maximum_velocity_error_px_s=args.velocity_gate_px_s,
    )
    if args.output is None:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        output = write_json_exclusive(args.output, report)
        print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
