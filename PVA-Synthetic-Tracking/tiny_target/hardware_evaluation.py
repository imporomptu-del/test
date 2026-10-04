"""Summarize checked Jetson pipeline and CUDA telemetry without overstating it."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
from typing import Any, Sequence

from .evaluation import EvaluationError, latency_summary
from .telemetry import run_identity, write_json_exclusive


REPOSITORY = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "seaqr.tiny-target.hardware-evaluation.v1"


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
        raise EvaluationError(f"Cannot read evaluation input {resolved}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvaluationError(f"Evaluation input must be a JSON object: {resolved}")
    return resolved, value, hashlib.sha256(raw).hexdigest()


def _totals(records: Sequence[dict[str, Any]]) -> list[float]:
    return [float(record["timings_ms"]["total"]) for record in records]


def analyze(
    motion_report_path: str | Path,
    cuda_report_path: str | Path,
    *,
    target_frame_rate_hz: float = 10.0,
) -> dict[str, Any]:
    motion_path, motion, motion_sha = _read(motion_report_path)
    cuda_path, cuda, cuda_sha = _read(cuda_report_path)
    if motion.get("schema_version") != "seaqr.tiny-target.motion.v9":
        raise EvaluationError("hardware evaluation requires a motion.v9 report")
    if cuda.get("schema_version") != "seaqr.tiny-target.cuda-tracking-benchmark.v1":
        raise EvaluationError("hardware evaluation requires a CUDA benchmark v1 report")
    summary = motion["summary"]
    frame_count = int(summary["frames_read"])
    wall_ms = float(summary["end_to_end_ms"])
    achieved_fps = frame_count / (wall_ms / 1000)
    alignments = [
        pair["image_alignment"]
        for pair in motion["pairs"]
        if pair["image_alignment"]["median_absolute_difference_after"] is not None
    ]
    stage_latency = {
        "pva_feature_motion": latency_summary(
            [float(pair["timings_ms"]["total"]) for pair in motion["pairs"]]
        ),
        "global_motion_fit": latency_summary(
            [float(pair["global_motion"]["timing_ms"]) for pair in motion["pairs"]]
        ),
        "full_resolution_stabilization": latency_summary(
            _totals(motion["stabilized_frames"])
        ),
        "preprocessing": latency_summary(_totals(motion["preprocessed_frames"])),
        "matched_filter": latency_summary(_totals(motion["matched_filter_frames"])),
        "cuda_synthetic_tracking": latency_summary(
            _totals(motion["synthetic_tracking_windows"])
        ),
        "candidate_extraction": latency_summary(_totals(motion["candidate_batches"])),
        "temporal_tracking": latency_summary(_totals(motion["track_batches"])),
    }
    cuda_windows = motion["synthetic_tracking_windows"]
    maximum_cuda_allocation = max(
        int(item["metrics"]["cuda"]["allocated_device_bytes"])
        for item in cuda_windows
    )
    maximum_upload = max(
        int(item["metrics"]["cuda"]["host_to_device_bytes"])
        for item in cuda_windows
    )
    maximum_download = max(
        int(item["metrics"]["cuda"]["device_to_host_bytes"])
        for item in cuda_windows
    )
    pva_memory = {
        key: max(int(pair["metrics"]["memory_bytes"][key]) for pair in motion["pairs"])
        for key in motion["pairs"][0]["metrics"]["memory_bytes"]
    }
    tegrastats = cuda["tegrastats"]
    return {
        "schema_version": SCHEMA_VERSION,
        "run": run_identity(REPOSITORY),
        "implementation_sha256": _identity(),
        "inputs": {
            "motion_report": {"path": str(motion_path), "sha256": motion_sha},
            "cuda_report": {"path": str(cuda_path), "sha256": cuda_sha},
        },
        "throughput": {
            "input_frames": frame_count,
            "completed_output_frames": frame_count,
            "wall_time_ms": wall_ms,
            "input_frames_per_second": achieved_fps,
            "completed_output_frames_per_second": achieved_fps,
            "target_frame_rate_hz": target_frame_rate_hz,
            "fraction_of_target_rate": achieved_fps / target_frame_rate_hz,
            "real_time_target_met": achieved_fps >= target_frame_rate_hz,
            "dropped_frames": None,
            "drop_accounting_note": (
                "Recorded pull-based replay blocks the decoder and has no application "
                "drop counter; this is not a live-camera drop measurement."
            ),
        },
        "latency": {
            "end_to_end_wall_mean_ms_per_frame": wall_ms / frame_count,
            "end_to_end_distribution": None,
            "end_to_end_distribution_note": (
                "motion.v9 records aggregate wall time and per-stage samples but not "
                "capture-to-completion latency for every frame. Controlled Phase 11 "
                "modes provide median/p90/p95/p99/max distributions."
            ),
            "per_stage": stage_latency,
        },
        "motion_and_stabilization": {
            "accepted_transform_fraction": (
                summary["accepted_global_transforms"] / summary["pairs_processed"]
            ),
            "stabilization_failure_reset_rate": summary["stabilization"][
                "failure_reset_rate"
            ],
            "comparable_alignment_pairs": len(alignments),
            "median_absolute_difference_before": (
                statistics.median(
                    float(item["median_absolute_difference_before"])
                    for item in alignments
                )
                if alignments
                else None
            ),
            "median_absolute_difference_after": (
                statistics.median(
                    float(item["median_absolute_difference_after"])
                    for item in alignments
                )
                if alignments
                else None
            ),
        },
        "memory_and_transfers": {
            "maximum_pva_pair_memory_bytes_by_buffer": pva_memory,
            "maximum_cuda_allocated_device_bytes": maximum_cuda_allocation,
            "maximum_cuda_host_to_device_bytes_per_window": maximum_upload,
            "maximum_cuda_device_to_host_bytes_per_window": maximum_download,
            "maximum_system_ram_used_mb_from_cuda_stress_run": tegrastats[
                "maximum_ram_used_mb"
            ],
        },
        "hardware_utilization": {
            "pva_utilization_percent": None,
            "pva_note": "VPI exposes submit/sync timings but no PVA utilization counter here.",
            "gpu_utilization_percent": tegrastats["gpu_utilization_percent"],
            "gpu_soc_power_mw": tegrastats["gpu_soc_power_mw"],
            "temperature_c": tegrastats["maximum_temperature_c"],
            "cuda_theoretical_occupancy_fraction": cuda[
                "maximum_intended_workload"
            ]["cuda_metrics"]["theoretical_occupancy_fraction"],
            "clock_and_throttling_events": None,
            "clock_note": "No per-sample clock/throttling event stream was retained.",
        },
        "detection_and_tracking": {
            "candidate_summary": summary["candidate_extraction"],
            "track_summary": summary["temporal_tracking"],
            "false_alarm_metrics_valid": False,
            "reason": (
                "The RAW16 recording is unlabeled and not verified empty; unmatched "
                "candidates cannot be declared false alarms."
            ),
        },
        "claims": {
            "real_time": False,
            "thermal_stability": (
                "short CUDA stress run completed without observed thermal instability"
            ),
            "soak_stability": False,
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-report", required=True, type=Path)
    parser.add_argument("--cuda-report", required=True, type=Path)
    parser.add_argument("--target-frame-rate-hz", type=float, default=10)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = analyze(
        args.motion_report,
        args.cuda_report,
        target_frame_rate_hz=args.target_frame_rate_hz,
    )
    if args.output is None:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        output = write_json_exclusive(args.output, report)
        print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
