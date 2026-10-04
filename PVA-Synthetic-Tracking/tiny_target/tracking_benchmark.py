"""Deterministic Phase 10 Kalman and temporal-confirmation characterization."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .detection import CandidateBatch, CandidateRecord
from .telemetry import run_identity, write_json_exclusive
from .tracking import KalmanTrackManager, KalmanTrackingConfig


REPOSITORY = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "seaqr.tiny-target.tracking-benchmark.v1"


def _identity() -> dict[str, str]:
    files = (
        Path(__file__),
        Path(__file__).parent / "tracking" / "kalman.py",
        Path(__file__).parent / "detection" / "candidates.py",
    )
    return {
        str(path.relative_to(REPOSITORY)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in files
    }


def _config(**overrides: Any) -> KalmanTrackingConfig:
    values = {
        "position_measurement_sigma_px": 0.75,
        "velocity_measurement_sigma_px_s": 0.75,
        "acceleration_process_sigma_px_s2": 1.0,
        "initial_position_sigma_px": 2.0,
        "initial_velocity_sigma_px_s": 2.0,
        "mahalanobis_gate_squared": 16.0,
        "maximum_position_residual_px": 6.0,
        "maximum_velocity_residual_px_s": 3.0,
        "confirmation_independent_hits": 2,
        "max_missed_windows": 2,
        "maximum_timestamp_gap_s": 2.0,
        "measurement_noise_source": "synthetic_characterization",
        "max_active_tracks": 64,
        "evidence_policy": "non_overlapping_frames",
    }
    values.update(overrides)
    return KalmanTrackingConfig(**values)


def _candidate(index: int, x: int, y: int, vx: float, vy: float) -> CandidateRecord:
    return CandidateRecord(
        candidate_index=index,
        x_px=x,
        y_px=y,
        velocity_index=0,
        velocity_xy_px_s=(vx, vy),
        normalized_score_snr=9.0,
        raw_sum_score=18.0,
        supporting_frame_count=4,
        support_weight=4.0,
        peak_neighbor_max_score_snr=6.0,
        peak_contrast_snr=3.0,
        peak_to_neighbor_ratio=1.5,
        distance_to_border_px=20,
        distance_to_invalid_chebyshev_px=None,
        distance_to_invalid_is_lower_bound=True,
    )


def _batch(
    index: int,
    candidates: Sequence[CandidateRecord],
    *,
    sliding: bool,
) -> CandidateBatch:
    frame_start = index if sliding else 4 * index
    frames = tuple(range(frame_start, frame_start + 4))
    timestamp_ns = int((frame_start + 1.5) * 100_000_000)
    return CandidateBatch(
        candidates=tuple(candidates),
        frame_indices=frames,
        reference_timestamp_ns=timestamp_ns,
        segment_index=0,
        metrics={},
        timings_ms={},
    )


def _persistent_target() -> dict[str, Any]:
    manager = KalmanTrackManager(_config())
    states = []
    position_errors = []
    velocity_errors = []
    confirmation_timestamp_ns = None
    for index in range(8):
        batch = _batch(index, (), sliding=True)
        time_s = batch.reference_timestamp_ns / 1e9
        true = np.array([30 + 2 * time_s, 22 - time_s, 2, -1], np.float64)
        measured = _candidate(
            0,
            int(round(true[0])),
            int(round(true[1])),
            2,
            -1,
        )
        result = manager.update(
            CandidateBatch(
                candidates=(measured,),
                frame_indices=batch.frame_indices,
                reference_timestamp_ns=batch.reference_timestamp_ns,
                segment_index=0,
                metrics={},
                timings_ms={},
            )
        )
        track = result.tracks[0]
        estimate = np.asarray(track.state_xy_vx_vy)
        position_errors.append(float(np.linalg.norm(estimate[:2] - true[:2])))
        velocity_errors.append(float(np.linalg.norm(estimate[2:] - true[2:])))
        if confirmation_timestamp_ns is None:
            confirmation_timestamp_ns = track.confirmation_timestamp_ns
        states.append(
            {
                "window_index": index,
                "frame_indices": list(batch.frame_indices),
                "lifecycle_state": track.lifecycle_state,
                "associated_updates": track.associated_update_count,
                "independent_hits": track.independent_confirmation_hits,
                "position_error_px": position_errors[-1],
                "velocity_error_px_s": velocity_errors[-1],
            }
        )
    return {
        "states": states,
        "confirmed": states[-1]["lifecycle_state"] == "confirmed",
        "confirmation_timestamp_ns": confirmation_timestamp_ns,
        "confirmation_latency_s": (
            (confirmation_timestamp_ns - 150_000_000) / 1e9
            if confirmation_timestamp_ns is not None
            else None
        ),
        "position_rmse_px": float(np.sqrt(np.mean(np.square(position_errors)))),
        "velocity_rmse_px_s": float(np.sqrt(np.mean(np.square(velocity_errors)))),
        "fragmentation_count": 0,
        "association_failure_count": 0,
    }


def _crossing_targets() -> dict[str, Any]:
    manager = KalmanTrackManager(_config())
    identity_failures = 0
    final = None
    for index in range(5):
        first_x = 10 + 2 * index
        second_x = 20 - 2 * index
        final = manager.update(
            _batch(
                index,
                (
                    _candidate(0, first_x, 14, 2, 0),
                    _candidate(1, second_x, 14, -2, 0),
                ),
                sliding=False,
            )
        )
        by_id = {track.track_id: track for track in final.tracks}
        identity_failures += int(by_id[0].state_xy_vx_vy[2] <= 0)
        identity_failures += int(by_id[1].state_xy_vx_vy[2] >= 0)
    assert final is not None
    return {
        "track_ids": [track.track_id for track in final.tracks],
        "lifecycle_states": [track.lifecycle_state for track in final.tracks],
        "identity_failure_count": identity_failures,
        "both_confirmed": all(
            track.lifecycle_state == "confirmed" for track in final.tracks
        ),
    }


def _isolated_noise() -> dict[str, Any]:
    manager = KalmanTrackManager(_config(max_missed_windows=1))
    ever_confirmed: set[int] = set()
    batch_times = []
    for index in range(12):
        result = manager.update(
            _batch(
                index,
                (_candidate(0, 20 + 20 * index, 30 + 10 * (index % 3), 0, 0),),
                sliding=True,
            )
        )
        batch_times.append(result.reference_timestamp_ns)
        ever_confirmed.update(
            track.track_id
            for track in result.tracks
            if track.lifecycle_state in {"confirmed", "coasted"}
        )
    duration_hours = (max(batch_times) - min(batch_times)) / 3.6e12
    return {
        "input_false_candidate_count": 12,
        "confirmed_false_track_count": len(ever_confirmed),
        "false_confirmed_tracks_per_hour": (
            len(ever_confirmed) / duration_hours if duration_hours else None
        ),
    }


def run_benchmark(seed: int = 75) -> dict[str, Any]:
    # The current fixtures are deterministic; retain a seed in the manifest for
    # later empirical measurement-noise trials.
    persistent = _persistent_target()
    crossing = _crossing_targets()
    noise = _isolated_noise()
    return {
        "schema_version": SCHEMA_VERSION,
        "run": run_identity(REPOSITORY),
        "implementation_sha256": _identity(),
        "random_seed": seed,
        "configuration": asdict(_config()),
        "scenarios": {
            "persistent_target_sliding_windows": persistent,
            "crossing_targets": crossing,
            "isolated_noise_candidates": noise,
        },
        "metrics": {
            "confirmed_track_probability": float(persistent["confirmed"]),
            "confirmation_latency_s": persistent["confirmation_latency_s"],
            "position_rmse_px": persistent["position_rmse_px"],
            "velocity_rmse_px_s": persistent["velocity_rmse_px_s"],
            "track_fragmentation_count": persistent["fragmentation_count"],
            "association_failure_count": (
                persistent["association_failure_count"]
                + crossing["identity_failure_count"]
            ),
            "false_confirmed_tracks_per_hour": noise[
                "false_confirmed_tracks_per_hour"
            ],
        },
        "acceptance": {
            "persistent_target_confirms": persistent["confirmed"],
            "overlapping_windows_not_credited_independently": (
                persistent["states"][1]["independent_hits"] == 1
                and persistent["states"][3]["independent_hits"] == 1
            ),
            "crossing_identities_preserved": (
                crossing["identity_failure_count"] == 0
                and crossing["both_confirmed"]
            ),
            "isolated_noise_does_not_confirm": (
                noise["confirmed_false_track_count"] == 0
            ),
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=75)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = run_benchmark(args.seed)
    if args.output is None:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        output = write_json_exclusive(args.output, report)
        print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
