"""Deterministic Phase 9 candidate-extraction characterization."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .detection import (
    CandidateExtractionConfig,
    CandidateExtractor,
    MatchedFilterFrame,
    ReferenceShiftAndStack,
    SyntheticTrackingConfig,
)
from .telemetry import run_identity, write_json_exclusive


REPOSITORY = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "seaqr.tiny-target.candidate-benchmark.v1"


def _identity() -> dict[str, str]:
    files = (
        Path(__file__),
        Path(__file__).parent / "detection" / "candidates.py",
        Path(__file__).parent / "detection" / "synthetic_reference.py",
        Path(__file__).parent / "detection" / "synthetic_types.py",
    )
    return {
        str(path.relative_to(REPOSITORY)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in files
    }


def _frame(response: np.ndarray, index: int, timestamp_ns: int) -> MatchedFilterFrame:
    valid = np.ones(response.shape, bool)
    return MatchedFilterFrame(
        response=np.asarray(response, np.float32),
        phase_index=np.zeros(response.shape, np.uint16),
        valid_mask=valid,
        valid_support_count=np.ones(response.shape, np.uint16),
        timestamp_ns=timestamp_ns,
        frame_index=index,
        reference_frame_index=0,
        segment_index=0,
        detection_ready=True,
        polarity="bright",
        backend="synthetic_candidate_benchmark",
        kernel_metadata={},
        metrics={},
        timings_ms={},
    )


def _gaussian(shape: tuple[int, int], x: float, y: float, amplitude: float) -> np.ndarray:
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    return (
        amplitude * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 0.7**2))
    ).astype(np.float32)


def _run_scenario(
    seed: int,
    tracks: Sequence[tuple[tuple[float, float], tuple[float, float]]],
) -> dict[str, Any]:
    shape = (48, 64)
    timestamps = (0, 500_000_000, 1_000_000_000, 1_500_000_000, 2_000_000_000)
    reference_timestamp_ns = 1_000_000_000
    rng = np.random.default_rng(seed)
    frames = []
    for index, timestamp_ns in enumerate(timestamps):
        response = rng.normal(0, 1, shape).astype(np.float32)
        dt_s = (timestamp_ns - reference_timestamp_ns) / 1e9
        for (x, y), (vx, vy) in tracks:
            response += _gaussian(shape, x + vx * dt_s, y + vy * dt_s, 5.0)
        frames.append(_frame(response, index, timestamp_ns))
    integration = ReferenceShiftAndStack(
        SyntheticTrackingConfig(
            window_frames=len(frames),
            window_stride_frames=1,
            vx_min_px_s=-3,
            vx_max_px_s=3,
            vy_min_px_s=-3,
            vy_max_px_s=3,
            velocity_step_px_s=1,
            min_valid_fraction=1,
            tile_rows=16,
        )
    ).integrate(frames)
    candidates = CandidateExtractor(
        CandidateExtractionConfig(
            score_threshold_snr=7,
            minimum_support_frames=5,
            local_maximum_radius_px=1,
            spatial_nms_radius_px=2.5,
            velocity_nms_radius_px_s=1.5,
            border_margin_px=3,
            invalid_margin_px=1,
            pre_nms_candidate_limit=128,
            max_candidates_per_window=32,
        )
    ).extract(integration)
    expected = []
    matched_indices: set[int] = set()
    for position, velocity in tracks:
        best = None
        for index, candidate in enumerate(candidates.candidates):
            position_error = float(
                np.hypot(candidate.x_px - position[0], candidate.y_px - position[1])
            )
            velocity_error = float(
                np.linalg.norm(np.asarray(candidate.velocity_xy_px_s) - velocity)
            )
            key = (position_error, velocity_error, index)
            if best is None or key < best[0]:
                best = (key, index, candidate)
        assert best is not None or not candidates.candidates
        if best is None:
            expected.append({"recovered": False})
        else:
            key, index, candidate = best
            recovered = key[0] <= 1.5 and key[1] <= 1.5
            if recovered:
                matched_indices.add(index)
            expected.append(
                {
                    "reference_position_xy_px": list(position),
                    "velocity_xy_px_s": list(velocity),
                    "recovered": recovered,
                    "position_error_px": key[0],
                    "velocity_error_px_s": key[1],
                    "matched_candidate_index": index,
                    "matched_candidate": candidate.to_dict(),
                }
            )
    return {
        "target_count": len(tracks),
        "candidate_count": len(candidates.candidates),
        "all_targets_recovered": all(item["recovered"] for item in expected),
        "unmatched_candidate_count": len(candidates.candidates) - len(matched_indices),
        "expected_tracks": expected,
        "candidate_batch": candidates.to_dict(),
    }


def run_benchmark(seed: int = 75) -> dict[str, Any]:
    scenarios = {
        "one_target": _run_scenario(seed, [((30, 24), (2, -1))]),
        "two_close_targets": _run_scenario(
            seed + 1,
            [((27, 23), (-2, 0)), ((33, 25), (2, 1))],
        ),
        "crossing_tracks": _run_scenario(
            seed + 2,
            [((28, 21), (2, 0)), ((34, 21), (-2, 0))],
        ),
        "no_target": _run_scenario(seed + 3, []),
    }
    threshold = 7.0
    velocity_trials = 49
    pixel_trials = 48 * 64
    gaussian_one_sided_tail = 0.5 * math.erfc(threshold / math.sqrt(2))
    return {
        "schema_version": SCHEMA_VERSION,
        "run": run_identity(REPOSITORY),
        "implementation_sha256": _identity(),
        "random_seed": seed,
        "operating_point": {
            "score_threshold_snr": threshold,
            "threshold_basis": (
                "unit-Gaussian normalized-noise union bound plus deterministic "
                "noise-only characterization; not a calibrated real-camera "
                "operating point"
            ),
            "noise_model": "independent unit Gaussian per matched-response pixel",
            "gaussian_one_sided_tail_probability_per_trial": gaussian_one_sided_tail,
            "pixel_count": pixel_trials,
            "velocity_trial_count": velocity_trials,
            "trial_count_upper_bound_per_window": pixel_trials * velocity_trials,
            "union_bound_probability_any_threshold_exceedance_per_window": (
                gaussian_one_sided_tail * pixel_trials * velocity_trials
            ),
            "empirical_noise_only_candidate_count": scenarios["no_target"][
                "candidate_count"
            ],
            "real_camera_calibrated": False,
        },
        "scenarios": scenarios,
        "acceptance": {
            "one_target_yields_one_candidate": (
                scenarios["one_target"]["candidate_count"] == 1
                and scenarios["one_target"]["all_targets_recovered"]
            ),
            "two_close_targets_recovered": scenarios["two_close_targets"][
                "all_targets_recovered"
            ],
            "crossing_tracks_recovered": scenarios["crossing_tracks"][
                "all_targets_recovered"
            ],
            "no_target_yields_no_candidates": scenarios["no_target"][
                "candidate_count"
            ]
            == 0,
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
