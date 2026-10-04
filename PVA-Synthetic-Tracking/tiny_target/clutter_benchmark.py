"""Deterministic Phase 12 clutter-normalization characterization."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .detection import (
    CandidateExtractionConfig,
    CandidateExtractor,
    SyntheticTrackWindow,
)
from .telemetry import run_identity, write_json_exclusive


REPOSITORY = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "seaqr.tiny-target.clutter-benchmark.v1"


def _identity() -> dict[str, str]:
    files = (
        Path(__file__),
        Path(__file__).parent / "detection" / "candidates.py",
    )
    return {
        str(path.relative_to(REPOSITORY)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in files
    }


def _window(score: np.ndarray) -> SyntheticTrackWindow:
    shape = score.shape
    return SyntheticTrackWindow(
        score=np.asarray(score, np.float32),
        velocity_index=np.zeros(shape, np.uint16),
        valid_support_count=np.full(shape, 4, np.uint16),
        valid_mask=np.ones(shape, bool),
        velocity_grid_xy_px_s=np.array([[0, 0]], np.float32),
        frame_indices=(0, 1, 2, 3),
        window_start_timestamp_ns=0,
        window_end_timestamp_ns=300_000_000,
        reference_timestamp_ns=150_000_000,
        segment_index=0,
        metrics={},
        timings_ms={},
    )


def _config(**overrides: Any) -> CandidateExtractionConfig:
    values: dict[str, Any] = {
        "score_threshold_snr": 5.0,
        "minimum_support_frames": 4,
        "local_maximum_radius_px": 1,
        "spatial_nms_radius_px": 0.0,
        "velocity_nms_radius_px_s": 0.0,
        "pre_nms_candidate_limit": 32,
        "max_candidates_per_window": 16,
    }
    values.update(overrides)
    return CandidateExtractionConfig(**values)


def _contains(batch: Any, position_xy: tuple[int, int]) -> bool:
    return any(
        (candidate.x_px, candidate.y_px) == position_xy
        for candidate in batch.candidates
    )


def _heterogeneous_clutter(seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    shape = (192, 256)
    tile_height = tile_width = 64
    score = np.empty(shape, np.float32)
    centers = ((45, 25, 2, 1), (30, 15, 0, -1), (18, 8, 1, 0))
    for tile_y, row in enumerate(centers):
        for tile_x, center in enumerate(row):
            y0, y1 = tile_y * tile_height, (tile_y + 1) * tile_height
            x0, x1 = tile_x * tile_width, (tile_x + 1) * tile_width
            scale = 8.0 if center >= 15 else 1.0
            score[y0:y1, x0:x1] = rng.normal(
                center,
                scale,
                (tile_height, tile_width),
            )
    target_xy = (220, 160)
    score[target_xy[1], target_xy[0]] = 10.0
    surface = _window(score)
    raw = CandidateExtractor(_config()).extract(surface)
    cfar_config = _config(
        ranking_mode="tile_robust_cfar",
        cfar_threshold_sigma=6.0,
        cfar_tile_height_px=tile_height,
        cfar_tile_width_px=tile_width,
        cfar_minimum_samples=2048,
        cfar_scale_floor_snr=0.5,
    )
    cfar_extractor = CandidateExtractor(cfar_config)
    ranking = cfar_extractor.ranking_surface(surface)
    cfar = cfar_extractor.extract(surface, ranking_surface=ranking)
    target_selection_score = float(ranking.score[target_xy[1], target_xy[0]])
    valid_selection = ranking.score[np.isfinite(ranking.score)]
    return {
        "target_position_xy_px": list(target_xy),
        "target_raw_score_snr": float(score[target_xy[1], target_xy[0]]),
        "target_raw_surface_rank_lower_bound": (
            1 + int(np.count_nonzero(score > score[target_xy[1], target_xy[0]]))
        ),
        "target_cfar_score_sigma": target_selection_score,
        "target_cfar_surface_rank_lower_bound": (
            1 + int(np.count_nonzero(valid_selection > target_selection_score))
        ),
        "raw_target_recovered": _contains(raw, target_xy),
        "cfar_target_recovered": _contains(cfar, target_xy),
        "raw_candidate_batch": raw.to_dict(),
        "cfar_candidate_batch": cfar.to_dict(),
        "cfar_configuration": asdict(cfar_config),
    }


def _spatial_monopoly(seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    shape = (120, 160)
    score = rng.normal(0, 0.25, shape).astype(np.float32)
    positions = []
    for cell_y in range(2):
        for cell_x in range(2):
            for index in range(8):
                x = cell_x * 80 + 4 + 8 * index
                y = cell_y * 60 + 8 + 5 * (index % 4)
                amplitude = 100.0 - index if (cell_y, cell_x) == (0, 0) else 20.0 - index
                score[y, x] = amplitude
                positions.append((x, y))
    surface = _window(score)
    unbalanced = CandidateExtractor(
        _config(
            pre_nms_candidate_limit=64,
            max_candidates_per_window=8,
        )
    ).extract(surface)
    balanced_config = _config(
        pre_nms_candidate_limit=64,
        max_candidates_per_window=8,
        quota_grid_rows=2,
        quota_grid_cols=2,
        pre_nms_candidates_per_cell=8,
        max_candidates_per_cell=2,
    )
    balanced = CandidateExtractor(balanced_config).extract(surface)
    return {
        "input_peak_count": len(positions),
        "unbalanced_occupied_output_cells": len(
            {
                (candidate.y_px // 60, candidate.x_px // 80)
                for candidate in unbalanced.candidates
            }
        ),
        "balanced_occupied_output_cells": len(
            {
                candidate.quota_cell_row_col for candidate in balanced.candidates
            }
        ),
        "unbalanced_candidate_batch": unbalanced.to_dict(),
        "balanced_candidate_batch": balanced.to_dict(),
        "balanced_configuration": asdict(balanced_config),
    }


def run_benchmark(seed: int = 75) -> dict[str, Any]:
    clutter = _heterogeneous_clutter(seed)
    quota = _spatial_monopoly(seed + 1)
    return {
        "schema_version": SCHEMA_VERSION,
        "run": run_identity(REPOSITORY),
        "implementation_sha256": _identity(),
        "random_seed": seed,
        "scenarios": {
            "heterogeneous_clutter": clutter,
            "spatial_monopoly": quota,
        },
        "acceptance": {
            "raw_global_ranking_misses_target": not clutter[
                "raw_target_recovered"
            ],
            "tile_robust_cfar_recovers_target": clutter[
                "cfar_target_recovered"
            ],
            "spatial_quota_fills_all_cells": (
                quota["unbalanced_occupied_output_cells"] == 1
                and quota["balanced_occupied_output_cells"] == 4
            ),
        },
        "scope": (
            "Controlled candidate-selection characterization only; RAW16 behavior "
            "is measured separately and unmatched real candidates are not labeled "
            "false alarms."
        ),
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
