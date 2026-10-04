"""Accelerator-independent feature selection and coordinate math."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _validate_size(size: tuple[int, int], label: str) -> tuple[int, int]:
    width, height = size
    if width <= 0 or height <= 0:
        raise ValueError(f"{label} must contain positive width and height")
    return int(width), int(height)


def lift_points_to_full_resolution(
    points: np.ndarray,
    motion_size: tuple[int, int],
    full_size: tuple[int, int],
) -> np.ndarray:
    """Map motion-image pixel centers onto full-resolution pixel centers."""

    motion_width, motion_height = _validate_size(motion_size, "motion_size")
    full_width, full_height = _validate_size(full_size, "full_size")
    result = np.ascontiguousarray(points, dtype=np.float32).copy()
    if result.ndim != 2 or result.shape[1] != 2:
        raise ValueError("points must have shape (N, 2)")
    result[:, 0] = (
        (result[:, 0] + 0.5) * (full_width / motion_width) - 0.5
    )
    result[:, 1] = (
        (result[:, 1] + 0.5) * (full_height / motion_height) - 0.5
    )
    return result


def lower_points_to_motion_resolution(
    points: np.ndarray,
    full_size: tuple[int, int],
    motion_size: tuple[int, int],
) -> np.ndarray:
    """Inverse of :func:`lift_points_to_full_resolution`."""

    full_width, full_height = _validate_size(full_size, "full_size")
    motion_width, motion_height = _validate_size(motion_size, "motion_size")
    result = np.ascontiguousarray(points, dtype=np.float32).copy()
    if result.ndim != 2 or result.shape[1] != 2:
        raise ValueError("points must have shape (N, 2)")
    result[:, 0] = (
        (result[:, 0] + 0.5) * (motion_width / full_width) - 0.5
    )
    result[:, 1] = (
        (result[:, 1] + 0.5) * (motion_height / full_height) - 0.5
    )
    return result


def select_spatially_distributed(
    points: np.ndarray,
    scores: np.ndarray,
    image_size: tuple[int, int],
    *,
    grid_rows: int,
    grid_cols: int,
    max_features: int,
    max_per_cell: int | None = None,
    eligible_mask: np.ndarray | None = None,
    execution: str = "reference",
) -> np.ndarray:
    """Return score-ranked indices subject to a strict per-cell quota."""

    if execution not in {"reference", "batched_exact_v1"}:
        raise ValueError("Unknown spatial-selection execution policy")
    width, height = _validate_size(image_size, "image_size")
    point_array = np.asarray(points, dtype=np.float32)
    score_array = np.asarray(scores).reshape(-1)
    if point_array.ndim != 2 or point_array.shape[1] != 2:
        raise ValueError("points must have shape (N, 2)")
    if len(point_array) != len(score_array):
        raise ValueError("scores must have one value per point")
    if grid_rows <= 0 or grid_cols <= 0 or max_features <= 0:
        raise ValueError("grid dimensions and max_features must be positive")
    quota = max_per_cell
    if quota is None:
        quota = math.ceil(max_features / (grid_rows * grid_cols))
    if quota <= 0:
        raise ValueError("max_per_cell must be positive")

    finite = np.isfinite(point_array).all(axis=1) & np.isfinite(score_array)
    in_bounds = (
        (point_array[:, 0] >= 0)
        & (point_array[:, 0] < width)
        & (point_array[:, 1] >= 0)
        & (point_array[:, 1] < height)
    )
    eligible = np.ones(len(point_array), dtype=bool)
    if eligible_mask is not None:
        eligible = np.asarray(eligible_mask, dtype=bool).reshape(-1)
        if len(eligible) != len(point_array):
            raise ValueError("eligible_mask must have one value per point")
    candidates = np.flatnonzero(finite & in_bounds & eligible)
    if len(candidates) == 0:
        return np.empty(0, dtype=np.int64)
    ranked = candidates[
        np.argsort(-score_array[candidates].astype(np.float64), kind="stable")
    ]
    if (execution == "batched_exact_v1"
            and _batchable_grid(width, height, grid_rows, grid_cols)
            and type(quota) is int and type(max_features) is int):
        cells = _batched_cell_ids(point_array[ranked], width, height, grid_rows, grid_cols)
        # Stable grouping retains score order (and original order for ties)
        # within each cell. Restore global rank before applying the clip limit.
        grouped = np.argsort(cells, kind="stable")
        grouped_cells = cells[grouped]
        positions = np.arange(len(grouped), dtype=np.int64)
        starts = np.maximum.accumulate(np.where(
            np.r_[True, grouped_cells[1:] != grouped_cells[:-1]], positions, 0))
        admitted_ranks = np.sort(grouped[positions - starts < quota])
        return np.asarray(ranked[admitted_ranks[:max_features]], dtype=np.int64)
    cell_counts = np.zeros((grid_rows, grid_cols), dtype=np.int32)
    selected: list[int] = []
    for index in ranked:
        x, y = point_array[index]
        col = min(grid_cols - 1, int(x * grid_cols / width))
        row = min(grid_rows - 1, int(y * grid_rows / height))
        if cell_counts[row, col] >= quota:
            continue
        selected.append(int(index))
        cell_counts[row, col] += 1
        if len(selected) >= max_features:
            break
    return np.asarray(selected, dtype=np.int64)


def _batchable_grid(width: int, height: int, rows: int, cols: int) -> bool:
    # Keep unusual legacy arguments on their original path. In this range
    # image bounds are exact float32 integers and flattened IDs fit int64.
    return (all(type(v) is int for v in (width, height, rows, cols))
            and 0 < width <= 2**24 and 0 < height <= 2**24
            and 0 < rows <= 2**20 and 0 < cols <= 2**20)


def _batched_cell_ids(points, width, height, rows, cols):
    def coordinates(values, count, extent):
        # Scalar float32 * Python-int promotion differs between NumPy 1.x
        # (Jetson) and 2.x (local). Preserve BOTH operations' scalar dtypes;
        # blindly vectorizing in float32 can change boundary cell assignment.
        scalar_product = np.float32(0) * count
        product = np.multiply(values, count, dtype=np.asarray(scalar_product).dtype)
        scaled = np.divide(product, extent,
                           dtype=np.asarray(scalar_product / extent).dtype)
        return np.minimum(count - 1, scaled.astype(np.int64))

    x = coordinates(points[:, 0], cols, width)
    y = coordinates(points[:, 1], rows, height)
    return y * cols + x


def grid_coverage(
    points: np.ndarray,
    image_size: tuple[int, int],
    *,
    grid_rows: int,
    grid_cols: int,
    execution: str = "reference",
) -> dict[str, Any]:
    if execution not in {"reference", "batched_exact_v1"}:
        raise ValueError("Unknown grid-coverage execution policy")
    width, height = _validate_size(image_size, "image_size")
    if grid_rows <= 0 or grid_cols <= 0:
        raise ValueError("grid dimensions must be positive")
    point_array = np.asarray(points, dtype=np.float32)
    if (execution == "batched_exact_v1" and point_array.ndim == 2
            and point_array.shape[1] == 2
            and _batchable_grid(width, height, grid_rows, grid_cols)):
        valid = (np.isfinite(point_array).all(axis=1)
                 & (point_array[:, 0] >= 0) & (point_array[:, 0] < width)
                 & (point_array[:, 1] >= 0) & (point_array[:, 1] < height))
        occupied_count = len(np.unique(_batched_cell_ids(
            point_array[valid], width, height, grid_rows, grid_cols)))
        total = grid_rows * grid_cols
        return {"occupied_cells": occupied_count, "total_cells": total,
                "fraction": occupied_count / total}
    occupied: set[tuple[int, int]] = set()
    for x, y in point_array:
        if not np.isfinite((x, y)).all() or x < 0 or y < 0 or x >= width or y >= height:
            continue
        col = min(grid_cols - 1, int(x * grid_cols / width))
        row = min(grid_rows - 1, int(y * grid_rows / height))
        occupied.add((row, col))
    total = grid_rows * grid_cols
    return {
        "occupied_cells": len(occupied),
        "total_cells": total,
        "fraction": len(occupied) / total,
    }


def correspondence_acceptance_mask(
    previous_points: np.ndarray,
    current_points: np.ndarray,
    forward_status: np.ndarray,
    *,
    image_size: tuple[int, int],
    max_displacement_px: float,
    backward_points: np.ndarray | None = None,
    backward_status: np.ndarray | None = None,
    max_forward_backward_error_px: float | None = None,
) -> tuple[np.ndarray, dict[str, int], np.ndarray]:
    """Apply track status, bounds, displacement, and optional FB checks."""

    width, height = _validate_size(image_size, "image_size")
    previous = np.asarray(previous_points, dtype=np.float32)
    current = np.asarray(current_points, dtype=np.float32)
    status = np.asarray(forward_status).reshape(-1)
    if previous.shape != current.shape or previous.ndim != 2 or previous.shape[1] != 2:
        raise ValueError("previous_points and current_points must share shape (N, 2)")
    if len(status) != len(previous):
        raise ValueError("forward_status must have one value per point")
    if max_displacement_px <= 0:
        raise ValueError("max_displacement_px must be positive")

    finite = np.isfinite(previous).all(axis=1) & np.isfinite(current).all(axis=1)
    status_ok = status == 0
    bounds_ok = (
        (current[:, 0] >= 0)
        & (current[:, 0] < width)
        & (current[:, 1] >= 0)
        & (current[:, 1] < height)
    )
    displacement = np.linalg.norm(current - previous, axis=1)
    displacement_ok = displacement <= max_displacement_px
    mask = finite & status_ok & bounds_ok & displacement_ok

    fb_error = np.full(len(previous), np.nan, dtype=np.float32)
    fb_ok = np.ones(len(previous), dtype=bool)
    if backward_points is not None or backward_status is not None:
        if backward_points is None or backward_status is None:
            raise ValueError("backward_points and backward_status must be supplied together")
        backward = np.asarray(backward_points, dtype=np.float32)
        back_status = np.asarray(backward_status).reshape(-1)
        if backward.shape != previous.shape or len(back_status) != len(previous):
            raise ValueError("backward results must match the forward point count")
        fb_error = np.linalg.norm(backward - previous, axis=1).astype(np.float32)
        threshold = max_forward_backward_error_px
        if threshold is None or threshold <= 0:
            raise ValueError("a positive forward/backward threshold is required")
        fb_ok = (
            np.isfinite(backward).all(axis=1)
            & np.isfinite(fb_error)
            & (back_status == 0)
            & (fb_error <= threshold)
        )
        mask &= fb_ok

    rejection_counts = {
        "nonfinite": int(np.count_nonzero(~finite)),
        "forward_status": int(np.count_nonzero(finite & ~status_ok)),
        "out_of_bounds": int(np.count_nonzero(finite & status_ok & ~bounds_ok)),
        "excessive_displacement": int(
            np.count_nonzero(finite & status_ok & bounds_ok & ~displacement_ok)
        ),
        "forward_backward": int(
            np.count_nonzero(finite & status_ok & bounds_ok & displacement_ok & ~fb_ok)
        ),
    }
    return mask, rejection_counts, fb_error
