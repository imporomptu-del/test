"""Model-specific, spatially held-out checks for sparse translation support.

No image-region hints, target labels, or previous transform reuse enter this gate.
Support is evaluated on ALL accepted flow pairs, not just RANSAC's inliers.
Agreement does not prove physical background identity outside observed support.
"""
from __future__ import annotations

import numpy as np


def sparse_translation_support(previous, current, image_size, translation, config):
    previous = np.asarray(previous, dtype=np.float64)
    flow = np.asarray(current, dtype=np.float64) - previous
    size = np.asarray(image_size, dtype=np.float64)
    result = dict(passed=False, rejection_reasons=[])
    if not np.isfinite(previous).all() or not np.isfinite(flow).all():
        result["rejection_reasons"].append("nonfinite_support")
        return result
    indices = np.floor(previous / size * [config.grid_cols, config.grid_rows]).astype(
        int
    )
    if np.any(indices < 0) or np.any(indices >= [config.grid_cols, config.grid_rows]):
        result["rejection_reasons"].append("out_of_bounds_support")
        return result
    keys = indices[:, 1] * config.grid_cols + indices[:, 0]
    groups = [np.flatnonzero(keys == key) for key in np.unique(keys)]
    groups = [g for g in groups if len(g) >= config.sparse_minimum_points_per_cell]
    result["supported_cells"] = len(groups)
    result["supported_points"] = sum(map(len, groups))
    result["supported_point_fraction"] = result["supported_points"] / max(
        1, len(previous)
    )
    if len(groups) < config.sparse_minimum_cells:
        result["rejection_reasons"].append("too_few_supported_cells")
        return result
    centers = np.asarray([np.median(previous[g], axis=0) for g in groups])
    span = np.ptp(centers, axis=0) / size
    result["cell_center_span_fraction_xy"] = span.tolist()
    if result["supported_point_fraction"] < config.minimum_inlier_ratio:
        result["rejection_reasons"].append("insufficient_supported_point_fraction")
    cell_translations = np.asarray([np.median(flow[g], axis=0) for g in groups])
    # Equal weight per cell prevents a dense moving cluster dominating the check.
    balanced = np.median(cell_translations, axis=0)
    result["balanced_translation_xy_px"] = balanced.tolist()
    result["fit_to_balanced_error_px"] = float(np.linalg.norm(balanced - translation))
    if result["fit_to_balanced_error_px"] > config.maximum_median_reprojection_px:
        result["rejection_reasons"].append("point_weighted_fit_disagrees_with_cells")
    cell_results = []
    passing = []
    for i, group in enumerate(groups):
        # The held-out cell contributes nothing to its predicting translation.
        predicted = np.median(np.delete(cell_translations, i, axis=0), axis=0)
        errors = np.linalg.norm(flow[group] - predicted, axis=1)
        median = float(np.median(errors))
        # Apply the SAME robust inlier radius/ratio as the global fit, but to
        # predictions from other cells. Isolated bad flows cannot veto a region;
        # an entire moving region still disagrees and cannot select its own fit.
        inliers = errors <= config.ransac_reprojection_px
        ratio = float(np.mean(inliers))
        inlier_count = int(np.count_nonzero(inliers))
        p90 = float(np.percentile(errors[inliers], 90)) if inlier_count else None
        agrees = (
            median <= config.maximum_median_reprojection_px
            and ratio >= config.minimum_inlier_ratio
            and inlier_count >= config.sparse_minimum_points_per_cell
            and p90 is not None
            and p90 <= config.maximum_p90_reprojection_px
        )
        cell_results.append(
            dict(
                cell_id=int(keys[group[0]]),
                points=len(group),
                held_out_median_error_px=median,
                held_out_raw_p90_error_px=float(np.percentile(errors, 90)),
                held_out_inlier_p90_error_px=p90,
                held_out_inlier_count=inlier_count,
                held_out_inlier_ratio=ratio,
                agrees_with_other_cells=agrees,
            )
        )
        if agrees:
            passing.append(i)
    # Local foreground motion is expected in a target detector. A bad region
    # loses its vote, not every other region's evidence. Require a robust spatial
    # majority AND enough supporting points, using the existing inlier ratio.
    result["consensus_cells"] = len(passing)
    result["consensus_cell_fraction"] = len(passing) / len(groups)
    result["consensus_inlier_points"] = sum(
        cell_results[i]["held_out_inlier_count"] for i in passing
    )
    result["consensus_point_fraction"] = result["consensus_inlier_points"] / len(
        previous
    )
    result["excluded_cell_ids"] = [
        c["cell_id"] for c in cell_results if not c["agrees_with_other_cells"]
    ]
    if (
        len(passing) < config.sparse_minimum_cells
        or result["consensus_cell_fraction"] < config.minimum_inlier_ratio
        or result["consensus_point_fraction"] < config.minimum_inlier_ratio
    ):
        result["rejection_reasons"].append("insufficient_held_out_consensus")
    if passing:
        consensus_span = np.ptp(centers[passing], axis=0) / size
        result["consensus_span_fraction_xy"] = consensus_span.tolist()
        if float(np.max(consensus_span)) < config.sparse_minimum_span_fraction:
            result["rejection_reasons"].append("compact_spatial_support")
    result["cells"] = cell_results
    result["passed"] = not result["rejection_reasons"]
    return result
