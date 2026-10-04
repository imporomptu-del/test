#!/usr/bin/env python3
"""Fixed six-pair independent patch diagnostic, gated by generated controls.

The matcher sees native pixels and previous feature locations only. Saved PVA
endpoints and the original fit enter comparisons after matching, never search.
This does not rerun the detector, refit global motion, or establish scene truth.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np

import review_feature_residual_pixels as pixels
import validate_motion_patch_controls as matcher

ANALYSIS_SHA = "ce1370499b8a2ca34352bd1e4cc9bb4af31f5464959c8e24b630422a0030b1b1"
PIXEL_HELPER_SHA = "b9d90c82ace5c6c1f0f4b12ca9d08cd194ba23c30417f2db61b17907081fe62f"
SCHEMA = "seaqr.independent-motion-pairs.v1"
LIMITATIONS = matcher.LIMITATIONS + [
    "Only previously LK-accepted feature locations in six fixed development pairs are measured.",
    "Neither independent local translation nor PVA disagreement establishes true camera motion.",
    "Native 31x31 patch support differs from half-resolution PVA pyramid support.",
    "Unavailable patch measurements are retained and excluded from displacement comparisons.",
    "No raw rejected PVA points, additional clips, detector runs or model refits are included.",
]


def validate_controls(path, expected_sha):
    """Reject a failed, altered, incomplete, or differently configured corpus."""
    path = Path(path).absolute()
    pixels.require(path.is_file() and not path.is_symlink()
                   and pixels.sha(path) == pixels.digest_value(expected_sha), "Control result identity differs")
    result = pixels.read_json(path)
    pixels.require(result.get("schema") == matcher.SCHEMA + ".result"
                   and result.get("passed") is True and result.get("generated_only") is True
                   and result.get("source_media_accessed") is False
                   and result.get("production_changes") is False and result.get("model_fits") == 0,
                   "Generated controls have not passed with isolated scope")
    pixels.require(result.get("inventory") == matcher.control_inventory(), "Generated control inventory differs")
    bindings = dict(result["input_sha256"])
    expected_code = (Path(matcher.__file__).resolve(),
                     pixels.REPOSITORY / "tests/unit/test_motion_patch_controls.py")
    pixels.require(set(bindings) == {str(p) for p in expected_code}, "Control code bindings differ")
    pixels.check_hashes(bindings)
    cases = result["inventory"]["cases"]
    rows = result.get("rows", [])
    pixels.require(len(rows) == len(cases) == 20, "Generated control coverage differs")
    for row, case in zip(rows, cases):
        pixels.require(row["case"] == case and len(row["queries"]) == 1, "Generated case/query differs")
        query = row["queries"][0]
        pixels.require(query["previous_xy"] == case["point_queries_xy"][0]
                       and query["assessment"] == matcher.assess_control(case, query["measurement"])
                       and query["assessment"]["passed"] is True, "Generated control assessment differs")
    bindings[str(path)] = expected_sha
    return bindings


def distribution(values):
    a = np.asarray(values, dtype=np.float64)
    pixels.require(a.ndim == 1 and np.isfinite(a).all(), "Invalid summary values")
    return dict(count=len(a), minimum=None if not len(a) else float(a.min()),
                median=None if not len(a) else float(np.median(a)),
                p90=None if not len(a) else float(np.percentile(a, 90)),
                maximum=None if not len(a) else float(a.max()))


def summarize_points(points):
    qualified = [p for p in points if p["independent"]["qualified"]]
    unavailable = [p for p in points if not p["independent"]["qualified"]]
    return dict(total=len(points), qualified=len(qualified), unavailable=len(unavailable),
        unavailable_reasons=dict(Counter(r for p in unavailable for r in p["independent"]["abstention_reasons"])),
        qualified_lk_disagreement_px=distribution([p["comparison"]["lk_difference_norm_px"] for p in qualified]),
        qualified_original_fit_disagreement_px=distribution([p["comparison"]["original_fit_difference_norm_px"] for p in qualified]),
        qualified_original_lk_residual_px=distribution([p["original_residual_px"] for p in qualified]),
        diagnostic_measurement_seconds=sum(p["independent"]["cost"]["elapsed_seconds"] for p in points))


def measure_pair(before, after, arrays, original_matrix):
    p, q, scores, fb, inliers, residuals = arrays
    pixels.require(0 < len(p) <= 384 and p.shape == q.shape == (len(p), 2), "Point bounds differ")
    pixels.require(all(a.shape == (len(p),) for a in (scores, fb, inliers, residuals)), "Point arrays differ")
    matrix = np.asarray(original_matrix, np.float64)
    pixels.require(matrix.shape == (3, 3) and np.isfinite(matrix).all(), "Original saved matrix required")
    projected = np.column_stack((p, np.ones(len(p)))) @ matrix.T
    pixels.require(np.all(projected[:, 2] != 0), "Invalid saved transform")
    original_prediction = projected[:, :2] / projected[:, 2:3]
    pixels.require(np.allclose(np.linalg.norm(q-original_prediction, axis=1), residuals, rtol=0, atol=1e-9),
                   "Original residuals differ")
    threshold = float(np.quantile(scores, .75))
    points = []
    for i in range(len(p)):
        # Independence boundary: q, scores, FB, residuals, inliers and the saved
        # matrix are deliberately not passed to the matcher.
        independent = matcher.measure_patch(before, after, p[i])
        comparison = None
        if independent["qualified"]:
            motion = np.asarray(independent["displacement_xy"], np.float64)
            lk_difference = motion - (q[i].astype(np.float64)-p[i])
            fit_difference = motion - (original_prediction[i]-p[i])
            comparison = dict(lk_difference_xy=lk_difference.tolist(),
                lk_difference_norm_px=float(np.linalg.norm(lk_difference)),
                original_fit_difference_xy=fit_difference.tolist(),
                original_fit_difference_norm_px=float(np.linalg.norm(fit_difference)),
                independent_measurement_is_ground_truth=False)
        cell_x = min(7, max(0, int(p[i, 0] / before.shape[1] * 8)))
        cell_y = min(5, max(0, int(p[i, 1] / before.shape[0] * 6)))
        points.append(dict(point_index=i, previous_xy=p[i].tolist(), original_current_xy=q[i].tolist(),
            harris_score=float(scores[i]), top_score_quartile=bool(scores[i] >= threshold),
            grid_cell=cell_y*8+cell_x, original_forward_backward_error_px=float(fb[i]),
            original_ransac_inlier=bool(inliers[i]), original_residual_px=float(residuals[i]),
            original_fit_displacement_xy=(original_prediction[i]-p[i]).tolist(),
            independent=independent, comparison=comparison))
    groups = {"all": points, "original_inliers": [p for p in points if p["original_ransac_inlier"]],
              "original_outliers": [p for p in points if not p["original_ransac_inlier"]],
              "top_score_quartile": [p for p in points if p["top_score_quartile"]]}
    return dict(points=points, summary={k: summarize_points(v) for k, v in groups.items()},
                grid_summaries={str(cell): summarize_points([p for p in points if p["grid_cell"] == cell])
                                for cell in sorted({p["grid_cell"] for p in points})})


def run(controls, controls_sha256, analysis, output):
    started = time.perf_counter()
    bindings = validate_controls(controls, controls_sha256)
    pixels.require(pixels.sha(Path(pixels.__file__).resolve()) == PIXEL_HELPER_SHA, "Pinned source helper changed")
    _, analyzer, rows, trace_bindings = pixels.load_verified_analysis(analysis, ANALYSIS_SHA)
    bindings.update(trace_bindings)
    for code in (Path(__file__).resolve(), pixels.REPOSITORY / "tests/unit/test_independent_motion_pairs.py"):
        bindings[str(code)] = pixels.sha(code)
    # Sources are first accessed only after all generated-control and trace checks.
    sources = {clip: pixels.SOURCE_DIRECTORY / ("chunk_" + clip + ".avi") for clip in pixels.PAIR_ENDS}
    for clip, source in sources.items():
        pixels.require(source.resolve() == source and source.is_file() and not source.is_symlink(), "Source is not approved native file")
        pixels.require(pixels.sha(source) == pixels.SOURCE_HASHES[clip], "Source hash changed")
        bindings[str(source)] = pixels.SOURCE_HASHES[clip]
    output = Path(output).absolute()
    pixels.require(output.parent.is_dir() and output.parent.resolve() == output.parent
                   and not output.exists() and not output.is_symlink(), "Fresh output directory required")
    output.mkdir()
    try:
        result = dict(schema=SCHEMA, passed_integrity=False, control_result_sha256=controls_sha256,
            analysis_sha256=ANALYSIS_SHA, contract=matcher.CONTRACT, pair_ends=pixels.PAIR_ENDS,
            input_sha256=bindings, limitations=LIMITATIONS, model_fits=0, detector_runs=0, production_changes=False, clips={})
        for clip, source in sources.items():
            measured = []
            for end, before, after in pixels.decode_verified_pairs(source, rows[clip]):
                row = rows[clip][end]
                arrays = analyzer.unpack_row(row, clip)
                matrix = analyzer.decode_array(row["fit"]["previous_to_current_matrix"], np.dtype("float64"), (3, 3))
                item = measure_pair(before, after, arrays, matrix)
                item.update(previous_frame_index=end-1, current_frame_index=end,
                            native_gray_pixel_sha256=row["native_gray_pixel_sha256"])
                measured.append(item)
                print(json.dumps(dict(clip=clip, pair_end=end, summary=item["summary"]["all"])), flush=True)
            pixels.require([r["current_frame_index"] for r in measured] == pixels.PAIR_ENDS[clip], "Pair inventory differs")
            result["clips"][clip] = dict(rows=measured, causally_decoded_from=0,
                causally_decoded_through=max(pixels.PAIR_ENDS[clip]),
                summary=summarize_points([p for row in measured for p in row["points"]]))
        pixels.check_hashes(bindings)
        result.update(passed_integrity=True, elapsed_seconds=time.perf_counter()-started,
                      completed_utc=datetime.now(timezone.utc).isoformat(), numpy_version=np.__version__)
        pixels.write_json(output / "measurements.json", result)
        pixels.write_json(output / "receipt.json", dict(schema=SCHEMA+".receipt", passed_integrity=True,
            measurements_sha256=pixels.sha(output / "measurements.json"), input_sha256=bindings,
            source_pixels_mutated=False, model_fits=0, detector_runs=0, production_changes=False))
        return result
    except BaseException as exc:
        pixels.write_json(output / "failed_receipt.json", dict(passed_integrity=False, error=repr(exc),
            partial_results_not_validated=True, input_sha256=bindings))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controls", type=Path, required=True)
    parser.add_argument("--controls-sha256", required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.controls, args.controls_sha256, args.analysis, args.output)
