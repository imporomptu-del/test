"""Metadata-only analysis of passively captured, original translation fits.

The input population has already passed LK correspondence filtering. This does
not recover rejected raw tracks, refit motion, infer ground truth, or tune gates.
"""
from __future__ import annotations

import argparse
import base64
import copy
from collections import Counter
import hashlib
from itertools import zip_longest
import json
import math
from pathlib import Path

import numpy as np

IMAGE_SIZE_WH = (4784, 3190)
GRID_ROWS, GRID_COLS = 6, 8
MAX_POINTS = 384
RESIDUAL_THRESHOLDS = (.35, .75, 1.0)
ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT.parent / "outputs/seaqr_feature_residual_trace_20260930"
ORIGINAL_ROOT = ROOT.parent / "outputs/seaqr_feature_selection_20260929/evidence"
CLIPS = ("0170", "0240")
IGNORED_PATHS = [["timings_ms"], ["motion", "pva_timings_ms"], ["motion", "motion_fit", "timing_ms"],
                 ["motion", "warp_timings_ms"], ["coverage", "detection_ms"]]
ARTIFACT_FILES = dict(execution_receipt="execution_receipt.json", preflight="preflight.json", journal="run/frames.jsonl",
                      report="run/report.json", launch="run/launch.json")
SOURCE_SHA = {"0170":"12848c0f0caedd697a3da51776ab1579bd634a7ae94343f8cbd2a8830ee340bc",
              "0240":"2f86f28785e302572a86e23688143edbd7f5f1f65e8a3434b86a427e79c6a585"}
CANDIDATE = dict(harris_gain=16,harris_capacity_policy="complete_grid",feature_image_scale=.5,
                 feature_cpu_policy="batched_exact_v1",max_features=384,max_features_per_cell=8,grid_rows=6,grid_cols=8)
ORIGINAL_FREEZE_SHA = "9ae62028ca25fe063241e62697cb3578a708078a1b03969061b4736f0c9c6b7e"
ORIGINAL_RUNNER_SHA = "db5a92d69fb7fc503a3ef2d56236460d1bb32656111ad3fd170203b9ab53bc4d"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def regular(path):
    path = Path(path)
    require(path.is_absolute() and path.resolve() == path and path.is_file() and not path.is_symlink()
            and path.suffix in (".json", ".jsonl", ".py"), "regular absolute metadata/code path required")
    return path


def sha(path):
    digest = hashlib.sha256()
    with regular(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def decode_json(text):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    def finite(value):
        result = float(value)
        require(math.isfinite(result), "nonfinite JSON number")
        return result
    return json.loads(text, object_pairs_hook=unique, parse_float=finite,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def read(path):
    return decode_json(regular(path).read_text())


def bind(path, hashes, expected=None):
    digest = sha(path)
    require(expected is None or digest == expected, "artifact hash differs: " + str(path))
    hashes[str(path)] = digest
    return digest


def decode_array(record, dtype, shape):
    require(isinstance(record, dict) and record.get("dtype") == np.dtype(dtype).str
            and record.get("shape") == list(shape) and all(type(n) is int for n in record["shape"])
            and all(type(n) is int and 0 <= n <= MAX_POINTS for n in shape),
            "array dtype/shape differs")
    require(np.dtype(dtype).str in ("<f4", "<f8", "|b1") and math.prod(shape) <= MAX_POINTS * 2,
            "unapproved array format/size")
    encoded = record.get("data_base64")
    expected_bytes = math.prod(shape) * np.dtype(dtype).itemsize
    require(isinstance(encoded, str) and len(encoded) <= 4*((expected_bytes+2)//3), "invalid encoded array size")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except Exception as exc:
        raise ValueError("invalid array base64") from exc
    require(len(raw) == expected_bytes and hashlib.sha256(raw).hexdigest() == record.get("sha256"), "array bytes/hash differ")
    if np.dtype(dtype) == np.dtype(bool):
        require(all(value in (0, 1) for value in raw), "noncanonical boolean mask bytes")
    return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()


def unpack_row(row, clip):
    require(clip in CLIPS and type(row["current_frame_index"]) is int, "invalid capture source/frame")
    index = row["current_frame_index"]
    require(1 <= index <= 672 and row["previous_frame_index"] == index-1
            and row["previous_timestamp_ns"] == (index-1)*100_000_000 and row["current_timestamp_ns"] == index*100_000_000
            and row["full_image_size"] == [4784,3190] and row["motion_image_size"] == [2392,1595], "capture pair geometry/timeline differs")
    for digest in row["native_gray_pixel_sha256"].values():
        require(isinstance(digest,str) and len(digest)==64 and all(c in "0123456789abcdef" for c in digest), "missing native gray identity")
    require(set(row["native_gray_pixel_sha256"]) == {"previous","current"}, "native gray pair identities differ")
    corr, fit = row["correspondence"], row["fit"]
    shape = corr["previous_points"].get("shape")
    require(isinstance(shape,list) and len(shape)==2 and type(shape[0]) is int and 0 <= shape[0] <= MAX_POINTS and shape[1]==2,
            "invalid accepted-point inventory")
    n = shape[0]
    arrays = (decode_array(corr["previous_points"],"<f4",(n,2)), decode_array(corr["current_points"],"<f4",(n,2)),
              decode_array(corr["harris_scores"],"<f4",(n,)), decode_array(corr["forward_backward_error_px"],"<f4",(n,)),
              decode_array(fit["inlier_mask"],"|b1",(n,)), decode_array(fit["residuals_px"],"<f8",(n,)))
    require(fit["model"] == "translation" and corr["metrics"]["accepted_count"] == n
            and fit["metrics"]["correspondence_count"] == n, "capture model/population differs")
    if fit["parameters"] is not None:
        matrix = decode_array(fit["previous_to_current_matrix"],"<f8",(3,3))
        tx,ty = fit["parameters"]["translation_x_px"],fit["parameters"]["translation_y_px"]
        require(np.array_equal(matrix,[[1,0,tx],[0,1,ty],[0,0,1]]), "saved matrix/parameters differ")
    else:
        require(fit["previous_to_current_matrix"] is None and fit["quality_status"] == "rejected", "undefined accepted fit")
    return arrays


def strip_timing(row):
    row = copy.deepcopy(row)
    for path in IGNORED_PATHS:
        parent = row
        for key in path[:-1]:
            parent = parent.get(key) if isinstance(parent,dict) else None
        if isinstance(parent,dict):
            parent.pop(path[-1],None)
    return row


def verify_full_parity(original, diagnostic, capture_pairs):
    captured, compared = {}, 0
    with regular(original).open() as left, regular(diagnostic).open() as right:
        for index,(old,new) in enumerate(zip_longest(left,right)):
            require(index < 673 and old is not None and new is not None, "full causal journal inventory differs")
            old,new = decode_json(old),decode_json(new)
            require(type(old["frame_index"]) is type(new["frame_index"]) is int
                    and old["frame_index"] == new["frame_index"] == index
                    and old["timestamp_ns"] == new["timestamp_ns"] == index*100_000_000, "full journal chronology differs")
            require(json.dumps(strip_timing(old),sort_keys=True,separators=(",",":"),allow_nan=False)
                    == json.dumps(strip_timing(new),sort_keys=True,separators=(",",":"),allow_nan=False),
                    "non-timing journal differs at frame " + str(index))
            if index in capture_pairs:
                captured[index] = new["motion"]
            compared += 1
    require(compared == 673 and sorted(captured) == list(capture_pairs), "incomplete full-causal parity")
    return captured


def verify_inputs(evidence_root, bundle_root, freeze_sha256):
    evidence_root,bundle_root = Path(evidence_root).absolute(),Path(bundle_root).absolute()
    require(evidence_root == OUTPUT_ROOT/"evidence" and bundle_root == OUTPUT_ROOT/"bundle", "outside fixed trace roots")
    hashes = {}
    bind(bundle_root/"freeze.json",hashes,freeze_sha256)
    freeze = read(bundle_root/"freeze.json")
    require(freeze["schema"] == "feature_residual_trace.v1" and set(freeze["sources"]) == set(CLIPS)
            and freeze["candidate"] == CANDIDATE and freeze["original_freeze_sha256"] == ORIGINAL_FREEZE_SHA,
            "trace freeze scope differs")
    files = freeze["files"]
    require({"run_feature_residual_trace.py","batch_feature_residual_trace.py","batch_discovery_pair.py",
             "feature_residual_trace_plan.json"} <= set(files), "incomplete trace bundle")
    for name,digest in files.items():
        require(Path(name).name == name and name not in (".","..","freeze.json"), "invalid bundle member")
        bind(bundle_root/name,hashes,digest)
    bind(evidence_root/"freeze.json",hashes,freeze_sha256)
    plan_path=bundle_root/"feature_residual_trace_plan.json"
    plan=read(plan_path)
    require(plan["schema"] == "seaqr.feature-residual-trace.plan.v1" and plan["parity"]["ignored_exact_paths"] == IGNORED_PATHS
            and plan["capture_pair_counts"] == {"0170":56,"0240":44} and plan["sources"] == freeze["sources"]
            and plan["candidate"] == CANDIDATE and plan["original_freeze_sha256"] == ORIGINAL_FREEZE_SHA
            and plan["original_workspace"] == freeze["original_workspace"] == "/tmp/seaqr_feature_selection_20260929_q4iI5B",
            "frozen trace plan differs")
    for clip in CLIPS:
        source=plan["sources"][clip]
        require(source["sha256"] == SOURCE_SHA[clip] and source["frames"] == 673
                and (source["width"],source["height"],source["fps"]) == (4784,3190,10), "trace source differs")
        pairs=plan["capture_pairs"][clip]
        require(len(pairs) == plan["capture_pair_counts"][clip] and pairs == sorted(set(pairs))
                and all(type(i) is int and 1 <= i <= 672 for i in pairs), "capture inventory differs")
        require(sorted(set(i for values in plan["capture_groups"][clip].values() for i in values)) == pairs,
                "capture group union differs")
    clips={}
    for clip in CLIPS:
        root=evidence_root/clip
        paths={k:root/f for k,f in ARTIFACT_FILES.items()}
        paths.update(trace=root/"trace.json",parity=root/"parity.json")
        for path in paths.values():bind(path,hashes)
        receipt,pre,trace,parity,report,launch = (read(paths[k]) for k in ("execution_receipt","preflight","trace","parity","report","launch"))
        require(receipt["schema"] == "seaqr.discovery-feature-residual-trace.v1" and receipt["passed"] is True
                and receipt["processed_frames"] == 673 and receipt["error"] is None
                and receipt["non_timing_journal_parity_passed"] is True
                and receipt["original_fit_calls"] == 672 and receipt["captured_pairs"] == len(plan["capture_pairs"][clip]),
                "failed or incomplete diagnostic execution")
        for flag in ("candidate_algorithm_changed_relative_to_original","global_motion_gates_changed","detector_configuration_changed",
                     "tracker_configuration_changed","extra_vpi_readbacks","estimator_method_source_changed"):
            require(receipt[flag] is False, "passive trace policy changed: " + flag)
        require(receipt["full_causal_history"] is True and pre["passed"] is True
                and pre["schema"] == receipt["schema"]+".preflight" and pre["detector_run"] is False
                and pre["passive_capture_check"] == dict(passed=True,generated_only=True,original_fit_calls=1,
                    same_return_object=True,input_arrays_unchanged=True,exact_array_byte_serialization=True,native_gray_hashes=True),
                "preflight/causal history incomplete")
        for key in ("preflight","trace","parity","journal","report","launch"):
            require(receipt[key+"_sha256"] == hashes[str(paths[key])], "receipt artifact binding differs: " + key)
        expected_input=receipt["input_sha256"]
        require(pre["input_sha256"] == trace["input_sha256"] == expected_input
                and expected_input["files"] == files and expected_input["freeze_sha256"] == freeze_sha256
                and expected_input["plan_sha256"] == hashes[str(plan_path)]
                and expected_input["original_artifacts"] == plan["original_artifacts"]
                and expected_input["original_workspace"] == plan["original_workspace"]
                and expected_input["original_freeze_sha256"] == plan["original_freeze_sha256"]
                and expected_input["original_runner_sha256"] == ORIGINAL_RUNNER_SHA, "trace input identity differs")
        require(receipt["source"] == trace["source"] == plan["sources"][clip]
                and receipt["candidate"] == trace["candidate"] == plan["candidate"]
                and receipt["clip"] == pre["clip"] == trace["clip"] == clip, "trace source/candidate differs")
        require(report["completed"] is True and report["full_clip"] is True and report["frames"] == 673
                and report["source_sha256"] == launch["source_sha256"] == SOURCE_SHA[clip], "diagnostic report differs")
        require(trace["schema"] == "seaqr.feature-residual-trace.v1.trace"
                and trace["capture_pairs"] == plan["capture_pairs"][clip]
                and trace["capture_groups"] == plan["capture_groups"][clip]
                and trace["original_fit_calls"] == 672 and trace["full_causal_history"] is True
                and trace["captured_pairs"] == len(plan["capture_pairs"][clip])
                and [r["current_frame_index"] for r in trace["rows"]] == plan["capture_pairs"][clip], "trace row inventory differs")
        old_paths={k:ORIGINAL_ROOT/clip/f for k,f in ARTIFACT_FILES.items()}
        for key,path in old_paths.items():bind(path,hashes,plan["original_artifacts"][clip][key])
        old_receipt=read(old_paths["execution_receipt"])
        require(old_receipt["passed"] is True and old_receipt["processed_frames"] == 673
                and old_receipt["input_sha256"] == expected_input["selection_inputs"], "original candidate execution binding differs")
        require(parity["schema"] == "seaqr.feature-residual-trace.v1.parity" and parity["passed"] is True
                and parity["rows_compared"] == 673 and parity["mismatch_frames"] == []
                and parity["excluded_paths"] == IGNORED_PATHS
                and parity["original_journal_sha256"] == hashes[str(old_paths["journal"])]
                and parity["diagnostic_journal_sha256"] == hashes[str(paths["journal"])], "full parity receipt differs")
        captures=verify_full_parity(old_paths["journal"],paths["journal"],plan["capture_pairs"][clip])
        clips[clip]=dict(trace=trace,receipt=receipt,captures=captures,trace_path=str(paths["trace"]),trace_sha256=hashes[str(paths["trace"])])
    for clip,data in clips.items():
        require(data["receipt"]["both_preflight_sha256"] == {c:hashes[str(evidence_root/c/"preflight.json")] for c in CLIPS},
                "both preflights were not bound before execution")
    return dict(plan=plan,clips=clips,input_sha256=hashes)


def distribution(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = values[np.isfinite(values)]
    result = dict(total=len(values), finite=len(finite), unavailable=len(values)-len(finite),
                  minimum=None, p10=None, median=None, p90=None, p95=None, maximum=None, mean=None)
    if len(finite):
        result.update(dict(zip(("minimum", "p10", "median", "p90", "p95", "maximum"),
                               map(float, np.quantile(finite, [0, .1, .5, .9, .95, 1])))))
        result["mean"] = float(np.mean(finite))
    return result


def midranks(values):
    """Zero-based average ranks for ties; never fabricate original U32 precision."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    require(np.isfinite(values).all(), "ranks require finite saved values")
    order = np.argsort(values, kind="stable")
    result = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        result[order[start:end]] = (start + end - 1) / 2
        start = end
    return result


def rank_association(left, right):
    left, right = np.asarray(left).reshape(-1), np.asarray(right).reshape(-1)
    require(left.shape == right.shape, "association dimensions differ")
    keep = np.isfinite(left) & np.isfinite(right)
    result = dict(total=len(left), paired_finite=int(keep.sum()), spearman=None,
                  descriptive_only=True, population_p_value=None)
    if keep.sum() >= 2:
        x, y = midranks(left[keep]), midranks(right[keep])
        x -= np.mean(x)
        y -= np.mean(y)
        denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
        if denominator > 0:
            result["spearman"] = float(np.clip(np.dot(x, y) / denominator, -1, 1))
    return result


def vector_summary(vectors):
    vectors = np.asarray(vectors, dtype=np.float64).reshape(-1, 2)
    finite = vectors[np.isfinite(vectors).all(axis=1)]
    return dict(total=len(vectors), finite=len(finite), component_median_xy=np.median(finite, axis=0).tolist() if len(finite) else None,
                x=distribution(vectors[:, 0]), y=distribution(vectors[:, 1]), norm=distribution(np.linalg.norm(vectors, axis=1)))


def cell_ids(previous):
    points = np.asarray(previous, dtype=np.float64)
    require(points.ndim == 2 and points.shape[1] == 2 and np.isfinite(points).all(), "invalid native previous points")
    width, height = IMAGE_SIZE_WH
    require(np.all((points >= 0) & (points < [width, height])), "previous points outside native image")
    return (np.floor(points[:, 1] * GRID_ROWS / height).astype(int) * GRID_COLS
            + np.floor(points[:, 0] * GRID_COLS / width).astype(int))


def spatial_summary(previous, displacement, residual_vectors, residuals, inliers):
    ids = cell_ids(previous)
    finite_residual = np.isfinite(residuals)
    cells = []
    for cell in range(GRID_ROWS * GRID_COLS):
        mask = ids == cell
        count = int(mask.sum())
        vectors = residual_vectors[mask]
        finite_vectors = vectors[np.isfinite(vectors).all(axis=1)]
        median = np.median(finite_vectors, axis=0) if len(finite_vectors) >= 5 else None
        cells.append(dict(cell_id=cell, row=cell // GRID_COLS, col=cell % GRID_COLS, points=count,
            original_inliers=int((mask & inliers).sum()), finite_residuals=int((mask & finite_residual).sum()),
            displacement_median_xy=np.median(displacement[mask], axis=0).tolist() if count >= 5 else None,
            residual_vector_median_xy=None if median is None else median.tolist(),
            within_cell_median_vector_scatter_px=None if median is None else float(np.median(np.linalg.norm(finite_vectors-median, axis=1))),
            minimum_points_for_vector_summary=5,
            tail_counts={str(t): int((mask & finite_residual & (residuals > t)).sum()) for t in RESIDUAL_THRESHOLDS}))
    tails = {}
    for threshold in RESIDUAL_THRESHOLDS:
        tail = finite_residual & (residuals > threshold)
        counts = np.bincount(ids[tail], minlength=48)
        total = int(tail.sum())
        tails[str(threshold)] = dict(point_count=total, finite_residual_denominator=int(finite_residual.sum()),
            occupied_cells=int(np.count_nonzero(counts)), maximum_cell_share=float(counts.max()/total) if total else None,
            largest_cell_ids=np.flatnonzero(counts == counts.max()).tolist() if total else [],
            concentration_hhi=float(np.sum((counts/total)**2)) if total else None,
            all_lk_accepted_maximum_cell_share=float(max((c["points"] for c in cells), default=0)/len(ids)) if len(ids) else None)
    return dict(grid_rows=6, grid_cols=8, coordinates="previous native pixels", cells=cells,
                descriptive_point_tail_concentration=tails)


def population_summary(mask, displacement, residual_vectors, residuals, scores, fb):
    return dict(points=int(mask.sum()), harris_score=distribution(scores[mask]), forward_backward_error_px=distribution(fb[mask]),
                saved_residual_norm_px=distribution(residuals[mask]), displacement=vector_summary(displacement[mask]),
                residual_vector=vector_summary(residual_vectors[mask]))


def point_analysis(previous, current, scores, fb, inliers, residuals, parameters):
    """Use the returned fit only. No search, refit, point deletion or new gate."""
    previous, current = np.asarray(previous), np.asarray(current)
    scores, fb = np.asarray(scores).reshape(-1), np.asarray(fb).reshape(-1)
    inliers, residuals = np.asarray(inliers), np.asarray(residuals, dtype=np.float64).reshape(-1)
    count = len(previous)
    require(previous.shape == current.shape == (count, 2) and count <= MAX_POINTS
            and scores.shape == fb.shape == inliers.shape == residuals.shape == (count,)
            and inliers.dtype == np.bool_, "capture dimensions/dtypes differ")
    require(np.isfinite(previous).all() and np.isfinite(current).all()
            and np.isfinite(scores).all() and np.isfinite(fb).all() and np.all(fb >= 0), "nonfinite accepted correspondence")
    displacement = current.astype(np.float64) - previous.astype(np.float64)
    if parameters is None:
        require(not inliers.any() and np.isnan(residuals).all(), "undefined fit has fabricated inliers/residuals")
        residual_vectors = np.full((count, 2), np.nan)
    else:
        translation = np.asarray([parameters["translation_x_px"], parameters["translation_y_px"]], dtype=np.float64)
        require(np.isfinite(translation).all() and np.isfinite(residuals).all() and np.all(residuals >= 0), "invalid saved translation/residuals")
        residual_vectors = current.astype(np.float64) - (previous.astype(np.float64) + translation)
        # The immutable returned transform is evaluated only to verify captured
        # scalar/vector identity, not to estimate another transform.
        require(np.allclose(np.linalg.norm(residual_vectors, axis=1), residuals, rtol=0, atol=1e-9), "saved residual identity differs")
        require(np.array_equal(inliers, residuals <= 1.0), "original RANSAC1px mask differs")
    finite_residuals = np.isfinite(residuals)
    populations = dict(all_lk_accepted=np.ones(count, bool), original_ransac_inliers=inliers,
                       original_ransac_outliers=finite_residuals & ~inliers, unavailable_original_residual=~finite_residuals)
    norm = np.linalg.norm(displacement, axis=1)
    ranks = (midranks(scores) + .5) / count if count else np.array([], dtype=np.float64)
    bins = np.searchsorted([.25, .5, .75], ranks, side="right")
    score_bins = []
    for index in range(4):
        mask = bins == index
        score_bins.append(dict(bin=index, points=int(mask.sum()), original_inliers=int((mask & inliers).sum()),
            original_outliers=int((mask & finite_residuals & ~inliers).sum()),
            residual_norm_px=distribution(residuals[mask]), fb_error_px=distribution(fb[mask]),
            score=distribution(scores[mask])))
    return dict(points=count, populations={name: population_summary(mask, displacement, residual_vectors, residuals, scores, fb)
                                         for name, mask in populations.items()},
        spatial=spatial_summary(previous, displacement, residual_vectors, residuals, inliers),
        harris_rank=dict(population="only original LK-accepted correspondences; not all selected or raw Harris points",
            ranking="average tied score rank; (zero_based_midrank+.5)/N; cuts.25/.5/.75 searchsortedright",
            original_score_precision_not_recovered=True, empty_tied_bins_retained=True, bins=score_bins),
        associations=dict(fb_vs_residual=rank_association(fb, residuals), fb_vs_displacement=rank_association(fb, norm),
            harris_vs_residual=rank_association(scores, residuals), harris_vs_fb=rank_association(scores, fb),
            harris_vs_displacement=rank_association(scores, norm)),
        original_threshold_context=dict(inlier_median_gate_px=.35, inlier_p90_gate_px=.75, ransac_inlier_gate_px=1,
            point_tail_counts_are_not_new_frame_acceptance_gates=True))


def analyze_row(row, clip, journal_motion, groups):
    previous,current,scores,fb,inliers,residuals=unpack_row(row,clip)
    fit,corr=row["fit"],row["correspondence"]
    require(corr["metrics"] == journal_motion["correspondence_metrics"] and corr["backends"] == journal_motion["motion_backends"],
            "captured correspondence diagnostics differ from causal journal")
    saved=journal_motion["motion_fit"]
    for key in ("model","quality_status","rejection_reasons","parameters","metrics"):
        require(fit[key] == saved[key], "captured original fit differs from journal: " + key)
    matrix=None if fit["previous_to_current_matrix"] is None else decode_array(fit["previous_to_current_matrix"],"<f8",(3,3)).tolist()
    require(matrix == saved["previous_to_current_matrix"], "captured original matrix differs from journal")
    metrics=fit["metrics"]
    require(metrics["inlier_count"] == int(inliers.sum())
            and abs(metrics["inlier_ratio"]-(float(inliers.mean()) if len(inliers) else 0)) <= 1e-15,
            "saved inlier accounting differs")
    if fit["parameters"] is not None:
        require(inliers.any(), "defined original fit has no inliers")
        for key,value in (("median_reprojection_error_px",np.median(residuals[inliers])),
                          ("p90_reprojection_error_px",np.percentile(residuals[inliers],90)),
                          ("maximum_reprojection_error_px",np.max(residuals[inliers]))):
            require(type(metrics[key]) in (int,float) and abs(metrics[key]-float(value)) <= 1e-9, "saved inlier statistic differs: " + key)
    stats=point_analysis(previous,current,scores,fb,inliers,residuals,fit["parameters"])
    return dict(previous_frame_index=row["previous_frame_index"],current_frame_index=row["current_frame_index"],
        groups=[name for name,indices in groups.items() if row["current_frame_index"] in indices],
        native_gray_pixel_sha256=row["native_gray_pixel_sha256"],
        original_fit={**{k:fit[k] for k in ("model","quality_status","rejection_reasons","parameters","metrics")},
                      "previous_to_current_matrix":matrix},
        original_correspondence_metrics=corr["metrics"],raw_lk_filtered_coordinates_available=False,
        original_fit_statistics_reproduced=True,analysis=stats)


def group_summary(rows, indices):
    by_index={row["current_frame_index"]:row for row in rows}
    require(len(indices) == len(set(indices)) and set(indices) <= set(by_index), "group inventory differs")
    selected=[by_index[i] for i in indices]
    reasons=Counter(reason for row in selected for reason in row["original_fit"]["rejection_reasons"])
    populations={}
    for name in ("all_lk_accepted","original_ransac_inliers","original_ransac_outliers","unavailable_original_residual"):
        values=[row["analysis"]["populations"][name] for row in selected]
        populations[name]=dict(point_total=sum(value["points"] for value in values),
            points_per_pair=distribution([value["points"] for value in values]),
            per_pair_residual_medians=distribution([value["saved_residual_norm_px"]["median"] for value in values]),
            per_pair_fb_medians=distribution([value["forward_backward_error_px"]["median"] for value in values]))
    return dict(frame_indices=list(indices),selected_pairs=len(selected),
        accepted_original_fits=sum(row["original_fit"]["quality_status"] == "accepted" for row in selected),
        rejection_reason_counts=dict(reasons),populations=populations,
        tail_cell_concentration={str(t):dict(
            pooled_point_count=sum(row["analysis"]["spatial"]["descriptive_point_tail_concentration"][str(t)]["point_count"] for row in selected),
            per_pair_maximum_cell_share=distribution([row["analysis"]["spatial"]["descriptive_point_tail_concentration"][str(t)]["maximum_cell_share"] for row in selected]))
            for t in RESIDUAL_THRESHOLDS},
        descriptive_association_by_pair={name:distribution([row["analysis"]["associations"][name]["spearman"] for row in selected])
            for name in ("fb_vs_residual","fb_vs_displacement","harris_vs_residual","harris_vs_fb","harris_vs_displacement")},
        statistical_unit="predeclared selected pair; temporally correlated and groups may overlap, not independent population samples")


def run(evidence_root,bundle_root,freeze_sha256,output):
    output=Path(output).absolute()
    require(output.parent == OUTPUT_ROOT and output.suffix == ".json" and output.parent.resolve() == output.parent
            and not output.exists() and not output.is_symlink(), "fresh analysis JSON in fixed root required")
    verified=verify_inputs(evidence_root,bundle_root,freeze_sha256)
    hashes=verified["input_sha256"]
    bind(Path(__file__).resolve(),hashes)
    bind(ROOT/"tests/unit/test_feature_residual_analysis.py",hashes)
    plan=verified["plan"]
    clips={}
    for clip,data in verified["clips"].items():
        rows=[analyze_row(row,clip,data["captures"][row["current_frame_index"]],plan["capture_groups"][clip]) for row in data["trace"]["rows"]]
        clips[clip]=dict(trace_path=data["trace_path"],trace_sha256=data["trace_sha256"],capture_pairs=plan["capture_pairs"][clip],
            source=plan["sources"][clip],rows=rows,
            groups={name:group_summary(rows,indices) for name,indices in {"all_captured":plan["capture_pairs"][clip],**plan["capture_groups"][clip]}.items()},
            full_causal_non_timing_parity_verified_frames=673)
    plan_path=Path(bundle_root)/"feature_residual_trace_plan.json"
    result=dict(schema="seaqr.feature-residual-trace.analysis.v1",passed_integrity=True,input_sha256=hashes,
        plan_path=str(plan_path),plan_sha256=hashes[str(plan_path)],freeze_sha256=freeze_sha256,clips=clips,
        source_media_read=False,new_fit_performed=False,new_acceptance_gate=False,independent_geometry_truth=False,production_promotion=False,
        limitations=["All point populations already passed LK/status/bounds/displacement/forward-backward filtering; rejected coordinates are unavailable, not zero.",
            "Pair-local correspondence indices and tied saved float32 Harris ranks do not recover raw Harris identities or original U32 precision.",
            "Reported original fit median/P90/max are inlier-only. All accepted points and original RANSAC outliers are shown separately.",
            "0.35/0.75/1px point-tail summaries are descriptive; original frame gates remain unchanged.",
            "Only the returned translation is evaluated to verify stored residuals. There is no alternative model, fit, threshold selection, or identity fallback.",
            "Failed/adjacent/temporal/positive groups were selected from earlier exposed development evidence and may overlap; adjacent controls are not independent geometry truth.",
            "Rank associations and spatial concentration are descriptive, without population p-values or causal interpretation.",
            "Full non-timing causal journal parity is required before interpretation. Instrumented runtime is not a performance benchmark."])
    require(all(sha(path)==digest for path,digest in hashes.items()),"metadata changed during analysis")
    with output.open("x") as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
        stream.write("\n")
    return result


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("evidence-root","bundle-root","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--freeze-sha256",required=True)
    result=run(**vars(parser.parse_args()))
    print(json.dumps(dict(passed_integrity=result["passed_integrity"],pairs=sum(len(c["rows"]) for c in result["clips"].values()))))
