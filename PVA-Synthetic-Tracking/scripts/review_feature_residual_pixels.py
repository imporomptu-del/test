"""Bounded, hash-verified native-pixel forensics of six saved motion pairs.

This is post-run visualization, not motion estimation or object detection.
The fixed 23x23 native grayscale patches are NOT the actual PVA proxy image,
pyramid, or LK support. Their statistics are descriptive, not motion truth.
No source image is opened until the complete trace-analysis bindings pass.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[1]
SOURCE_DIRECTORY = REPOSITORY.parent / "outputs/seaqr_discovery_pair_20260928/sources"
SOURCE_HASHES = {
    "0170": "12848c0f0caedd697a3da51776ab1579bd634a7ae94343f8cbd2a8830ee340bc",
    "0240": "2f86f28785e302572a86e23688143edbd7f5f1f65e8a3434b86a427e79c6a585",
}
PAIR_ENDS = {"0170": [451, 452, 453], "0240": [343, 344, 345]}
PATCH_SIZE = 23  # Frozen before trace. No CLI patch-size/ROI/threshold search.
PATCH_RADIUS = PATCH_SIZE // 2
CONTACT_NEAREST_SCALE = 8
VECTOR_MAGNIFICATION = 32
TOP_PER_POPULATION = 3
NATIVE_SHAPE = (3190, 4784)
SOURCE_FRAMES = 673
SOURCE_FPS = 10.0
SCHEMA = "seaqr.feature-residual-pixels.v1"
ANALYSIS_SCHEMA = "seaqr.feature-residual-trace.analysis.v1"
LIMITATIONS = [
    "Only the original LK-accepted correspondences are present; raw LK/status/FB rejects are absent.",
    "The 23x23 native-gray bilinear patches are not the actual PVA proxy/pyramid/LK support.",
    "Image statistics and residual vectors are descriptive, not independent motion or object-class truth.",
    "Displayed features are motion-estimation points, not object detections; no detector is rerun.",
    "Pair-local indices are not persistent feature identities across frames.",
    "The six exploratory forensic pairs are not an accuracy evaluation sample.",
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_value(value):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), "Invalid SHA256")
    return value


def _pairs(items):
    result = {}
    for key, value in items:
        require(key not in result, "Duplicate JSON key: " + key)
        result[key] = value
    return result


def _float(text):
    value = float(text)
    require(math.isfinite(value), "Nonfinite JSON number")
    return value


def _constant(text):
    raise ValueError("Nonfinite JSON constant: " + text)


def read_json(path):
    return json.loads(Path(path).read_text(), object_pairs_hook=_pairs,
                      parse_float=_float, parse_constant=_constant)


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def check_hashes(bindings):
    require(isinstance(bindings, dict) and bindings, "Missing input hash bindings")
    for raw_path, expected in bindings.items():
        path = Path(raw_path)
        require(path.is_absolute() and path.resolve() == path and path.is_file() and not path.is_symlink(), "Non-file or linked input: " + str(path))
        require(sha(path) == digest_value(expected), "Input changed: " + str(path))


def bind_file(path, expected, bindings):
    path = Path(path)
    require(str(path) in bindings and bindings[str(path)] == digest_value(expected),
            "Artifact absent from verified analysis input hashes: " + str(path))
    return path


def bilinear_patch(image, xy):
    """Sample fixed native support; unavailable support is not padded/clipped."""
    image = np.asarray(image)
    require(image.ndim == 2 and image.dtype == np.uint8, "Native uint8 grayscale required")
    point = np.asarray(xy, dtype=np.float64)
    require(point.shape == (2,), "A single native x/y point is required")
    if not np.isfinite(point).all():
        return None, "nonfinite_coordinate"
    x, y = point
    h, w = image.shape
    if x - PATCH_RADIUS < 0 or y - PATCH_RADIUS < 0 or x + PATCH_RADIUS > w - 1 or y + PATCH_RADIUS > h - 1:
        return None, "edge_insufficient_native_support"
    xs = x + np.arange(-PATCH_RADIUS, PATCH_RADIUS + 1, dtype=np.float64)
    ys = y + np.arange(-PATCH_RADIUS, PATCH_RADIUS + 1, dtype=np.float64)
    x0, y0 = np.floor(xs).astype(int), np.floor(ys).astype(int)
    # The min only handles exact integer samples at the final source pixel:
    # its fractional weight is zero. No out-of-image sample is admitted above.
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    wx, wy = xs - x0, ys - y0
    upper = image[y0[:, None], x0] * (1 - wx) + image[y0[:, None], x1] * wx
    lower = image[y1[:, None], x0] * (1 - wx) + image[y1[:, None], x1] * wx
    return upper * (1 - wy[:, None]) + lower * wy[:, None], "available"


def patch_statistics(patch):
    if patch is None:
        return None
    patch = np.asarray(patch, dtype=np.float64)
    require(patch.shape == (PATCH_SIZE, PATCH_SIZE) and np.isfinite(patch).all(), "Invalid sampled patch")
    gx = (patch[1:-1, 2:] - patch[1:-1, :-2]) / 2
    gy = (patch[2:, 1:-1] - patch[:-2, 1:-1]) / 2
    tensor = np.array([[np.mean(gx * gx), np.mean(gx * gy)],
                       [np.mean(gx * gy), np.mean(gy * gy)]], np.float64)
    eigenvalues = np.linalg.eigvalsh(tensor)
    total = float(eigenvalues.sum())
    return dict(mean=float(np.mean(patch)), std=float(np.std(patch)), std_ddof=0,
                minimum=float(np.min(patch)), maximum=float(np.max(patch)),
                gradient_structure_eigenvalues_ascending=eigenvalues.tolist(),
                gradient_structure_tensor=tensor.tolist(),
                anisotropy=None if total <= 0 else float(np.clip((eigenvalues[1] - eigenvalues[0]) / total, 0, 1)),
                gradient_support_pixels=int(gx.size),
                gradient_definition="central differences on the 21x21 patch interior; native gray levels/pixel")


def photometry(previous_gray, current_gray, previous_xy, current_xy):
    before, previous_status = bilinear_patch(previous_gray, previous_xy)
    after, current_status = bilinear_patch(current_gray, current_xy)
    result = dict(previous_status=previous_status, current_status=current_status,
                  previous=patch_statistics(before), current=patch_statistics(after),
                  aligned_zero_mean_rmse=None, aligned_raw_rmse=None, aligned_mean_change=None)
    if before is not None and after is not None:
        result.update(aligned_zero_mean_rmse=float(np.sqrt(np.mean(((after - np.mean(after)) - (before - np.mean(before))) ** 2))),
                      aligned_raw_rmse=float(np.sqrt(np.mean((after - before) ** 2))),
                      aligned_mean_change=float(np.mean(after) - np.mean(before)))
    return result


def selected_indices(residuals, inliers):
    """Preserve top-residual examples even if their patch is unavailable."""
    residuals, inliers = np.asarray(residuals), np.asarray(inliers)
    require(residuals.ndim == 1 and inliers.shape == residuals.shape and inliers.dtype == np.bool_, "Invalid selection arrays")
    require(np.isfinite(residuals).all() and np.all(residuals >= 0), "Finite original residuals required")
    return {name: sorted(np.flatnonzero(mask).tolist(), key=lambda i: (-float(residuals[i]), i))[:TOP_PER_POPULATION]
            for name, mask in (("inliers", inliers), ("outliers", ~inliers))}


def analyze_pixels(previous_gray, current_gray, arrays):
    p, q, scores, fb, mask, residuals = arrays
    require(len(p) <= 384 and p.shape == q.shape == (len(p), 2), "Point array scope differs")
    require(all(np.asarray(a).shape == (len(p),) for a in (scores, fb, mask, residuals)), "Point dimensions differ")
    shown = selected_indices(residuals, mask)
    points = []
    for i in range(len(p)):
        points.append(dict(point_index=i, previous_xy=p[i].tolist(), current_xy=q[i].tolist(),
            harris_score=float(scores[i]), forward_backward_error_px=float(fb[i]),
            original_ransac_inlier=bool(mask[i]), original_residual_px=float(residuals[i]),
            photometry=photometry(previous_gray, current_gray, p[i], q[i])))
    return dict(all_lk_accepted_points=len(points), original_ransac_inliers=int(mask.sum()),
                original_ransac_outliers=int((~mask).sum()), all_points=points,
                valid_aligned_patches=sum(point["photometry"]["aligned_zero_mean_rmse"] is not None for point in points),
                displayed_point_indices=shown,
                display_selection="Top3 original residual norms descending in each original inlier/outlier population; ties by pair-local index ascending; missing patches are not replaced")


def _cv2():
    import cv2
    return cv2


def _text(canvas, text, x, y, scale=.55, color=(225, 225, 225)):
    cv2 = _cv2()
    cv2.putText(canvas, str(text), (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def _save_png(path, image):
    cv2 = _cv2()
    require(not Path(path).exists(), "Refusing to replace figure")
    ok, encoded = cv2.imencode(".png", image)
    require(ok, "PNG encoding failed")
    with Path(path).open("xb") as stream:
        stream.write(encoded.tobytes())


def render_overview(path, gray, clip, pair_end, arrays, matrix):
    cv2 = _cv2()
    p, q, _, _, mask, residuals = arrays
    matrix = np.asarray(matrix, np.float64)
    require(matrix.shape == (3, 3) and np.isfinite(matrix).all(), "Original saved transform required")
    projected = np.column_stack((p.astype(np.float64), np.ones(len(p)))) @ matrix.T
    require(np.all(projected[:, 2] != 0), "Invalid homogeneous transform")
    prediction = projected[:, :2] / projected[:, 2:3]
    vectors = q.astype(np.float64) - prediction
    require(np.allclose(np.linalg.norm(vectors, axis=1), residuals, rtol=0, atol=1e-9), "Saved transform/residual mismatch")
    panel = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for index, (point, vector) in enumerate(zip(q, vectors)):
        start = tuple(np.rint(point).astype(int))
        end = tuple(np.rint(point + VECTOR_MAGNIFICATION * vector).astype(int))
        color = (65, 235, 100) if mask[index] else (240, 80, 230)
        cv2.circle(panel, start, 3, color, 1, cv2.LINE_AA)
        if start != end:
            cv2.arrowedLine(panel, start, end, color, 1, cv2.LINE_AA, tipLength=.15)
    header = 124
    canvas = np.full((gray.shape[0] + header, gray.shape[1], 3), 20, np.uint8)
    canvas[header:] = panel
    _text(canvas, f"chunk{clip} pair {pair_end-1}->{pair_end} | MOTION FEATURES, NOT OBJECT DETECTIONS", 20, 34, 1.1)
    _text(canvas, f"Native image 1:1. q - T(p) residual vectors anchored at q; arrows x{VECTOR_MAGNIFICATION} in native pixels.", 20, 71, 1.0)
    _text(canvas, "Green: original RANSAC inlier. Magenta: original outlier. No contrast enhancement. Image origin below124px header.", 20, 107, .9)
    _save_png(path, canvas)
    return dict(path=Path(path).name, sha256=sha(path), source_image_scale=1, source_origin_xy=[0, header],
                source_shape_hw=list(gray.shape), canvas_shape_hw=list(canvas.shape[:2]),
                vector_magnification=VECTOR_MAGNIFICATION, vector_definition="q - original T(p), anchored at q", contrast_enhancement=False)


def _show_patch(canvas, patch, status, x, y):
    cv2 = _cv2()
    _text(canvas, "1x", x, y + 14, .45)
    _text(canvas, f"{CONTACT_NEAREST_SCALE}x nearest", x + 38, y + 14, .45)
    if patch is None:
        _text(canvas, "UNAVAILABLE", x, y + 63, .45)
        _text(canvas, status, x, y + 89, .36)
        return
    pixels = np.rint(patch).clip(0, 255).astype(np.uint8)
    pixels = cv2.cvtColor(pixels, cv2.COLOR_GRAY2BGR)
    canvas[y + 25:y + 25 + PATCH_SIZE, x:x + PATCH_SIZE] = pixels
    enlarged = cv2.resize(pixels, None, fx=CONTACT_NEAREST_SCALE, fy=CONTACT_NEAREST_SCALE, interpolation=cv2.INTER_NEAREST)
    h, w = enlarged.shape[:2]
    canvas[y + 25:y + 25 + h, x + 38:x + 38 + w] = enlarged


def render_contact_sheet(path, previous_gray, current_gray, clip, pair_end, arrays, point_result):
    p, q, _, _, mask, residuals = arrays
    selection = point_result["displayed_point_indices"]
    indices = selection["inliers"] + selection["outliers"]
    height = 122 + max(len(indices), 1) * 250
    canvas = np.full((height, 1200, 3), 20, np.uint8)
    _text(canvas, f"chunk{clip} {pair_end-1}->{pair_end}: accepted LK feature patches, NOT object detections", 16, 28, .7)
    _text(canvas, "23x23 native bilinear samples; original0-255 brightness. 1x plus8x nearest display; no contrast enhancement.", 16, 57, .6)
    _text(canvas, "NOT actual PVA proxy / pyramid / LK support. Top3 residual inliers + top3 outliers, ties by index.", 16, 86, .6)
    if not indices:
        _text(canvas, "No accepted correspondences; populations not replaced by other points.", 16, 175, .65)
    for row, index in enumerate(indices):
        y = 122 + row * 250
        before, before_status = bilinear_patch(previous_gray, p[index])
        after, after_status = bilinear_patch(current_gray, q[index])
        label = "inlier" if mask[index] else "outlier"
        _text(canvas, f"index{index} original {label}; residual{residuals[index]:.6f}px", 16, y + 24, .58)
        _text(canvas, f"previous p=({p[index,0]:.3f},{p[index,1]:.3f})", 475, y + 1, .52)
        _text(canvas, f"current q=({q[index,0]:.3f},{q[index,1]:.3f})", 790, y + 1, .52)
        stats = point_result["all_points"][index]["photometry"]
        for line, key in enumerate(("aligned_zero_mean_rmse", "aligned_mean_change", "aligned_raw_rmse")):
            value = stats[key]
            _text(canvas, key + ": " + ("unavailable" if value is None else f"{value:.5f}"), 16, y + 64 + line * 29, .50)
        _show_patch(canvas, before, before_status, 475, y + 5)
        _show_patch(canvas, after, after_status, 790, y + 5)
    _save_png(path, canvas)
    return dict(path=Path(path).name, sha256=sha(path), displayed_point_indices=selection,
                native_sample_size=PATCH_SIZE, display_scales=[1, CONTACT_NEAREST_SCALE],
                interpolation="bilinear native sampling; nearest-neighbor display enlargement",
                display_quantization="round bilinear gray values to uint8", contrast_enhancement=False)


def native_pixel_sha(gray, index=0):
    if str(REPOSITORY) not in sys.path:
        sys.path.insert(0, str(REPOSITORY))
    from tiny_target.types import Frame, TimestampSource
    require(np.asarray(gray).dtype == np.uint8, "Native gray dtype differs")
    frame = Frame(gray, round(index / SOURCE_FPS * 1e9), index, "local-residual-pixel-review", 8,
                  TimestampSource.CONTAINER_RATE)
    return frame.pixel_sha256()


def expected_gray_hashes(rows):
    result = {}
    for end, row in rows.items():
        require(row["previous_frame_index"] == end - 1 and row["current_frame_index"] == end, "Nonadjacent trace pair")
        hashes = row["native_gray_pixel_sha256"]
        require(set(hashes) == {"previous", "current"}, "Native gray hash roles differ")
        for index, key in ((end - 1, "previous"), (end, "current")):
            value = digest_value(hashes[key])
            require(index not in result or result[index] == value, "Neighboring pairs disagree on native gray")
            result[index] = value
    return result


def decode_verified_pairs(source, rows):
    """Sequentially consume frame0 onward; no seek, skip, scale, or fit."""
    if str(REPOSITORY) not in sys.path:
        sys.path.insert(0, str(REPOSITORY))
    from tiny_target.visible_decode import VisibleFrameReader
    expected = expected_gray_hashes(rows)
    end = max(rows)
    previous = None
    with VisibleFrameReader(source, NATIVE_SHAPE, execution="sequential", max_frames=end + 1) as reader:
        require(reader.expected == SOURCE_FRAMES and reader.fps == SOURCE_FPS, "Local source container contract differs")
        for index in range(end + 1):
            decoded, _ = reader.read()
            require(decoded is not None and decoded.index == index, "Causal decode ended or reordered")
            gray = decoded.gray
            require(gray.shape == NATIVE_SHAPE and gray.dtype == np.uint8, "Native input contract differs")
            if index in expected:
                require(native_pixel_sha(gray, index) == expected[index], "Native gray differs from trace at frame" + str(index))
            if index in rows:
                require(previous is not None, "Previous native frame missing")
                # Both hashes were verified before yielding this pair.
                yield index, previous, gray
            previous = gray


def load_verified_analysis(path, expected_sha):
    path = Path(path).absolute()
    require(path.is_file() and not path.is_symlink() and sha(path) == digest_value(expected_sha), "Analysis identity differs")
    analysis = read_json(path)
    require(analysis.get("schema") == ANALYSIS_SCHEMA and analysis.get("passed_integrity") is True, "Full trace analysis has not passed")
    require(set(analysis["clips"]) == set(PAIR_ENDS), "Analysis source scope differs")
    bindings = dict(analysis["input_sha256"])
    require(all(Path(p).suffix in (".json", ".jsonl", ".py")
                and Path(p).is_relative_to(REPOSITORY.parent) for p in bindings), "Analysis binding is not local project metadata/code")
    check_hashes(bindings)
    analyzer_path = Path(__file__).with_name("analyze_feature_residual_trace.py").resolve()
    require(str(analyzer_path) in bindings and sha(analyzer_path) == bindings[str(analyzer_path)], "Analyzer code is not hash-bound")
    spec = importlib.util.spec_from_file_location("verified_pixel_trace_analysis", analyzer_path)
    analyzer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analyzer)
    plan_path = bind_file(analysis["plan_path"], analysis["plan_sha256"], bindings)
    plan = read_json(plan_path)
    require(plan["local_visual_followup"]["pair_ends"] == PAIR_ENDS, "Frozen pixel follow-up scope differs")
    require(set(plan["sources"]) == set(SOURCE_HASHES), "Frozen source scope differs")
    for clip, digest in SOURCE_HASHES.items():
        require(plan["sources"][clip]["sha256"] == digest and plan["sources"][clip]["frames"] == SOURCE_FRAMES,
                "Frozen source identity differs")
    rows = {}
    for clip, ends in PAIR_ENDS.items():
        item = analysis["clips"][clip]
        trace_path = bind_file(item["trace_path"], item["trace_sha256"], bindings)
        trace = read_json(trace_path)
        require(trace["schema"] == "seaqr.feature-residual-trace.v1.trace" and trace["clip"] == clip
                and trace["source"] == plan["sources"][clip] and trace["capture_pairs"] == plan["capture_pairs"][clip],
                "Trace source/scope binding differs")
        selected, seen = {}, []
        for row in trace["rows"]:
            index = row["current_frame_index"]
            require(type(index) is int and index not in seen, "Invalid or duplicated trace pair")
            seen.append(index)
            if index in ends:
                selected[index] = row
        require(set(selected) == set(ends) and seen == plan["capture_pairs"][clip], "Trace capture set differs")
        receipt_path = trace_path.parent / "execution_receipt.json"
        parity_path = trace_path.parent / "parity.json"
        require(str(receipt_path) in bindings and str(parity_path) in bindings, "Full trace receipts are unbound")
        receipt, parity = read_json(receipt_path), read_json(parity_path)
        require(receipt["schema"] == "seaqr.discovery-feature-residual-trace.v1" and receipt["passed"] is True
                and receipt["processed_frames"] == SOURCE_FRAMES and receipt["clip"] == clip
                and receipt["trace_sha256"] == item["trace_sha256"] and receipt["parity_sha256"] == bindings[str(parity_path)],
                "Full trace execution has not passed or hash binding differs")
        require(parity["schema"] == "seaqr.feature-residual-trace.v1.parity" and parity["passed"] is True
                and parity["rows_compared"] == SOURCE_FRAMES and parity["mismatch_frames"] == []
                and item["full_causal_non_timing_parity_verified_frames"] == SOURCE_FRAMES,
                "Full causal parity has not passed")
        analytical_rows = {r["current_frame_index"]: r for r in item["rows"]}
        for index, row in selected.items():
            require(analytical_rows[index]["native_gray_pixel_sha256"] == row["native_gray_pixel_sha256"], "Analysis/trace gray hash binding differs")
            analyzer.unpack_row(row, clip)
        expected_gray_hashes(selected)
        rows[clip] = selected
    bindings[str(path)] = expected_sha
    for code in (Path(__file__).resolve(), REPOSITORY / "tests/unit/test_feature_residual_pixels.py",
                 REPOSITORY / "tiny_target/types.py", REPOSITORY / "tiny_target/visible_decode.py"):
        bindings[str(code)] = sha(code)
    return analysis, analyzer, rows, bindings


def run(analysis_path, analysis_sha256, output):
    analysis, analyzer, rows, bindings = load_verified_analysis(analysis_path, analysis_sha256)
    sources = {clip: SOURCE_DIRECTORY / ("chunk_" + clip + ".avi") for clip in PAIR_ENDS}
    for clip, path in sources.items():
        require(path.resolve() == path and path.is_file() and not path.is_symlink(), "Approved local source missing or linked")
        require(sha(path) == SOURCE_HASHES[clip], "Local source identity differs: " + clip)
        bindings[str(path)] = SOURCE_HASHES[clip]
    output = Path(output).absolute()
    require(not output.exists() and not output.is_symlink(), "Fresh output directory required")
    output.mkdir(parents=False)
    try:
        result = dict(schema=SCHEMA + ".analysis", passed=False, limitations=LIMITATIONS,
                      patch_size=PATCH_SIZE, patch_interpolation="bilinear, no padding", point_statistics="all original LK-accepted points in each fixed pair",
                      input_sha256=bindings, pair_ends=PAIR_ENDS, clips={})
        figures = []
        for clip in PAIR_ENDS:
            pair_results = []
            for end, before, after in decode_verified_pairs(sources[clip], rows[clip]):
                row = rows[clip][end]
                arrays = analyzer.unpack_row(row, clip)
                stats = analyze_pixels(before, after, arrays)
                fit_matrix = analyzer.decode_array(row["fit"]["previous_to_current_matrix"], np.dtype("float64"), (3, 3))
                overview = render_overview(output / f"chunk{clip}_pair{end}_overview.png", after, clip, end, arrays, fit_matrix)
                contacts = render_contact_sheet(output / f"chunk{clip}_pair{end}_patches.png", before, after, clip, end, arrays, stats)
                figures.extend([overview, contacts])
                pair_results.append(dict(previous_frame_index=end-1, current_frame_index=end,
                    native_gray_pixel_sha256=row["native_gray_pixel_sha256"], statistics=stats, overview=overview, patch_contact_sheet=contacts))
                print(json.dumps(dict(clip=clip, pair_end=end, status="rendered", accepted_points=len(arrays[0]))), flush=True)
            require([r["current_frame_index"] for r in pair_results] == PAIR_ENDS[clip], "Rendered pair scope differs")
            result["clips"][clip] = dict(decoded_causally_from_frame=0, decoded_through_frame=max(PAIR_ENDS[clip]),
                decoded_frame_count=max(PAIR_ENDS[clip])+1, used_native_frame_indices=sorted(expected_gray_hashes(rows[clip])), rows=pair_results)
        check_hashes(bindings)
        for figure in figures:
            require(sha(output / figure["path"]) == figure["sha256"], "Rendered artifact changed")
        result["passed"] = True
        write_json(output / "pixel_statistics.json", result)
        check_hashes(bindings)
        receipt = dict(schema=SCHEMA + ".receipt", passed=True, completed_utc=datetime.now(timezone.utc).isoformat(),
                       input_sha256=bindings, figures=figures, statistics_sha256=sha(output / "pixel_statistics.json"),
                       source_pixels_mutated=False, model_fits=0, detector_runs=0, contrast_enhancement=False,
                       pair_count=6, native_patch_size=PATCH_SIZE, opencv_version=_cv2().__version__, numpy_version=np.__version__,
                       limitations=LIMITATIONS)
        write_json(output / "execution_receipt.json", receipt)
        return receipt
    except BaseException as exc:
        write_json(output / "failed_receipt.json", dict(schema=SCHEMA + ".failure", passed=False, error=repr(exc),
            input_sha256=bindings, partial_artifacts_are_not_validated=True))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--analysis-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    run(options.analysis, options.analysis_sha256, options.output)
