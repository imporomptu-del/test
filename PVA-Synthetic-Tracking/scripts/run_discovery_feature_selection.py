#!/usr/bin/env python3
"""Isolated two-clip Harris-selection candidate; not an unchanged-baseline claim.

Harris gain/capacity and an explicit eight-per-cell tracked-point budget change.
Existing exact CPU batching changes execution only. Native U8 images, LK,
global-motion gates, detector/tracker settings and failure policy stay fixed.
No source edits, clock writes, builds, fallback, labels, or automatic promotion.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, replace
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
from types import SimpleNamespace
from unittest.mock import patch

SCHEMA = "seaqr.discovery-feature-selection.v1"
WORKSPACE_PATTERN = r"/tmp/seaqr_feature_selection_20260929_[A-Za-z0-9]{6}"
BASELINE_WORKSPACE = Path("/tmp/seaqr_discovery_pair_20260928_ZGLHH7")
BASELINE_HELPER = BASELINE_WORKSPACE / "run_discovery_pair_baseline.py"
BASELINE_HELPER_SHA = "e5b56a8d5d6f6cae5aa40c5283405801c3556a3f39f80da801e128a4af6f9d41"
METHOD_SHA = "80aabc85b25b9204bc9de1838972ddd78a2b639d2b7b933609724d5ca2d7c733"
CANDIDATE = dict(harris_gain=16, harris_capacity_policy="complete_grid", feature_image_scale=0.5,
                 feature_cpu_policy="batched_exact_v1", max_features=384, max_features_per_cell=8,
                 grid_rows=6, grid_cols=8)
CONFIGURATION_CHANGES = {"harris_capacity_policy", "feature_cpu_policy", "max_features", "max_features_per_cell"}
PARITY_NAMES = ("native_gray8", "masked_and_excluded_gray8", "empty", "all_ineligible", "native_proxy_boundaries")
SOURCE_HASHES = {"0170": "12848c0f0caedd697a3da51776ab1579bd634a7ae94343f8cbd2a8830ee340bc",
                 "0240": "2f86f28785e302572a86e23688143edbd7f5f1f65e8a3434b86a427e79c6a585"}
GAIN_ANCHOR = '        conversion = {"offset": -32768.0} if uses_u16 else {}\n'
GAIN_REPLACEMENT = ('        conversion = {"offset": -32768.0} if uses_u16 '
                    'else {"scale": 16.0, "offset": 0.0}\n')
EXPECTED_BACKENDS = dict(intensity_conversion="CPU", motion_image_rescale="CUDA",
    gaussian_pyramid="PVA", harris_input_conversion="CUDA", harris="PVA",
    optical_flow_pyrlk="PVA", cpu_fallback=False)
CONTROL_NAMES = ("low_contrast_static", "low_contrast_translated", "flat_static")
CONTROL_SEED, CONTROL_WIDTH, CONTROL_HEIGHT, CONTROL_INTERIOR = 20260927, 640, 512, 128


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 32 * 1024 * 1024,
            f"missing/linked/oversize metadata: {path}")
    with path.open() as stream:
        return json.load(stream)


def validate_adapter_roundtrip(actual, saved):
    """Compare strict JSON values, since dataclass tuples serialize as lists."""
    for key in ("source_transformation", "effective_motion_configuration", "change_classification"):
        require(json.dumps(actual[key], sort_keys=True, allow_nan=False)
                == json.dumps(saved[key], sort_keys=True, allow_nan=False),
                "candidate implementation/config differs from preflight: " + key)


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def source_spec(clip):
    require(clip in SOURCE_HASHES, "unapproved source")
    return dict(path=f"/home/serg/project/camera_reader_sky/srcsky/chunks/chunk_{clip}.avi",
                sha256=SOURCE_HASHES[clip], frames=673, width=4784, height=3190,
                fps=10, codec="mjpeg", pixel_format="yuvj420p")


def scope_path(workspace):
    require(re.fullmatch(WORKSPACE_PATTERN, str(workspace)), "outside feature-selection workspace")
    return Path(workspace)


def workspace_guard(workspace, clip, mode):
    workspace = scope_path(workspace)
    source_spec(clip)
    require(mode in {"preflight", "run"}, "invalid mode")
    require(os.geteuid() != 0, "never run as root")
    require(workspace.is_dir() and not workspace.is_symlink(), "missing/linked workspace")
    directory = workspace / clip
    require(not directory.is_symlink() and (not directory.exists() or directory.is_dir()), "invalid clip directory")
    for name in ("run", "execution_receipt.json") + (("preflight.json",) if mode == "preflight" else ()):
        path = directory / name
        require(not path.exists() and not path.is_symlink(), "existing output; no overwrite")
    return workspace


def validate_freeze(value, hashes):
    require(value.get("schema") == "feature_selection.v1" and value.get("candidate") == CANDIDATE
            and value.get("baseline_workspace") == str(BASELINE_WORKSPACE), "candidate freeze differs")
    require(type(value["candidate"]["harris_gain"]) is int
            and type(value["candidate"]["feature_image_scale"]) in (int, float)
            and all(type(value["candidate"][key]) is int for key in
                    ("max_features", "max_features_per_cell", "grid_rows", "grid_cols")),
            "invalid candidate types")
    files = value.get("files", {})
    require(isinstance(files, dict) and "run_discovery_feature_selection.py" in files, "missing frozen runner")
    require(all(isinstance(name, str) and Path(name).name == name and name not in {"", ".", "..", "freeze.json"}
                and isinstance(digest, str) and re.fullmatch(r"[a-f0-9]{64}", digest)
                for name, digest in files.items()), "invalid frozen file names/hashes")
    require(files == hashes, "transferred artifacts differ from freeze")
    require(value.get("sources") == {c: source_spec(c) for c in SOURCE_HASHES}, "fixed sources differ")
    require(all(type(row[k]) is int for row in value["sources"].values()
                for k in ("frames", "width", "height", "fps")), "invalid source numerical types")


def load_baseline():
    require(BASELINE_HELPER.is_file() and not BASELINE_HELPER.is_symlink()
            and sha(BASELINE_HELPER) == BASELINE_HELPER_SHA, "baseline helper identity changed")
    spec = importlib.util.spec_from_file_location("feature_selection_baseline", BASELINE_HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inputs(workspace, clip, baseline):
    freeze = read(workspace / "freeze.json")
    validate_freeze(freeze, freeze.get("files", {}))
    hashes = {}
    for name in freeze["files"]:
        path = workspace / name
        require(path.is_file() and not path.is_symlink(), "missing/linked frozen artifact")
        hashes[name] = sha(path)
    validate_freeze(freeze, hashes)
    require(sha(Path(__file__)) == hashes["run_discovery_feature_selection.py"], "executed runner differs")
    require(sha(BASELINE_HELPER) == BASELINE_HELPER_SHA, "baseline helper changed")
    reference, runtime, baseline_hashes = baseline.inputs(BASELINE_WORKSPACE, clip)
    require(baseline.source_spec(clip) == source_spec(clip), "baseline source contract differs")
    return reference, runtime, dict(files=hashes, freeze_sha256=sha(workspace / "freeze.json"),
        baseline_helper_sha256=BASELINE_HELPER_SHA, baseline_inputs=baseline_hashes)


def transform_source(source):
    require(hashlib.sha256(source.encode()).hexdigest() == METHOD_SHA, "frozen generated estimator differs")
    require(source.count(GAIN_ANCHOR) == 1 and GAIN_REPLACEMENT not in source, "Harris gain anchor differs")
    transformed = source.replace(GAIN_ANCHOR, GAIN_REPLACEMENT)
    require(transformed.replace(GAIN_REPLACEMENT, GAIN_ANCHOR) == source,
            "non-gain estimator statements changed")
    return transformed


def candidate_config(config):
    require(config.feature_image_scale == 0.5 and config.harris_capacity_policy == "legacy_default"
            and config.flow_status_policy == "legacy_default" and config.minimum_accepted_features == 30
            and config.minimum_grid_coverage == 0.2 and config.feature_cpu_policy == "reference"
            and type(config.max_features) is int and config.max_features == 1000
            and config.max_features_per_cell is None and type(config.grid_rows) is int
            and type(config.grid_cols) is int and (config.grid_rows, config.grid_cols) == (6, 8),
            "unexpected baseline motion policy")
    result = replace(config, harris_capacity_policy="complete_grid", feature_cpu_policy="batched_exact_v1",
                     max_features=384, max_features_per_cell=8)
    before, after = asdict(config), asdict(result)
    changes = {key: dict(before=before[key], after=after[key]) for key in before if before[key] != after[key]}
    require(set(changes) == CONFIGURATION_CHANGES, "unexpected motion configuration change")
    return result, changes


def capacity_for_shape(shape):
    height, width = shape
    require(type(height) is int and type(width) is int and min(shape) > 0, "invalid native shape")
    mw, mh = round(width * 0.5), round(height * 0.5)
    return ((mw + 7) // 8 + 1) * ((mh + 7) // 8 + 1)


def validate_pyramid_dimensions(shape, config):
    """Conservative size bounds for the unchanged PVA pyramid, before controls.

    Flooring each half-scale level is a lower bound, not a claim about VPI's
    odd-dimension rounding. The generated control dimensions divide exactly.
    """
    height, width = shape
    require(type(height) is int and type(width) is int and min(shape) > 0,
            "invalid pyramid native shape")
    require(config.feature_image_scale == 0.5 and config.pyramid_scale == 0.5
            and type(config.pyramid_levels) is int and config.pyramid_levels == 4,
            "frozen pyramid configuration differs")
    mw, mh = round(width * config.feature_image_scale), round(height * config.feature_image_scale)
    require(mw <= 3264 and mh <= 2048, "proxy exceeds frozen PVA pyramid maximum")
    sizes = [[mw, mh]]
    for _ in range(config.pyramid_levels - 1):
        sizes.append([math.floor(value * config.pyramid_scale) for value in sizes[-1]])
    require(min(sizes[-1]) >= 32,
            f"PVA pyramid smallest-level lower bound {sizes[-1]} must be >=32x32")
    return dict(passed=True, native_shape_hw=list(shape), proxy_size_wh=[mw, mh],
                pyramid_levels=config.pyramid_levels, pyramid_scale=config.pyramid_scale,
                level_size_lower_bounds_wh=sizes, minimum_level_size_wh=[32, 32],
                dimension_rounding="conservative floor lower bounds; controls divide exactly")


def verify_correspondence(correspondence, shape):
    height, width = shape
    require(correspondence.backends == EXPECTED_BACKENDS, "unexpected motion backend/fallback")
    require(tuple(correspondence.full_image_size) == (width, height)
            and tuple(correspondence.motion_image_size) == (round(width * 0.5), round(height * 0.5)),
            "native/proxy dimensions changed")
    metrics = correspondence.metrics
    capacity = metrics.get("harris_output", {})
    require(capacity.get("capacity_policy") == "complete_grid"
            and capacity.get("capacity") == capacity_for_shape(shape)
            and capacity.get("capacity_exhausted") is False
            and 0 < metrics.get("detected_count", 0) < capacity["capacity"], "Harris capacity differs/exhausted")
    require(metrics.get("minimum_accepted_features") == 30 and metrics.get("minimum_grid_coverage") == 0.2
            and 0 < correspondence.count <= metrics.get("selected_count", 0) <= 384,
            "feature count/gates changed")


@contextmanager
def candidate_adapter(reuse, baseline_config, audit):
    effective, changes = candidate_config(baseline_config)
    original = reuse.generated_method()
    transformed = transform_source(original)
    namespace = dict(vars(reuse.pva))
    exec(compile(transformed, str(Path(__file__)) + ":harris_gain16", "exec"), namespace)
    audit.update(candidate=dict(CANDIDATE), original_motion_configuration=asdict(baseline_config),
        effective_motion_configuration=asdict(effective), configuration_changes=changes,
        change_classification=dict(algorithm=["Harris-only gain16", "complete Harris output capacity",
            "strongest eligible eight per unchanged6x8cell; at most384 tracked points"],
            execution_only=["existing batched_exact_v1 eligibility, selection and coverage"],
            harris_score_precision="unchanged float32 readback of U32 scores",
            motion_quality_gates_changed=False, detector_tracker_changed=False),
        source_transformation=dict(original_method_sha256=METHOD_SHA,
            transformed_method_sha256=hashlib.sha256(transformed.encode()).hexdigest(),
            exact_original_recovered=True, only_statement_change="U8 to Harris S16 scale16 offset0",
            heavy_capture_hooks=False), estimator_instances=0, successful_pair_backend_checks=0)

    class Candidate(reuse.ReuseMotionV12):
        def __init__(self, config):
            require(asdict(config) == asdict(baseline_config), "unexpected supplied estimator config")
            super().__init__(candidate_config(config)[0])
            audit["estimator_instances"] += 1

        def estimate(self, previous, current):
            require(previous.bit_depth == current.bit_depth == 8
                    and str(previous.image.dtype) == str(current.image.dtype) == "uint8"
                    and previous.shape == current.shape, "candidate accepts only native gray8")
            require(asdict(self.config) == asdict(effective), "effective feature configuration changed")
            result = super().estimate(previous, current)
            verify_correspondence(result, previous.shape)
            audit["successful_pair_backend_checks"] += 1
            return result

    with patch.object(reuse, "_ESTIMATE", namespace["_estimate_v12"]):
        yield Candidate


def translate_no_wrap(image, dx, dy, fill=128):
    import numpy as np
    require(image.ndim == 2 and image.dtype == np.uint8 and type(dx) is int and type(dy) is int,
            "control requires integer translation of gray8")
    h, w = image.shape
    require(abs(dx) < w and abs(dy) < h, "translation outside control")
    result = np.full_like(image, fill)
    x0, x1, y0, y1 = max(0, -dx), min(w, w - dx), max(0, -dy), min(h, h - dy)
    result[y0 + dy:y1 + dy, x0 + dx:x1 + dx] = image[y0:y1, x0:x1]
    return result


def generated_controls():
    import numpy as np
    rng = np.random.default_rng(CONTROL_SEED)
    cells = rng.integers(0, 16, size=((CONTROL_HEIGHT + 31) // 32, (CONTROL_WIDTH + 31) // 32), dtype=np.uint8)
    low = (120 + cells.repeat(32, 0).repeat(32, 1)[:CONTROL_HEIGHT, :CONTROL_WIDTH]).astype(np.uint8)
    flat = np.full_like(low, 128)
    return [(CONTROL_NAMES[0], low, low.copy(), (0, 0)),
            (CONTROL_NAMES[1], low, translate_no_wrap(low, 4, -2), (4, -2)),
            (CONTROL_NAMES[2], flat, flat.copy(), (0, 0))]


def control_metrics(correspondence, fit, expected):
    import numpy as np
    p, q = correspondence.previous_points.astype(np.float64), correspondence.current_points.astype(np.float64)
    width, height = correspondence.full_image_size
    dx, dy = expected
    margin = CONTROL_INTERIOR
    mask = ((p[:, 0] >= max(0, -dx) + margin) & (p[:, 0] < min(width, width - dx) - margin)
            & (p[:, 1] >= max(0, -dy) + margin) & (p[:, 1] < min(height, height - dy) - margin)
            & (q[:, 0] >= margin) & (q[:, 0] < width - margin)
            & (q[:, 1] >= margin) & (q[:, 1] < height - margin))
    errors = np.linalg.norm(q[mask] - p[mask] - expected, axis=1)
    parameters = fit.parameters or {}
    tx, ty = parameters.get("translation_x_px"), parameters.get("translation_y_px")
    vector_error = math.hypot(tx - dx, ty - dy) if all(type(v) in (int, float) and math.isfinite(v) for v in (tx, ty)) else None
    median, maximum = (float(np.median(errors)), float(errors.max())) if len(errors) else (None, None)
    gates = dict(original_fit_accepted=fit.accepted, at_least_30_interior=len(errors) >= 30,
        fit_vector_error_at_most_0_1=vector_error is not None and vector_error <= 0.1,
        interior_median_at_most_0_1=median is not None and median <= 0.1,
        interior_maximum_at_most_0_5=maximum is not None and maximum <= 0.5)
    return dict(passed=all(gates.values()), gates=gates, expected_displacement_xy=list(expected),
                all_accepted_points=correspondence.count, accepted_interior_points=len(errors),
                excluded_from_truth_summary=correspondence.count - len(errors), interior_margin_px=margin,
                truth_summary_uses_accepted_p_and_q_interior=True, translation_vector_error_px=vector_error,
                median_error_px=median, maximum_error_px=maximum)


def conversion_check():
    import numpy as np
    import vpi
    pixels = np.tile(np.arange(256, dtype=np.uint8), (32, 1))
    stream = vpi.Stream()
    image = vpi.asimage(pixels, vpi.Format.U8)
    converted = image.convert(vpi.Format.S16, backend=vpi.Backend.CUDA, stream=stream, scale=16.0, offset=0.0)
    stream.sync()
    with converted.rlock_cpu() as data:
        actual = np.array(data, copy=True)
    require(actual.dtype == np.int16 and np.array_equal(actual, pixels.astype(np.int16) * 16),
            "CUDA Harris conversion is not exact U8*16")
    return dict(passed=True, backend="CUDA", source_range=[0, 255], output_range=[int(actual.min()), int(actual.max())],
                no_input_clipping=True, internal_harris_arithmetic_precision_verified=False)


def generated_cpu_parity(motion_config):
    """Check the existing exact execution path on generated pixels only.

    The reference oracle uses the SAME candidate budget; this does not claim
    quota reduction is equivalent to the old algorithm. All operations execute
    under the caller's installed NumPy, including its scalar promotion rules.
    """
    import numpy as np
    from tiny_target.types import Frame, TimestampSource
    from tiny_target.motion.geometry import select_spatially_distributed, grid_coverage
    from tiny_target.motion.pva_pyrlk import _feature_eligibility

    effective, _ = candidate_config(motion_config)
    rng = np.random.default_rng(20260929)
    height, width = CONTROL_HEIGHT, CONTROL_WIDTH
    size = (width // 2, height // 2)
    image = rng.integers(120, 136, (height, width), dtype=np.uint8)
    image[64, 80] = 255
    mask = np.ones(image.shape, dtype=bool)
    mask[140, 320] = False
    dense = []
    for row in range(6):
        for col in range(8):
            for index in range(16):
                dense.append(((col + .2 + .6 * (index % 4) / 3) * size[0] / 8,
                              (row + .2 + .6 * (index // 4) / 3) * size[1] / 6))
    boundary = []
    for y in np.linspace(0, size[1], 7).astype(np.float32):
        for x in np.linspace(0, size[0], 9).astype(np.float32):
            boundary.extend(((np.nextafter(x, np.float32(-np.inf)), y), (x, y),
                (np.nextafter(x, np.float32(np.inf)), y),
                (x, np.nextafter(y, np.float32(-np.inf))),
                (x, np.nextafter(y, np.float32(np.inf)))))
    points = np.asarray(dense + boundary + [(39.75, 31.75), (159.75, 69.75),
        (np.nan, 1), (1, np.inf), (-np.inf, 0), (-1, 0)], dtype=np.float32)
    # Preserve production float32 score semantics, including deliberate ties
    # introduced by U32 conversion beyond2**24; do not improve precision here.
    integer_scores = np.resize(np.array([0, 1, 1, 9, 9, 2**24, 2**24 + 1, 2**32 - 1],
                                       dtype=np.uint32), len(points))
    scores = integer_scores.astype(np.float32)
    scores[-6] = np.nan
    scores[-5] = np.inf
    rows = []
    for name in PARITY_NAMES[:-1]:
        use_mask = mask if name == "masked_and_excluded_gray8" else None
        cfg = replace(effective, exclusion_regions_xyxy=((128., 96., 224., 176.),)) if use_mask is not None else effective
        p, s = (points[:0], scores[:0]) if name == "empty" else (points.copy(), scores.copy())
        if name == "all_ineligible":
            p[:] = -1
        source = Frame(image.copy(), 0, 0, "generated-selection-parity:" + name, 8,
                       TimestampSource.CONTAINER_RATE, valid_mask=use_mask)
        before_pixels = source.pixel_sha256()
        before_points, before_scores = p.tobytes(), s.tobytes()
        before_mask = None if use_mask is None else use_mask.tobytes()
        outputs = {}
        for policy in ("reference", "batched_exact_v1"):
            eligible, reasons = _feature_eligibility(p, source, size, replace(cfg, feature_cpu_policy=policy))
            options = dict(grid_rows=cfg.grid_rows, grid_cols=cfg.grid_cols, max_features=cfg.max_features,
                           max_per_cell=cfg.max_features_per_cell, eligible_mask=eligible, execution=policy)
            selected = select_spatially_distributed(p, s, size, **options)
            coverage = grid_coverage(p, size, grid_rows=cfg.grid_rows, grid_cols=cfg.grid_cols, execution=policy)
            outputs[policy] = dict(eligible=eligible, reasons=reasons, selected=selected, coverage=coverage)
        left, right = outputs["reference"], outputs["batched_exact_v1"]
        require(np.array_equal(left["eligible"], right["eligible"])
                and np.array_equal(left["selected"], right["selected"])
                and left["reasons"] == right["reasons"] and left["coverage"] == right["coverage"],
                "generated exact CPU parity failed: " + name)
        selected = right["selected"]
        require(len(selected) <= 384, "candidate exceeded total point budget")
        cells = {}
        for index in selected:
            x, y = p[index]
            cell = (min(5, int(y * 6 / size[1])), min(7, int(x * 8 / size[0])))
            cells[cell] = cells.get(cell, 0) + 1
        require(all(count <= 8 for count in cells.values()), "candidate exceeded per-cell quota")
        require(source.pixel_sha256() == before_pixels and p.tobytes() == before_points
                and s.tobytes() == before_scores
                and (use_mask is None or use_mask.tobytes() == before_mask), "generated parity inputs mutated")
        rows.append(dict(name=name, passed=True, generated_only=True, point_count=len(p),
            selected_count=len(selected), maximum_selected_per_cell=max(cells.values(), default=0),
            eligible_sha256=hashlib.sha256(right["eligible"].tobytes()).hexdigest(),
            selected_indices_sha256=hashlib.sha256(selected.tobytes()).hexdigest(),
            source_pixel_sha256=before_pixels, exclusions=right["reasons"], coverage=right["coverage"],
            masks_and_exclusions_generated_case_only=use_mask is not None))
    # The real2392x1595 proxy has fractional cell boundaries absent from the
    # small640x512controls. Check its exact index/coverage math under Jetson's
    # NumPy without allocating or reading a native-resolution source image.
    native_size = (2392, 1595)
    native_points = []
    for row in range(6):
        for col in range(8):
            for index in range(16):
                native_points.append(((col + .2 + .6 * (index % 4) / 3) * native_size[0] / 8,
                                      (row + .2 + .6 * (index // 4) / 3) * native_size[1] / 6))
    for y in np.linspace(0, native_size[1], 7).astype(np.float32):
        for x in np.linspace(0, native_size[0], 9).astype(np.float32):
            native_points.extend(((np.nextafter(x, np.float32(-np.inf)), y), (x, y),
                (np.nextafter(x, np.float32(np.inf)), y),
                (x, np.nextafter(y, np.float32(-np.inf))),
                (x, np.nextafter(y, np.float32(np.inf)))))
    native_points.extend(((np.nan, 1), (1, np.inf), (-np.inf, 0), (-1, 0)))
    p = np.asarray(native_points, dtype=np.float32)
    s = np.resize(integer_scores, len(p)).astype(np.float32)
    s[-6], s[-5] = np.nan, np.inf
    eligible = np.ones(len(p), dtype=bool)
    eligible[::17] = False
    before = (p.tobytes(), s.tobytes(), eligible.tobytes())
    outputs = {}
    for policy in ("reference", "batched_exact_v1"):
        selected = select_spatially_distributed(p, s, native_size, grid_rows=6, grid_cols=8,
            max_features=384, max_per_cell=8, eligible_mask=eligible, execution=policy)
        coverage = grid_coverage(p, native_size, grid_rows=6, grid_cols=8, execution=policy)
        outputs[policy] = (selected, coverage)
    left, right = outputs["reference"], outputs["batched_exact_v1"]
    require(np.array_equal(left[0], right[0]) and left[1] == right[1],
            "generated exact CPU parity failed: native_proxy_boundaries")
    cells = {}
    for index in right[0]:
        x, y = p[index]
        cell = (min(5, int(y * 6 / native_size[1])), min(7, int(x * 8 / native_size[0])))
        cells[cell] = cells.get(cell, 0) + 1
    require(len(right[0]) <= 384 and all(count <= 8 for count in cells.values()),
            "native-proxy generated budget exceeded")
    require(before == (p.tobytes(), s.tobytes(), eligible.tobytes()),
            "native-proxy generated inputs mutated")
    rows.append(dict(name=PARITY_NAMES[-1], passed=True, generated_only=True, point_count=len(p),
        selected_count=len(right[0]), maximum_selected_per_cell=max(cells.values(), default=0),
        eligible_sha256=hashlib.sha256(eligible.tobytes()).hexdigest(),
        selected_indices_sha256=hashlib.sha256(right[0].tobytes()).hexdigest(), coverage=right[1],
        proxy_size_wh=list(native_size), native_source_shape_hw=[3190, 4784],
        source_pixels_accessed=False, selection_and_coverage_only=True))
    return dict(passed=True, numpy_version=np.__version__, cases=rows,
                same_candidate_quota_in_both_paths=True, points_and_scores_unchanged=True,
                source_pixels_unchanged=True, exact_reference_comparison=True,
                score_precision_changed=False, max_features=384, max_features_per_cell=8,
                grid_rows=6, grid_cols=8)


def run_controls(candidate, motion_config, global_config, rows):
    from tiny_target.types import Frame, TimestampSource
    from tiny_target.motion import PvaMotionError, fit_global_motion
    for name, previous, current, expected in generated_controls():
        row = dict(name=name, passed=False, source_shape_hw=list(previous.shape), expected=list(expected),
                   previous_pixel_sha256=hashlib.sha256(previous.tobytes()).hexdigest(),
                   current_pixel_sha256=hashlib.sha256(current.tobytes()).hexdigest())
        rows.append(row)
        estimator = None
        try:
            row["pyramid_dimensions"] = validate_pyramid_dimensions(previous.shape, motion_config)
            estimator = candidate(motion_config)
            p = Frame(previous, 0, 0, "generated-control", 8, TimestampSource.CONTAINER_RATE)
            q = Frame(current, 100_000_000, 1, "generated-control", 8, TimestampSource.CONTAINER_RATE)
            try:
                correspondence = estimator.estimate(p, q)
            except PvaMotionError as exc:
                row.update(error=str(exc), pva_error=str(exc))
                require(name == "flat_static" and "PVA Harris returned zero features" in str(exc)
                        and not estimator.failed, "unexpected generated motion failure")
                row.update(passed=True, raw_harris_zero=True, expected_unavailable=True)
            else:
                require(name != "flat_static", "flat control unexpectedly produced tracked features")
                fit = fit_global_motion(correspondence, global_config)
                row.update(metrics=control_metrics(correspondence, fit, expected),
                           correspondence_metrics=correspondence.metrics, backends=correspondence.backends,
                           global_fit=fit.to_dict(include_inlier_indices=False))
                row["passed"] = row["metrics"]["passed"]
                require(row["passed"], "generated known-motion confidence gates failed")
        except BaseException as exc:
            row.update(passed=False, control_failure=repr(exc))
            row.setdefault("error", repr(exc))
            raise
        finally:
            if estimator is not None:
                try:
                    estimator.close()
                    row["closed"] = estimator.closed
                except BaseException as exc:
                    row.update(passed=False, cleanup_error=repr(exc))
                    raise
        require(row.get("closed") is True, "generated estimator did not close")
        print(json.dumps(dict(control=name, passed=row["passed"])), flush=True)


def configurations(helper):
    from tiny_target.config import load_config
    from tiny_target.motion import PvaMotionConfig, GlobalMotionConfig
    raw = load_config(helper.MOTION_CONFIG).raw
    motion = PvaMotionConfig.from_mapping(raw.get("motion"))
    global_config = GlobalMotionConfig.from_mapping(raw.get("global_motion"))
    candidate_config(motion)
    require(global_config.minimum_correspondences == global_config.minimum_inliers == 30,
            "global feature floor differs")
    return motion, global_config


def validate_preflight(pre, hashes, identities, workspace, clip):
    require(pre.get("schema") == SCHEMA + ".preflight" and pre.get("passed") is True
            and pre.get("workspace") == str(workspace) and pre.get("clip") == clip
            and pre.get("source") == source_spec(clip) and pre.get("candidate") == CANDIDATE
            and pre.get("input_sha256") == hashes and pre.get("probe_passed") is True,
            "missing/changed successful candidate preflight")
    require(pre.get("detector_run") is False and pre.get("conversion", {}).get("passed") is True
            and [r.get("name") for r in pre.get("controls", [])] == list(CONTROL_NAMES)
            and all(r.get("passed") is True and r.get("closed") is True for r in pre["controls"]),
            "generated control/conversion preflight incomplete")
    parity = pre.get("cpu_parity", {})
    require(parity.get("passed") is True and parity.get("same_candidate_quota_in_both_paths") is True
            and parity.get("exact_reference_comparison") is True and parity.get("source_pixels_unchanged") is True
            and parity.get("points_and_scores_unchanged") is True and parity.get("score_precision_changed") is False
            and [r.get("name") for r in parity.get("cases", [])] == list(PARITY_NAMES)
            and all(r.get("passed") is True and r.get("generated_only") is True for r in parity["cases"])
            and all(parity.get(key) == CANDIDATE[key] for key in
                    ("max_features", "max_features_per_cell", "grid_rows", "grid_cols"))
            and isinstance(parity.get("numpy_version"), str) and parity["numpy_version"],
            "generated CPU parity preflight incomplete")
    require(all(pre.get(k) == v for k, v in identities.items()), "candidate dependencies differ from preflight")


def preflight(workspace, clip):
    workspace = workspace_guard(workspace, clip, "preflight")
    directory = workspace / clip
    directory.mkdir(exist_ok=True)
    receipt = dict(schema=SCHEMA + ".preflight", passed=False, workspace=str(workspace), clip=clip,
                   source=source_spec(clip), candidate=dict(CANDIDATE), detector_run=False,
                   controls=[], full_pixel_predecode=False)
    try:
        baseline = load_baseline()
        reference, runtime, hashes = inputs(workspace, clip, baseline)
        helper = baseline.load_helper()
        modules, identities = helper.dependencies(reference)
        info = modules["profile_visible_interaction_v30"].runtime_info
        before, clocks = info(), helper.clock_policy_snapshot()
        helper.runtime_check(before, runtime)
        from tiny_target.frame_source import probe_video
        import cv2
        probe = probe_video(source_spec(clip)["path"])
        baseline.validate_probe(probe)
        motion, global_config = configurations(helper)
        receipt.update(input_sha256=hashes, runtime_before=before, clock_policy=clocks,
                       probe=probe.to_dict(), probe_passed=True, **identities)
        cv2.setNumThreads(2)  # Same process-local setting the frozen detector applies.
        receipt["cpu_parity"] = generated_cpu_parity(motion)
        receipt["conversion"] = conversion_check()
        receipt["feature_adapter"] = {}
        with candidate_adapter(modules["motion_reuse_v12"], motion, receipt["feature_adapter"]) as candidate:
            run_controls(candidate, motion, global_config, receipt["controls"])
        after = info()
        helper.runtime_check(after, runtime, after=True)
        require(helper.clock_policy_snapshot() == clocks, "clock controls changed in preflight")
        require(inputs(workspace, clip, baseline)[2] == hashes
                and helper.dependencies(reference)[1] == identities, "inputs/dependencies changed")
        receipt.update(passed=True, runtime_after=after, global_configuration=asdict(global_config),
                       clocks_changed=False, generated_pair_calls=3)
    except BaseException as exc:
        receipt["error"] = repr(exc)
        raise
    finally:
        write(directory / "preflight.json", receipt)
    return receipt


def run(workspace, clip):
    workspace = workspace_guard(workspace, clip, "run")
    directory = workspace / clip
    directory.mkdir(exist_ok=True)
    receipt = dict(schema=SCHEMA, passed=False, error=None, processed_frames=0,
        workspace=str(workspace), clip=clip, source=source_spec(clip), candidate=dict(CANDIDATE),
        algorithm_changed=True, feature_algorithm_changed=True, detector_configuration_changed=False,
        feature_quota_algorithm_changed=True, exact_cpu_execution_changed=True, harris_score_precision_changed=False,
        tracker_configuration_changed=False, global_motion_gates_changed=False, production_promotion=False,
        annotations_supplied_to_detector=False, raw16_accessed=False, sealed_holdouts_accessed=False,
        airborne_class_verified=False, clocks_changed=None, remote_clocks_unchanged=None,
        timestamp_basis="consumer frame index / nominal10Hz container rate; physical acquisition cadence unverified",
        interpretation="Candidate spatial feature quota and exact CPU execution only. Better readiness is not recall, registration truth or noise reduction.",
        launch_config_semantics="launch/report record unchanged file/package hashes; this receipt declares the in-memory Harris gain/capacity,384/8 point budget and exact CPU execution actually used.",
        timing_instrumentation="Native frame count/shape/order checks and small correspondence/backend/config checks; no real-pixel hashing or diagnostic stage readbacks. Current clocks, not a controlled v34 speed comparison.")
    try:
        baseline = load_baseline()
        reference, runtime, hashes = inputs(workspace, clip, baseline)
        helper = baseline.load_helper()
        modules, identities = helper.dependencies(reference)
        pre = read(directory / "preflight.json")
        validate_preflight(pre, hashes, identities, workspace, clip)
        info = modules["profile_visible_interaction_v30"].runtime_info
        before, clocks = info(), helper.clock_policy_snapshot()
        helper.runtime_check(before, runtime)
        require(clocks == pre.get("clock_policy"), "clock policy differs from preflight")
        motion, global_config = configurations(helper)
        receipt.update(input_sha256=hashes, preflight_sha256=sha(directory / "preflight.json"),
            runtime_before=before, clock_policy_before=clocks, feature_adapter={},
            global_configuration=asdict(global_config), tracking_transformed_sha256=helper.TRACKING_METHOD_SHA,
            **identities)
        with candidate_adapter(modules["motion_reuse_v12"], motion, receipt["feature_adapter"]) as candidate:
            changed_modules = dict(modules, motion_reuse_v12=SimpleNamespace(ReuseMotionV12=candidate))
            report = baseline.execute_baseline(helper, changed_modules, Path(source_spec(clip)["path"]),
                                               directory / "run", receipt)
        baseline.validate_output(report, read(directory / "run/launch.json"), reference, clip,
                                 receipt["decoded_frames_verified"])
        require(receipt["feature_adapter"]["estimator_instances"] == 1
                and receipt["feature_adapter"]["successful_pair_backend_checks"] ==
                sum(row["error"] is None for row in receipt["motion_attempts"]), "unverified candidate pair path")
        validate_adapter_roundtrip(receipt["feature_adapter"], pre["feature_adapter"])
        after, clock_after = info(), helper.clock_policy_snapshot()
        helper.runtime_check(after, runtime, after=True)
        receipt.update(runtime_after=after, clock_policy_after=clock_after,
                       clocks_changed=clock_after != clocks, remote_clocks_unchanged=clock_after == clocks)
        require(clock_after == clocks, "clock controls changed")
        require(inputs(workspace, clip, baseline)[2] == hashes
                and helper.dependencies(reference)[1] == identities, "frozen inputs/dependencies changed")
        receipt.update(passed=True, processed_frames=report["frames"],
            journal_sha256=sha(directory / "run/frames.jsonl"), report_sha256=sha(directory / "run/report.json"),
            launch_sha256=sha(directory / "run/launch.json"), availability=report["availability"],
            detection_status=report["detection_status"])
    except BaseException as exc:
        receipt["error"] = repr(exc)
        raise
    finally:
        write(directory / "execution_receipt.json", receipt)
    print(json.dumps(dict(clip=clip, passed=True, frames=receipt["processed_frames"],
                          candidate=CANDIDATE, detection_status=receipt["detection_status"])), flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--clip", choices=tuple(SOURCE_HASHES), required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--run", action="store_true")
    args = parser.parse_args()
    (preflight if args.preflight else run)(args.workspace, args.clip)


if __name__ == "__main__":
    main()
