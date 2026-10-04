"""Explicit, single-worker experimental adapters; no persistent default changes."""
from __future__ import annotations

import ast
from contextlib import ExitStack, contextmanager
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import tarfile
import types
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
from profile_raw16_efficiency import sha, compact
from profile_raw16_v6 import RUNTIME_SHA
from tiny_target import dense_screen as dense
from tiny_target.motion import global_motion as gm, pva_pyrlk as pva


def filter_parameters():
    cfg, _ = dense.load_dense_screen_config(ROOT/'configs/evaluation/raw16_background_v7.json')
    screen = dense.DensePointScreener(replace(cfg, synthetic_tracking_enabled=False))
    return screen._point_kernel, screen._point_kernel_l2


def cpu_filter(image, kernel, normalizer):
    cv = dense._load_cv2()
    out = cv.filter2D(image, cv.CV_32F, kernel, borderType=cv.BORDER_CONSTANT)
    out /= normalizer
    return out


def numeric_comparison(reference, candidate, image):
    if reference.shape != candidate.shape or reference.dtype != candidate.dtype:
        raise ValueError('Comparison contract mismatch')
    delta = reference.astype(np.float64) - candidate.astype(np.float64)
    bound = 2e-5 * max(1., float(np.max(np.abs(image))))
    maximum = float(np.max(np.abs(delta)))
    return dict(bit_exact=reference.tobytes() == candidate.tobytes(),
                different_pixels=int(np.count_nonzero(reference != candidate)),
                pixels=reference.size, max_abs_error=maximum,
                rms_error=float(np.sqrt(np.mean(delta*delta))), diagnostic_bound=bound,
                numerical_screen_passed=bool(np.isfinite(candidate).all() and maximum <= bound),
                threshold4_changes=int(np.count_nonzero((reference >= 4) != (candidate >= 4))))


class FeaturePixelsCache:
    """One immutable Frame identity, never a frame-number/content heuristic.

    Prepared buffers are private to this adapter and the synchronous estimator.
    This relies on the existing Frame immutable-array contract, not on timestamps.
    """
    def __init__(self, original):
        self.original = original
        self.frame = self.mapping = self.pixels = None
        self.hits = self.misses = 0

    def __call__(self, frame, mapping):
        if frame is self.frame and mapping == self.mapping:
            self.hits += 1
            return self.pixels
        pixels = self.original(frame, mapping)
        self.frame, self.mapping, self.pixels = frame, mapping, pixels
        self.misses += 1
        return pixels

    def clear(self):
        self.frame = self.mapping = self.pixels = None


@contextmanager
def cpu_motion_execution():
    cache = FeaturePixelsCache(pva._feature_pixels)
    original = gm.fit_global_motion
    def fit(correspondences, config=None):
        return original(correspondences, config, execution='translation_batched_exact_v1')
    with ExitStack() as stack:
        stack.enter_context(patch.object(pva, '_feature_pixels', cache))
        stack.enter_context(patch.object(dense, 'fit_global_motion', fit))
        # Generated PVA controls hold a separate imported function reference.
        import check_raw16_motion_controls as controls
        stack.enter_context(patch.object(controls, 'fit_global_motion', fit))
        try:
            yield cache
        finally:
            cache.clear()


def reference_ast_unchanged(source, current):
    """Verify the reference fitter has only the declared opt-in dispatch edits."""
    old, new = ast.parse(source), ast.parse(current)
    a = next(n for n in old.body if isinstance(n, ast.FunctionDef) and n.name == 'fit_global_motion')
    b = next(n for n in new.body if isinstance(n, ast.FunctionDef) and n.name == 'fit_global_motion')
    if [arg.arg for arg in b.args.kwonlyargs] != ['execution']:
        raise ValueError('Unexpected new fitter signature')
    if ast.literal_eval(b.args.kw_defaults[0]) != 'reference':
        raise ValueError('Default fitter execution changed')
    b.args.kwonlyargs = []
    b.args.kw_defaults = []
    expected = [
        'if execution not in {"reference", "translation_batched_exact_v1"}:\n    raise ValueError("Unknown global-motion execution policy")',
        'if execution == "translation_batched_exact_v1" and resolved.model != "translation":\n    raise ValueError("Batched translation execution cannot fit a similarity model")',
        'if execution == "translation_batched_exact_v1":\n    best_matrix, best_mask, best_score = _batched_translation_samples(previous, current, samples, resolved.ransac_reprojection_px)',
    ]
    for text in expected:
        target = ast.dump(ast.parse(text).body[0])
        matches = [n for n in b.body if ast.dump(n) == target]
        if len(matches) != 1:
            raise ValueError('Unknown reference dispatch edit')
        b.body.remove(matches[0])
    loop = next(n for n in b.body if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'sample')
    expected_iter = ast.parse('samples if execution == "reference" else ()', mode='eval').body
    if ast.dump(loop.iter) != ast.dump(expected_iter):
        raise ValueError('Unknown sample dispatch')
    loop.iter = ast.Name(id='samples', ctx=ast.Load())
    if ast.dump(a) != ast.dump(b):
        raise ValueError('Frozen reference fitting changed')
    # Every original top-level statement/function other than this fitter remains.
    new.body = [n for n in new.body if not (isinstance(n, ast.FunctionDef) and n.name == '_batched_translation_samples')]
    if ast.dump(old) != ast.dump(new):
        raise ValueError('Undeclared global-motion change')


def frozen_motion_module(archive):
    if sha(archive) != RUNTIME_SHA:
        raise ValueError('Frozen v6 archive changed')
    with tarfile.open(archive) as tar:
        code = tar.extractfile('tiny_target/motion/global_motion.py').read().decode()
    reference_ast_unchanged(code, (ROOT/'tiny_target/motion/global_motion.py').read_text())
    name = 'tiny_target.motion._v8_frozen_reference'
    mod = types.ModuleType(name)
    mod.__package__ = 'tiny_target.motion'
    sys.modules[name] = mod
    exec(compile(code, str(archive)+':global_motion', 'exec'), mod.__dict__)
    return mod


def estimate_identity(result):
    return compact(dict(report=result.to_dict(), mask=result.inlier_mask, residuals=result.residuals_px))


def package_identity():
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT/'tiny_target').rglob('*.py'))
            if not p.name.startswith('._')}
