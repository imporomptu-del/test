"""Predeclared generated-scene PVA tests; this program cannot open real media.

All motion-fit thresholds are taken unchanged from phase20_motion_v8.json.
The candidate must recover every supported known shift and reject every
unobservable/unsupported case. Rejections are not counted as successful recall.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import platform
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from profile_raw16_efficiency import sha, write_json
from tiny_target.config import load_config
from tiny_target.motion import PvaMotionConfig, PvaMotionError, PvaPyrLkMotionEstimator, GlobalMotionConfig, fit_global_motion
from tiny_target.motion.pva_pyrlk import _raw_asinh_lut
from tiny_target.types import Frame, TimestampSource

MOTION = ROOT / 'configs/evaluation/phase20_motion_v8.json'
MOTION_SHA256 = 'fe450546af91f01a0fb090d76df3ba4db24081b0077c6a220d5194990fdda5b1'
SCENE_SEED = 75316
MAX_TRANSLATION_ERROR_PX = .35


def quantize(values):
    return np.clip(np.rint(values / 16.) * 16., 0, 65520).astype(np.uint16)


def texture(seed, *, bright=False):
    rng = np.random.default_rng(seed)
    image = np.full((960, 1280), 12000. if bright else 1600., np.float32)
    image += np.linspace(0, 1000 if bright else 250, 1280, dtype=np.float32)[None, :]
    for row in range(6):
        for col in range(8):
            for _ in range(10):
                x = int(rng.integers(col * 160 + 20, (col + 1) * 160 - 35))
                y = int(rng.integers(row * 160 + 20, (row + 1) * 160 - 35))
                width, height = rng.integers(10, 26, size=2)
                contrast = rng.uniform(8000, 14000) if bright else rng.uniform(450, 1400)
                image[y:y + height, x:x + width] += contrast
    image = cv2.GaussianBlur(image, (5, 5), .7)
    return quantize(image)


def translated(image, shift):
    matrix = np.array([[1., 0., shift[0]], [0., 1., shift[1]]], np.float32)
    return quantize(cv2.warpAffine(image.astype(np.float32), matrix, (1280, 960),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101))


def cases(seed=SCENE_SEED):
    dark = texture(seed)
    bright = texture(seed + 1, bright=True)
    for name, shift in (('dim_stationary', (0., 0.)), ('dim_subpixel', (.75, -.5)),
                        ('dim_translation', (4., -3.)), ('dim_larger_shift', (-11., 9.))):
        yield name, dark, translated(dark, shift), shift
    hdr = dark.copy()
    hdr[50:180, 1000:1200] = 65520
    yield 'high_dynamic_range', hdr, translated(hdr, (3.25, -1.75)), (3.25, -1.75)
    yield 'bright_translation', bright, translated(bright, (4., -3.)), (4., -3.)
    moved = translated(dark, (4., -3.))
    yield 'gain_and_offset_change', dark, quantize(moved.astype(np.float32) * 1.1 + 80), (4., -3.)
    rng = np.random.default_rng(seed + 2)
    fpn = rng.normal(0, 80, dark.shape)
    yield 'sensor_fixed_pattern', quantize(dark + fpn), quantize(moved + fpn), (4., -3.)
    patch = quantize(rng.uniform(6000, 18000, (128, 192)).astype(np.float32))
    previous, current = dark.copy(), moved.copy()
    previous[350:478, 450:642] = patch
    current[352:480, 442:634] = patch  # foreground disagrees with camera motion
    yield 'moving_foreground_patch', previous, current, (4., -3.)
    flat = np.full(dark.shape, 2048, np.uint16)
    yield 'unobservable_flat', flat, flat.copy(), None
    noise1 = quantize(rng.uniform(1000, 10000, dark.shape))
    noise2 = quantize(rng.uniform(1000, 10000, dark.shape))
    yield 'independent_noise', noise1, noise2, None
    matrix = cv2.getRotationMatrix2D((639.5, 479.5), 1.2, 1.)
    rotated = quantize(cv2.warpAffine(bright.astype(np.float32), matrix, (1280, 960),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101))
    yield 'unsupported_rotation', bright, rotated, None


def frame(image, index, name):
    return Frame(image, index * 333_333_333, index, 'generated-motion-v3:' + name,
                 16, TimestampSource.MANIFEST)


def evaluate_mode(estimator, fitter, previous, current, truth):
    try:
        pairs = estimator.estimate(previous, current)
    except PvaMotionError as exc:
        expected_unavailable = any(t in str(exc) for t in ('zero features', 'No finite in-bounds'))
        return dict(accepted=False, unavailable=str(exc), runtime_failure=not expected_unavailable,
                    passed=truth is None and expected_unavailable)
    fit = fit_global_motion(pairs, fitter)
    error = None
    if fit.accepted and truth is not None:
        error = float(np.linalg.norm(fit.previous_to_current_matrix[:2, 2] - truth))
    passed = (fit.accepted and error <= MAX_TRANSLATION_ERROR_PX) if truth is not None else not fit.accepted
    return dict(accepted=fit.accepted, passed=passed, runtime_failure=False,
                translation_error_px=error, correspondences=pairs.to_dict(), fit=fit.to_dict())


def main(output, candidate='raw_robust_u16_v1', flow_backend='CUDA', seed=SCENE_SEED):
    if output.exists():
        raise ValueError('Never overwrite validation evidence')
    if sha(MOTION) != MOTION_SHA256:
        raise ValueError('The common motion gates changed')
    cv2.setNumThreads(2)
    cfg = load_config(MOTION).raw
    base = PvaMotionConfig.from_mapping(cfg['motion'])
    estimators = {'bit_shift': PvaPyrLkMotionEstimator(base),
                  candidate: PvaPyrLkMotionEstimator(replace(base,
                      feature_intensity_mapping=candidate, optical_flow_backend=flow_backend))}
    fitter = GlobalMotionConfig.from_mapping(cfg['global_motion'])
    observations = []
    for name, previous_pixels, current_pixels, truth in cases(seed):
        previous, current = frame(previous_pixels, 0, name), frame(current_pixels, 1, name)
        hashes = [f.pixel_sha256() for f in (previous, current)]
        modes = {name: evaluate_mode(estimator, fitter, previous, current, truth)
                 for name, estimator in estimators.items()}
        if hashes != [f.pixel_sha256() for f in (previous, current)]:
            raise ValueError('Input evidence was mutated')
        observations.append(dict(name=name, expected_translation_xy_px=truth, raw_pixel_sha256=hashes, modes=modes))
        print(json.dumps(dict(case=name, expected=truth, results={k:
            {key: value for key, value in v.items() if key not in ('correspondences', 'fit')}
            for k, v in modes.items()})), flush=True)
    passed = all(row['modes'][candidate]['passed'] for row in observations)
    write_json(output, dict(schema_version='seaqr.raw16-motion-controls.v3',
        candidate=candidate, candidate_flow_backend=flow_backend,
        candidate_passed=passed, case_count=len(observations), cases=observations,
        motion_configuration=cfg, motion_config_sha256=sha(MOTION), script_sha256=sha(__file__),
        runtime_sha256={str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT/'tiny_target').rglob('*.py'))},
        raw_feature_lut_sha256=hashlib.sha256(_raw_asinh_lut(16).tobytes()).hexdigest(),
        fixture_seed=seed, max_translation_error_px=MAX_TRANSLATION_ERROR_PX,
        python=platform.python_version(), numpy=np.__version__, opencv=cv2.__version__,
        warning='Generated-scene correctness controls only. No media read, no real-airborne recall or generalization claim.'))
    return 0 if passed else 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--candidate', choices=('raw_asinh_v1', 'raw_linear_u16_v1', 'raw_robust_u16_v1'), default='raw_robust_u16_v1')
    parser.add_argument('--flow-backend', choices=('PVA', 'CUDA'), default='CUDA')
    parser.add_argument('--seed', type=int, default=SCENE_SEED)
    args = parser.parse_args()
    raise SystemExit(main(args.output, args.candidate, args.flow_backend, args.seed))
