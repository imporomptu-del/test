"""Generated controls for the complete-grid Harris capacity correction.

Frozen v3 controls plus native-size dense/noisy scenes. No real media access.
Report both v3 and v4; a failed v4 check cannot be hidden by the aggregate.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from check_raw16_motion_controls import cases, evaluate_mode, frame, quantize
from profile_raw16_efficiency import sha, write_json
from tiny_target.config import load_config
from tiny_target.motion import PvaMotionConfig, PvaPyrLkMotionEstimator, GlobalMotionConfig

CONFIG = ROOT / 'configs/evaluation/raw16_motion_v4.json'
SEEDS = (75316, 129827, 85723)


def dense_cases(seed):
    """Different fixture family, full RAW geometry, known physical shifts."""
    rng = np.random.default_rng(seed + 40000)
    smooth = cv2.resize(rng.random((200, 299), dtype=np.float32), (4784, 3190),
                        interpolation=cv2.INTER_CUBIC)
    scene = np.clip(10000. + smooth * 36000., 4096., 60000.)
    # Spatially heterogeneous noise. Both samples contain independent noise;
    # this deliberately prevents a sensor-fixed pattern from providing truth.
    sigma = np.full(scene.shape, 35., np.float32)
    sigma[:1595, :2392] = 500.
    previous = quantize(scene + rng.standard_normal(scene.shape, dtype=np.float32) * sigma)
    for name, shift in (('dense_native_subpixel', (.75, -.5)), ('dense_native_translation', (4., -3.))):
        moved = cv2.warpAffine(scene, np.float32([[1., 0., shift[0]], [0., 1., shift[1]]]),
                              (4784, 3190), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)
        current = quantize(moved + rng.standard_normal(scene.shape, dtype=np.float32) * sigma)
        yield name, previous, current, shift
    # Same spatial statistics without any shared physical scene.
    independent = cv2.resize(rng.random((200, 299), dtype=np.float32), (4784, 3190),
                             interpolation=cv2.INTER_CUBIC)
    yield 'dense_native_independent_scene', previous, quantize(10000. + independent * 36000.), None


def run(output, seed):
    if output.exists():
        raise FileExistsError(output)
    cv2.setNumThreads(2)
    cfg = load_config(CONFIG).raw
    candidate = PvaMotionConfig.from_mapping(cfg['motion'])
    estimators = {
        'v3': PvaPyrLkMotionEstimator(replace(candidate, harris_capacity_policy='legacy_default')),
        'v4': PvaPyrLkMotionEstimator(candidate),
    }
    fitter = GlobalMotionConfig.from_mapping(cfg['global_motion'])
    rows = []
    from itertools import chain
    for name, before, after, truth in chain(cases(seed), dense_cases(seed)):
        previous, current = frame(before, 0, name), frame(after, 1, name)
        hashes = [f.pixel_sha256() for f in (previous, current)]
        modes = {key: evaluate_mode(estimator, fitter, previous, current, truth)
                 for key, estimator in estimators.items()}
        if hashes != [f.pixel_sha256() for f in (previous, current)]:
            raise ValueError('Source images were mutated')
        full_grid = None
        if name.startswith('dense_native_') and truth is not None:
            c = modes['v4'].get('correspondences', {})
            full_grid = c.get('metrics', {}).get('grid_coverage', {}).get('occupied_cells') == 48
        passed = modes['v4']['passed'] and full_grid is not False
        rows.append(dict(name=name, expected_translation_xy_px=truth, pixel_sha256=hashes,
            modes=modes, full_grid_supported=full_grid, passed=passed))
        print(name, {k: (v['passed'], v.get('translation_error_px'), v.get('unavailable')) for k, v in modes.items()},
              'full_grid', full_grid, flush=True)
    passed = all(r['passed'] for r in rows)
    write_json(output, dict(schema_version='seaqr.raw16-motion-generated.v4', seed=seed,
        passed=passed, cases=rows, config=cfg, config_sha256=sha(CONFIG),
        max_translation_error_px=.35, script_sha256=sha(__file__),
        fixture_helper_sha256=sha(ROOT/'scripts/check_raw16_motion_controls.py'),
        runtime_sha256={str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'tiny_target').rglob('*.py'))},
        warning='Generated development controls, not real airborne accuracy or speed evidence.'))
    return 0 if passed else 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, choices=SEEDS, required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.output, args.seed))
