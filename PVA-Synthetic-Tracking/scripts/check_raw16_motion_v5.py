"""Generated known-motion, rejection and reseeding controls; no real media."""
from __future__ import annotations

import argparse
from itertools import chain
from pathlib import Path
import sys

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
from check_raw16_motion_controls import cases, evaluate_mode, frame
from check_raw16_motion_v4 import dense_cases, SEEDS
from profile_raw16_efficiency import sha, write_json
from tiny_target.config import load_config
from tiny_target.motion import PvaMotionConfig, PvaPyrLkMotionEstimator, GlobalMotionConfig

CONFIG = ROOT/'configs/evaluation/raw16_motion_v5.json'


def run(output, seed):
    if output.exists():
        raise FileExistsError(output)
    cv2.setNumThreads(2)
    cfg = load_config(CONFIG).raw
    estimator = PvaPyrLkMotionEstimator(PvaMotionConfig.from_mapping(cfg['motion']))
    fitter = GlobalMotionConfig.from_mapping(cfg['global_motion'])
    rows = []

    def evaluate(name, previous, current, truth):
        a, b = frame(previous, 0, name), frame(current, 1, name)
        hashes = [f.pixel_sha256() for f in (a, b)]
        value = evaluate_mode(estimator, fitter, a, b, truth)
        if hashes != [f.pixel_sha256() for f in (a, b)]:
            raise ValueError('Original source pixels changed')
        return value, hashes

    for name, a, b, truth in chain(cases(seed), dense_cases(seed)):
        value, hashes = evaluate(name, a, b, truth)
        full_grid = None
        if name.startswith('dense_native_') and truth is not None:
            full_grid = value.get('correspondences', {}).get('metrics', {}).get('grid_coverage', {}).get('occupied_cells') == 48
        passed = value['passed'] and full_grid is not False
        rows.append(dict(name=name, result=value, pixel_sha256=hashes,
            expected_translation_xy_px=truth, full_grid_supported=full_grid, passed=passed))
        print(name, passed, value.get('translation_error_px'), value.get('unavailable'), flush=True)
    clean = next(c for c in cases(seed + 91) if c[0] == 'dim_subpixel')
    unrelated = next(c for c in cases(seed + 97) if c[0] == 'independent_noise')
    first, _ = evaluate(*clean)
    rejected, _ = evaluate(*unrelated)
    again, _ = evaluate(*clean)
    identical = first.get('correspondences', {}).get('correspondences') == again.get('correspondences', {}).get('correspondences')
    passed = first['passed'] and rejected['passed'] and again['passed'] and identical
    rows.append(dict(name='reseed_after_lost_tracks', passed=passed,
        first=first, unrelated=rejected, recovered=again, identical_points=identical))
    print('reseed_after_lost_tracks', passed, 'identical points', identical, flush=True)
    passed = all(r['passed'] for r in rows)
    write_json(output, dict(schema_version='seaqr.raw16-motion-generated.v5', seed=seed,
        passed=passed, cases=rows, config=cfg, config_sha256=sha(CONFIG),
        max_translation_error_px=.35, script_sha256=sha(__file__),
        helper_sha256={p:sha(ROOT/p) for p in ('scripts/check_raw16_motion_controls.py', 'scripts/check_raw16_motion_v4.py')},
        runtime_sha256={str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'tiny_target').rglob('*.py'))},
        warning='Generated development controls only. Fresh status applies before tracking new Harris points, never after tracking failures.'))
    return 0 if passed else 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, choices=SEEDS, required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.output, args.seed))
