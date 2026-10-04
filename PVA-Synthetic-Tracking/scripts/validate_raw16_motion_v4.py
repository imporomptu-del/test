"""Bounded, generated-control-gated full-native validation of capacity v4.

Only the existing 64-frame 0029/0040 experiments are authorized here. Detector,
motion acceptance gates, source allowlist and injected-control layout are frozen.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import validate_raw16_full_frame as validation
from profile_raw16_efficiency import sha, write_json
from tiny_target.config import load_config

CANDIDATE = ROOT / 'configs/evaluation/raw16_motion_v4.json'
CANDIDATE_SHA256 = '6986713c8243349fed9714970d750dcdaa390abb70e41deb6077a99b9d410eb4'
SEEDS = (75316, 129827, 85723)
POSITIVE = {
    'dim_stationary', 'dim_subpixel', 'dim_translation', 'dim_larger_shift',
    'high_dynamic_range', 'bright_translation', 'gain_and_offset_change',
    'sensor_fixed_pattern', 'moving_foreground_patch',
    'dense_native_subpixel', 'dense_native_translation',
}
NEGATIVE = {'unobservable_flat', 'independent_noise', 'unsupported_rotation', 'dense_native_independent_scene'}


def verify_candidate_config():
    base = json.loads(validation.MOTION.read_text())
    candidate = json.loads(CANDIDATE.read_text())
    if sha(validation.MOTION) != validation.FROZEN_HASHES[validation.MOTION] or sha(CANDIDATE) != CANDIDATE_SHA256:
        raise ValueError('Frozen experiment configuration changed')
    for key, expected in {'feature_intensity_mapping': 'raw_robust_u16_v1',
                          'optical_flow_backend': 'CUDA', 'harris_capacity_policy': 'complete_grid'}.items():
        if candidate['motion'].pop(key) != expected:
            raise ValueError('Unexpected motion option')
    if candidate != base:
        raise ValueError('Unapproved change to frozen thresholds or settings')


def verify_report(report, seed):
    if not (report['seed'] == seed and report['passed'] is True
            and report['config_sha256'] == CANDIDATE_SHA256
            and report['config'] == load_config(CANDIDATE).raw
            and report['max_translation_error_px'] == .35):
        raise ValueError('Generated-control header invalid')
    rows = report['cases']
    if len(rows) != 15 or {r['name'] for r in rows} != POSITIVE | NEGATIVE:
        raise ValueError('Generated-control cases missing or duplicated')
    for row in rows:
        v = row['modes']['v4']
        if not (row['passed'] is True and v['passed'] is True and v['runtime_failure'] is False):
            raise ValueError('Generated-control case failed')
        if row['name'] in POSITIVE:
            if not (v['accepted'] is True and 0 <= v['translation_error_px'] <= .35):
                raise ValueError('Known-motion error exceeds frozen limit')
            if row['name'].startswith('dense_native_'):
                if not (row['full_grid_supported'] is True
                        and v['correspondences']['metrics']['grid_coverage']['occupied_cells'] == 48):
                    raise ValueError('Full-frame generated coverage failed')
        elif v['accepted'] is not False:
            raise ValueError('Negative control was accepted')


def verify_generated_controls(directory):
    evidence = []
    for seed in SEEDS:
        path = directory / f'capacity_controls_seed{seed}.json'
        report = json.loads(path.read_text())
        verify_report(report, seed)
        if report['script_sha256'] != sha(ROOT/'scripts/check_raw16_motion_v4.py'):
            raise ValueError('Generated-control implementation changed')
        if report['fixture_helper_sha256'] != sha(ROOT/'scripts/check_raw16_motion_controls.py'):
            raise ValueError('Generated-control fixture changed')
        for module in sorted((ROOT/'tiny_target/motion').glob('*.py')):
            if sha(module) != report['runtime_sha256'].get(str(module.relative_to(ROOT))):
                raise ValueError(f'Motion runtime is not the tested version: {module.name}')
        evidence.append(dict(seed=seed, path=str(path), sha256=sha(path)))
    return evidence


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    verify_candidate_config()
    evidence = verify_generated_controls(args.generated_controls)
    old_motion, old_hashes = validation.MOTION, validation.FROZEN_HASHES
    validation.MOTION = CANDIDATE
    validation.FROZEN_HASHES = {**old_hashes, CANDIDATE: CANDIDATE_SHA256}
    try:
        status = validation.run(args)
        write_json(args.output/'motion_experiment.json', dict(
            schema_version='seaqr.raw16-motion-development.v4', generated_controls=evidence,
            candidate_sha256=sha(CANDIDATE), wrapper_sha256=sha(__file__),
            warning='Unlabeled development availability only, not real-airborne recall or full-clip validation.'))
        return status
    finally:
        validation.MOTION, validation.FROZEN_HASHES = old_motion, old_hashes


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clip', choices=('0029', '0040'), required=True)
    parser.add_argument('--injected', action='store_true')
    parser.add_argument('--generated-controls', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    raise SystemExit(run(parser.parse_args()))
