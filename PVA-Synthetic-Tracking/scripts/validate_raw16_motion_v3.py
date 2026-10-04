"""Gate a bounded development run on generated controls and a frozen config.

Reuses the Phase-v2 full-native 64-frame runner without changing the detector,
motion-fit acceptance thresholds, source allowlist, or injected-control layout.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
import validate_raw16_full_frame as validation
from profile_raw16_efficiency import sha, write_json

CANDIDATE = ROOT / 'configs/evaluation/raw16_motion_v3.json'
CANDIDATE_SHA256 = '5c35fd8239e105d0a4e3b76133adad1cfd7f11d7610fb89b9843387052f7a74b'
CONTROL_SEEDS = (75316, 129827, 85723)


def verify_candidate_config():
    baseline = json.loads(validation.MOTION.read_text())
    candidate = json.loads(CANDIDATE.read_text())
    if sha(validation.MOTION) != validation.FROZEN_HASHES[validation.MOTION] or sha(CANDIDATE) != CANDIDATE_SHA256:
        raise ValueError('Frozen motion configuration changed')
    candidate['motion'].pop('feature_intensity_mapping')
    candidate['motion'].pop('optical_flow_backend')
    if candidate != baseline:
        raise ValueError('Changes outside the two permitted motion options')


def verify_generated_controls(directory):
    evidence = []
    for seed in CONTROL_SEEDS:
        path = directory / f'final_controls_seed{seed}.json'
        report = json.loads(path.read_text())
        candidate = report['candidate']
        if not (candidate == 'raw_robust_u16_v1'
                and report['candidate_flow_backend'] == 'CUDA'
                and report['fixture_seed'] == seed
                and report['motion_config_sha256'] == validation.FROZEN_HASHES[validation.MOTION]
                and report['max_translation_error_px'] == .35
                and report['candidate_passed'] is True
                and len(report['cases']) == report['case_count'] == 12
                and all(row['modes'][candidate]['passed'] is True for row in report['cases'])):
            raise ValueError('Generated-scene controls did not pass unchanged')
        if sha(ROOT / 'scripts/check_raw16_motion_controls.py') != report['script_sha256']:
            raise ValueError('Generated-scene test implementation changed')
        for module in sorted((ROOT / 'tiny_target/motion').glob('*.py')):
            if sha(module) != report['runtime_sha256'].get(str(module.relative_to(ROOT))):
                raise ValueError(f'Motion runtime differs from tested version: {module.name}')
        evidence.append(dict(path=str(path), sha256=sha(path), seed=seed))
    return evidence


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    verify_candidate_config()
    evidence = verify_generated_controls(args.generated_controls)
    original_motion = validation.MOTION
    validation.MOTION = CANDIDATE
    validation.FROZEN_HASHES = {**validation.FROZEN_HASHES, CANDIDATE: CANDIDATE_SHA256}
    try:
        status = validation.run(args)
        write_json(args.output / 'motion_experiment.json', dict(
            schema_version='seaqr.raw16-motion-development.v3', generated_controls=evidence,
            candidate_sha256=sha(CANDIDATE), wrapper_sha256=sha(__file__),
            warning='Development availability only; no real-airborne accuracy or full-clip validation.'))
        return status
    finally:
        validation.MOTION = original_motion
        validation.FROZEN_HASHES.pop(CANDIDATE)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clip', choices=('0029', '0040'), required=True)
    parser.add_argument('--injected', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--generated-controls', type=Path, required=True)
    raise SystemExit(run(parser.parse_args()))
