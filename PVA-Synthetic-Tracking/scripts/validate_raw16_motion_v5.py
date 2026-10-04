"""Frozen, generated-control-gated 64-frame RAW motion status experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
import validate_raw16_full_frame as validation
import validate_raw16_motion_v4 as previous
from profile_raw16_efficiency import sha, write_json
from tiny_target.config import load_config

CANDIDATE = ROOT/'configs/evaluation/raw16_motion_v5.json'
CANDIDATE_SHA256 = '718c76a1924b1e9ff80ae92b433bddf2cf6791c9585bb83612ab3f6d15fa3f45'
SEEDS = previous.SEEDS


def verify_candidate_config():
    previous.verify_candidate_config()
    candidate = json.loads(CANDIDATE.read_text())
    if sha(CANDIDATE) != CANDIDATE_SHA256 or candidate['motion'].pop('flow_status_policy') != 'fresh_per_pair':
        raise ValueError('Frozen status policy changed')
    if candidate != json.loads(previous.CANDIDATE.read_text()):
        raise ValueError('Changes outside the per-pair status policy')


def verify_report(report, seed):
    if not (report['seed'] == seed and report['passed'] is True
            and report['config_sha256'] == CANDIDATE_SHA256
            and report['config'] == load_config(CANDIDATE).raw
            and report['max_translation_error_px'] == .35):
        raise ValueError('Generated-control header invalid')
    rows = report['cases']
    if len(rows) != 16 or {r['name'] for r in rows} != previous.POSITIVE | previous.NEGATIVE | {'reseed_after_lost_tracks'}:
        raise ValueError('Cases missing or duplicated')
    for row in rows:
        if row['passed'] is not True:
            raise ValueError('Generated case failed')
        if row['name'] == 'reseed_after_lost_tracks':
            if not (row['identical_points'] is True
                    and row['first']['correspondences']['correspondences'] == row['recovered']['correspondences']['correspondences']
                    and row['unrelated']['accepted'] is False
                    and all(row[k]['passed'] is True and row[k]['runtime_failure'] is False
                            for k in ('first', 'unrelated', 'recovered'))
                    and all(row[k]['accepted'] is True and 0 <= row[k]['translation_error_px'] <= .35
                            for k in ('first', 'recovered'))):
                raise ValueError('Reseeding did not recover exact valid correspondences')
            continue
        v = row['result']
        if not (v['passed'] is True and v['runtime_failure'] is False):
            raise ValueError('Case result failed')
        if row['name'] in previous.POSITIVE:
            if not (v['accepted'] is True and 0 <= v['translation_error_px'] <= .35):
                raise ValueError('Known-motion error exceeds limit')
            if row['name'].startswith('dense_native_') and not (row['full_grid_supported'] is True
                    and v['correspondences']['metrics']['grid_coverage']['occupied_cells'] == 48):
                raise ValueError('Full-image generated coverage failed')
        elif v['accepted'] is not False:
            raise ValueError('Negative control accepted')


def verify_generated_controls(directory):
    evidence = []
    for seed in SEEDS:
        path = directory/f'status_controls_seed{seed}.json'
        report = json.loads(path.read_text())
        verify_report(report, seed)
        if report['script_sha256'] != sha(ROOT/'scripts/check_raw16_motion_v5.py'):
            raise ValueError('Generated test changed')
        for name in ('scripts/check_raw16_motion_controls.py', 'scripts/check_raw16_motion_v4.py'):
            if report['helper_sha256'].get(name) != sha(ROOT/name):
                raise ValueError('Generated fixture changed')
        for p in sorted((ROOT/'tiny_target/motion').glob('*.py')):
            if sha(p) != report['runtime_sha256'].get(str(p.relative_to(ROOT))):
                raise ValueError(f'Untested motion runtime: {p.name}')
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
            schema_version='seaqr.raw16-motion-development.v5', generated_controls=evidence,
            candidate_sha256=sha(CANDIDATE), wrapper_sha256=sha(__file__),
            warning='Development prefixes only. Not real airborne accuracy or a speed benchmark.'))
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
