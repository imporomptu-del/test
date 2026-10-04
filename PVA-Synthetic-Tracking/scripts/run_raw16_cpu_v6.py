"""Frozen exact-CPU optimization checks on existing bounded development inputs."""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
import check_raw16_motion_v5 as generated
import validate_raw16_motion_v5 as v5
import validate_raw16_full_frame as full
import trace_raw16_status_v5 as sequence
from check_motion_cpu_parity import BASELINE_SHA
from profile_raw16_efficiency import sha, write_json
from tiny_target.config import load_config

CONFIG = ROOT/'configs/evaluation/raw16_motion_v6.json'
CONFIG_SHA = '7aa18c9c2d1e2bc2ee4463498a9936c4d2601d15fc0b070176b40eb59d50891f'


@contextmanager
def profile_motion(rows):
    estimator_type = sequence.PvaPyrLkMotionEstimator
    original = estimator_type.estimate

    def measured(estimator, previous, current):
        started = time.perf_counter()
        result = original(estimator, previous, current)
        elapsed = (time.perf_counter() - started) * 1000
        rows.append(dict(frame_index=current.frame_index, wall_ms=elapsed,
            timings_ms=result.timings_ms, identity=sequence.identity(result),
            accepted=result.count))
        return result

    with patch.object(estimator_type, 'estimate', measured):
        yield


def verify_config():
    v5.verify_candidate_config()
    if sha(CONFIG) != CONFIG_SHA:
        raise ValueError('Frozen v6 configuration changed')
    cfg = json.loads(CONFIG.read_text())
    if cfg['motion'].pop('feature_cpu_policy') != 'batched_exact_v1':
        raise ValueError('Unexpected CPU policy')
    if cfg != json.loads(v5.CANDIDATE.read_text()):
        raise ValueError('Only exact CPU execution may change')


def verify_cpu_report(report):
    if not (report['passed'] is True and report['baseline_sha256'] == BASELINE_SHA
            and len(report['cases']) == 5
            and {r['name'] for r in report['cases']} == {
                'native_raw16', 'native_raw16_masked', 'odd_u8',
                'odd_raw12_radius8', 'large_radius_fallback'}):
        raise ValueError('CPU parity gate missing or failed')
    for row in report['cases']:
        outputs = row['outputs']
        if not (row['exact'] is True
                and outputs['frozen_v5'] == outputs['current_reference'] == outputs['batched']):
            raise ValueError('CPU parity is not exact')
        observed = outputs['frozen_v5']
        if not (set(observed) == {'eligible', 'reasons', 'selected', 'coverage'}
                and observed['eligible']['shape'] == [row['points']]
                and row['points'] > 0
                and len(observed['eligible']['sha256']) == 64
                and len(observed['selected']['sha256']) == 64):
            raise ValueError('CPU output identity is missing')


def verify_cpu(directory):
    path = directory/'cpu_parity.json'
    report = json.loads(path.read_text())
    verify_cpu_report(report)
    if report['script_sha256'] != sha(ROOT/'scripts/check_motion_cpu_parity.py'):
        raise ValueError('CPU parity harness changed')
    for p in sorted((ROOT/'tiny_target/motion').glob('*.py')):
        if report['runtime_sha256'].get(str(p.relative_to(ROOT))) != sha(p):
            raise ValueError('CPU runtime changed after parity check')
    return dict(path=str(path), sha256=sha(path))


def verify_generated(directory):
    verify_config()
    cpu = verify_cpu(directory)
    with patch.object(v5, 'CANDIDATE', CONFIG), patch.object(v5, 'CANDIDATE_SHA256', CONFIG_SHA):
        controls = v5.verify_generated_controls(directory)
    return dict(cpu=cpu, generated=controls)


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    verify_config()
    if args.action == 'generated':
        verify_cpu(args.evidence)
        if args.seed not in v5.SEEDS:
            raise ValueError('A frozen generated seed is required')
        with patch.object(generated, 'CONFIG', CONFIG):
            return generated.run(args.output, args.seed)
    evidence = verify_generated(args.evidence)
    if args.clip not in ('0029', '0040'):
        raise ValueError('Only the existing development allowlist may be used')
    if args.action == 'sequence':
        if args.reference or args.injected or args.clip != '0040':
            raise ValueError('Sequence check uses the unchanged native candidate only')
        baseline = ROOT/'results/tiny_target/raw16_motion_v5_20260915/sequence_0040_v5.json'
        if sha(baseline) != 'a57cbf9aef65b94af0b6bb289e89d82151854b35245d6b051f30538eaaa576a8':
            raise ValueError('Frozen sequential reference changed')
        # Reuse the exact v5 sequence/order test, changing only its config and
        # evidence gate. It reads at most 64 frames from the explicit source.
        profile = []
        with ExitStack() as stack:
            stack.enter_context(profile_motion(profile))
            stack.enter_context(patch.object(sequence, 'CANDIDATE', CONFIG))
            stack.enter_context(patch.object(sequence, 'verify_candidate_config', verify_config))
            stack.enter_context(patch.object(sequence, 'verify_generated_controls', verify_generated))
            args.generated_controls = args.evidence
            status = sequence.run(args)
        current = json.loads(args.output.read_text())
        previous = json.loads(baseline.read_text())
        exact = (current['frames'] == previous['frames']
                 and [r['identity'] for r in current['pairs']] == [r['identity'] for r in previous['pairs']])
        write_json(args.output.with_suffix('.parity.json'), dict(
            passed=status == 0 and exact, exact_v5_points_and_source=exact,
            baseline_sha256=sha(baseline), sequence_sha256=sha(args.output), evidence=evidence))
        write_json(args.output.with_suffix('.timings.json'), profile)
        return 0 if status == 0 and exact else 2
    motion = v5.CANDIDATE if args.reference else CONFIG
    motion_sha = v5.CANDIDATE_SHA256 if args.reference else CONFIG_SHA
    profile = []
    with profile_motion(profile), patch.object(full, 'MOTION', motion), patch.object(full, 'FROZEN_HASHES', {
            **full.FROZEN_HASHES, motion: motion_sha}):
        status = full.run(args)
    write_json(args.output/'motion_profile.json', profile)
    write_json(args.output/'cpu_experiment.json', dict(
        schema_version='seaqr.raw16-cpu-development.v6', evidence=evidence,
        reference=args.reference, configuration_sha256=motion_sha,
        wrapper_sha256=sha(__file__),
        warning='Same 64-frame development prefixes, instrumented comparison only. '
                'No new accuracy evidence, mask tuning, cache policy change or deployment.'))
    return status


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action', choices=('generated', 'sequence', 'full'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--clip', choices=('0029', '0040'))
    parser.add_argument('--reference', action='store_true')
    parser.add_argument('--injected', action='store_true')
    raise SystemExit(run(parser.parse_args()))
