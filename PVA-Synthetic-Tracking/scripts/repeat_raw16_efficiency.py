"""One-worker, alternating, bounded RAW16 repeats after exact array audits."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from profile_raw16_efficiency import ROOT, sha, semantic_report, write_json
from compare_raw16_efficiency import compare, load


def run_child(command, log):
    """Terminate only this trial's process group on timeout/interruption."""
    with log.open('x') as handle:
        process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT,
                                   cwd=ROOT, start_new_session=True)
        try:
            if process.wait(timeout=600):
                raise RuntimeError(f'RAW16 child failed; see {log}')
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=15)
            raise


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    audits = {}
    # 0029's audited prefix resets on all 63 motion pairs and runs zero windows.
    # It is diagnostic evidence, NOT a speed sample. Do not weaken the comparator
    # to count suppressed work as successful detector throughput.
    for clip in ('0040',):
        ref = args.audit_root/f'audit_{clip}_reference'
        cand = args.audit_root/f'audit_{clip}_masked'
        checked = compare(ref, cand)
        if not checked['exact_array_audit'] or checked['frames'] != 64:
            raise ValueError('Requires a complete 64-frame array audit with synthetic windows')
        audits[clip] = dict(comparison=checked, reference=ref, candidate=cand)
    freeze = dict(started_at_utc=datetime.now(timezone.utc).isoformat(), frames=64,
        pairs_per_clip=3, clip_order=['0040'], alternating_order=['AB', 'BA', 'AB'],
        excluded_from_timing={'0029': 'All 63 global-motion fits rejected; zero synthetic windows'},
        scripts_sha256={name: sha(ROOT/'scripts'/name) for name in (
            'profile_raw16_efficiency.py', 'compare_raw16_efficiency.py', 'repeat_raw16_efficiency.py')},
        audits={clip: a['comparison'] for clip, a in audits.items()})
    write_json(args.output/'freeze.json', freeze)
    results = []
    with (ROOT/'raw16_trial.lock').open('a') as lock, (args.output/'tegrastats.log').open('x') as telemetry:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        monitor = subprocess.Popen(['/usr/bin/tegrastats', '--interval', '1000'],
            stdout=telemetry, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            for clip in freeze['clip_order']:
                for pair in range(3):
                    order = ['indexed_reference', 'masked_ufunc']
                    if pair % 2:
                        order.reverse()
                    outputs = {}
                    for execution in order:
                        name = f'{clip}_pair{pair}_{execution}'
                        output = args.output/name
                        print(json.dumps(dict(starting=name)), flush=True)
                        run_child([sys.executable, str(ROOT/'scripts/profile_raw16_efficiency.py'),
                            '--clip', clip, '--frames', '64', '--mode', 'timed',
                            '--background-execution', execution, '--output', str(output)],
                            args.output/(name+'.log'))
                        audit_key = 'reference' if execution == 'indexed_reference' else 'candidate'
                        audit = audits[clip][audit_key]
                        observation, prior = load(output/'observation.json'), load(audit/'observation.json')
                        for key in ('package_sha256', 'script_sha256', 'source', 'frozen_sha256',
                                    'execution_config_sha256', 'python', 'numpy'):
                            if observation[key] != prior[key]:
                                raise ValueError('Timed run differs from exact-audit provenance: '+key)
                        if semantic_report(load(output/'report.json')) != semantic_report(load(audit/'report.json')):
                            raise ValueError('Timed output differs from exact-audited output')
                        outputs[execution] = output
                    result = compare(outputs['indexed_reference'], outputs['masked_ufunc'])
                    result.update(pair=pair, execution_order=order)
                    write_json(args.output/f'{clip}_pair{pair}_comparison.json', result)
                    results.append(result)
                    print(json.dumps(result), flush=True)
        finally:
            monitor.terminate()
            try:
                monitor.wait(timeout=5)
            except subprocess.TimeoutExpired:
                monitor.kill()
                monitor.wait(timeout=5)
    summary = dict(passed=True, real_object_accuracy_validated=False, frames_per_run=64,
        total_timed_frames=len(results)*2*64, comparisons=results, per_clip={},
        freeze_sha256=sha(args.output/'freeze.json'), telemetry_sha256=sha(args.output/'tegrastats.log'),
        warning='One short prefix of one development clip, including startup/decoder teardown. '
                'Not full-clip or live-camera throughput, and not full-frame detection coverage.')
    for clip in freeze['clip_order']:
        subset = [r for r in results if r['clip'] == clip]
        summary['per_clip'][clip] = dict(
            reference_median_fps=statistics.median(r['reference_fps'] for r in subset),
            candidate_median_fps=statistics.median(r['candidate_fps'] for r in subset),
            median_pair_ratio=statistics.median(r['throughput_ratio'] for r in subset),
            minimum_pair_ratio=min(r['throughput_ratio'] for r in subset),
            maximum_pair_ratio=max(r['throughput_ratio'] for r in subset))
    write_json(args.output/'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.audit_root = args.audit_root.resolve()
    args.output = args.output.resolve()
    run(args)
