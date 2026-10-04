"""Independent saved-journal and diagnostic trace audit of completed v30."""
import argparse
from pathlib import Path
from unittest.mock import patch

from profile_visible_interaction_v30 import read, sha, write
from analyze_visible_interaction_v30 import analyze, require, statistics
from verify_visible_combined_v29 import verify_sources, audit_trial, transformed_hash, ROOT
import verify_visible_combined_v29 as previous_audit

V29_LOCAL = ROOT/'results/tiny_target/visible_combined_v29_20260920/evidence'


def audit_relocated_trial(directory, spec, row, frozen, generated, geometry, tracking_hash):
    """Only v29's freeze lives in its original archive; all new outputs live here.

    Preserve the old verifier without rewriting its archived identity. Redirect
    this one dependency pathname, not its expected value or any comparison.
    The old verifier still checks the actual receipt against the v29 freeze.
    """
    original_sha = previous_audit.sha
    def relocated_sha(path):
        return original_sha(V29_LOCAL/'freeze.json' if Path(path) == directory/'freeze.json' else path)
    with patch.object(previous_audit, 'sha', relocated_sha):
        return audit_trial(directory, spec, row, frozen, generated, geometry, tracking_hash)


def verify(directory):
    manifest = read(directory/'export_manifest.json')
    require(manifest['post_run'] and not manifest['media_included'], 'Invalid export')
    for name, digest in manifest['files'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'Unsafe export path')
        require(sha(directory/name) == digest, 'Artifact changed '+name)
    frozen, generated, geometry = verify_sources(V29_LOCAL)
    tracking_hash = transformed_hash()
    freeze = read(directory/'freeze.json')
    require(freeze['pre_run'] and freeze['baseline_freeze_sha256'] == sha(V29_LOCAL/'freeze.json'), 'Changed baseline freeze')
    expected_sources = {'profile_visible_interaction_v30.py', 'batch_visible_interaction_v30.py',
        'test_visible_interaction_v30.py', 'visible_interaction_v30_plan.md'}
    require(set(freeze['sources']) == expected_sources, 'Incomplete diagnostic source freeze')
    for name, digest in freeze['sources'].items():
        sub = 'tests/unit' if name.startswith('test_') else 'docs' if name.endswith('.md') else 'scripts'
        require(sha(directory/name) == digest == sha(ROOT/sub/name), 'Changed diagnostic source '+name)
    require(freeze['unit_log_sha256'] == sha(directory/'unit.log') and '\nOK\n' in (directory/'unit.log').read_text(), 'Failed harness tests')
    require(freeze['telemetry_probe_sha256'] == sha(directory/'telemetry_probe.json'), 'Changed telemetry preflight')
    batch = read(directory/'batch.json')
    require(batch['passed'] and batch['error'] is None and not read(directory/'status.json')['running'], 'Unfinished batch')
    require(batch['freeze_sha256'] == sha(directory/'freeze.json'), 'Changed batch freeze')
    pairs = [('0082','v26'), ('0082','combined'), ('0082','combined'), ('0082','v26'), ('0126','v26'), ('0126','combined')]
    expected = [dict(name=f'{mode}_{i}_{c}_{a}', mode=mode, clip=c, arm=a)
        for mode in ('trace','clean') for i,(c,a) in enumerate(pairs)]
    require(batch['schedule'] == expected and len(batch['rows']) == len(expected), 'Incomplete or changed schedule')
    trials, traces, clean, latencies = {}, {}, {}, {}
    for spec, row in zip(expected, batch['rows']):
        require(all(row[k] == v for k,v in spec.items()), 'Wrong trial order')
        name = spec['name']
        require(row['log_sha256'] == sha(directory/(name+'.log')), 'Changed command log')
        is_trace = spec['mode'] == 'trace'
        command = row['command']
        remote = Path('/tmp/seaqr_visible_interaction_v30_retry_UpdrpJ')
        original = Path('/tmp/seaqr_visible_combined_v29_s8XhL1')
        script = remote/'profile_visible_interaction_v30.py' if is_trace else original/'run_visible_combined_v29.py'
        expected_command = ['/usr/bin/python3',str(script),'--clip',spec['clip'],'--arm',spec['arm'],'--output',str(remote/name)]
        if is_trace:
            expected_command = ['nsys','profile','--trace=cuda,nvtx,osrt','--sample=none',
                '--cpuctxsw=process-tree','--osrt-threshold=100000','--cuda-flush-interval=0',
                '--kill=none','--force-overwrite=false','--export=sqlite','--output='+str(remote/name)] + expected_command
        else:
            expected_command += ['--frames','128']
        require(command == expected_command, 'Trace/clean command changed')
        trial_spec = dict(name=name, clip=spec['clip'], arm=spec['arm'], frames=128, state_audit=False)
        trial, samples = audit_relocated_trial(directory, trial_spec, row, frozen, generated, geometry, tracking_hash)
        trial['traced'] = is_trace
        trials[name] = trial
        if is_trace:
            require(row['trace_sha256'] == sha(directory/(name+'.trace30.json'))
                    and row['sqlite_sha256'] == sha(directory/(name+'.sqlite')), 'Changed trace data')
            receipt = read(directory/(name+'.trace30.json'))
            require(receipt['script_sha256'] == freeze['sources']['profile_visible_interaction_v30.py']
                    and receipt['baseline_freeze_sha256'] == sha(V29_LOCAL/'freeze.json')
                    and receipt['bridge_sha256'] == freeze['bridge_sha256']
                    and receipt['receipt_sha256'] == row['receipt_sha256'], 'Trace provenance changed')
            traces[name] = analyze(directory/(name+'.sqlite'), directory/(name+'.trace30.json'))
        else:
            clean.setdefault(spec['clip'], {}).setdefault(spec['arm'], []).append(trial)
            latencies[name] = {k: statistics(samples[k]) for k in ('consumer_cadence','queue_aware','request_to_complete')}
    control = {c: {a: dict(runs=len(rows), pooled_fps=sum(r['count'] for r in rows)/sum(r['wall_s'] for r in rows),
        individual_fps=[r['fps'] for r in rows]) for a, rows in arms.items()} for c, arms in clean.items()}
    failed = read(directory/'failed_attempt/batch.json')
    require(not failed['passed'] and failed['error'] is not None and len(failed['rows']) == 1, 'Missing failed attempt')
    return dict(verified=True, completed=True, diagnostic_only=True, clean_controls=control,
        trials=trials, traces=traces, clean_latency=latencies, full_regression_run=False,
        frame_instances=1536, unique_development_frames=256, failed_attempt_preserved=failed,
        raw16_accessed=False, defaults_changed=False, verifier_sha256=sha(__file__),
        analyzer_sha256=sha(ROOT/'scripts/analyze_visible_interaction_v30.py'),
        manifest_sha256=sha(directory/'export_manifest.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    write(args.output, verify(args.evidence))
