"""Verify saved v21 host traces and outputs; never opens source media."""
import argparse
from collections import defaultdict
from pathlib import Path
import statistics
from profile_visible_v17 import read, sha, write
from verify_visible_v20 import baseline

ROOT = Path(__file__).resolve().parents[1]
V20 = ROOT/'results/tiny_target/visible_speed_v20_20260918/evidence'


def verify_trace(rows):
    children = defaultdict(list)
    for i, row in enumerate(rows):
        if row['id'] != i or row['error'] is not None or row['duration_ns'] < 0:
            raise AssertionError('Invalid event or traced exception')
        if row['parent'] is not None:
            if not 0 <= row['parent'] < i:
                raise AssertionError('Invalid parent index')
            parent = rows[row['parent']]
            if parent['thread'] != row['thread']:
                raise AssertionError('Invalid parent')
            if not (parent['start_ns'] <= row['start_ns'] and
                    row['start_ns']+row['duration_ns'] <= parent['start_ns']+parent['duration_ns']):
                raise AssertionError('Child outside parent')
            children[row['parent']].append(row)
    for row in rows:
        child = children[row['id']]
        total = sum(c['duration_ns'] for c in child)
        if total != row['children_ns'] or row['exclusive_host_ns'] != row['duration_ns']-total or total > row['duration_ns']:
            raise AssertionError('Nested time accounting mismatch')
        for a, b in zip(child, child[1:]):
            if a['start_ns']+a['duration_ns'] > b['start_ns']:
                raise AssertionError('Overlapping children on same thread')
    for name in ('motion_and_warp', 'detector.total', 'tracks.total'):
        selected = [r for r in rows if r['name'] == name]
        if [r['frame'] for r in selected] != list(range(128)):
            raise AssertionError('Missing top-level frames: '+name)


def summarize(evidence):
    output = dict(verified=True, gpu_timeline_available=False, clean_speedup_claim=False,
                  raw16_accessed=False, defaults_changed=False, clips={})
    for clip in ('0126', '0082'):
        path = evidence/f'host_{clip}'
        v17, report, motion = baseline(path, clip, 128)
        trace = read(path.with_suffix('.host_trace.json'))
        v20 = read(path.with_suffix('.v20.json'))
        if not trace['passed'] or trace['error'] is not None or trace['clip'] != clip or trace['frames'] != 128:
            raise AssertionError('Trace incomplete')
        if trace['raw16_accessed'] or trace['defaults_changed'] or trace['script_sha256'] != sha(evidence/'profile_visible_architecture_v21.py'):
            raise AssertionError('Trace provenance/scope changed')
        if trace['script_sha256'] != sha(ROOT/'scripts/profile_visible_architecture_v21.py'):
            raise AssertionError('Local instrumentation differs')
        if not v20['passed'] or v20['error'] is not None or v20['mode'] != 'candidate' or v20['processed_frames'] != 128:
            raise AssertionError('Current v20 candidate missing')
        if v20['clip'] != clip or v20['frames'] != 128 or v20['native_geometry_calls'] <= 0 or v20['geometry_fallbacks'] != 0:
            raise AssertionError('Unexpected geometry execution')
        if any(v20[k] for k in ('raw16_accessed', 'defaults_changed', 'gpu_changed', 'noise_v18_enabled', 'median_v19_enabled')):
            raise AssertionError('Frozen scope changed')
        for key, filename in (('script_sha256', 'run_visible_v20.py'), ('freeze_sha256', 'freeze.json'),
                              ('gate_sha256', 'generated_01.json'), ('library_sha256', 'build/libtracking_geometry_v20.so')):
            if v20[key] != sha(V20/filename):
                raise AssertionError('Frozen v20 identity changed: '+key)
        if v20['transformed_sha256'] != read(V20/'generated_01.json')['transformed_sha256']:
            raise AssertionError('Unexpected geometry transformation')
        if trace['v20_receipt_sha256'] != sha(path.with_suffix('.v20.json')) or v20['baseline_receipt_sha256'] != sha(path.with_suffix('.v17.json')):
            raise AssertionError('Receipt hash mismatch')
        verify_trace(trace['events'])
        windows = {}
        for first in (0, 32):
            groups = defaultdict(list)
            for row in trace['events']:
                if row['frame'] is not None and first <= row['frame'] < 128:
                    groups[row['name']].append(row)
            windows[str(first)+'_127'] = {
                name:dict(calls=len(rows), **{
                    key.replace('_ns', '_ms_per_frame'):sum(r[key] for r in rows)/(128-first)/1e6
                    for key in ('duration_ns', 'exclusive_host_ns', 'thread_cpu_ns')})
                for name, rows in groups.items()}
        output['clips'][clip] = dict(exact=True, frames=128, windows=windows,
            instrumented_service_mean_ms=statistics.mean(v17['consumer_frame_ms']),
            journal_sha256=sha(path/'frames.jsonl'), trace_sha256=sha(path.with_suffix('.host_trace.json')),
            baseline_receipt_sha256=sha(path.with_suffix('.v20.json')))
    measured = read(ROOT/'results/tiny_target/visible_speed_v20_20260918/summary_verified_local.json')
    m = measured['full_stage_means_ms']
    total = measured['full_service']['mean_ms']
    other = total-m['motion_and_warp']-m['detection']-m['tracking']
    output['prior_clean_full_run'] = dict(frames=2741, service_mean_ms=total, stages_ms=m,
        fps=measured['full_candidate_fps'], other_including_decode_wait_ms=other)
    # Optimistic counterfactuals, not observed speedups or implementability claims.
    output['idealized_bounds_not_predictions'] = dict(
        delete_tracking_fps=1000/(total-m['tracking']),
        completely_hide_motion_warp_fps=1000/(total-m['motion_and_warp']),
        halve_all_three_compute_stages_fps=1000/((total-other)/2+other),
        provisional_target_fps=10, target_service_ms=100,
        required_service_reduction_fraction=1-100/total)
    output['warning'] = ('Host-only instrumented traces are not clean throughput measurements. '
        'CPU thread time excludes other workers; low caller CPU time does not prove GPU idle. '
        'Exclusive host intervals still contain unmeasured native work. No claimed GPU utilization '
        'or bandwidth. Idealized bounds ignore dependencies and contention.')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = summarize(args.evidence)
    write(args.output, result)
    print(result['idealized_bounds_not_predictions'])
