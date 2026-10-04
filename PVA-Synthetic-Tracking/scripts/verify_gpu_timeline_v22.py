"""Independent saved-output/provenance audit plus Nsight timeline analysis."""
import argparse
import hashlib
from pathlib import Path
import statistics
import tarfile
from profile_visible_v17 import read, sha, write
from verify_visible_v20 import baseline
from summarize_visible_architecture_v21 import verify_trace
from nsys_timeline_v22 import analyze

ROOT = Path(__file__).resolve().parents[1]
V20 = ROOT/'results/tiny_target/visible_speed_v20_20260918/evidence'
ARCHIVE_SHA = 'cc520754500406219825259e51e1a4788e5b7a295122c5d665d3c9326a711c44'


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def verify(evidence):
    archive = evidence.parent/'completed_01.tgz'
    require(sha(archive) == ARCHIVE_SHA, 'Transferred archive changed')
    with tarfile.open(archive) as bundle:
        for member in bundle.getmembers():
            p = Path(member.name)
            require(member.isfile() and not p.is_absolute() and '..' not in p.parts, 'Unsafe archive member')
            require(hashlib.sha256(bundle.extractfile(member).read()).hexdigest() == sha(evidence/p),
                    'Extracted artifact differs: '+member.name)
    batch = read(evidence/'batch_01.json')
    require(batch['passed'] and batch['error'] is None and len(batch['rows']) == 6
            and not batch['raw16_accessed'] and not batch['defaults_changed'], 'Incomplete batch')
    require(set(batch['sources']) == {'batch_gpu_timeline_v22.py', 'profile_gpu_timeline_v22.py',
                                     'nvtx_bridge_v22.cpp'}, 'Missing harness sources')
    for name,digest in batch['sources'].items():
        require(digest == sha(evidence/name) == sha(ROOT/'scripts'/name), 'Harness changed: '+name)
    require(batch['bridge_sha256'] == sha(evidence/'libnvtx_bridge.so') and bool(batch['headers']), 'Bridge provenance missing')
    for row in batch['rows']:
        require(row['returncode'] == 0 and row['log_sha256'] == sha(evidence/row['log']), 'Execution/log changed')
    schedule = [('0126', 'clean'), ('0126', 'trace'), ('0082', 'trace'), ('0082', 'clean')]
    require([(r['clip'],r['mode']) for r in batch['rows'][2:]] == schedule, 'Run order changed')
    trials = {}
    for clip,mode in schedule:
        path = evidence/f'{mode}_{clip}'
        old,report,motion = baseline(path, clip, 128)
        receipt = read(path.with_suffix('.v20.json'))
        require(receipt['passed'] and receipt['error'] is None and receipt['clip'] == clip
                and receipt['mode'] == 'candidate' and receipt['frames'] == 128
                and receipt['processed_frames'] == 128, 'v20 execution failed')
        require(not any(receipt[k] for k in ('raw16_accessed','defaults_changed','gpu_changed',
                                           'noise_v18_enabled','median_v19_enabled')), 'Frozen scope changed')
        for key,name in (('script_sha256','run_visible_v20.py'), ('freeze_sha256','freeze.json'),
                         ('gate_sha256','generated_01.json'), ('library_sha256','build/libtracking_geometry_v20.so')):
            require(receipt[key] == sha(V20/name), 'v20 identity changed: '+key)
        require(receipt['transformed_sha256'] == read(V20/'generated_01.json')['transformed_sha256']
                and receipt['native_geometry_calls'] > 0 and receipt['geometry_fallbacks'] == 0,
                'Unexpected tracker execution')
        require(receipt['baseline_receipt_sha256'] == sha(path.with_suffix('.v17.json')), 'Broken receipt chain')
        row = next(r for r in batch['rows'][2:] if r['clip'] == clip and r['mode'] == mode)
        require(row['receipt_sha256'] == sha(path.with_suffix('.v20.json'))
                and row['fps'] == receipt['fps'] == report['processed_fps'], 'Receipt/timing mismatch')
        trials[path.name] = dict(exact=True, frames=128, fps=receipt['fps'], wall_s=receipt['wall_s'],
            service_mean_ms=statistics.mean(old['consumer_frame_ms']),
            stages_ms={k:v['mean'] for k,v in report['timings_ms'].items()},
            journal_sha256=sha(path/'frames.jsonl'))
        if mode == 'trace':
            nvtx = read(path.with_suffix('.nvtx.json'))
            host = read(path.with_suffix('.host_trace.json'))
            require(nvtx['passed'] and nvtx['error'] is None and nvtx['clip'] == clip
                    and nvtx['frames'] == 128 and not nvtx['raw16_accessed'] and not nvtx['defaults_changed']
                    and nvtx['script_sha256'] == sha(evidence/'profile_gpu_timeline_v22.py')
                    and nvtx['host_script_sha256'] == host['script_sha256'] == sha(ROOT/'scripts/profile_visible_architecture_v21.py')
                    and nvtx['bridge_sha256'] == batch['bridge_sha256']
                    and nvtx['host_receipt_sha256'] == sha(path.with_suffix('.host_trace.json')), 'NVTX provenance changed')
            require(host['passed'] and host['error'] is None and host['clip'] == clip
                    and host['frames'] == 128 and not host['raw16_accessed'] and not host['defaults_changed']
                    and host['v20_receipt_sha256'] == sha(path.with_suffix('.v20.json')), 'Host trace failed')
            verify_trace(host['events'])
            timeline = analyze(evidence/f'gpu_{clip}.sqlite')
            require(nvtx['push_pop_counts'] == [len(host['events'])]*2
                    and timeline['stage_range_count'] == len(host['events']), 'Missing NVTX annotations')
            trials[path.name]['timeline'] = timeline
    comparisons = {}
    for clip in ('0126','0082'):
        clean,trace = [trials[f'{m}_{clip}'] for m in ('clean','trace')]
        comparisons[clip] = dict(clean_fps=clean['fps'], traced_fps=trace['fps'],
            observed_elapsed_ratio=trace['wall_s']/clean['wall_s'],
            note='One control/trace comparison; opposite order across clips. Includes instrumentation '
                 'and run variation; not a causal profiler-overhead estimate or a speedup result.')
    return dict(verified=True, exact_frame_instances=512, unique_frames=256, raw16_accessed=False,
                defaults_changed=False, clean_speedup_claim=False, archive_sha256=ARCHIVE_SHA,
                analyzer_sha256=sha(ROOT/'scripts/nsys_timeline_v22.py'), verifier_sha256=sha(__file__),
                trials=trials, comparisons=comparisons,
                no_reported_cuda_loss=True, expected_per_frame_kernel_counts_passed=True,
                warning='No whole-device utilization, PVA hardware timeline, SM occupancy, bandwidth '
                        'counters, or system-wide CPU trace. Diagnostic CPU-backtrace permission and '
                        'auxiliary-process no-CUDA/NVTX warnings are retained, not hidden. '
                        'No new airborne accuracy or real-time claim.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.evidence)
    write(args.output, result)
    print(result['comparisons'])
