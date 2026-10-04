"""Independently check frozen decode-overlap repeats, ownership and exact output."""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tiny_target.visible_baseline import VisibleConfig, sha256
from compare_phase20_exact_runs import validate_decode, shape_accelerator, transition_for_pair, EXECUTION_FIELDS
from repeat_phase20_kernel_speed import check_prefix
from finalize_phase20_host_speed import verify_package


def check_config(before, after):
    configs = [asdict(VisibleConfig(**cfg)) for cfg in (before, after)]
    if configs[0]['frame_decode_execution'] != 'sequential' or configs[1]['frame_decode_execution'] != 'prefetch_one':
        raise ValueError('Expected explicit sequential/prefetch comparison')
    for key in configs[0]:
        if key not in EXECUTION_FIELDS and configs[0][key] != configs[1][key]:
            raise ValueError('Decode experiment changed policy: ' + key)
    if configs[0]['native_shape_library_sha256'] != configs[1]['native_shape_library_sha256']:
        raise ValueError('Decode experiment changed native binary')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'reference', 'full-summary', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    candidate = json.loads((args.root / 'freeze.json').read_text())
    previous = json.loads((args.reference / 'freeze.json').read_text())
    if candidate['compiled_library_sha256'] != previous['compiled_library_sha256']:
        raise ValueError('Decode experiment changed GPU binary')
    check_config(*[json.loads((p / 'config.json').read_text()) for p in (args.reference, args.root)])
    for base, frozen in ((args.reference, previous), (args.root, candidate)):
        if sha256(base / 'config.json') != frozen['config_sha256']:
            raise ValueError('Config hash changed')
        for job in frozen['jobs']:
            verify_package(base, frozen, job)
    full = json.loads(args.full_summary.read_text())
    if (full['freeze_sha256'] != sha256(args.root / 'freeze.json')
            or not full['passed_execution_equivalence'] or full['full_clip_frames'] != 2741
            or full['timing_reference']['freeze_sha256'] != sha256(args.reference / 'freeze.json')
            or full['production_ready'] is not False):
        raise ValueError('Missing complete four-clip equivalence result')
    repeats = args.root / 'repeats'
    summary = json.loads((repeats / 'repeat_summary.json').read_text())
    if (summary['passed'] is not True or summary['prefix_frames'] != 128
            or summary['alternating_pairs_per_clip'] != 3 or summary['clip_ids'] != ['0126','0082']
            or summary['candidate_freeze_sha256'] != sha256(args.root / 'freeze.json')
            or summary['reference_freeze_sha256'] != sha256(args.reference / 'freeze.json')
            or summary['script_sha256'] != candidate['files_sha256']['scripts/repeat_phase20_kernel_speed.py']
            or not (repeats / 'tegrastats.log').stat().st_size):
        raise ValueError('Repeat scope/provenance changed')
    expected = [(cid, pair) for cid in ('0126','0082') for pair in range(3)]
    if [(r['clip_id'], r['pair']) for r in summary['comparisons']] != expected:
        raise ValueError('Missing, duplicate or reordered repeat pair')
    package = {n.removeprefix('tiny_target/'): d for n, d in candidate['files_sha256'].items()
        if n.startswith('tiny_target/')}
    results = []
    for row in summary['comparisons']:
        cid, pair = row['clip_id'], row['pair']
        if row['order'] != (['before','after'] if pair % 2 == 0 else ['after','before']):
            raise ValueError('Repeat order changed')
        samples = {}
        for label, frozen in (('before',previous), ('after',candidate)):
            path = repeats / f'{cid}_pair{pair}_{label}'
            report = json.loads((path / 'report.json').read_text())
            launch = json.loads((path / 'launch.json').read_text())
            check_prefix(args.reference / ('pva_' + cid), path, 128)
            if (launch['package_sha256'] != package or launch['config_sha256'] != frozen['config_sha256']
                    or launch['source_sha256'] != frozen['sources'][cid]['sha256']
                    or launch['fps'] != 10 or launch['max_frames'] != 128
                    or report['completed'] is not True or report['full_clip'] is not False
                    or report['frames'] != 128 or report['source_sha256'] != launch['source_sha256']
                    or report['configuration'] != launch['configuration']):
                raise ValueError('Repeat launch/report provenance changed')
            config_base = args.reference if label == 'before' else args.root
            expected_cfg = asdict(VisibleConfig(**json.loads((config_base / 'config.json').read_text())))
            if launch['configuration'] != expected_cfg:
                raise ValueError('Repeat runtime configuration changed')
            for name, digest in package.items():
                if sha256(path / 'implementation' / name) != digest:
                    raise ValueError('Repeat implementation snapshot changed')
            shape_accelerator(launch)
            previous_launch = json.loads((args.reference / ('pva_' + cid) / 'launch.json').read_text())
            transition_for_pair(previous_launch, launch, None)
            stats = validate_decode(launch, report)
            if stats['read_calls'] != 128:
                raise ValueError('Prefix decoded beyond frame bound')
            if (not math.isfinite(report['elapsed_seconds']) or report['elapsed_seconds'] <= 0
                    or report['processed_fps'] != 128 / report['elapsed_seconds']
                    or row[label]['fps'] != report['processed_fps']
                    or row[label]['timings_ms'] != report['timings_ms']
                    or row[label]['frame_decode'] != stats
                    or row[label]['journal_sha256'] != sha256(path / 'frames.jsonl')):
                raise ValueError('Repeat timing/ownership evidence changed')
            samples[label] = dict(fps=report['processed_fps'], timings_ms=report['timings_ms'],
                frame_decode=stats, journal_sha256=sha256(path / 'frames.jsonl'))
        ratio = samples['after']['fps'] / samples['before']['fps']
        if ratio != row['speedup']:
            raise ValueError('Repeat speedup changed')
        results.append(dict(clip_id=cid, pair=pair, speedup=ratio, **samples))
    aggregate = {cid: dict(
        before_fps_median=statistics.median(r['before']['fps'] for r in results if r['clip_id'] == cid),
        after_fps_median=statistics.median(r['after']['fps'] for r in results if r['clip_id'] == cid),
        paired_speedups=[r['speedup'] for r in results if r['clip_id'] == cid]) for cid in ('0126','0082')}
    if aggregate != summary['summary']:
        raise ValueError('Repeat aggregate changed')
    result = dict(passed=True, repeated_frames_compared=1536, full_clip_frames=2741,
        unchanged_gpu_library_sha256=candidate['compiled_library_sha256'],
        repeated_summary=aggregate, comparisons=results,
        sequential_and_prefetch_use_same_current_package=True,
        production_ready=False, holdouts_accessed=False, labels_used_during_processing=False,
        caveat='Paired prefixes on natural dynamic clocks. No repeated full-clip distribution or accuracy gain claim.',
        full_summary_sha256=sha256(args.full_summary), repeat_summary_sha256=sha256(repeats / 'repeat_summary.json'),
        freeze_sha256=sha256(args.root / 'freeze.json'), verifier_sha256=sha256(__file__))
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps({k:v for k,v in result.items() if k != 'comparisons'}, indent=2))


if __name__ == '__main__':
    main()
