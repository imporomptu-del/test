"""Verify completed RAW16 execution evidence and keep suppression out of speed claims."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_raw16_efficiency import compare, load
from profile_raw16_efficiency import compact, semantic_report, sha, write_json


def audit_diagnostic(root, clip):
    paths = [root/f'audit_{clip}_{suffix}' for suffix in ('reference', 'masked')]
    observations = [load(p/'observation.json') for p in paths]
    reports = [load(p/'report.json') for p in paths]
    for base, observation in zip(paths, observations):
        if sha(base/'audit.jsonl') != observation['audit_sha256']:
            raise ValueError('Audit changed after generation')
    for key in ('source', 'package_sha256', 'script_sha256', 'frozen_sha256', 'requested_frames'):
        if observations[0][key] != observations[1][key]:
            raise ValueError('Audit pair scope/provenance differs')
    if observations[0]['audit_sha256'] != observations[1]['audit_sha256']:
        raise ValueError('Array/state audit differs')
    if semantic_report(reports[0]) != semantic_report(reports[1]):
        raise ValueError('Audit report differs')
    events = [json.loads(line) for line in (paths[0]/'audit.jsonl').open()]
    sources = [r['value'] for r in events if r['stage'] == 'source_frame']
    motions = [r['value']['metrics'] for r in events if r['stage'] == 'pva_motion']
    rejections = Counter(reason for r in events if r['stage'] == 'global_fit'
                         for reason in r['value']['rejection_reasons'])
    if [s['frame_index'] for s in sources] != list(range(64)) or len(motions) != 63:
        raise ValueError('Incomplete diagnostic sample')
    if any(s['image']['dtype'] != '<u2' or s['bit_depth'] != 16
           or s['image']['shape'] != [3190, 4784] for s in sources):
        raise ValueError('Unexpected RAW16 source contract')
    intervals = [(b['source_timestamp_ns']-a['source_timestamp_ns'])/1e9
                 for a, b in zip(sources, sources[1:])]
    if min(intervals) <= 0:
        raise ValueError('Source timestamps do not strictly increase')
    metrics = reports[0]['source']['pva_stabilization']['metrics']
    screened = reports[0]['screening']
    return dict(clip=clip, exact_recorded_state=True,
        complete_detector_execution=screened['synthetic_tracking']['window_count'] > 0,
        audited_unique_frames=64, pva=compact(metrics), rejection_reasons=dict(rejections),
        accepted_features_min=min(m['accepted_count'] for m in motions),
        accepted_features_max=max(m['accepted_count'] for m in motions),
        occupied_grid_fraction_min=min(m['grid_coverage']['fraction'] for m in motions),
        occupied_grid_fraction_max=max(m['grid_coverage']['fraction'] for m in motions),
        screened_frames=screened['frames_screened_after_background_warmup'],
        synthetic_windows=screened['synthetic_tracking']['window_count'],
        intermediate_candidate_count=screened['synthetic_tracking']['candidate_count_before_clip_pool'],
        recorded_timestamp_span_seconds=sum(intervals),
        recorded_interval_seconds=dict(minimum=min(intervals), median=statistics.median(intervals),
                                       maximum=max(intervals)),
        timestamp_warning='Host timestamps after pull/copy, not verified exposure timing',
        audit_sha256=observations[0]['audit_sha256'], real_object_accuracy_validated=False)


def summarize(root):
    audit = compare(root/'audit_0040_reference', root/'audit_0040_masked')
    diagnostics = {clip: audit_diagnostic(root, clip) for clip in ('0040', '0029')}
    suppressed = diagnostics['0029']
    if suppressed['synthetic_windows'] != 0 or suppressed['screened_frames'] != 0:
        raise ValueError('Unexpected suppressed-control state')
    try:
        compare(root/'audit_0029_reference', root/'audit_0029_masked')
    except ValueError as exc:
        if 'No synthetic windows' not in str(exc):
            raise
        suppressed['end_to_end_gate'] = 'rejected_no_synthetic_windows'
    else:
        raise ValueError('Suppressed work was admitted by the detector gate')
    repeats = root/'repeats_0040'
    saved = load(repeats/'summary.json')
    freeze = load(repeats/'freeze.json')
    if (saved['freeze_sha256'] != sha(repeats/'freeze.json')
            or saved['telemetry_sha256'] != sha(repeats/'tegrastats.log')
            or not (repeats/'tegrastats.log').stat().st_size
            or freeze['clip_order'] != ['0040'] or freeze['pairs_per_clip'] != 3
            or saved['total_timed_frames'] != 384 or saved['passed'] is not True):
        raise ValueError('Incomplete or changed repeat evidence')
    fresh = []
    for pair in range(3):
        ref = repeats/f'0040_pair{pair}_indexed_reference'
        cand = repeats/f'0040_pair{pair}_masked_ufunc'
        result = compare(ref, cand)
        for job, audit_name in ((ref, 'audit_0040_reference'), (cand, 'audit_0040_masked')):
            observation, prior = load(job/'observation.json'), load(root/audit_name/'observation.json')
            for key in ('package_sha256', 'script_sha256', 'source', 'frozen_sha256',
                        'execution_config_sha256', 'python', 'numpy'):
                if observation[key] != prior[key]:
                    raise ValueError('Timed/audited scope differs: '+key)
            if semantic_report(load(job/'report.json')) != semantic_report(load(root/audit_name/'report.json')):
                raise ValueError('Timed/audited output differs')
        result.update(pair=pair, execution_order=(['masked_ufunc', 'indexed_reference'] if pair%2
                                                else ['indexed_reference', 'masked_ufunc']))
        if result != saved['comparisons'][pair]:
            raise ValueError('Saved comparison differs from recomputation')
        fresh.append(result)
    baseline = load(root/'profile_0040_baseline/observation.json')
    elapsed_ms = baseline['performance']['elapsed_seconds']*1000
    stages = {name: dict(exclusive_seconds=metrics['exclusive_ms']/1000,
                        percent_of_instrumented_loop=100*metrics['exclusive_ms']/elapsed_ms)
              for name, metrics in baseline['instrumented_stages'].items() if name != 'finalize'}
    return dict(verified=True, algorithm_policy_changed=False, default_execution_changed=False,
        full_detector_audit=audit, diagnostic_samples=diagnostics,
        baseline_profile_stages=stages, profile_warning='Instrumented diagnostic costs, not benchmark FPS',
        speed=dict(clip='0040', frames_per_run=64, alternating_pairs=3,
            reference_median_fps=statistics.median(r['reference_fps'] for r in fresh),
            candidate_median_fps=statistics.median(r['candidate_fps'] for r in fresh),
            pair_throughput_ratios=[r['throughput_ratio'] for r in fresh],
            median_pair_throughput_ratio=statistics.median(r['throughput_ratio'] for r in fresh),
            all_pairs_improved=all(r['throughput_ratio']>1 for r in fresh),
            reference_weighted_fps=192/sum(64/r['reference_fps'] for r in fresh),
            candidate_weighted_fps=192/sum(64/r['candidate_fps'] for r in fresh)),
        real_object_accuracy_validated=False, live_camera_throughput_validated=False,
        coverage_warning='Legacy upper 1920/3190 rows, bright synthetic polarity, narrow motion grid',
        next_priority='RAW16 motion availability and full-coverage accuracy baseline before broader GPU acceleration',
        freeze_sha256=saved['freeze_sha256'], summary_script_sha256=sha(__file__))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Never overwrite evidence')
    result = summarize(args.root)
    write_json(args.output, result)
    print(json.dumps(result, indent=2))
