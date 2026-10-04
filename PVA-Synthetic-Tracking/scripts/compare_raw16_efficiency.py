"""Fail-closed comparison of two bounded RAW16 development executions."""
from __future__ import annotations

import argparse
from collections import Counter
from itertools import zip_longest
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from profile_raw16_efficiency import ALLOWED_CLIPS, compact, semantic_report, sha, write_json


def load(path):
    return json.loads(Path(path).read_text())


def compare(reference, candidate):
    roots = [Path(reference), Path(candidate)]
    obs = [load(p/'observation.json') for p in roots]
    reports = [load(p/'report.json') for p in roots]
    if not all(o['passed'] is True for o in obs):
        raise ValueError('Incomplete or failed run')
    for key in ('mode', 'clip', 'requested_frames', 'source', 'frozen_sha256',
                'package_sha256', 'script_sha256', 'python', 'numpy'):
        if obs[0][key] != obs[1][key]:
            raise ValueError('Execution scope/provenance differs: ' + key)
    if obs[0]['clip'] not in ALLOWED_CLIPS:
        raise ValueError('Unexpected development clip')
    if [o['background_execution'] for o in obs] != ['indexed_reference', 'masked_ufunc']:
        raise ValueError('Expected explicit reference/candidate execution pair')
    for base, observation, report in zip(roots, obs, reports):
        if sha(base/'execution_config.json') != observation['execution_config_sha256']:
            raise ValueError('Execution config hash mismatch')
        if report['configuration']['identity']['sha256'] != observation['execution_config_sha256']:
            raise ValueError('Report configuration does not match launch')
        if report['configuration']['effective']['background_execution'] != observation['background_execution']:
            raise ValueError('Report execution mode does not match launch')
        if report['screening']['frames_seen'] != observation['requested_frames']:
            raise ValueError('Frame count mismatch')
        if report['source']['pva_stabilization']['metrics']['pva_failures']:
            raise ValueError('PVA execution errors')
        if report['screening']['synthetic_tracking']['window_count'] <= 0:
            raise ValueError('No synthetic windows: cannot validate downstream execution')
        if observation['mode'] == 'audit' and sha(base/'audit.jsonl') != observation['audit_sha256']:
            raise ValueError('Audit content hash mismatch')
    if semantic_report(reports[0]) != semantic_report(reports[1]):
        raise ValueError('Non-timing report output differs')
    counts = Counter()
    if obs[0]['mode'] == 'audit':
        with (roots[0]/'audit.jsonl').open() as left, (roots[1]/'audit.jsonl').open() as right:
            for line, (a, b) in enumerate(zip_longest(left, right), 1):
                if a is None or b is None or json.loads(a) != json.loads(b):
                    raise ValueError(f'Intermediate evidence differs at event {line}')
                counts[json.loads(a)['stage']] += 1
        n = obs[0]['requested_frames']
        for stage in ('source_frame', 'full_resolution_warp', 'crop', 'background_and_filter'):
            if counts[stage] != n:
                raise ValueError('Incomplete frame-array audit: ' + stage)
        for stage in ('pva_motion', 'global_fit'):
            if counts[stage] != n - 1:
                raise ValueError('Incomplete motion audit: ' + stage)
        windows = reports[0]['screening']['synthetic_tracking']['window_count']
        for stage in ('cuda_shift_stack', 'candidate_ranking', 'candidate_extract', 'synthetic_association'):
            if counts[stage] != windows:
                raise ValueError('Incomplete window audit: ' + stage)
        if sum(counts.values()) != obs[0]['audit_events'] or obs[0]['audit_events'] != obs[1]['audit_events']:
            raise ValueError('Audit event count mismatch')
    result = dict(passed=True, clip=obs[0]['clip'], frames=obs[0]['requested_frames'],
        mode=obs[0]['mode'], exact_array_audit=obs[0]['mode'] == 'audit',
        audit_events=dict(counts), exact_non_timing_report=True,
        reference_report_sha256=sha(roots[0]/'report.json'),
        candidate_report_sha256=sha(roots[1]/'report.json'),
        reference_observation_sha256=sha(roots[0]/'observation.json'),
        candidate_observation_sha256=sha(roots[1]/'observation.json'),
        pva=compact(reports[0]['source']['pva_stabilization']['metrics']),
        screened_frames=reports[0]['screening']['frames_screened_after_background_warmup'],
        synthetic_windows=reports[0]['screening']['synthetic_tracking']['window_count'],
        candidate_count=reports[0]['screening']['synthetic_tracking']['candidate_count_before_clip_pool'],
        real_object_accuracy_validated=False)
    if obs[0]['mode'] == 'timed':
        perf = [r['performance'] for r in reports]
        result.update(reference_fps=perf[0]['processed_frames_per_second'],
                      candidate_fps=perf[1]['processed_frames_per_second'],
                      throughput_ratio=perf[1]['processed_frames_per_second']/perf[0]['processed_frames_per_second'])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Never overwrite evidence')
    result = compare(args.reference, args.candidate)
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
