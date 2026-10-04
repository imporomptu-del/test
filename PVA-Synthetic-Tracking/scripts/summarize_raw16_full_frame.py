"""Verify and summarize existing bounded RAW16 correctness reports; no media reads."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from profile_raw16_efficiency import write_json
from validate_raw16_full_frame import assess


def summarize(directory):
    trials = {}
    for name in ('full_frame_0040', 'full_frame_0040_injected', 'full_frame_0029'):
        folder = directory / name
        report = json.loads((folder / 'report.json').read_text())
        provenance = json.loads((folder / 'provenance.json').read_text())
        recorded = json.loads((folder / 'checks.json').read_text())
        frames = json.loads((folder / 'source_frames.json').read_text())
        checks = assess(report)
        if checks != recorded['checks']:
            raise ValueError(f'{name}: saved checks disagree with report')
        if len(frames) != 64 or [f['frame_index'] for f in frames] != list(range(64)):
            raise ValueError(f'{name}: incomplete source evidence')
        expected_clip = '0029' if name.endswith('0029') else '0040'
        if provenance['clip'] != expected_clip or provenance['injected'] != name.endswith('injected'):
            raise ValueError(f'{name}: wrong experiment identity')
        trials[name] = dict(report=report, provenance=provenance, frames=frames, checks=checks, run=recorded)
    base = trials['full_frame_0040']
    injected = trials['full_frame_0040_injected']
    for name, trial in trials.items():
        for key in ('frozen_sha256', 'package_sha256', 'script_sha256', 'helper_sha256'):
            if base['provenance'][key] != trial['provenance'][key]:
                raise ValueError(f'{name}: inconsistent {key}')
    same_source = base['frames'] == injected['frames'] and base['provenance']['source'] == injected['provenance']['source']
    same_motion = (base['report']['source']['pva_stabilization']['metrics']['motion_pairs'] ==
                   injected['report']['source']['pva_stabilization']['metrics']['motion_pairs'])
    injection = injected['report']['injection']
    expected = sorted(t['target_id'] for t in injection['specification']['targets'])
    pool_ids = sorted(injection['synthetic_track_pool_evaluation']['detected_target_ids'])
    shortlist_ids = sorted(injection['synthetic_track_shortlist_evaluation']['detected_target_ids'])
    control_checks = {
        'same_raw_frames_and_source_metadata': same_source,
        'same_motion_decisions': same_motion,
        'both_runs_have_supported_full_image_windows': all(t['checks']['detection_availability_passed'] for t in (base, injected)),
        'all_controls_in_track_pool': bool(expected) and pool_ids == expected,
        'all_controls_in_bounded_shortlist': bool(expected) and shortlist_ids == expected,
    }
    summaries = {}
    for name, trial in trials.items():
        report = trial['report']
        screen = report['screening']
        motion = report['source']['pva_stabilization']['metrics']
        summaries[name] = dict(checks=trial['checks'], frames=screen['frames_seen'],
            requested_crop_xywh=report['source']['crop_xywh'],
            frames_by_filter_state=screen['availability']['frames_by_filter_state'],
            synthetic_windows=screen['synthetic_tracking']['window_count'],
            synthetic_candidate_count=screen['synthetic_tracking']['candidate_count_before_clip_pool'],
            synthetic_qualified_track_pool=screen['synthetic_tracking']['qualified_track_pool_count'],
            bounded_shortlist_count=screen['synthetic_tracking']['shortlist_count'],
            accepted_transforms_applied=motion['accepted_transforms_applied'],
            reference_resets=motion['reference_resets'], pva_failures=motion['pva_failures'],
            source_elapsed_seconds=(trial['frames'][-1]['timestamp_ns'] - trial['frames'][0]['timestamp_ns']) / 1e9,
            instrumented_wall_seconds=trial['run']['elapsed_wall_s'])
    return dict(schema_version='seaqr.raw16-full-frame-validation.v2',
        processing_integrity_passed=all(t['checks']['processing_integrity_passed'] for t in trials.values()),
        full_frame_control_checks=control_checks, full_frame_controls_passed=all(control_checks.values()),
        all_development_sources_detection_available=all(t['checks']['detection_availability_passed'] for t in trials.values()),
        real_target_accuracy_validated=False, summaries=summaries,
        control_track_pool_evaluation=injection['synthetic_track_pool_evaluation'],
        control_bounded_shortlist_evaluation=injection['synthetic_track_shortlist_evaluation'],
        control_probe_summary={k: {key: value for key, value in v.items() if key != 'probes'}
                               for k, v in injection['synthetic_tracking_truth_probes'].items()},
        warning='Positive controls are injected after stabilization and validate only the bounded bright/slow-motion downstream screen. '
                'Unlabeled candidates are review workload, not real airborne objects or false alarms. '
                'Unavailable motion must not be interpreted as a successful negative search; timings include diagnostics.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    summary = summarize(args.directory)
    write_json(args.output, summary)
    print(json.dumps(summary, indent=2))
