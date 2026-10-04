"""Report-only checks for the bounded RAW16 motion experiment; no media reads."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from profile_raw16_efficiency import sha, write_json


def read(path):
    return json.loads(path.read_text())


def trial(root, name):
    folder = root / name
    report = read(folder / 'report.json')
    check = read(folder / 'checks.json')
    screen = report['screening']
    motion = report['source']['pva_stabilization']['metrics']
    synthetic = screen['synthetic_tracking']
    injection = report.get('injection')
    return dict(report_sha256=sha(folder/'report.json'),
        frames=screen['frames_seen'],
        applied_transforms=motion['accepted_transforms_applied'],
        reference_resets=motion['reference_resets'], runtime_failures=motion['pva_failures'],
        ready_filter_frames=screen['availability']['frames_with_valid_filter_support'],
        windows=synthetic['window_count'], candidates=synthetic['candidate_count_before_clip_pool'],
        retained_tracks=synthetic['qualified_track_pool_count'], preview_tracks=synthetic['shortlist_count'],
        checks=check['checks'], diagnostic_wall_s=check['elapsed_wall_s'],
        injected_control_matches=None if injection is None else injection['synthetic_track_pool_evaluation'],
        motion_pair_median_ms=motion.get('median_pva_total_ms'))


def summarize(current, previous):
    names = ('full_frame_0029', 'full_frame_0040', 'full_frame_0040_injected')
    results = {name: dict(previous=trial(previous, name), current=trial(current, name+'_v3')) for name in names}
    comparisons = {}
    for name in names:
        old, new = previous/name, current/(name+'_v3')
        a, b = read(old/'report.json'), read(new/'report.json')
        comparisons[name] = dict(
            identical_decoded_source_frames=read(old/'source_frames.json') == read(new/'source_frames.json'),
            identical_detector_configuration=a['configuration']['effective'] == b['configuration']['effective'],
            identical_full_native_geometry=a['source']['crop_xywh'] == b['source']['crop_xywh'] == [0, 0, 4784, 3190])
    plain = read(current/'full_frame_0040_v3/report.json')
    injected = read(current/'full_frame_0040_injected_v3/report.json')
    paired = dict(
        source_frames_identical=read(current/'full_frame_0040_v3/source_frames.json') == read(current/'full_frame_0040_injected_v3/source_frames.json'),
        motion_decisions_identical=plain['source']['pva_stabilization']['metrics']['motion_pairs'] == injected['source']['pva_stabilization']['metrics']['motion_pairs'])
    controls = []
    for seed in (75316, 129827, 85723):
        path = current/f'final_controls_seed{seed}.json'
        report = read(path)
        rows = [r['modes'][report['candidate']] for r in report['cases']]
        controls.append(dict(seed=seed, report_sha256=sha(path), cases=len(rows),
            passed=sum(row['passed'] for row in rows),
            max_accepted_translation_error_px=max(row.get('translation_error_px') or 0 for row in rows),
            runtime_failures=sum(row['runtime_failure'] for row in rows)))
    old_controls = results['full_frame_0040_injected']['previous']['injected_control_matches']['detected_target_ids']
    new_controls = results['full_frame_0040_injected']['current']['injected_control_matches']['detected_target_ids']
    gates = dict(
        generated_controls_all_pass=all(row['passed'] == row['cases'] == 12 and row['runtime_failures'] == 0 for row in controls),
        source_detector_geometry_unchanged=all(all(row.values()) for row in comparisons.values()),
        paired_injection_did_not_change_source_or_motion=all(paired.values()),
        all_trials_processing_integrity=all(v['current']['checks']['processing_integrity_passed'] for v in results.values()),
        all_trials_search_available=all(v['current']['checks']['detection_availability_passed'] for v in results.values()),
        previous_injected_control_matches_preserved=set(old_controls).issubset(new_controls),
        all_three_injected_controls_recovered=results['full_frame_0040_injected']['current']['checks']['synthetic_controls_passed'],
        real_airborne_accuracy_validated=False,
        speed_improvement_validated=False,
        production_ready=False)
    return dict(schema_version='seaqr.raw16-motion-summary.v3', trials=results,
        generated_controls=controls, paired_comparisons=comparisons, injected_pair=paired, gates=gates,
        warning='Only 64-frame development prefixes. Candidate/track counts are unlabeled review workload, not objects or false positives. Diagnostic elapsed times are not throughput benchmarks.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--current', type=Path, required=True)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.current, args.previous)
    write_json(args.output, result)
    print(json.dumps(result, indent=2))
