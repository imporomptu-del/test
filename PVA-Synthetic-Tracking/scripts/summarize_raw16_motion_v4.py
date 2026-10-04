"""Report-only capacity-correction comparisons; never opens camera media."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from profile_raw16_efficiency import sha, write_json
from summarize_raw16_motion_v3 import read, trial


def summarize(current, v2, v3):
    names = ('full_frame_0029', 'full_frame_0040', 'full_frame_0040_injected')
    trials = {n: {'v2': trial(v2, n), 'v3': trial(v3, n+'_v3'), 'v4': trial(current, n+'_v4')} for n in names}
    comparisons = {}
    for name in names:
        now = current/(name+'_v4')
        report = read(now/'report.json')
        for label, old in (('v2', v2/name), ('v3', v3/(name+'_v3'))):
            prior = read(old/'report.json')
            comparisons[name+'_'+label] = dict(
                identical_decoded_source_frames=read(now/'source_frames.json') == read(old/'source_frames.json'),
                identical_detector_configuration=report['configuration']['effective'] == prior['configuration']['effective'],
                identical_full_native_geometry=report['source']['crop_xywh'] == prior['source']['crop_xywh'] == [0, 0, 4784, 3190])
    plain, injected = (current/'full_frame_0040_v4', current/'full_frame_0040_injected_v4')
    paired = dict(
        source_frames_identical=read(plain/'source_frames.json') == read(injected/'source_frames.json'),
        motion_decisions_identical=read(plain/'report.json')['source']['pva_stabilization']['metrics']['motion_pairs'] ==
                                  read(injected/'report.json')['source']['pva_stabilization']['metrics']['motion_pairs'])
    generated = []
    for seed in (75316, 129827, 85723):
        path = current/f'capacity_controls_seed{seed}.json'
        report = read(path)
        generated.append(dict(seed=seed, sha256=sha(path), passed=sum(r['passed'] for r in report['cases']),
            cases=len(report['cases']), runtime_failures=sum(r['modes']['v4']['runtime_failure'] for r in report['cases']),
            worst_translation_error_px=max(r['modes']['v4'].get('translation_error_px') or 0 for r in report['cases'])))
    previous_controls = trials['full_frame_0040_injected']['v2']['injected_control_matches']['detected_target_ids']
    new_controls = trials['full_frame_0040_injected']['v4']['injected_control_matches']['detected_target_ids']
    gates = dict(
        generated_controls_all_pass=all(r['cases'] == r['passed'] == 15 and r['runtime_failures'] == 0 for r in generated),
        source_detector_geometry_unchanged=all(all(r.values()) for r in comparisons.values()),
        injection_did_not_change_source_or_motion=all(paired.values()),
        processing_integrity=all(r['v4']['checks']['processing_integrity_passed'] for r in trials.values()),
        all_samples_search_available=all(r['v4']['checks']['detection_availability_passed'] for r in trials.values()),
        motion_and_search_availability_nonregression=all(
            r['v4'][key] >= max(r['v2'][key], r['v3'][key])
            for r in trials.values() for key in ('applied_transforms', 'ready_filter_frames', 'windows')),
        previous_injected_matches_preserved=set(previous_controls).issubset(new_controls),
        all_three_injected_controls_recovered=trials['full_frame_0040_injected']['v4']['checks']['synthetic_controls_passed'],
        real_airborne_accuracy_validated=False, speed_improvement_validated=False,
        default_configuration_changed=False, production_ready=False)
    return dict(schema_version='seaqr.raw16-motion-summary.v4', trials=trials,
        generated_controls=generated, paired_comparisons=comparisons, injected_pair=paired, gates=gates,
        warning='64-frame development prefixes only. Track/candidate counts are unlabeled bounded review workload. '
                'Known-control recovery and motion availability are not real-object recall. Times include validation instrumentation.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--current', type=Path, required=True)
    parser.add_argument('--v2', type=Path, required=True)
    parser.add_argument('--v3', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = summarize(args.current, args.v2, args.v3)
    write_json(args.output, result)
    print(json.dumps(result['gates'], indent=2))
