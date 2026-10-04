"""Report-only summary for capacity plus fresh flow-state isolation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from profile_raw16_efficiency import sha, write_json
from summarize_raw16_motion_v3 import read, trial


def summarize(current, v4):
    previous = read(v4/'summary.json')
    names = ('full_frame_0029', 'full_frame_0040', 'full_frame_0040_injected')
    trials = {n: {**previous['trials'][n], 'v5': trial(current, n+'_v5')} for n in names}
    comparisons = {}
    for name in names:
        now, old = current/(name+'_v5'), v4/(name+'_v4')
        a, b = read(now/'report.json'), read(old/'report.json')
        comparisons[name] = dict(
            same_source=read(now/'source_frames.json') == read(old/'source_frames.json'),
            same_detector=a['configuration']['effective'] == b['configuration']['effective'],
            same_full_native_geometry=a['source']['crop_xywh'] == b['source']['crop_xywh'] == [0, 0, 4784, 3190])
    plain, injected = current/'full_frame_0040_v5', current/'full_frame_0040_injected_v5'
    paired = dict(
        same_source=read(plain/'source_frames.json') == read(injected/'source_frames.json'),
        same_motion=read(plain/'report.json')['source']['pva_stabilization']['metrics']['motion_pairs'] ==
                    read(injected/'report.json')['source']['pva_stabilization']['metrics']['motion_pairs'])
    controls = []
    for seed in (75316, 129827, 85723):
        path = current/f'status_controls_seed{seed}.json'
        report = read(path)
        observations = []
        for case in report['cases']:
            observations.extend([case['result']] if 'result' in case else [case[k] for k in ('first', 'unrelated', 'recovered')])
        controls.append(dict(seed=seed, sha256=sha(path), cases=len(report['cases']),
            passed=sum(c['passed'] for c in report['cases']),
            runtime_failures=sum(o['runtime_failure'] for o in observations),
            worst_translation_error_px=max(o.get('translation_error_px') or 0 for o in observations)))
    sequence = read(current/'sequence_0040_v5.json')
    sequence_check = dict(passed=sequence['passed'], pairs=len(sequence['pairs']),
        accepted_pairs=sum(p['fit']['quality_status'] == 'accepted' for p in sequence['pairs']),
        exact_reverse_replays=sum(r['identical_points'] for r in sequence['reverse_replays']),
        reverse_replays=len(sequence['reverse_replays']), sha256=sha(current/'sequence_0040_v5.json'))
    old_targets = trials['full_frame_0040_injected']['v2']['injected_control_matches']['detected_target_ids']
    now_targets = trials['full_frame_0040_injected']['v5']['injected_control_matches']['detected_target_ids']
    gates = dict(
        generated_controls_all_pass=all(r['cases'] == r['passed'] == 16 and r['runtime_failures'] == 0 for r in controls),
        source_detector_geometry_unchanged=all(all(r.values()) for r in comparisons.values()),
        injection_does_not_change_source_or_motion=all(paired.values()),
        processing_integrity=all(r['v5']['checks']['processing_integrity_passed'] for r in trials.values()),
        search_available=all(r['v5']['checks']['detection_availability_passed'] for r in trials.values()),
        motion_and_search_availability_nonregression=all(
            r['v5'][key] >= max(r[v][key] for v in ('v2', 'v3', 'v4'))
            for r in trials.values() for key in ('applied_transforms', 'ready_filter_frames', 'windows')),
        previous_injected_matches_preserved=set(old_targets).issubset(now_targets),
        order_invariance=sequence_check['passed'] and sequence_check['pairs'] == sequence_check['accepted_pairs'] == 63
                        and sequence_check['exact_reverse_replays'] == sequence_check['reverse_replays'] == 8)
    development_passed = all(gates.values())
    gates.update(all_three_injected_controls_recovered=trials['full_frame_0040_injected']['v5']['checks']['synthetic_controls_passed'],
        real_airborne_accuracy_validated=False, speed_improvement_validated=False,
        default_configuration_changed=False, production_ready=False)
    return dict(schema_version='seaqr.raw16-motion-summary.v5', development_nonregression_passed=development_passed,
        trials=trials, generated_controls=controls, sequence_check=sequence_check,
        source_comparisons=comparisons, injected_pair=paired, gates=gates,
        warning='Only 64-frame development prefixes; bounded track counts are unlabeled review workload. '
                'VPI cache isolation is a correctness workaround, not a deployment or speed claim.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--current', type=Path, required=True)
    parser.add_argument('--v4', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = summarize(args.current, args.v4)
    write_json(args.output, result)
    print(json.dumps(result['gates'], indent=2))
