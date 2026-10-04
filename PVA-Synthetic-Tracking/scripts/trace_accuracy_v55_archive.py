"""Read-only compact-evidence joins. Never imports or executes a detector."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/tiny_target'
OUT = BASE / 'accuracy_v55_20260926'
DATA = {
    'diagnostic': 'accuracy_v48_20260926/diagnostic_breakdown_01.json',
    'ledger': 'accuracy_v48_20260926/shadow_01/selected_ledger.json',
    'references': 'accuracy_v48_20260926/shadow_01/reference_evidence.json',
    'v48_summary': 'accuracy_v48_20260926/shadow_01/summary.json',
    'v36_summary': 'accuracy_v36_20260924/full_context_01/summary.json',
    'v39_summary': 'accuracy_v39_20260925/continuity_01/summary.json',
    'motion_history': 'accuracy_v39_20260925/motion_history_01.json',
    'coverage': 'accuracy_v40_20260925/coverage_workload_01/window_evidence.json',
    'coverage_summary': 'accuracy_v40_20260925/coverage_workload_01/summary.json',
    'context0029': 'accuracy_v36_20260924/full_context_01/0029_decisions.jsonl',
    'context0126': 'accuracy_v36_20260924/full_context_01/0126_decisions.jsonl',
    'continuity0029': 'accuracy_v39_20260925/continuity_01/0029_decisions.jsonl',
    'continuity0126': 'accuracy_v39_20260925/continuity_01/0126_decisions.jsonl',
    'launch0029': 'visible_validation_v34_20260923/audit_20260924/evidence/run/full_repeat0_0029/launch.json',
    'launch0126': 'visible_validation_v34_20260923/audit_20260924/evidence/run/full_repeat0_0126/launch.json',
    'cuda_build': 'visible_front_v26_20260920/evidence/build_03/build.json',
    'candidate_config': 'visible_front_v26_20260920/evidence/candidate_config.json',
}
RECEIPTS = {
    'v48': ('accuracy_v48_20260926/shadow_01/completion_receipt.json', 'files_sha256'),
    'v36': ('accuracy_v36_20260924/full_context_independent_audit_01.json', 'checked_files_sha256'),
    'v39': ('accuracy_v39_20260925/continuity_01/completion_receipt.json', 'checked_files_sha256'),
    'v40': ('accuracy_v40_20260925/coverage_workload_01/completion_receipt.json', 'inputs_outputs_sha256'),
}
CODE = ('visible_baseline.py', 'visible_quality.py', 'visible_resident.py',
        'visible_shapes.py', 'tracking/kalman.py')
SOURCE_NAMES = ('docs/accuracy_v55_plan.md', 'scripts/trace_accuracy_v55_archive.py',
                'tests/unit/test_accuracy_v55_archive.py',
                'results/tiny_target/visible_front_v26_20260920/evidence/build_03/source/phase20_cuda_resident.cu',
                'results/tiny_target/visible_front_v26_20260920/evidence/build_03/source/visible_front_v26.cu',
                'results/tiny_target/visible_front_v26_20260920/evidence/visible_front_v26.py')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    path = Path(path)
    require(path.is_absolute() and path.resolve() == path and path.is_file() and not path.is_symlink(),
            'Canonical regular file required')
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def index_rows(rows, key):
    result = {}
    for row in rows:
        identity = key(row)
        require(identity not in result, 'Duplicate identity')
        result[identity] = row
    return result


def state_key(row):
    return row['clip'], row['frame_index'], row['segment'], row['track_id']


def join_state(detail, ledger_row, references, context_row, continuity_row):
    key = tuple(detail['state_key'])
    require(key == state_key(ledger_row), 'State identity mismatch')
    require(detail['previous_stage'] == 'available_positive', 'Not an original numerical positive')
    require(detail['original_strict_qualified'] is ledger_row['qualified_moving'], 'Qualification mismatch')
    assigned, alternatives = [], []
    for ref in references:
        matches = [a for a in ref['measured_alternatives'] if tuple(a['state_key']) == key]
        require(len(matches) <= 1, 'Duplicate reference alternative')
        if matches:
            alternatives.append(ref['sample_index'])
            if matches[0]['identity'] == ref['original_strict_assigned_identity']:
                assigned.append(ref['sample_index'])
    for row in (context_row, continuity_row):
        if row is not None:
            require((row['segment'], row['track_id']) == key[2:], 'Joined track identity mismatch')
            require(row['measured'] is True, 'Prediction cannot stand in for actual measurement')
            require(row['measurement_source_xy'] == ledger_row['actual_source_xy'], 'Actual coordinate mismatch')
    if ledger_row['qualified_moving']:
        require(context_row is not None and continuity_row is not None, 'Missing qualified decision')
        require(continuity_row['baseline_qualified'] is True and
                continuity_row['status'] == 'baseline_qualified_measured', 'Not unchanged baseline output')
    else:
        require(context_row is None and continuity_row is None, 'Unexpected output for this unqualified state')
    return dict(state_key=list(key), actual_source_xy=ledger_row['actual_source_xy'],
        baseline_qualified_measured=ledger_row['qualified_moving'],
        numerical_previous=detail['previous_stage'], numerical_later=detail['new_stage'],
        numerical_guard_reasons=detail['gain_reasons'], original_assigned_reference_samples=assigned,
        all_reference_alternative_samples=alternatives,
        v36_parallel_shadow=None if context_row is None else
            {k: context_row[k] for k in ('accepted', 'reason', 'features')},
        v39_parallel_shadow=None if continuity_row is None else
            {k: continuity_row[k] for k in ('status', 'baseline_qualified', 'measured', 'renderable')},
        absent_decision_means_rejected=False, numerical_unknown_means_detector_miss=False)


def summarize_references(refs):
    index_rows(refs, lambda r: r['sample_index'])
    panels, assigned_keys, all_keys, misses, records = {}, set(), set(), [], []
    assigned_count = alternative_count = 0
    for ref in refs:
        original = ref['original']
        counts = panels.setdefault(original['panel'], Counter())
        counts['samples'] += 1
        stages = original['stages']
        strict = 'baseline_qualified' if 'baseline_qualified' in stages else 'strict_qualified_measurement'
        require(strict in stages, 'Unknown strict-stage schema')
        for stage, value in stages.items():
            require(type(value[0]) is bool, 'Stage hit must be boolean')
            counts[stage] += int(value[0])
        alternatives = ref['measured_alternatives']
        index_rows(alternatives, lambda a: a['identity'])
        for alternative in alternatives:
            key = alternative['state_key']
            require(len(key) == 4 and key[:2] == [original['clip_id'], original['frame_index']],
                    'Reference alternative frame/clip mismatch')
            require(alternative['identity'] == f'{key[2]}/{key[3]}', 'Alternative identity mismatch')
            require(key[3].split(':')[0] == original['polarity'], 'Alternative polarity mismatch')
            require(type(alternative['original_qualified_moving']) is bool, 'Alternative qualification not boolean')
        selected = [a for a in alternatives if a['identity'] == ref['original_strict_assigned_identity']]
        if ref['original_strict_assigned_identity'] is None:
            require(not stages[strict][0] and stages[strict][1] is None and stages[strict][2] == [],
                    'Unassigned strict hit or alternatives')
            misses.append(original)
        else:
            require(len(selected) == 1 and stages[strict][0], 'Original assignment missing or non-strict')
            require(stages[strict][1] == ref['original_strict_assigned_identity'], 'Original identity replaced')
            require(selected[0]['original_qualified_moving'] is True, 'Original assigned state not qualified')
            assigned_count += 1
            assigned_keys.add(tuple(selected[0]['state_key']))
        alternative_count += len(alternatives)
        all_keys.update(tuple(a['state_key']) for a in alternatives)
        records.append(dict(sample_index=ref['sample_index'], original=original,
            original_strict_assigned_identity=ref['original_strict_assigned_identity'],
            measured_alternatives=[{k: a[k] for k in ('identity', 'state_key', 'original_qualified_moving')}
                                   for a in alternatives]))
    return dict(samples=len(refs), assigned_samples=assigned_count, unique_assigned_states=len(assigned_keys),
        alternative_entries=alternative_count, unique_alternative_states=len(all_keys),
        panels={k: dict(v) for k, v in panels.items()}, original_unassigned=misses,
        records=records, overlapping_not_independent=True, airborne_truth=False)


def summarize_controls(summary):
    arms = summary['clips']['0126']['arms']
    left, right = arms['baseline']['provisional_controls'], arms['point_context']['provisional_controls']
    key = lambda r: (tuple(r['frames_inclusive']), tuple(r['crop_xywh']), r['label'])
    before, after = index_rows(left, key), index_rows(right, key)
    require(before.keys() == after.keys(), 'Control scopes differ')
    rows = [dict(clip='0126', frames_inclusive=a['frames_inclusive'], crop_xywh=a['crop_xywh'],
                 label=a['label'], baseline_measured=a['measured_states'],
                 baseline_predicted=a['predicted_states'], v36_measured=after[k]['measured_states'],
                 v36_predicted=after[k]['predicted_states'], verified_airborne_negative=False)
            for k, a in before.items()]
    return dict(windows=rows, totals={name: sum(r[name] for r in rows) for name in
        ('baseline_measured', 'baseline_predicted', 'v36_measured', 'v36_predicted')},
        false_positive_rate=None)


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def run(output):
    output = Path(output).absolute()
    require(output.parent == OUT and output.resolve() == output and not output.exists(), 'Fresh V55 child required')
    paths = {name: BASE/rel for name, rel in DATA.items()}
    sources = [ROOT/name for name in SOURCE_NAMES] + [ROOT/'tiny_target'/name for name in CODE]
    receipts = {name: json.loads((BASE/rel).read_text()) for name, (rel, _) in RECEIPTS.items()}
    receipt_paths = [BASE/rel for rel, _ in RECEIPTS.values()]
    bindings = {str(p): sha(p) for p in [*paths.values(), *sources, *receipt_paths]}
    historical = {}
    for name, path in paths.items():
        matched = []
        for label, receipt in receipts.items():
            prior = receipt[RECEIPTS[label][1]].get(str(path))
            if prior is not None:
                require(prior == bindings[str(path)], 'Historical binding changed: '+name)
                matched.append(label)
        historical[name] = dict(matching_historical_receipts=matched,
                               newly_hashed_sidecar_without_prior_binding=not matched)
    output.mkdir(parents=True)
    write(output/'freeze.json', dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        files_sha256=bindings, historical_bindings=historical, retrospective_exposed_diagnostic=True,
        paths_inside_receipts_not_followed=True, media_journal_packets_not_opened=True))
    data = {name: ([json.loads(line) for line in path.open()] if path.suffix == '.jsonl'
                  else json.loads(path.read_text())) for name, path in paths.items()}
    provenance = {}
    for clip in ('0029', '0126'):
        launch = data['launch'+clip]
        build = data['cuda_build']
        require(sha(paths['candidate_config']) == launch['config_sha256'], 'Archived config hash differs')
        for filename in ('phase20_cuda_resident.cu', 'visible_front_v26.cu'):
            require(sha(paths['cuda_build'].parent/'source'/filename) == build['source_sha256'][filename],
                    'Archived CUDA source hash differs')
        require(sha(paths['cuda_build'].parent.parent/'visible_front_v26.py') == build['adapter_sha256'],
                'Archived CUDA adapter hash differs')
        require(build['library_sha256'] == launch['external_accelerators']['median']['library_sha256'],
                'CUDA build/library metadata differs')
        for name in CODE:
            require(sha(ROOT/'tiny_target'/name) == launch['package_sha256'][name], 'Production Python source changed')
        provenance[clip] = dict(configuration=launch['configuration'], config_sha256=launch['config_sha256'],
            code_sha256={name: launch['package_sha256'][name] for name in CODE},
            external_accelerators=launch['external_accelerators'], runtime_revalidated=False,
            archived_cuda_source_sha256={name: build['source_sha256'][name] for name in
                ('phase20_cuda_resident.cu', 'visible_front_v26.cu')},
            archived_adapter_sha256=build['adapter_sha256'], native_library_not_opened_or_executed=True)
    ledger = index_rows(data['ledger']['states'], state_key)
    refs = data['references']['samples']
    joined = []
    lookup = {}
    frame_metadata = {}
    for prefix, field in (('context', 'tracks'), ('continuity', 'output_states')):
        for clip in ('0029', '0126'):
            frames = index_rows(data[prefix+clip], lambda row: (row['frame_index'], row['segment']))
            frame_metadata[prefix, clip] = frames
            require(all(track['segment'] == row['segment'] for row in frames.values() for track in row[field]),
                    'Track/frame segment mismatch')
            lookup[prefix, clip] = {key: index_rows(row[field], lambda r: r['track_id']) for key, row in frames.items()}
    details = data['diagnostic']['previous_positive_states_followup']
    index_rows(details, lambda d: tuple(d['state_key']))
    for detail in details:
        clip, frame, segment, track = detail['state_key']
        require(all((frame, segment) in lookup[prefix, clip] for prefix in ('context', 'continuity')),
                'Selected frame missing; cannot infer absent track')
        require(frame_metadata['context', clip][frame, segment]['timestamp_ns'] ==
                frame_metadata['continuity', clip][frame, segment]['timestamp_ns'], 'Frame timestamps differ')
        values = [lookup[prefix, clip][frame, segment].get(track) for prefix in ('context', 'continuity')]
        require(values[0] is None or values[0]['measurement_frame'] == frame, 'Context evidence is from another frame')
        joined.append(join_state(detail, ledger[tuple(detail['state_key'])], refs, *values))
    reference_summary = summarize_references(refs)
    require(len(joined) == 7 and len(ledger) == 1211 and reference_summary['samples'] == 355,
            'Frozen scope denominator differs')
    require(tuple(reference_summary[k] for k in ('assigned_samples', 'unique_assigned_states',
            'alternative_entries', 'unique_alternative_states')) == (352, 300, 424, 367),
            'Original assignment/alternative denominators differ')
    require(reference_summary['original_unassigned'] == data['diagnostic']['original_unassigned_references'],
            'Original misses changed')
    coverage = data['coverage']
    require(len(coverage) == 108 and all(r['source_review']['authoritative_negative'] is False for r in coverage),
            'Coverage/negative interpretation changed')
    result = dict(completed=True, retrospective_archive_trace=True, production_changed=False,
        source_media_or_journals_read=False, new_detector_run=False, seven_numerical_states=joined,
        reference_summary=reference_summary, provisional_controls=summarize_controls(data['v36_summary']),
        coverage=dict(windows=108, scene_counts=dict(Counter(r['source_review']['scene_type'] for r in coverage)),
                      authoritative_negative_windows=0, saved_workload=data['coverage_summary']),
        motion_qualification_evidence=dict(parameters=data['motion_history']['parameters'],
            focus_states=data['motion_history']['focus_states'], independently_refitted_in_this_run=False),
        production_python_provenance=provenance,
        missing_candidate_diagnostics=['prelearning spatial response', 'prelearning background and temporal residual',
            'tile center and tile noise', 'pixel variance and support', 'threshold margins and failure masks',
            'raw-absolute peak competition', 'prequota rank and suppression', 'shape relocation lineage'],
        earliest_known_0126_frame216_absence='saved_post_quota_post_shape_candidate',
        cause_of_0126_frame216_candidate_absence=None,
        v36_and_v39_are_parallel_nonpromoted_shadows=True)
    write(output/'trace.json', result)
    for p, digest in bindings.items():
        require(sha(Path(p)) == digest, 'Bound input changed during trace')
    write(output/'completion_receipt.json', dict(completed=True, created_at_utc=datetime.now(timezone.utc).isoformat(),
        files_sha256={**bindings, **{str(p): sha(p) for p in (output/'freeze.json', output/'trace.json')}},
        production_changed=False, old_evidence_changed=False))
    print('V55 complete: seven exact joins, all 355 reference rows and control scopes retained; no detector run.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
