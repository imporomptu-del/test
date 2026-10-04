"""Independent V43 accounting/provenance audit; no producer imports or decoding."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/tiny_target/accuracy_v43_20260925'
V42 = ROOT / 'results/tiny_target/accuracy_v42_20260925'
BASELINE = V42 / 'localized_01'
BASELINE_RECEIPT_SHA = '9077ba90861a05bea591ff4961c818755002e9f11a28d06a7d6a1e86854c3c46'
INVENTORY_SHA = '444df960282569ef4c60a80a15bc6a50756fe0eadba421969061f6fb5aab86df'
ARMS = ('baseline', 'stable_only', 'protected_only', 'combined')


def invalid_constant(value):
    raise ValueError('Nonfinite JSON constant: ' + value)


def read(path):
    return json.loads(Path(path).read_text(), parse_constant=invalid_constant)


def records(path):
    with Path(path).open() as stream:
        return [json.loads(line, parse_constant=invalid_constant) for line in stream]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def key(record):
    return record['clip'], record['frame_index'], record['segment'], record['track_id']


def require_finite(value):
    if isinstance(value, dict):
        for item in value.values(): require_finite(item)
    elif isinstance(value, list):
        for item in value: require_finite(item)
    elif type(value) is float:
        assert math.isfinite(value)


def check_arm(value):
    if value is None: return
    require_finite(value)
    assert type(value['available']) is bool and type(value['ambiguous']) is bool
    if value['available']:
        assert not value['reasons']
        assert len(value['folds']) == 2
        assert all(type(value[k]) in (int, float) and math.isfinite(value[k]) for k in
                   ('mse_stationary', 'mse_augmented', 'advantage_stationary_minus_augmented'))
        assert value['mse_stationary'] >= 0 and value['mse_augmented'] >= 0
        assert math.isclose(value['mse_stationary'] - value['mse_augmented'],
                            value['advantage_stationary_minus_augmented'], rel_tol=1e-8, abs_tol=1e-6)
        assert value['common_support_count'] == sum(value['fold_support_counts'])
        assert value['common_support_sha256'] is not None
    else:
        assert value['mse_stationary'] is None and value['mse_augmented'] is None
        assert value['advantage_stationary_minus_augmented'] is None
        assert value['folds'] == []
        assert value['reasons']
    assert value['ambiguous'] == bool(value['ambiguity_reasons'])
    metadata = value['components']
    for reason in metadata.get('fixed_learning_ambiguity_reasons', []):
        assert reason in value['ambiguity_reasons']
    if metadata['anchor_dictionary_truncated']:
        assert 'persistent_anchor_dictionary_truncated' in value['ambiguity_reasons']
    if metadata['persistent_anchor_overlaps_prediction']:
        assert 'causal_prediction_overlaps_persistent_fixed_anchor' in value['ambiguity_reasons']


def recount(rows, arm):
    out = Counter(states=0, strict_states=0, geometry_available=0, score_available=0,
                  score_with_ambiguity=0, score_without_recorded_ambiguity=0)
    for row in rows:
        out['states'] += 1
        out['strict_states'] += int(row['qualified_moving'])
        out['geometry_available'] += int(row['geometry']['available'])
        value = row['arms'][arm]
        if value is None:
            out['geometry_unavailable'] += 1
            continue
        out['score_available'] += int(value['available'])
        out['score_with_ambiguity'] += int(value['available'] and value['ambiguous'])
        out['score_without_recorded_ambiguity'] += int(value['available'] and not value['ambiguous'])
        out.update('unavailable:' + reason for reason in value['reasons'])
        out.update('ambiguity:' + reason for reason in value['ambiguity_reasons'])
    return dict(out)


def independent_pair(rows, arm):
    counts, changes = Counter(states=len(rows)), []
    for row in rows:
        first, second = row['arms']['baseline'], row['arms'][arm]
        a, b = bool(first and first['available']), bool(second and second['available'])
        counts['available_' + str(int(a)) + '_' + str(int(b))] += 1
        if a and b:
            same = first['common_support_sha256'] == second['common_support_sha256']
            counts['both_available_same_support' if same else 'both_available_different_support'] += 1
            if same:
                assert first['common_support_count'] == second['common_support_count']
                assert first['fold_support_counts'] == second['fold_support_counts']
                changes.append(second['advantage_stationary_minus_augmented'] - first['advantage_stationary_minus_augmented'])
    return dict(first='baseline', second=arm, counts=dict(counts),
                same_support_advantage_change_quantiles=None if not changes else np.quantile(changes, [0, .5, 1]).tolist(),
                changed_support_mse_not_compared=True, improvement_is_not_accuracy=True)


def date(value):
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    assert parsed.tzinfo is not None
    return parsed


def run(directory, output):
    directory, output = Path(directory).resolve(), Path(output).resolve()
    if output.exists(): raise FileExistsError('Fresh audit output required')
    receipt_path = directory / 'completion_receipt.json'
    if not receipt_path.is_file(): raise RuntimeError('Completed run receipt required')
    checked = {}

    def bind(path, expected=None):
        path = str(Path(path).resolve())
        digest = sha(path)
        if expected is not None: assert digest == expected, path
        if path in checked: assert digest == checked[path], path
        checked[path] = digest

    bind(receipt_path)
    receipt = read(receipt_path)
    assert receipt['completed'] and receipt['all_inputs_and_outputs_rehashed']
    assert receipt['production_changed'] is False
    for path, digest in receipt['files_sha256'].items(): bind(path, digest)
    bind(BASELINE / 'completion_receipt.json', BASELINE_RECEIPT_SHA)
    old_receipt = read(BASELINE / 'completion_receipt.json')
    assert len(old_receipt['files_sha256']) == 52
    freeze, cache, score_start = [read(directory / name) for name in
                                   ('freeze.json', 'cache_manifest.json', 'score_start.json')]
    for path, digest in old_receipt['files_sha256'].items():
        assert freeze['inputs_sha256'][path] == receipt['files_sha256'][path] == digest
    assert freeze['arms'] == list(ARMS) and freeze['pre_native_decode_and_score'] is True
    assert freeze['production_changed'] is False
    assert cache['completed_before_any_arm_score'] is True
    assert cache['all_geometry_exactly_reproduced'] is True
    assert cache['all_seven_v42_saved_inputs_exactly_reproduced'] is True
    assert score_start['all_bound_files_rehashed_before_scoring'] is True
    assert score_start['cache_manifest_sha256'] == sha(directory / 'cache_manifest.json')
    assert score_start['pre_score_files_sha256'][str(directory / 'freeze.json')] == sha(directory / 'freeze.json')
    assert score_start['pre_score_files_sha256'][str(directory / 'cache_manifest.json')] == sha(directory / 'cache_manifest.json')
    for path, digest in score_start['pre_score_files_sha256'].items():
        assert receipt['files_sha256'][path] == digest
    for name in ('states.jsonl', 'summary.json', 'reference_evidence.json'):
        assert str(directory / name) not in score_start['pre_score_files_sha256']
    inventory_path = V42 / 'reference_inventory.json'
    bind(inventory_path, INVENTORY_SHA)
    bind(__file__)
    bind(ROOT / 'tests/unit/test_accuracy_v43_accounting_audit.py')
    inventory = read(inventory_path)
    selection = read(BASELINE / 'selection.json')
    old_records, new_records = records(BASELINE / 'states.jsonl'), records(directory / 'states.jsonl')
    old_lookup, lookup = {key(r): r for r in old_records}, {key(r): r for r in new_records}
    cached = {key(r): r for r in cache['states']}
    selected = {key(r): r for r in selection['states']}
    assert len(old_records) == len(new_records) == len(lookup) == len(old_lookup) == len(cached) == len(selected) == 1698
    assert set(lookup) == set(old_lookup) == set(cached) == set(selected)
    archive_count, old_archive_comparisons = 0, 0
    for state_key, row in lookup.items():
        old, item = old_lookup[state_key], cached[state_key]
        assert row['geometry'] == old['geometry'] == item['geometry']
        for name in ('actual_source_xy', 'qualified_moving', 'grid_windows', 'reference_samples', 'save_audit_inputs'):
            assert row[name] == old[name] == item[name] == selected[state_key][name]
        assert set(row['arms']) == set(ARMS)
        assert row['arms']['baseline'] == old['localized']
        assert row['archive'] == item['archive']
        if row['geometry']['available']:
            assert row['archive'] is not None
            archive_count += 1
            archive = (directory / row['archive']['path']).resolve()
            assert archive.parent == directory / 'inputs' and archive.suffix == '.npz'
            digest = row['archive']['sha256']
            assert checked[str(archive)] == receipt['files_sha256'][str(archive)] == digest
            assert score_start['pre_score_files_sha256'][str(archive)] == digest
            assert all(row['arms'][arm] is not None for arm in ARMS)
            old_archive_comparisons += int(old['audit_inputs'] is not None)
        else:
            assert row['archive'] is None
            assert all(row['arms'][arm] is None for arm in ARMS)
        for value in row['arms'].values(): check_arm(value)
    assert archive_count == 605 and old_archive_comparisons == 7
    summary, reference = read(directory / 'summary.json'), read(directory / 'reference_evidence.json')
    assert (date(freeze['created_at_utc']) <= date(cache['cache_completed_at_utc'])
            <= date(score_start['started_at_utc']) <= date(summary['created_at_utc'])
            <= date(receipt['created_at_utc']))
    assert summary['completed'] and summary['baseline_exact_reproduction']
    assert summary['production_changed'] is False and summary['no_filter_applied'] is True
    assert summary['no_new_labels'] is True
    groups = dict(all=new_records, strict=[r for r in new_records if r['qualified_moving']],
                  grid=[r for r in new_records if r['grid_windows']],
                  grid_strict=[r for r in new_records if r['grid_windows'] and r['qualified_moving']])
    assert [len(groups[k]) for k in ('all', 'strict', 'grid', 'grid_strict')] == [1698, 675, 1367, 381]
    group_counts = {group: {arm: recount(rows, arm) for arm in ARMS} for group, rows in groups.items()}
    assert summary['groups'] == group_counts
    paired = [independent_pair(new_records, arm) for arm in ARMS[1:]]
    assert summary['paired'] == paired
    for clip, metadata in cache['source_metadata'].items():
        assert metadata['retrieved_indices'] == selection['needed_source_frames'][clip]
        assert metadata['sequential_frames'] == inventory['source_scopes'][clip]['declared_frame_count']
        assert metadata['eof_verified'] and metadata['max_retained_native_gray_frames'] == 9
        assert metadata['states'] == sum(r['clip'] == clip for r in new_records)
        assert metadata['packets'] == sum(r['clip'] == clip and r['archive'] is not None for r in new_records)
    assert len(reference['samples']) == len(inventory['samples']) == 356
    assert len(inventory['original_gated_measured_states']) == 368
    assert {tuple(r[:4]) for r in inventory['original_gated_measured_states']} <= set(lookup)
    for i, raw in enumerate(inventory['samples']):
        original = dict(zip(inventory['sample_columns'], raw))
        reported = reference['samples'][i]
        assert reported['sample_index'] == i and reported['original'] == original
        ids = original['stages']['actual_measurement'][2]
        assert [a['identity'] for a in reported['measured_alternatives']] == ids
        for identity, alternative in zip(ids, reported['measured_alternatives']):
            segment, tid = identity.split('/', 1)
            row = lookup[(original['clip_id'], original['frame_index'], int(segment), tid)]
            assert alternative['qualified_moving'] == row['qualified_moving']
            assert alternative['geometry_available'] == row['geometry']['available']
            assert alternative['arms'] == row['arms']
    panels = {}
    for panel, original in inventory['panels'].items():
        subset = [r for r in reference['samples'] if r['original']['panel'] == panel]
        assert len(subset) == original['samples']
        panel_arms = {}
        for arm in ARMS:
            values = Counter(samples=len(subset), original_strict_assignment=0, same_assignment_score_available=0,
                             same_assignment_score_without_recorded_ambiguity=0)
            for sample in subset:
                identity = sample['original']['stages'][original['original_strict_stage']][1]
                if identity is None: continue
                values['original_strict_assignment'] += 1
                alternatives = [a for a in sample['measured_alternatives'] if a['identity'] == identity]
                assert len(alternatives) == 1
                value = alternatives[0]['arms'][arm]
                if value is not None and value['available']:
                    values['same_assignment_score_available'] += 1
                    values['same_assignment_score_without_recorded_ambiguity'] += int(not value['ambiguous'])
            assert values['original_strict_assignment'] == original['hits'][original['original_strict_stage']]
            panel_arms[arm] = dict(values)
        panels[panel] = dict(original=original, arms=panel_arms)
    assert panels == reference['panels'] == summary['reference_panels']
    for path, digest in checked.items(): assert sha(path) == digest, path
    result = dict(schema='seaqr.accuracy-v43-independent-accounting-audit.v1', verified=True,
        completed_at_utc=datetime.now(timezone.utc).isoformat(), experiment=str(directory),
        checked_files_sha256=checked, all_bound_files_rehashed=True,
        states=1698, strict_states=675, grid_states=1367, grid_strict_states=381,
        reference_unique_states=368, reference_samples=356, cached_archives=605,
        v42_completion_bound_files_verified=52, every_v42_geometry_exactly_reproduced=True,
        every_v42_baseline_outcome_exactly_reproduced_including_none=True,
        every_original_alternative_assignment_and_miss_retained=True,
        no_nonfinite_arm_output=True, no_operative_mse_on_unavailable=True,
        protected_component_ambiguity_propagated=True,
        producer_declared_cache_before_score_chronology_hash_chain_verified=True,
        independent_wall_clock_attestation=False,
        chronology_caveat='UTC ordering and immutable producer code/hash-chain agree; producer timestamps are not independent clock attestation.',
        cache_array_contents_independently_recomputed=False,
        seven_prior_saved_array_equality_claim_not_recomputed_here=True,
        source_media_bytes_hashed_only=True, source_media_decoded=False,
        no_producer_model_or_runner_functions_imported=True,
        numerical_sensitivity_math_independently_recomputed=False,
        group_counts=group_counts, paired_support_accounting=paired, reference_panels=panels,
        no_best_arm_or_identity_selection=True, no_accuracy_gain_claimed=True)
    with output.open('x') as stream: json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({k: result[k] for k in ('verified', 'states', 'cached_archives', 'group_counts')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=BASE / 'stability_01')
    parser.add_argument('--output', type=Path, default=BASE / 'accounting_independent_audit_01.json')
    args = parser.parse_args()
    run(args.run, args.output)
