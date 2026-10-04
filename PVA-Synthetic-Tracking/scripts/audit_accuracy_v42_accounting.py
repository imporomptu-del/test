"""Independent completed-run accounting audit; no model imports or media decode."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/tiny_target/accuracy_v42_20260925'
INVENTORY = BASE / 'reference_inventory.json'
INVENTORY_SHA = '444df960282569ef4c60a80a15bc6a50756fe0eadba421969061f6fb5aab86df'
GRID = ROOT / 'results/tiny_target/accuracy_v40_20260925/coverage_workload_01/window_evidence.json'
GRID_SHA = 'cefb4ccc97a894d16a02c5a11c9ce4696f4b49a279a5df22477fcd175a7b87a3'
CLIPS = ('0029', '0126', '0055', '0082')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def key(record):
    return (record['clip'], record['frame_index'], record['segment'], record['track_id'])


def counts(records):
    out = Counter(states=0, strict_states=0, geometry_available=0, localized_available=0, ambiguous=0)
    for record in records:
        out['states'] += 1
        out['strict_states'] += int(record['qualified_moving'])
        geometry, localized = record['geometry'], record['localized']
        out['geometry_available'] += int(geometry['available'])
        out.update('geometry_reason:' + reason for reason in geometry['reasons'])
        if localized is not None:
            out['localized_available'] += int(localized['available'])
            out['ambiguous'] += int(localized['ambiguous'])
            out.update('localized_reason:' + reason for reason in localized['reasons'])
            out.update('ambiguity_reason:' + reason for reason in localized['ambiguity_reasons'])
    return dict(out)


def outcome(record):
    value = record['localized']
    return dict(state_key=list(key(record)), qualified_moving=record['qualified_moving'],
        geometry_available=record['geometry']['available'],
        geometry_reasons=record['geometry']['reasons'],
        localized_available=bool(value and value['available']),
        ambiguous=None if value is None else value['ambiguous'],
        reasons=None if value is None else value['reasons'],
        ambiguity_reasons=None if value is None else value['ambiguity_reasons'],
        mse_stationary=None if value is None else value['mse_stationary'],
        mse_augmented=None if value is None else value['mse_augmented'],
        predicted_to_actual_distance_px=record['predicted_to_actual_distance_px'])


def outcome_counts(outcomes):
    return dict(alternatives=len(outcomes),
        geometry_available=sum(o['geometry_available'] for o in outcomes),
        localized_available=sum(o['localized_available'] for o in outcomes),
        available_unambiguous=sum(o['localized_available'] and o['ambiguous'] is False for o in outcomes),
        ambiguous_including_unavailable=sum(o['ambiguous'] is True for o in outcomes))


def run(directory, output):
    directory, output = Path(directory).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Fresh audit artifact required')
    receipt_path = directory / 'completion_receipt.json'
    if not receipt_path.is_file():
        raise RuntimeError('Completed receipt is required; no partial audit')
    checked = {}

    def bind(path, expected=None):
        path = str(Path(path).resolve())
        value = sha(path)
        if expected is not None and value != expected:
            raise AssertionError('Changed hash: ' + path)
        if path in checked and checked[path] != value:
            raise AssertionError('Input changed during audit: ' + path)
        checked[path] = value

    bind(receipt_path)
    receipt = read(receipt_path)
    assert receipt['completed'] and receipt['all_inputs_and_outputs_rehashed']
    assert receipt['production_changed'] is False
    for path, digest in receipt['files_sha256'].items():
        bind(path, digest)
    bind(INVENTORY, INVENTORY_SHA)
    bind(GRID, GRID_SHA)
    bind(__file__)
    inventory, grid = read(INVENTORY), read(GRID)
    freeze = read(directory / 'freeze.json')
    selection = read(directory / 'selection.json')
    summary = read(directory / 'summary.json')
    reference = read(directory / 'reference_evidence.json')
    assert freeze['pre_native_decode_and_score'] is True
    assert freeze['production_change'] is False
    assert freeze['exposed_development'] is True
    assert freeze['selection_sha256'] == sha(directory / 'selection.json')
    assert set(freeze['media_allowlist']) == set(CLIPS)
    for path, digest in freeze['inputs_sha256'].items():
        assert receipt['files_sha256'][path] == digest
    for path, digest in inventory['inputs_sha256'].items():
        assert freeze['inputs_sha256'][path] == digest
    expected = {}
    grid_keys = set()
    grid_states = grid_strict = 0
    for window in grid:
        for frame in window['frames']:
            for track in frame['actual_measurements']:
                state_key = (window['clip_id'], frame['frame_index'], track['segment'], track['track_id'])
                value = [track['source_xy'], track['qualified_moving']]
                if state_key in expected:
                    assert expected[state_key][:2] == value
                    expected[state_key][2].add(window['window_id'])
                else:
                    expected[state_key] = [*value, {window['window_id']}, set()]
                grid_keys.add(state_key)
                grid_states += 1
                grid_strict += int(track['qualified_moving'])
    assert len(grid) == 108 and grid_states == len(grid_keys) == 1367 and grid_strict == 381
    reference_keys = set()
    for row in inventory['original_gated_measured_states']:
        state_key = tuple(row[:4])
        reference_keys.add(state_key)
        if state_key in expected:
            assert expected[state_key][:2] == row[4:6]
            expected[state_key][3].update(row[6])
        else:
            expected[state_key] = [row[4], row[5], set(), set(row[6])]
    assert len(reference_keys) == 368
    assert len(reference_keys & grid_keys) == 37
    assert len(expected) == 1698
    with (directory / 'states.jsonl').open() as stream:
        records = [json.loads(line) for line in stream]
    lookup = {key(record): record for record in records}
    assert len(records) == len(lookup) == 1698 and set(lookup) == set(expected)
    assert len(selection['states']) == 1698
    selected = {key(record): record for record in selection['states']}
    assert len(selected) == len(selection['states']) and set(selected) == set(expected)
    for state_key, (xy, qualified, windows, samples) in expected.items():
        record, chosen = lookup[state_key], selected[state_key]
        for entry in (record, chosen):
            assert entry['actual_source_xy'] == xy and entry['qualified_moving'] == qualified
            assert set(entry['grid_windows']) == windows
            assert set(entry['reference_samples']) == samples
        assert record['save_audit_inputs'] == chosen['save_audit_inputs']
        if record['audit_inputs'] is not None:
            assert record['save_audit_inputs'] and record['geometry']['available']
            path = str(directory / record['audit_inputs']['path'])
            assert receipt['files_sha256'][path] == record['audit_inputs']['sha256']
        else:
            assert not record['save_audit_inputs'] or not record['geometry']['available']
    assert summary['totals'] == counts(records)
    assert summary['grid'] == counts([lookup[k] for k in grid_keys])
    assert summary['grid_strict'] == counts([lookup[k] for k in grid_keys if lookup[k]['qualified_moving']])
    assert summary['by_clip'] == {c: counts([r for r in records if r['clip'] == c]) for c in CLIPS}
    assert summary['production_additions'] == summary['production_removals'] == 0
    assert summary['no_filter_applied'] is True and summary['airborne_accuracy_established'] is False
    assert summary['reference_panels'] == inventory['panels'] == reference['panels']
    for clip in CLIPS:
        wanted = sorted({i for k in expected if k[0] == clip for i in range(k[1] - 8, k[1] + 1)})
        assert wanted == selection['needed_source_frames'][clip]
        metadata = summary['source_metadata'][clip]
        assert metadata['retrieved_indices'] == wanted
        assert metadata['actual_sequential_frames'] == inventory['source_scopes'][clip]['declared_frame_count']
        assert metadata['eof_verified'] and metadata['max_retained_gray_frames'] == 9
        assert metadata['evaluated_states'] == sum(k[0] == clip for k in expected)
    assert len(reference['samples']) == len(inventory['samples']) == 356
    sample_outcomes, per_panel = [], {}
    for i, raw in enumerate(inventory['samples']):
        original = dict(zip(inventory['sample_columns'], raw))
        observed = reference['samples'][i]
        assert observed['sample_index'] == i and observed['original'] == original
        actual_ids = original['stages']['actual_measurement'][2]
        assert [a['identity'] for a in observed['measured_alternatives']] == actual_ids
        alternatives = []
        for ident, actual in zip(actual_ids, observed['measured_alternatives']):
            seg, tid = ident.split('/', 1)
            record = lookup[(original['clip_id'], original['frame_index'], int(seg), tid)]
            for field in ('qualified_moving', 'geometry', 'localized', 'predicted_to_actual_distance_px'):
                assert actual[field] == record[field]
            alternatives.append(outcome(record))
        sample_outcomes.append(dict(sample_index=i, panel=original['panel'], clip=original['clip_id'],
            window=original['window_id'], frame=original['frame_index'],
            original_actual_hit=original['stages']['actual_measurement'][0],
            alternatives=alternatives))
    for panel, old in inventory['panels'].items():
        chosen = [s for s in sample_outcomes if s['panel'] == panel]
        assert len(chosen) == old['samples']
        alternatives = [a for s in chosen for a in s['alternatives']]
        all_available = lambda s: bool(s['alternatives']) and all(a['localized_available'] for a in s['alternatives'])
        all_unambiguous = lambda s: all_available(s) and all(a['ambiguous'] is False for a in s['alternatives'])
        strict_assigned = []
        for sample in chosen:
            raw = inventory['samples'][sample['sample_index']]
            assigned = raw[8][old['original_strict_stage']][1]
            if assigned is not None:
                segment, track_id = assigned.split('/', 1)
                strict_assigned.append(outcome(lookup[(raw[1], raw[3], int(segment), track_id)]))
        per_panel[panel] = dict(samples=old['samples'], original_scores=old,
            actual_alternative_counts=outcome_counts(alternatives),
            original_strict_assigned_outcomes=outcome_counts(strict_assigned),
            samples_without_original_actual_measurement=sum(not s['alternatives'] for s in chosen),
            samples_all_alternatives_localized_available=sum(all_available(s) for s in chosen),
            samples_all_alternatives_available_unambiguous=sum(all_unambiguous(s) for s in chosen),
            unavailable_or_ambiguous_alternatives_are_not_rejections=True,
            no_best_alternative_or_best_score_selection=True)
    known = []
    for frame in (346, 347):
        record = lookup[('0029', frame, 0, 'bright:2641')]
        grid_samples = [raw for raw in inventory['samples'] if raw[0] == 'grid' and raw[1] == '0029' and raw[3] == frame]
        assert len(grid_samples) == 1
        assert grid_samples[0][8]['strict_qualified_measurement'][2] == ['0/bright:2641']
        known.append(dict(**outcome(record), original_grid_strict_gated_ids=['0/bright:2641'],
            localized_full=record['localized']))
    for path, digest in checked.items():
        assert sha(path) == digest, path
    result = dict(schema='seaqr.accuracy-v42-independent-accounting-audit.v1', verified=True,
        completed_at_utc=datetime.now(timezone.utc).isoformat(), experiment=str(directory),
        audited_states=1698, grid_actual_states=1367, grid_strict_states=381,
        original_reference_unique_states=368, reference_grid_overlap=37,
        all_reference_samples_preserved=356, all_gated_alternatives_preserved=True,
        original_misses_and_stage_assignments_unchanged=True,
        all_completed_and_frozen_hashes_recomputed=True, checked_files_sha256=checked,
        independently_recounted_totals=counts(records), per_panel=per_panel,
        known_v41_counterexamples=known, sample_outcomes=sample_outcomes,
        source_metadata_checked_against_frozen_selection=True,
        native_decode_reexecuted=False, source_media_bytes_hashed_only=True,
        no_source_media_decoded=True, no_model_or_producer_imports=True,
        numerical_model_audit_not_performed_here=True, no_new_labels=True,
        no_filter_applied=True, no_accuracy_gain_claimed=True,
        interpretation='Availability and ambiguity are evidence bookkeeping, not acceptance, rejection, airborne truth, false-positive reduction, or independent generalization.')
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({k: result[k] for k in ('verified', 'audited_states', 'per_panel')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=BASE / 'localized_01')
    parser.add_argument('--output', type=Path, default=BASE / 'accounting_independent_audit_01.json')
    args = parser.parse_args()
    run(args.run, args.output)
