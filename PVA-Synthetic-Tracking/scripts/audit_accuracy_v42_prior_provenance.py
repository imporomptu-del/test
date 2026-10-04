"""Independently bind every serialized V42 prior measurement to original journals."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/tiny_target/accuracy_v42_20260925'
INVENTORY_SHA = '444df960282569ef4c60a80a15bc6a50756fe0eadba421969061f6fb5aab86df'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            value.update(block)
    return value.hexdigest()


def run(directory, output):
    directory, output = Path(directory).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Fresh supplemental audit output required')
    checked = {}

    def bind(path, expected=None):
        path = str(Path(path).resolve())
        digest = sha(path)
        if expected is not None:
            assert digest == expected, path
        checked[path] = digest

    receipt_path = directory / 'completion_receipt.json'
    bind(receipt_path)
    receipt = read(receipt_path)
    assert receipt['completed'] and receipt['all_inputs_and_outputs_rehashed']
    for name in ('states.jsonl', 'selection.json', 'freeze.json'):
        path = directory / name
        bind(path, receipt['files_sha256'][str(path)])
    inventory_path = BASE / 'reference_inventory.json'
    bind(inventory_path, INVENTORY_SHA)
    bind(__file__)
    inventory, selection = read(inventory_path), read(directory / 'selection.json')
    freeze = read(directory / 'freeze.json')
    journals = {}
    for clip, specification in inventory['source_scopes'].items():
        path = Path(specification['journal_path'])
        digest = specification['journal_sha256']
        assert freeze['inputs_sha256'][str(path)] == receipt['files_sha256'][str(path)] == digest
        bind(path, digest)
        needed = set(selection['needed_source_frames'][clip])
        rows = {}
        count = 0
        with path.open() as stream:
            for index, line in enumerate(stream):
                row = json.loads(line)
                assert row['frame_index'] == index
                count += 1
                if index not in needed:
                    continue
                tracks = {}
                for track in row['tracks']:
                    state = (track['segment'], track['track_id'])
                    assert state not in tracks
                    assert type(track['measured']) is bool
                    assert track['measured'] == (track['measurement_source_xy'] is not None)
                    tracks[state] = (track['measured'], track['measurement_source_xy'])
                rows[index] = dict(frame_index=index, timestamp_ns=row['timestamp_ns'],
                    segment=row['segment'], reset=row['motion']['reset'],
                    source_to_reference=row['source_to_reference'], tracks=tracks)
        assert count == specification['declared_frame_count'] and set(rows) == needed
        journals[clip] = rows
    totals = Counter()
    prior_count_histogram = Counter()
    max_reference_error = max_warp_error = 0.0
    evidence = []
    seen = set()
    with (directory / 'states.jsonl').open() as stream:
        for line in stream:
            record = json.loads(line)
            clip, frame, segment, tid = record['clip'], record['frame_index'], record['segment'], record['track_id']
            state_key = (clip, frame, segment, tid)
            assert state_key not in seen
            seen.add(state_key)
            outer, geometry = record['geometry'], record['geometry']['geometry']
            current = journals[clip][frame]
            priors = [journals[clip][i] for i in range(frame - 8, frame)]
            assert geometry['current_frame_index'] == frame == current['frame_index']
            assert geometry['current_timestamp_ns'] == current['timestamp_ns'] == frame * 100000000
            assert geometry['prior_frame_indices'] == [r['frame_index'] for r in priors]
            assert geometry['prior_timestamps_ns'] == [r['timestamp_ns'] for r in priors]
            assert geometry['segment'] == segment and geometry['track_id'] == tid
            assert geometry['origins_xy'] == [[0.0, 0.0]] * 9
            assert geometry['current_target_fields_accessed'] is False
            assert geometry['current_global_transform_uses_current_whole_frame'] is True
            assert geometry['required_measured_prior_count'] == 5
            for offset, prior in enumerate(priors, start=-8):
                assert prior['frame_index'] == frame + offset
                assert prior['timestamp_ns'] == current['timestamp_ns'] + offset * 100000000
                assert prior['segment'] == segment and prior['reset'] is False
            assert current['segment'] == segment and current['reset'] is False
            selected_current = current['tracks'][(segment, tid)]
            assert selected_current[0] is True and selected_current[1] == record['actual_source_xy']
            actual = [(r, r['tracks'][(segment, tid)][1]) for r in priors
                      if (segment, tid) in r['tracks'] and r['tracks'][(segment, tid)][0]]
            assert len(actual) == geometry['measured_prior_count'] == len(geometry['prior_measurements'])
            for (prior, xy), serialized in zip(actual, geometry['prior_measurements']):
                assert serialized['frame_index'] == prior['frame_index']
                assert serialized['timestamp_ns'] == prior['timestamp_ns']
                assert serialized['source_xy'] == xy
                assert serialized['source_to_reference'] == prior['source_to_reference']
                projected = np.asarray(prior['source_to_reference']) @ np.array([*xy, 1.])
                reference_xy = projected[:2] / projected[2]
                error = float(np.max(np.abs(reference_xy - serialized['reference_xy'])))
                assert error <= 1e-10
                max_reference_error = max(max_reference_error, error)
                totals['prior_measurement_entries_verified'] += 1
            if outer['available']:
                assert len(actual) >= 5 and not outer['reasons']
                assert geometry['current_source_to_reference'] == current['source_to_reference']
                assert len(geometry['current_to_prior_matrices']) == 8
                for prior, saved_warp in zip(priors, geometry['current_to_prior_matrices']):
                    independent = np.linalg.inv(np.asarray(prior['source_to_reference'])) @ np.asarray(current['source_to_reference'])
                    error = float(np.max(np.abs(independent - saved_warp)))
                    assert error <= 1e-10
                    max_warp_error = max(max_warp_error, error)
                    totals['current_to_prior_matrices_verified'] += 1
                totals['available_current_H_exactly_verified'] += 1
            else:
                assert len(actual) < 5
                assert outer['reasons'] == ['fewer_than_five_prior_actual_same_id_measurements']
                assert 'current_source_to_reference' not in geometry
                assert 'current_to_prior_matrices' not in geometry
                assert record['localized'] is None
                totals['unavailable_before_current_H_serialization'] += 1
            totals['states'] += 1
            prior_count_histogram[len(actual)] += 1
            evidence.append(dict(state_key=list(state_key), geometry_available=outer['available'],
                measured_prior_count=len(actual), actual_prior_frame_indices=[r['frame_index'] for r, xy in actual],
                exact_original_prior_positions_and_H=True,
                current_H_recorded_and_exactly_verified=outer['available']))
    assert totals['states'] == 1698
    assert totals['available_current_H_exactly_verified'] == 605
    assert totals['unavailable_before_current_H_serialization'] == 1093
    assert seen == {(r['clip'], r['frame_index'], r['segment'], r['track_id']) for r in selection['states']}
    for path, digest in checked.items():
        assert sha(path) == digest, path
    result = dict(schema='seaqr.accuracy-v42-independent-prior-provenance-audit.v1', verified=True,
        completed_at_utc=datetime.now(timezone.utc).isoformat(), experiment=str(directory),
        checked_files_sha256=checked, all_bound_files_rehashed=True, counts=dict(totals),
        prior_count_histogram=dict(prior_count_histogram),
        maximum_reference_coordinate_error_px=max_reference_error,
        maximum_current_to_prior_matrix_element_error=max_warp_error,
        all1698_exact_segment_and_ID_histories_checked=True,
        current_H_caveat='Current H is serialized and exactly matches the original current journal in all605 geometry-available states. In1093 short-history states it is intentionally absent because history processing returned before forecast/geometry serialization; their original current journal H is hash-bound, but this artifact does not claim comparison to an absent field.',
        no_current_or_future_measurements_in_serialized_prior_lists=True,
        unknown_geometry_states_retained=True,
        every_recorded_prior_source_coordinate_and_H_matches_original_exactly=True,
        per_state_evidence=evidence, source_media_decoded=False, source_media_opened=False,
        producer_or_model_functions_imported=False, no_frozen_artifacts_changed=True)
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({k: result[k] for k in ('verified', 'counts', 'prior_count_histogram',
          'maximum_reference_coordinate_error_px', 'maximum_current_to_prior_matrix_element_error')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=BASE / 'localized_01')
    parser.add_argument('--output', type=Path, default=BASE / 'prior_provenance_audit_01.json')
    arguments = parser.parse_args()
    run(arguments.run, arguments.output)
