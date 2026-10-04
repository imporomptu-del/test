"""Freeze, cache all V42 inputs, then evaluate the four offline V43 arms."""
import argparse
from collections import Counter, defaultdict, deque
import json
from pathlib import Path
import platform
import shutil

import cv2
import numpy as np

from accuracy_v42_history import prepare_history
import run_accuracy_v42_localized as old
from accuracy_v43_localized import ARMS, evaluate_arm

ROOT = old.ROOT
BASELINE = old.V42/'localized_01'
RECEIPT_SHA = '9077ba90861a05bea591ff4961c818755002e9f11a28d06a7d6a1e86854c3c46'
ARRAY_KEYS = ('current129', 'history129', 'prior_centers_xy', 'predicted_offset_xy')
NEW_FILES = ('docs/accuracy_v43_plan.md', 'scripts/run_accuracy_v43_verifier.py',
    'scripts/accuracy_v43_localized.py', 'scripts/accuracy_v43_bounds.py',
    'scripts/accuracy_v43_components.py', 'scripts/accuracy_v43_stable_fit.py',
    'tests/unit/test_accuracy_v43_localized.py', 'tests/unit/test_accuracy_v43_bounds.py',
    'tests/unit/test_accuracy_v43_components.py', 'tests/unit/test_accuracy_v43_stable_fit.py',
    'tests/unit/test_accuracy_v43_runner.py')


def key(record):
    return record['clip'], record['frame_index'], record['segment'], record['track_id']


def causal_rows(rows):
    """Original measured journal schema; no current-track fields can cross."""
    output = []
    for index, row in enumerate(rows):
        item = {k: row[k] for k in ('frame_index', 'timestamp_ns', 'segment', 'source_to_reference')}
        item['motion'] = {'reset': row['motion']['reset']}
        if index < len(rows)-1:
            item['tracks'] = []
            for track in row['tracks']:
                if type(track['measured']) is not bool or track['segment'] != row['segment']:
                    raise ValueError('Malformed journal measurement')
                item['tracks'].append(dict(track_id=track['track_id'], predicted=not track['measured'],
                                           measurement_source_xy=track['measurement_source_xy']))
        output.append(item)
    return output


def counts(records, arm):
    result = Counter(states=0, strict_states=0, geometry_available=0, score_available=0,
                     score_with_ambiguity=0, score_without_recorded_ambiguity=0)
    for record in records:
        result['states'] += 1
        result['strict_states'] += int(record['qualified_moving'])
        result['geometry_available'] += int(record['geometry']['available'])
        value = record['arms'][arm]
        if value is None:
            result['geometry_unavailable'] += 1
            continue
        result['score_available'] += int(value['available'])
        result['score_with_ambiguity'] += int(value['available'] and value['ambiguous'])
        result['score_without_recorded_ambiguity'] += int(value['available'] and not value['ambiguous'])
        for reason in value['reasons']: result['unavailable:'+reason] += 1
        for reason in value['ambiguity_reasons']: result['ambiguity:'+reason] += 1
    return dict(result)


def paired(records, first, second):
    result = Counter(states=len(records))
    changes = []
    for record in records:
        a, b = record['arms'][first], record['arms'][second]
        aa, ba = a is not None and a['available'], b is not None and b['available']
        result[f'available_{int(aa)}_{int(ba)}'] += 1
        if aa and ba:
            same = a['common_support_sha256'] == b['common_support_sha256']
            result['both_available_same_support' if same else 'both_available_different_support'] += 1
            if same:
                changes.append(b['advantage_stationary_minus_augmented']-a['advantage_stationary_minus_augmented'])
    return dict(first=first, second=second, counts=dict(result),
        same_support_advantage_change_quantiles=None if not changes else np.quantile(changes, [0,.5,1]).tolist(),
        changed_support_mse_not_compared=True, improvement_is_not_accuracy=True)


def reference_report(inventory, records):
    lookup = {key(r): r for r in records}
    samples, panels = [], {}
    for index, raw in enumerate(inventory['samples']):
        sample = dict(zip(inventory['sample_columns'], raw))
        alternatives = []
        for identity in sample['stages']['actual_measurement'][2]:
            segment, track = identity.split('/', 1)
            record = lookup[(sample['clip_id'], sample['frame_index'], int(segment), track)]
            alternatives.append(dict(identity=identity, qualified_moving=record['qualified_moving'],
                geometry_available=record['geometry']['available'], arms=record['arms']))
        samples.append(dict(sample_index=index, original=sample, measured_alternatives=alternatives))
    for panel, original in inventory['panels'].items():
        subset = [s for s in samples if s['original']['panel'] == panel]
        panel_arms = {}
        for arm in ARMS:
            c = Counter(samples=len(subset), original_strict_assignment=0, same_assignment_score_available=0,
                        same_assignment_score_without_recorded_ambiguity=0)
            for sample in subset:
                identity = sample['original']['stages'][original['original_strict_stage']][1]
                if identity is None: continue
                c['original_strict_assignment'] += 1
                matches = [a for a in sample['measured_alternatives'] if a['identity'] == identity]
                if len(matches) != 1: raise ValueError('Lost original strict identity')
                value = matches[0]['arms'][arm]
                if value is not None and value['available']:
                    c['same_assignment_score_available'] += 1
                    c['same_assignment_score_without_recorded_ambiguity'] += int(not value['ambiguous'])
            panel_arms[arm] = dict(c)
        panels[panel] = dict(original=original, arms=panel_arms)
    return dict(samples=samples, panels=panels, overlapping_samples_not_independent=True,
                no_best_alternative_or_arm_selection=True, original_scoring_unchanged=True)


def run(output):
    output = Path(output).resolve()
    if output.exists(): raise FileExistsError('Fresh output required')
    if shutil.disk_usage(output.parent if output.parent.exists() else ROOT).free < 2*1024**3:
        raise ValueError('Require two GiB headroom before caching')
    bound = {}
    def bind(path, expected=None):
        path = str(Path(path).resolve()); value = old.sha(path)
        if expected is not None and value != expected: raise ValueError('Hash mismatch: '+path)
        if path in bound and bound[path] != value: raise ValueError('Changed input: '+path)
        bound[path] = value
    bind(BASELINE/'completion_receipt.json', RECEIPT_SHA)
    receipt = old.read(BASELINE/'completion_receipt.json')
    for path, value in receipt['files_sha256'].items(): bind(path, value)
    for name in NEW_FILES: bind(ROOT/name)
    selection, inventory = old.read(BASELINE/'selection.json'), old.read(old.INVENTORY)
    baseline = [json.loads(line) for line in (BASELINE/'states.jsonl').read_text().splitlines()]
    previous = {key(r): r for r in baseline}
    if len(previous) != 1698 or len(baseline) != 1698 or {key(s) for s in selection['states']} != set(previous):
        raise ValueError('Wrong frozen state union')
    output.mkdir(parents=True); (output/'inputs').mkdir()
    old.write(output/'freeze.json', dict(created_at_utc=old.now(), inputs_sha256=bound.copy(),
        pre_native_decode_and_score=True, arms=ARMS, production_changed=False,
        schema='seaqr.accuracy-v43-freeze.v1',
        versions=dict(python=platform.python_version(), numpy=np.__version__, opencv=cv2.__version__)))
    bind(output/'freeze.json')
    cached, metadata = [], {}
    for clip in old.CLIPS:
        specification = inventory['source_scopes'][clip]
        needed = set(selection['needed_source_frames'][clip])
        by_frame = defaultdict(list)
        for state in selection['states']:
            if state['clip'] == clip: by_frame[state['frame_index']].append(state)
        journal, frame_count = {}, 0
        with Path(specification['journal_path']).open() as stream:
            for index, line in enumerate(stream):
                row = json.loads(line)
                if row['frame_index'] != index: raise ValueError('Noncontiguous journal')
                if index in needed: journal[index] = row
                frame_count += 1
        if frame_count != specification['declared_frame_count'] or set(journal) != needed:
            raise ValueError('Incomplete journal')
        cap = cv2.VideoCapture(str(old.SOURCES[clip]))
        if not cap.isOpened(): raise ValueError('Cannot open explicit source')
        native = [cap.get(p) for p in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT,
                                       cv2.CAP_PROP_FRAME_COUNT, cv2.CAP_PROP_FPS)]
        if native != [4784, 3190, frame_count, 10]:
            cap.release(); raise ValueError('Changed native source metadata')
        buffer = deque(maxlen=9); retrieved, states_count, packet_count = [], 0, 0
        try:
            if cap.get(cv2.CAP_PROP_POS_FRAMES) != 0: raise ValueError('Decoder must start at zero')
            for index in range(frame_count):
                if len(buffer) == 9: buffer.popleft()
                if not cap.grab() or cap.get(cv2.CAP_PROP_POS_FRAMES) != index+1:
                    raise ValueError('Incomplete sequential decode')
                gray = None
                if index in needed:
                    ok, image = cap.retrieve()
                    if not ok or image.shape != (3190,4784,3) or image.dtype != np.uint8:
                        raise ValueError('Invalid native decoded frame')
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY); retrieved.append(index)
                buffer.append((index, gray))
                if index not in by_frame: continue
                if len(buffer) != 9 or any(frame is None for _,frame in buffer):
                    raise ValueError('Missing requested history')
                images = [frame for _,frame in buffer]
                rows = causal_rows([journal[i] for i,_ in buffer])
                for state in by_frame[index]:
                    geom = prepare_history(images, rows, state['segment'], state['track_id'])
                    meta = old.clean({k:v for k,v in geom.items() if k not in ARRAY_KEYS})
                    prior = previous[key(state)]
                    if meta != prior['geometry']: raise ValueError('V42 geometry did not reproduce')
                    item = dict(**state, geometry=meta, archive=None)
                    if geom['available']:
                        if prior['audit_inputs']:
                            with np.load(BASELINE/prior['audit_inputs']['path'], allow_pickle=False) as saved:
                                for name in ARRAY_KEYS: np.testing.assert_equal(geom[name], saved[name])
                        path = output/'inputs'/f'{clip}_{index:04d}_{state["segment"]}_{state["track_id"].replace(":","_")}.npz'
                        np.savez_compressed(path, **{name:geom[name] for name in ARRAY_KEYS})
                        bind(path)
                        item['archive'] = dict(path='inputs/'+path.name, sha256=bound[str(path)])
                        packet_count += 1
                    cached.append(item); states_count += 1
                del images, rows
            if cap.grab(): raise ValueError('Extra source frame')
        finally:
            cap.release()
        if set(retrieved) != needed: raise ValueError('Missing native retrieval')
        metadata[clip] = dict(sequential_frames=frame_count, retrieved_indices=retrieved,
            states=states_count, packets=packet_count, eof_verified=True, max_retained_native_gray_frames=9)
        print(f'Cached {clip}: {states_count} states / {packet_count} inputs', flush=True)
    if len(cached) != 1698 or sum(c['archive'] is not None for c in cached) != 605:
        raise ValueError('Cache scope differs from V42')
    old.write(output/'cache_manifest.json', dict(states=cached, source_metadata=metadata,
        cache_completed_at_utc=old.now(),
        completed_before_any_arm_score=True, all_geometry_exactly_reproduced=True,
        all_seven_v42_saved_inputs_exactly_reproduced=True))
    bind(output/'cache_manifest.json')
    for path, value in bound.items():
        if old.sha(path) != value: raise ValueError('Input changed before score: '+path)
    old.write(output/'score_start.json', dict(started_at_utc=old.now(),
        cache_manifest_sha256=old.sha(output/'cache_manifest.json'),
        pre_score_files_sha256=bound.copy(), all_bound_files_rehashed_before_scoring=True))
    bind(output/'score_start.json')
    print('Entire 605-input cache frozen; starting all four arms', flush=True)
    results = []
    with (output/'states.jsonl').open('x') as stream:
        for index, item in enumerate(cached):
            record = dict(item, arms={name:None for name in ARMS})
            if item['archive'] is not None:
                with np.load(output/item['archive']['path'], allow_pickle=False) as packet:
                    centers = [p.tolist() if np.isfinite(p).all() else None for p in packet['prior_centers_xy']]
                    for arm in ARMS:
                        value = evaluate_arm(packet['current129'], packet['history129'], centers,
                            packet['predicted_offset_xy'], item['track_id'].split(':')[0], arm)
                        # No quiet conversion of infinities into missing metrics.
                        encoded = json.dumps(value, default=lambda v:v.tolist(), allow_nan=False)
                        record['arms'][arm] = json.loads(encoded)
                if record['arms']['baseline'] != previous[key(item)]['localized']:
                    raise ValueError('Frozen V42 baseline score did not reproduce exactly')
            elif previous[key(item)]['localized'] is not None:
                raise ValueError('Missing baseline localized evidence')
            stream.write(json.dumps(record, allow_nan=False)+'\n'); results.append(record)
            if index % 200 == 0: print(f'Evaluated {index+1}/1698 states', flush=True)
    references = reference_report(inventory, results)
    groups = dict(all=results, strict=[r for r in results if r['qualified_moving']],
        grid=[r for r in results if r['grid_windows']],
        grid_strict=[r for r in results if r['grid_windows'] and r['qualified_moving']])
    summary = dict(completed=True, created_at_utc=old.now(),
        groups={group:{arm:counts(rows,arm) for arm in ARMS} for group, rows in groups.items()},
        paired=[paired(results,'baseline',arm) for arm in ARMS[1:]],
        reference_panels=references['panels'], baseline_exact_reproduction=True,
        production_changed=False, no_filter_applied=True, no_new_labels=True,
        conditional_sensitivity_not_calibrated_airborne_confidence=True)
    old.write(output/'summary.json', summary); old.write(output/'reference_evidence.json', references)
    for name in ('states.jsonl','summary.json','reference_evidence.json'): bind(output/name)
    for path, value in bound.items():
        if old.sha(path) != value: raise ValueError('Bound file changed: '+path)
    old.write(output/'completion_receipt.json', dict(completed=True, created_at_utc=old.now(),
        files_sha256=bound, all_inputs_and_outputs_rehashed=True, production_changed=False))
    print(json.dumps(summary['groups'], indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute: parser.error('Explicit --execute required after preflight/tests')
    run(args.output)
