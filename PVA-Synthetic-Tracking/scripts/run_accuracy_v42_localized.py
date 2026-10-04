"""Frozen offline native-AVI replay of a causal localized-source evidence stage.

No production detector/tracker is rerun or changed. Selection uses saved actual
states; current positions and labels are not inputs to the evidence model.
"""
import argparse
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform

import cv2
import numpy as np

from accuracy_v42_history import prepare_history
from accuracy_v42_localized import evaluate_localized

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'results/tiny_target'
V42 = BASE/'accuracy_v42_20260925'
INVENTORY = V42/'reference_inventory.json'
INVENTORY_SHA = '444df960282569ef4c60a80a15bc6a50756fe0eadba421969061f6fb5aab86df'
GRID = BASE/'accuracy_v40_20260925/coverage_workload_01/window_evidence.json'
GRID_SHA = 'cefb4ccc97a894d16a02c5a11c9ce4696f4b49a279a5df22477fcd175a7b87a3'
CLIPS = ('0029', '0126', '0055', '0082')
SOURCES = {
    '0029': ROOT.parent/'outputs/jetson_review_clips_20260913/chunk_0029.avi',
    '0126': ROOT.parent/'outputs/jetson_review_clips_20260913/chunk_0126.avi',
    '0055': ROOT.parent/'outputs/v7_frozen_evaluation_20260913/sources/chunk_0055.avi',
    '0082': ROOT.parent/'outputs/v7_frozen_evaluation_20260913/sources/chunk_0082.avi',
}


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            result.update(chunk)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def now():
    return datetime.now(timezone.utc).isoformat()


def make_selection(inventory, grid):
    selected = {}

    def add(key, xy, qualified, window=None, samples=()):
        if key not in selected:
            selected[key] = dict(clip=key[0], frame_index=key[1], segment=key[2], track_id=key[3],
                actual_source_xy=xy, qualified_moving=qualified, grid_windows=[], reference_samples=[])
        item = selected[key]
        if item['actual_source_xy'] != xy or item['qualified_moving'] != qualified:
            raise ValueError('Disagreeing saved state provenance')
        if window is not None and window not in item['grid_windows']:
            item['grid_windows'].append(window)
        item['reference_samples'] = sorted(set(item['reference_samples']) | set(samples))

    grid_count = strict_count = 0
    for window in grid:
        for row in window['frames']:
            for track in row['actual_measurements']:
                key = (window['clip_id'], row['frame_index'], track['segment'], track['track_id'])
                add(key, track['source_xy'], track['qualified_moving'], window['window_id'])
                grid_count += 1
                strict_count += int(track['qualified_moving'])
    if (len(grid), grid_count, strict_count) != (108, 1367, 381):
        raise ValueError('Wrong grid denominator')
    for raw in inventory['original_gated_measured_states']:
        add(tuple(raw[:4]), raw[4], raw[5], samples=raw[6])
    keys = sorted(selected)
    audited = set()
    for clip in CLIPS:
        audited.update([key for key in keys if key[0] == clip][:2])
    for panel in inventory['panels']:
        audited.update([key for key in keys if any(inventory['samples'][i][0] == panel
                       for i in selected[key]['reference_samples'])][:2])
    audited.update(key for key in keys if key[0] == '0029' and key[1] in (346, 347))
    states = []
    for key in keys:
        item = selected[key]
        item['save_audit_inputs'] = key in audited
        states.append(item)
    return dict(states=states, unique_states=len(states), grid_actual_states=grid_count,
        grid_strict_states=strict_count, grid_windows=108,
        reference_unique_states=len(inventory['original_gated_measured_states']),
        reference_panel_denominators={k:v['samples'] for k,v in inventory['panels'].items()},
        audit_state_keys=[list(key) for key in sorted(audited)],
        needed_source_frames={clip:sorted({i for key in keys if key[0] == clip
            for i in range(max(0, key[1]-8), key[1]+1)}) for clip in CLIPS},
        no_ground_truth_coordinates_enter_inference=True)


def clean(value):
    """Arrays outside inference-input archives must be finite JSON metadata."""
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, dict):
        return {k:clean(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def run_state(state, images, rows, output):
    # Deliberately provide no current measurement or reference position to either
    # inference function. A poison-current-track unit test guards this boundary.
    causal_rows = []
    for index, row in enumerate(rows):
        item = {key: row[key] for key in
                ('frame_index', 'timestamp_ns', 'segment', 'source_to_reference')}
        item['motion'] = {'reset': row['motion']['reset']}
        if index < len(rows)-1:
            item['tracks'] = []
            for track in row['tracks']:
                if type(track['measured']) is not bool or track['segment'] != row['segment']:
                    raise ValueError('Malformed original journal measurement state')
                item['tracks'].append(dict(track_id=track['track_id'], predicted=not track['measured'],
                    measurement_source_xy=track['measurement_source_xy']))
        causal_rows.append(item)
    geometry = prepare_history(images, causal_rows, state['segment'], state['track_id'])
    metadata = {k:v for k,v in geometry.items() if k not in
                ('current129','history129','prior_centers_xy','predicted_offset_xy')}
    result = dict(**state, geometry=clean(metadata), localized=None,
                  predicted_to_actual_distance_px=None, audit_inputs=None)
    if not geometry['available']:
        return result
    centers = [None if not np.isfinite(point).all() else list(map(float, point))
               for point in geometry['prior_centers_xy']]
    result['localized'] = clean(evaluate_localized(geometry['current129'], geometry['history129'],
        centers, geometry['predicted_offset_xy'], state['track_id'].split(':')[0]))
    # Current actual coordinates are read only AFTER inference for diagnostics.
    predicted = geometry['geometry']['predicted_source_xy']
    result['predicted_to_actual_distance_px'] = math.dist(predicted, state['actual_source_xy'])
    if state['save_audit_inputs']:
        name = f'{state["clip"]}_{state["frame_index"]:04d}_{state["segment"]}_{state["track_id"].replace(":","_")}.npz'
        path = output/'audit_inputs'/name
        np.savez_compressed(path, current129=geometry['current129'], history129=geometry['history129'],
            prior_centers_xy=geometry['prior_centers_xy'], predicted_offset_xy=geometry['predicted_offset_xy'])
        result['audit_inputs'] = dict(path='audit_inputs/'+name, sha256=sha(path),
            role='Deterministically preselected numerical-audit inputs, not new ground truth.')
    return result


def state_summary(records):
    counts = Counter(states=0, strict_states=0, geometry_available=0, localized_available=0,
                     ambiguous=0)
    for r in records:
        counts['states'] += 1
        counts['strict_states'] += int(r['qualified_moving'])
        counts['geometry_available'] += int(r['geometry']['available'])
        for reason in r['geometry']['reasons']:
            counts['geometry_reason:'+reason] += 1
        if r['localized'] is not None:
            value = r['localized']
            counts['localized_available'] += int(value['available'])
            counts['ambiguous'] += int(value['ambiguous'])
            for reason in value['reasons']:
                counts['localized_reason:'+reason] += 1
            for reason in value['ambiguity_reasons']:
                counts['ambiguity_reason:'+reason] += 1
    return dict(counts)


def reference_report(inventory, records):
    lookup = {(r['clip'], r['frame_index'], r['segment'], r['track_id']):r for r in records}
    samples = []
    for i, raw in enumerate(inventory['samples']):
        sample = dict(zip(inventory['sample_columns'], raw))
        alternatives = []
        for identity in sample['stages']['actual_measurement'][2]:
            segment, track_id = identity.split('/', 1)
            record = lookup[(sample['clip_id'], sample['frame_index'], int(segment), track_id)]
            alternatives.append(dict(identity=identity, qualified_moving=record['qualified_moving'],
                geometry=record['geometry'], localized=record['localized'],
                predicted_to_actual_distance_px=record['predicted_to_actual_distance_px']))
        samples.append(dict(sample_index=i, original=sample, measured_alternatives=alternatives))
    return dict(panels=inventory['panels'], samples=samples,
        original_scoring_unchanged=True, no_filter_applied=True,
        physical_class_or_identity_established=False,
        missing_original_states_not_dropped=True)


def run(output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Fresh output required; frozen results are never overwritten')
    bound = {}

    def bind(path, expected=None):
        path = str(Path(path).resolve())
        value = sha(path)
        if expected is not None and value != expected:
            raise ValueError('Hash mismatch '+path)
        if path in bound and value != bound[path]:
            raise ValueError('Input changed '+path)
        bound[path] = value

    bind(INVENTORY, INVENTORY_SHA); bind(GRID, GRID_SHA)
    inventory, grid = read(INVENTORY), read(GRID)
    for path, value in inventory['inputs_sha256'].items():
        bind(path, value)
    for clip in CLIPS:
        bind(SOURCES[clip], inventory['source_scopes'][clip]['source_sha256_inherited_not_media_recomputed'])
    for name in ('docs/accuracy_v42_plan.md', 'scripts/run_accuracy_v42_localized.py',
                 'scripts/accuracy_v42_history.py', 'scripts/accuracy_v42_localized.py',
                 'scripts/accuracy_v41_geometry.py', 'scripts/accuracy_v38_source_pairs.py',
                 'tests/unit/test_accuracy_v42_history.py', 'tests/unit/test_accuracy_v42_localized.py',
                 'tests/unit/test_accuracy_v42_runner.py'):
        bind(ROOT/name)
    selection = make_selection(inventory, grid)
    output.mkdir(parents=True); (output/'audit_inputs').mkdir()
    write(output/'selection.json', selection); bind(output/'selection.json')
    write(output/'freeze.json', dict(schema='seaqr.accuracy-v42-freeze.v1', created_at_utc=now(),
        inputs_sha256=bound.copy(), pre_native_decode_and_score=True,
        version=dict(python=platform.python_version(), numpy=np.__version__, opencv=cv2.__version__),
        media_allowlist={k:str(v) for k,v in SOURCES.items()}, selection_sha256=sha(output/'selection.json'),
        production_change=False, exposed_development=True))
    bind(output/'freeze.json')
    print('Frozen before decode: '+str(output)+'; selected states '+str(selection['unique_states']), flush=True)
    all_results, source_metadata = [], {}
    with (output/'states.jsonl').open('x') as destination:
        for clip in CLIPS:
            specification = inventory['source_scopes'][clip]
            needed = set(selection['needed_source_frames'][clip])
            by_frame = defaultdict(list)
            for item in selection['states']:
                if item['clip'] == clip:
                    by_frame[item['frame_index']].append(item)
            journal = {}
            count = 0
            with Path(specification['journal_path']).open() as stream:
                for index, line in enumerate(stream):
                    row = json.loads(line)
                    if row['frame_index'] != index:
                        raise ValueError('Noncontiguous journal')
                    if index in needed:
                        journal[index] = row
                    count += 1
            if count != specification['declared_frame_count'] or set(journal) != needed:
                raise ValueError('Incomplete journal scope')
            capture = cv2.VideoCapture(str(SOURCES[clip]))
            if not capture.isOpened():
                raise ValueError('Could not open allowlisted source')
            metadata = dict(width=capture.get(cv2.CAP_PROP_FRAME_WIDTH), height=capture.get(cv2.CAP_PROP_FRAME_HEIGHT),
                frames=capture.get(cv2.CAP_PROP_FRAME_COUNT), fps=capture.get(cv2.CAP_PROP_FPS),
                backend=capture.getBackendName())
            if (metadata['width'],metadata['height'],metadata['frames'],metadata['fps']) != (4784,3190,count,10):
                capture.release(); raise ValueError('Changed decoder/source metadata')
            frame_buffer = deque(maxlen=9)
            retrieved, produced = [], 0
            try:
                if capture.get(cv2.CAP_PROP_POS_FRAMES) != 0:
                    raise ValueError('Decoder must start at frame zero')
                for index in range(count):
                    # Evict before allocation, not only after creating a tenth gray frame.
                    if len(frame_buffer) == 9:
                        frame_buffer.popleft()
                    if not capture.grab() or capture.get(cv2.CAP_PROP_POS_FRAMES) != index+1:
                        raise ValueError('Wrong sequential decoder index')
                    gray = None
                    if index in needed:
                        ok, image = capture.retrieve()
                        if not ok or image.shape != (3190,4784,3) or image.dtype != np.uint8:
                            raise ValueError('Wrong native decoded frame')
                        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        retrieved.append(index)
                    frame_buffer.append((index, gray))
                    if index not in by_frame:
                        continue
                    if index < 8 or any(frame is None for _,frame in frame_buffer) or len(frame_buffer) != 9:
                        raise ValueError('Selected state has no complete requested eight-frame history')
                    images = [frame for _,frame in frame_buffer]
                    rows = [journal[i] for i,_ in frame_buffer]
                    for state in by_frame[index]:
                        matches = [t for t in rows[-1]['tracks'] if
                                   (t['segment'],t['track_id']) == (state['segment'],state['track_id'])]
                        if (len(matches) != 1 or matches[0]['measured'] is not True or
                            matches[0]['measurement_source_xy'] != state['actual_source_xy'] or
                            matches[0]['qualified_moving'] != state['qualified_moving']):
                            raise ValueError('Selected current state differs from original journal')
                        result = run_state(state, images, rows, output)
                        destination.write(json.dumps(result, allow_nan=False)+'\n')
                        all_results.append(result); produced += 1
                    # Do not keep an old nine-image list alive as the deque rolls.
                    del images, rows
                    if index % 25 == 0:
                        print(clip+' frame '+str(index)+': '+str(produced)+' evaluated states', flush=True)
                if capture.grab():
                    raise ValueError('Unexpected extra source frame')
            finally:
                capture.release()
            if set(retrieved) != needed:
                raise ValueError('Incomplete requested native history')
            metadata.update(actual_sequential_frames=count, eof_verified=True, retrieved_indices=retrieved,
                            evaluated_states=produced, max_retained_gray_frames=9)
            source_metadata[clip] = metadata
            print('Completed '+clip+': '+str(produced)+' states', flush=True)
    if len(all_results) != selection['unique_states']:
        raise ValueError('Wrong completed state denominator')
    grid_records = [r for r in all_results if r['grid_windows']]
    if len(grid_records) != 1367 or sum(r['qualified_moving'] for r in grid_records) != 381:
        raise ValueError('Grid states lost')
    summary = dict(completed=True, created_at_utc=now(), totals=state_summary(all_results),
        grid=state_summary(grid_records), grid_strict=state_summary([r for r in grid_records if r['qualified_moving']]),
        by_clip={clip:state_summary([r for r in all_results if r['clip']==clip]) for clip in CLIPS},
        reference_panels=inventory['panels'], source_metadata=source_metadata,
        production_additions=0, production_removals=0, no_filter_applied=True,
        airborne_accuracy_established=False, no_new_labels=True)
    write(output/'summary.json', summary)
    write(output/'reference_evidence.json', reference_report(inventory, all_results))
    for name in ('states.jsonl','summary.json','reference_evidence.json'):
        bind(output/name)
    for result in all_results:
        if result['audit_inputs']:
            bind(output/result['audit_inputs']['path'], result['audit_inputs']['sha256'])
    for path, value in bound.items():
        if sha(path) != value:
            raise ValueError('Bound file changed during experiment: '+path)
    write(output/'completion_receipt.json', dict(completed=True, created_at_utc=now(), files_sha256=bound,
        all_inputs_and_outputs_rehashed=True, production_changed=False))
    print(json.dumps({k:v for k,v in summary.items() if k not in ('source_metadata','reference_panels')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        parser.error('Use --execute after reviewing the frozen experiment and synthetic tests')
    run(args.output)
