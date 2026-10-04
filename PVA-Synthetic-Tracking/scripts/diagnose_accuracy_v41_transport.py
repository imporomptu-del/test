"""Frozen saved-source diagnostic; no detector, media decode or rejection policy."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import cv2
import numpy as np

from accuracy_v41_geometry import LAGS, SHIFTS, source_templates
from accuracy_v41_transport import compare_templates

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/tiny_target'
V40 = BASE / 'accuracy_v40_20260925'
V41 = BASE / 'accuracy_v41_20260925'
PACKET = ROOT.parent / 'outputs/seaqr_accuracy_v40_20260925/coverage_01'
PACKET_SHA = 'ca4fa702a0106bb88e15c94131b09b955de69fa14e21e5abec9ea670c48d0a77'
AUDIT_SHA = '763a287bfc69f0e0a72617dfd4d60ffd5acb43326e0fd7584888eb22d4506237'
EVIDENCE_SHA = 'cefb4ccc97a894d16a02c5a11c9ce4696f4b49a279a5df22477fcd175a7b87a3'
CLIPS = ('0029', '0126', '0055', '0082')
TIE_EPSILON = 1e-9


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def now():
    return datetime.now(timezone.utc).isoformat()


def preference(difference):
    if difference > TIE_EPSILON:
        return 'transported'
    if difference < -TIE_EPSILON:
        return 'stationary'
    return 'numerical_tie'


def actual_identity(row, segment, track_id):
    found = [t for t in row['tracks'] if t['segment'] == segment and t['track_id'] == track_id]
    if len(found) > 1:
        raise ValueError('Duplicate segment/track ID')
    return found[0] if found else None


def evaluate_lag(window, rows, grayscale, frame, track, lag):
    first = window['frame_start']
    prior_frame = frame-lag
    result = dict(lag=lag, prior_frame=prior_frame, pair_available=False, reasons=[],
                  variants=[], all_nine_available=False, offset_sign_agreement=None,
                  zero_offset_preference=None)
    if prior_frame < first:
        result['reasons'] = ['insufficient_retained_window_history']
        return result
    row, prior_row = rows[frame], rows[prior_frame]
    if (prior_row['segment'] != row['segment'] or
            any(rows[i]['motion']['reset'] for i in range(prior_frame+1, frame+1))):
        result['reasons'] = ['segment_or_reset_boundary']
        return result
    if row['timestamp_ns'] - prior_row['timestamp_ns'] != lag*100_000_000:
        result['reasons'] = ['non_nominal_timestamp_spacing']
        return result
    previous = actual_identity(prior_row, track['segment'], track['track_id'])
    if previous is None or previous['measured'] is not True:
        result['reasons'] = ['prior_identity_absent' if previous is None else 'prior_identity_not_measured']
        return result
    if previous.get('measurement_source_xy') is None:
        raise ValueError('Actual prior measurement is missing its source position')
    result['prior_actual_source_xy'] = previous['measurement_source_xy']
    result['pair_available'] = True
    for shift in SHIFTS:
        try:
            data = source_templates(grayscale[frame-first], grayscale[prior_frame-first],
                window['crop_xywh'][:2], window['crop_xywh'][:2],
                row['source_to_reference'], prior_row['source_to_reference'],
                track['measurement_source_xy'], previous['measurement_source_xy'], shift)
        except ValueError as exc:
            result['variants'].append(dict(shift=list(shift), available=False,
                                           reasons=['geometry_unavailable: '+str(exc)]))
            continue
        probe = compare_templates(data['current'], data['stationary_prior'], data['transported_prior'])
        probe.update(shift=list(shift), geometry=data['geometry'])
        if probe['available']:
            probe['preference'] = preference(probe['advantage_stationary_minus_transported'])
            if shift == (0, 0):
                result['zero_offset_preference'] = probe['preference']
        result['variants'].append(probe)
    result['all_nine_available'] = all(v['available'] for v in result['variants'])
    if result['all_nine_available']:
        signs = {v['preference'] for v in result['variants']}
        result['offset_sign_agreement'] = next(iter(signs)) if len(signs) == 1 else 'mixed'
    return result


def count_state(target, state):
    if tuple(lag['lag'] for lag in state['lags']) != LAGS:
        raise ValueError('All four frozen lags must remain in the denominator')
    target['actual_states'] += 1
    target['strict_states'] += int(state['qualified_moving'])
    for lag in state['lags']:
        prefix = str(lag['lag'])+':'
        target[prefix+'planned_pairs'] += 1
        target[prefix+'same_id_actual_pair'] += int(lag['pair_available'])
        for reason in lag['reasons']:
            target[prefix+'unavailable:'+reason] += 1
        target[prefix+'planned_variants'] += len(SHIFTS)
        target[prefix+'available_variants'] += sum(v['available'] for v in lag['variants'])
        zero = lag['zero_offset_preference']
        target[prefix+'zero:'+(zero if zero else 'unavailable')] += 1
        agreement = lag['offset_sign_agreement']
        target[prefix+'nine:'+(agreement if agreement else 'unavailable')] += 1
    all_lags = [lag['offset_sign_agreement'] for lag in state['lags']]
    all36 = all(v is not None for v in all_lags)
    target['all_36_available'] += int(all36)
    if all36:
        target['all_36_preference:'+(all_lags[0] if len(set(all_lags)) == 1 else 'mixed')] += 1


def run(output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Fresh output directory required')
    bound = {}

    def bind(path, expected=None):
        path = Path(path).resolve()
        digest = sha(path)
        if expected is not None and digest != expected:
            raise ValueError('Changed bound input: '+str(path))
        if str(path) in bound and bound[str(path)] != digest:
            raise ValueError('Input changed during binding')
        bound[str(path)] = digest

    bind(PACKET/'packet.json', PACKET_SHA)
    bind(PACKET/'completion.json')
    bind(V40/'independent_audit_01.json', AUDIT_SHA)
    bind(V40/'coverage_workload_01/window_evidence.json', EVIDENCE_SHA)
    audit = read(V40/'independent_audit_01.json')
    receipt_path = V40/'coverage_workload_01/completion_receipt.json'
    bind(receipt_path, audit['checked_files_sha256'][str(receipt_path)])
    receipt = read(receipt_path)
    for path, digest in receipt['inputs_outputs_sha256'].items():
        bind(path, digest)
    packet = read(PACKET/'packet.json')
    native_windows = {w['window_id']: w for w in packet['windows']}
    for w in packet['windows']:
        bind(PACKET/w['native_archive']['path'], w['native_archive']['sha256'])
    for name in ['docs/accuracy_v41_plan.md', 'scripts/accuracy_v41_geometry.py',
                 'scripts/accuracy_v41_transport.py', 'scripts/diagnose_accuracy_v41_transport.py',
                 'scripts/accuracy_v38_source_pairs.py', 'tests/unit/test_accuracy_v41_geometry.py',
                 'tests/unit/test_accuracy_v41_transport.py', 'tests/unit/test_accuracy_v41_runner.py']:
        bind(ROOT/name)
    for name in ['source_mechanism_0126.json', 'source_mechanism_0082.json']:
        bind(V41/name)
    evidence = read(V40/'coverage_workload_01/window_evidence.json')
    if len(evidence) != 108 or sum(w['counts']['actual_measured_states'] for w in evidence) != 1367:
        raise ValueError('Changed exact V40 denominator')
    if sum(w['counts']['qualified_measured_states'] for w in evidence) != 381:
        raise ValueError('Changed strict V40 denominator')
    output.mkdir(parents=True)
    write(output/'freeze.json', dict(schema='seaqr.accuracy-v41-transport-freeze.v1',
        created_at_utc=now(), inputs_sha256=bound.copy(), lags=LAGS, shifts=SHIFTS,
        numerical_tie_epsilon_dn_squared=TIE_EPSILON,
        libraries=dict(python=platform.python_version(), numpy=np.__version__, opencv=cv2.__version__),
        actual_states=1367, strict_states=381, windows=108, real_scores_not_yet_computed=True,
        held_out=False, production_change=False))
    bind(output/'freeze.json')
    print('Frozen before new source-template scores: '+str(output), flush=True)
    journals = {}
    for clip in CLIPS:
        path = BASE / ('visible_validation_v34_20260923/audit_20260924/evidence/run/full_repeat0_'+clip+'/frames.jsonl')
        needed = {i for w in packet['windows'] if w['clip'] == clip for i in w['frame_indices']}
        rows = {}
        count = 0
        with path.open() as stream:
            for index, line in enumerate(stream):
                row = json.loads(line)
                if row['frame_index'] != index:
                    raise ValueError('Noncontiguous original journal')
                if index in needed:
                    rows[index] = row
                count += 1
        if (count != packet['actual_decoder_metadata'][clip]['actual_sequential_frame_count']
                or set(rows) != needed):
            raise ValueError('Changed journal frame denominator')
        journals[clip] = rows
    total, strict, by_window, by_clip = Counter(), Counter(), {}, {c: Counter() for c in CLIPS}
    references = read(V40/'coverage_workload_01/visible_reference_evidence.json')
    reference_keys = {(r['window_id'], r['frame_index'], r['stages']['actual_measurement']['assigned_id']): r
                      for r in references}
    reference_evidence = []
    rows_written = 0
    with (output/'states.jsonl').open('x') as stream:
        for window in evidence:
            wid, clip = window['window_id'], window['clip_id']
            spec = native_windows[wid]
            by_window[wid] = Counter()
            # Zero-workload windows stay inventoried; their arrays are hash-bound
            # but need not be opened because no state requests a comparison.
            if not window['counts']['actual_measured_states']:
                continue
            with np.load(PACKET/spec['native_archive']['path'], allow_pickle=False) as archive:
                bgr = archive['native_bgr']
                if (bgr.shape != (20, 192, 256, 3) or bgr.dtype != np.uint8 or
                    archive['frame_indices'].tolist() != spec['frame_indices'] or
                    hashlib.sha256(bgr.tobytes()).hexdigest() != spec['native_array']['sha256']):
                    raise ValueError('Changed native crop arrays')
                grayscale = np.stack([cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in bgr])
            clip_rows = journals[clip]
            for frame in window['frames']:
                index = frame['frame_index']
                for original in frame['actual_measurements']:
                    track = actual_identity(clip_rows[index], original['segment'], original['track_id'])
                    if (not track or track['measured'] is not True or
                        track['measurement_source_xy'] != original['source_xy'] or
                        track['qualified_moving'] != original['qualified_moving']):
                        raise ValueError('V40 actual state differs from original journal')
                    state = dict(window_id=wid, clip=clip, frame_index=index, segment=track['segment'],
                        track_id=track['track_id'], actual_source_xy=track['measurement_source_xy'],
                        qualified_moving=track['qualified_moving'],
                        lags=[evaluate_lag(spec, clip_rows, grayscale, index, track, lag) for lag in LAGS])
                    count_state(total, state); count_state(by_window[wid], state); count_state(by_clip[clip], state)
                    if state['qualified_moving']:
                        count_state(strict, state)
                    key = (wid, index, str(track['segment'])+'/'+track['track_id'])
                    if key in reference_keys:
                        reference_evidence.append(dict(reference=reference_keys[key], diagnostic=state))
                    stream.write(json.dumps(state, allow_nan=False)+'\n')
                    rows_written += 1
            print(wid+': '+str(by_window[wid]['actual_states'])+' measured states', flush=True)
    if rows_written != 1367 or strict['actual_states'] != 381 or len(reference_evidence) != 11:
        raise ValueError('Incomplete comparison denominator')
    summary = dict(schema='seaqr.accuracy-v41-transport-summary.v1', completed=True,
        created_at_utc=now(), total=dict(total), strict=dict(strict),
        by_clip={k:dict(v) for k,v in by_clip.items()}, by_window={k:dict(v) for k,v in by_window.items()},
        windows=108, actual_states=rows_written, provisional_reference_samples=11,
        source_decodes=0, candidate_policy=None, physical_class_inferred=False, production_changed=False,
        interpretation='Model preference is a conditional predictive-fit diagnostic, not motion/class truth or rejection authority.')
    write(output/'summary.json', summary)
    write(output/'reference_evidence.json', reference_evidence)
    for name in ['states.jsonl', 'summary.json', 'reference_evidence.json']:
        bind(output/name)
    for path, digest in bound.items():
        if sha(path) != digest:
            raise ValueError('Bound file changed during diagnostic: '+path)
    write(output/'completion_receipt.json', dict(completed=True, created_at_utc=now(),
                                               all_bound_files_rehashed=True, files_sha256=bound))
    print(json.dumps({k:v for k,v in summary.items() if k not in ('by_clip','by_window')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    run(parser.parse_args().output)
