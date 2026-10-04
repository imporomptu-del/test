"""Literal, JSON-only V50 guard inputs for the V52 shadow experiment.

No image/archive reader, source model, fit or detector is imported. Old receipt
paths are lookup keys only, never an authorization to open other resources.
"""
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/tiny_target/accuracy_v50_20260926/prediction_01'
RECEIPT_SHA = '9213ed07fa7e8efd81f7c0032290dbe2f8c37bd06f6376b86d0b53a1af7dc2b1'
ALLOWED_NAMES = (
    'freeze.json', 'state_results.jsonl',
    'calibration_forecasts.jsonl', 'calibration_measurements.jsonl',
    'evaluation_forecasts.jsonl', 'evaluation_measurements.jsonl',
    'reference_context.json',
)
ARMS = ('median8_unit_scale', 'median8_temporal_scale', 'median3_temporal_scale')
PARTITIONS = ('calibration', 'embargo', 'evaluation')


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _regular_bytes(path):
    if path.is_symlink() or not path.is_file() or path.resolve() != path:
        raise ValueError('Expected literal regular artifact path')
    return path.read_bytes()


def read_authorized():
    """Return (documents, bindings), verifying each literal byte stream first."""
    receipt_path = BASE / 'completion_receipt.json'
    raw = _regular_bytes(receipt_path)
    if hashlib.sha256(raw).hexdigest() != RECEIPT_SHA:
        raise ValueError('V50 receipt hash mismatch')
    receipt = json.loads(raw)
    if receipt.get('completed') is not True:
        raise ValueError('V50 did not complete')
    bindings = {str(receipt_path): RECEIPT_SHA}
    documents = {}
    # Do not iterate the receipt map, the freeze packet map, or any old paths.
    for name in ALLOWED_NAMES:
        path = BASE / name
        raw = _regular_bytes(path)
        digest = hashlib.sha256(raw).hexdigest()
        if digest != receipt['files_sha256'].get(str(path)):
            raise ValueError('V50 artifact hash mismatch: ' + name)
        bindings[str(path)] = digest
        documents[name] = ([json.loads(line) for line in raw.splitlines()]
                           if name.endswith('.jsonl') else json.loads(raw))
    for path, digest in bindings.items():
        if hashlib.sha256(_regular_bytes(Path(path))).hexdigest() != digest:
            raise ValueError('Input changed during literal read')
    return documents, bindings


def _integer(value, minimum=0):
    return type(value) is int and value >= minimum


def _key(row):
    value = row.get('state_key')
    if (not isinstance(value, list) or len(value) != 4
            or value[0] not in ('0029', '0126') or not _integer(value[1])
            or not _integer(value[2]) or not isinstance(value[3], str) or not value[3]):
        raise ValueError('Invalid state key')
    return tuple(value)


def _keyed(rows):
    result = {}
    for row in rows:
        key = _key(row)
        if key in result:
            raise ValueError('Duplicate state key')
        result[key] = row
    return result


def _vector(value, count, name):
    if (not isinstance(value, list) or len(value) != count
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in value)):
        raise ValueError('Invalid finite vector: ' + name)
    return value


def _forecast(forecast):
    fingerprint = canonical_hash({k: v for k, v in forecast.items() if k != 'forecast_sha256'})
    if forecast.get('forecast_sha256') != fingerprint:
        raise ValueError('Forecast fingerprint mismatch')
    if forecast.get('available') is not True or forecast.get('reasons') != []:
        raise ValueError('Expected available frozen V50 forecast')
    points = forecast['used_points_xy']
    count = forecast['used_count']
    if not _integer(count, 1) or not isinstance(points, list) or len(points) != count:
        raise ValueError('Invalid used support count')
    for point in points:
        if (not isinstance(point, list) or len(point) != 2
                or any(not _integer(p) or p % 8 for p in point)
                or not all(8 <= p <= 120 for p in point)
                or not 40 <= max(abs(point[0]-64), abs(point[1]-64)) <= 56):
            raise ValueError('Invalid prior-selected guard geometry')
    if (len({tuple(p) for p in points}) != count
            or points != sorted(points, key=lambda p: (p[1], p[0]))
            or forecast['used_support_sha256'] != canonical_hash(points)):
        raise ValueError('Used support binding/order mismatch')
    stencils = forecast['stencils']
    if (forecast['stencil_count'] != len(stencils)
            or forecast['stencil_sha256'] != canonical_hash(stencils)):
        raise ValueError('Stencil binding mismatch')
    union = set()
    for stencil in stencils:
        axis = stencil['axis']
        center = stencil['center_xy']
        if axis not in ('x', 'y') or center not in points or stencil['weights'] != [1, -2, 1]:
            raise ValueError('Invalid guard stencil')
        x, y = center
        dx, dy = (8, 0) if axis == 'x' else (0, 8)
        expected = [[x-dx, y-dy], [x, y], [x+dx, y+dy]]
        if stencil['pixels_xy'] != expected or any(p not in points for p in expected):
            raise ValueError('Invalid guard stencil geometry')
        union.update(tuple(p) for p in expected)
    if union != {tuple(p) for p in points}:
        raise ValueError('Used support is not exactly the stencil union')
    arms = forecast['arms']
    if set(arms) != set(ARMS):
        raise ValueError('Unexpected forecast arms')
    for name in ARMS:
        if set(arms[name]) != {'prediction', 'scale'}:
            raise ValueError('Unexpected forecast arm fields')
        _vector(arms[name]['prediction'], count, name + ' prediction')
        if any(v < 1 for v in _vector(arms[name]['scale'], count, name + ' scale')):
            raise ValueError('Invalid forecast scale')
    if (arms[ARMS[0]]['prediction'] != arms[ARMS[1]]['prediction']
            or arms[ARMS[1]]['scale'] != arms[ARMS[2]]['scale']
            or arms[ARMS[0]]['scale'] != [1.0] * count):
        raise ValueError('V50 shared predictions/scales changed')
    return points, arms[ARMS[0]]['prediction'], arms[ARMS[2]]['prediction']


def _current(measurement, forecast):
    count = forecast['used_count']
    if (measurement['forecast_sha256'] != forecast['forecast_sha256']
            or measurement['used_support_sha256'] != forecast['used_support_sha256']
            or measurement['used_count'] != count):
        raise ValueError('Measurement and forecast bindings differ')
    if type(measurement['available']) is not bool:
        raise ValueError('Invalid current availability')
    invalid = measurement['current_nonfinite_used_point_count']
    if measurement['available'] is False:
        if (measurement['arms'] != {} or not _integer(invalid, 1) or invalid > count
                or measurement['reasons'] != ['nonfinite_current_on_fixed_used_guard_support']):
            raise ValueError('Unavailable packet contract changed')
        return [None] * count
    if invalid != 0 or measurement['reasons'] != [] or set(measurement['arms']) != set(ARMS):
        raise ValueError('Available measurement contract changed')
    reconstructed = []
    for name in ARMS:
        residual = _vector(measurement['arms'][name]['residuals'], count, name + ' residuals')
        prediction = forecast['arms'][name]['prediction']
        current = [p+r for p, r in zip(prediction, residual)]
        _vector(current, count, name + ' reconstructed current')
        reconstructed.append(current)
    # The pinned V50 artifact reconstructs identical native current samples
    # through all three arms. No tolerance or integer rounding hides mismatch.
    if not reconstructed[0] == reconstructed[1] == reconstructed[2]:
        raise ValueError('Current reconstruction differs between arms')
    return reconstructed[0]


def assemble(documents):
    """Validate the frozen ledger and return JSON-safe, isolated guard inputs.

    References remain a separate output document, never fields in fit inputs.
    Every state is copied intact. Incomplete current packets keep their entire
    support with null values; nonfinite-point locations cannot be reconstructed.
    """
    if set(documents) != set(ALLOWED_NAMES):
        raise ValueError('Expected only the seven literal V50 documents')
    states = _keyed(documents['state_results.jsonl'])
    scope = documents['freeze.json']['scope']
    assignments = _keyed(scope['assignments'])
    if set(states) != set(assignments):
        raise ValueError('State membership differs from frozen V50 scope')
    cutoffs = {}
    for cutoff in scope['cutoffs']:
        pair = (cutoff['clip'], cutoff['segment'])
        if pair in cutoffs or not _integer(cutoff['cutoff_frame_index']):
            raise ValueError('Invalid duplicate or noninteger cutoff')
        cutoffs[pair] = cutoff['cutoff_frame_index']
    if scope['history_frames'] != 8 or scope['embargo_frames'] != 8:
        raise ValueError('V50 eight-frame history/embargo changed')
    for key, state in states.items():
        assignment = assignments[key]
        cutoff = cutoffs[(key[0], key[2])]
        expected = ('calibration' if key[1] <= cutoff else
                    'embargo' if key[1] <= cutoff + 8 else 'evaluation')
        if (state['partition'] != expected or assignment['partition'] != expected
                or type(assignment['archived']) is not bool):
            raise ValueError('State partition differs from frozen chronological scope')
        if state.get('source_scores_and_original_detections_unchanged') is not True:
            raise ValueError('Original source decisions not preserved')
    for partition in PARTITIONS:
        listed = scope['partitions'][partition]
        keys = [_key({'state_key': key}) for key in listed]
        expected = {k for k, a in assignments.items() if a['partition'] == partition}
        if len(keys) != len(set(keys)) or set(keys) != expected:
            raise ValueError('Frozen partition membership mismatch')
        frames = {(k[0], k[2], k[1]) for k in expected}
        archived = {k for k in expected if assignments[k]['archived']}
        archive_frames = {(k[0], k[2], k[1]) for k in archived}
        counts = dict(states=len(expected), archived_states=len(archived),
                      history_unknown_states=len(expected)-len(archived),
                      unique_response_frames=len(frames), archived_response_frames=len(archive_frames),
                      frames_without_archives=len(frames-archive_frames))
        if scope['counts'][partition] != counts:
            raise ValueError('Frozen scope denominator mismatch')
    packets = []
    for partition in ('calibration', 'evaluation'):
        forecasts = _keyed(documents[partition + '_forecasts.jsonl'])
        measurements = _keyed(documents[partition + '_measurements.jsonl'])
        expected = {k for k, a in assignments.items() if a['partition'] == partition and a['archived']}
        if set(forecasts) != expected or set(measurements) != expected:
            raise ValueError('Packet membership differs from frozen V50 scope')
        for key in sorted(expected):
            f = forecasts[key]['forecast']
            record = measurements[key]
            m = record['measurement']
            if (record['clip'] != key[0] or record['frame_index'] != key[1]
                    or record['segment'] != key[2] or record['forecast_available'] is not True):
                raise ValueError('Measurement state identity mismatch')
            points, median8, median3 = _forecast(f)
            current = _current(m, f)
            status = 'background_measured' if m['available'] else 'response_unavailable'
            if states[key]['v50_status'] != status:
                raise ValueError('State and current availability differ')
            packets.append(dict(state_key=list(key), partition=partition, available=m['available'],
                reasons=deepcopy(m['reasons']), points_xy=deepcopy(points),
                median8=list(median8), median3=list(median3), current=current,
                metadata=dict(original_v50_status=status, forecast_sha256=f['forecast_sha256'],
                    used_support_sha256=f['used_support_sha256'],
                    original_nonfinite_current_point_count=m['current_nonfinite_used_point_count'],
                    unavailable_current_packet_not_partially_reconstructed=not m['available'],
                    no_core_or_reference_values_in_fit_input=True,
                    current_whole_frame_registration_caveat_preserved=True)))
    for key, state in states.items():
        if state['partition'] == 'embargo':
            if state['v50_status'] != 'embargo_not_scored':
                raise ValueError('Embargo state must remain unscored')
        elif not assignments[key]['archived'] and state['v50_status'] != 'history_unknown':
            raise ValueError('Unarchived state must remain history unknown')
    return dict(states=deepcopy(documents['state_results.jsonl']), packets=packets,
        references=deepcopy(documents['reference_context.json']), scope=deepcopy(scope),
        counts=dict(states=len(states), status_counts=dict(Counter(s['v50_status'] for s in states.values())),
            scored_archived_packets=len(packets), available_current_packets=sum(p['available'] for p in packets),
            unavailable_current_packets=sum(not p['available'] for p in packets),
            guard_point_opportunities=sum(len(p['points_xy']) for p in packets),
            available_current_points=sum(len(p['points_xy']) for p in packets if p['available']),
            unavailable_current_packet_points=sum(len(p['points_xy']) for p in packets if not p['available'])))
