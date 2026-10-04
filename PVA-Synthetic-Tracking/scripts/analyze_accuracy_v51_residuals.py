"""Retrospective spatial/temporal V50 residual diagnostics, never predictors.

Only six literal, receipt-bound V50 JSON artifacts are read. No receipt map is
traversed and no image/archive, reference label or source dataset is accessed.
Running this module prints JSON; it does not write artifacts or change models.
"""
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/tiny_target/accuracy_v50_20260926/prediction_01'
RECEIPT_SHA = '9213ed07fa7e8efd81f7c0032290dbe2f8c37bd06f6376b86d0b53a1af7dc2b1'
ALLOWED_NAMES = (
    'freeze.json', 'state_results.jsonl',
    'calibration_forecasts.jsonl', 'calibration_measurements.jsonl',
    'evaluation_forecasts.jsonl', 'evaluation_measurements.jsonl',
)
ARMS = ('median8_unit_scale', 'median8_temporal_scale', 'median3_temporal_scale')
QUADRANTS = ('upper_left', 'upper_right', 'lower_left', 'lower_right')


def _regular_bytes(path):
    if path.is_symlink() or not path.is_file() or path.resolve() != path:
        raise ValueError('Expected literal regular artifact path')
    return path.read_bytes()


def read_authorized():
    """Read literal allowlist only, validating bytes before JSON decoding."""
    receipt_path = BASE / 'completion_receipt.json'
    payload = _regular_bytes(receipt_path)
    if hashlib.sha256(payload).hexdigest() != RECEIPT_SHA:
        raise ValueError('V50 receipt hash mismatch')
    receipt = json.loads(payload)
    if receipt.get('completed') is not True:
        raise ValueError('V50 did not complete')
    bindings = {str(receipt_path): RECEIPT_SHA}
    documents = {}
    # Never iterate/open paths from the receipt map or from freeze metadata.
    for name in ALLOWED_NAMES:
        path = BASE / name
        payload = _regular_bytes(path)
        digest = hashlib.sha256(payload).hexdigest()
        if digest != receipt['files_sha256'].get(str(path)):
            raise ValueError('V50 artifact hash mismatch: ' + name)
        bindings[str(path)] = digest
        documents[name] = ([json.loads(line) for line in payload.splitlines()]
                           if name.endswith('.jsonl') else json.loads(payload))
    return documents, bindings


def distribution(values):
    a = np.asarray(values, dtype=float)
    if a.ndim != 1 or not np.isfinite(a).all():
        raise ValueError('Expected finite one-dimensional metrics')
    if not a.size:
        return dict(count=0, mean=None, median=None, p90=None, max=None)
    return dict(count=len(a), mean=float(a.mean()), median=float(np.median(a)),
                p90=float(np.quantile(a, .9)), max=float(a.max()))


def top_tenth_share(values):
    """Share in the largest ceil(n/10) values; zero total is undefined."""
    a = np.asarray(values, dtype=float)
    if a.ndim != 1 or not np.isfinite(a).all() or (a < 0).any():
        raise ValueError('Expected finite nonnegative masses')
    count = (len(a) + 9) // 10
    total = math.fsum(a)
    return dict(count=len(a), selected_count=count,
                share=float(math.fsum(np.sort(a)[-count:]) / total) if total else None)


def residual_metrics(points_xy, residuals):
    points = np.asarray(points_xy, dtype=float)
    r = np.asarray(residuals, dtype=float)
    if (r.ndim != 1 or not r.size or points.shape != (len(r), 2)
            or not np.isfinite(r).all() or not np.isfinite(points).all()
            or (points != np.floor(points)).any() or (points < 0).any()
            or (points > 128).any() or len(np.unique(points, axis=0)) != len(r)):
        raise ValueError('Invalid guard support or residuals')
    absolute = np.abs(r)
    mass = math.fsum(absolute)
    quadrant_ids = (points[:, 1] >= 64).astype(int) * 2 + (points[:, 0] >= 64)
    quadrants = {}
    for index, name in enumerate(QUADRANTS):
        values = absolute[quadrant_ids == index]
        qmass = math.fsum(values)
        quadrants[name] = dict(point_count=len(values), absolute_mass_dn=qmass,
            mae_dn=float(values.mean()) if len(values) else None,
            absolute_mass_share=qmass / mass if mass else None)
    return dict(point_count=len(r), absolute_mass_dn=mass, mae_dn=float(absolute.mean()),
        signed_median_residual_dn=float(np.median(r)),
        coherence=abs(math.fsum(r)) / mass if mass else None,
        positive_fraction=float(np.mean(r > 0)), negative_fraction=float(np.mean(r < 0)),
        zero_fraction=float(np.mean(r == 0)), top_tenth_absolute_mass=top_tenth_share(absolute),
        quadrants=quadrants)


def _keyed(rows):
    output = {}
    for row in rows:
        key = tuple(row['state_key'])
        if key in output:
            raise ValueError('Duplicate state key')
        output[key] = row
    return output


def packet_diagnostics(documents):
    states = _keyed(documents['state_results.jsonl'])
    scope = documents['freeze.json']['scope']
    assignments = _keyed(scope['assignments'])
    if set(states) != set(assignments):
        raise ValueError('State membership differs from frozen V50 scope')
    for key, state in states.items():
        if state['partition'] != assignments[key]['partition']:
            raise ValueError('State partition changed')
    packets = {}
    for partition in ('calibration', 'evaluation'):
        forecasts = _keyed(documents[partition + '_forecasts.jsonl'])
        measurements = _keyed(documents[partition + '_measurements.jsonl'])
        expected = {key for key, a in assignments.items()
                    if a['partition'] == partition and a['archived']}
        if set(forecasts) != expected or set(measurements) != expected:
            raise ValueError('Packet membership differs from frozen V50 scope')
        for key in sorted(expected):
            f = forecasts[key]['forecast']
            m = measurements[key]['measurement']
            if m['forecast_sha256'] != f['forecast_sha256']:
                raise ValueError('Measurement and forecast binding differs')
            if m['available'] and (not f['available']
                    or m['used_support_sha256'] != f['used_support_sha256']):
                raise ValueError('Available measurement support mismatch')
            row = dict(state_key=list(key), partition=partition, available=m['available'],
                       reasons=m['reasons'], arms={})
            if m['available']:
                if set(m['arms']) != set(ARMS) or set(f['arms']) != set(ARMS):
                    raise ValueError('Unexpected arms')
                for arm in ARMS:
                    row['arms'][arm] = residual_metrics(f['used_points_xy'], m['arms'][arm]['residuals'])
            packets[key] = row
    return states, packets


def aggregate(states, packets):
    """Frame means give each measurable response frame equal temporal weight."""
    ordered = sorted(states)
    available = [packets[k] for k in ordered if k in packets and packets[k]['available']]
    archived = [packets[k] for k in ordered if k in packets]
    frames = defaultdict(list)
    for key in ordered:
        frames[(key[0], key[2], key[1])].append(key)
    measurable_frames = {key: values for key, values in frames.items()
                         if any(k in packets for k in values)}
    complete = {key: values for key, values in measurable_frames.items()
                if all(packets[k]['available'] for k in values if k in packets)}
    result = dict(states=len(states), status_counts=dict(Counter(v['v50_status'] for v in states.values())),
        unique_response_frames=len(frames), scored_archived_packets=len(archived),
        available_packets=len(available), unavailable_packets=len(archived)-len(available),
        unavailable_reasons=dict(Counter(reason for p in archived if not p['available'] for reason in p['reasons'])),
        response_frames_with_scored_archives=len(measurable_frames),
        complete_archived_response_frames=len(complete),
        unavailable_archived_response_frames=len(measurable_frames)-len(complete),
        response_frames_without_scored_archives=len(frames)-len(measurable_frames),
        states_without_scored_archives=len(states)-len(archived), arms={})
    for arm in ARMS:
        values = [p['arms'][arm] for p in available]
        frame_means = [float(np.mean([packets[k]['arms'][arm]['mae_dn'] for k in keys if k in packets]))
                       for keys in complete.values()]
        mass = math.fsum(v['absolute_mass_dn'] for v in values)
        quadrants = {}
        for name in QUADRANTS:
            count = sum(v['quadrants'][name]['point_count'] for v in values)
            qmass = math.fsum(v['quadrants'][name]['absolute_mass_dn'] for v in values)
            quadrants[name] = dict(point_pairs=count, absolute_mass_dn=qmass,
                pooled_mae_dn=qmass/count if count else None, absolute_mass_share=qmass/mass if mass else None)
        result['arms'][arm] = dict(packet_mae_dn=distribution([v['mae_dn'] for v in values]),
            packet_signed_median_residual_dn=distribution([v['signed_median_residual_dn'] for v in values]),
            packet_coherence=distribution([v['coherence'] for v in values if v['coherence'] is not None]),
            zero_mass_packets=sum(v['absolute_mass_dn'] == 0 for v in values),
            packet_top_tenth_absolute_mass_share=distribution([v['top_tenth_absolute_mass']['share'] for v in values
                                                              if v['top_tenth_absolute_mass']['share'] is not None]),
            equal_frame_mean_packet_mae_dn=distribution(frame_means),
            top_tenth_frame_mean_error_share=top_tenth_share(frame_means),
            pooled_guard_quadrants=quadrants)
    return result


def summarize(states, packets):
    groups = {}
    bins = defaultdict(dict)
    for partition in ('calibration', 'embargo', 'evaluation'):
        groups[partition] = {}
        for clip in sorted({k[0] for k in states}):
            selected = {k: s for k, s in states.items() if s['partition'] == partition and k[0] == clip}
            groups[partition][clip] = aggregate(selected, packets)
    # Fixed absolute response-frame bins [9*b,9*b+8], partition never pooled.
    for key, state in states.items():
        bins[(state['partition'], key[0], key[2], key[1] // 9)][key] = state
    temporal = [dict(partition=p, clip=c, segment=s, frame_start=9*b, frame_end=9*b+8,
                     metrics=aggregate(rows, packets)) for (p, c, s, b), rows in sorted(bins.items())]
    return dict(groups=groups, nine_response_frame_bins=temporal,
        all_state_count=len(states), scored_archive_count=len(packets),
        retrospective_current_response_diagnostics_not_predictors=True,
        no_object_labels_or_detector_decisions=True,
        complete_archived_frame_means_do_not_cover_history_unknown_states=True,
        duplicated_and_overlapping_guard_samples_not_independent=True,
        quadrants_are_local_guard_coordinates_not_full_frame_locations=True,
        all_zero_absolute_mass_shares_are_null_not_zero=True,
        temporal_top_tenth_uses_equal_frame_mean_packet_mae_not_raw_point_mass=True)


def analyze():
    documents, bindings = read_authorized()
    states, packets = packet_diagnostics(documents)
    if len(states) != 1211 or len(packets) != 493:
        raise ValueError('Unexpected frozen V50 count')
    result = summarize(states, packets)
    for path, expected in bindings.items():
        if hashlib.sha256(_regular_bytes(Path(path))).hexdigest() != expected:
            raise ValueError('V50 artifact changed during diagnostic')
    result['input_files_sha256'] = bindings
    result['packet_diagnostics'] = list(packets.values())
    return result


if __name__ == '__main__':
    print(json.dumps(analyze(), allow_nan=False, sort_keys=True))
