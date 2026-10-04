"""Offline, audit-gated V56 diagnostics. Reads arrays/journals, never media.

Reference-gate pixel extrema are diagnostic selections, not object locations or
physical truth. This utility never imports detector inference or a media reader.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from accuracy_v56_capture import FLOAT_FIELDS, FLAG_FIELDS, rank_candidates, validate_rectangle


FRAMES = tuple(range(213, 219))
FRAME_COUNT = 674
GATE = 7.0
VALUE = {name: index for index, name in enumerate(FLOAT_FIELDS)}
FLAG = {name: index for index, name in enumerate(FLAG_FIELDS)}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            value.update(block)
    return value.hexdigest()


def _object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'Duplicate JSON key: ' + key)
        result[key] = value
    return result


def loads(text):
    def nonfinite(value):
        raise ValueError('Nonfinite JSON number: ' + value)
    return json.loads(text, object_pairs_hook=_object, parse_constant=nonfinite)


def exact(a, b, name):
    require(json.dumps(a, sort_keys=True, allow_nan=False) ==
            json.dumps(b, sort_keys=True, allow_nan=False), 'Changed ' + name)


def point(value):
    require(isinstance(value, (list, tuple)) and len(value) == 2 and
            all(type(v) in (int, float) and math.isfinite(v) for v in value), 'Finite XY required')
    return np.asarray(value, dtype=np.float64)


def mapped(matrix, xy):
    result = np.asarray(matrix, dtype=np.float64) @ np.asarray([*point(xy), 1.0])
    require(np.isfinite(result).all() and result[2] != 0, 'Invalid coordinate transform')
    return (result[:2] / result[2]).tolist()


def number(value):
    result = float(value)
    return result if math.isfinite(result) else None


def source_gate(metadata):
    rect = validate_rectangle(metadata['rectangle'])
    require(rect['radius_px'] == GATE, 'Changed probe radius')
    matrix = np.asarray(metadata['source_to_reference'], dtype=np.float64)
    require(matrix.shape == (3, 3) and np.isfinite(matrix).all(), 'Invalid source-to-reference matrix')
    # The production transform is affine. Refuse unsupported projective bounds
    # rather than claiming that a possibly truncated gate is complete.
    require(np.array_equal(matrix[2], [0., 0., 1.]), 'Projective gate bounds unsupported')
    inverse = np.linalg.inv(matrix)
    center = point(metadata['source_probe']['source_xy'])
    ref = mapped(matrix, center.tolist())
    exact(ref, rect['probe_xy'], 'transformed fixed probe')
    require(metadata['source_probe']['original_reference_match_gate_px'] == GATE,
            'Changed original circular source gate')
    x0, y0, x1, y1 = rect['capture_bounds_exclusive_xyxy']
    tx0, ty0, tx1, ty1 = rect['tile_bounds_exclusive_xyxy']
    h, w = rect['shape_hw']
    radii = GATE * np.linalg.norm(matrix[:2, :2], axis=1)
    lower = np.maximum(np.ceil(np.asarray(ref)-radii), [0, 0])
    upper = np.minimum(np.floor(np.asarray(ref)+radii), [w-1, h-1])
    require(lower[0] >= tx0 and lower[1] >= ty0 and upper[0] < tx1 and upper[1] < ty1,
            'Full ranked tiles truncate the fixed source gate')
    yy, xx = np.mgrid[y0:y1, x0:x1]
    source_x = inverse[0, 0]*xx + inverse[0, 1]*yy + inverse[0, 2]
    source_y = inverse[1, 0]*xx + inverse[1, 1]*yy + inverse[1, 2]
    mask = (source_x-center[0])**2 + (source_y-center[1])**2 <= GATE**2
    return mask, inverse, rect


def _pixel(snapshot, metadata, inverse, yy, xx):
    rect = metadata['rectangle']
    x0, y0, _, _ = rect['capture_bounds_exclusive_xyxy']
    x, y = int(xx+x0), int(yy+y0)
    values, flags = snapshot['values'], snapshot['flags']
    local = {name: number(values[yy, xx, index]) for name, index in VALUE.items()}
    bits = {name: bool(flags[yy, xx, index]) for name, index in FLAG.items()}
    local['tile_sigma_precise'] = number(snapshot['precise_sigmas'][yy, xx])
    for sign in ('positive', 'negative'):
        for domain in ('temporal', 'spatial'):
            a, b = local[sign+'_signed_'+domain], local[domain+'_threshold_dn']
            local[sign+'_'+domain+'_margin_dn'] = None if a is None or b is None else a-b
    competitors = []
    own = local['temporal']
    full_h, full_w = rect['shape_hw']
    complete = True
    for py in range(max(0, y-2), min(full_h, y+3)):
        for px in range(max(0, x-2), min(full_w, x+3)):
            cy, cx = py-y0, px-x0
            if not (0 <= cy < values.shape[0] and 0 <= cx < values.shape[1]):
                complete = False
                continue
            response = number(values[cy, cx, VALUE['temporal']])
            if own is not None and response is not None and abs(response) > abs(own):
                competitors.append(dict(reference_xy=[px, py], source_xy=mapped(inverse, [px, py]),
                    raw_temporal_dn=response, absolute_raw_temporal_dn=abs(response),
                    support=bool(flags[cy, cx, FLAG['support']]),
                    eligible=bool(flags[cy, cx, FLAG['eligible']])))
    competitors.sort(key=lambda item: (-item['absolute_raw_temporal_dn'],
                                      item['reference_xy'][1], item['reference_xy'][0]))
    return dict(reference_xy=[x, y], source_xy=mapped(inverse, [x, y]),
                selected_pixel_is_object_truth=False, values=local, flags=bits,
                positive_prequota_rank=int(snapshot['positive_prequota_rank'][yy, xx]),
                stronger_raw_absolute_neighbors=competitors,
                neighborhood_complete_in_capture=complete)


def selected_pixel(snapshot, metadata, mask, inverse, field):
    values = snapshot['values'][..., VALUE[field]]
    candidates = mask & np.isfinite(values)
    if not candidates.any():
        return None
    # argmax on row-major pixels gives the fixed y/x tie-break, without
    # conditioning this selection on support, threshold passage or detection.
    index = int(np.argmax(np.where(candidates, values, -np.inf)))
    yy, xx = np.unravel_index(index, values.shape)
    result = _pixel(snapshot, metadata, inverse, yy, xx)
    result['selection'] = 'maximum finite '+field+' within original source gate; y/x tie-break'
    return result


def _in_gate(xy, center):
    return math.dist(point(xy), point(center)) <= GATE


def _proposal_record(proposal, inverse):
    result = dict(proposal)
    result['source_xy'] = mapped(inverse, [proposal['x'], proposal['y']])
    return result


def summarize_probe(snapshot, metadata, row):
    require(type(metadata['frame']) is int and metadata['frame'] == row['frame_index'], 'Capture/journal frame mismatch')
    require(metadata['prelearning'] is True and metadata['full_exposed_state_unchanged'] is True,
            'Missing pre-learning read-only capture guarantee')
    exact(metadata['native_state_before'], metadata['native_state_after'], 'capture state fingerprints')
    exact(metadata['source_to_reference'], row['source_to_reference'], 'capture/journal transform')
    exact(metadata['tracks'], row['tracks'], 'capture/journal tracks')
    exact(metadata['float_fields'], list(FLOAT_FIELDS), 'float schema')
    exact(metadata['flag_fields'], list(FLAG_FIELDS), 'flag schema')
    mask, inverse, rect = source_gate(metadata)
    snapshot = dict(snapshot, metadata=metadata)
    ranks = rank_candidates(snapshot, max_candidates_per_tile_polarity=12, verify_native=True)
    for name in ('positive_prequota_rank', 'negative_prequota_rank'):
        require(snapshot[name].dtype == np.int32 and np.array_equal(snapshot[name], ranks[name]),
                'Saved prequota ranks do not match native selection')
    flags, values = snapshot['flags'], snapshot['values']
    require(snapshot['precise_sigmas'].dtype == np.float64 and
            snapshot['precise_sigmas'].shape == mask.shape, 'Invalid precise sigmas')
    eligible = flags[..., FLAG['eligible']].astype(bool)
    temporal = flags[..., FLAG['positive_temporal_pass']].astype(bool)
    spatial = flags[..., FLAG['positive_spatial_pass']].astype(bool)
    peak = flags[..., FLAG['raw_absolute_peak']].astype(bool)
    expected = eligible & temporal & spatial & peak
    require(np.array_equal(expected, flags[..., FLAG['positive_candidate']].astype(bool)), 'Candidate predicate mismatch')
    cells = metadata['decoded_pre_frame_quota_cells']
    require(type(cells) is list and all(type(cell) is list and 0 < len(cell) <= 12 for cell in cells),
            'Invalid decoded cells')
    decoded = []
    for index, cell in enumerate(snapshot['native_peaks']):
        present = cell[cell['x'] >= 0].tolist()
        if present:
            decoded.append([dict(x=x, y=y, polarity='bright' if index % 2 == 0 else 'dark',
                                 score=score, response_dn=response, noise_sigma_dn=noise)
                            for x,y,score,response,noise in present])
    exact(decoded, cells, 'native decoded peak cells')
    interleaved = [cell[r] for r in range(12) for cell in cells if r < len(cell)]
    pre_shape = metadata['pre_shape_post_frame_quota']
    exact(interleaved[:512], pre_shape, 'frame quota selection order')
    exact(metadata['post_shape'], [{k:v for k,v in p.items() if k != 'source_xy'} for p in row['candidates']],
          'shape/journal proposals')
    center = metadata['source_probe']['source_xy']
    post_frame = [_proposal_record(p, inverse) for p in pre_shape
                  if p['polarity'] == 'bright' and _in_gate(mapped(inverse, [p['x'],p['y']]), center)]
    actual = []
    for index, p in enumerate(row['candidates']):
        exact(p['source_xy'], mapped(inverse, [p['x'],p['y']]), 'candidate source coordinate')
        if p['polarity'] != 'bright':
            continue
        actual.append(dict(candidate_index=index, identity='candidate:'+str(index),
                           source_distance_px=math.dist(p['source_xy'], center), **p))
    actual.sort(key=lambda p: (p['source_distance_px'], p['candidate_index']))
    gate_candidates = [p for p in actual if p['source_distance_px'] <= GATE]
    nearest = None if not actual else actual[0]
    if nearest is not None:
        nearest = dict(nearest, within_original_gate=nearest['source_distance_px'] <= GATE)
        seeds = nearest.get('shape', {}).get('member_peak_reference_xy', [[nearest['x'], nearest['y']]])
        nearest['shape_seed_lineage'] = []
        for xy in seeds:
            x0,y0,x1,y1 = rect['capture_bounds_exclusive_xyxy']
            x,y = point(xy)
            available = bool(x.is_integer() and y.is_integer() and x0 <= x < x1 and y0 <= y < y1)
            nearest['shape_seed_lineage'].append(dict(reference_xy=xy, capture_available=available,
                diagnostic=None if not available else _pixel(snapshot, metadata, inverse, int(y)-y0, int(x)-x0)))
    measured = [t for t in row['tracks'] if t['measured'] and t['track_id'].startswith('bright:')
                and _in_gate(t['measurement_source_xy'], center)]
    count = lambda m: int(np.count_nonzero(mask & m))
    tile_survive = (snapshot['positive_prequota_rank'] > 0) & (snapshot['positive_prequota_rank'] <= 12)
    counts = dict(integer_pixels_in_original_source_gate=int(mask.sum()),
        current_support=count(flags[..., FLAG['support']].astype(bool)),
        previous_support=count(flags[..., FLAG['previous_support']].astype(bool)),
        eligible=count(eligible), eligible_temporal_pass=count(eligible & temporal),
        eligible_spatial_pass=count(eligible & spatial),
        eligible_both_thresholds_pass=count(eligible & temporal & spatial),
        eligible_both_thresholds_raw_peak_pass=count(expected),
        tile_quota_surviving_seeds=count(tile_survive),
        frame_quota_surviving_seeds=len(post_frame),
        post_shape_candidates_in_original_gate=len(gate_candidates),
        actual_measurements_in_original_gate=len(measured),
        qualified_actual_measurements_in_original_gate=sum(t['qualified_moving'] for t in measured),
        nonfinite_value_pixels=count(~np.isfinite(values).all(axis=2)))
    return dict(frame_index=row['frame_index'], reference=metadata['source_probe'],
        source_to_reference=metadata['source_to_reference'], capture_rectangle=rect,
        gate_counts=counts, gate_counts_are_not_independent_objects=True,
        pixel_seed_and_shape_centroid_counts_are_distinct=True,
        strongest_bright_spatial_pixel=selected_pixel(snapshot, metadata, mask, inverse, 'positive_signed_spatial'),
        strongest_positive_temporal_pixel=selected_pixel(snapshot, metadata, mask, inverse, 'positive_signed_temporal'),
        nearest_actual_bright_candidate=nearest, post_frame_quota_gate_seeds=post_frame,
        post_shape_gate_candidates=gate_candidates, measured_gate_tracks=measured,
        original_assignment_preserved=metadata['source_probe']['original_strict_assigned_identity'],
        selected_diagnostic_pixel_is_object_truth=False, detector_accuracy_claim=False)


def control_counts(rows, controls):
    results = [dict(scope=c, measured=0, predicted=0, ids=set()) for c in controls]
    for row in rows:
        identities = set()
        for t in row['tracks']:
            identity = (t['segment'], t['track_id'])
            require(identity not in identities, 'Duplicate track identity')
            identities.add(identity)
            require(type(t['qualified_moving']) is bool and type(t['measured']) is bool, 'Invalid track flags')
            if not t['qualified_moving']:
                continue
            xy = point(t['measurement_source_xy'] if t['measured'] else t['source_xy'])
            for result in results:
                c = result['scope']
                first,last = c['frames_inclusive']
                x,y,w,h = c['crop_xywh']
                if first <= row['frame_index'] <= last and x <= xy[0] < x+w and y <= xy[1] < y+h:
                    result['measured' if t['measured'] else 'predicted'] += 1
                    result['ids'].add(identity)
    output = []
    for r in results:
        c = r['scope']
        output.append(dict(scope=c, qualified_measured_states=r['measured'], qualified_predicted_states=r['predicted'],
            distinct_segment_track_ids=len(r['ids']), verified_airborne_negative=False,
            matches_archived_scope_counts=(r['measured'] == c['baseline_measured'] and r['predicted'] == c['baseline_predicted'])))
    return dict(windows=output, qualified_measured_states=sum(r['measured'] for r in results),
                qualified_predicted_states=sum(r['predicted'] for r in results),
                all_archived_scope_counts_match=all(r['matches_archived_scope_counts'] for r in output),
                false_positive_rate=None)


class Inputs:
    def __init__(self, root):
        self.root = Path(root).absolute()
        require(self.root.is_dir() and self.root.resolve() == self.root, 'Canonical run directory required')
        self.bound = {}

    def path(self, name, expected=None):
        require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'Unsafe evidence path')
        path = self.root/name
        require(path.is_file() and path.resolve() == path, 'Regular nonsymlink evidence required: '+name)
        digest = sha(path)
        require(expected is None or expected == digest, 'Audited evidence hash mismatch: '+name)
        require(name not in self.bound or self.bound[name] == digest, 'Evidence changed: '+name)
        self.bound[name] = digest
        return path

    def read(self, name, expected=None):
        value = loads(self.path(name, expected).read_text())
        require(type(value) is dict, 'JSON object required')
        return value

    def unchanged(self):
        for name, digest in list(self.bound.items()):
            self.path(name, digest)


def summarize_run(directory):
    data = Inputs(directory)
    audit = data.read('independent_audit.json')
    require(audit.get('schema') == 'seaqr.accuracy-v56-independent-audit.v1' and audit.get('passed') is True
            and audit.get('frames') == FRAME_COUNT and audit.get('capture_frames') == list(FRAMES)
            and audit.get('exact_full_journal_semantics') is True
            and audit.get('exact_private_state_and_learning') is True, 'Successful complete independent audit required')
    bound = audit.get('files_sha256')
    require(type(bound) is dict, 'Missing audit evidence bindings')
    required = ['freeze.json', 'probes.json', 'probe/frames.jsonl']
    required += [f'captures/frame_{f:06d}.{ext}' for f in FRAMES for ext in ('npz', 'json')]
    require(all(name in bound for name in required), 'Audit omits summary input bindings')
    frozen = data.read('freeze.json', bound['freeze.json'])
    probes = data.read('probes.json', bound['probes.json'])
    require(frozen.get('pre_run') is True and frozen['files_sha256']['probes.json'] == bound['probes.json'],
            'Probe manifest not frozen before replay')
    specs = probes['reference_probes']
    require([p['frame_index'] for p in specs] == list(FRAMES) and len(probes['provisional_controls']) == 7,
            'Changed fixed reference/control scope')
    selected, frame_count = {}, 0
    journal = data.path('probe/frames.jsonl', bound['probe/frames.jsonl'])
    def rows():
        nonlocal frame_count
        with journal.open() as stream:
            for index, line in enumerate(stream):
                row = loads(line)
                require(type(row['frame_index']) is int and row['frame_index'] == index, 'Noncontiguous journal')
                frame_count += 1
                if index in FRAMES:
                    selected[index] = row
                yield row
    controls = control_counts(rows(), probes['provisional_controls'])
    require(frame_count == FRAME_COUNT, 'Incomplete journal')
    results = []
    for spec in specs:
        frame = spec['frame_index']
        stem = f'captures/frame_{frame:06d}'
        meta = data.read(stem+'.json', bound[stem+'.json'])
        exact(meta['source_probe'], spec, 'fixed reference probe')
        with np.load(data.path(stem+'.npz', bound[stem+'.npz']), allow_pickle=False) as arrays:
            snapshot = {key: arrays[key] for key in arrays.files}
        results.append(summarize_probe(snapshot, meta, selected[frame]))
    controls['matches_archived_total_70_measured_64_predicted'] = (
        controls['qualified_measured_states'] == 70 and controls['qualified_predicted_states'] == 64)
    data.unchanged()
    return dict(schema='seaqr.accuracy-v56-capture-summary.v1', completed=True,
        created_at_utc=datetime.now(timezone.utc).isoformat(), frames=FRAME_COUNT,
        capture_frames=list(FRAMES), probes=results, provisional_controls=controls,
        independent_parity_audit_passed=True, source_media_accessed=False,
        reference_selected_pixels_are_not_object_truth=True, new_accuracy_claim=False,
        production_change=False, promotion=False, false_positive_rate=None,
        files_sha256=data.bound, summary_source_sha256=sha(__file__))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists() and not args.output.is_symlink(), 'New output file required')
    result = summarize_run(args.run)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
