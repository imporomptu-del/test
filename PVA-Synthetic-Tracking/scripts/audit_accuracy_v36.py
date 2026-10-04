"""Independent, read-only audit of the frozen v36 shadow_01 experiment.

Rebuilds decisions from original v34 journals without importing the experiment's
policy or evaluator. Only the hash-verified historical reference scorer is reused.
No media, detector, GPU, remote state or closed-loop feedback is accessed.
"""
import argparse
from collections import defaultdict
import hashlib
import importlib.util
from itertools import zip_longest
import json
import math
from pathlib import Path
import re

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / 'results/tiny_target/accuracy_v36_20260924/shadow_01'
PARENT = ROOT / 'results/tiny_target/visible_validation_v34_20260923/audit_20260924'
FROZEN_DIGEST = '9f1439c9eb9c63c820dfaea571cc7d3b8894cfb28ef6463f93c198602c6c6b70'
COUNTS = {'0029': 687, '0126': 674, '0055': 689, '0082': 691}
ARMS = ('baseline', 'recent_support', 'recent_excursion', 'bounded_shape', 'combined')
SOURCE_HASHES = {
    '0029': '0330bc3e390a793c2bf6afe7b16720ad3cd6bb8943ee162caf9ce6eff800f359',
    '0126': 'c5302b873656793da47f1da3c03f05df595f17c3f9bc407ce0bfd99b7e718344',
    '0055': 'c59abb3dad5c8787aab8a5a86eff466928a918be79a536960a7713d8a5dc539f',
    '0082': '465ba7b00393e655534f78d9be07aa965d6d404c1761b4d0e0841c1766bce117',
}
SOURCES = (
    'scripts/accuracy_v36_policy.py', 'scripts/evaluate_accuracy_v36.py',
    'scripts/score_phase20_accuracy.py', 'tiny_target/visible_regression.py',
    'tests/unit/test_accuracy_v36_policy.py', 'docs/accuracy_v36_plan.md',
)
REFERENCES = {
    'dense_labels': ROOT / 'results/tiny_target/phase20/encounter_accuracy_v2_20260914/annotations.json',
    'dense_packet': ROOT / 'results/tiny_target/phase20/encounter_accuracy_v2_20260914/scoring_packet.json',
    'pilot_labels': ROOT / 'results/tiny_target/phase20/accuracy_baseline_v1_20260914/annotations.json',
    'pilot_packet': ROOT / 'results/tiny_target/phase20/accuracy_baseline_v1_20260914/source_review/packet.json',
    'anchors_0029': ROOT / 'results/tiny_target/phase19/chunk0029_visual_review_20260913/visual_annotations.json',
    'anchors_0126': ROOT / 'results/tiny_target/phase19/chunk0126_avi_20260913/visual_review/visual_annotations.json',
    'controls': ROOT / 'results/tiny_target/phase20/clutter_review_20260913/background_controls.json',
}
POLICY = {'window_hits': 8, 'recent_frame_window': 8, 'minimum_recent_hits': 5,
          'minimum_excursion_px': 12.0}
CAVEAT = ('Shadow output eligibility, not a closed-loop detector rerun. Counts are unlabeled '
          'workload or selected provisional nuisance cases, not false-positive rates.')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'Regular non-symlink file required: ' + str(path))
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def decode(text):
    def reject(value):
        raise ValueError('Nonfinite JSON constant: ' + value)

    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'Duplicate JSON key: ' + key)
            result[key] = value
        return result

    return json.loads(text, parse_constant=reject, object_pairs_hook=unique)


def read(path):
    return decode(Path(path).read_text())


def exact(actual, expected, context):
    # JSON comparison preserves integer/boolean distinction and ignores only
    # dictionary insertion order, not list order, fields or numeric values.
    options = dict(sort_keys=True, separators=(',', ':'), allow_nan=False)
    require(json.dumps(actual, **options) == json.dumps(expected, **options),
            'Independent comparison failed: ' + context)


class Integrity:
    def __init__(self):
        self.files = {}

    def bind(self, path, expected=None):
        path = Path(path)
        current = digest(path)
        if expected is not None:
            require(current == expected, 'SHA256 mismatch: ' + str(path))
        previous = self.files.setdefault(str(path), current)
        require(current == previous, 'File changed during audit: ' + str(path))
        return current

    def recheck(self):
        for path, expected in self.files.items():
            require(digest(path) == expected, 'File changed during audit: ' + path)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_scope(experiment, integrity):
    require(Path(experiment).resolve() == EXPERIMENT.resolve(), 'Only frozen shadow_01 is in scope')
    integrity.bind(EXPERIMENT / 'freeze.json', FROZEN_DIGEST)
    frozen = read(EXPERIMENT / 'freeze.json')
    exact(frozen['policy'], POLICY, 'frozen policy')
    exact(frozen['arms'], list(ARMS), 'frozen arms')
    require(frozen['pre_run'] is True and frozen['detector_feedback_unchanged'] is True,
            'Expected pre-run shadow freeze')
    for name in ('media_decoded', 'raw16_accessed', 'defaults_changed'):
        require(frozen[name] is False, 'Scope changed: ' + name)
    require(set(frozen['source_sha256']) == set(SOURCES), 'Source inventory changed')
    for source in SOURCES:
        expected = frozen['source_sha256'][source]
        integrity.bind(ROOT / source, expected)
        integrity.bind(EXPERIMENT / 'implementation' / source, expected)
    require(set(frozen['references']) == set(REFERENCES)
            and set(frozen['reference_sha256']) == set(REFERENCES), 'Reference inventory changed')
    for name, path in REFERENCES.items():
        require(frozen['references'][name] == str(path), 'Unexpected reference path')
        integrity.bind(path, frozen['reference_sha256'][name])
    integrity.bind(EXPERIMENT / 'unit.log', frozen['unit_log_sha256'])
    log = (EXPERIMENT / 'unit.log').read_text()
    require(re.search(r'Ran 31 tests in [0-9.]+s\s+OK\s*$', log) is not None,
            'Frozen 31-test success log missing')
    integrity.bind(PARENT / 'summary_verified_01.json', frozen['v34_audit_sha256'])
    parent = read(PARENT / 'summary_verified_01.json')
    require(parent['verified'] is True and parent['completed'] is True
            and parent['completed_trials'] == 16, 'Completed verified v34 required')
    integrity.bind(PARENT / 'evidence/export_manifest_v34_01.json', parent['export_manifest_sha256'])
    manifest = read(PARENT / 'evidence/export_manifest_v34_01.json')['files']
    require(set(frozen['inputs']) == set(COUNTS), 'Input clip inventory changed')
    for clip, frames in COUNTS.items():
        spec = frozen['inputs'][clip]
        directory = PARENT / 'evidence/run' / ('full_repeat0_' + clip)
        require(spec['path'] == str(directory) and spec['frames'] == frames
                and spec['source_sha256'] == SOURCE_HASHES[clip], 'Input scope changed: ' + clip)
        suffixes = ('/launch.json', '/report.json', '/frames.jsonl', '.v34.json', '.v29.json')
        expected_paths = {str(PARENT / 'evidence' / ('run/full_repeat0_' + clip + suffix))
                          for suffix in suffixes}
        require(set(spec['files_sha256']) == expected_paths, 'Input file inventory changed')
        for suffix in suffixes:
            relative = 'run/full_repeat0_' + clip + suffix
            path = PARENT / 'evidence' / relative
            require(spec['files_sha256'][str(path)] == manifest[relative], 'Parent manifest conflict')
            integrity.bind(path, manifest[relative])
        launch, report = read(directory / 'launch.json'), read(directory / 'report.json')
        require(launch['fps'] == 10 and launch['expected_frames'] == frames
                and report['frames'] == frames and report['completed'] is True
                and report['full_clip'] is True and launch['source_sha256'] == SOURCE_HASHES[clip]
                and report['source_sha256'] == SOURCE_HASHES[clip], 'Incomplete or wrong video journal')
        cfg = launch['configuration']
        require(cfg['motion_quality_window_hits'] == 8 and cfg['motion_quality_minimum_hits'] == 5
                and cfg['minimum_moving_excursion_px'] == 12
                and cfg['shape_measurement_mode'] == 'mutual_half_height_r8'
                and cfg['learning_protection_geometry'] == 'observed_shape', 'Unexpected parent policy')
        for suffix in ('_decisions.jsonl', '_results.json'):
            integrity.bind(EXPERIMENT / (clip + suffix))
    integrity.bind(EXPERIMENT / 'summary.json')
    return frozen


def read_references(integrity, scorer, anchor_reader):
    references = {}
    denominators = {}
    for name, expected in (('dense', 285), ('pilot', 28)):
        historical = REFERENCES[name + '_labels'].parent / 'scoring_freeze.json'
        integrity.bind(historical)
        old = read(historical)
        integrity.bind(REFERENCES[name + '_labels'], old['labels_sha256'])
        integrity.bind(REFERENCES[name + '_packet'], old['packet_sha256'])
        labels, packet = read(REFERENCES[name + '_labels']), read(REFERENCES[name + '_packet'])
        scorer.validate_labels(labels, packet)
        require(not labels['negative_windows'], 'No verified negative intervals in this experiment')
        source_ids = [s['clip_id'] for s in packet['plan']['sources']]
        require(len(source_ids) == len(set(source_ids)) and set(source_ids) == set(COUNTS),
                'Reference source coverage changed')
        for source in packet['plan']['sources']:
            require(source['sha256'] == SOURCE_HASHES[source['clip_id']], 'Reference source SHA mismatch')
        windows = {w['id']: w for w in packet['windows']}
        require(all(w['clip_id'] in COUNTS for w in windows.values()), 'Reference clip outside allowlist')
        samples = [(w['window_id'], s['frame_index']) for w in labels['positive_windows']
                   for s in w['visible_samples']]
        require(len(samples) == len(set(samples)) == expected, 'Visible reference denominator changed')
        for entry in labels['positive_windows']:
            window = windows[entry['window_id']]
            require(window['clip_id'] in ('0029', '0126'), 'Unexpected positive clip')
            require(all(0 <= sample['frame_index'] < COUNTS[window['clip_id']]
                        for sample in entry['visible_samples']), 'Visible reference outside journal')
        references[name] = (labels, packet)
        denominators[name] = expected
    anchors = {}
    for clip in ('0029', '0126'):
        source_hash, events = anchor_reader.load_reference(REFERENCES['anchors_' + clip])
        require(source_hash == SOURCE_HASHES[clip], 'Anchor source mismatch')
        require(len({e['event_id'] for e in events}) == len(events), 'Duplicate anchor event')
        anchors[clip] = []
        for event in events:
            required = [a for a in event['anchors'] if a['required']]
            require(len(required) >= 2, 'Incomplete required anchor event')
            for item in required:
                require(0 <= item['frame_index'] < COUNTS[clip], 'Anchor outside complete journal')
                anchors[clip].append(dict(event_id=event['event_id'], polarity=event['polarity'], **item))
        require(len({(a['event_id'], a['frame_index']) for a in anchors[clip]}) == len(anchors[clip]),
                'Duplicate required anchor')
    require(sum(map(len, anchors.values())) == 24, 'Required anchor denominator changed')
    controls = read(REFERENCES['controls'])
    require(controls['source_sha256'] == SOURCE_HASHES['0126']
            and controls['human_confirmed'] is False
            and controls['eligible_for_authoritative_false_alarm_rate'] is False
            and len(controls['controls']) == 7, 'Provisional control scope changed')
    for control in controls['controls']:
        first, last = control['frames_inclusive']
        x, y, width, height = control['crop_xywh']
        require(0 <= first <= last < COUNTS['0126'] and width > 0 and height > 0
                and 0 <= x < x + width <= 4784 and 0 <= y < y + height <= 3190,
                'Invalid provisional control coverage')
    return references, anchors, controls['controls'], denominators


def excursion(history, native_grid=False):
    if not history:
        return 0.0
    points = [(round(x), round(y)) if native_grid else (x, y) for _, x, y in history]
    return math.hypot(max(x for x, _ in points) - min(x for x, _ in points),
                      max(y for _, y in points) - min(y for _, y in points))


class IndependentHistory:
    """Own list-based causal reconstruction, not the tested experiment policy."""
    def __init__(self):
        self.frame = 0
        self.segment = None
        self.observations = {}
        self.shapes = {}
        self.measurements = 0
        self.noninteger_roundtrips = 0
        self.maximum_native_grid_error = 0.0
        self.native_grid_decision_differences = 0
        self.evaluated_track_states = 0

    def step(self, row):
        frame, segment = row['frame_index'], row['segment']
        require(type(frame) is int and frame == self.frame
                and type(row['timestamp_ns']) is int and row['timestamp_ns'] == frame * 100000000
                and type(segment) is int and segment >= 0, 'Invalid frame/time/segment sequence')
        matrix = np.asarray(row['source_to_reference'], dtype=float)
        require(matrix.shape == (3, 3) and np.isfinite(matrix).all()
                and abs(np.linalg.det(matrix)) >= 1e-12, 'Invalid reference transform')
        if segment != self.segment:
            self.observations, self.shapes = {}, {}
        present = {(t['segment'], t['track_id']) for t in row['tracks']}
        require(len(present) == len(row['tracks']), 'Duplicate track state')
        self.observations = {k: v for k, v in self.observations.items() if k in present}
        self.shapes = {k: v for k, v in self.shapes.items() if k in present}
        result = {}
        for track in row['tracks']:
            require(track['segment'] == segment and isinstance(track['track_id'], str)
                    and type(track['measured']) is bool and type(track['qualified_moving']) is bool,
                    'Invalid track observation')
            key = (segment, track['track_id'])
            history = self.observations.setdefault(key, [])
            footprint = track.get('learning_shape_reference_xy')
            if footprint is not None:
                require(isinstance(footprint, list) and all(isinstance(p, list) and len(p) == 2
                        and all(type(v) in (int, float) and math.isfinite(v) for v in p)
                        for p in footprint), 'Invalid observed footprint')
            if track['measured']:
                point = track['measurement_source_xy']
                require(isinstance(point, list) and len(point) == 2 and np.isfinite(point).all(),
                        'Invalid raw source observation')
                transformed = matrix @ np.asarray([point[0], point[1], 1.0], dtype=float)
                require(np.isfinite(transformed).all() and abs(transformed[2]) >= 1e-12,
                        'Invalid homogeneous observation')
                x, y = (float(transformed[i] / transformed[2]) for i in (0, 1))
                error = max(abs(x - round(x)), abs(y - round(y)))
                require(error <= 1e-8, 'Measurement is not the frozen native reference pixel grid')
                self.maximum_native_grid_error = max(self.maximum_native_grid_error, error)
                self.noninteger_roundtrips += error != 0.0
                self.measurements += 1
                history.append((frame, x, y))
                del history[:-8]
                self.shapes[key] = bool(footprint)
            else:
                require(track.get('measurement_source_xy') is None, 'Prediction supplied measurement')
            count = sum(frame - observation[0] < 8 for observation in history)
            distance = excursion(history)
            base = track['qualified_moving']
            support = count >= 5
            motion = len(history) >= 5 and distance >= 12.0
            shape = self.shapes.get(key, False)
            decisions = dict(baseline=base, recent_support=base and support,
                             recent_excursion=base and motion, bounded_shape=base and shape,
                             combined=base and support and motion and shape)
            native_motion = len(history) >= 5 and excursion(history, True) >= 12.0
            alternate = dict(baseline=base, recent_support=base and support,
                             recent_excursion=base and native_motion, bounded_shape=base and shape,
                             combined=base and support and native_motion and shape)
            self.native_grid_decision_differences += sum(decisions[a] != alternate[a] for a in ARMS)
            self.evaluated_track_states += 1
            result[key] = dict(decisions=decisions, recent_hits=count, recent_excursion_px=distance,
                               last_measurement_age_frames=frame - history[-1][0] if history else None,
                               bounded_shape=shape)
        self.frame += 1
        self.segment = segment
        return result


def inside(point, crop):
    x, y, width, height = crop
    return x <= point[0] < x + width and y <= point[1] < y + height


def retention(baseline, candidate):
    def records(score):
        items = [((window['window_id'], entry['frame_index']), entry)
                 for window in score['positive_windows'] for entry in window['evidence']]
        require(len(items) == len(dict(items)), 'Duplicate scored reference sample')
        return dict(items)
    before, after = records(baseline), records(candidate)
    require(before.keys() == after.keys(), 'Scored reference denominator changed')
    lost, changed = [], []
    for (window, frame), old in before.items():
        new = after[window, frame]
        if old['qualified_measured_hit'] and not new['qualified_measured_hit']:
            lost.append(dict(window=window, frame=frame))
        elif old['qualified_measured_hit'] and old['assigned_track_id'] != new['assigned_track_id']:
            changed.append(dict(window=window, frame=frame, before=old['assigned_track_id'],
                                after=new['assigned_track_id']))
    return dict(no_new_misses=not lost, lost_visible_samples=lost, changed_assignments=changed,
                baseline_hits=sum(x['qualified_measured_hit'] for x in before.values()),
                candidate_hits=sum(x['qualified_measured_hit'] for x in after.values()), samples=len(before))


def empty_counts(controls):
    return dict(measured=0, predicted=0, ids=set(), per_frame=[],
                controls=[dict(measured=0, predicted=0, ids=set()) for _ in controls])


def audit_clip(clip, frozen, references, anchors, controls, scorer):
    directory = Path(frozen['inputs'][clip]['path'])
    source = directory / 'frames.jsonl'
    log = EXPERIMENT / (clip + '_decisions.jsonl')
    score_frames = set()
    for labels, packet in references.values():
        lookup = {w['id']: w for w in packet['windows']}
        for window in labels['positive_windows']:
            if lookup[window['window_id']]['clip_id'] == clip:
                score_frames.update(sample['frame_index'] for sample in window['visible_samples'])
    clip_anchors = anchors.get(clip, [])
    score_frames.update(a['frame_index'] for a in clip_anchors)
    clip_controls = controls if clip == '0126' else []
    anchor_frames = defaultdict(list)
    for anchor in clip_anchors:
        anchor_frames[anchor['frame_index']].append(anchor)
    history = IndependentHistory()
    statistics = {arm: empty_counts(clip_controls) for arm in ARMS}
    scoring_rows = {arm: [] for arm in ARMS}
    anchor_results = {arm: [] for arm in ARMS}
    with source.open() as originals, log.open() as decisions:
        for raw, saved in zip_longest(originals, decisions):
            require(raw is not None and saved is not None, 'Decision/journal length mismatch: ' + clip)
            row, reported = decode(raw), decode(saved)
            computed = history.step(row)
            eligible = [t for t in row['tracks'] if t['qualified_moving']]
            expected_log = dict(frame_index=row['frame_index'], timestamp_ns=row['timestamp_ns'],
                                segment=row['segment'], tracks=[])
            for track in eligible:
                expected_log['tracks'].append(dict(track_id=track['track_id'], measured=track['measured'],
                    source_xy=track['source_xy'], measurement_source_xy=track['measurement_source_xy'],
                    **computed[track['segment'], track['track_id']]))
            exact(reported, expected_log, clip + ' decision frame ' + str(row['frame_index']))
            for arm in ARMS:
                accepted = [track for track in eligible
                            if computed[track['segment'], track['track_id']]['decisions'][arm]]
                stat = statistics[arm]
                stat['per_frame'].append(len(accepted))
                for track in accepted:
                    kind = 'measured' if track['measured'] else 'predicted'
                    stat[kind] += 1
                    identity = (track['segment'], track['track_id'])
                    stat['ids'].add(identity)
                    point = track['measurement_source_xy'] if track['measured'] else track['source_xy']
                    for index, control in enumerate(clip_controls):
                        first, last = control['frames_inclusive']
                        if first <= row['frame_index'] <= last and inside(point, control['crop_xywh']):
                            stat['controls'][index][kind] += 1
                            stat['controls'][index]['ids'].add(identity)
                if row['frame_index'] in score_frames:
                    scoring_rows[arm].append(dict(frame_index=row['frame_index'], segment=row['segment'],
                        coverage=row['coverage'], candidates=[dict(source_xy=p['source_xy'], polarity=p['polarity'])
                            for p in row['candidates']], tracks=[{name: t[name] for name in
                            ('track_id', 'segment', 'measured', 'qualified_moving', 'measurement_source_xy')}
                            for t in accepted]))
                for anchor in anchor_frames.get(row['frame_index'], []):
                    matched = []
                    for track in accepted:
                        if (track['measured'] and track['track_id'].split(':')[0] == anchor['polarity']
                                and math.dist(track['measurement_source_xy'], anchor['xy']) <= anchor['uncertainty_px'] + 2):
                            matched.append(str(track['segment']) + '/' + track['track_id'])
                    anchor_results[arm].append(dict(event_id=anchor['event_id'], frame=anchor['frame_index'], ids=matched))
    require(history.frame == COUNTS[clip], 'Incomplete source clip: ' + clip)
    require(history.native_grid_decision_differences == 0, 'Native-grid reconstruction changed eligibility')
    scores = {arm: {name: scorer.score_rows(scoring_rows[arm], labels, packet, clip, 10)
                   for name, (labels, packet) in references.items()} for arm in ARMS}
    expected_arms = {}
    for arm in ARMS:
        stat = statistics[arm]
        require(len(anchor_results[arm]) == len(clip_anchors), 'Dropped required anchors')
        control_counts = [dict(frames_inclusive=control['frames_inclusive'], crop_xywh=control['crop_xywh'],
                              label=control['label'], measured_states=counts['measured'],
                              predicted_states=counts['predicted'], distinct_segment_track_ids=len(counts['ids']))
                          for control, counts in zip(clip_controls, stat['controls'])]
        comparisons = {name: retention(scores['baseline'][name], scores[arm][name]) for name in references}
        base_anchors = {(a['event_id'], a['frame']): a for a in anchor_results['baseline']}
        lost_anchors = [dict(event_id=a['event_id'], frame=a['frame']) for a in anchor_results[arm]
                        if base_anchors[a['event_id'], a['frame']]['ids'] and not a['ids']]
        by_event = defaultdict(list)
        for item in anchor_results[arm]:
            by_event[item['event_id']].append(set(item['ids']))
        intersections = {event: sorted(set.intersection(*sets)) for event, sets in by_event.items()}
        passed = (all(item['no_new_misses'] and not item['changed_assignments'] for item in comparisons.values())
                  and not lost_anchors and all(intersections.values()))
        expected_arms[arm] = dict(qualified_measured_states=stat['measured'], qualified_predicted_states=stat['predicted'],
            distinct_segment_track_ids=len(stat['ids']), qualified_states_per_frame=(stat['measured'] + stat['predicted']) / history.frame,
            maximum_qualified_states_in_frame=max(stat['per_frame']), provisional_controls=control_counts,
            provisional_control_measured_states=sum(c['measured_states'] for c in control_counts),
            provisional_control_predicted_states=sum(c['predicted_states'] for c in control_counts),
            references=scores[arm], retention=comparisons, required_anchor_evidence=anchor_results[arm],
            lost_required_anchors=lost_anchors, common_id_across_required_anchors=intersections,
            known_reference_retention_passed=bool(passed))
    expected = dict(clip=clip, frames=history.frame, source_sha256=SOURCE_HASHES[clip], arms=expected_arms,
                    decisions_sha256=digest(log), airborne_precision=None, airborne_recall=None,
                    false_alarms_per_minute=None, detector_rerun=False, feedback_changed=False)
    exact(read(EXPERIMENT / (clip + '_results.json')), expected, clip + ' complete results and reference scores')
    diagnostics = dict(frames=history.frame, track_states_evaluated=history.evaluated_track_states,
                       measured_states_mapped=history.measurements, noninteger_roundtrips=history.noninteger_roundtrips,
                       maximum_native_grid_roundtrip_error_px=history.maximum_native_grid_error,
                       native_grid_algorithmic_decision_differences=history.native_grid_decision_differences,
                       labeled_positive_scope=clip in ('0029', '0126'),
                       unlabeled_workload_only=clip in ('0055', '0082'))
    return expected, diagnostics


def global_totals(results):
    totals = {}
    for arm in ARMS:
        totals[arm] = {}
        for name in ('dense', 'pilot'):
            comparisons = [r['arms'][arm]['retention'][name] for r in results.values()]
            totals[arm][name] = dict(samples=sum(x['samples'] for x in comparisons),
                hits=sum(x['candidate_hits'] for x in comparisons),
                new_misses=sum(len(x['lost_visible_samples']) for x in comparisons),
                changed_assignments=sum(len(x['changed_assignments']) for x in comparisons))
        evidence = [a for result in results.values() for a in result['arms'][arm]['required_anchor_evidence']]
        totals[arm]['anchors'] = dict(samples=len(evidence), hits=sum(bool(a['ids']) for a in evidence))
        require(totals[arm]['dense']['samples'] == 285 and totals[arm]['pilot']['samples'] == 28
                and totals[arm]['anchors']['samples'] == 24, 'Global reference denominator changed')
    require(totals['baseline']['dense']['hits'] == 284 and totals['baseline']['pilot']['hits'] == 28
            and totals['baseline']['anchors']['hits'] == 24, 'Frozen baseline reference score changed')
    require(results['0126']['arms']['baseline']['provisional_control_measured_states'] == 70,
            'Baseline provisional control count changed')
    return totals


def audit(experiment=EXPERIMENT):
    integrity = Integrity()
    auditor_hash = integrity.bind(Path(__file__))
    auditor_test_hash = integrity.bind(ROOT / 'tests/unit/test_audit_accuracy_v36.py')
    frozen = check_scope(experiment, integrity)
    scorer = load_module(ROOT / 'scripts/score_phase20_accuracy.py', 'v36_audited_reference_scorer')
    anchor_reader = load_module(ROOT / 'tiny_target/visible_regression.py', 'v36_audited_anchor_reader')
    references, anchors, controls, denominators = read_references(integrity, scorer, anchor_reader)
    results, diagnostics = {}, {}
    for clip in COUNTS:
        results[clip], diagnostics[clip] = audit_clip(clip, frozen, references, anchors, controls, scorer)
    totals = global_totals(results)
    gates = {}
    for arm in ARMS[1:]:
        passed = all(r['arms'][arm]['known_reference_retention_passed'] for r in results.values())
        count = results['0126']['arms'][arm]['provisional_control_measured_states']
        gates[arm] = dict(known_reference_retention_passed=passed, control_measured_before=70,
                         control_measured_after=count, eligible_for_further_study=passed and count < 70,
                         promoted=False, reason='known_reference_regression' if not passed else 'further_source_review_required')
    clips = {clip: dict(frames=result['frames'], arms={arm: {key: value for key, value in values.items()
             if key not in ('references', 'required_anchor_evidence')} for arm, values in result['arms'].items()})
             for clip, result in results.items()}
    expected_summary = dict(completed=True, freeze_sha256=FROZEN_DIGEST, clips=clips, gates=gates,
        defaults_changed=False, media_decoded=False, raw16_accessed=False, airborne_accuracy_established=False, caveat=CAVEAT)
    exact(read(EXPERIMENT / 'summary.json'), expected_summary, 'complete experiment summary')
    integrity.recheck()
    return dict(schema='seaqr.accuracy-v36-independent-audit.v1', verified=True, experiment=str(EXPERIMENT),
        experiment_freeze_sha256=FROZEN_DIGEST, auditor_sha256=auditor_hash,
        auditor_test_sha256=auditor_test_hash, original_reference_freezes_verified=True,
        frozen_code_reference_journal_and_test_hashes_verified=True,
        all_bound_files_rehashed_after_analysis=True, checked_file_count=len(integrity.files),
        checked_files_sha256=integrity.files, frozen_policy_tests_passed=31,
        independent_policy_recomputation=True, candidate_policy_imported=False,
        comparison='Every decision-log record, diagnostic, complete result/reference score and summary field matches.',
        frames=sum(COUNTS.values()), clips=diagnostics, reference_totals=totals, gates=gates,
        bounded_controls=dict(clip='0126', count=7, roi_frame_instances=sum(c['frames_inclusive'][1] - c['frames_inclusive'][0] + 1 for c in controls),
            baseline_measured_states=70, authoritative_negative_labels=False, false_alarms_per_minute=None),
        native_grid_validation=dict(algorithmic_decision_differences=sum(d['native_grid_algorithmic_decision_differences'] for d in diagnostics.values()),
            note='Separate diagnostic comparison to nearest native reference pixels; recorded policy decisions are not changed or rounded.'),
        raw16_accessed=False, media_decoded=False, source_video_files_read=False,
        source_video_identity='Checked against frozen audited v34 source SHA metadata; source footage is not read.',
        remote_state_accessed=False, detector_rerun=False,
        defaults_changed=False, feedback_changed=False, airborne_accuracy_established=False,
        caveat='This independently verifies shadow output ablations on development journals, not operational airborne accuracy or a closed-loop improvement.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Optional new JSON receipt; existing files are never overwritten')
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        raise FileExistsError('Fresh audit output required')
    result = audit()
    if args.output is not None:
        with args.output.open('x') as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({k: result[k] for k in ('verified', 'frames', 'reference_totals', 'gates', 'native_grid_validation')}, indent=2))


if __name__ == '__main__':
    main()
