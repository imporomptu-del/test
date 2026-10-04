"""Independent journal/causality/scoring audit of full_context_01, without video.

No gate or full-runner helpers are imported. Saved feature algebra is validated,
and all 358 previously independently fitted patch observations are crosschecked.
Features for the remaining observations are not independently refitted from pixels.
"""
import argparse
from collections import Counter, defaultdict
from itertools import zip_longest
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
# Reuse only independently written audit/integrity/reference-scoring utilities.
import audit_accuracy_v36 as prior
import audit_accuracy_v36_context as context_audit

BASE = ROOT / 'results/tiny_target/accuracy_v36_20260924'
FULL = BASE / 'full_context_01'
CONTEXT = BASE / 'context_01'
SHADOW = BASE / 'shadow_01'
CONTEXT_RECEIPT = BASE / 'context_independent_audit_01.json'
CONTEXT_RECEIPT_SHA = '88e07ac5581c445c53578a1c8bb2db24ed0d454f5fc075aeaf3e9c5afdd192a2'
FULL_FREEZE = 'f3152e3788bae0f16ece35f33b3835ce38f5af0df0fb9267755b8995865305b3'
ARMS = ('baseline', 'point_context')
IMPLEMENTATION = (
    'scripts/accuracy_v36_context.py', 'scripts/accuracy_v36_context_gate.py',
    'scripts/evaluate_accuracy_v36_full_context.py', 'scripts/evaluate_accuracy_v36.py',
    'scripts/accuracy_v36_policy.py', 'scripts/score_phase20_accuracy.py', 'tiny_target/visible_regression.py',
    'tests/unit/test_accuracy_v36_context.py', 'tests/unit/test_accuracy_v36_context_runner.py',
    'tests/unit/test_accuracy_v36_context_gate.py', 'tests/unit/test_accuracy_v36_full_context.py',
    'docs/accuracy_v36_full_context_plan.md',
)
POLICY = 'informative point-minus-edge > 0; unknown conservatively keeps baseline; coasts inherit latest measured decision'
CAVEAT = ('Output-only shadow eligibility on four development clips; original detection, association, state and learning feedback are unchanged. '
          'Unlabeled workload and selected provisional controls are not false-positive rates. Known moving image features have unknown physical class.')
FEATURE_KEYS = {
    'informative', 'background_residual_energy', 'point_gain_fraction', 'edge_gain_fraction',
    'point_minus_edge_fraction', 'point_amplitude_dn', 'point_sigma_px', 'point_offset_xy',
    'edge_width_px', 'edge_orientation_rad', 'edge_offset_px', 'edge_amplitude_dn', 'residual_rms_dn',
    'point_absolute_gain', 'edge_absolute_gain', 'edge_residual_energy', 'conditional_informative',
    'point_gain_after_edge_fraction', 'point_after_edge_absolute_gain', 'point_after_edge_amplitude_dn',
    'point_after_edge_sigma_px', 'point_after_edge_offset_xy',
}
require, exact, read, sha = prior.require, prior.exact, prior.read, prior.digest


def bind(registry, path, expected=None):
    path = Path(path)
    require(path.resolve().is_relative_to(ROOT.resolve()) and path.suffix in ('.json', '.jsonl', '.py', '.md', '.log', '.npz'),
            'Read-only audit input outside allowed metadata/code scope')
    return registry.bind(path, expected)


def verify_freeze(registry):
    require(isinstance(FULL_FREEZE, str), 'Completed full-context freeze has not yet been pinned')
    bind(registry, FULL / 'freeze.json', FULL_FREEZE)
    bind(registry, CONTEXT_RECEIPT, CONTEXT_RECEIPT_SHA)
    freeze = read(FULL / 'freeze.json')
    receipt = read(CONTEXT_RECEIPT)
    require(receipt['verified'] is True and receipt['schema'] == 'seaqr.accuracy-v36-context-independent-audit.v1'
            and receipt['context_freeze_sha256'] == context_audit.CONTEXT_FREEZE
            and receipt['complete_summary_verified'] is True and receipt['zero_margin_any_alternative_decisions_match'] is True,
            'Verified bounded-context dependency required')
    expected_inputs = {**receipt['checked_files_sha256'], str(CONTEXT_RECEIPT): CONTEXT_RECEIPT_SHA}
    exact(freeze['inputs_sha256'], expected_inputs, 'complete bound full-context input inventory')
    for path, digest in expected_inputs.items():
        bind(registry, path, digest)
    exact(freeze['inputs'], read(SHADOW / 'freeze.json')['inputs'], 'frozen parent journals')
    require(freeze['schema'] == 'seaqr.accuracy-v36-full-context-freeze.v1'
            and freeze['pre_extraction'] is True and freeze['frames'] == 2741 and freeze['patch_size'] == 25
            and freeze['context_audit_sha256'] == CONTEXT_RECEIPT_SHA and freeze['policy'] == POLICY,
            'Full experiment scope/policy changed')
    exact(freeze['clips'], list(prior.COUNTS), 'four-clip scope')
    exact(freeze['arms'], list(ARMS), 'two-arm scope')
    for flag in ('detector_rerun', 'feedback_changed', 'raw16_accessed', 'remote_state_accessed', 'classifier_promoted', 'thresholds_tuned'):
        require(freeze[flag] is False, 'Scope flag changed: ' + flag)
    require(set(freeze['source_videos']) == set(prior.COUNTS), 'Source-video metadata inventory changed')
    for clip, metadata in freeze['source_videos'].items():
        source_dir = ('jetson_review_clips_20260913' if clip in ('0029', '0126')
                      else 'v7_frozen_evaluation_20260913/sources')
        expected_path = ROOT.parent / 'outputs' / source_dir / ('chunk_' + clip + '.avi')
        require(metadata['path'] == str(expected_path) and metadata['sha256'] == prior.SOURCE_HASHES[clip],
                'Unapproved source-video metadata')
    require(set(freeze['implementation_sha256']) == set(IMPLEMENTATION), 'Implementation inventory changed')
    for relative in IMPLEMENTATION:
        bind(registry, ROOT / relative, freeze['implementation_sha256'][relative])
        bind(registry, FULL / 'implementation' / relative, freeze['implementation_sha256'][relative])
    require(freeze['implementation_sha256']['scripts/accuracy_v36_context.py'] ==
            read(CONTEXT / 'freeze.json')['implementation_sha256']['scripts/accuracy_v36_context.py'],
            'Previously audited feature function changed')
    bind(registry, FULL / 'unit.log', freeze['unit_log_sha256'])
    log = (FULL / 'unit.log').read_text()
    tests = re.search(r'Ran (\d+) tests in [0-9.]+s\s+OK\s*$', log)
    require(tests is not None and int(tests.group(1)) == 130, 'Frozen 130-test successful full-context unit log required')
    bind(registry, FULL / 'summary.json')
    summary = read(FULL / 'summary.json')
    require(summary['completed'] is True and summary['freeze_sha256'] == FULL_FREEZE, 'Incomplete full-context run')
    expected_outputs = {clip + suffix for clip in prior.COUNTS for suffix in ('_decisions.jsonl', '_results.json', '_workload.jsonl')}
    require(set(summary['outputs_sha256']) == expected_outputs, 'Output inventory mismatch')
    for name, digest in summary['outputs_sha256'].items():
        bind(registry, FULL / name, digest)
    return freeze, summary, int(tests.group(1))


def near(actual, expected, label):
    require(math.isclose(actual, expected, rel_tol=2e-10, abs_tol=2e-8), 'Feature algebra mismatch: ' + label)


def validate_features(features):
    require(isinstance(features, dict) and set(features) == FEATURE_KEYS, 'Complete feature inventory required')
    for flag in ('informative', 'conditional_informative'):
        require(type(features[flag]) is bool, 'Boolean informativeness required')
    for name, value in features.items():
        if name in ('informative', 'conditional_informative'):
            continue
        if name in ('point_offset_xy', 'point_after_edge_offset_xy'):
            require(isinstance(value, list) and len(value) == 2 and all(type(x) in (int, float) and x in (-1, 0, 1) for x in value),
                    'Invalid point-template offset')
        else:
            require(type(value) in (int, float) and math.isfinite(value), 'Finite numeric feature required: ' + name)
    for gain in ('point_gain_fraction', 'edge_gain_fraction', 'point_gain_after_edge_fraction'):
        require(0 <= features[gain] <= 1, 'Fractional gain outside [0,1]')
    for amplitude in ('point_amplitude_dn', 'point_after_edge_amplitude_dn'):
        require(features[amplitude] >= 0, 'Nonnegative point contrast magnitude required')
    for energy in ('background_residual_energy', 'edge_residual_energy', 'point_absolute_gain',
                   'edge_absolute_gain', 'point_after_edge_absolute_gain', 'residual_rms_dn'):
        require(features[energy] >= 0, 'Negative feature energy/RMS')
    for sigma in ('point_sigma_px', 'point_after_edge_sigma_px'):
        require(features[sigma] in (1, 2, 3), 'Point sigma outside frozen bank')
    require(features['edge_width_px'] in (1, 2, 4) and features['edge_offset_px'] in (-2, 0, 2)
            and features['edge_orientation_rad'] in tuple(k * math.pi / 8 for k in range(8)), 'Edge parameters outside frozen bank')
    energy = features['background_residual_energy']
    edge_energy = features['edge_residual_energy']
    require(features['point_minus_edge_fraction'] == features['point_gain_fraction'] - features['edge_gain_fraction'],
            'Exact point-edge subtraction required; eligibility has no numerical acceptance tolerance')
    near(features['residual_rms_dn']**2, energy / 625, 'residual RMS')
    near(features['point_absolute_gain'], features['point_gain_fraction'] * energy, 'point absolute gain')
    near(features['edge_absolute_gain'], features['edge_gain_fraction'] * energy, 'edge absolute gain')
    near(edge_energy, energy - features['edge_absolute_gain'], 'remaining edge energy')
    near(features['point_after_edge_absolute_gain'], features['point_gain_after_edge_fraction'] * edge_energy, 'conditional absolute gain')
    require(energy <= 625 * 255**2 / 4 + 2e-8, 'Residual energy exceeds possible uint8 variance')
    for flag, local_energy, gains in (
        ('informative', energy, ('point_gain_fraction', 'edge_gain_fraction', 'point_amplitude_dn', 'edge_amplitude_dn')),
        ('conditional_informative', edge_energy, ('point_gain_after_edge_fraction', 'point_after_edge_amplitude_dn')),
    ):
        # Without the patch, the precise centered-energy guard is not available.
        # These bounds follow from its known uint8 range; ambiguous tiny energies
        # are not silently relabeled by this audit.
        if local_energy > 1e-24 * (625 * 255**2 / 4):
            require(features[flag], 'Informative flag contradicts uint8 energy bound')
        if local_energy <= 1e-24:
            require(not features[flag], 'Informative flag contradicts minimum numerical guard')
        if not features[flag]:
            require(all(features[name] == 0 for name in gains), 'Uninformative gains/coefficients must be zero-coded')


class CausalCacheAudit:
    """Rebuild only gate history from original states plus logged current evidence."""
    def __init__(self):
        self.next_frame = 0
        self.segment = None
        self.cache = {}

    def step(self, original, compact):
        frame, segment = original['frame_index'], original['segment']
        require(type(frame) is int and frame == self.next_frame and original['timestamp_ns'] == frame * 100000000,
                'Noncontiguous original journal')
        exact({k: compact[k] for k in ('frame_index', 'timestamp_ns', 'segment')},
              {k: original[k] for k in ('frame_index', 'timestamp_ns', 'segment')}, 'compact frame provenance')
        require(set(compact) == {'frame_index', 'timestamp_ns', 'segment', 'tracks'}, 'Unexpected compact frame fields')
        original_keys = [(t['segment'], t['track_id']) for t in original['tracks']]
        require(len(original_keys) == len(set(original_keys)) and all(key[0] == segment for key in original_keys), 'Invalid original identities')
        relevant = [t for t in original['tracks'] if t['qualified_moving']]
        expected_keys = [(t['segment'], t['track_id']) for t in relevant]
        logged_keys = [(t['segment'], t['track_id']) for t in compact['tracks']]
        require(logged_keys == expected_keys, 'Every baseline-qualified state must appear exactly once, without extras/reordering')
        log = {key: item for key, item in zip(logged_keys, compact['tracks'])}
        if self.segment != segment:
            self.cache.clear()
        self.cache = {key: value for key, value in self.cache.items() if key in set(original_keys)}
        for track in original['tracks']:
            key = (track['segment'], track['track_id'])
            if not track['qualified_moving']:
                if track['measured']:
                    self.cache.pop(key, None)
                continue
            item = log[key]
            copied = ('track_id', 'segment', 'measured', 'source_xy', 'measurement_source_xy')
            require(set(item) == set(copied) | {'accepted', 'reason', 'measurement_frame', 'features'}, 'Unexpected decision record fields')
            exact({k: item[k] for k in copied}, {k: track[k] for k in copied}, 'original track fields')
            require(type(item['accepted']) is bool, 'Boolean gate decision required')
            if track['measured']:
                x, y = (math.floor(v + 0.5) for v in track['measurement_source_xy'])
                truncated = x - 12 < 0 or y - 12 < 0 or x + 12 >= 4784 or y + 12 >= 3190
                if truncated:
                    require(item['features'] is None, 'Truncated measurement cannot claim full patch features')
                    accepted, reason = True, 'unknown_truncated_patch'
                else:
                    validate_features(item['features'])
                    if not item['features']['informative']:
                        accepted, reason = True, 'unknown_uninformative_patch'
                    elif item['features']['point_minus_edge_fraction'] > 0:
                        accepted, reason = True, 'point_preferred'
                    else:
                        accepted, reason = False, 'edge_preferred_or_tie'
                self.cache[key] = (accepted, reason, frame)
                expected_frame = frame
            else:
                require(item['features'] is None, 'A coast cannot create or copy current pixel evidence')
                prior_value = self.cache.get(key)
                if prior_value is None:
                    accepted, reason, expected_frame = True, 'unknown_missing_history', None
                else:
                    accepted, previous_reason, expected_frame = prior_value
                    require(expected_frame < frame, 'Coast provenance must precede current frame')
                    reason = 'coast_' + previous_reason
            exact({k: item[k] for k in ('accepted', 'reason', 'measurement_frame')},
                  dict(accepted=accepted, reason=reason, measurement_frame=expected_frame), 'independent causal eligibility')
        self.next_frame += 1
        self.segment = segment
        return relevant, log


def score_spec(clip, references, anchors):
    wanted = set()
    for labels, packet in references.values():
        windows = {w['id']: w for w in packet['windows']}
        for window in labels['positive_windows']:
            if windows[window['window_id']]['clip_id'] == clip:
                wanted.update(s['frame_index'] for s in window['visible_samples'])
    wanted.update(a['frame_index'] for a in anchors.get(clip, []))
    return wanted


def audit_clip(clip, freeze, references, anchors, controls, scorer, known):
    source = Path(freeze['inputs'][clip]['path']) / 'frames.jsonl'
    known_lookup = {o['key']: o for o in known if o['clip'] == clip}
    known_seen = set()
    clip_controls = controls if clip == '0126' else []
    stats = {arm: prior.empty_counts(clip_controls) for arm in ARMS}
    sparse = {arm: [] for arm in ARMS}
    anchor_rows = {arm: [] for arm in ARMS}
    required = score_spec(clip, references, anchors)
    by_frame = defaultdict(list)
    for anchor in anchors.get(clip, []):
        by_frame[anchor['frame_index']].append(anchor)
    cache = CausalCacheAudit()
    availability = Counter()
    reasons = {'measured': Counter(), 'predicted': Counter()}
    with source.open() as originals, (FULL / (clip + '_decisions.jsonl')).open() as decisions, (FULL / (clip + '_workload.jsonl')).open() as workload:
        for raw, saved, counts in zip_longest(originals, decisions, workload):
            require(raw is not None and saved is not None and counts is not None, 'Original/decisions/workload lengths differ')
            row, compact, logged_counts = prior.decode(raw), prior.decode(saved), prior.decode(counts)
            relevant, log = cache.step(row, compact)
            for track in relevant:
                decision = log[track['segment'], track['track_id']]
                kind = 'measured' if track['measured'] else 'predicted'
                reasons[kind][decision['reason']] += 1
                if track['measured']:
                    features = decision['features']
                    status = 'missing' if features is None else ('informative' if features['informative'] else 'uninformative')
                else:
                    status = 'missing_history' if decision['measurement_frame'] is None else 'inherited_decision'
                availability[kind + '_' + status] += 1
                oid = f'{clip}/{row["frame_index"]}/{track["segment"]}/{track["track_id"]}'
                if oid in known_lookup:
                    require(oid not in known_seen and track['measured'], 'Duplicate/nonmeasured bounded observation')
                    known_seen.add(oid)
                    known_item = known_lookup[oid]
                    exact(track['measurement_source_xy'], known_item['measurement_source_xy'], 'bounded observation coordinate')
                    context_audit.close_tree(decision['features'], known_item['features'], 'bounded independent-fit feature crosscheck')
                    require(decision['accepted'] == known_item['zero_margin_ablation_passed'], 'Bounded context decision changed')
            per_frame = {}
            for arm in ARMS:
                selected = relevant if arm == 'baseline' else [t for t in relevant if log[t['segment'], t['track_id']]['accepted']]
                stat = stats[arm]
                measured = sum(t['measured'] for t in selected)
                predicted = len(selected) - measured
                stat['measured'] += measured
                stat['predicted'] += predicted
                stat['ids'].update((t['segment'], t['track_id']) for t in selected)
                stat['per_frame'].append(measured + predicted)
                per_frame[arm] = dict(frame_index=row['frame_index'], measured=measured, predicted=predicted)
                for control, count in zip(clip_controls, stat['controls']):
                    first, last = control['frames_inclusive']
                    if first <= row['frame_index'] <= last:
                        for track in selected:
                            xy = track['measurement_source_xy'] if track['measured'] else track['source_xy']
                            if prior.inside(xy, control['crop_xywh']):
                                count['measured' if track['measured'] else 'predicted'] += 1
                                count['ids'].add((track['segment'], track['track_id']))
                if row['frame_index'] in required:
                    sparse[arm].append(dict(frame_index=row['frame_index'], segment=row['segment'], coverage=row['coverage'],
                        candidates=[dict(source_xy=p['source_xy'], polarity=p['polarity']) for p in row['candidates']],
                        tracks=[{k: t[k] for k in ('track_id', 'segment', 'measured', 'qualified_moving', 'measurement_source_xy')} for t in selected]))
                for anchor in by_frame.get(row['frame_index'], []):
                    identities = [f'{t["segment"]}/{t["track_id"]}' for t in selected if t['measured']
                        and t['track_id'].split(':')[0] == anchor['polarity']
                        and math.dist(t['measurement_source_xy'], anchor['xy']) <= anchor['uncertainty_px'] + 2]
                    anchor_rows[arm].append(dict(event_id=anchor['event_id'], frame=anchor['frame_index'], ids=identities))
            exact(logged_counts, dict(frame_index=row['frame_index'], arms=per_frame), 'per-frame workload')
    require(cache.next_frame == prior.COUNTS[clip] and known_seen == set(known_lookup), 'Incomplete frames/bounded crosschecks')
    scores = {arm: {name: scorer.score_rows(sparse[arm], labels, packet, clip, 10)
                   for name, (labels, packet) in references.items()} for arm in ARMS}
    result_arms = {}
    for arm in ARMS:
        stat = stats[arm]
        comparisons = {name: prior.retention(scores['baseline'][name], scores[arm][name]) for name in references}
        control_rows = [dict(frames_inclusive=c['frames_inclusive'], crop_xywh=c['crop_xywh'], label=c['label'],
            measured_states=n['measured'], predicted_states=n['predicted'], distinct_segment_track_ids=len(n['ids']))
            for c, n in zip(clip_controls, stat['controls'])]
        require(len(anchor_rows[arm]) == len(anchors.get(clip, [])), 'Required anchors omitted')
        intersections = {}
        for event in sorted({a['event_id'] for a in anchor_rows[arm]}):
            intersections[event] = sorted(set.intersection(*(set(a['ids']) for a in anchor_rows[arm] if a['event_id'] == event)))
        lost = [dict(event_id=a['event_id'], frame=a['frame']) for a, b in zip(anchor_rows['baseline'], anchor_rows[arm]) if a['ids'] and not b['ids']]
        result_arms[arm] = dict(qualified_measured_states=stat['measured'], qualified_predicted_states=stat['predicted'],
            distinct_segment_track_ids=len(stat['ids']), qualified_states_per_frame=(stat['measured'] + stat['predicted']) / cache.next_frame,
            maximum_qualified_states_in_frame=max(stat['per_frame']), provisional_controls=control_rows,
            provisional_control_measured_states=sum(c['measured_states'] for c in control_rows),
            provisional_control_predicted_states=sum(c['predicted_states'] for c in control_rows), references=scores[arm],
            retention=comparisons, required_anchor_evidence=anchor_rows[arm], lost_required_anchors=lost,
            common_id_across_required_anchors=intersections,
            known_reference_retention_passed=bool(all(c['no_new_misses'] and not c['changed_assignments'] for c in comparisons.values())
                and not lost and all(intersections.values())))
    exact(result_arms['baseline'], read(SHADOW / (clip + '_results.json'))['arms']['baseline'], 'entire prior baseline')
    expected = dict(clip=clip, frames=cache.next_frame, arms=result_arms,
        availability=dict(state_evidence_counts=dict(availability), reasons={kind: dict(n) for kind, n in reasons.items()}),
        decisions_sha256=sha(FULL / (clip + '_decisions.jsonl')), airborne_precision=None, airborne_recall=None,
        false_alarms_per_minute=None, detector_rerun=False, feedback_changed=False, source_sha256=prior.SOURCE_HASHES[clip],
        decoder=dict(backend='FFMPEG', fps=10, width=4784, height=3190, reported_frame_count=prior.COUNTS[clip]),
        decoded_frames=cache.next_frame, eof_checked=True, workload_sha256=sha(FULL / (clip + '_workload.jsonl')),
        bounded_context_observations_crosschecked=len(known_seen))
    exact(read(FULL / (clip + '_results.json')), expected, clip + ' full independent result')
    return expected


def totals(results):
    result = {}
    for arm in ARMS:
        result[arm] = {}
        for name in ('dense', 'pilot'):
            samples = [window for r in results.values() for window in r['arms'][arm]['references'][name]['positive_windows']]
            result[arm][name] = dict(samples=sum(w['visible_samples'] for w in samples),
                                    hits=sum(w['qualified_measured_hits'] for w in samples))
        anchors = [a for r in results.values() for a in r['arms'][arm]['required_anchor_evidence']]
        result[arm]['anchors'] = dict(samples=len(anchors), hits=sum(bool(a['ids']) for a in anchors))
        require(result[arm]['dense']['samples'] == 285 and result[arm]['pilot']['samples'] == 28
                and result[arm]['anchors']['samples'] == 24, 'Dropped reference denominator')
    exact(result['baseline'], dict(dense=dict(samples=285, hits=284), pilot=dict(samples=28, hits=28), anchors=dict(samples=24, hits=24)),
          'frozen baseline totals')
    return result


def audit():
    registry = prior.Integrity()
    auditor_hash = bind(registry, Path(__file__))
    test_hash = bind(registry, ROOT / 'tests/unit/test_audit_accuracy_v36_full_context.py')
    freeze, summary, test_count = verify_freeze(registry)
    scorer = prior.load_module(ROOT / 'scripts/score_phase20_accuracy.py', 'full_context_independent_scorer')
    anchor_reader = prior.load_module(ROOT / 'tiny_target/visible_regression.py', 'full_context_independent_anchors')
    references, anchors, controls, _ = prior.read_references(registry, scorer, anchor_reader)
    known = read(CONTEXT / 'observations.json')
    require(len(known) == len({o['key'] for o in known}) == 358, 'Bounded crosscheck inventory changed')
    results = {clip: audit_clip(clip, freeze, references, anchors, controls, scorer, known) for clip in prior.COUNTS}
    reference_totals = totals(results)
    retained = all(r['arms']['point_context']['known_reference_retention_passed'] for r in results.values())
    before = results['0126']['arms']['baseline']['provisional_control_measured_states']
    after = results['0126']['arms']['point_context']['provisional_control_measured_states']
    require(before == 70, 'Baseline provisional control denominator changed')
    expected = dict(schema='seaqr.accuracy-v36-full-context-summary.v1', completed=True, freeze_sha256=FULL_FREEZE,
        outputs_sha256={clip + suffix: sha(FULL / (clip + suffix)) for clip in prior.COUNTS
                        for suffix in ('_decisions.jsonl', '_results.json', '_workload.jsonl')}, frames=2741,
        reference_totals=reference_totals, clips={clip: {k: v for k, v in result.items() if k != 'arms'} |
            {'arms': {arm: {k: v for k, v in values.items() if k not in ('references', 'required_anchor_evidence')}
                      for arm, values in result['arms'].items()}} for clip, result in results.items()},
        known_reference_retention_passed=retained, provisional_control_measured_before=before,
        provisional_control_measured_after=after, eligible_for_further_study=retained and after < before,
        promoted=False, detector_rerun=False, feedback_changed=False, defaults_changed=False,
        thresholds_tuned=False, raw16_accessed=False, remote_state_accessed=False, airborne_precision=None,
        airborne_recall=None, false_alarms_per_minute=None, accuracy_established=False, performance_benchmark=False, caveat=CAVEAT)
    exact(summary, expected, 'entire full-context summary')
    registry.recheck()
    workload = {clip: {arm: {k: result['arms'][arm][k] for k in
        ('qualified_measured_states', 'qualified_predicted_states', 'distinct_segment_track_ids', 'qualified_states_per_frame')}
        for arm in ARMS} for clip, result in results.items()}
    return dict(schema='seaqr.accuracy-v36-full-context-independent-audit.v1', verified=True,
        experiment=str(FULL), full_context_freeze_sha256=FULL_FREEZE, auditor_sha256=auditor_hash,
        auditor_test_sha256=test_hash, checked_file_count=len(registry.files), checked_files_sha256=registry.files,
        all_bound_files_rehashed_after_analysis=True, frozen_full_context_tests_passed=test_count,
        frames=2741, clips=list(prior.COUNTS), every_baseline_qualified_state_verified_once=True,
        copied_identity_coordinates_time_and_measurement_status_verified=True,
        independent_causal_cache_reconstruction=True, gate_or_full_runner_imported=False,
        unknown_evidence_conservatively_retained_verified=True, every_logged_feature_inventory_range_and_algebra_verified=True,
        zero_margin_acceptance_checked_without_tolerance=True,
        qualified_states_verified=sum(r['arms']['baseline']['qualified_measured_states'] +
            r['arms']['baseline']['qualified_predicted_states'] for r in results.values()),
        measured_feature_records_verified=sum(r['availability']['state_evidence_counts'].get('measured_informative', 0) +
            r['availability']['state_evidence_counts'].get('measured_uninformative', 0) for r in results.values()),
        bounded_context_observation_features_crosschecked=358, bounded_patch_fit_independent_parent_audit=CONTEXT_RECEIPT_SHA,
        reference_totals=reference_totals, known_reference_retention_passed=retained,
        provisional_control_measured_before=before, provisional_control_measured_after=after,
        workload=workload, complete_result_workload_and_summary_verified=True,
        full_source_patch_fits_independently_recomputed=False, source_video_decoding_independently_validated=False,
        boundary='No AVI files or original pixels were reopened. The audit reconstructs decisions, history, workload and scoring from frozen journals and logged features. All358 independently fitted saved-patch features match; additional full-clip fits, source decoding and decoder EOF are not independently rerun.',
        video_files_read=False, raw16_accessed=False, remote_state_accessed=False, defaults_changed=False,
        detector_rerun=False, feedback_changed=False, classifier_promoted=False,
        airborne_accuracy_established=False, performance_benchmark=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        raise FileExistsError('Fresh audit receipt required')
    result = audit()
    if args.output is not None:
        with args.output.open('x') as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({k: result[k] for k in ('verified', 'frames', 'reference_totals', 'known_reference_retention_passed',
        'provisional_control_measured_before', 'provisional_control_measured_after', 'workload')}, indent=2))


if __name__ == '__main__':
    main()
