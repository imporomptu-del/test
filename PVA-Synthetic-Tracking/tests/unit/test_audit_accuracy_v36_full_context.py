"""Independent full-context causal/log audit tests; no images or videos read."""
import ast
import copy
import importlib.util
import math
from pathlib import Path
import unittest

PATH = Path(__file__).resolve().parents[2] / 'scripts/audit_accuracy_v36_full_context.py'
SPEC = importlib.util.spec_from_file_location('independent_full_context_audit_under_test', PATH)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def features(pass_=True, informative=True):
    energy = 100.0 if informative else 0.0
    pg = (0.6 if pass_ else 0.1) if informative else 0.0
    eg = (0.2 if pass_ else 0.6) if informative else 0.0
    edge_energy = energy * (1 - eg)
    cg = 0.75 if informative else 0.0
    return dict(informative=informative, background_residual_energy=energy,
        point_gain_fraction=pg, edge_gain_fraction=eg, point_minus_edge_fraction=pg - eg,
        point_amplitude_dn=2.0 if informative else 0.0, point_sigma_px=1.0, point_offset_xy=[0.0, 0.0],
        edge_width_px=1.0, edge_orientation_rad=0.0, edge_offset_px=0.0,
        edge_amplitude_dn=1.0 if informative else 0.0, residual_rms_dn=math.sqrt(energy / 625),
        point_absolute_gain=pg * energy, edge_absolute_gain=eg * energy, edge_residual_energy=edge_energy,
        conditional_informative=informative, point_gain_after_edge_fraction=cg,
        point_after_edge_absolute_gain=cg * edge_energy,
        point_after_edge_amplitude_dn=2.0 if informative else 0.0,
        point_after_edge_sigma_px=1.0, point_after_edge_offset_xy=[0.0, 0.0])


def track(*, measured=True, qualified=True, x=20, y=30, segment=0, tid='bright:1'):
    return dict(track_id=tid, segment=segment, measured=measured, qualified_moving=qualified,
                source_xy=[21.0, 31.0], measurement_source_xy=[x, y] if measured else None)


def row(frame, tracks=None, segment=0):
    return dict(frame_index=frame, timestamp_ns=frame * 100000000, segment=segment,
                tracks=[track(segment=segment)] if tracks is None else tracks)


def compact(original, accepted=True, reason='point_preferred', measurement_frame=None, feature=None):
    measured_frame = original['frame_index'] if measurement_frame is None else measurement_frame
    values = dict(frame_index=original['frame_index'], timestamp_ns=original['timestamp_ns'], segment=original['segment'], tracks=[])
    for item in original['tracks']:
        if item['qualified_moving']:
            values['tracks'].append(dict(**{k: item[k] for k in ('track_id', 'segment', 'measured', 'source_xy', 'measurement_source_xy')},
                accepted=accepted, reason=reason, measurement_frame=measured_frame,
                features=feature))
    return values


def missing_history(original):
    result = compact(original, reason='unknown_missing_history', feature=None)
    for item in result['tracks']:
        item['measurement_frame'] = None
    return result


class FullContextAuditTests(unittest.TestCase):
    def test_no_gate_runner_or_feature_import(self):
        tree = ast.parse(PATH.read_text())
        imports = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        imports += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
        for forbidden in ('accuracy_v36_context_gate', 'evaluate_accuracy_v36_full_context', 'accuracy_v36_context'):
            self.assertNotIn(forbidden, imports)

    def test_complete_feature_algebra_accepts_valid_informative_and_zero_case(self):
        audit.validate_features(features())
        audit.validate_features(features(False))
        audit.validate_features(features(informative=False))

    def test_feature_inventory_and_nonfinite_fields_rejected(self):
        for change in ('missing', 'extra', 'nan', 'boolean'):
            value = features()
            if change == 'missing':
                del value['point_sigma_px']
            elif change == 'extra':
                value['confidence'] = 1.0
            elif change == 'nan':
                value['edge_amplitude_dn'] = math.nan
            else:
                value['point_amplitude_dn'] = True
            with self.subTest(change=change), self.assertRaises(ValueError):
                audit.validate_features(value)

    def test_wrong_gain_range_or_margin_cannot_pass(self):
        for name, value in (('point_gain_fraction', 1.1), ('point_minus_edge_fraction', -0.4),
                            ('point_absolute_gain', 55.0), ('residual_rms_dn', 1.0),
                            ('edge_residual_energy', 70.0), ('point_after_edge_absolute_gain', 1.0)):
            bad = features()
            bad[name] = value
            with self.subTest(name=name), self.assertRaises(ValueError):
                audit.validate_features(bad)

    def test_template_parameters_outside_frozen_bank_rejected(self):
        for name, value in (('point_sigma_px', 4.0), ('edge_width_px', 3.0), ('point_offset_xy', [2.0, 0.0]),
                            ('point_after_edge_offset_xy', [0.0]), ('edge_orientation_rad', 0.123)):
            bad = features()
            bad[name] = value
            with self.subTest(name=name), self.assertRaises(ValueError):
                audit.validate_features(bad)

    def test_numerical_validation_tolerance_cannot_flip_zero_margin_decision(self):
        bad = features()
        bad['point_gain_fraction'] = bad['edge_gain_fraction']
        bad['point_absolute_gain'] = bad['edge_absolute_gain']
        bad['point_minus_edge_fraction'] = 1e-15
        with self.assertRaises(ValueError):
            audit.validate_features(bad)

    def test_false_informative_flag_cannot_hide_real_energy(self):
        bad = features()
        bad['informative'] = False
        with self.assertRaises(ValueError):
            audit.validate_features(bad)
        bad = features(informative=False)
        bad['informative'] = True
        with self.assertRaises(ValueError):
            audit.validate_features(bad)

    def test_qualified_measurement_and_coast_inherit_rejection(self):
        engine = audit.CausalCacheAudit()
        first = row(0)
        engine.step(first, compact(first, accepted=False, reason='edge_preferred_or_tie', feature=features(False)))
        second = row(1, [track(measured=False)])
        engine.step(second, compact(second, accepted=False, reason='coast_edge_preferred_or_tie', measurement_frame=0))
        self.assertEqual(engine.cache[(0, 'bright:1')], (False, 'edge_preferred_or_tie', 0))

    def test_unknown_missing_history_conservatively_preserves_baseline(self):
        original = row(0, [track(measured=False)])
        audit.CausalCacheAudit().step(original, missing_history(original))
        bad = missing_history(original)
        bad['tracks'][0]['accepted'] = False
        with self.assertRaises(ValueError):
            audit.CausalCacheAudit().step(original, bad)

    def test_truncated_patch_is_retained_and_coast_remembers_reason(self):
        engine = audit.CausalCacheAudit()
        first = row(0, [track(x=0)])
        engine.step(first, compact(first, reason='unknown_truncated_patch'))
        second = row(1, [track(measured=False)])
        engine.step(second, compact(second, reason='coast_unknown_truncated_patch', measurement_frame=0))

    def test_uninformative_patch_retains_baseline(self):
        original = row(0)
        audit.CausalCacheAudit().step(original, compact(original, reason='unknown_uninformative_patch', feature=features(informative=False)))
        with self.assertRaises(ValueError):
            audit.CausalCacheAudit().step(original, compact(original, accepted=False, reason='edge_preferred_or_tie', feature=features(informative=False)))

    def test_measured_unqualified_state_clears_cached_decision(self):
        engine = audit.CausalCacheAudit()
        original = row(0)
        engine.step(original, compact(original, accepted=False, reason='edge_preferred_or_tie', feature=features(False)))
        unqualified = row(1, [track(qualified=False)])
        engine.step(unqualified, compact(unqualified))
        coast = row(2, [track(measured=False)])
        engine.step(coast, missing_history(coast))

    def test_unqualified_prediction_does_not_create_new_evidence_or_erase_cache(self):
        engine = audit.CausalCacheAudit()
        first = row(0)
        engine.step(first, compact(first, feature=features()))
        unqualified = row(1, [track(qualified=False, measured=False)])
        engine.step(unqualified, compact(unqualified))
        coast = row(2, [track(measured=False)])
        engine.step(coast, compact(coast, reason='coast_point_preferred', measurement_frame=0))

    def test_deletion_prunes_cache(self):
        engine = audit.CausalCacheAudit()
        first = row(0)
        engine.step(first, compact(first, feature=features()))
        deleted = row(1, [])
        engine.step(deleted, compact(deleted))
        reappeared = row(2, [track(measured=False)])
        engine.step(reappeared, missing_history(reappeared))

    def test_segment_change_cannot_inherit_previous_segment_history(self):
        engine = audit.CausalCacheAudit()
        first = row(0)
        engine.step(first, compact(first, feature=features()))
        reset = row(1, [track(measured=False, segment=1)], segment=1)
        engine.step(reset, missing_history(reset))

    def test_predicted_state_cannot_copy_features_or_invent_current_measurement(self):
        for modification in ('features', 'measurement_frame'):
            engine = audit.CausalCacheAudit()
            first = row(0)
            engine.step(first, compact(first, feature=features()))
            second = row(1, [track(measured=False)])
            logged = compact(second, reason='coast_point_preferred', measurement_frame=0)
            logged['tracks'][0][modification] = features() if modification == 'features' else 1
            with self.subTest(modification=modification), self.assertRaises(ValueError):
                engine.step(second, logged)

    def test_original_copied_coordinates_identity_status_and_time_are_immutable(self):
        for field, replacement in (('measurement_source_xy', [20.1, 30]), ('source_xy', [22, 31]),
                                    ('track_id', 'bright:2'), ('measured', False)):
            original = row(0)
            logged = compact(original, feature=features())
            logged['tracks'][0][field] = replacement
            with self.subTest(field=field), self.assertRaises(ValueError):
                audit.CausalCacheAudit().step(original, logged)
        original = row(0)
        logged = compact(original, feature=features())
        logged['timestamp_ns'] = 1
        with self.assertRaises(ValueError):
            audit.CausalCacheAudit().step(original, logged)

    def test_missing_duplicate_or_extra_baseline_states_fail(self):
        for kind in ('missing', 'duplicate', 'extra'):
            original = row(0)
            logged = compact(original, feature=features())
            if kind == 'missing':
                logged['tracks'] = []
            elif kind == 'duplicate':
                logged['tracks'] *= 2
            else:
                other = copy.deepcopy(logged['tracks'][0])
                other['track_id'] = 'dark:2'
                logged['tracks'].append(other)
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                audit.CausalCacheAudit().step(original, logged)

    def test_future_data_does_not_change_previously_rebuilt_cache(self):
        a, b = audit.CausalCacheAudit(), audit.CausalCacheAudit()
        for frame in range(3):
            original = row(frame)
            saved = compact(original, feature=features())
            a.step(original, saved)
            b.step(original, saved)
        snapshot = copy.deepcopy(a.cache)
        for frame in range(3, 8):
            original = row(frame)
            b.step(original, compact(original, accepted=False, reason='edge_preferred_or_tie', feature=features(False)))
        self.assertEqual(a.cache, snapshot)
        self.assertNotEqual(a.cache, b.cache)


if __name__ == '__main__':
    unittest.main()
