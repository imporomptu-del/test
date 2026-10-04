"""Generated-only independent-auditor regressions; no experiment artifacts read."""
import ast
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts'))
import audit_accuracy_v54_sources as audit
import accuracy_v54_adapter as adapter
import accuracy_v54_benchmark as benchmark
import run_accuracy_v54_sources as runner


def specification(**fields):
    defaults = dict(condition='stable', motion='appearing', amplitude=4,
                    background='constant', noise_level=0, missingness=None)
    defaults.update(fields)
    return next(s for s in audit.specifications() if all(s[k] == v for k, v in defaults.items()))


def row_for(**fields):
    return audit.generated_input(specification(**fields))


def score(row):
    return audit.expected_score(row, audit.reconstructed_prediction(row))


def refresh_forecast(value):
    value['prediction_sha256'] = audit.digest(value, ('prediction_sha256',))
    return value


class IndependentSourcesAuditTests(unittest.TestCase):
    def test_literal_scope_and_inherited_pins(self):
        self.assertEqual(audit.SOURCE_NAMES, runner.SOURCE_NAMES)
        self.assertEqual(len(set(audit.SOURCE_NAMES)), 11)
        self.assertEqual(audit.INHERITED_SHA256, runner.INHERITED_SHA256)
        audit.verify_inherited_sources(audit.ROOT)

    def test_no_producer_or_prior_auditor_imports(self):
        tree = ast.parse(Path(audit.__file__).read_text())
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(n.name for n in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.append(node.module)
        self.assertEqual(set(imported), {'argparse', 'collections', 'datetime', 'fractions',
                                        'hashlib', 'json', 'math', 'pathlib', 'numpy'})

    def test_independent_exact_ordered_420_specs(self):
        actual = benchmark.specifications()
        audit.compare(audit.specifications(), actual, rtol=0, atol=0)
        self.assertEqual(len({s['case_id'] for s in actual}), 420)
        self.assertEqual(Counter(s['stratum'] for s in actual), dict(factorial=416, availability=4))
        self.assertEqual(Counter(s['condition'] for s in actual[:416]), {k: 52 for k in audit.CONDITIONS})
        self.assertEqual([s['missingness'] for s in actual[-4:]], list(audit.MISSINGNESS))

    def test_generated_arrays_all_conditions_motions_profiles_and_missingness(self):
        # Inputs only: covers all formulas without executing the 420-case run.
        choices = [specification(condition=c, motion='turning', amplitude=-4,
                                  background='textured', noise_level=.5) for c in audit.CONDITIONS]
        choices += [specification(motion=m, amplitude=a) for m in audit.MOTIONS for a in (4, -4)]
        choices += [specification(motion='absent', amplitude=0)] + audit.specifications()[-4:]
        for spec in choices:
            with self.subTest(case=spec['case_id']):
                audit.check_generated_input(spec, runner.prepare_input(spec))

    def test_exact_truth_tiny_template_support_tamper_rejected(self):
        spec = specification(); row = audit.generated_input(spec)
        row['truth']['source_template'][0] = 1e-13
        with self.assertRaisesRegex(ValueError, 'exact analytic truth'):
            audit.check_generated_input(spec, row)

    def test_exact_source_and_contamination_epsilon_tamper_rejected(self):
        for field in ('current_source', 'guard_contamination_current'):
            row = row_for(); row['truth'][field][0] += 1e-13
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'exact analytic truth'):
                audit.check_generated_input(specification(), row)

    def test_input_metadata_and_missingness_tampering_rejected(self):
        for change in ('label', 'missing', 'order', 'extra'):
            row = row_for()
            if change == 'label': row['spec']['motion'] = 'stationary'
            elif change == 'missing': row['observed']['core_current_on'][0] = None
            elif change == 'order': row['observed']['guard_xy'].reverse()
            else: row['truth']['unexpected'] = 0
            with self.subTest(change=change), self.assertRaises(ValueError):
                audit.check_generated_input(specification(), row)

    def test_current_tent_has_nine_positive_points_and_guard_disjoint(self):
        row = row_for(); p = np.array(row['truth']['source_template'])
        self.assertEqual(np.count_nonzero(p), 9)
        self.assertEqual(p.sum(), 4); self.assertEqual(p@p, 2.25)
        self.assertEqual(set(audit.CORE_XY) & set(audit.GUARD_XY), set())
        for motion in audit.MOTIONS:
            for time in range(-8, 1):
                self.assertEqual(np.count_nonzero(audit.template(audit.GUARD_XY, audit.source_center(motion, time))), 0)

    def test_missing_current_does_not_erase_known_truth(self):
        spec = audit.specifications()[-1]; row = audit.generated_input(spec)
        self.assertIsNone(row['observed']['core_current_on'][312])
        self.assertIsNone(row['observed']['core_current_off'][312])
        self.assertEqual(row['truth']['current_source'][312], 4)
        self.assertTrue(all(math.isfinite(x) for x in row['truth']['clean_current_background']))

    def test_constants_independently_match_frozen_adapter(self):
        audit.compare(audit.adapter_constants(), adapter.model_constants(), rtol=0, atol=0)
        audit.compare(audit.evaluation_constants(), runner.plain(runner.evaluation_constants()), rtol=0, atol=0)

    def test_complete_predictions_exact_for_representative_generated_cases(self):
        specs = [specification(), specification(motion='stationary', amplitude=-4),
                 specification(condition='local_guard_minus16', motion='slow_linear', background='textured', noise_level=.5),
                 specification(condition='recent_plus8', motion='move_stop', background='textured', noise_level=.5)]
        specs += audit.specifications()[-4:]
        for spec in specs:
            with self.subTest(case=spec['case_id']):
                row = runner.prepare_input(spec)
                audit.check_prediction(row, runner.predict_input(row)['forecast'])

    def test_scalar_fit_exact_midpoint_and_order_certificate(self):
        fit = audit.reconstructed_fit(audit.GUARD_XY[:4], [0.]*4, [1., 3., 7., 9.])
        self.assertEqual(fit['offset_dn'], 5)
        self.assertEqual(fit['median_interval_dn'], [3., 7.])
        self.assertEqual(fit['objective_mae_dn'], 3)
        self.assertEqual(fit['subgradient_interval'], [0., 0.])
        self.assertEqual(fit['residual_order_counts'], dict(below=2, above=2, tied=0))
        audit.compare(fit, audit.plain(adapter._V53.fit(np.array(audit.GUARD_XY[:4]), [0.]*4, [1., 3., 7., 9.])), rtol=0, atol=0)

    def test_exact_subnormal_midpoint_does_not_double_round(self):
        smallest = np.nextafter(0., 1.)
        values = [0., smallest, smallest*2, smallest*3]
        fit = audit.reconstructed_fit(audit.GUARD_XY[:4], [0.]*4, values)
        self.assertEqual(fit['offset_dn'], smallest*2)
        audit.compare(fit, audit.plain(adapter._V53.fit(np.array(audit.GUARD_XY[:4]), [0.]*4, values)), rtol=0, atol=0)

    def test_insufficient_training_rows_remain_unavailable(self):
        fit = audit.reconstructed_fit(audit.GUARD_XY[:4], [0.]*4, [1., 1., 1., None])
        self.assertEqual(fit['training_used_count'], 3)
        self.assertEqual(fit['unavailable_reason'], 'insufficient_finite_training_rows')
        self.assertIsNone(fit['offset_dn']); self.assertFalse(fit['available'])

    def test_finite_input_residual_overflow_invalidates_entire_fit(self):
        fit = audit.reconstructed_fit(audit.GUARD_XY[:4], [-1e308, 0., 0., 0.], [1e308, 0., 0., 0.])
        self.assertEqual(fit['training_used_count'], 4)
        self.assertEqual(fit['unavailable_reason'], 'nonfinite_training_residual_arithmetic')
        self.assertFalse(fit['available'])

    def test_objective_overflow_preserves_diagnostic_candidate_only(self):
        fit = audit.reconstructed_fit(audit.GUARD_XY[:4], [0.]*4, [-1e308, 1e308, 1e308, 1e308])
        self.assertEqual(fit['candidate_offset_dn'], 1e308)
        self.assertIsNone(fit['offset_dn'])
        self.assertEqual(fit['unavailable_reason'], 'nonfinite_objective_arithmetic')

    def test_strict_median_arithmetic_and_missingness(self):
        history = np.array([[1e308, 0., 1.]]*8)
        history[0, 1] = np.nan
        result = audit.strict_median(history, 8)
        self.assertEqual(result['values'], [None, None, 1.])
        self.assertEqual(result['unavailable_reasons'], ['nonfinite_median_arithmetic', 'nonfinite_history', None])
        self.assertEqual(audit.strict_median(history, 3)['values'], [1e308, 0., 1.])

    def test_recomputed_prediction_hash_cannot_hide_model_tampering(self):
        row = row_for(); original = audit.reconstructed_prediction(row)
        for field in ('offset_dn', 'training_used_count', 'training_input_sha256', 'residual_order_counts'):
            forecast = deepcopy(original)
            fit = forecast['guard_crossfits']['left_right']['fits']['median_offset']['0']
            if field == 'training_input_sha256': fit[field] = '0'*64
            elif field == 'residual_order_counts': fit[field]['tied'] -= 1
            else: fit[field] += 1
            fit['model_sha256'] = audit.digest(fit, ('model_sha256',))
            cross = forecast['guard_crossfits']['left_right']
            cross['crossfit_sha256'] = audit.digest(cross, ('crossfit_sha256',))
            with self.subTest(field=field), self.assertRaises(ValueError):
                audit.check_prediction(row, refresh_forecast(forecast))

    def test_recomputed_hash_cannot_hide_branch_or_schema_tampering(self):
        row = row_for(); original = audit.reconstructed_prediction(row)
        for change in ('value', 'available', 'reason', 'fit_binding', 'extra', 'missing', 'bool_schema'):
            forecast = deepcopy(original); branch = forecast['predictions']['median8']
            if change == 'value': branch['on']['values'][0] += 1
            elif change == 'available': branch['on']['available'][0] = False
            elif change == 'reason': branch['on']['unavailable_reasons'][0] = 'invented'
            elif change == 'fit_binding': branch['guard_fit_sha256'] = '0'*64
            elif change == 'extra': forecast['extra'] = 1
            elif change == 'missing': del forecast['metadata']
            else: forecast['schema_version'] = True
            with self.subTest(change=change), self.assertRaises(ValueError):
                audit.check_prediction(row, refresh_forecast(forecast))

    def test_geometry_shape_and_order_fail_before_prediction(self):
        for change in ('guard_order', 'core_order', 'history_shape'):
            row = row_for()
            if change == 'guard_order': row['observed']['guard_xy'].reverse()
            elif change == 'core_order': row['observed']['core_xy'].reverse()
            else: row['observed']['core_history_on'].pop()
            with self.subTest(change=change), self.assertRaises(ValueError):
                audit.reconstructed_prediction(row)

    def test_current_core_truth_and_spec_do_not_change_forecast(self):
        row = row_for(); before = audit.reconstructed_prediction(row)
        row['observed']['core_current_on'] = [999.]*625
        row['truth']['source_template'] = [1.]*625
        row['spec']['motion'] = 'not_a_fit_parameter'
        self.assertEqual(before, audit.reconstructed_prediction(row))

    def test_heldout_guard_current_cannot_change_complement_fit(self):
        row = row_for(); before = audit.reconstructed_prediction(row)
        for index, (x, _) in enumerate(audit.GUARD_XY):
            if x < 64: row['observed']['guard_current'][index] += 99
        after = audit.reconstructed_prediction(row)
        self.assertEqual(before['guard_crossfits']['left_right']['fits']['median_offset']['0'],
                         after['guard_crossfits']['left_right']['fits']['median_offset']['0'])
        self.assertNotEqual(before['guard_crossfits']['left_right']['fits']['median_offset']['1'],
                            after['guard_crossfits']['left_right']['fits']['median_offset']['1'])

    def test_noiseless_signed_appearing_projection(self):
        for amplitude in (4, -4):
            result = score(row_for(amplitude=amplitude))
            for name in audit.METHODS:
                a = result['methods'][name]['metrics']['amplitudes']
                self.assertTrue(a['available']); self.assertEqual(a['raw_on_dn'], amplitude)
                self.assertEqual(a['paired_retention'], 1); self.assertEqual(a['raw_retention'], 1)
                self.assertEqual(a['oracle_retention'], 1); self.assertEqual(a['raw_sign'], 'same')

    def test_stationary_absorption_is_zero_not_unknown(self):
        result = score(row_for(motion='stationary'))
        for name in audit.METHODS:
            a = result['methods'][name]['metrics']['amplitudes']
            self.assertTrue(a['available']); self.assertEqual(a['paired_retention'], 0)
            self.assertEqual(a['paired_sign'], 'zero'); self.assertEqual(a['oracle_retention'], 1)

    def test_raw_guard_bias_does_not_erase_paired_increment(self):
        result = score(row_for(condition='all_guard_plus8'))
        for name in audit.METHODS[2:]:
            a = result['methods'][name]['metrics']['amplitudes']
            self.assertEqual(a['paired_retention'], 1); self.assertEqual(a['raw_sign'], 'reversed')
            self.assertAlmostEqual(a['raw_on_dn'], 4-128/9)

    def test_absent_diagnostic_not_perfect_retention_or_sign(self):
        result = score(row_for(motion='absent', amplitude=0, condition='all_guard_plus8'))
        for name in audit.METHODS:
            a = result['methods'][name]['metrics']['amplitudes']
            self.assertFalse(a['source_present']); self.assertTrue(a['available'])
            self.assertEqual(a['paired_increment_dn'], 0)
            for field in ('raw_retention', 'paired_retention', 'oracle_retention', 'raw_sign', 'paired_sign',
                          'raw_amplitude_error_dn', 'paired_amplitude_error_dn'):
                self.assertIsNone(a[field])

    def test_current_footprint_missing_forces_projection_and_oracle_unknown(self):
        result = score(audit.generated_input(audit.specifications()[-1]))
        for name in audit.METHODS:
            r = result['methods'][name]; a = r['metrics']['amplitudes']
            self.assertEqual(len(r['scored_indices']), 624)
            self.assertFalse(a['available']); self.assertFalse(a['oracle_available'])
            self.assertIsNone(a['raw_on_dn']); self.assertIsNone(a['oracle_on_dn'])

    def test_outer_current_missing_leaves_template_projection_known(self):
        row = row_for(); row['observed']['core_current_on'][0] = None
        result = score(row)
        for name in audit.METHODS:
            self.assertFalse(result['methods'][name]['complete'])
            self.assertTrue(result['methods'][name]['metrics']['amplitudes']['available'])
            self.assertEqual(result['methods'][name]['metrics']['point_errors']['off_residual']['count'], 624)

    def test_median3_extra_support_not_borrowed_by_matched_controls(self):
        result = score(audit.generated_input(audit.specifications()[-2]))
        self.assertEqual(len(result['methods']['median3']['scored_indices']), 625)
        self.assertTrue(result['methods']['median3']['metrics']['amplitudes']['available'])
        for name in audit.METHODS[2:]:
            matched = result['matched'][name]
            self.assertEqual(len(matched['scored_indices']), 624)
            for method in matched['methods'].values():
                self.assertFalse(method['amplitudes']['available'])
                self.assertTrue(method['amplitudes']['oracle_available'])
                self.assertEqual(method['amplitudes']['oracle_on_dn'], 4)

    def test_unavailable_guard_fit_preserves_oracle_and_unknown_counts(self):
        row = audit.generated_input(audit.specifications()[-3]); result = score(row)
        for name in audit.METHODS[2:]:
            r = result['methods'][name]; a = r['metrics']['amplitudes']
            self.assertEqual(r['scored_indices'], []); self.assertFalse(a['available'])
            self.assertTrue(a['oracle_available']); self.assertEqual(a['oracle_retention'], 1)
            aggregate = audit.aggregate([result])['methods'][name]
            self.assertEqual(aggregate['present_amplitude_unknown_cases'], 1)
            self.assertEqual(aggregate['present_oracle_available_cases'], 1)
            self.assertEqual(aggregate['amplitudes']['raw_retention']['count'], 0)

    def test_independent_scores_and_summaries_match_runner(self):
        specs = [specification(motion='absent', amplitude=0),
                 specification(condition='local_guard_plus16', motion='slow_linear', background='textured', noise_level=.5)]
        specs += audit.specifications()[-4:]
        independent, producer = [], []
        for spec in specs:
            row = runner.prepare_input(spec); forecast = runner.predict_input(row)['forecast']
            expected = audit.expected_score(row, forecast); actual = runner.evaluate(row, forecast)
            audit.compare(expected, actual)
            independent.append(expected); producer.append(actual)
        summary = runner.summarize(producer, specs)
        audit.compare(audit.expected_summary(independent, summary['created_at_utc']), summary)
        self.assertEqual(summary['strata']['factorial']['case_count'], 2)
        self.assertEqual(summary['strata']['availability']['case_count'], 4)
        self.assertEqual(sum(x['case_count'] for x in summary['groups']['condition'].values()), 2)

    def test_score_denominator_and_summary_tampering_detected(self):
        result = score(row_for()); bad = deepcopy(result)
        bad['methods']['median8']['scored_indices'].pop()
        with self.assertRaises(ValueError): audit.compare(result, bad)
        summary = audit.expected_summary([result], '2026-09-26T00:00:00+00:00'); bad = deepcopy(summary)
        bad['strata']['factorial']['methods']['median8']['present_cases'] = 2
        with self.assertRaises(ValueError): audit.compare(summary, bad)

    def test_integer_indices_counts_and_schema_do_not_accept_boolean_or_float(self):
        result = score(row_for())
        for field, value in (('index', False), ('index', 0.), ('count', 625.), ('count', True)):
            bad = deepcopy(result)
            if field == 'index': bad['methods']['median8']['scored_indices'][0] = value
            else: bad['total_core_points'] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                audit.compare(result, bad)
        row = row_for(); row['observed']['guard_xy'][0][0] = 8.
        with self.assertRaises(ValueError): audit.check_generated_input(specification(), row)

    def test_empty_metrics_null_not_zero_and_quantile_scalar_reference(self):
        self.assertEqual(audit.point_metrics([]), dict(count=0, mae_dn=None, rmse_dn=None, max_abs_dn=None))
        self.assertEqual(audit.distribution([])['count'], 0)
        self.assertIsNone(audit.distribution([])['mean'])
        audit.compare(audit.distribution([-4., 0., 2., 3., 17.]), runner.distribution([-4., 0., 2., 3., 17.]))
        for bad in ([float('nan')], [float('inf')], [1e308]):
            with self.subTest(bad=bad), self.assertRaises((ValueError, OverflowError)):
                audit.point_metrics(bad)

    def test_sign_category_polarity_and_tolerance(self):
        for amplitude in (-4, 4):
            self.assertEqual(audit.sign_category(amplitude, amplitude), 'same')
            self.assertEqual(audit.sign_category(-amplitude, amplitude), 'reversed')
            self.assertEqual(audit.sign_category(1e-11, amplitude), 'zero')

    def test_checked_read_verifies_hash_before_json_decode(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve()/'bad.json'; path.write_text('not JSON')
            with self.assertRaisesRegex(ValueError, 'File hash differs'):
                audit.checked_read(path, '0'*64, {})
            with self.assertRaises(json.JSONDecodeError):
                audit.checked_read(path, audit.file_sha(path), {})

    def test_canonical_hash_rejects_symlink_and_nonabsolute_path(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve(); path = base/'file'; path.write_text('fixture')
            link = base/'link'; link.symlink_to(path)
            self.assertEqual(audit.file_sha(path), hashlib.sha256(b'fixture').hexdigest())
            for invalid in (link, Path('relative')):
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    audit.file_sha(invalid)

    def test_streaming_jsonl_reader_not_eager(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve()/'rows.jsonl'; path.write_text('{"x":1}\nnot json\n')
            rows = audit.read_lines(path); self.assertEqual(next(rows), {'x': 1})
            with self.assertRaises(json.JSONDecodeError): next(rows)

    def test_run_rejects_outside_path_and_unlisted_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve(); child = base/'generated_fixture'; child.mkdir()
            with self.assertRaisesRegex(ValueError, 'immediate'):
                audit.audit_run(child)
            with patch.object(audit, 'OUTPUT', base):
                with self.assertRaisesRegex(ValueError, 'Unexpected or missing'):
                    audit.audit_run(child)

    def test_receipt_extra_path_rejected_without_following_it(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve(); child = base/'generated_fixture'; child.mkdir()
            for name in audit.RUN_NAMES: (child/name).write_text('{}')
            receipt = dict(completed=True, source_decisions_changed=False, production_changed=False,
                           real_data_accessed=False, files_sha256={'/never/access/this/path': '0'*64})
            (child/'completion_receipt.json').write_text(json.dumps(receipt))
            with patch.object(audit, 'OUTPUT', base):
                with self.assertRaisesRegex(ValueError, 'allowlist'):
                    audit.audit_run(child)

    def test_inherited_hash_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            for name in audit.INHERITED_SHA256:
                path = base/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text('not inherited source')
            with self.assertRaisesRegex(ValueError, 'Inherited V53 source changed'):
                audit.verify_inherited_sources(base)

    def test_prediction_and_scoring_leave_input_unchanged(self):
        row = row_for(background='textured', noise_level=.5); before = audit.digest(row)
        forecast = audit.reconstructed_prediction(row); frozen = audit.digest(forecast)
        audit.expected_score(row, forecast)
        self.assertEqual(before, audit.digest(row)); self.assertEqual(frozen, audit.digest(forecast))


if __name__ == '__main__':
    unittest.main()
