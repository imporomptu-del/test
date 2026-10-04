from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import validate_raw16_motion_v4 as validation


class RawMotionV4ValidationTests(unittest.TestCase):
    def fixture(self):
        rows = []
        for name in sorted(validation.POSITIVE | validation.NEGATIVE):
            positive = name in validation.POSITIVE
            rows.append(dict(name=name, passed=True, full_grid_supported=True,
                modes={'v4': dict(passed=True, accepted=positive, runtime_failure=False,
                    translation_error_px=.1 if positive else None,
                    correspondences={'metrics': {'grid_coverage': {'occupied_cells': 48}}})}))
        return dict(seed=75316, passed=True, config_sha256=validation.CANDIDATE_SHA256,
                    config=validation.load_config(validation.CANDIDATE).raw,
                    max_translation_error_px=.35, cases=rows)

    def test_frozen_config_and_complete_report(self):
        validation.verify_candidate_config()
        validation.verify_report(self.fixture(), 75316)

    def test_case_removal_or_duplication_is_not_a_pass(self):
        for duplicate in (False, True):
            r = self.fixture()
            r['cases'].pop()
            if duplicate:
                r['cases'].append(deepcopy(r['cases'][0]))
            with self.assertRaises(ValueError):
                validation.verify_report(r, 75316)

    def test_aggregate_pass_does_not_hide_wrong_motion_or_false_acceptance(self):
        for negative in (False, True):
            r = self.fixture()
            case = next(c for c in r['cases'] if (c['name'] in validation.NEGATIVE) == negative)
            if negative:
                case['modes']['v4']['accepted'] = True
            else:
                case['modes']['v4']['translation_error_px'] = .36
            with self.assertRaises(ValueError):
                validation.verify_report(r, 75316)

    def test_missing_full_grid_and_runtime_failures_are_not_success(self):
        for missing_grid in (False, True):
            r = self.fixture()
            case = next(c for c in r['cases'] if c['name'] == 'dense_native_translation')
            if missing_grid:
                case['modes']['v4']['correspondences']['metrics']['grid_coverage']['occupied_cells'] = 47
            else:
                case['modes']['v4']['runtime_failure'] = True
            with self.assertRaises(ValueError):
                validation.verify_report(r, 75316)


if __name__ == '__main__':
    unittest.main()
