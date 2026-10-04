from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts'))
import validate_raw16_motion_v5 as validation


class RawMotionV5ValidationTests(unittest.TestCase):
    def fixture(self):
        rows = []
        for name in sorted(validation.previous.POSITIVE | validation.previous.NEGATIVE):
            positive = name in validation.previous.POSITIVE
            rows.append(dict(name=name, passed=True, full_grid_supported=True,
                result=dict(passed=True, accepted=positive, runtime_failure=False,
                    translation_error_px=.1 if positive else None,
                    correspondences={'metrics': {'grid_coverage': {'occupied_cells': 48}}})))
        clean = dict(passed=True, accepted=True, runtime_failure=False, translation_error_px=.1,
                     correspondences={'correspondences': [{'previous_xy': [1., 2.], 'current_xy': [1.75, 1.5]}]})
        rows.append(dict(name='reseed_after_lost_tracks', passed=True, identical_points=True,
            first=deepcopy(clean), recovered=deepcopy(clean),
            unrelated=dict(passed=True, accepted=False, runtime_failure=False)))
        return dict(seed=75316, passed=True, config_sha256=validation.CANDIDATE_SHA256,
            config=validation.load_config(validation.CANDIDATE).raw, max_translation_error_px=.35, cases=rows)

    def test_frozen_config_and_complete_report(self):
        validation.verify_candidate_config()
        validation.verify_report(self.fixture(), 75316)

    def test_reseeding_must_be_measured_not_just_labeled_passed(self):
        r = self.fixture()
        r['cases'][-1]['recovered']['correspondences']['correspondences'] = []
        with self.assertRaises(ValueError):
            validation.verify_report(r, 75316)

    def test_no_relaxed_error_or_false_noise_acceptance(self):
        for key in ('recovered', 'unrelated'):
            r = self.fixture()
            if key == 'recovered':
                r['cases'][-1][key]['translation_error_px'] = .36
            else:
                r['cases'][-1][key]['accepted'] = True
            with self.assertRaises(ValueError):
                validation.verify_report(r, 75316)

    def test_missing_case_or_changed_seed_fails(self):
        r = self.fixture()
        r['cases'].pop()
        with self.assertRaises(ValueError):
            validation.verify_report(r, 75316)
        with self.assertRaises(ValueError):
            validation.verify_report(self.fixture(), 85723)


if __name__ == '__main__':
    unittest.main()
