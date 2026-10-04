from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest

from tiny_target.dense_screen import load_dense_screen_config

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('validate_raw16_full_frame', ROOT / 'scripts/validate_raw16_full_frame.py')
validation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validation)


def supported_report():
    return {'source': {'crop_xywh': [0, 0, 4784, 3190], 'requested_image_area_fraction': 1.,
                      'pva_stabilization': {'metrics': {'pva_failures': 0}}},
            'screening': {'frames_seen': 64, 'availability': {'frames': [{}] * 64,
                'frames_with_valid_filter_support': 60}, 'synthetic_tracking': {'windows': [
                {'availability': {'valid_ranking_pixels_3x3_row_major': [1] * 9}} for _ in range(6)]}},
            'injection': None}


class Raw16FullFrameValidationTests(unittest.TestCase):
    def test_policy_changes_are_explicit_and_bounded(self):
        old, _ = load_dense_screen_config(ROOT / 'configs/evaluation/phase19_dense_screen_v1.json')
        new, _ = load_dense_screen_config(ROOT / 'configs/evaluation/raw16_full_frame_v2.json')
        diff = {k for k, value in old.to_dict().items() if value != new.to_dict()[k]}
        self.assertEqual(diff, {'coverage_mode', 'crop_height', 'background_execution'})
        self.assertEqual(new.coverage_mode, 'full_frame')
        self.assertEqual(new.background_execution, 'masked_ufunc')

    def test_zero_window_processing_is_not_successful_detection_availability(self):
        report = supported_report()
        good = validation.assess(report)
        self.assertTrue(good['detection_availability_passed'])
        self.assertFalse(good['real_target_accuracy_validated'])
        report['screening']['synthetic_tracking']['windows'] = []
        report['screening']['availability']['frames_with_valid_filter_support'] = 0
        unavailable = validation.assess(report)
        self.assertTrue(unavailable['processing_integrity_passed'])
        self.assertFalse(unavailable['detection_availability_passed'])

    def test_bottom_region_gap_and_missing_control_cannot_pass(self):
        report = supported_report()
        for window in report['screening']['synthetic_tracking']['windows']:
            window['availability']['valid_ranking_pixels_3x3_row_major'][8] = 0
        self.assertFalse(validation.assess(report)['detection_availability_passed'])
        report['injection'] = {'specification': {'targets': [{'target_id': 'a'}, {'target_id': 'b'}]},
                               'synthetic_track_pool_evaluation': {'detected_target_ids': ['a']}}
        self.assertFalse(validation.assess(report)['synthetic_controls_passed'])
        complete = deepcopy(report)
        complete['injection']['synthetic_track_pool_evaluation']['detected_target_ids'] = ['a', 'b']
        self.assertTrue(validation.assess(complete)['synthetic_controls_passed'])
        self.assertFalse(validation.assess(complete)['real_target_accuracy_validated'])


if __name__ == '__main__':
    unittest.main()
