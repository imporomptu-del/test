"""Ensure RAW16 evidence comparisons fail on changed output or provenance."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts'))
from compare_raw16_efficiency import compare
from profile_raw16_efficiency import sha, write_json


class CompareRaw16Tests(unittest.TestCase):
    def fixture(self, root, mode):
        root.mkdir()
        config = dict(background_execution=mode, crop_width=4784)
        write_json(root/'execution_config.json', config)
        report = dict(configuration=dict(identity=dict(sha256=sha(root/'execution_config.json')),
            effective=config), source=dict(pva_stabilization=dict(metrics=dict(
                pva_failures=0, frames=32, accepted_global_transforms=31, rejected_global_transforms=0,
                median_pva_total_ms=1))), injection=None, performance=dict(
                    elapsed_seconds=40., processed_frames_per_second=.8),
            screening=dict(frames_seen=32, frames_screened_after_background_warmup=28,
                synthetic_tracking=dict(window_count=1, candidate_count_before_clip_pool=8)))
        observation = dict(passed=True, mode='timed', clip='0040', requested_frames=32,
            source={'path': 'generated'}, frozen_sha256={}, package_sha256={'fake.py': 'abc'},
            script_sha256='script', python='3.10', numpy='1.26', background_execution=mode,
            execution_config_sha256=sha(root/'execution_config.json'))
        write_json(root/'report.json', report)
        write_json(root/'observation.json', observation)

    def check(self, mutate=None):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp)/'reference', Path(tmp)/'candidate'
            self.fixture(a, 'indexed_reference')
            self.fixture(b, 'masked_ufunc')
            if mutate:
                path = b/mutate[0]
                value = json.loads(path.read_text())
                mutate[1](value)
                # Generated test fixtures, not application data.
                path.write_text(json.dumps(value))
            return compare(a, b)

    def test_matching_output_passes(self):
        result = self.check()
        self.assertTrue(result['passed'])
        self.assertFalse(result['exact_array_audit'])
        self.assertFalse(result['real_object_accuracy_validated'])

    def test_execution_timing_may_change(self):
        result = self.check(('report.json', lambda r: r['performance'].update(
            elapsed_seconds=32, processed_frames_per_second=1)))
        self.assertEqual(result['throughput_ratio'], 1.25)

    def test_changed_counts_are_not_excused_as_timing(self):
        with self.assertRaises(ValueError):
            self.check(('report.json', lambda r: r['screening']['synthetic_tracking'].update(
                candidate_count_before_clip_pool=7)))

    def test_different_package_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'provenance'):
            self.check(('observation.json', lambda o: o['package_sha256'].update({'fake.py': 'changed'})))

    def test_no_windows_cannot_claim_equivalence(self):
        with self.assertRaisesRegex(ValueError, 'No synthetic windows'):
            self.check(('report.json', lambda r: r['screening']['synthetic_tracking'].update(window_count=0)))

    def test_pva_error_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'PVA'):
            self.check(('report.json', lambda r: r['source']['pva_stabilization']['metrics'].update(pva_failures=1)))

    def test_modified_execution_config_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'config hash'):
            self.check(('execution_config.json', lambda c: c.update(crop_width=4000)))


if __name__ == '__main__':
    unittest.main()
