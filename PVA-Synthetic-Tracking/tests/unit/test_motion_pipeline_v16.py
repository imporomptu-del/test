from pathlib import Path
import sys
import unittest
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts'))
from run_motion_pipeline_v16 import validate_scope
from batch_motion_pipeline_v16 import schedule
import summarize_motion_pipeline_v16 as summary


class PipelineScopeTests(unittest.TestCase):
    def test_closed_scope(self):
        for clip, mode, injected in [('0126', 'candidate', False), ('0029', 'candidate', True),
                                     ('0040', 'unknown', False)]:
            with self.assertRaises(ValueError):
                validate_scope(clip, mode, injected)

    def test_frozen_schedule(self):
        rows = schedule()
        self.assertEqual(len(rows), 11)
        self.assertEqual(len({r['name'] for r in rows}), 11)
        self.assertEqual([r['mode'] for r in rows[:8]], ['reference', 'candidate']*2+['candidate', 'reference']*2)
        self.assertEqual(sum(r['injected'] for r in rows), 1)
        self.assertEqual(sum(r['profile'] for r in rows), 2)
        for row in rows:
            validate_scope(row['clip'], row['mode'], row['injected'])

    def test_incomplete_summary_rejected(self):
        with patch.object(summary, 'read', return_value={'passed': False, 'error': None}):
            with self.assertRaisesRegex(ValueError, 'Batch incomplete'):
                summary.summarize(Path('/unused'))

    def compare_fixture(self, changed=None):
        def reader(path):
            value = {'report.json': {'tracks': [1]},
                     'candidate_decisions.json': [{'accepted': True}],
                     'global_fit_identities.json': [{'fit': 1}]}[path.name]
            return [] if path.parent.name == 'candidate' and path.name == changed else value
        with patch.object(summary, 'read', side_effect=reader), \
             patch.object(summary, 'normalized_report', side_effect=lambda r: r), \
             patch.object(summary, 'compare_source_motion', return_value={'source': True, 'motion': True}):
            summary.compare(Path('/reference'), Path('/candidate'))

    def test_comparator_accepts_identical(self):
        self.compare_fixture()

    def test_comparator_rejects_track_candidate_and_fit_mutations(self):
        for name in ('report.json', 'candidate_decisions.json', 'global_fit_identities.json'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.compare_fixture(name)

    def test_comparator_rejects_source_changes(self):
        with patch.object(summary, 'read', return_value={}), \
             patch.object(summary, 'normalized_report', side_effect=lambda r: r), \
             patch.object(summary, 'compare_source_motion', return_value={'source': False, 'motion': True}):
            with self.assertRaisesRegex(ValueError, 'Source/point'):
                summary.compare(Path('/reference'), Path('/candidate'))
