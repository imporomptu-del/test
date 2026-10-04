from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from tiny_target.dense_review import analyze_report, describe_track_outputs, export_review


def track(i):
    return dict(track_id=i, first_reference_frame_index=12, last_reference_frame_index=52, hit_count=5)


def report():
    pool = [track(i) for i in range(20)]
    return {'schema_version': 'seaqr.tiny-target.dense-screen.v1',
        'configuration': {'effective': {'synthetic_retained_track_pool_size': 512, 'max_shortlist_tracks_per_clip': 8}},
        'screening': {'synthetic_tracking': {'track_pool': pool, 'shortlist': pool[:8],
            'qualified_track_pool_count': 20, 'shortlist_count': 8, 'window_count': 6},
            'availability': {'synthetic_windows_with_valid_ranking': 6}}, 'injection': None}


class DenseReviewTests(unittest.TestCase):
    def test_invalid_counts_and_inconsistent_availability_are_rejected(self):
        for bad in (-1, True, 1.5):
            source = report()
            source['screening']['synthetic_tracking']['window_count'] = bad
            with self.assertRaises(ValueError):
                analyze_report(source)
        for valid in (-1, True, 7, 0):
            source = report()
            source['screening']['availability']['synthetic_windows_with_valid_ranking'] = valid
            with self.assertRaises(ValueError):
                analyze_report(source)
        source = report()
        source['screening']['availability'] = {}
        self.assertIsNone(analyze_report(source)['detection_available'])

    def test_malformed_track_does_not_create_partial_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = report()
            del source['screening']['synthetic_tracking']['track_pool'][10]['hit_count']
            path = root / 'report.json'
            path.write_text(json.dumps(source))
            with self.assertRaises(KeyError):
                export_review(path, root / 'review')
            self.assertFalse((root / 'review').exists())

    def test_preview_does_not_change_or_hide_retained_output(self):
        source = report()
        before = deepcopy(source)
        result = analyze_report(source)
        self.assertEqual(source, before)
        self.assertEqual(len(result['retained_tracks']), 20)
        self.assertEqual(result['retained_tracks'][16]['track_id'], 16)
        self.assertEqual(len(result['review_preview']), 8)
        preview = result['output_contract']['review_preview']
        self.assertFalse(preview['is_complete_retained_set'])
        self.assertEqual(preview['omitted_retained_count'], 12)
        self.assertFalse(result['output_contract']['retained_tracks']['unbounded_all_tracks'])

    def test_mismatched_duplicate_and_out_of_pool_preview_fail_closed(self):
        for bad in ([track(21)], [track(1), track(1)], [dict(track(1), hit_count=9)]):
            with self.assertRaises(ValueError):
                describe_track_outputs([track(1)], bad, capacity=8, preview_limit=8)
        with self.assertRaises(ValueError):
            describe_track_outputs([track(1), track(1)], [], capacity=8, preview_limit=8)
        with self.assertRaises(ValueError):
            describe_track_outputs([track(1), track(2)], [], capacity=1, preview_limit=1)

    def test_empty_unavailable_search_is_not_a_successful_negative(self):
        source = report()
        synthetic = source['screening']['synthetic_tracking']
        synthetic.update(track_pool=[], shortlist=[], qualified_track_pool_count=0, shortlist_count=0, window_count=0)
        source['screening']['availability']['synthetic_windows_with_valid_ranking'] = 0
        self.assertEqual(analyze_report(source)['interpretation'], 'detection_unavailable')
        del source['screening']['availability']
        self.assertEqual(analyze_report(source)['interpretation'], 'availability_not_recorded')

    def test_exports_exact_arrays_and_never_overwrites_existing_review(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = report()
            path = root / 'report.json'
            path.write_text(json.dumps(source))
            output = root / 'review'
            manifest = export_review(path, output)
            self.assertEqual(manifest['interpretation'], 'unlabeled_review_workload')
            self.assertEqual(json.loads((output/'retained_tracks.json').read_text())['tracks'],
                             source['screening']['synthetic_tracking']['track_pool'])
            self.assertEqual(json.loads((output/'review_preview.json').read_text())['tracks'],
                             source['screening']['synthetic_tracking']['shortlist'])
            self.assertIn('| 17 | 16 |', (output/'results.md').read_text())
            with self.assertRaises(FileExistsError):
                export_review(path, output)
