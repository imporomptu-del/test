"""Full-native geometry and honest detection-availability regression tests."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from tiny_target import dense_screen as dense
from tiny_target.evaluation import SyntheticInjectionSpec, SyntheticInjector, SyntheticTarget
from tiny_target.motion.types import MotionCorrespondences
from tiny_target.types import Discontinuity, Frame, TimestampSource


def frame(image, index, valid=None):
    return Frame(image=image, timestamp_ns=index * 333_333_333, frame_index=index,
                 source_id='generated-raw16-coverage', bit_depth=16,
                 timestamp_source=TimestampSource.CONTAINER_RATE, valid_mask=valid)


class DenseCoverageTests(unittest.TestCase):
    def test_legacy_crop_is_unchanged_and_full_frame_resolves_geometry(self):
        positional = dense.DenseScreenConfig(0, 0, 96, 80)
        self.assertEqual((positional.crop_width, positional.crop_height), (96, 80))
        old = dense.DenseScreenConfig()
        self.assertIs(dense.resolve_dense_geometry(old, 4784, 3190), old)
        self.assertEqual(old.crop_height, 1920)
        for width, height in ((4784, 3190), (101, 77), (640, 480)):
            result = dense.resolve_dense_geometry(replace(old, coverage_mode='full_frame'), width, height)
            self.assertEqual((result.crop_x, result.crop_y, result.crop_width, result.crop_height),
                             (0, 0, width, height))
        with self.assertRaises(dense.DenseScreenError):
            dense.resolve_dense_geometry(old, 100, 100)
        with self.assertRaises(ValueError):
            dense.DenseScreenConfig(coverage_mode='automatic')
        with self.assertRaises(ValueError):
            dense.DenseScreenConfig(coverage_mode='full_frame', crop_y=10)

    def test_both_source_types_resolve_full_frame_before_bounds_checks(self):
        cfg = dense.DenseScreenConfig(coverage_mode='full_frame')
        probe = SimpleNamespace(width=101, height=77, pixel_format='gray16le', interval_ns=100_000_000)
        with patch.object(dense, 'probe_video', return_value=probe):
            cropped = dense.CroppedVideoSource('generated', cfg, timestamp_csv=None)
        with patch.object(dense, 'FfmpegVideoSource', return_value=SimpleNamespace(probe=probe)):
            stabilized = dense.StabilizedCropSource('generated', cfg,
                Path(__file__).resolve().parents[2] / 'configs/evaluation/phase20_motion_v8.json',
                timestamp_csv=None, bit_depth=16, max_frames=8, injector=None)
        raw = np.arange(77 * 101, dtype=np.uint16).reshape(77, 101)
        valid = raw % 3 != 0
        source_frame = frame(raw, 0, valid)
        result = stabilized._crop(source_frame)
        self.assertEqual(cropped.config, stabilized.config)
        self.assertEqual(result.image.tobytes(), raw.tobytes())
        self.assertEqual(result.valid_mask.tobytes(), valid.tobytes())

    def test_all_frames_accounted_for_including_resets_masks_and_disabled_branch(self):
        cfg = dense.DenseScreenConfig(crop_width=61, crop_height=47,
            background_warmup_frames=2, per_frame_event_screen_enabled=False, opencv_threads=1)
        screener = dense.DensePointScreener(cfg)
        for index in range(9):
            mask = np.ones((47, 61), bool)
            if index == 4:
                mask[:] = False
            image = np.full(mask.shape, 20000, np.uint16)
            screener.process(frame(image, index, mask), segment_index=int(index >= 6))
        result = screener.finalize()['availability']
        self.assertEqual(len(result['frames']), 9)
        self.assertEqual(result['frames_by_filter_state'],
                         {'background_warmup': 4, 'ready': 4, 'no_valid_filter_pixels': 1})
        self.assertEqual(result['reference_reset_count'], 1)
        self.assertEqual(result['frames_emitting_synthetic_windows'], 0)
        for entry in result['frames']:
            self.assertEqual(entry['synthetic_window_state'], 'disabled')
            self.assertEqual(sum(entry['filter_valid_pixels_3x3_row_major']), entry['filter_valid_pixels'])
        self.assertEqual(result['frames'][4]['filter_valid_pixels'], 0)
        self.assertTrue(result['frames'][6]['reference_reset'])

    def test_warmed_but_fully_saturated_is_not_reported_as_available(self):
        screener = dense.DensePointScreener(dense.DenseScreenConfig(
            crop_width=48, crop_height=48, background_warmup_frames=2, opencv_threads=1))
        for index in range(6):
            screener.process(frame(np.full((48, 48), 65535, np.uint16), index))
        result = screener.finalize()
        self.assertEqual(result['frames_screened_after_background_warmup'], 4)  # legacy field
        self.assertEqual(result['availability']['frames_with_valid_filter_support'], 0)
        self.assertEqual(result['availability']['frames_by_filter_state']['no_valid_filter_pixels'], 4)
        self.assertEqual(result['shortlist'], [])

    def test_motion_report_distinguishes_accepted_fit_from_applied_transform(self):
        frames = [frame(np.full((77, 101), 20000, np.uint16), i) for i in range(5)]
        frames[2] = replace(frames[2], discontinuities=(Discontinuity.TIMESTAMP_GAP,))
        class Source:
            probe = SimpleNamespace(width=101, height=77)
            def __iter__(self):
                return iter(frames)
        def correspondences(previous, current):
            if current.frame_index == 3:
                raise dense.PvaMotionError('generated runtime failure')
            points = np.array([(x, y) for y in np.linspace(5, 72, 6)
                               for x in np.linspace(5, 96, 8)], np.float32)
            if current.frame_index == 4:
                points = points[:2]
            return MotionCorrespondences(points, points, np.ones(len(points)), np.zeros(len(points)),
                previous.frame_index, current.frame_index, previous.timestamp_ns, current.timestamp_ns,
                (101, 77), (51, 39), {'usable_for_transform': True}, {'total': 1.0}, {'optical_flow_pyrlk': 'PVA'})
        with patch.object(dense, 'FfmpegVideoSource', return_value=Source()), \
             patch.object(dense, 'PvaPyrLkMotionEstimator') as estimator:
            estimator.return_value.estimate.side_effect = correspondences
            source = dense.StabilizedCropSource('generated', dense.DenseScreenConfig(coverage_mode='full_frame'),
                Path(__file__).resolve().parents[2] / 'configs/evaluation/phase20_motion_v8.json',
                timestamp_csv=None, bit_depth=16, max_frames=5, injector=None)
            observed = list(source)
        self.assertEqual([segment for _, segment in observed], [0, 0, 1, 2, 3])
        self.assertEqual(source.metrics['accepted_global_transforms'], 2)
        self.assertEqual(source.metrics['accepted_transforms_applied'], 1)
        self.assertEqual(source.metrics['reference_resets'], 3)
        self.assertEqual(source.metrics['pva_failures'], 1)
        pairs = source.metrics['motion_pairs']
        self.assertTrue(pairs[2]['fit_accepted'])
        self.assertTrue(pairs[2]['reference_reset'])
        self.assertEqual(pairs[2]['discontinuities'], ['timestamp_gap'])
        self.assertEqual(pairs[3]['rejection_reasons'], ['pva_failure'])
        self.assertEqual(pairs[4]['rejection_reasons'], ['insufficient_correspondences'])

    def test_generated_uint16_target_detected_in_each_image_region(self):
        # These generated controls test spatial coverage, not real-scene recall.
        cfg = dense.resolve_dense_geometry(dense.DenseScreenConfig(
            coverage_mode='full_frame', background_warmup_frames=3,
            tile_size_px=64, cfar_sample_stride_px=2, sigma_floor_dn=1,
            minimum_track_hits=4, threshold_sigma=4, opencv_threads=1), 192, 192)
        for row in range(3):
            for col in range(3):
                with self.subTest(row=row, col=col):
                    target = SyntheticTarget(f'r{row}c{col}', 2400., 0,
                        (col * 64 + 16.25, row * 64 + 24.25), (2., 1.), first_frame_index=5, last_frame_index=29)
                    injector = SyntheticInjector(SyntheticInjectionSpec(75, .8, 3, (target,)))
                    screener = dense.DensePointScreener(cfg, truth_targets=(target,))
                    rng = np.random.default_rng(75)
                    for index in range(30):
                        raw = np.rint(20000 + rng.normal(0, 2, (192, 192))).astype(np.uint16)
                        original = frame(raw, index)
                        before = original.pixel_sha256()
                        injected = injector.inject(original)
                        self.assertEqual(original.pixel_sha256(), before)
                        self.assertEqual(injected.image.dtype, np.uint16)
                        screener.process(injected)
                    result = screener.finalize()
                    evaluation = dense._evaluate_injected_targets(result['shortlist'],
                        SyntheticInjectionSpec(75, .8, 3, (target,)), 3.)
                    self.assertEqual(evaluation['detected_target_ids'], [target.target_id])
                    self.assertTrue(all(v > 0 for v in result['availability']['frames'][-1]['filter_valid_pixels_3x3_row_major']))


if __name__ == '__main__':
    unittest.main()
