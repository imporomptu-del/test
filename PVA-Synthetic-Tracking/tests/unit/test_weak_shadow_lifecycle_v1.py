"""Generated integration tests for runner hooks; no media/native libraries.

The fake surfaces preserve the runner's call/ownership contracts, not detector
physics. Actual weak selection and covariance arithmetic have separate tests.
This file also runs from the flat, transferred experiment bundle.
"""
from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SCRIPTS = ROOT / 'scripts' if (ROOT / 'scripts').is_dir() else HERE
SPEC = importlib.util.spec_from_file_location(
    'weak_shadow_lifecycle_runner', SCRIPTS / 'run_weak_continuation_shadow_v1.py')
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class GeneratedSurfaces:
    def __init__(self, output):
        self.calls = []
        self.native_version = 0
        self.capture_mutates_native = False
        self.shadow_mutates_array = False
        self.shadow_mutates_baseline = False
        self.oversized_rectangle = False
        self.capture_calls = []
        self.supplied = []
        self.original_results = []
        self.front_module = types.ModuleType('visible_front_v26')
        harness = self

        class Motion:
            def update(instance, gray, index, timestamp):
                harness.calls.append(('motion', index, timestamp, gray))
                result = (harness.image, harness.valid, np.eye(3), 0, {'accepted': True})
                harness.motion_result = result
                return result

        class Front:
            def __init__(instance):
                instance.shape = (128, 128)
                instance.config = types.SimpleNamespace(tile_size=64, warmup_frames=0)
                instance.count = 0
                instance.busy = False
                instance.finish_calls = 0
                instance.peaks = object()

            def update(instance, image, valid, segment, learning_centers=()):
                harness.calls.append(('front', image, valid, segment, learning_centers))
                if instance.busy:
                    raise AssertionError('Fake front reentered')
                instance.busy = True
                instance.count += 1
                try:
                    # This is the same boundary intercepted by the native runner:
                    # prepare has finished, decode precedes learning/finish.
                    cells = harness.front_module.decode_peak_cells(instance.peaks)
                    instance.finish_calls += 1
                    harness.calls.append(('finish', instance.count - 1))
                    return cells, {'native_finished': True}
                finally:
                    instance.busy = False

        class Detector:
            def update(instance, image, valid, segment, learning_centers=()):
                harness.calls.append(('detector', image, valid, segment, learning_centers))
                result = harness.front.update(image, valid, segment, learning_centers)
                harness.detector_result = result
                return result

        class Tracks:
            def __init__(instance, name):
                instance.name = name
                instance.state = {'updates': 0}

            def learning_centers(instance, timestamp, segment):
                harness.calls.append(('learning', instance.name, timestamp, segment))
                return harness.centers

            def update(instance, proposals, index, timestamp, segment, matrix, shape):
                harness.calls.append(('tracking', instance.name, index, proposals,
                                      timestamp, segment, matrix, shape))
                instance.state['updates'] += 1
                result = ([{'track_id': instance.name, 'frame': index}], {'frame': index})
                if instance.name == 'baseline':
                    harness.original_results.append(result)
                return result

        self.visible = types.ModuleType('tiny_target.visible_baseline')
        self.visible.PvaMotion = Motion
        self.visible.VisiblePointDetector = Detector
        self.visible.VisibleTracks = Tracks
        self.package = types.ModuleType('tiny_target')
        self.package.__path__ = []
        self.package.visible_baseline = self.visible
        self.front_module.ResidentFrontV26 = Front

        def decode(peaks):
            if peaks is not self.front.peaks:
                raise AssertionError('Original decoder received different peaks')
            self.calls.append(('decode', self.front.count - 1))
            return self.proposals

        self.front_module.decode_peak_cells = decode
        self.state_module = types.ModuleType('combined_v29_state')
        self.state_module.state_of = lambda tracker: deepcopy(tracker.state)
        self.digest_module = types.ModuleType('replay_tracking_v27')
        self.digest_module.digest = _digest
        self.capture_module = types.ModuleType('accuracy_v56_capture')
        self.capture_module.tile_rectangle = self.rectangle
        self.capture_module.capture_prepared = self.capture
        self.modules = {
            'tiny_target': self.package,
            'tiny_target.visible_baseline': self.visible,
            'visible_front_v26': self.front_module,
            'combined_v29_state': self.state_module,
            'replay_tracking_v27': self.digest_module,
            'accuracy_v56_capture': self.capture_module,
        }
        self.motion, self.front, self.detector = Motion(), Front(), Detector()
        self.baseline = Tracks('baseline')
        self.image = types.SimpleNamespace(owner=types.SimpleNamespace(handle=object()))
        self.valid = np.ones((128, 128), dtype=np.bool_)
        self.centers = [{'support_reference_xy': [[60, 60]]}]
        self.proposals = [{'x': 61.0, 'y': 62.0, 'polarity': 'bright',
                           'score': 5.0, 'response_dn': 2.5}]
        self.bridge = object()
        self.experiment = RUNNER.Experiment(
            'clean', '0126', {'frames': 2, 'weak_windows_inclusive': [[0, 0]]},
            output, types.SimpleNamespace(native_state=self.native_state),
            self.bridge, None, 10.0)

        class Shadow:
            def __init__(instance):
                instance.tracker = Tracks('shadow')
                instance.context = None

            def prepare(instance, index, timestamp, segment):
                harness.calls.append(('prepare_shadow', index))
                instance.context = (index, timestamp, segment)
                return deepcopy(harness.forecasts())

            def step(instance, proposals, matrix, shape, provider):
                index, timestamp, segment = instance.context
                # Deliberately go through the patched method: its shadow-owner
                # dispatch must invoke the original once, not recurse.
                result = instance.tracker.update(
                    proposals, index, timestamp, segment, matrix, shape)
                for forecast in harness.forecasts():
                    supplied = provider(deepcopy(forecast))
                    harness.supplied.append((index, forecast['identity'], supplied))
                    if supplied is not None and harness.shadow_mutates_array:
                        supplied['values'][0, 0, 0] = 99
                if harness.shadow_mutates_baseline:
                    harness.baseline.state['updates'] += 100
                return result

        self.experiment.shadow = Shadow()

    def forecasts(self):
        return [
            {'identity': '0/bright:1', 'query_eligible': True, 'reference_xy': [63.0, 63.0]},
            {'identity': '0/bright:2', 'query_eligible': True, 'reference_xy': [65.0, 65.0]},
            {'identity': '0/bright:3', 'query_eligible': True, 'reference_xy': [-1.0, 63.0]},
            {'identity': '0/bright:4', 'query_eligible': False, 'reference_xy': [64.0, 64.0]},
        ]

    def rectangle(self, shape, tile_size, center, radius):
        if shape != (128, 128) or tile_size != 64 or radius != 45:
            raise AssertionError('Capture geometry arguments changed')
        extent = 3000 if self.oversized_rectangle else 128
        return {'shape_hw': list(shape), 'tile_size': tile_size, 'probe_xy': list(center),
                'radius_px': radius, 'capture_bounds_exclusive_xyxy': [0, 0, extent, extent],
                'tile_bounds_exclusive_xyxy': [0, 0, extent, extent]}

    def native_state(self, instance):
        if instance is not self.front or not instance.busy:
            raise AssertionError('Native guard outside prepared/unfinished owner')
        self.calls.append(('native_guard', self.native_version))
        return {'native_version': self.native_version}

    def capture(self, instance, bridge, *, rectangle, ready, warp_handle):
        if (instance is not self.front or bridge is not self.bridge or not instance.busy
                or warp_handle is not self.image.owner.handle or ready is not True
                or instance.finish_calls != instance.count - 1):
            raise AssertionError('Capture outside exact pre-finish lifecycle')
        self.capture_calls.append(deepcopy(rectangle))
        self.calls.append(('capture', instance.count - 1))
        if self.capture_mutates_native:
            self.native_version += 1
        return {'metadata': {'rectangle': deepcopy(rectangle)},
                'values': np.zeros((128, 128, 20), dtype=np.float32),
                'flags': np.ones((128, 128, 13), dtype=np.uint8),
                'precise_sigmas': np.ones(1, dtype=np.float64)}

    def install(self, stack):
        stack.enter_context(patch.dict(sys.modules, self.modules))
        stack.enter_context(redirect_stdout(io.StringIO()))
        self.experiment.install(stack)

    def frame(self, index):
        timestamp = index * 100000000
        returned = self.motion.update('generated-gray', index, timestamp)
        if returned is not self.motion_result:
            raise AssertionError('Original motion return replaced')
        image, valid, matrix, segment = returned[:4]
        centers = self.baseline.learning_centers(timestamp, segment)
        result = self.detector.update(image, valid, segment, centers)
        if result is not self.detector_result:
            raise AssertionError('Original detector return replaced')
        proposals = result[0]
        returned_tracks = self.baseline.update(
            proposals, index, timestamp, segment, matrix, (128, 128))
        if returned_tracks is not self.original_results[-1]:
            raise AssertionError('Original baseline tracker return replaced')
        return returned_tracks


class LifecycleTests(unittest.TestCase):
    def test_two_frames_dedup_prelearning_missing_and_nonrecursive_dispatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'run'
            h = GeneratedSurfaces(output)
            originals = (h.visible.PvaMotion.update, h.visible.VisibleTracks.update,
                         h.front_module.decode_peak_cells)
            with ExitStack() as stack:
                h.install(stack)
                h.frame(0)
                self.assertEqual(h.experiment.pending, {})
                self.assertIsNone(h.experiment.active)
                self.assertIsNone(h.experiment.learning)
                h.frame(1)
                self.assertEqual(h.experiment.pending, {})
                self.assertIsNone(h.experiment.learning)
            self.assertEqual(originals, (h.visible.PvaMotion.update, h.visible.VisibleTracks.update,
                                         h.front_module.decode_peak_cells))
            self.assertEqual(h.front.finish_calls, 2)
            self.assertEqual(h.baseline.state['updates'], 2)
            self.assertEqual(h.experiment.shadow.tracker.state['updates'], 2)
            self.assertEqual(len(h.experiment.rows), 2)
            self.assertEqual([row['frame'] for row in h.experiment.rows], [0, 1])
            self.assertEqual(len(h.capture_calls), 1)
            self.assertEqual(h.experiment.capture_count, 1)
            first = [item for frame, _, item in h.supplied if frame == 0]
            self.assertIs(first[0]['values'], first[1]['values'])
            self.assertIsNone(first[2])
            self.assertIsNone(first[3])
            self.assertTrue(all(item is None for frame, _, item in h.supplied if frame == 1))
            for name in ('values', 'flags'):
                self.assertFalse(first[0][name].flags.writeable)
            self.assertEqual(first[0]['metadata']['frame'], 0)
            self.assertTrue(first[0]['metadata']['prelearning'])
            rows = [json.loads(line) for line in (output / 'shadow_trace.jsonl').read_text().splitlines()]
            self.assertEqual([row['capture_scheduled'] for row in rows], [True, False])
            self.assertEqual([len(row['capture_files']) for row in rows], [2, 0])
            self.assertEqual(rows[0]['capture_files'][0]['path'], rows[0]['capture_files'][1]['path'])
            self.assertEqual(rows[0]['native_state_before'], rows[0]['native_state_after'])
            self.assertIsNone(rows[1]['native_state_before'])
            self.assertEqual(len(list((output / 'captures').glob('*.npz'))), 1)
            # Original dispatch arguments are untouched; shadow proposals are detached.
            baseline_calls = [c for c in h.calls if c[:2] == ('tracking', 'baseline')]
            shadow_calls = [c for c in h.calls if c[:2] == ('tracking', 'shadow')]
            self.assertEqual(len(baseline_calls), 2)
            self.assertEqual(len(shadow_calls), 2)
            self.assertIs(baseline_calls[0][3], h.proposals)
            self.assertIsNot(shadow_calls[0][3], h.proposals)
            self.assertEqual(shadow_calls[0][3], h.proposals)
            detector_call = next(c for c in h.calls if c[0] == 'detector')
            front_call = next(c for c in h.calls if c[0] == 'front')
            self.assertIs(detector_call[1], h.image)
            self.assertIs(front_call[2], h.valid)
            self.assertIs(front_call[4], h.centers)
            names = [c[0] for c in h.calls]
            self.assertLess(names.index('capture'), names.index('finish'))
            self.assertLess(names.index('finish'), names.index('tracking'))

    def test_native_capture_mutation_aborts_before_finish_and_restores_hooks(self):
        with tempfile.TemporaryDirectory() as temporary:
            h = GeneratedSurfaces(Path(temporary) / 'run')
            h.capture_mutates_native = True
            original = h.visible.PvaMotion.update
            with self.assertRaisesRegex(ValueError, 'Native capture changed baseline state'):
                with ExitStack() as stack:
                    h.install(stack)
                    h.frame(0)
            self.assertIs(h.visible.PvaMotion.update, original)
            self.assertEqual(h.front.finish_calls, 0)
            self.assertIsNone(h.experiment.active)
            self.assertEqual(h.experiment.rows, [])

    def test_readonly_snapshot_mutation_is_not_swallowed(self):
        with tempfile.TemporaryDirectory() as temporary:
            h = GeneratedSurfaces(Path(temporary) / 'run')
            h.shadow_mutates_array = True
            with self.assertRaisesRegex(ValueError, 'read-only'):
                with ExitStack() as stack:
                    h.install(stack)
                    h.frame(0)
            self.assertEqual(h.front.finish_calls, 1)
            self.assertEqual(h.baseline.state['updates'], 1)
            self.assertEqual(h.experiment.shadow.tracker.state['updates'], 1)
            self.assertEqual(len(h.experiment.rows), 1)

    def test_shadow_cannot_mutate_baseline_private_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            h = GeneratedSurfaces(Path(temporary) / 'run')
            h.shadow_mutates_baseline = True
            with self.assertRaisesRegex(ValueError, 'Shadow changed baseline state or proposals'):
                with ExitStack() as stack:
                    h.install(stack)
                    h.frame(0)

    def test_shadow_learning_is_forbidden(self):
        with tempfile.TemporaryDirectory() as temporary:
            h = GeneratedSurfaces(Path(temporary) / 'run')
            with ExitStack() as stack:
                h.install(stack)
                with self.assertRaisesRegex(ValueError, 'never supply detector learning'):
                    h.experiment.shadow.tracker.learning_centers(100000000, 0)

    def test_pixel_budget_aborts_before_capture_without_partial_track_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            h = GeneratedSurfaces(Path(temporary) / 'run')
            h.oversized_rectangle = True
            with self.assertRaisesRegex(ValueError, 'Capture resource bound exceeded'):
                with ExitStack() as stack:
                    h.install(stack)
                    h.frame(0)
            self.assertEqual(h.capture_calls, [])
            self.assertEqual(h.front.finish_calls, 0)
            self.assertIsNone(h.experiment.active)


if __name__ == '__main__':
    unittest.main()
