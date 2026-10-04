"""Generated, no-GPU contracts for the opt-in visible execution adapter."""

from contextlib import contextmanager, ExitStack
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import threading
import types
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from visible_overlap_v23 import Binding, install


@contextmanager
def mock_runtime(*, count=5, fps=29.97, motion_failure=None,
                 second_warp_failure=False, execution="cuda_cubic_resident",
                 source_cancel_race=False):
    """Only tiny_target imports are replaced; the real owner worker runs."""
    state = types.SimpleNamespace(
        warps=[], estimators=[], motions=[], sources=[], calls=[], markers=[],
        prepared=[threading.Event() for _ in range(count)],
        source_blocked=threading.Event(), source_stop=threading.Event(),
    )

    class Warp:
        def __new__(cls, *args, **kwargs):
            instance = super().__new__(cls)
            instance.owner = threading.get_ident()
            instance.close_count = 0
            instance.close_threads = []
            state.warps.append(instance)
            return instance

        def __init__(self, library):
            self.lib = types.SimpleNamespace(_name=library)
            self.image = np.zeros((4, 4), dtype=np.uint8)
            self.valid = np.ones((4, 4), dtype=bool)
            if second_warp_failure and len(state.warps) == 2:
                raise RuntimeError("synthetic second warp allocation failure")

        def close(self):
            self.close_count += 1
            self.close_threads.append(threading.get_ident())

    class Estimator:
        def __init__(self):
            self.owner = threading.get_ident()
            self.close_count = 0
            self.close_threads = []
            state.estimators.append(self)

        def close(self):
            self.close_count += 1
            self.close_threads.append(threading.get_ident())

    class Source:
        def __init__(self, *args, **kwargs):
            self.fps = fps
            self.position = 0
            self.close_count = 0
            self.closed = False
            self.started = False
            self.frames = [types.SimpleNamespace(index=i, gray=np.full((4, 4), i, np.uint8))
                           for i in range(count)]
            state.sources.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

        def start(self):
            self.started = True

        def _decode(self):
            if self.position == count:
                return None
            frame = self.frames[self.position]
            self.position += 1
            return frame

        def read(self):
            if source_cancel_race and self.position == 1:
                state.source_blocked.set()
                if not state.source_stop.wait(3):
                    raise RuntimeError("test did not cancel source")
                raise RuntimeError("synthetic source closed during read")
            if self.closed:
                raise RuntimeError("unexpected source read after close")
            return self._decode(), 0.25

        def close(self):
            if not self.closed:
                self.closed = True
                self.close_count += 1
                state.source_stop.set()

    class Motion:
        def __init__(self, *args, **kwargs):
            self.owner = threading.get_ident()
            state.motions.append(self)
            self.estimator = Estimator()
            if motion_failure == "estimator":
                raise RuntimeError("synthetic motion failure after estimator")
            self.cuda_warp = Warp("synthetic-library.so")
            if motion_failure == "warp":
                raise RuntimeError("synthetic motion failure after warp")
            self.execution = execution
            self.conformance = {"validation": {"passed": True}}
            self.matrix = np.eye(3)
            self.meta = {"nested": {"history": []}}

        def update(self, gray, frame_index, timestamp_ns):
            state.calls.append((frame_index, timestamp_ns, threading.get_ident()))
            self.cuda_warp.image[:] = gray + 1
            self.matrix[0, 2] = frame_index
            self.meta["frame"] = frame_index
            self.meta["reset"] = frame_index in (0, 2)
            self.meta["nested"]["history"][:] = [frame_index]
            self.conformance["validation"]["after_prepare"] = frame_index
            state.prepared[frame_index].set()
            return (self.cuda_warp.image, self.cuda_warp.valid, self.matrix,
                    frame_index // 2, self.meta)

        def close(self):
            self.cuda_warp.close()
            self.estimator.close()

    class Detector:
        def update(self, image):
            return int(image.sum())

    class Tracks:
        def update(self, measurement):
            return {"sum": measurement}

    package = types.ModuleType("tiny_target")
    visible = types.ModuleType("tiny_target.visible_baseline")
    decode = types.ModuleType("tiny_target.visible_decode")
    warp_module = types.ModuleType("tiny_target.visible_warp_exact")
    visible.VisibleFrameReader = decode.VisibleFrameReader = Source
    visible.PvaMotion, visible.VisiblePointDetector, visible.VisibleTracks = Motion, Detector, Tracks
    warp_module.CudaCubicTranslation = Warp
    package.visible_baseline, package.visible_decode = visible, decode
    package.visible_warp_exact = warp_module
    state.visible, state.decode = visible, decode
    state.original_reader, state.original_motion = Source, Motion
    modules = {"tiny_target": package, "tiny_target.visible_baseline": visible,
               "tiny_target.visible_decode": decode,
               "tiny_target.visible_warp_exact": warp_module}
    with patch.dict(sys.modules, modules):
        yield state


class VisibleOverlapV23Tests(unittest.TestCase):
    def test_batch_log_is_allowed_but_existing_receipts_and_data_are_rejected(self):
        from run_visible_overlap_v23 import ensure_output_fresh
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'trial'
            output.with_suffix('.log').touch()
            ensure_output_fresh(output)
            receipt = output.with_suffix('.v23.json')
            receipt.touch()
            with self.assertRaises(FileExistsError):
                ensure_output_fresh(output)
            receipt.unlink()
            output.mkdir()
            with self.assertRaises(FileExistsError):
                ensure_output_fresh(output)

    def assert_resources_closed(self, state, *, worker=True):
        self.assertGreaterEqual(len(state.estimators), 1)
        for resource in state.warps + state.estimators:
            self.assertEqual(resource.close_count, 1)
            self.assertEqual(resource.close_threads, [resource.owner])
            if worker:
                self.assertNotEqual(resource.owner, threading.get_ident())
        for motion in state.motions:
            if worker:
                self.assertFalse(any(t.ident == motion.owner and t.is_alive()
                                     for t in threading.enumerate()))

    def test_reference_serial_and_overlap_produce_identical_ordered_outputs(self):
        all_outputs = []
        for mode in ("reference", "serial", "overlap"):
            with self.subTest(mode=mode), mock_runtime() as state, ExitStack() as stack:
                binding = Binding(mode)
                install(stack, binding)
                reader = state.decode.VisibleFrameReader("synthetic")
                self.assertIs(state.visible.VisibleFrameReader, state.decode.VisibleFrameReader)
                self.assertIsInstance(reader, state.visible.VisibleFrameReader)
                self.assertIsInstance(reader.inner, state.original_reader)
                with reader:
                    reader.start()
                    motion = state.visible.PvaMotion()
                    detector, tracks = state.visible.VisiblePointDetector(), state.visible.VisibleTracks()
                    outputs = []
                    try:
                        while True:
                            frame, wait_ms = reader.read()
                            if frame is None:
                                break
                            self.assertEqual(wait_ms, 0.25)
                            image, valid, matrix, segment, metadata = motion.update(
                                frame.gray, frame.index, round(frame.index / reader.fps * 1e9))
                            result = tracks.update(detector.update(image))
                            outputs.append((frame.index, image.tobytes(), valid.tobytes(),
                                            matrix.tobytes(), segment, deepcopy(metadata), result))
                    finally:
                        motion.close()
                all_outputs.append(outputs)
                self.assertEqual([c[:2] for c in state.calls],
                                 [(i, round(i / reader.fps * 1e9)) for i in range(5)])
                snapshot = binding.snapshot()
                self.assertEqual([row["frame"] for row in snapshot["frames"]], list(range(5)))
                for row in snapshot["frames"]:
                    self.assertLessEqual(row["ready_ns"], row["consumer_received_ns"])
                    self.assertLessEqual(row["prepare_start_ns"], row["prepare_end_ns"])
                    self.assertLessEqual(row["detector_start_ns"], row["detector_end_ns"])
                    self.assertLessEqual(row["tracking_start_ns"], row["tracking_end_ns"])
                    self.assertLessEqual(row["tracking_end_ns"], row["consumer_complete_ns"])
                self.assertEqual(state.sources[0].close_count, 1)
                self.assert_resources_closed(state, worker=mode != "reference")
                if mode != "reference":
                    self.assertTrue(snapshot["engine"]["closed"])
                    self.assertTrue(snapshot["engine"]["joined"])
                    self.assertLessEqual(snapshot["engine"]["max_owned"], 1 if mode == "serial" else 2)
        self.assertEqual(len(all_outputs), 3)
        self.assertEqual(all_outputs[0], all_outputs[1])
        self.assertEqual(all_outputs[1], all_outputs[2])

    def test_install_restores_classes_and_wrapped_updates(self):
        with mock_runtime() as state:
            motion_update = state.original_motion.update
            detector_update = state.visible.VisiblePointDetector.update
            tracks_update = state.visible.VisibleTracks.update
            for mode in ("reference", "serial", "overlap"):
                with ExitStack() as stack:
                    install(stack, Binding(mode))
                    self.assertIsNot(state.decode.VisibleFrameReader, state.original_reader)
                self.assertIs(state.decode.VisibleFrameReader, state.original_reader)
                self.assertIs(state.visible.VisibleFrameReader, state.original_reader)
                self.assertIs(state.visible.PvaMotion, state.original_motion)
                self.assertIs(state.original_motion.update, motion_update)
                self.assertIs(state.visible.VisiblePointDetector.update, detector_update)
                self.assertIs(state.visible.VisibleTracks.update, tracks_update)

    def test_optional_markers_enclose_worker_and_consumer_stages_on_correct_threads(self):
        with mock_runtime(count=1) as state, ExitStack() as stack:
            @contextmanager
            def marker(frame, stage):
                state.markers.append((frame, stage, "enter", threading.get_ident()))
                try:
                    yield
                finally:
                    state.markers.append((frame, stage, "exit", threading.get_ident()))

            binding = Binding("overlap", marker=marker)
            install(stack, binding)
            reader = state.decode.VisibleFrameReader()
            motion = state.visible.PvaMotion()
            try:
                frame, _ = reader.read()
                image, *_ = motion.update(frame.gray, frame.index, 0)
                value = state.visible.VisiblePointDetector().update(image)
                self.assertEqual(state.visible.VisibleTracks().update(value), {"sum": 16})
            finally:
                motion.close()
            self.assertEqual([m[:3] for m in state.markers],
                             [(0, stage, position)
                              for stage in ("motion_worker", "detector", "tracking")
                              for position in ("enter", "exit")])
            self.assertEqual({m[3] for m in state.markers[:2]}, {state.motions[0].owner})
            self.assertEqual({m[3] for m in state.markers[2:]}, {threading.get_ident()})
            self.assert_resources_closed(state)

    def test_prepared_metadata_matrix_and_conformance_are_independent_of_next_frame(self):
        with mock_runtime() as state, ExitStack() as stack:
            binding = Binding("overlap")
            install(stack, binding)
            reader = state.decode.VisibleFrameReader()
            motion = state.visible.PvaMotion()
            try:
                first, _ = reader.read()
                self.assertTrue(state.prepared[1].wait(3))
                image, valid, matrix, segment, metadata = motion.update(first.gray, 0, 0)
                self.assertEqual(metadata, {"frame": 0, "reset": True, "nested": {"history": [0]}})
                self.assertEqual(matrix[0, 2], 0)
                self.assertEqual(segment, 0)
                self.assertTrue(np.all(image == 1))
                self.assertEqual(motion.conformance, {"validation": {"passed": True}})
                reader.read()  # release frame 0; frame 2 may now reuse its image slot
                self.assertTrue(state.prepared[2].wait(3))
                self.assertEqual(matrix[0, 2], 0, "small mapping must outlive slot reuse")
                self.assertEqual(metadata["nested"]["history"], [0])
            finally:
                motion.close()
            self.assert_resources_closed(state)

    def test_motion_rejects_wrong_identity_index_timestamp_duplicate_and_stale_frame(self):
        for mode in ("serial", "overlap"):
            with self.subTest(mode=mode), mock_runtime() as state, ExitStack() as stack:
                binding = Binding(mode)
                install(stack, binding)
                reader = state.decode.VisibleFrameReader()
                motion = state.visible.PvaMotion()
                try:
                    with self.assertRaisesRegex(RuntimeError, "Stale, mismatched"):
                        motion.update(np.zeros((4, 4), np.uint8), 0, 0)
                    first, _ = reader.read()
                    for gray, index, timestamp in ((first.gray.copy(), 0, 0),
                                                   (first.gray, 1, 0),
                                                   (first.gray, 0, 1)):
                        with self.assertRaisesRegex(RuntimeError, "Stale, mismatched"):
                            motion.update(gray, index, timestamp)
                    motion.update(first.gray, 0, 0)
                    with self.assertRaisesRegex(RuntimeError, "already consumed"):
                        motion.update(first.gray, 0, 0)
                    second, _ = reader.read()
                    with self.assertRaisesRegex(RuntimeError, "Stale, mismatched"):
                        motion.update(first.gray, 0, 0)
                    timestamp = round(1 / reader.fps * 1e9)
                    motion.update(second.gray, 1, timestamp)
                finally:
                    motion.close()
                self.assert_resources_closed(state)

    def test_partial_motion_initialization_closes_created_resources_on_owner(self):
        for failure in ("estimator", "warp"):
            with self.subTest(failure=failure), mock_runtime(motion_failure=failure) as state, ExitStack() as stack:
                binding = Binding("overlap")
                install(stack, binding)
                reader = state.decode.VisibleFrameReader()
                try:
                    with self.assertRaisesRegex(RuntimeError, "synthetic motion failure"):
                        state.visible.PvaMotion()
                    self.assertIsNone(binding.engine)
                    self.assertEqual(state.sources[0].position, 0)
                    self.assert_resources_closed(state)
                finally:
                    reader.close()

    def test_partial_second_warp_initialization_closes_both_warps_and_estimator(self):
        with mock_runtime(second_warp_failure=True) as state, ExitStack() as stack:
            binding = Binding("overlap")
            install(stack, binding)
            reader = state.decode.VisibleFrameReader()
            try:
                with self.assertRaisesRegex(RuntimeError, "second warp allocation failure"):
                    state.visible.PvaMotion()
                self.assertEqual(len(state.warps), 2)
                self.assert_resources_closed(state)
            finally:
                reader.close()

    def test_unsupported_execution_is_rejected_with_owner_cleanup(self):
        with mock_runtime(execution="cpu") as state, ExitStack() as stack:
            binding = Binding("serial")
            install(stack, binding)
            reader = state.decode.VisibleFrameReader()
            try:
                with self.assertRaisesRegex(ValueError, "Only frozen resident"):
                    state.visible.PvaMotion()
                self.assert_resources_closed(state)
                self.assertEqual(state.sources[0].position, 0)
            finally:
                reader.close()

    def test_invalid_mode_construction_order_and_multiple_sources_are_rejected(self):
        with self.assertRaises(ValueError):
            Binding("unknown")
        with mock_runtime() as state, ExitStack() as stack:
            binding = Binding("overlap")
            install(stack, binding)
            with self.assertRaisesRegex(RuntimeError, "construction order"):
                state.visible.PvaMotion()
            reader = state.decode.VisibleFrameReader()
            try:
                with self.assertRaisesRegex(RuntimeError, "Only one source"):
                    state.decode.VisibleFrameReader()
                motion = state.visible.PvaMotion()
                with self.assertRaisesRegex(RuntimeError, "construction order"):
                    state.visible.PvaMotion()
                motion.close()
            finally:
                reader.close()
            self.assert_resources_closed(state)

    def test_cancellation_racing_source_close_fails_closed_but_joins_and_cleans(self):
        with mock_runtime(source_cancel_race=True) as state, ExitStack() as stack:
            binding = Binding("overlap")
            install(stack, binding)
            reader = state.decode.VisibleFrameReader()
            motion = state.visible.PvaMotion()
            try:
                first, _ = reader.read()
                motion.update(first.gray, 0, 0)
                self.assertTrue(state.source_blocked.wait(3))
                with self.assertRaisesRegex(RuntimeError, "source closed during read"):
                    motion.close()
                self.assertTrue(binding.engine.stats["joined"])
                self.assertFalse(binding.engine.stats["closed"])
                self.assertEqual(binding.engine.stats["delivered"], 1)
                self.assert_resources_closed(state)
                with self.assertRaisesRegex(RuntimeError, "source closed during read"):
                    reader.close()
            finally:
                state.source_stop.set()


if __name__ == "__main__":
    unittest.main()
