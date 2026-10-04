"""Ownership and ordering contracts for the bounded, opt-in frame preparer.

Synthetic frames only: these tests never open media, create GPU contexts, or
depend on how quickly preparation happens. Events coordinate positive progress;
short bounded waits are used only to assert the absence of forbidden lookahead.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import sys
import threading
import time
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from frame_lookahead_v23 import PreparedFrames


@dataclass(frozen=True)
class Frame:
    index: int
    payload: bytes


class SyntheticSource:
    def __init__(self, indices=range(6)):
        self.frames = [Frame(i, bytes([i % 256]) * 16) for i in indices]
        self.position = 0
        self.calls = []
        self.entered = [threading.Event() for _ in range(len(self.frames) + 1)]
        self.shutdown = threading.Event()

    def read(self):
        position = self.position
        self.position += 1
        self.calls.append(threading.get_ident())
        if position < len(self.entered):
            self.entered[position].set()
        frame = self.frames[position] if position < len(self.frames) else None
        return frame, 0.125

    def close(self):
        self.shutdown.set()


class SyntheticProcessor:
    def __init__(self, count=6, fail_index=None, fail_close=False):
        self.metadata = {"width": 16, "kind": "synthetic"}
        self.owner = threading.get_ident()
        self.threads = [self.owner]
        self.buffers = [bytearray(16), bytearray(16)]
        self.prepared = []
        self.done = [threading.Event() for _ in range(count)]
        self.failed = threading.Event()
        self.close_event = threading.Event()
        self.close_count = 0
        self.fail_index = fail_index
        self.fail_close = fail_close

    def prepare(self, frame, slot):
        self.threads.append(threading.get_ident())
        if frame.index == self.fail_index:
            self.failed.set()
            raise ValueError("synthetic preparation failure")
        self.buffers[slot][:] = frame.payload
        self.prepared.append((frame.index, slot))
        self.done[frame.index].set()
        return memoryview(self.buffers[slot])

    def close(self):
        self.threads.append(threading.get_ident())
        self.close_count += 1
        self.close_event.set()
        if self.fail_close:
            raise ValueError("synthetic cleanup failure")


class FrameLookaheadV23Tests(unittest.TestCase):
    WAIT = 3.0
    ABSENCE_WAIT = 0.05

    def construct(self, source=None, capacity=2, **processor_options):
        source = SyntheticSource() if source is None else source
        processors = []

        def factory():
            processor = SyntheticProcessor(**processor_options)
            processors.append(processor)
            return processor

        prepared = PreparedFrames(
            source.read, factory, capacity=capacity,
            shutdown_source=source.close, timeout=self.WAIT,
        )
        self.assertEqual(len(processors), 1, "construction must await factory")
        return prepared, source, processors[0]

    def assertJoined(self, prepared, processor=None, clean=True):
        self.assertEqual(prepared.stats["closed"], clean)
        self.assertTrue(prepared.stats["joined"])
        owner = prepared.stats["owner_thread_id"]
        self.assertFalse(any(t.ident == owner and t.is_alive()
                             for t in threading.enumerate()))
        if processor is not None:
            self.assertEqual(processor.close_count, 1)
            self.assertTrue(processor.close_event.is_set())

    def test_lazy_start_metadata_copy_and_single_owner_thread(self):
        prepared, source, processor = self.construct()
        try:
            self.assertFalse(source.entered[0].wait(self.ABSENCE_WAIT))
            self.assertIsNot(prepared.metadata, processor.metadata)
            processor.metadata["kind"] = "changed-after-copy"
            self.assertEqual(prepared.metadata["kind"], "synthetic")
            first = prepared.read()  # read implicitly starts the worker
            self.assertEqual(bytes(first.result), first.frame.payload)
            self.assertEqual(first.slot, 0)
        finally:
            prepared.close()
        self.assertNotEqual(processor.owner, threading.get_ident())
        self.assertEqual(set(processor.threads + source.calls), {processor.owner})
        self.assertEqual(prepared.stats["owner_thread_id"], processor.owner)
        self.assertJoined(prepared, processor)

    def test_ordered_exact_results_and_timestamps_in_both_modes(self):
        for capacity in (1, 2):
            with self.subTest(capacity=capacity):
                prepared, source, processor = self.construct(capacity=capacity)
                jobs = []
                try:
                    prepared.start()
                    prepared.start()  # starting twice must not spawn another worker
                    while (job := prepared.read()) is not None:
                        self.assertEqual(bytes(job.result), job.frame.payload)
                        self.assertEqual(job.slot, job.frame.index % 2)
                        self.assertGreater(job.prepare_start_ns, 0)
                        self.assertLessEqual(job.prepare_start_ns, job.prepare_end_ns)
                        self.assertTrue(math.isfinite(job.read_wait_ms))
                        self.assertGreaterEqual(job.read_wait_ms, 0)
                        self.assertEqual(job.read_wait_ms, 0.125)
                        jobs.append(job.frame.index)
                    self.assertEqual(jobs, list(range(6)))
                    self.assertEqual(prepared.stats["prepared"], 6)
                    self.assertEqual(prepared.stats["delivered"], 6)
                    self.assertLessEqual(prepared.stats["max_owned"], capacity)
                finally:
                    prepared.close()
                self.assertEqual(processor.prepared, [(i, i % 2) for i in range(6)])
                self.assertJoined(prepared, processor)

    def test_capacity_two_preserves_current_lease_and_blocks_third_frame(self):
        prepared, source, processor = self.construct(capacity=2)
        try:
            first = prepared.read()
            self.assertTrue(processor.done[1].wait(self.WAIT))
            self.assertFalse(source.entered[2].wait(self.ABSENCE_WAIT),
                             "leased + queued work must consume both slots")
            self.assertEqual(bytes(first.result), first.frame.payload)
            self.assertEqual(prepared.stats["max_owned"], 2)
            second = prepared.read()  # release slot 0; permit preparing frame 2
            self.assertTrue(processor.done[2].wait(self.WAIT))
            self.assertEqual(second.frame.index, 1)
            self.assertEqual(bytes(second.result), second.frame.payload)
            self.assertFalse(source.entered[3].wait(self.ABSENCE_WAIT))
        finally:
            prepared.close()
        self.assertJoined(prepared, processor)

    def test_capacity_one_never_prepares_while_delivered_lease_is_held(self):
        prepared, source, processor = self.construct(capacity=1)
        try:
            first = prepared.read()
            self.assertFalse(source.entered[1].wait(self.ABSENCE_WAIT))
            self.assertEqual(bytes(first.result), first.frame.payload)
            second = prepared.read()
            self.assertEqual(second.frame.index, 1)
            self.assertFalse(source.entered[2].wait(self.ABSENCE_WAIT))
            self.assertEqual(prepared.stats["max_owned"], 1)
        finally:
            prepared.close()
        self.assertJoined(prepared, processor)

    def test_eof_keeps_processor_resources_alive_until_explicit_close(self):
        prepared, source, processor = self.construct(SyntheticSource(range(1)))
        try:
            first = prepared.read()
            self.assertIsNone(prepared.read())
            self.assertIsNone(prepared.read())
            self.assertFalse(processor.close_event.is_set())
            self.assertEqual(bytes(first.result), first.frame.payload)
        finally:
            prepared.close()
        prepared.close()  # successful close is idempotent
        self.assertJoined(prepared, processor)

    def test_preparation_error_is_delivered_after_earlier_valid_result(self):
        prepared, source, processor = self.construct(fail_index=1)
        try:
            prepared.start()
            self.assertTrue(processor.failed.wait(self.WAIT))
            first = prepared.read()
            self.assertEqual(first.frame.index, 0)
            self.assertEqual(bytes(first.result), first.frame.payload)
            with self.assertRaisesRegex(Exception, "synthetic preparation failure"):
                prepared.read()
            self.assertFalse(processor.close_event.is_set())
            self.assertEqual(prepared.stats["delivered"], 1)
        finally:
            with self.assertRaisesRegex(Exception, "synthetic preparation failure"):
                prepared.close()
        self.assertJoined(prepared, processor, clean=False)

    def test_source_error_preserves_earlier_result_and_closes_on_owner(self):
        source = SyntheticSource()
        original_read = source.read
        failed = threading.Event()

        def read():
            if source.position == 1:
                failed.set()
                raise ValueError("synthetic source failure")
            return original_read()

        source.read = read
        prepared, _, processor = self.construct(source)
        try:
            prepared.start()
            self.assertTrue(failed.wait(self.WAIT))
            self.assertEqual(prepared.read().frame.index, 0)
            with self.assertRaisesRegex(Exception, "synthetic source failure"):
                prepared.read()
            self.assertFalse(processor.close_event.is_set())
        finally:
            with self.assertRaisesRegex(Exception, "synthetic source failure"):
                prepared.close()
        self.assertJoined(prepared, processor, clean=False)

    def test_noncontiguous_indices_are_rejected_before_second_prepare(self):
        prepared, source, processor = self.construct(SyntheticSource((0, 2)))
        try:
            self.assertEqual(prepared.read().frame.index, 0)
            with self.assertRaisesRegex(Exception, "(?i)(index|indices|sequence|order)"):
                prepared.read()
            self.assertEqual(processor.prepared, [(0, 0)])
        finally:
            with self.assertRaisesRegex(Exception, "(?i)(index|indices|sequence|order)"):
                prepared.close()
        self.assertJoined(prepared, processor, clean=False)

    def test_close_cancels_blocked_source_and_joins_without_draining_media(self):
        source = SyntheticSource()
        original_read = source.read
        blocked = threading.Event()
        release = threading.Event()

        def read():
            if source.position == 1:
                blocked.set()
                if not release.wait(self.WAIT):
                    raise RuntimeError("test source was not cancelled")
                return None, 0.0
            return original_read()

        def shutdown():
            source.shutdown.set()
            release.set()

        source.read, source.close = read, shutdown
        prepared, _, processor = self.construct(source)
        try:
            self.assertEqual(prepared.read().frame.index, 0)
            self.assertTrue(blocked.wait(self.WAIT))
            prepared.close()
        finally:
            release.set()
            prepared.close()
        self.assertTrue(source.shutdown.is_set())
        self.assertEqual(source.position, 1)
        self.assertJoined(prepared, processor)

    def test_close_before_start_never_reads_source(self):
        prepared, source, processor = self.construct()
        prepared.close()
        prepared.close()
        self.assertEqual(source.position, 0)
        self.assertTrue(source.shutdown.is_set())
        self.assertEqual(prepared.stats["prepared"], 0)
        self.assertEqual(prepared.stats["delivered"], 0)
        self.assertJoined(prepared, processor)

    def test_cleanup_error_is_reported_after_owner_thread_joins(self):
        prepared, source, processor = self.construct(fail_close=True)
        prepared.read()
        with self.assertRaisesRegex(Exception, "synthetic cleanup failure"):
            prepared.close()
        self.assertJoined(prepared, processor, clean=False)

    def test_invalid_capacity_or_timeout_is_rejected_without_source_activity(self):
        for capacity, timeout in ((0, 3), (3, 3), (-1, 3), (1.5, 3), (1, 0), (2, -1)):
            with self.subTest(capacity=capacity, timeout=timeout):
                source = SyntheticSource()
                with self.assertRaises(ValueError):
                    PreparedFrames(source.read, SyntheticProcessor, capacity=capacity,
                                   shutdown_source=source.close, timeout=timeout)
                self.assertEqual(source.position, 0)

    def test_read_and_close_are_restricted_to_the_consumer_owner(self):
        prepared, source, processor = self.construct()
        failures = []

        def other_thread():
            for call in (prepared.start, prepared.read, prepared.close):
                try:
                    call()
                except Exception as exc:
                    failures.append(exc)

        foreign = threading.Thread(target=other_thread)
        try:
            foreign.start()
            foreign.join(self.WAIT)
            self.assertFalse(foreign.is_alive())
            self.assertEqual(len(failures), 3)
            for failure in failures:
                self.assertRegex(str(failure), "(?i)(owner|consumer)")
            self.assertEqual(source.position, 0)
            self.assertEqual(prepared.read().frame.index, 0)
        finally:
            prepared.close()
        self.assertJoined(prepared, processor)

    def test_source_shutdown_failure_does_not_skip_owner_cleanup_or_join(self):
        source = SyntheticSource()

        def shutdown():
            source.shutdown.set()
            raise ValueError("synthetic source shutdown failure")

        source.close = shutdown
        prepared, _, processor = self.construct(source)
        with self.assertRaisesRegex(Exception, "synthetic source shutdown failure"):
            prepared.close()
        self.assertTrue(source.shutdown.is_set())
        self.assertJoined(prepared, processor, clean=False)

    def test_cancellation_during_prepare_discards_only_after_owner_finishes(self):
        source = SyntheticSource()
        started, release = threading.Event(), threading.Event()
        processors = []

        class BlockingProcessor(SyntheticProcessor):
            def prepare(self, frame, slot):
                started.set()
                if not release.wait(FrameLookaheadV23Tests.WAIT):
                    raise RuntimeError("test preparation was not cancelled")
                return super().prepare(frame, slot)

        def factory():
            processor = BlockingProcessor()
            processors.append(processor)
            return processor

        def shutdown():
            source.close()
            release.set()

        prepared = PreparedFrames(source.read, factory, capacity=2,
                                  shutdown_source=shutdown, timeout=self.WAIT)
        try:
            prepared.start()
            self.assertTrue(started.wait(self.WAIT))
            prepared.close()
        finally:
            release.set()
            prepared.close()
        self.assertEqual(prepared.stats["prepared"], 1)
        self.assertEqual(prepared.stats["delivered"], 0)
        self.assertEqual(source.position, 1)
        self.assertJoined(prepared, processors[0])

    def test_factory_failure_propagates_and_leaves_no_owner_thread(self):
        source = SyntheticSource()
        threads = []

        def factory():
            threads.append(threading.current_thread())
            raise ValueError("synthetic factory failure")

        with self.assertRaisesRegex(Exception, "synthetic factory failure"):
            PreparedFrames(source.read, factory, capacity=2,
                           shutdown_source=source.close, timeout=self.WAIT)
        self.assertEqual(len(threads), 1)
        threads[0].join(self.WAIT)
        self.assertFalse(threads[0].is_alive())
        self.assertEqual(source.position, 0)

    def test_metadata_failure_cleans_constructed_processor_on_owner_thread(self):
        source = SyntheticSource()
        processors, threads = [], []

        def factory():
            processor = SyntheticProcessor()
            processor.metadata = None  # dict(None) must fail during initialization
            processors.append(processor)
            threads.append(threading.current_thread())
            return processor

        with self.assertRaises(TypeError):
            PreparedFrames(source.read, factory, capacity=2,
                           shutdown_source=source.close, timeout=self.WAIT)
        self.assertEqual(len(processors), 1)
        self.assertEqual(source.position, 0)
        self.assertEqual(processors[0].close_count, 1)
        self.assertEqual(set(processors[0].threads), {threads[0].ident})
        self.assertFalse(threads[0].is_alive())

    def test_close_timeout_is_bounded_and_remains_a_failed_close(self):
        entered, release = threading.Event(), threading.Event()
        shutdown = threading.Event()
        processors, threads = [], []

        def read():
            threads.append(threading.current_thread())
            entered.set()
            if not release.wait(self.WAIT):
                raise RuntimeError("test failed to release blocked source")
            return None, 0.0

        def factory():
            processor = SyntheticProcessor()
            processors.append(processor)
            return processor

        prepared = PreparedFrames(read, factory, capacity=2,
                                  shutdown_source=shutdown.set, timeout=0.05)
        try:
            prepared.start()
            self.assertTrue(entered.wait(self.WAIT))
            begin = time.monotonic()
            with self.assertRaisesRegex(Exception, "(?i)(did not stop|timeout|timed out)") as first:
                prepared.close()
            self.assertLess(time.monotonic() - begin, 2.0)
            self.assertTrue(shutdown.is_set())
            self.assertFalse(prepared.stats["closed"])
            self.assertFalse(prepared.stats["joined"])
            self.assertFalse(processors[0].close_event.is_set())
        finally:
            release.set()
            if threads:
                threads[0].join(self.WAIT)
        self.assertFalse(threads[0].is_alive())
        self.assertEqual(processors[0].close_count, 1)
        self.assertEqual(set(processors[0].threads), {threads[0].ident})
        with self.assertRaises(Exception) as repeated:
            prepared.close()
        self.assertIs(repeated.exception, first.exception)
        self.assertFalse(prepared.stats["closed"], "timeout cannot become a success record")


if __name__ == "__main__":
    unittest.main()
