"""Lossless, ordered visible-video decode with at most one frame ahead.

Only read/cvtColor overlap the consumer. No motion, CUDA or tracker state is
shared with the producer. The capture is opened/queried by the caller, then
exclusively transferred to the producer at start (including release).
"""
from dataclasses import dataclass
import math
import threading
import time

import cv2


def decode_contract(execution):
    if execution not in {"sequential", "prefetch_one"}:
        raise ValueError("Unknown frame_decode_execution")
    return dict(schema="seaqr.visible-decode.v1", execution=execution,
        producer_threads=int(execution == "prefetch_one"),
        maximum_frames_ahead=int(execution == "prefetch_one"),
        stages=["VideoCapture.read", "BGR2GRAY"], drop_frames=False,
        preserve_order=True, timestamp_basis="consumer frame index / container fps",
        processing_state_owner="consumer only", capture_owner="single thread at a time",
        grayscale_ownership="new cvtColor output per frame",
        shutdown="cancel, join, then report; blocked native read fails closed")


@dataclass(frozen=True)
class DecodedFrame:
    index: int
    gray: object
    decode_ms: float
    grayscale_ms: float


class VisibleFrameReader:
    """Single-consumer context manager; call start only inside the timed run.

    The single slot is reserved before decoding, not after: queued + in-flight
    frames <= 1, excluding the current consumer frame and codec-internal memory.
    A stuck native decoder cannot safely be interrupted by another thread. Join
    is bounded and raises, never claiming successful cleanup or completion.
    """
    def __init__(self, source, shape, execution="sequential", max_frames=None):
        self.contract = decode_contract(execution)
        if max_frames is not None and (isinstance(max_frames, bool)
                or not isinstance(max_frames, int) or max_frames <= 0):
            raise ValueError("max_frames must be a positive integer")
        self.source, self.shape = str(source), tuple(shape)
        self.execution, self.max_frames = execution, max_frames
        self._condition = threading.Condition()
        self._slot = None
        self._stop = False
        self._done = False
        self._error = None
        self._thread = None
        self._cap = None
        self._started = False
        self._closed = False
        self._released = False
        self._join_timeout = 5.0
        self.decoded = self.consumed = self.read_calls = 0
        self.maximum_observed_frames_ahead = 0

    def __enter__(self):
        if self._cap is not None or self._closed:
            raise RuntimeError("Reader cannot be reopened")
        self._cap = cv2.VideoCapture(self.source)
        try:
            if not self._cap.isOpened():
                raise ValueError("Cannot open source")
            self.fps = float(self._cap.get(cv2.CAP_PROP_FPS))
            self.expected = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if not math.isfinite(self.fps) or self.fps <= 0:
                raise ValueError("Valid container fps required")
        except BaseException:
            self.close()
            raise
        return self

    def start(self):
        if self._closed or self._cap is None:
            raise RuntimeError("Reader is not open")
        if self._started:
            return
        self._started = True
        if self.execution == "prefetch_one":
            self._thread = threading.Thread(target=self._produce,
                name="seaqr-visible-decode", daemon=True)
            try:
                self._thread.start()
            except BaseException:
                self._thread = None
                self.close()
                raise

    def _decode(self):
        if self.max_frames is not None and self.decoded == self.max_frames:
            return None
        started = time.perf_counter()
        self.read_calls += 1
        ok, bgr = self._cap.read()
        decode_ms = 1000 * (time.perf_counter() - started)
        if not ok:
            return None
        if bgr.shape[:2] != self.shape:
            raise ValueError("Source dimensions changed")
        started = time.perf_counter()
        # No dst argument: independent storage even if the decoder reuses BGR.
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        grayscale_ms = 1000 * (time.perf_counter() - started)
        frame = DecodedFrame(self.decoded, gray, decode_ms, grayscale_ms)
        self.decoded += 1
        return frame

    def _release(self):
        if self._cap is not None and not self._released:
            self._cap.release()
            self._released = True

    def _produce(self):
        try:
            while True:
                with self._condition:
                    self._condition.wait_for(lambda: self._stop or self._slot is None)
                    if self._stop:
                        return
                frame = self._decode()
                if frame is None:
                    return
                with self._condition:
                    if self._stop:
                        return
                    self._slot = frame
                    self.maximum_observed_frames_ahead = max(
                        self.maximum_observed_frames_ahead, self.decoded - self.consumed)
                    self._condition.notify_all()
                # Do not retain an old gray buffer across the next decode.
                frame = None
        except BaseException as exc:
            self._error = exc
        finally:
            try:
                self._release()
            except BaseException as exc:
                self._error = exc
            with self._condition:
                self._done = True
                self._condition.notify_all()

    def read(self):
        self.start()
        if self.execution == "sequential":
            if self._done:
                return None, 0.0
            frame = self._decode()
            self._done = frame is None
            if frame is not None:
                self.consumed += 1
            return frame, 0.0
        started = time.perf_counter()
        with self._condition:
            self._condition.wait_for(lambda: self._slot is not None or self._done)
            if self._slot is not None:
                frame, self._slot = self._slot, None
                self.consumed += 1
                self._condition.notify_all()
                return frame, 1000 * (time.perf_counter() - started)
            if self._error is not None:
                raise self._error
            return None, 1000 * (time.perf_counter() - started)

    def close(self):
        if self._closed:
            return
        with self._condition:
            self._stop = True
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=self._join_timeout)
            if self._thread.is_alive():
                raise RuntimeError("Decode worker did not stop; native read still active")
        else:
            self._release()
        self._closed = True
        self._slot = None
        if self._error is not None:
            raise self._error

    def completed_stats(self):
        if not self._closed or not self._released or self.decoded != self.consumed:
            raise RuntimeError("Decode did not drain and close cleanly")
        return dict(decoded_frames=self.decoded, consumed_frames=self.consumed,
            read_calls=self.read_calls, maximum_observed_frames_ahead=self.maximum_observed_frames_ahead,
            worker_joined=True, capture_released=True, dropped_frames=0)

    def __exit__(self, exc_type, exc, traceback):
        self.close()
