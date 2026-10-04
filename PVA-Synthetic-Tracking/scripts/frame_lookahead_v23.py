"""Two explicitly leased preparation slots; no algorithm or source dependencies."""
from collections import deque
from dataclasses import dataclass
import threading
import time


@dataclass(frozen=True)
class Job:
    frame: object
    result: object
    slot: int
    read_wait_ms: float
    prepare_start_ns: int
    prepare_end_ns: int


class PreparedFrames:
    def __init__(self, source_read, factory, capacity=2, shutdown_source=lambda: None, timeout=10.0):
        if capacity not in (1, 2) or timeout <= 0:
            raise ValueError('Capacity must be one/two and timeout positive')
        self._source_read, self._factory = source_read, factory
        self._shutdown_source, self._timeout, self.capacity = shutdown_source, timeout, capacity
        self._condition = threading.Condition()
        self._ready = threading.Event()
        self._queue = deque()
        self._occupied = set()
        self._leased = None
        self._started = self._stop = self._eof = False
        self._error = self._cleanup_error = self._close_error = None
        self._close_attempted = False
        self.metadata = {}
        self.stats = dict(prepared=0, delivered=0, max_owned=0, owner_thread_id=None,
                          closed=False, joined=False, released=0, discarded_on_cancel=0)
        self._consumer = threading.get_ident()
        self._thread = threading.Thread(target=self._produce, name='seaqr-prepare-v23', daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout):
            with self._condition:
                self._stop = True
                self._condition.notify_all()
            raise RuntimeError('Preparation initialization timed out; worker may still be active')
        if self._error is not None:
            with self._condition:
                self._stop = True
                self._condition.notify_all()
            self._thread.join(timeout)
            if self._thread.is_alive():
                raise RuntimeError('Initialization failed and worker did not stop') from self._error
            raise self._error

    def _owner(self):
        if threading.get_ident() != self._consumer:
            raise RuntimeError('PreparedFrames has one consumer/close owner')

    def _produce(self):
        processor = None
        try:
            self.stats['owner_thread_id'] = threading.get_ident()
            processor = self._factory()
            self.metadata = dict(processor.metadata)
            self._ready.set()
            index = 0
            while True:
                with self._condition:
                    self._condition.wait_for(lambda: self._stop or
                        (self._started and len(self._occupied) < self.capacity))
                    if self._stop:
                        break
                    slot = index % 2
                    if slot in self._occupied:
                        raise RuntimeError('Attempt to overwrite a leased preparation slot')
                    self._occupied.add(slot)  # reserve BEFORE reading or GPU work
                    self.stats['max_owned'] = max(self.stats['max_owned'], len(self._occupied))
                frame, wait_ms = self._source_read()
                if frame is None:
                    with self._condition:
                        self._occupied.remove(slot)
                        self._eof = True
                        self._condition.notify_all()
                        self._condition.wait_for(lambda: self._stop)
                    break
                if frame.index != index:
                    raise ValueError('Preparation input order changed')
                with self._condition:
                    if self._stop:
                        self.stats['discarded_on_cancel'] += 1
                        break
                begin = time.perf_counter_ns()
                result = processor.prepare(frame, slot)
                end = time.perf_counter_ns()
                job = Job(frame, result, slot, wait_ms, begin, end)
                with self._condition:
                    self.stats['prepared'] += 1
                    if self._stop:
                        self.stats['discarded_on_cancel'] += 1
                        break
                    self._queue.append(job)
                    self._condition.notify_all()
                index += 1
                frame = result = job = None
        except BaseException as exc:
            with self._condition:
                self._error = exc
                self._ready.set()
                self._condition.notify_all()
                # A failure in the future frame must not destroy a prior lease.
                if processor is not None:
                    self._condition.wait_for(lambda: self._stop)
        finally:
            if processor is not None:
                try:
                    processor.close()
                except BaseException as exc:
                    self._cleanup_error = exc
            with self._condition:
                self._condition.notify_all()

    def start(self):
        self._owner()
        with self._condition:
            if self._stop:
                raise RuntimeError('Preparation is closed/stopping')
            self._started = True
            self._condition.notify_all()

    def read(self):
        self._owner()
        self.start()
        with self._condition:
            if self._leased is not None:
                self._occupied.remove(self._leased.slot)
                self.stats['released'] += 1
                self._leased = None
                self._condition.notify_all()
            self._condition.wait_for(lambda: self._queue or self._error is not None or self._eof or self._stop)
            if self._queue:
                self._leased = self._queue.popleft()
                self.stats['delivered'] += 1
                return self._leased
            if self._error is not None:
                raise self._error
            if self._stop:
                raise RuntimeError('Preparation stopped')
            return None

    def close(self):
        self._owner()
        if self._close_attempted:
            if self._close_error is not None:
                raise self._close_error
            return
        self._close_attempted = True
        with self._condition:
            self._stop = True
            self._condition.notify_all()
        source_error = None
        try:
            self._shutdown_source()
        except BaseException as exc:
            source_error = exc
        self._thread.join(self._timeout)
        if self._thread.is_alive():
            self._close_error = RuntimeError('Preparation worker did not stop; resources remain worker-owned')
        else:
            self.stats['joined'] = True
            self.stats['discarded_on_cancel'] += len(self._queue)
            self._queue.clear()
            if self._leased is not None:
                self.stats['released'] += 1
            self._leased = None
            self._occupied.clear()
            self._close_error = self._cleanup_error or self._error or source_error
            self.stats['closed'] = self._close_error is None
        if self._close_error is not None:
            raise self._close_error
