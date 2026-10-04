"""Shared decoded/prepared frame admission and ordered GPU-stage release."""
import threading
import time


class StageCancelled(RuntimeError):
    pass


class Admission:
    """Capacity includes decode-in-flight, queued, preparing and main leases."""
    def __init__(self, capacity=2):
        if type(capacity) is not int or capacity != 2:
            raise ValueError('Frozen shared capacity is two frames')
        self.capacity = capacity
        self.condition = threading.Condition()
        self.stopped = False
        self.held = set()
        self.events = []
        self.next_index = 0
        self.maximum = 0

    def acquire(self, index):
        request = time.perf_counter_ns()  # BEFORE waiting: never hide admission age
        with self.condition:
            if index != self.next_index:
                raise ValueError('Admission must follow source order')
            self.condition.wait_for(lambda: self.stopped or len(self.held) < self.capacity)
            if self.stopped:
                raise StageCancelled('Admission stopped')
            if index != self.next_index:
                raise ValueError('Duplicate/concurrent admission request')
            self.next_index += 1
            self.held.add(index)
            self.maximum = max(self.maximum, len(self.held))
            event = dict(frame=index, request_ns=request, admitted_ns=time.perf_counter_ns(),
                         released_ns=None, disposition=None)
            self.events.append(event)
            return dict(event)

    def release(self, index, disposition='consumed'):
        with self.condition:
            if index not in self.held or disposition not in ('consumed','eof','error'):
                raise ValueError('Unknown/double admission release')
            self.held.remove(index)
            self.events[index].update(released_ns=time.perf_counter_ns(), disposition=disposition)
            self.condition.notify_all()

    def stop(self):
        with self.condition:
            self.stopped = True
            self.condition.notify_all()

    def snapshot(self):
        with self.condition:
            return dict(capacity=self.capacity, maximum=self.maximum, held=sorted(self.held),
                        stopped=self.stopped, events=[dict(e) for e in self.events])


class GpuRelease:
    """Frame n's full-resolution warp starts only after detector n-1 completes."""
    def __init__(self):
        self.condition = threading.Condition()
        self.completed = -1
        self.stopped = False
        self.release_times = []

    def detector_complete(self, frame):
        with self.condition:
            if self.stopped or frame != self.completed+1:
                raise ValueError('Invalid detector completion order')
            now = time.perf_counter_ns()
            self.completed = frame
            self.release_times.append(now)
            self.condition.notify_all()
            return now

    def wait(self, frame):
        if type(frame) is not int or frame < 0:
            raise ValueError('Invalid GPU frame index')
        with self.condition:
            self.condition.wait_for(lambda: self.stopped or self.completed >= frame-1)
            if self.stopped:
                raise StageCancelled('GPU release stopped')

    def stop(self):
        with self.condition:
            self.stopped = True
            self.condition.notify_all()

    def snapshot(self):
        with self.condition:
            return dict(completed=self.completed,stopped=self.stopped,release_times_ns=list(self.release_times))
