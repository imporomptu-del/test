"""Smoke-only private-state and actual detector-learning-input checks."""
from unittest.mock import patch
from replay_tracking_v27 import digest


def state_of(tracker):
    return dict(managers={p: vars(m) for p, m in tracker.managers.items()},
        extents=tracker.extents, qualified=tracker.qualified, summary=tracker.summary,
        ever_qualified=tracker.ever_qualified, previous_records=tracker.previous_records,
        previous_timestamp_ns=tracker.previous_timestamp_ns,
        quality={k: vars(v) for k, v in tracker.quality.items()})


class StateAudit:
    def __init__(self, expected):
        self.expected = expected
        self.rows = []
        self.learning = None

    def install(self, stack):
        from tiny_target.visible_baseline import VisibleTracks, VisiblePointDetector
        detector, update = VisiblePointDetector.update, VisibleTracks.update
        audit = self
        def checked_detector(instance, image, valid, segment, learning_centers=()):
            if audit.learning is not None:
                raise AssertionError('Repeated/unconsumed learning input')
            audit.learning = digest(learning_centers)
            return detector(instance, image, valid, segment, learning_centers)
        def checked_update(instance, proposals, index, timestamp_ns, segment, matrix, shape):
            if index != len(audit.rows) or audit.learning is None:
                raise AssertionError('Smoke tracker/learning order changed')
            tracks, metrics = update(instance, proposals, index, timestamp_ns, segment, matrix, shape)
            row = dict(frame=index, output=digest([tracks, metrics]), state=digest(state_of(instance)), learning=audit.learning)
            if row != audit.expected[index]:
                raise AssertionError('Smoke output/private state/learning mismatch at ' + str(index))
            audit.rows.append(row)
            audit.learning = None
            return tracks, metrics
        stack.enter_context(patch.object(VisiblePointDetector, 'update', checked_detector))
        stack.enter_context(patch.object(VisibleTracks, 'update', checked_update))

    def finish(self, count):
        if count != len(self.expected) or self.rows != self.expected or self.learning is not None:
            raise AssertionError('Incomplete smoke state gate')
