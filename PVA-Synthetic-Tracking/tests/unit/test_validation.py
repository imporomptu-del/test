from __future__ import annotations

import unittest

import numpy as np

from tiny_target.types import Discontinuity, Frame, TimestampSource
from tiny_target.validation import FrameSequenceValidator


def make_frame(index: int, timestamp_ns: int, sequence: int | None = None) -> Frame:
    return Frame(
        image=np.zeros((2, 3), dtype=np.uint8),
        timestamp_ns=timestamp_ns,
        timestamp_source=TimestampSource.MANIFEST,
        frame_index=index,
        sequence=index if sequence is None else sequence,
        source_id="fixture",
        bit_depth=8,
    )


class FrameSequenceValidatorTests(unittest.TestCase):
    def test_marks_timestamp_and_sequence_gaps(self) -> None:
        validator = FrameSequenceValidator(expected_interval_ns=100, gap_factor=1.5)
        first = validator.observe(make_frame(0, 0, 10))
        second = validator.observe(make_frame(1, 250, 12))

        self.assertEqual(first.discontinuities, ())
        self.assertEqual(
            set(second.discontinuities),
            {Discontinuity.SEQUENCE_GAP, Discontinuity.TIMESTAMP_GAP},
        )

    def test_marks_duplicate_and_non_monotonic_timestamps(self) -> None:
        validator = FrameSequenceValidator(expected_interval_ns=100)
        validator.observe(make_frame(0, 100))
        duplicate = validator.observe(make_frame(1, 100))
        backwards = validator.observe(make_frame(2, 50))

        self.assertIn(Discontinuity.DUPLICATE_TIMESTAMP, duplicate.discontinuities)
        self.assertIn(
            Discontinuity.NON_MONOTONIC_TIMESTAMP,
            backwards.discontinuities,
        )


if __name__ == "__main__":
    unittest.main()

