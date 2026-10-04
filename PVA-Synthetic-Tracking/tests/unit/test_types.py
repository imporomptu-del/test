from __future__ import annotations

import unittest

import numpy as np

from tiny_target.types import Discontinuity, Frame, TimestampSource


class FrameTests(unittest.TestCase):
    def test_preserves_uint16_and_makes_arrays_read_only(self) -> None:
        image = np.arange(12, dtype=np.uint16).reshape(3, 4)
        mask = np.ones((3, 4), dtype=bool)
        frame = Frame(
            image=image,
            timestamp_ns=10,
            source_timestamp_ns=1_000_000_010,
            timestamp_source=TimestampSource.SIDECAR_UNIX_NS,
            frame_index=0,
            sequence=20,
            source_id="fixture",
            bit_depth=16,
            valid_mask=mask,
        )

        self.assertEqual(frame.image.dtype, np.dtype("uint16"))
        self.assertFalse(frame.image.flags.writeable)
        self.assertFalse(frame.valid_mask.flags.writeable)
        self.assertTrue(image.flags.writeable)
        image[0, 0] = 999
        self.assertNotEqual(frame.image[0, 0], image[0, 0])
        self.assertEqual(frame.metadata_dict()["shape"], [3, 4])
        with self.assertRaises(ValueError):
            frame.image[0, 0] = 0

    def test_rejects_color_image(self) -> None:
        with self.assertRaisesRegex(ValueError, "2-D grayscale"):
            Frame(
                image=np.zeros((2, 2, 3), dtype=np.uint8),
                timestamp_ns=0,
                timestamp_source=TimestampSource.MANIFEST,
                frame_index=0,
                source_id="fixture",
                bit_depth=8,
            )

    def test_discontinuities_are_deduplicated(self) -> None:
        frame = Frame(
            image=np.zeros((2, 2), dtype=np.uint8),
            timestamp_ns=0,
            timestamp_source=TimestampSource.MANIFEST,
            frame_index=0,
            source_id="fixture",
            bit_depth=8,
        )
        changed = frame.with_discontinuities(
            Discontinuity.TIMESTAMP_GAP,
            Discontinuity.TIMESTAMP_GAP,
        )
        self.assertEqual(changed.discontinuities, (Discontinuity.TIMESTAMP_GAP,))


if __name__ == "__main__":
    unittest.main()
