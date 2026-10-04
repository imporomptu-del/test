from __future__ import annotations

import unittest

import numpy as np

from tiny_target.motion import MotionCorrespondences, PvaMotionConfig, PvaMotionError
from tiny_target.motion.pva_pyrlk import _feature_eligibility, _motion_u8
from tiny_target.types import Frame, TimestampSource


def frame(image: np.ndarray, *, bit_depth: int = 16) -> Frame:
    return Frame(
        image=image,
        timestamp_ns=0,
        frame_index=0,
        source_id="fixture",
        bit_depth=bit_depth,
        timestamp_source=TimestampSource.MANIFEST,
    )


class MotionTypeTests(unittest.TestCase):
    def test_raw16_motion_conversion_is_explicit_high_byte(self) -> None:
        source = np.array([[0, 255, 256, 65535]], dtype=np.uint16)
        converted = _motion_u8(frame(source))
        np.testing.assert_array_equal(converted, [[0, 0, 1, 255]])
        self.assertEqual(converted.dtype, np.uint8)

    def test_motion_config_rejects_non_pva_backend_and_unknown_keys(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be PVA"):
            PvaMotionConfig.from_mapping({"backend": "CPU"})
        with self.assertRaisesRegex(ValueError, "Unknown"):
            PvaMotionConfig.from_mapping({"surprise": 4})

    def test_feature_eligibility_excludes_border_mask_saturation_and_region(self) -> None:
        image = np.zeros((20, 20), np.uint16)
        image[10, 10] = 65535
        valid = np.ones((20, 20), bool)
        valid[15, 15] = False
        source = Frame(
            image=image,
            valid_mask=valid,
            timestamp_ns=0,
            frame_index=0,
            source_id="fixture",
            bit_depth=16,
            timestamp_source=TimestampSource.MANIFEST,
        )
        points = np.array([[0, 0], [5, 5], [10, 10], [15, 15], [18, 10]], np.float32)
        eligible, reasons = _feature_eligibility(
            points,
            source,
            (20, 20),
            PvaMotionConfig(
                feature_image_scale=1,
                feature_border_px=1,
                saturated_neighborhood_radius_px=0,
                exclusion_regions_xyxy=((4, 4, 7, 7),),
            ),
        )
        self.assertEqual(eligible.tolist(), [False, False, False, False, True])
        self.assertEqual(reasons["unreliable_border"], 1)
        self.assertEqual(reasons["exclusion_region"], 1)
        self.assertEqual(reasons["saturated_neighborhood"], 1)
        self.assertEqual(reasons["invalid_source_mask"], 1)

    def test_correspondence_contract_copies_and_serializes_nonfinite_fb_as_null(self) -> None:
        previous = np.array([[1, 2]], np.float32)
        value = MotionCorrespondences(
            previous_points=previous,
            current_points=np.array([[2, 3]], np.float32),
            harris_scores=np.array([4], np.float32),
            forward_backward_error_px=np.array([np.nan], np.float32),
            previous_frame_index=1,
            current_frame_index=2,
            previous_timestamp_ns=10,
            current_timestamp_ns=20,
            full_image_size=(20, 10),
            motion_image_size=(10, 5),
            metrics={},
            timings_ms={},
            backends={"cpu_fallback": False},
        )
        previous[0] = 99
        self.assertEqual(value.previous_points.tolist(), [[1.0, 2.0]])
        self.assertFalse(value.previous_points.flags.writeable)
        self.assertIsNone(value.to_dict()["correspondences"][0]["forward_backward_error_px"])

    def test_building_estimator_off_jetson_has_clear_error(self) -> None:
        try:
            import vpi  # type: ignore[import-not-found]  # noqa: F401
        except ImportError:
            from tiny_target.motion import PvaPyrLkMotionEstimator

            with self.assertRaisesRegex(PvaMotionError, "Jetson"):
                PvaPyrLkMotionEstimator()


if __name__ == "__main__":
    unittest.main()
