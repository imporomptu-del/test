from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from tiny_target.detection import (
    CudaShiftAndStack,
    CudaSyntheticTrackingError,
    MatchedFilterFrame,
    ReferenceShiftAndStack,
    SyntheticTrackingConfig,
)
from tiny_target.detection.synthetic_cuda import DEFAULT_CUDA_LIBRARY


def matched(
    response: np.ndarray,
    valid: np.ndarray,
    index: int,
    timestamp_ns: int,
    *,
    polarity: str = "bright",
) -> MatchedFilterFrame:
    return MatchedFilterFrame(
        response=np.asarray(response, np.float32),
        phase_index=np.zeros(response.shape, np.uint16),
        valid_mask=np.asarray(valid, bool),
        valid_support_count=np.asarray(valid, np.uint16),
        timestamp_ns=timestamp_ns,
        frame_index=index,
        reference_frame_index=0,
        segment_index=0,
        detection_ready=True,
        polarity=polarity,
        backend="fixture",
        kernel_metadata={},
        metrics={},
        timings_ms={},
    )


def config(
    frame_count: int,
    *,
    batch_size: int = 3,
    support: float = 1.0,
) -> SyntheticTrackingConfig:
    return SyntheticTrackingConfig(
        backend="cuda",
        window_frames=frame_count,
        window_stride_frames=1,
        vx_min_px_s=-1,
        vx_max_px_s=1,
        vy_min_px_s=-1,
        vy_max_px_s=1,
        velocity_step_px_s=1,
        min_valid_fraction=support,
        velocity_batch_size=batch_size,
        cuda_threads_per_block=256,
    )


def compare(reference: object, cuda: object, tolerance: float = 2e-6) -> None:
    np.testing.assert_allclose(
        cuda.score, reference.score, rtol=1e-6, atol=tolerance
    )
    np.testing.assert_array_equal(cuda.velocity_index, reference.velocity_index)
    np.testing.assert_array_equal(
        cuda.valid_support_count, reference.valid_support_count
    )
    np.testing.assert_array_equal(cuda.valid_mask, reference.valid_mask)


class SyntheticCudaIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not DEFAULT_CUDA_LIBRARY.is_file():
            raise unittest.SkipTest(
                f"CUDA test library is not built at {DEFAULT_CUDA_LIBRARY}"
            )

    def test_integer_shift_matches_reference_exactly(self) -> None:
        shape = (13, 17)
        frames = []
        for index, timestamp_ns in enumerate((0, 1_000_000_000, 2_000_000_000)):
            response = np.zeros(shape, np.float32)
            response[7 - index, 7 + index] = 4
            frames.append(
                matched(response, np.ones(shape, bool), index, timestamp_ns)
            )
        settings = config(len(frames), batch_size=2)
        reference = ReferenceShiftAndStack(settings).integrate(frames)
        cuda = CudaShiftAndStack(settings).integrate(frames)
        compare(reference, cuda, tolerance=0)
        self.assertEqual(cuda.metrics["backend"], "cuda")
        self.assertEqual(cuda.metrics["cuda"]["velocity_batch_count"], 5)

    def test_fractional_irregular_masked_cases_match_with_tolerance(self) -> None:
        for seed in range(6):
            rng = np.random.default_rng(seed)
            frame_count = 3 + seed % 4
            shape = (11 + 2 * seed, 14 + seed)
            intervals = rng.integers(70_000_000, 260_000_000, frame_count - 1)
            timestamps = np.concatenate(([0], np.cumsum(intervals))).astype(np.int64)
            frames = []
            for index, timestamp_ns in enumerate(timestamps):
                response = rng.normal(0, 1, shape).astype(np.float32)
                valid = rng.random(shape) > (0.05 + 0.02 * (seed % 3))
                frames.append(
                    matched(response, valid, index, int(timestamp_ns))
                )
            settings = config(
                frame_count,
                batch_size=1 + seed % 5,
                support=0.5 if seed % 2 else 1.0,
            )
            reference = ReferenceShiftAndStack(settings).integrate(frames)
            cuda = CudaShiftAndStack(settings).integrate(frames)
            compare(reference, cuda, tolerance=3e-6)

    def test_batch_size_repeatability_and_stale_output_protection(self) -> None:
        rng = np.random.default_rng(75)
        shape = (19, 23)
        timestamps = (0, 170_000_000, 390_000_000, 710_000_000)
        first_frames = [
            matched(
                rng.normal(0, 1, shape).astype(np.float32),
                np.ones(shape, bool),
                index,
                timestamp,
            )
            for index, timestamp in enumerate(timestamps)
        ]
        zero_frames = [
            matched(
                np.zeros(shape, np.float32),
                np.ones(shape, bool),
                index,
                timestamp,
            )
            for index, timestamp in enumerate(timestamps)
        ]
        baseline = None
        for batch_size in (1, 2, 4, 20):
            settings = config(len(timestamps), batch_size=batch_size)
            tracker = CudaShiftAndStack(settings)
            first = tracker.integrate(first_frames)
            repeated = tracker.integrate(first_frames)
            compare(first, repeated, tolerance=0)
            cleared = tracker.integrate(zero_frames)
            reference_zero = ReferenceShiftAndStack(settings).integrate(zero_frames)
            compare(reference_zero, cleared, tolerance=0)
            if baseline is None:
                baseline = first
            else:
                compare(baseline, first, tolerance=0)

    def test_dark_polarity_matches_reference(self) -> None:
        rng = np.random.default_rng(17)
        frames = [
            matched(
                rng.normal(0, 1, (15, 18)).astype(np.float32),
                np.ones((15, 18), bool),
                index,
                index * 230_000_000,
                polarity="dark",
            )
            for index in range(4)
        ]
        settings = config(4)
        compare(
            ReferenceShiftAndStack(settings).integrate(frames),
            CudaShiftAndStack(settings).integrate(frames),
        )


class SyntheticCudaFailureTests(unittest.TestCase):
    def test_missing_library_fails_without_reference_fallback(self) -> None:
        settings = SyntheticTrackingConfig(
            backend="cuda",
            window_frames=4,
            window_stride_frames=1,
            vx_min_px_s=0,
            vx_max_px_s=0,
            vy_min_px_s=0,
            vy_max_px_s=0,
            velocity_step_px_s=1,
            min_valid_fraction=1,
            cuda_library_path="definitely-missing-cuda-library.so",
        )
        with self.assertRaisesRegex(
            CudaSyntheticTrackingError, "CUDA synthetic-tracking library is missing"
        ):
            CudaShiftAndStack(settings, base_path=Path("/tmp"))


if __name__ == "__main__":
    unittest.main()
