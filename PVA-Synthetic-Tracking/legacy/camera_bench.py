"""SkyEye62AM live capture helpers for skymove bench scripts."""

from __future__ import annotations

import queue
import time
from typing import Any

import numpy as np

import pipeline as pl

# Full frame at RESOLUTION_INDEX=2 (see pipeline.py / save_video_of_two.py)
SKY26_FULL_WIDTH = 3184
SKY26_FULL_HEIGHT = 2124


def center_crop(gray: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    h, w = gray.shape[:2]
    if w == out_w and h == out_h:
        return np.ascontiguousarray(gray, dtype=np.uint8)
    x0 = max(0, (w - out_w) // 2)
    y0 = max(0, (h - out_h) // 2)
    return np.ascontiguousarray(
        gray[y0 : y0 + out_h, x0 : x0 + out_w], dtype=np.uint8,
    )


class BenchCameraSession:
    """Pull-mode SkyEye session for synchronous bench frame capture."""

    def __init__(
        self,
        hcam: Any,
        capture: pl.CaptureThread,
        cam_idx: int,
        sensor_w: int,
        sensor_h: int,
        out_w: int,
        out_h: int,
        cam_info: Any,
    ):
        self.hcam = hcam
        self.capture = capture
        self.cam_idx = cam_idx
        self.sensor_w = sensor_w
        self.sensor_h = sensor_h
        self.out_w = out_w
        self.out_h = out_h
        self.cam_info = cam_info

    def close(self) -> None:
        self.capture.stop()
        try:
            self.hcam.Close()
        except Exception:
            pass

    def grab_gray(self, timeout: float = 15.0) -> np.ndarray:
        gray = pl.frame_q.get(timeout=timeout)
        return center_crop(gray, self.out_w, self.out_h)


def _drain_frame_queue() -> None:
    while True:
        try:
            pl.frame_q.get_nowait()
        except queue.Empty:
            break


def open_bench_camera(
    camera_id: int | None = None,
    camera_name: str = "SkyEye",
    *,
    roi_w: int | None = None,
    roi_h: int | None = None,
    resolution_index: int | None = None,
    requested_gain: int | None = None,
) -> BenchCameraSession:
    """Open SkyEye62AM; optional center crop to roi_w×roi_h."""
    ri = resolution_index if resolution_index is not None else pl.RESOLUTION_INDEX
    gain = requested_gain if requested_gain is not None else pl.REQUESTED_GAIN
    hcam, sensor_w, sensor_h, cam_idx, cam_info = pl.open_skyeye_camera(
        camera_id, camera_name, ri, gain,
    )
    if roi_w is not None and roi_h is not None:
        out_w = roi_w - (roi_w % 2)
        out_h = roi_h - (roi_h % 2)
        if out_w > sensor_w or out_h > sensor_h:
            raise SystemExit(
                f"Requested {out_w}×{out_h} exceeds sensor {sensor_w}×{sensor_h}"
            )
        print(
            f"Output crop {out_w}×{out_h} centered on {sensor_w}×{sensor_h} sensor",
            flush=True,
        )
    else:
        out_w, out_h = sensor_w, sensor_h

    capture = pl.CaptureThread(hcam, sensor_w, sensor_h)
    capture.start()
    _drain_frame_queue()
    for _ in range(5):
        pl.frame_q.get(timeout=15.0)

    return BenchCameraSession(
        hcam, capture, cam_idx, sensor_w, sensor_h, out_w, out_h, cam_info,
    )


def capture_grays(
    session: BenchCameraSession, n: int,
) -> tuple[list[np.ndarray], list[float]]:
    """Return gray frames + per-frame crop/copy ms (replaces ASI debayer timing)."""
    grays: list[np.ndarray] = []
    crop_ms: list[float] = []
    print(f"Streaming {n} live frames…", flush=True)
    for i in range(n):
        t0 = time.perf_counter()
        gray = session.grab_gray()
        crop_ms.append((time.perf_counter() - t0) * 1000.0)
        grays.append(gray)
        if (i + 1) % 5 == 0 or i + 1 == n:
            print(f"  frame {i + 1}/{n}", flush=True)
    return grays, crop_ms
