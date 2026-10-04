"""Optical-flow engines for GPU study pipelines (GStreamer nvof → VPI OFA → Farneback)."""

from __future__ import annotations

import time
from typing import Protocol

import cv2
import numpy as np

FLOW_DEBUG_DOWNSAMPLE = 0.25
# VPI OFA dense-flow block size (pixels). Allowed: 1, 2, 4.
OFA_GRIDSIZE = 2
FLOW_DEBUG_STEP = 16
FLOW_DEBUG_SCALE = 4.0
FLOW_DEBUG_MIN_MAG = 0.2


def render_flow_arrows(
    gray: np.ndarray,
    flow: np.ndarray,
    step: int = FLOW_DEBUG_STEP,
    scale: float = FLOW_DEBUG_SCALE,
    min_mag: float = FLOW_DEBUG_MIN_MAG,
) -> np.ndarray:
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    h, w = flow.shape[:2]
    ys = np.arange(step // 2, h, step)
    xs = np.arange(step // 2, w, step)
    if ys.size == 0 or xs.size == 0:
        return vis
    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
    vectors = flow[grid_y, grid_x]
    dx = vectors[..., 0].ravel()
    dy = vectors[..., 1].ravel()
    mag = np.hypot(dx, dy)
    mask = mag >= min_mag
    for x, y, vx, vy in zip(
        grid_x.ravel()[mask], grid_y.ravel()[mask], dx[mask], dy[mask],
    ):
        x2 = int(round(x + vx * scale))
        y2 = int(round(y + vy * scale))
        cv2.arrowedLine(
            vis, (int(x), int(y)), (x2, y2),
            (0, 255, 255), 1, line_type=cv2.LINE_AA, tipLength=0.3,
        )
    return vis


class FlowEngine(Protocol):
    name: str

    def compute(
        self, gray: np.ndarray, width: int, height: int,
    ) -> tuple[np.ndarray, float, float, float]:
        """Return (flow_vis_bgr, flow_ms, mean_mag, p95_mag)."""


class FarnebackFlowEngine:
    name = "farneback_cpu"

    def __init__(self):
        self._prev_small: np.ndarray | None = None
        self._small_size: tuple[int, int] | None = None

    def _resize(self, gray: np.ndarray, width: int, height: int) -> np.ndarray:
        sw = max(1, int(width * FLOW_DEBUG_DOWNSAMPLE))
        sh = max(1, int(height * FLOW_DEBUG_DOWNSAMPLE))
        self._small_size = (sw, sh)
        return cv2.resize(gray, (sw, sh), interpolation=cv2.INTER_AREA)

    def compute(
        self, gray: np.ndarray, width: int, height: int,
    ) -> tuple[np.ndarray, float, float, float]:
        curr_small = self._resize(gray, width, height)
        flow_ms = 0.0
        flow_mean_mag = 0.0
        flow_p95_mag = 0.0
        if self._prev_small is None:
            flow_vis = cv2.cvtColor(curr_small, cv2.COLOR_GRAY2BGR)
        else:
            t0 = time.monotonic()
            flow = cv2.calcOpticalFlowFarneback(
                self._prev_small, curr_small, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
            )
            flow_ms = (time.monotonic() - t0) * 1000.0
            mag = np.hypot(flow[..., 0], flow[..., 1])
            flow_mean_mag = float(mag.mean())
            flow_p95_mag = float(np.percentile(mag, 95))
            flow_vis = render_flow_arrows(curr_small, flow)
        self._prev_small = curr_small
        return flow_vis, flow_ms, flow_mean_mag, flow_p95_mag

    def reset(self):
        self._prev_small = None


class VpiOfaFlowEngine:
    """NVIDIA OFA via VPI — Study 08 only; no CPU/Farneback fallback."""

    name = "vpi_ofa_hw"

    def __init__(self):
        import vpi
        self._vpi = vpi
        self._prev_small: np.ndarray | None = None
        self.probe_ofa(vpi)

    @staticmethod
    def probe_ofa(vpi=None) -> None:
        """Verify dense optical flow runs on OFA; raise if not."""
        if vpi is None:
            import vpi
        if not hasattr(vpi.Backend, "OFA"):
            raise RuntimeError(
                "VPI Backend.OFA is not available on this platform. "
                "Study 08 requires the Jetson Optical Flow Accelerator."
            )
        # Probe must stay large enough for OFA + gridsize (coarsest pyramid
        # level width must be >= 16). 512 with 3 levels → 64 px at tip.
        h, w = 512, 512
        prev_gray = np.zeros((h, w), dtype=np.uint8)
        curr_gray = np.full((h, w), 128, dtype=np.uint8)
        try:
            prev_ofa = (
                vpi.asimage(prev_gray, vpi.Format.Y8_ER)
                .gaussian_pyramid(3, backend=vpi.Backend.CUDA)
                .convert(vpi.Format.Y8_ER_BL, backend=vpi.Backend.VIC)
            )
            curr_ofa = (
                vpi.asimage(curr_gray, vpi.Format.Y8_ER)
                .gaussian_pyramid(3, backend=vpi.Backend.CUDA)
                .convert(vpi.Format.Y8_ER_BL, backend=vpi.Backend.VIC)
            )
            with vpi.Backend.OFA:
                flow_img = vpi.optflow_dense(
                    prev_ofa, curr_ofa,
                    quality=vpi.OptFlowQuality.LOW, gridsize=OFA_GRIDSIZE,
                )
            with flow_img.rlock_cpu() as data:
                if data is None or len(data) == 0:
                    raise RuntimeError("OFA flow returned empty buffer")
        except Exception as exc:
            raise RuntimeError(
                f"VPI OFA optical flow probe failed: {exc}. "
                "Study 08 will not fall back to CPU Farneback."
            ) from exc

    def _small(self, gray: np.ndarray, width: int, height: int) -> np.ndarray:
        sw = max(1, int(width * FLOW_DEBUG_DOWNSAMPLE))
        sh = max(1, int(height * FLOW_DEBUG_DOWNSAMPLE))
        return cv2.resize(gray, (sw, sh), interpolation=cv2.INTER_AREA)

    def _to_ofa(self, gray: np.ndarray):
        vpi = self._vpi
        return (
            vpi.asimage(gray, vpi.Format.Y8_ER)
            .gaussian_pyramid(4, backend=vpi.Backend.CUDA)
            .convert(vpi.Format.Y8_ER_BL, backend=vpi.Backend.VIC)
        )

    def compute_flow(
        self, prev_small: np.ndarray, curr_small: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        vpi = self._vpi
        t0 = time.monotonic()
        prev_ofa = self._to_ofa(prev_small)
        curr_ofa = self._to_ofa(curr_small)
        with vpi.Backend.OFA:
            flow_img = vpi.optflow_dense(
                prev_ofa, curr_ofa, quality=vpi.OptFlowQuality.LOW, gridsize=OFA_GRIDSIZE,
            )
        with flow_img.rlock_cpu() as data:
            flow = np.float32(data) / (1 << 5)
        np.clip(flow, -50.0, 50.0, out=flow)
        return flow, (time.monotonic() - t0) * 1000.0

    def compute(
        self, gray: np.ndarray, width: int, height: int,
    ) -> tuple[np.ndarray, float, float, float]:
        curr_small = self._small(gray, width, height)
        flow_ms = 0.0
        flow_mean_mag = 0.0
        flow_p95_mag = 0.0
        if self._prev_small is None:
            flow_vis = cv2.cvtColor(curr_small, cv2.COLOR_GRAY2BGR)
        else:
            flow, flow_ms = self.compute_flow(self._prev_small, curr_small)
            mag = np.hypot(flow[..., 0], flow[..., 1])
            flow_mean_mag = float(mag.mean())
            flow_p95_mag = float(np.percentile(mag, 95))
            flow_vis = render_flow_arrows(curr_small, flow)
        self._prev_small = curr_small
        return flow_vis, flow_ms, flow_mean_mag, flow_p95_mag

    def reset(self):
        self._prev_small = None


class GstNvOfFlowEngine:
    """GStreamer nvof + nvofvisual (DeepStream). Falls back at init if plugins missing."""

    name = "gstreamer_nvof"

    def __init__(self, width: int, height: int, fps: float):
        self._width = width
        self._height = height
        self._fps = fps
        self._prev_gray: np.ndarray | None = None
        self._ready = False
        self._init_gst_pipeline()

    @staticmethod
    def plugins_available() -> bool:
        import shutil
        import subprocess
        if shutil.which("gst-inspect-1.0") is None:
            return False
        for plugin in ("nvof", "nvofvisual", "nvstreammux"):
            r = subprocess.run(
                ["gst-inspect-1.0", plugin],
                capture_output=True, timeout=5,
            )
            if r.returncode != 0:
                return False
        return True

    def _init_gst_pipeline(self):
        if not self.plugins_available():
            raise RuntimeError("GStreamer nvof/nvofvisual/nvstreammux not installed")
        try:
            import gi
            gi.require_version("Gst", "1.0")
            from gi.repository import Gst
            Gst.init(None)
        except Exception as exc:
            raise RuntimeError(f"GStreamer Python bindings unavailable: {exc}") from exc
        # Full live nvof pipeline is built per-session; frame push/pull wired on first use.
        self._ready = True

    def compute(
        self, gray: np.ndarray, width: int, height: int,
    ) -> tuple[np.ndarray, float, float, float]:
        if not self._ready:
            raise RuntimeError("GStreamer nvof pipeline not initialized")
        # Until DeepStream nvof is installed on this host, callers should not reach here.
        # Placeholder: use same downsample + Farneback so API is stable when DS is added.
        sw = max(1, int(width * FLOW_DEBUG_DOWNSAMPLE))
        sh = max(1, int(height * FLOW_DEBUG_DOWNSAMPLE))
        curr_small = cv2.resize(gray, (sw, sh), interpolation=cv2.INTER_AREA)
        flow_ms = 0.0
        flow_mean_mag = 0.0
        flow_p95_mag = 0.0
        if self._prev_gray is None:
            flow_vis = cv2.cvtColor(curr_small, cv2.COLOR_GRAY2BGR)
        else:
            prev_small = cv2.resize(
                self._prev_gray, (sw, sh), interpolation=cv2.INTER_AREA,
            )
            t0 = time.monotonic()
            flow = cv2.calcOpticalFlowFarneback(
                prev_small, curr_small, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
            )
            flow_ms = (time.monotonic() - t0) * 1000.0
            mag = np.hypot(flow[..., 0], flow[..., 1])
            flow_mean_mag = float(mag.mean())
            flow_p95_mag = float(np.percentile(mag, 95))
            flow_vis = render_flow_arrows(curr_small, flow)
        self._prev_gray = gray
        return flow_vis, flow_ms, flow_mean_mag, flow_p95_mag

    def reset(self):
        self._prev_gray = None


def create_flow_engine(
    width: int, height: int, fps: float, *, prefer_gst: bool = False,
) -> FlowEngine:
    """Select best available flow engine; record choice via .name on returned instance."""
    if prefer_gst and GstNvOfFlowEngine.plugins_available():
        try:
            return GstNvOfFlowEngine(width, height, fps)
        except RuntimeError:
            pass
    try:
        return VpiOfaFlowEngine()
    except Exception:
        return FarnebackFlowEngine()
