"""Detection backends for GPU study pipelines — one class per technology."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

MOG2_HISTORY = 200
MOG2_VAR_THRESHOLD = 16
MOG2_DETECT_SHADOWS = False
MIN_BLOB_AREA = 1
MAX_BLOB_AREA = 50
MIN_DETECTION_INTENSITY = 20


@dataclass
class ProcessResult:
    detections: list[dict]
    fg_mask: np.ndarray
    fg_clean: np.ndarray
    mog2_ms: float
    fg_pixels: int
    clean_pixels: int
    extra: dict[str, Any] = field(default_factory=dict)


def _connected_components_detect(gray: np.ndarray, fg_clean: np.ndarray) -> list[dict]:
    num_labels, _labels, stats, centroids = cv2.connectedComponentsWithStats(
        fg_clean, connectivity=8,
    )
    detections = []
    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area < MIN_BLOB_AREA or area > MAX_BLOB_AREA:
            continue
        cx, cy = centroids[label_id]
        x, y = int(round(cx)), int(round(cy))
        if 0 <= y < gray.shape[0] and 0 <= x < gray.shape[1]:
            peak_intensity = int(gray[max(0, y - 1):y + 2, max(0, x - 1):x + 2].max())
        else:
            peak_intensity = 0
        if peak_intensity < MIN_DETECTION_INTENSITY:
            continue
        detections.append({
            "x": x, "y": y,
            "area_px": int(area),
            "peak_intensity": peak_intensity,
        })
    return detections


class CpuMog2Backend:
    """Reference CPU OpenCV MOG2 (pointer_bench baseline)."""

    study_id = "cpu_mog2"
    study_title = "CPU MOG2 (reference)"
    uses_hw_encode = False

    def __init__(self):
        self.frame_count = 0
        self.backsub = cv2.createBackgroundSubtractorMOG2(
            history=MOG2_HISTORY,
            varThreshold=MOG2_VAR_THRESHOLD,
            detectShadows=MOG2_DETECT_SHADOWS,
        )
        self._kernel = np.ones((1, 1), np.uint8)

    def process(self, gray: np.ndarray) -> ProcessResult:
        self.frame_count += 1
        t0 = time.monotonic()
        fg_mask = self.backsub.apply(gray)
        fg_clean = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self._kernel)
        detections = _connected_components_detect(gray, fg_clean)
        mog2_ms = (time.monotonic() - t0) * 1000.0
        return ProcessResult(
            detections=detections,
            fg_mask=fg_mask,
            fg_clean=fg_clean,
            mog2_ms=mog2_ms,
            fg_pixels=int(np.count_nonzero(fg_mask)),
            clean_pixels=int(np.count_nonzero(fg_clean)),
        )


class OpenCvCudaBackend:
    """OpenCV CUDA MOG2 — mask round-trips to CPU for connected components."""

    study_id = "01_opencv_cuda"
    study_title = "OpenCV CUDA MOG2"
    uses_hw_encode = False

    def __init__(self):
        self.frame_count = 0
        if not hasattr(cv2, "cuda"):
            raise RuntimeError("cv2.cuda not available")
        if cv2.cuda.getCudaEnabledDeviceCount() <= 0:
            raise RuntimeError("No CUDA devices for OpenCV")
        self.backsub = cv2.cuda.createBackgroundSubtractorMOG2(
            history=MOG2_HISTORY,
            varThreshold=MOG2_VAR_THRESHOLD,
            detectShadows=MOG2_DETECT_SHADOWS,
        )
        self._gpu_frame = cv2.cuda_GpuMat()
        self._kernel = np.ones((1, 1), np.uint8)
        self._upload_ms = 0.0
        self._download_ms = 0.0

    def process(self, gray: np.ndarray) -> ProcessResult:
        self.frame_count += 1
        t0 = time.monotonic()
        t_up = time.monotonic()
        self._gpu_frame.upload(gray)
        upload_ms = (time.monotonic() - t_up) * 1000.0
        fg_gpu = self.backsub.apply(self._gpu_frame, learningRate=-1, stream=None)
        t_dn = time.monotonic()
        fg_mask = fg_gpu.download()
        download_ms = (time.monotonic() - t_dn) * 1000.0
        fg_clean = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self._kernel)
        detections = _connected_components_detect(gray, fg_clean)
        mog2_ms = (time.monotonic() - t0) * 1000.0
        return ProcessResult(
            detections=detections,
            fg_mask=fg_mask,
            fg_clean=fg_clean,
            mog2_ms=mog2_ms,
            fg_pixels=int(np.count_nonzero(fg_mask)),
            clean_pixels=int(np.count_nonzero(fg_clean)),
            extra={"gpu_upload_ms": upload_ms, "gpu_download_ms": download_ms},
        )


class GstFlowBackend:
    """GStreamer appsrc pipeline — optical flow in GStreamer, no MOG2."""

    study_id = "02_jetson_encode"
    study_title = "Jetson GStreamer optical flow (no MOG2)"
    uses_hw_encode = True
    uses_gst_pipeline = True
    warmup_frames = 2

    def __init__(self):
        self.frame_count = 0
        self._pipeline = None

    def bind_pipeline(self, pipeline) -> None:
        self._pipeline = pipeline

    def process(self, gray: np.ndarray) -> ProcessResult:
        if self._pipeline is None:
            raise RuntimeError("GstFlowBackend: pipeline not bound")
        self.frame_count += 1
        r = self._pipeline.process(gray)
        return ProcessResult(
            detections=r.detections,
            fg_mask=r.fg_mask,
            fg_clean=r.fg_clean,
            mog2_ms=r.process_ms,
            fg_pixels=r.fg_pixels,
            clean_pixels=r.clean_pixels,
            extra={
                "flow_vis": r.flow_vis,
                "flow_ms": r.flow_ms,
                "flow_mean_mag": r.flow_mean_mag,
                "flow_p95_mag": r.flow_p95_mag,
                "gst_pipeline": self._pipeline.name,
                "algorithm": "gst_optical_flow",
            },
        )


# Backward-compatible alias
JetsonEncodeBackend = GstFlowBackend


class VpiBackend(CpuMog2Backend):
    """CPU MOG2 — VPI studied for preprocessing fit (not MOG2 replacement)."""

    study_id = "03_vpi"
    study_title = "NVIDIA VPI (detection unchanged)"
    uses_hw_encode = False

    def __init__(self):
        super().__init__()
        self._vpi_info: dict = {}
        try:
            import vpi
            self._vpi_info = {
                "version": getattr(vpi, "__version__", "unknown"),
                "backends": [n for n in ("CUDA", "VIC", "OFA", "PVA", "CPU")
                             if hasattr(vpi.Backend, n)],
            }
        except ImportError as exc:
            self._vpi_info = {"error": str(exc)}


class CupyFrameDiffBackend:
    """CuPy GPU frame-differencing — same pipeline outputs as Study 02 (mask/clean/flow videos)."""

    study_id = "04_custom_cuda"
    study_title = "CuPy frame-diff"
    uses_hw_encode = True
    uses_gst_flow = True
    warmup_frames = 2

    FRAME_DIFF_THRESHOLD = 25

    def __init__(self):
        self.frame_count = 0
        self.flow_engine = None  # wired by run_study when camera resolution is known
        try:
            import cupy as cp
            self._cp = cp
        except ImportError as exc:
            raise RuntimeError("cupy not installed") from exc
        self._prev_gpu = None
        self._kernel = np.ones((1, 1), np.uint8)

    def reset(self) -> None:
        """Clear frame-diff state at chunk boundaries (matches GstFlowPipeline.reset)."""
        self._prev_gpu = None
        self.frame_count = 0

    def process(self, gray: np.ndarray) -> ProcessResult:
        self.frame_count += 1
        cp = self._cp
        t0 = time.monotonic()
        upload_ms = download_ms = 0.0

        t_up = time.monotonic()
        g_curr = cp.asarray(gray)
        upload_ms = (time.monotonic() - t_up) * 1000.0

        if self._prev_gpu is None:
            self._prev_gpu = g_curr.copy()
            fg_mask = np.zeros_like(gray)
            fg_clean = fg_mask.copy()
            detections: list[dict] = []
        else:
            diff = cp.abs(g_curr.astype(cp.int16) - self._prev_gpu.astype(cp.int16))
            fg_gpu = (diff > self.FRAME_DIFF_THRESHOLD).astype(cp.uint8) * 255
            t_dn = time.monotonic()
            fg_mask = cp.asnumpy(fg_gpu)
            download_ms = (time.monotonic() - t_dn) * 1000.0
            self._prev_gpu = g_curr.copy()
            fg_clean = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self._kernel)
            detections = _connected_components_detect(gray, fg_clean)

        mog2_ms = (time.monotonic() - t0) * 1000.0
        return ProcessResult(
            detections=detections,
            fg_mask=fg_mask,
            fg_clean=fg_clean,
            mog2_ms=mog2_ms,
            fg_pixels=int(np.count_nonzero(fg_mask)),
            clean_pixels=int(np.count_nonzero(fg_clean)),
            extra={
                "algorithm": "cupy_frame_diff",
                "gpu_upload_ms": upload_ms,
                "gpu_download_ms": download_ms,
            },
        )


class TensorRtBackend(CpuMog2Backend):
    """CPU MOG2 + TensorRT/DNN stack probe in session metadata."""

    study_id = "05_tensorrt"
    study_title = "TensorRT / DNN feasibility"
    uses_hw_encode = False

    def __init__(self):
        super().__init__()
        self._stack: dict = {}
        for mod in ("tensorrt", "onnxruntime", "torch"):
            try:
                __import__(mod)
                self._stack[mod] = True
            except ImportError:
                self._stack[mod] = False


class PvaBackend(CpuMog2Backend):
    """CPU MOG2 — PVA device presence recorded in metadata."""

    study_id = "06_pva"
    study_title = "Jetson PVA"
    uses_hw_encode = False

    def __init__(self):
        super().__init__()
        from pathlib import Path
        nodes = sorted(Path("/dev").glob("nvhost-pva*"))
        self._pva_nodes = [str(p) for p in nodes]


class OfaCpuFlowBackend:
    """VPI dense optical flow via OFA hardware — same outputs as Study 02, no MOG2.

    Requires VPI + OFA. Fails at startup if OFA is unavailable; never falls back
    to CPU Farneback or vpi.Backend.CPU.
    """

    study_id = "08_ofa_cpu"
    study_title = "VPI OFA optical flow (no MOG2)"
    uses_hw_encode = True
    warmup_frames = 2
    uses_flow_detection = True

    def __init__(self):
        import vpi
        from flow_engines import VpiOfaFlowEngine
        from gst_flow_pipeline import OFA_PROFILE, flow_magnitude_to_detection

        self.frame_count = 0
        self._prev_small: np.ndarray | None = None
        self._flow_w = 0
        self._flow_h = 0
        self._flow_profile = OFA_PROFILE
        self._flow_magnitude_to_detection: Callable = (
            lambda flow, gray, fw, fh: flow_magnitude_to_detection(
                flow, gray, fw, fh, profile=OFA_PROFILE,
            )
        )
        self._engine = VpiOfaFlowEngine()
        self._vpi_info = {
            "version": getattr(vpi, "__version__", "unknown"),
            "flow_backend": "OFA",
            "pyramid_backend": "CUDA",
            "convert_backend": "VIC",
            "algorithm": "vpi_ofa_hw",
            "fallback_allowed": False,
            "flow_detect_profile": {
                "name": OFA_PROFILE.name,
                "mag_floor": OFA_PROFILE.mag_floor,
                "p95_factor": OFA_PROFILE.p95_factor,
                "max_mask_frac": OFA_PROFILE.max_mask_frac,
                "escalate_percentile": OFA_PROFILE.escalate_percentile,
            },
        }
        print(
            "VPI OFA probe OK — pyramid=CUDA, convert=VIC, "
            f"flow=OFA (vpi {self._vpi_info['version']})  "
            f"detect profile: floor={OFA_PROFILE.mag_floor} "
            f"p95×{OFA_PROFILE.p95_factor} escalate>{OFA_PROFILE.max_mask_frac:.0%}"
        )

    def reset(self) -> None:
        self._prev_small = None
        self.frame_count = 0
        self._engine.reset()

    def _flow_size(self, gray: np.ndarray) -> tuple[int, int]:
        from flow_engines import FLOW_DEBUG_DOWNSAMPLE
        fh, fw = gray.shape[:2]
        self._flow_w = max(1, int(fw * FLOW_DEBUG_DOWNSAMPLE))
        self._flow_h = max(1, int(fh * FLOW_DEBUG_DOWNSAMPLE))
        return self._flow_w, self._flow_h

    def process(self, gray: np.ndarray) -> ProcessResult:
        from flow_engines import FLOW_DEBUG_DOWNSAMPLE

        self.frame_count += 1
        t0 = time.monotonic()
        flow_w, flow_h = self._flow_size(gray)
        curr_small = cv2.resize(
            gray, (flow_w, flow_h), interpolation=cv2.INTER_AREA,
        )

        flow_ms = 0.0
        flow_mean_mag = 0.0
        flow_p95_mag = 0.0
        flow_vis = cv2.cvtColor(curr_small, cv2.COLOR_GRAY2BGR)
        fg_mask = np.zeros_like(gray)
        fg_clean = fg_mask.copy()
        detections: list[dict] = []

        if self._prev_small is not None:
            flow, flow_ms = self._engine.compute_flow(self._prev_small, curr_small)
            fg_mask, fg_clean, detections, flow_mean_mag, flow_p95_mag, flow_vis = (
                self._flow_magnitude_to_detection(flow, gray, flow_w, flow_h)
            )

        self._prev_small = curr_small
        process_ms = (time.monotonic() - t0) * 1000.0
        return ProcessResult(
            detections=detections,
            fg_mask=fg_mask,
            fg_clean=fg_clean,
            mog2_ms=process_ms,
            fg_pixels=int(np.count_nonzero(fg_mask)),
            clean_pixels=int(np.count_nonzero(fg_clean)),
            extra={
                "flow_vis": flow_vis,
                "flow_ms": flow_ms,
                "flow_mean_mag": flow_mean_mag,
                "flow_p95_mag": flow_p95_mag,
                "algorithm": "vpi_ofa_hw",
                "flow_backend": "OFA",
                "flow_downsample": FLOW_DEBUG_DOWNSAMPLE,
            },
        )


class ZeroCopyBackend(OpenCvCudaBackend):
    """OpenCV CUDA MOG2 with explicit transfer timing in metrics."""

    study_id = "07_zero_copy"
    study_title = "Zero-copy / GPU transfer"
    uses_hw_encode = False
