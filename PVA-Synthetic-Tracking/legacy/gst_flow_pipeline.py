"""Live GStreamer pipeline: camera frames via appsrc → NVIDIA optical flow (no MOG2)."""

from __future__ import annotations

import queue
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

from flow_engines import (
    FLOW_DEBUG_DOWNSAMPLE,
    FLOW_DEBUG_MIN_MAG,
    FLOW_DEBUG_SCALE,
    FLOW_DEBUG_STEP,
    render_flow_arrows,
)

try:
    import gi
    gi.require_version("Gst", "1.0")
    from gi.repository import Gst, GLib
    HAS_GST = True
except (ImportError, ValueError):
    HAS_GST = False

from nvof_meta import extract_flow_from_buffer

# nvof block vectors are fixed-point; scale to pixel displacement.
NVOF_FLOW_SCALE = 1.0 / 32.0
NVOF_MAG_THRESHOLD = 0.35
FRAME_DIFF_THRESHOLD = 25
MIN_BLOB_AREA = 1
MAX_BLOB_AREA = 50
MIN_DETECTION_INTENSITY = 20


@dataclass(frozen=True)
class FlowMagProfile:
    """Per-engine flow-magnitude mask thresholds (magnitude scales differ by backend)."""

    name: str
    mag_floor: float          # minimum |flow| to count as motion (px)
    p95_factor: float         # adaptive: thresh >= p95_mag * factor
    max_mask_frac: float      # if more than this fraction passes, escalate percentile
    escalate_percentile: float  # keep only top (100-p)% motion pixels


NVOF_PROFILE = FlowMagProfile("nvof", 0.35, 0.25, 0.50, 99.5)
# OFA baseline noise ~5 px; p95 tail ~15 px — floor must sit above noise band.
OFA_PROFILE = FlowMagProfile("ofa", 12.0, 0.85, 0.05, 99.9)


def _detect_limits() -> tuple[int, int, int]:
    """Use CLI-tuned limits from backends when available."""
    try:
        import backends as be
        return be.MIN_BLOB_AREA, be.MAX_BLOB_AREA, be.MIN_DETECTION_INTENSITY
    except ImportError:
        return MIN_BLOB_AREA, MAX_BLOB_AREA, MIN_DETECTION_INTENSITY


@dataclass
class GstFrameResult:
    detections: list[dict]
    fg_mask: np.ndarray
    fg_clean: np.ndarray
    flow_vis: np.ndarray | None
    process_ms: float
    flow_ms: float
    flow_mean_mag: float
    flow_p95_mag: float
    fg_pixels: int
    clean_pixels: int


def _nvof_plugins_available() -> bool:
    if not HAS_GST:
        return False
    for plugin in ("nvof", "nvofvisual", "nvstreammux"):
        r = subprocess.run(
            ["gst-inspect-1.0", plugin],
            capture_output=True, timeout=5,
        )
        if r.returncode != 0:
            return False
    return True


def _connected_components_detect(
    gray: np.ndarray, fg_clean: np.ndarray,
) -> list[dict]:
    min_area, max_area, min_intensity = _detect_limits()
    num_labels, _labels, stats, centroids = cv2.connectedComponentsWithStats(
        fg_clean, connectivity=8,
    )
    detections = []
    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area < min_area or area > max_area:
            continue
        cx, cy = centroids[label_id]
        x, y = int(round(cx)), int(round(cy))
        if 0 <= y < gray.shape[0] and 0 <= x < gray.shape[1]:
            peak_intensity = int(gray[max(0, y - 1):y + 2, max(0, x - 1):x + 2].max())
        else:
            peak_intensity = 0
        if peak_intensity < min_intensity:
            continue
        detections.append({
            "x": x, "y": y,
            "area_px": int(area),
            "peak_intensity": peak_intensity,
        })
    return detections


def flow_magnitude_to_detection(
    flow: np.ndarray,
    full_gray: np.ndarray,
    flow_w: int,
    flow_h: int,
    profile: FlowMagProfile = NVOF_PROFILE,
) -> tuple[np.ndarray, np.ndarray, list[dict], float, float, np.ndarray]:
    """Convert dense flow vectors to masks, detections, and arrow-ready flow field."""
    fh, fw = full_gray.shape[:2]
    flow_up = cv2.resize(flow, (flow_w, flow_h), interpolation=cv2.INTER_LINEAR)
    flow_up[..., 0] *= flow_w / max(flow.shape[1], 1)
    flow_up[..., 1] *= flow_h / max(flow.shape[0], 1)

    mag = np.hypot(flow_up[..., 0], flow_up[..., 1])
    flow_mean_mag = float(mag.mean())
    flow_p95_mag = float(np.percentile(mag, 95))

    small_gray = cv2.resize(full_gray, (flow_w, flow_h), interpolation=cv2.INTER_AREA)
    thresh = max(profile.mag_floor, flow_p95_mag * profile.p95_factor)
    small_mask = (mag >= thresh).astype(np.uint8) * 255
    mask_frac = int(np.count_nonzero(small_mask)) / max(flow_w * flow_h, 1)
    if mask_frac > profile.max_mask_frac:
        thresh = float(np.percentile(mag, profile.escalate_percentile))
        small_mask = (mag >= thresh).astype(np.uint8) * 255

    kernel = np.ones((1, 1), np.uint8)
    small_clean = cv2.morphologyEx(small_mask, cv2.MORPH_OPEN, kernel)
    fg_mask = cv2.resize(small_mask, (fw, fh), interpolation=cv2.INTER_NEAREST)
    fg_clean = cv2.resize(small_clean, (fw, fh), interpolation=cv2.INTER_NEAREST)
    detections = _connected_components_detect(full_gray, fg_clean)
    flow_vis = render_flow_arrows(small_gray, flow_up)
    return fg_mask, fg_clean, detections, flow_mean_mag, flow_p95_mag, flow_vis


def _frame_diff_masks(
    prev_small: np.ndarray, curr_small: np.ndarray, full_gray: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    diff = cv2.absdiff(prev_small, curr_small)
    _, small_mask = cv2.threshold(diff, FRAME_DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
    kernel = np.ones((1, 1), np.uint8)
    small_clean = cv2.morphologyEx(small_mask, cv2.MORPH_OPEN, kernel)
    fh, fw = full_gray.shape[:2]
    sh, sw = small_clean.shape[:2]
    scale_x = fw / sw
    scale_y = fh / sh
    fg_mask = cv2.resize(small_mask, (fw, fh), interpolation=cv2.INTER_NEAREST)
    fg_clean = cv2.resize(small_clean, (fw, fh), interpolation=cv2.INTER_NEAREST)
    detections = _connected_components_detect(full_gray, fg_clean)
    if not detections and int(np.count_nonzero(small_clean)) > 0:
        detections = _local_maxima_detect(full_gray, small_clean, scale_x, scale_y)
    return fg_mask, fg_clean, detections


def _full_frame_diff_masks(
    prev_gray: np.ndarray, curr_gray: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    diff = cv2.absdiff(prev_gray, curr_gray)
    _, fg_mask = cv2.threshold(diff, FRAME_DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
    kernel = np.ones((1, 1), np.uint8)
    fg_clean = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    detections = _connected_components_detect(curr_gray, fg_clean)
    return fg_mask, fg_clean, detections


def _local_maxima_detect(
    full_gray: np.ndarray,
    small_clean: np.ndarray,
    scale_x: float,
    scale_y: float,
) -> list[dict]:
    """Find point sources from a small motion mask when CC blobs are too large."""
    dilated = cv2.dilate(small_clean, np.ones((3, 3), np.uint8))
    local_max = cv2.erode(dilated, np.ones((3, 3), np.uint8))
    peaks = (local_max == small_clean) & (small_clean > 0)
    ys, xs = np.where(peaks)
    detections = []
    for x, y in zip(xs, ys):
        fx = int(round(x * scale_x))
        fy = int(round(y * scale_y))
        if not (0 <= fy < full_gray.shape[0] and 0 <= fx < full_gray.shape[1]):
            continue
        peak_intensity = int(full_gray[max(0, fy - 1):fy + 2, max(0, fx - 1):fx + 2].max())
        if peak_intensity < MIN_DETECTION_INTENSITY:
            continue
        detections.append({
            "x": fx, "y": fy,
            "area_px": 1,
            "peak_intensity": peak_intensity,
        })
    return detections


class GstFlowPipeline:
    """Push GRAY8 frames into GStreamer; optical flow runs inside the pipeline."""

    def __init__(self, width: int, height: int, fps: float):
        if not HAS_GST:
            raise RuntimeError("python3-gi and GStreamer 1.0 required for Study 02")
        self.width = width
        self.height = height
        self.fps = max(1.0, fps)
        self.flow_w = max(1, int(width * FLOW_DEBUG_DOWNSAMPLE))
        self.flow_h = max(1, int(height * FLOW_DEBUG_DOWNSAMPLE))
        self.name = "gstreamer_nvof" if _nvof_plugins_available() else "gstreamer_appsrc_flow"
        self._frame_count = 0
        self._pts = 0
        self._duration = int(Gst.SECOND / self.fps)
        self._pending_gray: np.ndarray | None = None
        self._result_q: queue.Queue[GstFrameResult | Exception] = queue.Queue(maxsize=8)
        self._prev_small: np.ndarray | None = None
        self._prev_full_gray: np.ndarray | None = None
        self._last_nvof_flow: np.ndarray | None = None
        self._lock = threading.Lock()
        self._appsrc = None
        self._pipeline = None
        if self.name == "gstreamer_nvof":
            self._build_nvof_pipeline()
        else:
            self._build_appsrc_flow_pipeline()
        self._start()

    def _start(self):
        bus = self._pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message)
        ret = self._pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("GStreamer pipeline failed to enter PLAYING")

    def _on_bus_message(self, bus, message):
        t = message.type
        if t == Gst.MessageType.ERROR:
            err, _dbg = message.parse_error()
            self._result_q.put(RuntimeError(str(err)))

    def _nvof_probe(self, pad, info, _user_data):
        buf = info.get_buffer()
        if buf is None:
            return Gst.PadProbeReturn.OK
        flow = extract_flow_from_buffer(buf)
        if flow is not None:
            flow = flow.astype(np.float32) * NVOF_FLOW_SCALE
            with self._lock:
                self._last_nvof_flow = flow
        return Gst.PadProbeReturn.OK

    def _build_appsrc_flow_pipeline(self):
        Gst.init(None)
        pipeline = Gst.Pipeline.new("study02-flow")
        appsrc = Gst.ElementFactory.make("appsrc", "src")
        convert = Gst.ElementFactory.make("videoconvert", "convert")
        scale = Gst.ElementFactory.make("videoscale", "scale")
        capsfilter = Gst.ElementFactory.make("capsfilter", "flow_caps")
        flowstep = Gst.ElementFactory.make("identity", "flowstep")
        sink = Gst.ElementFactory.make("appsink", "sink")
        if not all((appsrc, convert, scale, capsfilter, flowstep, sink)):
            raise RuntimeError("Failed to create GStreamer elements for flow pipeline")

        in_caps = Gst.Caps.from_string(
            f"video/x-raw,format=GRAY8,width={self.width},height={self.height},"
            f"framerate={int(round(self.fps))}/1",
        )
        flow_caps = Gst.Caps.from_string(
            f"video/x-raw,format=GRAY8,width={self.flow_w},height={self.flow_h}",
        )
        appsrc.set_property("is-live", True)
        appsrc.set_property("do-timestamp", True)
        appsrc.set_property("format", Gst.Format.TIME)
        appsrc.set_property("caps", in_caps)
        appsrc.set_property("block", True)
        capsfilter.set_property("caps", flow_caps)
        sink.set_property("emit-signals", True)
        sink.set_property("sync", False)
        sink.set_property("max-buffers", 2)
        sink.set_property("drop", True)
        sink.connect("new-sample", self._on_flow_sample)

        for el in (appsrc, convert, scale, capsfilter, flowstep, sink):
            pipeline.add(el)
        appsrc.link(convert)
        convert.link(scale)
        scale.link(capsfilter)
        capsfilter.link(flowstep)
        flowstep.link(sink)
        self._pipeline = pipeline
        self._appsrc = appsrc

    def _build_nvof_pipeline(self):
        Gst.init(None)
        fps_i = max(1, int(round(self.fps)))
        desc = (
            f"appsrc name=src is-live=true block=true format=3 do-timestamp=true "
            f"caps=video/x-raw,format=GRAY8,width={self.width},height={self.height},"
            f"framerate={fps_i}/1 ! "
            f"videoconvert ! nvvideoconvert ! video/x-raw(memory:NVMM),format=NV12 ! "
            f"mux.sink_0 nvstreammux name=mux width={self.width} height={self.height} "
            f"batch-size=1 live-source=1 batched-push-timeout=40000 ! "
            f"queue ! nvof name=nvof ! queue ! nvofvisual ! queue ! "
            f"nvvideoconvert ! videoconvert ! video/x-raw,format=BGR ! "
            f"appsink name=sink emit-signals=true sync=false max-buffers=2 drop=true"
        )
        pipeline = Gst.parse_launch(desc)
        appsrc = pipeline.get_by_name("src")
        sink = pipeline.get_by_name("sink")
        nvof = pipeline.get_by_name("nvof")
        if appsrc is None or sink is None or nvof is None:
            raise RuntimeError("Failed to build nvof pipeline from launch string")
        nvof.get_static_pad("src").add_probe(
            Gst.PadProbeType.BUFFER, self._nvof_probe, None,
        )
        sink.connect("new-sample", self._on_nvof_visual_sample)
        self._pipeline = pipeline
        self._appsrc = appsrc

    def _on_flow_sample(self, sink):
        sample = sink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.ERROR
        buf = sample.get_buffer()
        caps = sample.get_caps().get_structure(0)
        w, h = caps.get_value("width"), caps.get_value("height")
        success, map_info = buf.map(Gst.MapFlags.READ)
        if not success:
            return Gst.FlowReturn.ERROR
        try:
            curr_small = np.frombuffer(map_info.data, dtype=np.uint8).reshape(h, w).copy()
        finally:
            buf.unmap(map_info)

        with self._lock:
            full_gray = self._pending_gray
            self._pending_gray = None

        if full_gray is None:
            return Gst.FlowReturn.OK

        t_flow = time.monotonic()
        flow_ms = 0.0
        flow_mean_mag = 0.0
        flow_p95_mag = 0.0
        flow_vis = cv2.cvtColor(curr_small, cv2.COLOR_GRAY2BGR)
        fg_mask = np.zeros_like(full_gray)
        fg_clean = fg_mask.copy()
        detections: list[dict] = []

        if self._prev_small is not None:
            flow = cv2.calcOpticalFlowFarneback(
                self._prev_small, curr_small, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
            )
            flow_ms = (time.monotonic() - t_flow) * 1000.0
            mag = np.hypot(flow[..., 0], flow[..., 1])
            flow_mean_mag = float(mag.mean())
            flow_p95_mag = float(np.percentile(mag, 95))
            flow_vis = render_flow_arrows(curr_small, flow)
            fg_mask, fg_clean, detections = _frame_diff_masks(
                self._prev_small, curr_small, full_gray,
            )
        self._prev_small = curr_small
        self._prev_full_gray = full_gray.copy()
        self._emit_result(
            detections, fg_mask, fg_clean, flow_vis,
            flow_ms, flow_mean_mag, flow_p95_mag,
        )
        return Gst.FlowReturn.OK

    def _on_nvof_visual_sample(self, sink):
        sample = sink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.ERROR
        buf = sample.get_buffer()

        with self._lock:
            full_gray = self._pending_gray
            self._pending_gray = None
            flow_grid = self._last_nvof_flow
            self._last_nvof_flow = None

        if full_gray is None:
            return Gst.FlowReturn.OK

        t_flow = time.monotonic()
        curr_small = cv2.resize(
            full_gray, (self.flow_w, self.flow_h), interpolation=cv2.INTER_AREA,
        )
        flow_ms = (time.monotonic() - t_flow) * 1000.0
        flow_mean_mag = 0.0
        flow_p95_mag = 0.0
        flow_vis = cv2.cvtColor(curr_small, cv2.COLOR_GRAY2BGR)
        fg_mask = np.zeros_like(full_gray)
        fg_clean = fg_mask.copy()
        detections: list[dict] = []

        if flow_grid is not None:
            _, _, _, flow_mean_mag, flow_p95_mag, flow_vis = (
                flow_magnitude_to_detection(flow_grid, full_gray, self.flow_w, self.flow_h)
            )
            if self._prev_full_gray is not None:
                fg_mask, fg_clean, detections = _full_frame_diff_masks(
                    self._prev_full_gray, full_gray,
                )
            else:
                fg_mask = np.zeros_like(full_gray)
                fg_clean = fg_mask.copy()
                detections = []
        elif self._prev_small is not None:
            flow = cv2.calcOpticalFlowFarneback(
                self._prev_small, curr_small, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
            )
            flow_vis = render_flow_arrows(curr_small, flow)
            mag = np.hypot(flow[..., 0], flow[..., 1])
            flow_mean_mag = float(mag.mean())
            flow_p95_mag = float(np.percentile(mag, 95))
            fg_mask, fg_clean, detections = _frame_diff_masks(
                self._prev_small, curr_small, full_gray,
            )
        self._prev_small = curr_small
        self._prev_full_gray = full_gray.copy()
        self._emit_result(
            detections, fg_mask, fg_clean, flow_vis,
            flow_ms, flow_mean_mag, flow_p95_mag,
        )
        return Gst.FlowReturn.OK

    def _emit_result(
        self,
        detections: list[dict],
        fg_mask: np.ndarray,
        fg_clean: np.ndarray,
        flow_vis: np.ndarray,
        flow_ms: float,
        flow_mean_mag: float,
        flow_p95_mag: float,
    ):
        result = GstFrameResult(
            detections=detections,
            fg_mask=fg_mask,
            fg_clean=fg_clean,
            flow_vis=flow_vis,
            process_ms=0.0,
            flow_ms=flow_ms,
            flow_mean_mag=flow_mean_mag,
            flow_p95_mag=flow_p95_mag,
            fg_pixels=int(np.count_nonzero(fg_mask)),
            clean_pixels=int(np.count_nonzero(fg_clean)),
        )
        try:
            self._result_q.put_nowait(result)
        except queue.Full:
            try:
                self._result_q.get_nowait()
            except queue.Empty:
                pass
            self._result_q.put_nowait(result)

    def process(self, gray: np.ndarray) -> GstFrameResult:
        if self._appsrc is None:
            raise RuntimeError("GStreamer pipeline not initialized")
        t0 = time.monotonic()
        with self._lock:
            self._pending_gray = gray

        buf = Gst.Buffer.new_wrapped(gray.tobytes())
        buf.pts = self._pts
        buf.duration = self._duration
        self._pts += self._duration

        ret = self._appsrc.emit("push-buffer", buf)
        if ret != Gst.FlowReturn.OK and ret != Gst.FlowReturn.FLUSHING:
            raise RuntimeError(f"appsrc push-buffer failed: {ret}")

        try:
            item = self._result_q.get(timeout=15.0)
        except queue.Empty as exc:
            raise RuntimeError("GStreamer flow pipeline timed out waiting for output") from exc
        if isinstance(item, Exception):
            raise item
        item.process_ms = (time.monotonic() - t0) * 1000.0
        self._frame_count += 1
        return item

    def reset(self):
        self._prev_small = None
        self._prev_full_gray = None
        self._last_nvof_flow = None
        self._frame_count = 0
        self._pts = 0
        while not self._result_q.empty():
            try:
                self._result_q.get_nowait()
            except queue.Empty:
                break

    def close(self):
        if self._pipeline is None:
            return
        if self._appsrc is not None:
            self._appsrc.emit("end-of-stream")
        self._pipeline.set_state(Gst.State.NULL)
        self._pipeline = None
        self._appsrc = None
