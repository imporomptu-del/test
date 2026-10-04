#!/usr/bin/env python3
"""
Shared live-camera pipeline for GPU study scripts.

Same options as pointer_bench.py plus mog2 mask/clean debug videos (pointer_report).
Each study script supplies a DetectorBackend from backends.py.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import json
import os
import queue
import socket
import statistics
import sys
import threading
import time
from datetime import datetime, timezone
from http import server as http_server
from socketserver import ThreadingMixIn
from typing import TYPE_CHECKING, Callable

import cv2
import numpy as np

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

TOUPCAM_PATH = "/usr/local/ToupLite"
sys.path.insert(0, TOUPCAM_PATH)
try:
    import toupcam
except ImportError as exc:
    raise SystemExit(
        f"toupcam SDK not found under {TOUPCAM_PATH}. "
        "Install ToupLite for SkyEye62AM."
    ) from exc

from backends import MOG2_HISTORY, ProcessResult

if TYPE_CHECKING:
    from backends import CpuMog2Backend

# ══════════════════════════════════════════════════════════════════════════════
# Defaults (overridden by CLI / run_study)
# ══════════════════════════════════════════════════════════════════════════════

# SkyEye62AM settings — match save_video_of_two.py
RESOLUTION_INDEX = 2          # 3184×2124
EXPOSURE_US = 100_000         # 0.1 s → ~10 fps max
REQUESTED_GAIN = 100
TARGET_FPS = 10.0

CHUNK_SECONDS = 60
MAX_CHUNKS = 0
RESULTS_ROOT = "/home/a/projects/SkyFortress/skymove/results"
GPU_STUDY_SUBDIR = "gpu_study"
OUTPUT_DIR = os.path.join(RESULTS_ROOT, GPU_STUDY_SUBDIR)
DET_SUBDIR = "detections"
METRICS_SUBDIR = "metrics"
REPORT_SUBDIR = "reports"
FILE_PREFIX = "chunk"

MOG2_VAR_THRESHOLD = 16
MIN_BLOB_AREA = 1
MAX_BLOB_AREA = 500
MIN_DETECTION_INTENSITY = 20
HARDWARE_SAMPLE_SEC = 1.0

OUTPUT_MODE = "avi"
STREAM_SCALE = 0.25
STREAM_MODE = "local"
STREAM_PORT = 8765
STREAM_WINDOW = "gpu_study"

ANNOTATE_VIDEO = True
SAVE_FLOW_DEBUG_VIDEO = True
SAVE_MOG2_MASK_VIDEO = True
SAVE_MOG2_CLEAN_VIDEO = True
SAVE_DETECTIONS_VIDEO = True

FLOW_DEBUG_DOWNSAMPLE = 0.25
FLOW_DEBUG_STEP = 16
FLOW_DEBUG_SCALE = 4.0
FLOW_DEBUG_MIN_MAG = 0.2

STUDY_ID = "gpu_study"
STUDY_TITLE = "GPU Study"
USE_HW_ENCODE = False

POINTER_BENCH_REF = {
    "worker_frame_ms_mean_avi": 320.0,
    "worker_frame_ms_mean_novideo": 65.0,
    "mog2_ms_mean": 55.0,
    "video_write_ms_mean": 250.0,
    "frames_per_60s_chunk_novideo": 600.0,
}

frame_q: queue.Queue = queue.Queue(maxsize=32)
_FLUSH = object()
_STOP = object()
detect_q: queue.Queue = queue.Queue(maxsize=64)
session_chunk_summaries: list[dict] = []


def allocate_run_output_dir(study_slug: str) -> str:
    """Create a unique run folder under results/gpu_study/<study_slug>/."""
    base = os.path.join(RESULTS_ROOT, GPU_STUDY_SUBDIR, study_slug)
    os.makedirs(base, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base, f"run_{stamp}")
    suffix = 1
    while os.path.exists(run_dir):
        run_dir = os.path.join(base, f"run_{stamp}_{suffix:02d}")
        suffix += 1
    os.makedirs(run_dir, exist_ok=False)
    return run_dir


def local_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"


class MjpegHttpServer(threading.Thread):
    def __init__(self, port: int = STREAM_PORT):
        super().__init__(daemon=True, name="MjpegHttpServer")
        self.port = port
        self._lock = threading.Lock()
        self._latest_jpeg: bytes | None = None
        self._httpd = None

    def update_frame(self, bgr: np.ndarray):
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            return
        with self._lock:
            self._latest_jpeg = buf.tobytes()

    def _get_jpeg(self) -> bytes | None:
        with self._lock:
            return self._latest_jpeg

    def run(self):
        outer = self

        class Handler(http_server.BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                pass

            def do_GET(self):
                if self.path in ("/", "/index.html"):
                    html = (
                        "<html><body style='margin:0;background:#111'>"
                        "<img src='/stream' style='max-width:100%'>"
                        "</body></html>"
                    ).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.send_header("Content-Length", str(len(html)))
                    self.end_headers()
                    self.wfile.write(html)
                    return
                if self.path != "/stream":
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                while True:
                    jpeg = outer._get_jpeg()
                    if jpeg is None:
                        time.sleep(0.05)
                        continue
                    try:
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                        self.wfile.write(jpeg)
                        self.wfile.write(b"\r\n")
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        break
                    time.sleep(0.033)

        class ThreadedServer(ThreadingMixIn, http_server.HTTPServer):
            daemon_threads = True

        self._httpd = ThreadedServer(("0.0.0.0", self.port), Handler)
        self._httpd.serve_forever()

    def stop(self):
        if self._httpd is not None:
            self._httpd.shutdown()


def _camera_callback(event, ctx: "CaptureThread"):
    if event == toupcam.TOUPCAM_EVENT_IMAGE:
        ctx.on_image()


class CaptureThread(threading.Thread):
    """Pull-mode SkyEye62AM capture — same model as save_video_of_two.py."""

    def __init__(self, hcam, width: int, height: int):
        super().__init__(daemon=True, name="CaptureThread")
        self.hcam = hcam
        self.width = width
        self.height = height
        self.frame = np.zeros((height, width), dtype=np.uint8)
        self._stop_event = threading.Event()
        self.frames_captured = 0
        self.frame_drops = 0

    def stop(self):
        self._stop_event.set()

    def on_image(self):
        row_pitch = self.width
        self.hcam.PullImageV3(
            ctypes.c_char_p(self.frame.ctypes.data),
            0, 8, row_pitch, None,
        )
        gray = self.frame.copy()
        self.frames_captured += 1
        if frame_q.full():
            try:
                frame_q.get_nowait()
                self.frame_drops += 1
            except queue.Empty:
                pass
        frame_q.put_nowait(gray)

    def run(self):
        self.hcam.StartPullModeWithCallback(_camera_callback, self)
        while not self._stop_event.wait(0.1):
            pass


def find_camera_index(camera_id: int | None, name_hint: str, cameras) -> int:
    if not cameras:
        raise RuntimeError("No ToupTek / SkyEye camera detected")
    print(f"Detected {len(cameras)} SkyEye camera(s):")
    for idx, cam in enumerate(cameras):
        print(f"  [{idx}] {cam.displayname}  id={cam.id}  model={cam.model.name}")
    if camera_id is not None:
        if camera_id < 0 or camera_id >= len(cameras):
            raise RuntimeError(f"Invalid --camera-id {camera_id}")
        return camera_id
    for idx, cam in enumerate(cameras):
        haystack = f"{cam.displayname} {cam.model.name}".lower()
        if name_hint.lower() in haystack:
            return idx
    return 0


def open_skyeye_camera(
    camera_id: int | None,
    camera_name: str,
    resolution_index: int,
    requested_gain: int,
) -> tuple[object, int, int, int, object]:
    cameras = toupcam.Toupcam.EnumV2()
    cam_idx = find_camera_index(camera_id, camera_name, cameras)
    cam_info = cameras[cam_idx]
    hcam = toupcam.Toupcam.Open(cam_info.id)
    if hcam is None:
        raise RuntimeError("Failed to open SkyEye camera")

    hcam.put_eSize(resolution_index)
    width, height = hcam.get_Size()
    hcam.put_AutoExpoEnable(1)
    try:
        gain_min, gain_max, _ = hcam.get_ExpoAGainRange()
        applied_gain = max(gain_min, min(requested_gain, gain_max))
    except Exception as exc:
        print(f"Could not read gain range: {exc} — using {requested_gain}")
        applied_gain = requested_gain

    print(f"Exposure target: {EXPOSURE_US / 1_000_000:.3f} s  "
          f"({1_000_000 / EXPOSURE_US:.1f} fps max)")
    print(f"Gain            : {applied_gain}")
    return hcam, width, height, cam_idx, cam_info


def measure_fps(capture: CaptureThread, seconds: float = 3.0) -> float:
    start_count = capture.frames_captured
    time.sleep(seconds)
    frames = capture.frames_captured - start_count
    measured = frames / seconds if seconds > 0 else 0.0
    if measured <= 0:
        return TARGET_FPS
    return max(measured, 1.0)


def open_writer(path: str, width: int, height: int, fps: float,
                color: bool = True, use_hw: bool = False) -> tuple[cv2.VideoWriter, str]:
    if use_hw and "GStreamer" in cv2.getBuildInformation():
        if color:
            pipeline = (
                f"appsrc ! videoconvert ! video/x-raw,format=I420 ! "
                f"nvvidconv ! video/x-raw(memory:NVMM),format=NV12 ! "
                f"nvjpegenc ! jpegparse ! avimux ! filesink location={path}"
            )
        else:
            pipeline = (
                f"appsrc ! videoconvert ! video/x-raw,format=GRAY8 ! "
                f"nvjpegenc ! jpegparse ! avimux ! filesink location={path}"
            )
        writer = cv2.VideoWriter(pipeline, cv2.CAP_GSTREAMER, 0, fps, (width, height), color)
        if writer.isOpened():
            return writer, "gstreamer_nvjpeg"
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height), isColor=color)
    if not writer.isOpened():
        raise RuntimeError(f"VideoWriter could not open '{path}'.")
    return writer, "software_mjpeg"


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    return float(np.percentile(np.asarray(values, dtype=np.float64), pct))


def summarize(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "count": len(values),
        "mean": float(statistics.mean(values)),
        "median": float(statistics.median(values)),
        "p95": percentile(values, 95),
        "max": float(max(values)),
    }


class HardwareSampler(threading.Thread):
    def __init__(self, interval_sec: float = HARDWARE_SAMPLE_SEC):
        super().__init__(daemon=True, name="HardwareSampler")
        self.interval_sec = interval_sec
        self._stop_event = threading.Event()
        self.samples: list[dict] = []

    def stop(self):
        self._stop_event.set()

    def _read_sysfs_text(self, path: str) -> str | None:
        try:
            with open(path, "rb") as f:
                data = f.read()
            return data.decode("utf-8", errors="ignore").strip() if data else None
        except OSError:
            return None

    def _read_thermal_c(self) -> dict[str, float]:
        temps: dict[str, float] = {}
        base = "/sys/class/thermal"
        if not os.path.isdir(base):
            return temps
        for name in sorted(os.listdir(base)):
            if not name.startswith("thermal_zone"):
                continue
            try:
                raw = self._read_sysfs_text(os.path.join(base, name, "temp"))
                if not raw:
                    continue
                label = self._read_sysfs_text(os.path.join(base, name, "type")) or name
                temps[label] = int(raw) / 1000.0
            except (OSError, ValueError, TypeError):
                continue
        return temps

    def _sample_once(self) -> dict:
        sample: dict = {"timestamp": time.time()}
        if HAS_PSUTIL:
            sample["cpu_percent"] = psutil.cpu_percent(interval=None)
            vm = psutil.virtual_memory()
            sample["ram_used_mb"] = vm.used / (1024 * 1024)
            sample["ram_percent"] = vm.percent
        else:
            sample["cpu_percent"] = sample["ram_used_mb"] = sample["ram_percent"] = None
        try:
            l1, l5, l15 = os.getloadavg()
            sample["load_1"], sample["load_5"], sample["load_15"] = l1, l5, l15
        except OSError:
            sample["load_1"] = sample["load_5"] = sample["load_15"] = None
        sample["thermal_c"] = self._read_thermal_c()
        return sample

    def run(self):
        try:
            self.samples.append(self._sample_once())
        except Exception as exc:
            print(f"\n  [HardwareSampler] initial sample error: {exc}")
        while not self._stop_event.wait(self.interval_sec):
            try:
                self.samples.append(self._sample_once())
            except Exception as exc:
                print(f"\n  [HardwareSampler] sample error: {exc}")

    def summary(self) -> dict:
        if not self.samples:
            return {}
        cpu_vals = [s["cpu_percent"] for s in self.samples if s.get("cpu_percent") is not None]
        ram_vals = [s["ram_percent"] for s in self.samples if s.get("ram_percent") is not None]
        load_vals = [s["load_1"] for s in self.samples if s.get("load_1") is not None]
        temp_keys: set[str] = set()
        for s in self.samples:
            temp_keys.update(s.get("thermal_c", {}).keys())
        thermal_summary = {
            key: summarize([s["thermal_c"][key] for s in self.samples if key in s.get("thermal_c", {})])
            for key in sorted(temp_keys)
        }
        return {
            "sample_count": len(self.samples),
            "cpu_percent": summarize(cpu_vals),
            "ram_percent": summarize(ram_vals),
            "load_1": summarize(load_vals),
            "thermal_c": thermal_summary,
        }


def render_detections_frame(
    width: int, height: int, detections: list[dict],
    frame_idx: int, is_warming_up: bool, study_label: str = "",
) -> np.ndarray:
    vis = np.zeros((height, width, 3), dtype=np.uint8)
    if not is_warming_up:
        for det in detections:
            x, y = det["x"], det["y"]
            cv2.drawMarker(vis, (x, y), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=12, thickness=1)
            cv2.circle(vis, (x, y), 4, (0, 255, 255), 1)
    status = "MOG2 warmup" if is_warming_up else f"detections={len(detections)}"
    hud = f"frame={frame_idx}  {status}"
    if study_label:
        hud = f"{study_label}  {hud}"
    cv2.putText(vis, hud, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return vis


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


def render_stream_frame(
    det_vis: np.ndarray,
    flow_vis: np.ndarray,
    stream_scale: float,
) -> np.ndarray:
    """Side-by-side detections (left) and optical flow (right) for live preview."""
    if stream_scale != 1.0:
        sw = max(1, int(det_vis.shape[1] * stream_scale))
        sh = max(1, int(det_vis.shape[0] * stream_scale))
        det_vis = cv2.resize(det_vis, (sw, sh), interpolation=cv2.INTER_AREA)
        flow_vis = cv2.resize(flow_vis, (sw, sh), interpolation=cv2.INTER_AREA)
    cv2.putText(
        det_vis, "detections", (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA,
    )
    cv2.putText(
        flow_vis, "optical flow", (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA,
    )
    return np.hstack([det_vis, flow_vis])


class ChunkMetrics:
    def __init__(self, chunk_idx: int, width: int, height: int, fps: float, study_id: str):
        self.chunk_idx = chunk_idx
        self.width = width
        self.height = height
        self.fps = fps
        self.study_id = study_id
        self.frame_rows: list[dict] = []
        self.detections_total = 0
        self.frames_with_detections = 0
        self.area_1px = self.area_2_3px = self.area_4_plus_px = 0
        self.max_detect_q = 0

    def add_frame(self, row: dict, detections: list[dict], detect_q_depth: int):
        self.frame_rows.append(row)
        self.max_detect_q = max(self.max_detect_q, detect_q_depth)
        if row["detection_count"] > 0 and not row["mog2_warmup"]:
            self.frames_with_detections += 1
        for det in detections:
            if row["mog2_warmup"]:
                continue
            self.detections_total += 1
            area = det["area_px"]
            if area == 1:
                self.area_1px += 1
            elif area <= 3:
                self.area_2_3px += 1
            else:
                self.area_4_plus_px += 1

    def summary(self, chunk_seconds: float, chunk_frames: int,
                hardware_summary: dict, output_mode: str,
                video_outputs: list[str], encoder: str) -> dict:
        def col(name: str) -> list[float]:
            return [float(r[name]) for r in self.frame_rows if r.get(name) is not None]

        active_rows = [r for r in self.frame_rows if not r["mog2_warmup"]]
        elapsed = max(chunk_seconds, 1e-6)
        mog2 = summarize(col("mog2_ms"))
        csv_w = summarize(col("csv_write_ms"))
        det_vid = summarize(col("detections_video_ms"))
        annot_vid = summarize(col("annot_video_ms"))
        flow_vid = summarize(col("flow_video_ms"))
        mask_vid = summarize(col("mask_video_ms"))
        clean_vid = summarize(col("clean_video_ms"))
        stream_w = summarize(col("stream_ms"))
        flow_ms = summarize(col("flow_ms"))
        worker = summarize(col("worker_frame_ms"))
        video_total = summarize(col("video_write_ms"))
        compute_mean = mog2["mean"] + flow_ms["mean"]
        io_mean = csv_w["mean"] + video_total["mean"] + stream_w["mean"]
        other_mean = max(0.0, worker["mean"] - compute_mean - io_mean)

        return {
            "chunk_index": self.chunk_idx,
            "study_id": self.study_id,
            "frames": chunk_frames,
            "frames_processed": len(self.frame_rows),
            "duration_sec": elapsed,
            "capture_fps_mean": chunk_frames / elapsed,
            "processed_fps_mean": len(self.frame_rows) / elapsed,
            "resolution": {"width": self.width, "height": self.height},
            "mog2_warmup_frames": MOG2_HISTORY,
            "flow_downsample": FLOW_DEBUG_DOWNSAMPLE,
            "pipeline_mode": f"gpu_study_{self.study_id}_{output_mode}",
            "output_mode": output_mode,
            "video_encoder": encoder,
            "video_outputs": video_outputs,
            "software": {
                "mog2_ms": mog2,
                "flow_ms": flow_ms,
                "csv_write_ms": csv_w,
                "detections_video_ms": det_vid,
                "annot_video_ms": annot_vid,
                "flow_video_ms": flow_vid,
                "mask_video_ms": mask_vid,
                "clean_video_ms": clean_vid,
                "video_write_ms": video_total,
                "stream_ms": stream_w,
                "worker_frame_ms": worker,
                "gpu_upload_ms": summarize(col("gpu_upload_ms")),
                "gpu_download_ms": summarize(col("gpu_download_ms")),
                "flow_mean_mag": summarize(col("flow_mean_mag")),
                "flow_p95_mag": summarize(col("flow_p95_mag")),
                "compute_ms_mean": compute_mean,
                "io_ms_mean": io_mean,
                "other_ms_mean": other_mean,
                "io_fraction_of_worker": io_mean / worker["mean"] if worker["mean"] > 0 else 0.0,
                "compute_fraction_of_worker": compute_mean / worker["mean"] if worker["mean"] > 0 else 0.0,
                "mog2_fg_pixels": summarize(col("mog2_fg_pixels")),
                "mog2_clean_pixels": summarize(col("mog2_clean_pixels")),
                "detection_count_per_frame": summarize(
                    [float(r["detection_count"]) for r in active_rows],
                ),
                "detect_queue_depth_max": self.max_detect_q,
            },
            "detections": {
                "total": self.detections_total,
                "frames_with_detections": self.frames_with_detections,
                "area_1px": self.area_1px,
                "area_2_3px": self.area_2_3px,
                "area_4_plus_px": self.area_4_plus_px,
            },
            "hardware": hardware_summary,
        }


class StudyWorker(threading.Thread):
    def __init__(self, width: int, height: int, fps: float, chunk_seconds: int,
                 hw_sampler: HardwareSampler, detector, output_mode: str,
                 stream_scale: float, stream_mode: str,
                 mjpeg_server: MjpegHttpServer | None, study_id: str, study_title: str,
                 use_hw_encode: bool):
        super().__init__(daemon=True, name="StudyWorker")
        self.width = width
        self.height = height
        self.fps = fps
        self.chunk_seconds = chunk_seconds
        self.hw_sampler = hw_sampler
        self.detector = detector
        self.output_mode = output_mode
        self.stream_scale = stream_scale
        self.stream_mode = stream_mode
        self.mjpeg_server = mjpeg_server
        self.study_id = study_id
        self.study_title = study_title
        self.use_hw_encode = use_hw_encode
        self.done_event = threading.Event()
        self.encoder_label = "none"
        self._reset()

    def _video_outputs(self) -> list[str]:
        if self.output_mode == "avi":
            outs = []
            if ANNOTATE_VIDEO:
                outs.append("annotated.avi")
            if SAVE_DETECTIONS_VIDEO:
                outs.append("detections.avi")
            if SAVE_FLOW_DEBUG_VIDEO:
                outs.append("flow_arrows.avi")
            if SAVE_MOG2_MASK_VIDEO:
                outs.append("mog2_mask.avi")
            if SAVE_MOG2_CLEAN_VIDEO:
                outs.append("mog2_clean.avi")
            return outs
        if self.output_mode == "stream":
            if self.stream_mode == "http":
                return [f"http_mjpeg://0.0.0.0:{STREAM_PORT}/stream@{self.stream_scale:.2f}x"]
            return [f"local_preview@{self.stream_scale:.2f}x"]
        return []

    def _reset(self):
        self._chunk_idx = None
        self._csv_file = self._csv_writer = None
        self._metrics_writer = self._metrics_csv = None
        self._annot_writer = self._flow_writer = None
        self._detections_writer = self._mask_writer = self._clean_writer = None
        self._frame_in_chunk = 0
        self._total_detections = 0
        self._prev_small = None
        self._chunk_metrics: ChunkMetrics | None = None

    def _metrics_dir(self) -> str:
        return os.path.join(OUTPUT_DIR, METRICS_SUBDIR)

    def _det_dir(self) -> str:
        return os.path.join(OUTPUT_DIR, DET_SUBDIR)

    def _open_outputs(self, chunk_idx: int):
        os.makedirs(self._det_dir(), exist_ok=True)
        os.makedirs(self._metrics_dir(), exist_ok=True)
        prefix = f"{FILE_PREFIX}_{chunk_idx:04d}"
        csv_path = os.path.join(self._det_dir(), f"{prefix}_detections.csv")
        self._csv_file = open(csv_path, "w", newline="", encoding="utf-8")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(["frame", "timestamp", "x", "y", "area_px", "peak_intensity"])

        metrics_path = os.path.join(self._metrics_dir(), f"{prefix}_frames.csv")
        self._metrics_writer = open(metrics_path, "w", newline="", encoding="utf-8")
        self._metrics_fields = [
            "frame", "timestamp", "mog2_warmup", "detection_count",
            "mog2_ms", "flow_ms", "csv_write_ms",
            "detections_video_ms", "annot_video_ms", "flow_video_ms",
            "mask_video_ms", "clean_video_ms", "video_write_ms", "stream_ms",
            "worker_frame_ms", "gpu_upload_ms", "gpu_download_ms",
            "mog2_fg_pixels", "mog2_clean_pixels",
            "flow_mean_mag", "flow_p95_mag", "detect_queue_depth",
        ]
        self._metrics_csv = csv.DictWriter(self._metrics_writer, fieldnames=self._metrics_fields)
        self._metrics_csv.writeheader()

        use_hw = self.use_hw_encode and self.output_mode == "avi"
        if self.output_mode == "avi":
            if ANNOTATE_VIDEO:
                self._annot_writer, enc = open_writer(
                    os.path.join(self._det_dir(), f"{prefix}_annotated.avi"),
                    self.width, self.height, self.fps, color=True, use_hw=use_hw,
                )
                self.encoder_label = enc
            if SAVE_DETECTIONS_VIDEO:
                self._detections_writer, enc = open_writer(
                    os.path.join(self._det_dir(), f"{prefix}_detections.avi"),
                    self.width, self.height, self.fps, color=True, use_hw=use_hw,
                )
                self.encoder_label = enc
            if SAVE_FLOW_DEBUG_VIDEO:
                flow_w = max(1, int(self.width * FLOW_DEBUG_DOWNSAMPLE))
                flow_h = max(1, int(self.height * FLOW_DEBUG_DOWNSAMPLE))
                self._flow_writer, enc = open_writer(
                    os.path.join(self._det_dir(), f"{prefix}_flow_arrows.avi"),
                    flow_w, flow_h, self.fps, color=True, use_hw=False,
                )
                self.encoder_label = enc
            if SAVE_MOG2_MASK_VIDEO:
                self._mask_writer, enc = open_writer(
                    os.path.join(self._det_dir(), f"{prefix}_mog2_mask.avi"),
                    self.width, self.height, self.fps, color=False, use_hw=use_hw,
                )
                self.encoder_label = enc
            if SAVE_MOG2_CLEAN_VIDEO:
                self._clean_writer, enc = open_writer(
                    os.path.join(self._det_dir(), f"{prefix}_mog2_clean.avi"),
                    self.width, self.height, self.fps, color=False, use_hw=use_hw,
                )
                self.encoder_label = enc

        self._chunk_metrics = ChunkMetrics(chunk_idx, self.width, self.height, self.fps, self.study_id)

    def _compute_flow_vis(self, gray: np.ndarray) -> tuple[np.ndarray, float, float, float]:
        engine = getattr(self.detector, "flow_engine", None)
        if engine is not None:
            return engine.compute(gray, self.width, self.height)
        small_w = max(1, int(self.width * FLOW_DEBUG_DOWNSAMPLE))
        small_h = max(1, int(self.height * FLOW_DEBUG_DOWNSAMPLE))
        curr_small = cv2.resize(
            gray, (small_w, small_h), interpolation=cv2.INTER_AREA,
        )
        flow_ms = 0.0
        flow_mean_mag = 0.0
        flow_p95_mag = 0.0
        if self._prev_small is None:
            flow_vis = cv2.cvtColor(curr_small, cv2.COLOR_GRAY2BGR)
        else:
            t_flow = time.monotonic()
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
        self._prev_small = curr_small
        return flow_vis, flow_ms, flow_mean_mag, flow_p95_mag

    def _close_outputs(self, chunk_idx: int | None):
        if self._chunk_metrics is not None and chunk_idx is not None:
            summary = self._chunk_metrics.summary(
                self.chunk_seconds, self._frame_in_chunk, self.hw_sampler.summary(),
                self.output_mode, self._video_outputs(), self.encoder_label,
            )
            summary_path = os.path.join(self._metrics_dir(), f"{FILE_PREFIX}_{chunk_idx:04d}_summary.json")
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            session_chunk_summaries.append(summary)
            sw = summary.get("software", {})
            print(
                f"\n  [Metrics] chunk {chunk_idx:04d}: "
                f"processed={summary['frames_processed']}  "
                f"worker={sw.get('worker_frame_ms', {}).get('mean', 0):.1f}ms  "
                f"mog2={sw.get('mog2_ms', {}).get('mean', 0):.1f}ms  "
                f"video_io={sw.get('video_write_ms', {}).get('mean', 0):.1f}ms  "
                f"encoder={self.encoder_label}  -> {summary_path}"
            )

        if self._csv_file is not None:
            print(f"\n  [Detect] chunk {chunk_idx}: {self._total_detections} detections "
                  f"-> {self._csv_file.name}")
            self._csv_file.close()
            self._csv_file = self._csv_writer = None
        if self._metrics_writer is not None:
            self._metrics_writer.close()
            self._metrics_writer = None
        for w in (
            self._annot_writer, self._flow_writer,
            self._detections_writer, self._mask_writer, self._clean_writer,
        ):
            if w is not None:
                w.release()
        self._annot_writer = self._flow_writer = None
        self._detections_writer = self._mask_writer = self._clean_writer = None
        self._frame_in_chunk = 0
        self._total_detections = 0
        self._prev_small = None
        engine = getattr(self.detector, "flow_engine", None)
        if engine is not None and hasattr(engine, "reset"):
            engine.reset()
        gst_pipe = getattr(self.detector, "_pipeline", None)
        if gst_pipe is not None and hasattr(gst_pipe, "reset"):
            gst_pipe.reset()
        detector_reset = getattr(self.detector, "reset", None)
        if callable(detector_reset):
            detector_reset()
        self._chunk_metrics = None

    def run(self):
        try:
            while True:
                item = detect_q.get()
                if item is _FLUSH or item is _STOP:
                    self._close_outputs(self._chunk_idx)
                    if item is _STOP:
                        break
                    continue

                chunk_idx, gray = item
                if chunk_idx != self._chunk_idx:
                    self._chunk_idx = chunk_idx
                    self._open_outputs(chunk_idx)

                self._frame_in_chunk += 1
                frame_t0 = time.monotonic()
                queue_depth = detect_q.qsize()

                result: ProcessResult = self.detector.process(gray)
                warmup_frames = getattr(self.detector, "warmup_frames", MOG2_HISTORY)
                is_warming_up = self.detector.frame_count <= warmup_frames

                csv_write_ms = 0.0
                if not is_warming_up:
                    t_csv = time.monotonic()
                    ts = time.time()
                    for det in result.detections:
                        self._csv_writer.writerow([
                            self._frame_in_chunk, f"{ts:.3f}",
                            det["x"], det["y"], det["area_px"], det["peak_intensity"],
                        ])
                        self._total_detections += 1
                    csv_write_ms = (time.monotonic() - t_csv) * 1000.0

                flow_ms = 0.0
                flow_mean_mag = 0.0
                flow_p95_mag = 0.0
                annot_video_ms = flow_video_ms = 0.0
                detections_video_ms = mask_video_ms = clean_video_ms = stream_ms = 0.0
                flow_vis = None

                if self.output_mode == "avi":
                    if self._annot_writer is not None:
                        t0 = time.monotonic()
                        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                        if not is_warming_up:
                            for det in result.detections:
                                cv2.circle(bgr, (det["x"], det["y"]), 8, (0, 0, 255), 1)
                        self._annot_writer.write(bgr)
                        annot_video_ms = (time.monotonic() - t0) * 1000.0
                    if self._detections_writer is not None:
                        t0 = time.monotonic()
                        vis = render_detections_frame(
                            self.width, self.height, result.detections,
                            self._frame_in_chunk, is_warming_up, self.study_id,
                        )
                        self._detections_writer.write(vis)
                        detections_video_ms = (time.monotonic() - t0) * 1000.0
                    if self._mask_writer is not None:
                        t0 = time.monotonic()
                        self._mask_writer.write(result.fg_mask)
                        mask_video_ms = (time.monotonic() - t0) * 1000.0
                    if self._clean_writer is not None:
                        t0 = time.monotonic()
                        self._clean_writer.write(result.fg_clean)
                        clean_video_ms = (time.monotonic() - t0) * 1000.0

                need_flow = (
                    self._flow_writer is not None
                    or self.output_mode == "stream"
                )
                flow_from_backend = result.extra.get("flow_vis")
                if flow_from_backend is not None:
                    flow_vis = flow_from_backend
                    flow_ms = float(result.extra.get("flow_ms", 0.0))
                    flow_mean_mag = float(result.extra.get("flow_mean_mag", 0.0))
                    flow_p95_mag = float(result.extra.get("flow_p95_mag", 0.0))
                elif need_flow:
                    flow_vis, flow_ms, flow_mean_mag, flow_p95_mag = (
                        self._compute_flow_vis(gray)
                    )
                else:
                    flow_vis = None
                if need_flow and flow_vis is not None:
                    if self._flow_writer is not None:
                        t0 = time.monotonic()
                        self._flow_writer.write(flow_vis)
                        flow_video_ms = (time.monotonic() - t0) * 1000.0

                if self.output_mode == "stream":
                    t0 = time.monotonic()
                    det_vis = render_detections_frame(
                        self.width, self.height, result.detections,
                        self._frame_in_chunk, is_warming_up, self.study_id,
                    )
                    preview = render_stream_frame(det_vis, flow_vis, self.stream_scale)
                    if self.stream_mode == "http" and self.mjpeg_server is not None:
                        self.mjpeg_server.update_frame(preview)
                    else:
                        cv2.imshow(STREAM_WINDOW, preview)
                        cv2.waitKey(1)
                    stream_ms = (time.monotonic() - t0) * 1000.0

                video_write_ms = (
                    detections_video_ms + annot_video_ms + flow_video_ms
                    + mask_video_ms + clean_video_ms
                )
                worker_frame_ms = (time.monotonic() - frame_t0) * 1000.0

                row = {
                    "frame": self._frame_in_chunk,
                    "timestamp": f"{time.time():.3f}",
                    "mog2_warmup": int(is_warming_up),
                    "detection_count": len(result.detections) if not is_warming_up else 0,
                    "mog2_ms": round(result.mog2_ms, 3),
                    "flow_ms": round(flow_ms, 3),
                    "csv_write_ms": round(csv_write_ms, 3),
                    "detections_video_ms": round(detections_video_ms, 3),
                    "annot_video_ms": round(annot_video_ms, 3),
                    "flow_video_ms": round(flow_video_ms, 3),
                    "mask_video_ms": round(mask_video_ms, 3),
                    "clean_video_ms": round(clean_video_ms, 3),
                    "video_write_ms": round(video_write_ms, 3),
                    "stream_ms": round(stream_ms, 3),
                    "worker_frame_ms": round(worker_frame_ms, 3),
                    "gpu_upload_ms": round(result.extra.get("gpu_upload_ms", 0.0), 3),
                    "gpu_download_ms": round(result.extra.get("gpu_download_ms", 0.0), 3),
                    "mog2_fg_pixels": result.fg_pixels,
                    "mog2_clean_pixels": result.clean_pixels,
                    "flow_mean_mag": round(flow_mean_mag, 4),
                    "flow_p95_mag": round(flow_p95_mag, 4),
                    "detect_queue_depth": queue_depth,
                }
                self._metrics_csv.writerow(row)
                if self._chunk_metrics is not None:
                    self._chunk_metrics.add_frame(row, result.detections, queue_depth)

        finally:
            if self.output_mode == "stream" and self.stream_mode == "local":
                cv2.destroyAllWindows()
            self.done_event.set()


def build_study_analysis(chunks: list[dict], study_id: str) -> dict:
    if not chunks:
        return {"verdict": "no_data", "study_id": study_id, "notes": []}

    worker_vals, mog2_vals, video_vals, frames_list = [], [], [], []
    for ch in chunks:
        sw = ch.get("software", {})
        if sw.get("worker_frame_ms", {}).get("count"):
            worker_vals.append(sw["worker_frame_ms"]["mean"])
        if sw.get("mog2_ms", {}).get("count"):
            mog2_vals.append(sw["mog2_ms"]["mean"])
        video_vals.append(sw.get("video_write_ms", {}).get("mean", 0.0))
        frames_list.append(ch.get("frames_processed", ch.get("frames", 0)))

    ref = POINTER_BENCH_REF
    worker_mean = statistics.mean(worker_vals) if worker_vals else 0.0
    mog2_mean = statistics.mean(mog2_vals) if mog2_vals else 0.0
    video_mean = statistics.mean(video_vals) if video_vals else 0.0
    frames_mean = statistics.mean(frames_list) if frames_list else 0.0

    notes = [
        f"Study `{study_id}` worker_frame_ms mean: {worker_mean:.1f} ms",
        f"mog2_ms mean: {mog2_mean:.1f} ms (pointer_bench ref ~{ref['mog2_ms_mean']:.0f} ms)",
        f"video_write_ms mean: {video_mean:.1f} ms (pointer_bench ref ~{ref['video_write_ms_mean']:.0f} ms)",
        f"frames/chunk mean: {frames_mean:.0f}",
    ]
    if worker_mean < ref["worker_frame_ms_mean_novideo"] * 1.2 and video_mean < 10:
        verdict = "compute_bound_like_novideo"
    elif video_mean > mog2_mean * 2:
        verdict = "video_io_bound"
    else:
        verdict = "mixed"

    return {
        "verdict": verdict,
        "study_id": study_id,
        "worker_frame_ms_mean": worker_mean,
        "mog2_ms_mean": mog2_mean,
        "video_write_ms_mean": video_mean,
        "frames_per_chunk_mean": frames_mean,
        "pointer_bench_reference": ref,
        "notes": notes,
    }


def write_session_report(output_dir: str, session_info: dict, capture: CaptureThread,
                         hw_sampler: HardwareSampler, study_id: str, study_title: str,
                         backend_meta: dict) -> tuple[str, str]:
    report_dir = os.path.join(output_dir, REPORT_SUBDIR)
    os.makedirs(report_dir, exist_ok=True)
    analysis = build_study_analysis(session_chunk_summaries, study_id)

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "study_id": study_id,
        "study_title": study_title,
        "session": session_info,
        "chunks": session_chunk_summaries,
        "study_analysis": analysis,
        "capture": {
            "frames_captured": capture.frames_captured,
            "frame_drops_at_capture": capture.frame_drops,
        },
        "hardware": hw_sampler.summary(),
        "backend_meta": backend_meta,
        "config": {
            "mog2_history": MOG2_HISTORY,
            "mog2_var_threshold": MOG2_VAR_THRESHOLD,
            "min_blob_area": MIN_BLOB_AREA,
            "max_blob_area": MAX_BLOB_AREA,
            "min_detection_intensity": MIN_DETECTION_INTENSITY,
            "annotate_video": ANNOTATE_VIDEO,
            "save_flow_debug_video": SAVE_FLOW_DEBUG_VIDEO,
            "flow_debug_downsample": FLOW_DEBUG_DOWNSAMPLE,
            "save_mog2_mask_video": SAVE_MOG2_MASK_VIDEO,
            "save_mog2_clean_video": SAVE_MOG2_CLEAN_VIDEO,
            "save_detections_video": SAVE_DETECTIONS_VIDEO,
            "output_mode": session_info.get("output_mode"),
            "use_hw_encode": USE_HW_ENCODE,
            "auto_exposure": True,
            "auto_gain": True,
            "camera": "SkyEye62AM",
            "resolution_index": RESOLUTION_INDEX,
        },
    }

    json_path = os.path.join(report_dir, "session_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    md_path = os.path.join(report_dir, "session_report.md")
    lines = [
        f"# GPU Study Session Report — {study_title}",
        "",
        f"- Study ID: `{study_id}`",
        f"- Generated (UTC): `{payload['generated_at_utc']}`",
        f"- Output dir: `{output_dir}`",
        f"- Chunks: `{len(session_chunk_summaries)}`",
        f"- Verdict: `{analysis['verdict']}`",
        "",
        "## vs pointer_bench reference",
        "",
        "| metric | this run | pointer_bench ref |",
        "|--------|---------:|------------------:|",
        f"| worker_frame_ms | {analysis.get('worker_frame_ms_mean', 0):.1f} | "
        f"{POINTER_BENCH_REF['worker_frame_ms_mean_avi']:.1f} (avi) / "
        f"{POINTER_BENCH_REF['worker_frame_ms_mean_novideo']:.1f} (no-video) |",
        f"| mog2_ms | {analysis.get('mog2_ms_mean', 0):.1f} | {POINTER_BENCH_REF['mog2_ms_mean']:.1f} |",
        f"| video_write_ms | {analysis.get('video_write_ms_mean', 0):.1f} | "
        f"{POINTER_BENCH_REF['video_write_ms_mean']:.1f} |",
        "",
        "## Per-chunk",
        "",
        "| chunk | processed | mog2 ms | video io ms | worker ms | detections | queue max |",
        "|------:|----------:|--------:|------------:|----------:|-----------:|----------:|",
    ]
    for ch in session_chunk_summaries:
        sw, det = ch.get("software", {}), ch.get("detections", {})
        lines.append(
            f"| {ch['chunk_index']:04d} | {ch.get('frames_processed', 0)} | "
            f"{sw.get('mog2_ms', {}).get('mean', 0):.1f} | "
            f"{sw.get('video_write_ms', {}).get('mean', 0):.1f} | "
            f"{sw.get('worker_frame_ms', {}).get('mean', 0):.1f} | "
            f"{det.get('total', 0)} | {sw.get('detect_queue_depth_max', 0)} |"
        )
    lines.extend([
        "",
        "## Output files",
        "",
        "- `detections/chunk_*_detections.csv`",
        "- `detections/chunk_*_annotated.avi`",
        "- `detections/chunk_*_detections.avi`",
        "- `detections/chunk_*_flow_arrows.avi`",
        "- `detections/chunk_*_mog2_mask.avi`",
        "- `detections/chunk_*_mog2_clean.avi`",
        "- `metrics/chunk_*_frames.csv`",
        "- `metrics/chunk_*_summary.json`",
        "",
    ])
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return json_path, md_path


def parse_args(description: str):
    p = argparse.ArgumentParser(description=description)
    p.add_argument(
        "-o", "--output", default=None,
        help=(
            "Exact output directory for this run. "
            f"Default: auto-create run_YYYYMMDD_HHMMSS under "
            f"{RESULTS_ROOT}/{GPU_STUDY_SUBDIR}/<study>/"
        ),
    )
    p.add_argument("--chunk-seconds", type=int, default=CHUNK_SECONDS)
    p.add_argument("--max-chunks", type=int, default=MAX_CHUNKS)
    p.add_argument("--min-area", type=int, default=MIN_BLOB_AREA)
    p.add_argument("--max-area", type=int, default=MAX_BLOB_AREA)
    p.add_argument("--min-intensity", type=int, default=MIN_DETECTION_INTENSITY)
    p.add_argument("--var-threshold", type=float, default=MOG2_VAR_THRESHOLD)
    p.add_argument("--no-video", action="store_true",
                   help="CSV + metrics only (no disk video)")
    p.add_argument("--no-annotate", action="store_true", help="Skip annotated.avi")
    p.add_argument("--no-flow-debug", action="store_true", help="Skip flow_arrows.avi")
    p.add_argument("--no-detections-video", action="store_true",
                   help="Skip detections.avi (still saves mask/clean if enabled)")
    p.add_argument("--no-mog2-mask", action="store_true", help="Skip mog2_mask.avi")
    p.add_argument("--no-mog2-clean", action="store_true", help="Skip mog2_clean.avi")
    p.add_argument("--stream", action="store_true", help="Live preview instead of disk video")
    p.add_argument("--stream-mode", choices=("local", "http"), default=STREAM_MODE)
    p.add_argument("--stream-port", type=int, default=STREAM_PORT)
    p.add_argument("--stream-scale", type=float, default=STREAM_SCALE)
    p.add_argument("--camera-id", type=int, default=None,
                   help="Index from detected SkyEye camera list")
    p.add_argument("--camera-name", default="SkyEye",
                   help="Substring match for auto-select (default: SkyEye)")
    p.add_argument("--resolution-index", type=int, default=RESOLUTION_INDEX,
                   help="ToupCam resolution index (default 2 = 3184×2124)")
    p.add_argument("--gain", type=int, default=REQUESTED_GAIN,
                   help="Requested auto-exposure gain cap")
    return p.parse_args()


def apply_cli_globals(args):
    global MIN_BLOB_AREA, MAX_BLOB_AREA, MIN_DETECTION_INTENSITY, MOG2_VAR_THRESHOLD
    global OUTPUT_DIR, OUTPUT_MODE, STREAM_SCALE, STREAM_MODE, STREAM_PORT
    global ANNOTATE_VIDEO, SAVE_FLOW_DEBUG_VIDEO
    global SAVE_MOG2_MASK_VIDEO, SAVE_MOG2_CLEAN_VIDEO, SAVE_DETECTIONS_VIDEO
    global RESOLUTION_INDEX, REQUESTED_GAIN

    MIN_BLOB_AREA = args.min_area
    MAX_BLOB_AREA = args.max_area
    MIN_DETECTION_INTENSITY = args.min_intensity
    MOG2_VAR_THRESHOLD = args.var_threshold
    OUTPUT_DIR = args.output
    RESOLUTION_INDEX = args.resolution_index
    REQUESTED_GAIN = args.gain
    STREAM_SCALE = max(0.05, min(1.0, args.stream_scale))
    STREAM_MODE = args.stream_mode
    STREAM_PORT = args.stream_port
    ANNOTATE_VIDEO = not args.no_annotate
    SAVE_FLOW_DEBUG_VIDEO = not args.no_flow_debug
    SAVE_MOG2_MASK_VIDEO = not args.no_mog2_mask
    SAVE_MOG2_CLEAN_VIDEO = not args.no_mog2_clean
    SAVE_DETECTIONS_VIDEO = not args.no_detections_video

    if args.no_video and args.stream:
        raise SystemExit("Use either --no-video or --stream, not both.")
    if args.stream:
        OUTPUT_MODE = "stream"
    elif args.no_video:
        OUTPUT_MODE = "none"
    else:
        OUTPUT_MODE = "avi"


def run_study(
    backend_factory: Callable[[], object],
    study_slug: str,
    cli_description: str = "GPU study pipeline — full pointer_bench options",
) -> None:
    args = parse_args(cli_description)
    if args.output is None:
        args.output = allocate_run_output_dir(study_slug)
    apply_cli_globals(args)

    import backends as be
    be.MOG2_VAR_THRESHOLD = MOG2_VAR_THRESHOLD
    be.MIN_BLOB_AREA = MIN_BLOB_AREA
    be.MAX_BLOB_AREA = MAX_BLOB_AREA
    be.MIN_DETECTION_INTENSITY = MIN_DETECTION_INTENSITY

    detector = backend_factory()
    study_id = getattr(detector, "study_id", STUDY_ID)
    study_title = getattr(detector, "study_title", STUDY_TITLE)
    use_hw = getattr(detector, "uses_hw_encode", USE_HW_ENCODE)

    backend_meta = {
        "study_id": study_id,
        "study_title": study_title,
        "uses_hw_encode": use_hw,
    }
    if hasattr(detector, "_vpi_info"):
        backend_meta["vpi"] = detector._vpi_info
    if hasattr(detector, "_stack"):
        backend_meta["dnn_stack"] = detector._stack
    if hasattr(detector, "_pva_nodes"):
        backend_meta["pva_nodes"] = detector._pva_nodes
    if getattr(detector, "flow_engine", None) is not None:
        backend_meta["flow_engine"] = detector.flow_engine.name
    if getattr(detector, "uses_gst_flow", False):
        backend_meta["flow_prefer_gst"] = True
    if getattr(detector, "uses_gst_pipeline", False):
        backend_meta["uses_gst_pipeline"] = True
        backend_meta["detection"] = "flow_magnitude"
    if getattr(detector, "uses_flow_detection", False):
        backend_meta["detection"] = "flow_magnitude"

    if not os.path.isdir(TOUPCAM_PATH):
        raise RuntimeError(f"ToupCam SDK path not found: {TOUPCAM_PATH}")

    os.makedirs(args.output, exist_ok=True)
    session_chunk_summaries.clear()

    hcam, width, height, cam_idx, cam_info = open_skyeye_camera(
        args.camera_id, args.camera_name, RESOLUTION_INDEX, REQUESTED_GAIN,
    )

    capture = hw_sampler = worker = mjpeg_server = gst_pipeline = None
    session_start = time.monotonic()
    chunk_index = 1
    total_frames = 0

    try:
        print(f"Study           : {study_title} ({study_id})")
        print(f"Camera          : {cam_info.displayname} ({cam_info.model.name})")
        print(f"Camera index    : {cam_idx}")
        print(f"Resolution      : {width} x {height} (index {RESOLUTION_INDEX})")
        print(f"Video encoder   : {'hw (GStreamer)' if use_hw and OUTPUT_MODE == 'avi' else 'software/none'}")
        print(f"Output dir      : {args.output}")

        capture = CaptureThread(hcam, width, height)
        capture.start()
        fps_est = round(measure_fps(capture))
        print(f"Measured FPS    : {fps_est:.1f}")

        if getattr(detector, "uses_gst_pipeline", False):
            from gst_flow_pipeline import GstFlowPipeline
            gst_pipeline = GstFlowPipeline(width, height, float(fps_est))
            detector.bind_pipeline(gst_pipeline)
            backend_meta["gst_pipeline"] = gst_pipeline.name
            print(f"GStreamer pipe  : {gst_pipeline.name}")
            print(f"Detection       : flow magnitude (no MOG2)")
        elif getattr(detector, "uses_flow_detection", False):
            print(f"Detection       : flow magnitude (no MOG2)")
        elif getattr(detector, "uses_gst_flow", False) and detector.flow_engine is None:
            from flow_engines import create_flow_engine
            detector.flow_engine = create_flow_engine(
                width, height, float(fps_est), prefer_gst=True,
            )
            backend_meta["flow_engine"] = detector.flow_engine.name
            print(f"Flow engine     : {detector.flow_engine.name}")

        hw_sampler = HardwareSampler()
        hw_sampler.start()

        if OUTPUT_MODE == "stream" and STREAM_MODE == "http":
            mjpeg_server = MjpegHttpServer(port=STREAM_PORT)
            mjpeg_server.start()
            print(f"HTTP stream     : http://{local_ip()}:{STREAM_PORT}/")

        worker = StudyWorker(
            width, height, fps_est, args.chunk_seconds, hw_sampler, detector,
            OUTPUT_MODE, STREAM_SCALE, STREAM_MODE, mjpeg_server,
            study_id, study_title, use_hw,
        )
        worker.start()
        print("Study worker started.\n")

        try:
            while True:
                if args.max_chunks and chunk_index > args.max_chunks:
                    print("\nMax chunks reached.")
                    break
                chunk_start = time.monotonic()
                chunk_frames = 0
                frame_timeout = max(5.0, 3.0 / fps_est)
                print(f"[Chunk {chunk_index:04d}]")
                while time.monotonic() - chunk_start < args.chunk_seconds:
                    try:
                        gray = frame_q.get(timeout=frame_timeout)
                    except queue.Empty:
                        raise KeyboardInterrupt
                    chunk_frames += 1
                    total_frames += 1
                    if not detect_q.full():
                        detect_q.put_nowait((chunk_index, gray))
                    elapsed = time.monotonic() - chunk_start
                    print(
                        f"\r  {elapsed:5.1f}s/{args.chunk_seconds}s  frames={chunk_frames:5d}  "
                        f"detect_q={detect_q.qsize()}  drops={capture.frame_drops}",
                        end="", flush=True,
                    )
                print()
                detect_q.put(_FLUSH)
                chunk_index += 1
        except KeyboardInterrupt:
            print("\nStopped by user.")
        finally:
            if capture:
                capture.stop()
                capture.join(timeout=5)
            if worker:
                detect_q.put(_FLUSH)
                detect_q.put(_STOP)
                worker.done_event.wait(timeout=120)
            if hw_sampler:
                hw_sampler.stop()
                hw_sampler.join(timeout=2)
            if mjpeg_server:
                mjpeg_server.stop()
                mjpeg_server.join(timeout=2)
            if gst_pipeline is not None:
                gst_pipeline.close()

        session_info = {
            "study_id": study_id,
            "study_slug": study_slug,
            "study_title": study_title,
            "output_dir": args.output,
            "camera_index": cam_idx,
            "camera_model": cam_info.model.name,
            "camera_name": cam_info.displayname,
            "resolution_index": RESOLUTION_INDEX,
            "resolution": {"width": width, "height": height},
            "fps_measured": fps_est,
            "chunk_seconds": args.chunk_seconds,
            "chunks_written": chunk_index - 1,
            "total_frames": total_frames,
            "total_minutes": (time.monotonic() - session_start) / 60.0,
            "output_mode": OUTPUT_MODE,
            "stream_mode": STREAM_MODE if OUTPUT_MODE == "stream" else None,
            "stream_scale": STREAM_SCALE if OUTPUT_MODE == "stream" else None,
        }
        json_path, md_path = write_session_report(
            args.output, session_info, capture, hw_sampler,
            study_id, study_title, backend_meta,
        )
        print(f"\nReport JSON: {json_path}")
        print(f"Report MD  : {md_path}")
    finally:
        if hcam is not None:
            hcam.Close()
        print("Camera closed.")
