#!/usr/bin/env python3
"""Live-camera flow-only bench: Study 01 Farneback vs Study 02 nvof.

Only steps before flow + the flow kernel. No MOG2, no detection, no video.

  01_opencv_cuda flow path:
      gray (already debayered) → [optional resize] → Farneback
      With downsample=1.0: NO resize — Farneback on full frame.

  02_jetson_encode flow path:
      gray → appsrc GRAY8 → videoconvert → nvvideoconvert → NV12 NVMM
        → nvstreammux → nvof
      prep_ms = push → just before nvof
      flow_ms = nvof sink → nvof src

Defaults: camera full frame (3184×2124), downsample=1.0, 20 timed pairs.
Gridsize note: Farneback has none; nvof uses fixed HW block size (~4×4),
not a VPI-style gridsize knob.

  source ~/optical-flow/bin/activate
  python3 bench_01_02_flow.py --runs 20 --verbose
"""

from __future__ import annotations

import argparse
import json
import queue
import statistics
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

SKYMOVE = Path(__file__).resolve().parent
sys.path.insert(0, str(SKYMOVE))

import flow_engines as fe  # noqa: E402
import pipeline as pl  # noqa: E402
from camera_bench import capture_grays, open_bench_camera  # noqa: E402

RESULTS_ROOT = SKYMOVE / "results" / "ofa"

NO_DOWNSAMPLE = 1.0
DEFAULT_RUNS = 20
DEFAULT_WARMUP = 3


def _stats_ms(samples: list[float]) -> dict:
    if not samples:
        return {
            "count": 0, "mean_ms": 0.0, "std_ms": 0.0,
            "min_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0,
        }
    xs = sorted(samples)
    n = len(xs)
    return {
        "count": n,
        "mean_ms": round(statistics.mean(xs), 3),
        "std_ms": round(statistics.pstdev(xs), 3) if n > 1 else 0.0,
        "min_ms": round(xs[0], 3),
        "p50_ms": round(xs[n // 2], 3),
        "p95_ms": round(xs[max(0, int(n * 0.95) - 1)], 3),
        "max_ms": round(xs[-1], 3),
    }


# ---------------------------------------------------------------------------
# 01 — Farneback only (StudyWorker._compute_flow_vis path, no MOG2)
# ---------------------------------------------------------------------------

def bench_01_farneback(
    grays: list[np.ndarray], width: int, height: int, warmup: int, downsample: float,
) -> dict:
    """Prep = resize if downsample<1; submit = Farneback; rlock = 0 (already CPU)."""
    fw = max(1, int(width * downsample))
    fh = max(1, int(height * downsample))
    do_resize = downsample < 1.0 - 1e-9

    prep_s: list[float] = []
    submit_s: list[float] = []
    rlock_s: list[float] = []
    total_s: list[float] = []

    def _prep_pair(a: np.ndarray, b: np.ndarray):
        t0 = time.perf_counter()
        if do_resize:
            pa = cv2.resize(a, (fw, fh), interpolation=cv2.INTER_AREA)
            pb = cv2.resize(b, (fw, fh), interpolation=cv2.INTER_AREA)
        else:
            pa, pb = a, b
        prep_ms = (time.perf_counter() - t0) * 1000.0
        return pa, pb, prep_ms

    for i in range(warmup):
        if i + 1 >= len(grays):
            break
        pa, pb, _ = _prep_pair(grays[i], grays[i + 1])
        cv2.calcOpticalFlowFarneback(
            pa, pb, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )

    for i in range(warmup, len(grays) - 1):
        pa, pb, prep_ms = _prep_pair(grays[i], grays[i + 1])
        t1 = time.perf_counter()
        _flow = cv2.calcOpticalFlowFarneback(
            pa, pb, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )
        submit_ms = (time.perf_counter() - t1) * 1000.0
        _ = _flow
        rlock_ms = 0.0  # Farneback returns NumPy on CPU already
        prep_s.append(prep_ms)
        submit_s.append(submit_ms)
        rlock_s.append(rlock_ms)
        total_s.append(prep_ms + submit_ms + rlock_ms)

    total_stats = _stats_ms(total_s)
    fps = round(1000.0 / total_stats["mean_ms"], 2) if total_stats["mean_ms"] > 0 else 0.0
    return {
        "script": "01_opencv_cuda.py",
        "method": "Farneback (CPU, per-pixel)",
        "architecture": "GRAY8 → [resize?] → Farneback",
        "prep_ms": _stats_ms(prep_s),
        "submit_ms": _stats_ms(submit_s),
        "rlock_ms": _stats_ms(rlock_s),
        "total_ms": total_stats,
        "fps": fps,
        # aliases for older report fields
        "flow_ms": _stats_ms(submit_s),
        "time_spent_on_optical_flow": total_stats,
        "flow_resolution": f"{fw}×{fh}",
        "downsample": downsample,
        "gridsize": "n/a",
        "gridsize_label": "n/a (Farneback per-pixel)",
        "source": "live SkyEye camera",
        "detection": False,
    }


# ---------------------------------------------------------------------------
# 02 — GStreamer GRAY8→NV12→nvof only (no detection / nvofvisual)
# ---------------------------------------------------------------------------

class Gst02NvofBench:
    """Push gray; split prep (→ pre_nvof) vs flow (nvof in→out)."""

    def __init__(self, width: int, height: int, fps: float = 30.0):
        self.width = width
        self.height = height
        self._gst: Any = None
        self._pipeline: Any = None
        self._appsrc: Any = None
        self._pts = 0
        self._duration = 0
        self._pending: dict = {}
        self._lock = threading.Lock()
        self._results: queue.Queue = queue.Queue(maxsize=32)
        self._fps = fps

    def _on_pre_nvof(self, pad, info, _ud):
        buf = info.get_buffer()
        if buf is None:
            return self._gst.PadProbeReturn.OK
        with self._lock:
            if "t0" in self._pending and "t_prep" not in self._pending:
                self._pending["t_prep"] = time.perf_counter()
        return self._gst.PadProbeReturn.OK

    def _on_nvof_src(self, pad, info, _ud):
        buf = info.get_buffer()
        if buf is None:
            return self._gst.PadProbeReturn.OK
        t_flow = time.perf_counter()
        with self._lock:
            if "t0" not in self._pending or "t_prep" not in self._pending:
                return self._gst.PadProbeReturn.OK
            t0 = self._pending.pop("t0")
            t_prep = self._pending.pop("t_prep")
        self._results.put_nowait({
            "prep_ms": (t_prep - t0) * 1000.0,
            "flow_ms": (t_flow - t_prep) * 1000.0,
            "total_ms": (t_flow - t0) * 1000.0,
        })
        return self._gst.PadProbeReturn.OK

    def start(self) -> None:
        import gi
        gi.require_version("Gst", "1.0")
        from gi.repository import Gst
        self._gst = Gst
        Gst.init(None)
        fps_i = max(1, int(round(self._fps)))
        self._duration = int(Gst.SECOND / self._fps)
        w, h = self.width, self.height
        # Same pre-nvof chain as GstFlowPipeline._build_nvof_pipeline, but
        # fakesink (no nvofvisual / appsink / detection).
        desc = (
            f"appsrc name=src is-live=true block=true format=3 do-timestamp=true "
            f"caps=video/x-raw,format=GRAY8,width={w},height={h},framerate={fps_i}/1 ! "
            f"videoconvert ! nvvideoconvert ! "
            f"video/x-raw(memory:NVMM),format=NV12,width={w},height={h} ! "
            f"mux.sink_0 nvstreammux name=mux batch-size=1 width={w} height={h} "
            f"live-source=1 batched-push-timeout=40000 ! "
            f"queue ! identity name=pre_nvof ! nvof name=nvof ! fakesink sync=false"
        )
        self._pipeline = Gst.parse_launch(desc)
        self._appsrc = self._pipeline.get_by_name("src")
        pre = self._pipeline.get_by_name("pre_nvof")
        nvof = self._pipeline.get_by_name("nvof")
        if not all((self._appsrc, pre, nvof)):
            raise RuntimeError("Failed to build Study 02 nvof bench pipeline")
        pre.get_static_pad("src").add_probe(
            Gst.PadProbeType.BUFFER, self._on_pre_nvof, None,
        )
        nvof.get_static_pad("src").add_probe(
            Gst.PadProbeType.BUFFER, self._on_nvof_src, None,
        )
        ret = self._pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("GStreamer PLAYING failed for Study 02 bench")

    def push_once(self, gray: np.ndarray) -> tuple[float, float, float]:
        Gst = self._gst
        data = np.ascontiguousarray(gray, dtype=np.uint8).reshape(-1)
        with self._lock:
            self._pending = {"t0": time.perf_counter()}
        buf = Gst.Buffer.new_wrapped(data.tobytes())
        buf.pts = self._pts
        buf.duration = self._duration
        self._pts += self._duration
        ret = self._appsrc.emit("push-buffer", buf)
        if ret != Gst.FlowReturn.OK and ret != Gst.FlowReturn.FLUSHING:
            raise RuntimeError(f"appsrc push failed: {ret}")
        item = self._results.get(timeout=30.0)
        return item["prep_ms"], item["flow_ms"], item["total_ms"]

    def close(self) -> None:
        if self._pipeline is not None:
            self._pipeline.set_state(self._gst.State.NULL)
            self._pipeline = None


def bench_02_nvof(
    grays: list[np.ndarray], width: int, height: int, warmup: int,
) -> dict:
    """prep = GRAY8→NV12→mux; submit = nvof; rlock = 0 (no CPU vector pull in this bench)."""
    bench = Gst02NvofBench(width, height)
    prep_s: list[float] = []
    submit_s: list[float] = []
    rlock_s: list[float] = []
    total_s: list[float] = []
    try:
        bench.start()
        for g in grays[: warmup + 1]:
            bench.push_once(g)
        prep_s.clear()
        submit_s.clear()
        rlock_s.clear()
        total_s.clear()
        timed = grays[warmup:]
        for g in timed:
            p, f, t = bench.push_once(g)
            rlock_ms = 0.0  # stripped pipeline: no flow-meta CPU readback
            prep_s.append(p)
            submit_s.append(f)
            rlock_s.append(rlock_ms)
            total_s.append(p + f + rlock_ms)
    finally:
        bench.close()

    total_stats = _stats_ms(total_s)
    fps = round(1000.0 / total_stats["mean_ms"], 2) if total_stats["mean_ms"] > 0 else 0.0
    return {
        "script": "02_jetson_encode.py",
        "method": "GStreamer nvof (HW)",
        "architecture": "GRAY8 → appsrc → NV12 → mux → nvof",
        "prep_ms": _stats_ms(prep_s),
        "submit_ms": _stats_ms(submit_s),
        "rlock_ms": _stats_ms(rlock_s),
        "total_ms": total_stats,
        "fps": fps,
        "flow_ms": _stats_ms(submit_s),
        "time_spent_on_optical_flow": total_stats,
        "flow_resolution": f"{width}×{height}",
        "downsample": 1.0,
        "gridsize": "~4×4",
        "gridsize_label": "~4×4 nvof HW fixed",
        "source": "live SkyEye camera",
        "detection": False,
    }


def _write_report(out_dir: Path, payload: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "flow_report.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8",
    )
    md = out_dir / "flow_report.md"
    cfg = payload["config"]
    lines = [
        "# Live-camera Study 01 vs 02 — flow only",
        "",
        f"- Generated (UTC): `{payload['generated_at_utc']}`",
        f"- **Source: live SkyEye62AM camera**",
        f"- **Detection/MOG2: OFF**",
        f"- Resolution: **{cfg['width']}×{cfg['height']}**",
        f"- Downsample: **{cfg['downsample']}**",
        f"- Runs: {cfg['runs']} (warmup {cfg['warmup']})",
        f"- Shared frame crop/copy mean: {cfg['debayer_ms_mean']:.3f} ms",
        f"- `stream_fps = 1000 / total_ms`",
        "",
        "## Metric definitions",
        "",
        "| Column | Meaning |",
        "|--------|---------|",
        "| **prep** | Format / resize before the flow kernel |",
        "| **submit** | Flow kernel only (Farneback or nvof) |",
        "| **rlock** | CPU readback of flow (0 here: Farneback already CPU; nvof bench has no meta pull) |",
        "| **total** | prep + submit + rlock |",
        "| **fps** | `1000 / total` |",
        "",
        "## Timing",
        "",
        "| method | resolution | grid | prep | submit | rlock | total | **fps** |",
        "|--------|------------|------|-----:|-------:|------:|------:|--------:|",
    ]
    for r in payload["rows"]:
        prep = r["prep_ms"]["mean_ms"]
        submit = r["submit_ms"]["mean_ms"]
        rlock = r["rlock_ms"]["mean_ms"]
        total = r["total_ms"]["mean_ms"]
        lines.append(
            f"| **{r['method']}** | {r.get('flow_resolution', '?')} | "
            f"{r.get('gridsize', '?')} | {prep:.1f} | {submit:.1f} | "
            f"{rlock:.1f} | {total:.1f} | **{r['fps']:.1f}** |"
        )
    lines.extend([
        "",
        "## Architecture",
        "",
        "### Shared",
        "",
        "```",
        "SkyEye62AM GRAY8 (mono) → [center crop if ROI] → flow kernels",
        "```",
        "",
        "### 01 Farneback",
        "",
        "```",
        "GRAY8 → [resize if ds<1] → Farneback(CPU) → NumPy flow",
        "         └─ prep ─┘      └── submit ──┘   rlock=0",
        "```",
        "",
        "### 02 nvof",
        "",
        "```",
        "GRAY8 → appsrc → NV12 → mux → nvof → fakesink",
        "        └──────── prep ────────┘  └submit┘",
        "```",
        "",
    ])
    md.write_text("\n".join(lines), encoding="utf-8")
    return md


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live camera: Study 01 Farneback vs 02 nvof — prep+flow only",
    )
    parser.add_argument("--camera-id", type=int, default=None)
    parser.add_argument("--camera-name", default="SkyEye")
    parser.add_argument(
        "--resolution-index", type=int, default=None,
        help="ToupCam resolution index (default: pipeline RESOLUTION_INDEX=2 → 3184×2124)",
    )
    parser.add_argument(
        "--width", type=int, default=None,
        help="Camera ROI width (default: full sensor). e.g. 1920",
    )
    parser.add_argument(
        "--height", type=int, default=None,
        help="Camera ROI height (default: full sensor). e.g. 1080",
    )
    parser.add_argument(
        "--flow-downsample", type=float, default=NO_DOWNSAMPLE,
        help="Farneback resize factor (default 1.0 = no resize)",
    )
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("-o", "--output-dir", default="")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if (args.width is None) ^ (args.height is None):
        raise SystemExit("Pass both --width and --height, or neither")

    ds = max(0.05, min(1.0, args.flow_downsample))
    fe.FLOW_DEBUG_DOWNSAMPLE = ds
    pl.FLOW_DEBUG_DOWNSAMPLE = ds

    # Need pairs for Farneback; nvof needs successive pushes ≈ runs+warmup
    need = args.warmup + args.runs + 1

    print("=" * 72, flush=True)
    print("LIVE CAMERA — Study 01 Farneback vs Study 02 nvof (flow only)", flush=True)
    if args.width is not None:
        print(f"  ROI         : {args.width}×{args.height}", flush=True)
    else:
        print("  ROI         : full sensor", flush=True)
    print(f"  downsample : {ds}", flush=True)
    print(f"  runs/warmup: {args.runs}/{args.warmup}", flush=True)
    print("  gridsize   : 01=n/a (Farneback); 02=nvof HW ~4×4 (fixed)", flush=True)
    print("=" * 72, flush=True)

    session = open_bench_camera(
        args.camera_id, args.camera_name,
        roi_w=args.width, roi_h=args.height,
        resolution_index=args.resolution_index,
    )
    width, height = session.out_w, session.out_h
    print(f"Camera [{session.cam_idx}]  output {width}×{height}", flush=True)

    try:
        grays, crop_ms = capture_grays(session, need)
        print(
            f"Frame crop/copy mean: {statistics.mean(crop_ms):.3f} ms (shared)",
            flush=True,
        )

        print("\n[1/2] Study 01 Farneback @ "
              f"{max(1, int(width * ds))}×{max(1, int(height * ds))} …", flush=True)
        row01 = bench_01_farneback(grays, width, height, args.warmup, ds)

        print("[2/2] Study 02 GStreamer nvof …", flush=True)
        row02 = bench_02_nvof(grays, width, height, args.warmup)

        payload = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "config": {
                "source": "live_skyeye_camera",
                "synthetic": False,
                "detection": False,
                "mog2": False,
                "camera_index": session.cam_idx,
                "width": width,
                "height": height,
                "downsample": ds,
                "runs": args.runs,
                "warmup": args.warmup,
                "crop_ms_mean": round(statistics.mean(crop_ms), 3),
                "crop_ms": _stats_ms(crop_ms),
                "debayer_ms_mean": round(statistics.mean(crop_ms), 3),
                "debayer_ms": _stats_ms(crop_ms),
                "gridsize_note": (
                    "01 Farneback: n/a; 02 nvof: fixed HW ~4×4 (not VPI gridsize)"
                ),
            },
            "rows": [row01, row02],
        }

        out_dir = Path(args.output_dir) if args.output_dir else (
            RESULTS_ROOT / datetime.now().strftime("flow_01_02_run_%Y%m%d_%H%M%S")
        )
        md_path = _write_report(out_dir, payload)

        print(
            f"\n{'method':<28} {'prep':>8} {'submit':>8} {'rlock':>8} "
            f"{'total':>8} {'fps':>8}",
            flush=True,
        )
        print("-" * 72, flush=True)
        for r in payload["rows"]:
            print(
                f"{r['method']:<28} "
                f"{r['prep_ms']['mean_ms']:8.1f} "
                f"{r['submit_ms']['mean_ms']:8.1f} "
                f"{r['rlock_ms']['mean_ms']:8.1f} "
                f"{r['total_ms']['mean_ms']:8.1f} "
                f"{r['fps']:8.1f}",
                flush=True,
            )
        print(f"\nReport: {md_path}", flush=True)
    finally:
        session.close()


if __name__ == "__main__":
    main()
