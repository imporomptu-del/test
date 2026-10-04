#!/usr/bin/env python3
"""Live-camera Farneback: CPU vs OpenCV CUDA GPU.

Same metric columns as the OFA / 01–02 reports:
  prep | submit | rlock | total | fps

  CPU:  prep = optional resize; submit = calcOpticalFlowFarneback; rlock = 0
  GPU:  prep = optional resize + GpuMat upload; submit = cuda Farneback.calc
        (+ device sync); rlock = flow.download()

No MOG2 / detection. Live SkyEye62AM only.

  source ~/optical-flow/bin/activate
  python3 bench_01_farneback_gpu.py --width 1920 --height 1080 --runs 20
  python3 bench_01_farneback_gpu.py --runs 20   # full sensor
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

SKYMOVE = Path(__file__).resolve().parent
sys.path.insert(0, str(SKYMOVE))

from camera_bench import capture_grays, open_bench_camera  # noqa: E402

RESULTS_ROOT = SKYMOVE / "results" / "ofa"
NO_DOWNSAMPLE = 1.0
DEFAULT_RUNS = 20
DEFAULT_WARMUP = 3

# Match CPU Farneback params used elsewhere in gpu_study
FB_KW = dict(
    pyr_scale=0.5, levels=3, winsize=15,
    iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
)


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


def _row(
    *,
    method: str,
    architecture: str,
    prep_s: list[float],
    submit_s: list[float],
    rlock_s: list[float],
    flow_resolution: str,
    downsample: float,
    gridsize: str,
) -> dict:
    total_s = [p + s + r for p, s, r in zip(prep_s, submit_s, rlock_s)]
    total_stats = _stats_ms(total_s)
    fps = round(1000.0 / total_stats["mean_ms"], 2) if total_stats["mean_ms"] > 0 else 0.0
    return {
        "method": method,
        "architecture": architecture,
        "prep_ms": _stats_ms(prep_s),
        "submit_ms": _stats_ms(submit_s),
        "rlock_ms": _stats_ms(rlock_s),
        "total_ms": total_stats,
        "fps": fps,
        "flow_resolution": flow_resolution,
        "downsample": downsample,
        "gridsize": gridsize,
        "source": "live SkyEye camera",
        "detection": False,
    }


def _prep_pair(a: np.ndarray, b: np.ndarray, fw: int, fh: int, do_resize: bool):
    t0 = time.perf_counter()
    if do_resize:
        pa = cv2.resize(a, (fw, fh), interpolation=cv2.INTER_AREA)
        pb = cv2.resize(b, (fw, fh), interpolation=cv2.INTER_AREA)
    else:
        pa, pb = a, b
    prep_ms = (time.perf_counter() - t0) * 1000.0
    return pa, pb, prep_ms


def bench_cpu_farneback(
    grays: list[np.ndarray], width: int, height: int, warmup: int, downsample: float,
) -> dict:
    fw = max(1, int(width * downsample))
    fh = max(1, int(height * downsample))
    do_resize = downsample < 1.0 - 1e-9
    prep_s: list[float] = []
    submit_s: list[float] = []
    rlock_s: list[float] = []

    for i in range(warmup):
        if i + 1 >= len(grays):
            break
        pa, pb, _ = _prep_pair(grays[i], grays[i + 1], fw, fh, do_resize)
        cv2.calcOpticalFlowFarneback(pa, pb, None, **FB_KW)

    for i in range(warmup, len(grays) - 1):
        pa, pb, prep_ms = _prep_pair(grays[i], grays[i + 1], fw, fh, do_resize)
        t1 = time.perf_counter()
        _flow = cv2.calcOpticalFlowFarneback(pa, pb, None, **FB_KW)
        submit_ms = (time.perf_counter() - t1) * 1000.0
        _ = _flow
        prep_s.append(prep_ms)
        submit_s.append(submit_ms)
        rlock_s.append(0.0)

    return _row(
        method="Farneback CPU",
        architecture="GRAY8 → [resize?] → calcOpticalFlowFarneback",
        prep_s=prep_s,
        submit_s=submit_s,
        rlock_s=rlock_s,
        flow_resolution=f"{fw}×{fh}",
        downsample=downsample,
        gridsize="n/a",
    )


def bench_gpu_farneback(
    grays: list[np.ndarray], width: int, height: int, warmup: int, downsample: float,
) -> dict:
    """prep = resize + upload; submit = cuda Farneback; rlock = download."""
    if cv2.cuda.getCudaEnabledDeviceCount() < 1:
        raise RuntimeError("No OpenCV CUDA device available")

    fw = max(1, int(width * downsample))
    fh = max(1, int(height * downsample))
    do_resize = downsample < 1.0 - 1e-9

    fo = cv2.cuda.FarnebackOpticalFlow_create(
        numLevels=FB_KW["levels"],
        pyrScale=FB_KW["pyr_scale"],
        fastPyramids=False,
        winSize=FB_KW["winsize"],
        numIters=FB_KW["iterations"],
        polyN=FB_KW["poly_n"],
        polySigma=FB_KW["poly_sigma"],
        flags=FB_KW["flags"],
    )
    stream = cv2.cuda.Stream()

    prep_s: list[float] = []
    submit_s: list[float] = []
    rlock_s: list[float] = []

    def _one(a: np.ndarray, b: np.ndarray, record: bool) -> None:
        t0 = time.perf_counter()
        if do_resize:
            pa = cv2.resize(a, (fw, fh), interpolation=cv2.INTER_AREA)
            pb = cv2.resize(b, (fw, fh), interpolation=cv2.INTER_AREA)
        else:
            pa, pb = a, b
        d0 = cv2.cuda_GpuMat()
        d1 = cv2.cuda_GpuMat()
        d0.upload(pa, stream)
        d1.upload(pb, stream)
        stream.waitForCompletion()
        prep_ms = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        flow_gpu = fo.calc(d0, d1, None, stream)
        stream.waitForCompletion()
        submit_ms = (time.perf_counter() - t1) * 1000.0

        t2 = time.perf_counter()
        flow_host = flow_gpu.download()
        rlock_ms = (time.perf_counter() - t2) * 1000.0
        _ = flow_host

        if record:
            prep_s.append(prep_ms)
            submit_s.append(submit_ms)
            rlock_s.append(rlock_ms)

    for i in range(warmup):
        if i + 1 >= len(grays):
            break
        _one(grays[i], grays[i + 1], record=False)

    for i in range(warmup, len(grays) - 1):
        _one(grays[i], grays[i + 1], record=True)

    return _row(
        method="Farneback GPU (cv2.cuda)",
        architecture="GRAY8 → [resize?] → upload → cuda.FarnebackOpticalFlow → download",
        prep_s=prep_s,
        submit_s=submit_s,
        rlock_s=rlock_s,
        flow_resolution=f"{fw}×{fh}",
        downsample=downsample,
        gridsize="n/a",
    )


def _write_report(out_dir: Path, payload: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "flow_report.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8",
    )
    md = out_dir / "flow_report.md"
    cfg = payload["config"]
    lines = [
        "# Live-camera Farneback — CPU vs OpenCV CUDA GPU",
        "",
        f"- Generated (UTC): `{payload['generated_at_utc']}`",
        f"- **Source: live ASI camera**",
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
        "| **prep** | CPU: optional resize. GPU: resize + GpuMat upload |",
        "| **submit** | Farneback kernel (CPU API or `cuda.FarnebackOpticalFlow.calc` + sync) |",
        "| **rlock** | CPU: 0. GPU: `GpuMat.download()` of flow |",
        "| **total** | prep + submit + rlock |",
        "| **fps** | `1000 / total` |",
        "",
        "## Timing",
        "",
        "| method | resolution | grid | prep | submit | rlock | total | **fps** |",
        "|--------|------------|------|-----:|-------:|------:|------:|--------:|",
    ]
    for r in payload["rows"]:
        lines.append(
            f"| **{r['method']}** | {r.get('flow_resolution', '?')} | "
            f"{r.get('gridsize', '?')} | "
            f"{r['prep_ms']['mean_ms']:.1f} | "
            f"{r['submit_ms']['mean_ms']:.1f} | "
            f"{r['rlock_ms']['mean_ms']:.1f} | "
            f"{r['total_ms']['mean_ms']:.1f} | "
            f"**{r['fps']:.1f}** |"
        )
    lines.extend([
        "",
        "## Architecture",
        "",
        "### Farneback CPU",
        "",
        "```",
        "GRAY8 → [resize if ds<1] → calcOpticalFlowFarneback → NumPy",
        "         └─ prep ─┘      └──────── submit ────────┘  rlock=0",
        "```",
        "",
        "### Farneback GPU",
        "",
        "```",
        "GRAY8 → [resize?] → upload GpuMat → cuda.FarnebackOpticalFlow.calc → download",
        "        └──────── prep ────────┘  └────────── submit ──────────┘  └ rlock ┘",
        "```",
        "",
    ])
    md.write_text("\n".join(lines), encoding="utf-8")
    return md


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live camera: Farneback CPU vs OpenCV CUDA GPU",
    )
    parser.add_argument("--camera-id", type=int, default=None)
    parser.add_argument("--camera-name", default="SkyEye")
    parser.add_argument("--resolution-index", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--flow-downsample", type=float, default=NO_DOWNSAMPLE)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--skip-cpu", action="store_true", help="Only run GPU Farneback")
    parser.add_argument("-o", "--output-dir", default="")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if (args.width is None) ^ (args.height is None):
        raise SystemExit("Pass both --width and --height, or neither")

    ds = max(0.05, min(1.0, args.flow_downsample))
    need = args.warmup + args.runs + 1

    print("=" * 72, flush=True)
    print("LIVE CAMERA — Farneback CPU vs OpenCV CUDA GPU", flush=True)
    if args.width is not None:
        print(f"  ROI         : {args.width}×{args.height}", flush=True)
    else:
        print("  ROI         : full sensor", flush=True)
    print(f"  downsample : {ds}", flush=True)
    print(f"  runs/warmup: {args.runs}/{args.warmup}", flush=True)
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

        rows: list[dict] = []
        if not args.skip_cpu:
            print(
                f"\n[1/2] Farneback CPU @ "
                f"{max(1, int(width * ds))}×{max(1, int(height * ds))} …",
                flush=True,
            )
            rows.append(bench_cpu_farneback(grays, width, height, args.warmup, ds))

        label = "[2/2]" if not args.skip_cpu else "[1/1]"
        print(f"{label} Farneback GPU (cv2.cuda) …", flush=True)
        rows.append(bench_gpu_farneback(grays, width, height, args.warmup, ds))

        payload = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "config": {
                "source": "live_skyeye_camera",
                "synthetic": False,
                "detection": False,
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
                "farneback_params": FB_KW,
                "gpu_api": "cv2.cuda.FarnebackOpticalFlow",
            },
            "rows": rows,
        }

        out_dir = Path(args.output_dir) if args.output_dir else (
            RESULTS_ROOT / datetime.now().strftime("farneback_gpu_run_%Y%m%d_%H%M%S")
        )
        md_path = _write_report(out_dir, payload)

        print(
            f"\n{'method':<32} {'prep':>8} {'submit':>8} {'rlock':>8} "
            f"{'total':>8} {'fps':>8}",
            flush=True,
        )
        print("-" * 76, flush=True)
        for r in rows:
            print(
                f"{r['method']:<32} "
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
