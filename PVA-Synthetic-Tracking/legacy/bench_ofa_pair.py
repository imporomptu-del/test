#!/usr/bin/env python3
"""Live-camera OFA-only pair bench — no detection, no masks, no video.

Opens SkyEye62AM, streams GRAY8, then runs ONLY optical-flow for:

  A) Study 08 OFA path  (flow_engines.VpiOfaFlowEngine)
       gray → Y8_ER → gaussian_pyramid(CUDA) → Y8_ER_BL(VIC) → optflow_dense(OFA)

  B) bench_optflow format path
       gray → Y8_ER → NV12_ER_BL(VIC) → optflow_dense(OFA)

Same live frames for both. No OfaCpuFlowBackend.process() (that adds detection).

Defaults: camera full ROI, downsample=1.0, gridsize=2, 50 pairs.

  source ~/optical-flow/bin/activate
  python3 bench_ofa_pair.py --runs 50 --gridsize 2 --verbose
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SKYMOVE = Path(__file__).resolve().parent
sys.path.insert(0, str(SKYMOVE))

import flow_engines as fe  # noqa: E402
from flow_engines import VpiOfaFlowEngine  # noqa: E402
from camera_bench import capture_grays, open_bench_camera  # noqa: E402

RESULTS_ROOT = SKYMOVE / "results" / "ofa"

DEFAULT_GRIDSIZE = 2
DEFAULT_RUNS = 50
DEFAULT_WARMUP = 5
NO_DOWNSAMPLE = 1.0


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


def _fps_from_ms(mean_ms: float) -> float:
    """Sustainable FPS if this stage alone limited the stream (1000 / mean_ms)."""
    if mean_ms <= 0:
        return 0.0
    return round(1000.0 / mean_ms, 2)


def _attach_fps(row: dict) -> dict:
    """Add fps for prep / submit / rlock / total (stream rate limited by that stage)."""
    prep = row["prep_ms"]["mean_ms"]
    submit = row.get("submit_ms", row["flow_ms"])["mean_ms"]
    rlock = row.get("rlock_ms", {"mean_ms": 0.0})["mean_ms"]
    total = row["time_spent_on_optical_flow"]["mean_ms"]
    row["fps"] = {
        "prep_fps": _fps_from_ms(prep),
        "submit_fps": _fps_from_ms(submit),
        "rlock_fps": _fps_from_ms(rlock),
        "stream_fps": _fps_from_ms(total),  # end-to-end pairs/s
    }
    return row


def bench_08_ofa_only(
    grays: list[np.ndarray], warmup: int, gridsize: int,
) -> dict:
    """OFA only — no detection. Splits format / submit / rlock like docs."""
    import vpi

    if fe.OFA_GRIDSIZE != gridsize:
        raise RuntimeError(f"fe.OFA_GRIDSIZE={fe.OFA_GRIDSIZE} != {gridsize}")

    engine = VpiOfaFlowEngine()
    prep_s: list[float] = []
    submit_s: list[float] = []
    rlock_s: list[float] = []
    total_s: list[float] = []

    for i in range(warmup):
        if i + 1 >= len(grays):
            break
        engine.compute_flow(grays[i], grays[i + 1])

    for i in range(warmup, len(grays) - 1):
        prev, curr = grays[i], grays[i + 1]
        t0 = time.perf_counter()
        prev_ofa = engine._to_ofa(prev)
        curr_ofa = engine._to_ofa(curr)
        prep_ms = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        with vpi.Backend.OFA:
            flow_img = vpi.optflow_dense(
                prev_ofa, curr_ofa,
                quality=vpi.OptFlowQuality.LOW, gridsize=gridsize,
            )
        t2 = time.perf_counter()
        with flow_img.rlock_cpu() as data:
            _ = np.float32(data)
        t3 = time.perf_counter()

        submit_ms = (t2 - t1) * 1000.0
        rlock_ms = (t3 - t2) * 1000.0
        prep_s.append(prep_ms)
        submit_s.append(submit_ms)
        rlock_s.append(rlock_ms)
        total_s.append(prep_ms + submit_ms + rlock_ms)

    return _attach_fps({
        "script": "08_flow_676_OFA_CPU.py",
        "method": "VpiOfaFlowEngine: Y8_ER→pyramid→Y8_ER_BL→OFA (no detection)",
        "prep_ms": _stats_ms(prep_s),
        "submit_ms": _stats_ms(submit_s),
        "rlock_ms": _stats_ms(rlock_s),
        "flow_ms": _stats_ms([a + b for a, b in zip(submit_s, rlock_s)]),
        "time_spent_on_optical_flow": _stats_ms(total_s),
        "gridsize": gridsize,
        "source": "live ASI camera",
        "detection": False,
        "note": "submit_ms ≈ docs table; flow_ms = submit+rlock; stream_fps = 1000/total_ms",
    })


# ---------------------------------------------------------------------------
# Method B — bench_optflow format path on same live gray
# ---------------------------------------------------------------------------

def bench_docs_ofa_only(
    grays: list[np.ndarray], warmup: int, gridsize: int,
) -> dict:
    """Docs-comparable NV12 path: convert once, then time submit vs rlock separately.

    Docs 2.921 ms = optflow_dense submit only on pre-built NV12_ER_BL buffers.
    Rebuilding NV12 every frame (and rlock_cpu) is NOT in that number.
    """
    import vpi

    def _to_nv12(gray: np.ndarray):
        return (
            vpi.asimage(gray, vpi.Format.Y8_ER)
            .convert(vpi.Format.NV12_ER_BL, backend=vpi.Backend.VIC)
        )

    # Pre-convert all frames once (docs: buffers created beforehand).
    t_fmt0 = time.perf_counter()
    nv12s = [_to_nv12(g) for g in grays]
    format_once_ms = (time.perf_counter() - t_fmt0) * 1000.0 / max(1, len(grays))

    submit_s: list[float] = []
    rlock_s: list[float] = []
    total_s: list[float] = []

    for i in range(warmup):
        if i + 1 >= len(nv12s):
            break
        with vpi.Backend.OFA:
            flow = vpi.optflow_dense(
                nv12s[i], nv12s[i + 1],
                quality=vpi.OptFlowQuality.LOW, gridsize=gridsize,
            )
        with flow.rlock_cpu() as data:
            _ = np.float32(data)

    for i in range(warmup, len(nv12s) - 1):
        t0 = time.perf_counter()
        with vpi.Backend.OFA:
            flow = vpi.optflow_dense(
                nv12s[i], nv12s[i + 1],
                quality=vpi.OptFlowQuality.LOW, gridsize=gridsize,
            )
        t1 = time.perf_counter()
        with flow.rlock_cpu() as data:
            _ = np.float32(data)
        t2 = time.perf_counter()
        submit_s.append((t1 - t0) * 1000.0)
        rlock_s.append((t2 - t1) * 1000.0)
        total_s.append((t2 - t0) * 1000.0)

    return _attach_fps({
        "script": "bench_optflow.py (camera-fed, docs-style)",
        "method": "prebuilt NV12_ER_BL → OFA submit (+ separate rlock)",
        "prep_ms": _stats_ms([format_once_ms]),  # one-time avg per frame
        "submit_ms": _stats_ms(submit_s),
        "rlock_ms": _stats_ms(rlock_s),
        "flow_ms": _stats_ms(submit_s),  # docs-comparable column
        "time_spent_on_optical_flow": _stats_ms(total_s),
        "gridsize": gridsize,
        "source": "live ASI camera",
        "detection": False,
        "format_once_ms_per_frame": round(format_once_ms, 3),
        "docs_reference_ms": 2.921,
        "note": "stream_fps = 1000/total_ms; submit_fps ≈ docs-comparable rate",
    })


def _write_report(out_dir: Path, payload: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "flow_report.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8",
    )
    md_path = out_dir / "flow_report.md"
    cfg = payload["config"]
    lines = [
        "# Live-camera OFA-only pair benchmark",
        "",
        f"- Generated (UTC): `{payload['generated_at_utc']}`",
        f"- **Source: live ASI camera**",
        f"- **Detection: OFF** (OFA only)",
        f"- Camera index: {cfg['camera_index']}",
        f"- Resolution: **{cfg['width']}×{cfg['height']}**",
        f"- Downsample: **{cfg['downsample']}**",
        f"- OFA gridsize: **{cfg['ofa_gridsize']}×{cfg['ofa_gridsize']}**",
        f"- Timed pairs: {cfg['runs']} (warmup {cfg['warmup']})",
        f"- Shared debayer mean: {cfg['debayer_ms_mean']:.3f} ms",
        "",
        "## Timing (OFA only — same live frames)",
        "",
        "| method | prep_ms | **submit_ms** (≈docs) | rlock_ms | total_ms | **stream_fps** |",
        "|--------|--------:|----------------------:|---------:|---------:|---------------:|",
    ]
    for r in payload["rows"]:
        sub = r.get("submit_ms", r["flow_ms"])["mean_ms"]
        rl = r.get("rlock_ms", {"mean_ms": 0.0})["mean_ms"]
        fps = r.get("fps", {})
        lines.append(
            f"| `{r['script']}` | {r['prep_ms']['mean_ms']:.3f} | "
            f"**{sub:.3f}** | {rl:.3f} | "
            f"{r['time_spent_on_optical_flow']['mean_ms']:.3f} | "
            f"**{fps.get('stream_fps', 0):.2f}** |"
        )
    lines.extend([
        "",
        "## FPS breakdown (1000 / mean_ms for that stage)",
        "",
        "| method | prep_fps | submit_fps | rlock_fps | stream_fps |",
        "|--------|---------:|-----------:|----------:|-----------:|",
    ])
    for r in payload["rows"]:
        fps = r.get("fps", {})
        lines.append(
            f"| `{r['script']}` | {fps.get('prep_fps', 0):.2f} | "
            f"{fps.get('submit_fps', 0):.2f} | {fps.get('rlock_fps', 0):.2f} | "
            f"**{fps.get('stream_fps', 0):.2f}** |"
        )
    lines.extend([
        "",
        f"Docs reference (AGX Orin): **2.921 ms** submit only "
        f"(≈ **{1000.0 / 2.921:.1f} fps** if submit alone limited the stream) — "
        f"1920×1080 NV12_ER_BL quality=LOW gridsize=4.",
        "",
        "## What is timed",
        "",
        "- **submit_ms** — `vpi.optflow_dense(OFA)` only. This is what the docs table reports.",
        "- **rlock_ms** — CPU readback of the flow grid (NOT in docs 2.921 ms).",
        "- **prep_ms** — format conversion (08: every pair; docs path: one-time NV12 build amortized).",
        "- **stream_fps** — `1000 / total_ms` (pairs/s if this pipeline ran continuously).",
        "- Docs methodology also uses max clocks (`jetson_clocks`), pre-allocated buffers, batch median.",
        "",
    ])
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live SkyEye camera → OFA only (08 vs docs format). No detection.",
    )
    parser.add_argument("--camera-id", type=int, default=None)
    parser.add_argument("--camera-name", default="SkyEye")
    parser.add_argument("--resolution-index", type=int, default=None)
    parser.add_argument(
        "--gridsize", type=int, default=DEFAULT_GRIDSIZE, choices=(1, 2, 4),
    )
    parser.add_argument("--width", type=int, default=None,
                        help="Camera ROI width (default: full sensor)")
    parser.add_argument("--height", type=int, default=None,
                        help="Camera ROI height (default: full sensor)")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("-o", "--output-dir", default="")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if (args.width is None) ^ (args.height is None):
        raise SystemExit("Pass both --width and --height, or neither")

    fe.FLOW_DEBUG_DOWNSAMPLE = NO_DOWNSAMPLE
    fe.OFA_GRIDSIZE = args.gridsize

    need = args.warmup + args.runs + 1
    print("=" * 72, flush=True)
    print("LIVE CAMERA → OFA ONLY (no detection)", flush=True)
    print(f"  gridsize   : {args.gridsize}×{args.gridsize}", flush=True)
    print(f"  downsample : {NO_DOWNSAMPLE}", flush=True)
    print(f"  runs/warmup: {args.runs}/{args.warmup}", flush=True)
    if args.width is not None:
        print(f"  ROI        : {args.width}×{args.height}", flush=True)
    print(f"  methods    : Study 08 VpiOfaFlowEngine | bench_optflow NV12 path", flush=True)
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
            f"Frame crop/copy mean: {statistics.mean(crop_ms):.3f} ms "
            f"(shared, not in method prep)",
            flush=True,
        )

        print("\n[1/2] Study 08 OFA path (no detection)…", flush=True)
        row08 = bench_08_ofa_only(grays, args.warmup, args.gridsize)

        print("[2/2] bench_optflow NV12 path on SAME frames…", flush=True)
        row_docs = bench_docs_ofa_only(grays, args.warmup, args.gridsize)

        payload = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "config": {
                "source": "live_skyeye_camera",
                "synthetic": False,
                "detection": False,
                "camera_index": session.cam_idx,
                "width": width,
                "height": height,
                "downsample": NO_DOWNSAMPLE,
                "ofa_gridsize": args.gridsize,
                "fe_ofa_gridsize": fe.OFA_GRIDSIZE,
                "runs": args.runs,
                "warmup": args.warmup,
                "frames_captured": need,
                "crop_ms_mean": round(statistics.mean(crop_ms), 3),
                "crop_ms": _stats_ms(crop_ms),
                "debayer_ms_mean": round(statistics.mean(crop_ms), 3),
                "debayer_ms": _stats_ms(crop_ms),
            },
            "rows": [row08, row_docs],
        }

        out_dir = Path(args.output_dir) if args.output_dir else (
            RESULTS_ROOT / datetime.now().strftime("ofa_cam_run_%Y%m%d_%H%M%S")
        )
        md_path = _write_report(out_dir, payload)

        print(
            f"\n{'method':<40} {'prep':>8} {'submit':>8} {'rlock':>8} "
            f"{'total':>8} {'fps':>8}",
            flush=True,
        )
        print("-" * 86, flush=True)
        for r in payload["rows"]:
            sub = r.get("submit_ms", r["flow_ms"])["mean_ms"]
            rl = r.get("rlock_ms", {"mean_ms": 0.0})["mean_ms"]
            fps = r.get("fps", {}).get("stream_fps", 0.0)
            print(
                f"{r['script']:<40} "
                f"{r['prep_ms']['mean_ms']:8.3f} "
                f"{sub:8.3f} "
                f"{rl:8.3f} "
                f"{r['time_spent_on_optical_flow']['mean_ms']:8.3f} "
                f"{fps:8.2f}",
                flush=True,
            )
        print(
            "\nstream_fps = 1000/total_ms. Docs ref submit ≈ 2.921 ms "
            f"(≈ {1000.0 / 2.921:.1f} fps if submit-only).",
            flush=True,
        )
        print(f"\nReport: {md_path}", flush=True)
    finally:
        session.close()


if __name__ == "__main__":
    main()
