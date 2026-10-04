#!/usr/bin/env python3
"""Build combined 01/02 + Farneback GPU summary table from flow_report.json files."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

SKYMOVE = Path(__file__).resolve().parent
RESULTS = SKYMOVE / "results" / "ofa"
OUT_DIR = RESULTS / "combined_01_02_report"


def _method_label(row: dict) -> str:
    method = row.get("method", "")
    if "nvof" in method.lower():
        return "nvof"
    if "gpu" in method.lower():
        return "Farneback GPU"
    if "cpu" in method.lower() or "farneback" in method.lower():
        return "Farneback CPU"
    return method


def _row_from_report(report_path: Path, *, want: set[str] | None = None) -> list[dict]:
    data = json.loads(report_path.read_text(encoding="utf-8"))
    cfg = data["config"]
    res = f"{cfg['width']}×{cfg['height']}"
    out: list[dict] = []
    for r in data["rows"]:
        label = _method_label(r)
        if want is not None and label not in want:
            continue
        prep = r["prep_ms"]["mean_ms"]
        submit = r["submit_ms"]["mean_ms"]
        rlock = r["rlock_ms"]["mean_ms"]
        total = r["total_ms"]["mean_ms"]
        fps = r.get("fps", round(1000.0 / total, 2) if total else 0.0)
        grid = r.get("gridsize", "n/a")
        if grid in ("~4×4", "~4"):
            grid = "~4"
        out.append({
            "resolution": res,
            "grid": grid,
            "method": label,
            "prep": round(prep, 1),
            "submit": round(submit, 1),
            "rlock": round(rlock, 1),
            "total": round(total, 1),
            "fps": round(float(fps), 1),
            "report_dir": report_path.parent.name,
            "script": r.get("script", ""),
        })
    return out


def build_summary(
    *,
    batch: str,
    runs: int,
    warmup: int,
    flow_01_02_1080: Path,
    flow_01_02_full: Path,
    farneback_gpu_1080: Path,
    farneback_gpu_full: Path,
) -> tuple[str, dict]:
    rows_1080_01 = _row_from_report(flow_01_02_1080)
    rows_full_01 = _row_from_report(flow_01_02_full)
    rows_1080_gpu = _row_from_report(farneback_gpu_1080, want={"Farneback GPU"})
    rows_full_gpu = _row_from_report(farneback_gpu_full, want={"Farneback GPU"})

    def _pick(rows: list[dict], method: str) -> dict:
        for r in rows:
            if r["method"] == method:
                return r
        raise KeyError(f"{method} not found in {[r['method'] for r in rows]}")

    matrix = [
        {**_pick(rows_1080_01, "Farneback CPU"), "num": 1},
        {**rows_1080_gpu[0], "num": 1},
        {**_pick(rows_1080_01, "nvof"), "num": 1},
        {**_pick(rows_full_01, "Farneback CPU"), "num": 2},
        {**rows_full_gpu[0], "num": 2},
        {**_pick(rows_full_01, "nvof"), "num": 2},
    ]

    report_dirs = [
        ("1920×1080 Farneback CPU + nvof", flow_01_02_1080.parent.name),
        ("3184×2124 Farneback CPU + nvof", flow_01_02_full.parent.name),
        ("1920×1080 Farneback GPU", farneback_gpu_1080.parent.name),
        ("3184×2124 Farneback GPU", farneback_gpu_full.parent.name),
    ]

    lines = [
        "# Summary — Farneback CPU / Farneback GPU / nvof (flow only)",
        "",
        f"Live SkyEye62AM · downsample **1.0** · **{runs}** timed pairs · flow only (no detection)  ",
        "`stream_fps = 1000 / total_ms`",
        "",
        f"**Source:** `{SKYMOVE}/` · `bench_01_02_flow.py` + `bench_01_farneback_gpu.py`  ",
        f"**Run batch:** `{batch}`",
        "",
        "---",
        "",
        "## Full matrix (all methods)",
        "",
        "| # | resolution | grid | method | prep | submit | rlock | total | **fps** |",
        "|---|------------|------|--------|-----:|-------:|------:|------:|--------:|",
    ]
    for r in matrix:
        lines.append(
            f"| {r['num']} | {r['resolution']} | {r['grid']} | {r['method']} | "
            f"{r['prep']} | {r['submit']} | {r['rlock']} | {r['total']} | "
            f"**{r['fps']:.1f}** |"
        )

    lines.extend([
        "",
        "### Per-run reports",
        "",
        "| config | report dir |",
        "|--------|------------|",
    ])
    for label, dirname in report_dirs:
        lines.append(f"| {label} | `{dirname}/` |")

    lines.extend([
        "",
        "### Notes",
        "",
        "- GPU **prep** = upload (resize if needed); **submit** = CUDA Farneback; **rlock** = download.",
        "- nvof **grid** = ~4×4 fixed HW block (not VPI gridsize).",
        "",
        f"_Generated {datetime.now(timezone.utc).isoformat()}_",
        "",
    ])

    payload = {
        "title": "Farneback CPU / Farneback GPU / nvof (flow only)",
        "batch": batch,
        "generated_from": [d for _, d in report_dirs],
        "config": {
            "source": "live_skyeye_camera",
            "downsample": 1.0,
            "runs": runs,
            "warmup": warmup,
            "stream_fps_formula": "1000 / total_ms",
        },
        "rows": matrix,
    }
    return "\n".join(lines), payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", required=True)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--flow-01-02-1080", required=True)
    parser.add_argument("--flow-01-02-full", required=True)
    parser.add_argument("--farneback-gpu-1080", required=True)
    parser.add_argument("--farneback-gpu-full", required=True)
    parser.add_argument("--manifest", default="")
    args = parser.parse_args()

    md, payload = build_summary(
        batch=args.batch,
        runs=args.runs,
        warmup=args.warmup,
        flow_01_02_1080=Path(args.flow_01_02_1080) / "flow_report.json",
        flow_01_02_full=Path(args.flow_01_02_full) / "flow_report.json",
        farneback_gpu_1080=Path(args.farneback_gpu_1080) / "flow_report.json",
        farneback_gpu_full=Path(args.farneback_gpu_full) / "flow_report.json",
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    md_path = OUT_DIR / "combined_report.md"
    json_path = OUT_DIR / "combined_report.json"
    md_path.write_text(md, encoding="utf-8")
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if args.manifest:
        manifest = {
            "batch": args.batch,
            "reports": {
                "flow_01_02_1080": args.flow_01_02_1080,
                "flow_01_02_full": args.flow_01_02_full,
                "farneback_gpu_1080": args.farneback_gpu_1080,
                "farneback_gpu_full": args.farneback_gpu_full,
            },
            "combined_md": str(md_path),
        }
        Path(args.manifest).write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote {md_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
