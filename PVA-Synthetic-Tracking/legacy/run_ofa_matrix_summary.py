#!/usr/bin/env python3
"""Build combined_ofa_summary.md from a batch of bench_ofa_pair flow_report.json files."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

SKYMOVE = Path(__file__).resolve().parent
RESULTS = SKYMOVE / "results" / "ofa"

METHOD_LABELS = {
    "08_flow_676_OFA_CPU.py": "Y8_ER_BL",
    "bench_optflow.py (camera-fed, docs-style)": "NV12_ER_BL",
}

CONFIG_ORDER = [
    ("1920x1080_g4", "—", "1920×1080", 4),
    ("1920x1080_g2", "1", "1920×1080", 2),
    ("1920x1080_g1", "2", "1920×1080", 1),
    ("3184x2124_g4", "3", "3184×2124", 4),
    ("3184x2124_g1", "4", "3184×2124", 1),
    ("3184x2124_g2", "5", "3184×2124", 2),
]


def _row_from_report(report_path: Path) -> list[dict]:
    data = json.loads(report_path.read_text(encoding="utf-8"))
    cfg = data["config"]
    res = f"{cfg['width']}×{cfg['height']}"
    grid = cfg["ofa_gridsize"]
    rows = []
    for r in data["rows"]:
        method = METHOD_LABELS.get(r["script"], r["script"])
        prep = r["prep_ms"]["mean_ms"]
        submit = r.get("submit_ms", r["flow_ms"])["mean_ms"]
        rlock = r.get("rlock_ms", {"mean_ms": 0.0})["mean_ms"]
        total = r["time_spent_on_optical_flow"]["mean_ms"]
        fps = r.get("fps", {}).get("stream_fps", round(1000.0 / total, 2) if total else 0.0)
        rows.append({
            "resolution": res,
            "grid": grid,
            "method": method,
            "prep": round(prep, 1),
            "submit": round(submit, 1),
            "rlock": round(rlock, 1),
            "total": round(total, 1),
            "fps": fps,
            "report_dir": report_path.parent.name,
        })
    return rows


def _find_report(batch: str, tag: str) -> Path:
    return RESULTS / f"ofa_cam_run_{tag}_{batch}" / "flow_report.json"


def build_summary(batch: str, runs: int, warmup: int, dirs_file: Path | None) -> str:
    all_rows: list[dict] = []
    report_dirs: list[tuple[str, str]] = []

    for tag, num, res, grid in CONFIG_ORDER:
        report = _find_report(batch, tag)
        if not report.exists():
            raise FileNotFoundError(f"Missing report: {report}")
        pair_rows = _row_from_report(report)
        for row in pair_rows:
            row["num"] = num
            all_rows.append(row)
        report_dirs.append((f"{res} g{grid}", report.parent.name))

    g4_rows = [r for r in all_rows if r["resolution"] == "1920×1080" and r["grid"] == 4]

    lines = [
        "# Summary — Study 08 vs docs-style (camera-fed)",
        "",
        f"Live SkyEye62AM · downsample **1.0** · **{runs}** timed pairs · OFA only (no detection)  ",
        "`stream_fps = 1000 / total_ms`",
        "",
        f"**Source:** `{SKYMOVE}/` · `bench_ofa_pair.py` via `run_ofa_matrix.sh`  ",
        f"**Run batch:** `{batch}`",
        "",
        "---",
        "",
        "## Settings: live SkyEye62AM 1920×1080, downsample 1.0, gridsize 4×4, "
        f"{runs} runs",
        "",
        "| method | prep | submit | rlock | total | **fps** |",
        "|--------|-----:|-------:|------:|------:|--------:|",
    ]
    for r in g4_rows:
        lines.append(
            f"| **{r['method']}** | {r['prep']} | {r['submit']} | {r['rlock']} | "
            f"{r['total']} | **{r['fps']:.2f}** |"
        )
    lines.extend([
        "",
        f"Report: `{report_dirs[0][1]}/`",
        "",
        "---",
        "",
        "## Full matrix (all configs)",
        "",
        "| # | resolution | grid | method | prep | submit | rlock | total | **fps** |",
        "|---|------------|------|--------|-----:|-------:|------:|------:|--------:|",
    ])
    for r in all_rows:
        lines.append(
            f"| {r['num']} | {r['resolution']} | **{r['grid']}** | {r['method']} | "
            f"{r['prep']} | {r['submit']} | {r['rlock']} | {r['total']} | "
            f"**{r['fps']:.2f}** |"
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
        f"_Generated {datetime.now(timezone.utc).isoformat()}_",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", required=True)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--dirs-file", default="")
    args = parser.parse_args()

    dirs_file = Path(args.dirs_file) if args.dirs_file else None
    md = build_summary(args.batch, args.runs, args.warmup, dirs_file)
    out = RESULTS / "combined_ofa_summary.md"
    out.write_text(md, encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
