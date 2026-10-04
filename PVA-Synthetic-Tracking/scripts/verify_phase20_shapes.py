"""Verify shape-only runs preserve original threshold seeds and frozen truth."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tiny_target.visible_baseline import VisibleConfig
from tiny_target.visible_regression import score
from score_phase20_accuracy import digest
from diagnose_phase20_v3_tracks import inspect


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--summary", type=Path)
    p.add_argument("--diagnostics-output", type=Path)
    a = p.parse_args()
    summary = a.summary or a.root / "verified_summary.json"
    diagnostics_output = (
        a.diagnostics_output or a.root / "remaining_failure_diagnostics.json"
    )
    if a.output.exists() or diagnostics_output.exists():
        raise FileExistsError("Verification outputs must be new files")
    comparison = json.loads(summary.read_text())
    reference = ROOT / "results/tiny_target/phase20/encounter_accuracy_v2_20260914"
    if (
        digest(reference / "annotations.json")
        != "56b98b77b8acdfabc564978b80203ca0e9a698bffbca9c8d44e46ff886e67fed"
    ):
        raise ValueError("Frozen labels changed")
    specs = json.loads((reference / "scoring_freeze.json").read_text())["runs"]
    frozen_scores = json.loads((reference / "baseline_summary.json").read_text())[
        "runs"
    ]
    dev = json.loads(
        (ROOT / "configs/evaluation/phase20_visible_development.json").read_text()
    )
    results = []
    diagnostics = []
    for result in comparison["runs"]:
        cid, backend = result["clip_id"], result["backend"]
        source = next(
            s for s in specs if s["clip_id"] == cid and s["backend"] == backend
        )
        before = Path(source["run"])
        provenance = next(
            r["provenance"]
            for r in frozen_scores
            if r["clip_id"] == cid and r["provenance"]["backend"] == backend
        )
        for name, sha in provenance["artifacts_sha256"].items():
            if digest(before / name) != sha:
                raise ValueError("Frozen baseline artifact changed")
        after = a.root / (("pva_" if backend == "pva" else "cpu_") + cid)
        counts = Counter()
        with (before / "frames.jsonl").open() as old_file, (
            after / "frames.jsonl"
        ).open() as new_file:
            for old_line, new_line in zip(old_file, new_file):
                old, new = json.loads(old_line), json.loads(new_line)
                if old["frame_index"] != new["frame_index"]:
                    raise ValueError("Frame mismatch")
                seeds = Counter(
                    (c["x"], c["y"], c["polarity"]) for c in old["candidates"]
                )
                old_peaks = {
                    (c["x"], c["y"], c["polarity"]): c for c in old["candidates"]
                }
                recovered = Counter()
                for c in new["candidates"]:
                    shape = c.get("shape")
                    members = (
                        shape["member_peak_reference_xy"]
                        if shape
                        else [[c["x"], c["y"]]]
                    )
                    recovered.update((x, y, c["polarity"]) for x, y in members)
                    peak = shape["peak_reference_xy"] if shape else [c["x"], c["y"]]
                    original = old_peaks.get((*peak, c["polarity"]))
                    if original is None or any(
                        c[k] != original[k]
                        for k in ("score", "response_dn", "noise_sigma_dn")
                    ):
                        raise ValueError("Changed accepted-peak evidence")
                if seeds != recovered:
                    raise ValueError("Shape pass lost or invented threshold seeds")
                metric = new["coverage"]["shape_measurement"]
                if (
                    metric["input_peaks"] - metric["output_features"]
                    != metric["merged_peak_count"]
                ):
                    raise ValueError("Shape accounting mismatch")
                counts.update(
                    frames=1,
                    original_peaks=len(old["candidates"]),
                    output_features=len(new["candidates"]),
                    merged_peaks=metric["merged_peak_count"],
                    unchanged_shape_rejection_peaks=metric[
                        "unmodified_unbounded_or_unsupported_peaks"
                    ],
                )
        if counts["frames"] != result["frames"]:
            raise ValueError("Incomplete seed verification")
        anchors = []
        if result["full_clip"]:
            ref = next(c for c in dev["clips"] if c["clip_id"] == "chunk" + cid)
            old_score, new_score = (
                score(before, ROOT / ref["annotations"]),
                score(after, ROOT / ref["annotations"]),
            )
            for old, new in zip(old_score["events"], new_score["events"]):
                anchors.append(
                    dict(
                        event=old["event_id"],
                        required=old["required_anchor_count"],
                        baseline_dominant_hits=old["dominant_track_anchor_hits"],
                        shape_dominant_hits=new["dominant_track_anchor_hits"],
                    )
                )
        results.append(
            dict(
                clip_id=cid,
                backend=backend,
                shape_counts=dict(counts),
                original_threshold_seeds_and_peak_evidence_identical=True,
                original_anchors=anchors,
            )
        )
        old_result = next(
            r
            for r in frozen_scores
            if r["clip_id"] == cid and r["provenance"]["backend"] == backend
        )
        wanted = {}
        for old_window, window in zip(
            old_result["positive_windows"], result["evaluation"]["positive_windows"]
        ):
            frequencies = Counter(
                e["assigned_track_id"]
                for e in window["evidence"]
                if e["qualified_measured_hit"]
            )
            dominant = frequencies.most_common(1)[0][0] if frequencies else None
            selected = set(old_window["missed_visible_frames"]) | set(
                window["missed_visible_frames"]
            )
            selected.update(
                e["frame_index"]
                for e in window["evidence"]
                if e["qualified_measured_hit"] and e["assigned_track_id"] != dominant
            )
            for e in window["evidence"]:
                if e["frame_index"] in selected or e["frame_index"] + 1 in selected:
                    wanted[e["frame_index"]] = e["reference_xy"]
        diagnostics.append(
            dict(
                clip_id=cid,
                backend=backend,
                baseline=inspect(before, wanted),
                shape=inspect(after, wanted),
            )
        )
    frozen = ROOT / "results/tiny_target/phase20/v8c_motion_fix_20260913"
    manifest = json.loads((frozen / "freeze.json").read_text())
    for name, sha in manifest["files_sha256"].items():
        if "phase18" in name:
            raise ValueError("Sealed split access forbidden")
        if digest(frozen / "snapshot" / name) != sha:
            raise ValueError("Frozen implementation changed")
    if VisibleConfig().shape_measurement_mode != "none":
        raise ValueError("Shape policy unexpectedly enabled by default")
    tests = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests/unit"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    match = re.search(r"Ran (\d+) tests", tests.stderr)
    if tests.returncode or not match:
        raise RuntimeError(tests.stdout + tests.stderr)
    record = dict(
        runs=results,
        tests_passed=int(match.group(1)),
        test_stdout=tests.stdout,
        test_stderr=tests.stderr,
        frozen_v8c_files_intact=len(manifest["files_sha256"]),
        labels_sha256=comparison["labels_sha256"],
        no_new_visible_misses=all(
            not c["newly_missed_frames"]
            for r in comparison["runs"]
            for c in r["comparison"]
        ),
        hardware_validation_included=comparison["hardware_validation_included"],
        defaults_unchanged=True,
        promoted=False,
        fresh_nuisance_runs_completed=False,
        airborne_accuracy_verified=False,
    )
    with a.output.open("x") as f:
        json.dump(record, f, indent=2)
    with diagnostics_output.open("x") as f:
        json.dump(
            dict(
                scope="Post-run frozen-reference diagnostics only; no detector inputs",
                runs=diagnostics,
            ),
            f,
            indent=2,
        )
    print(
        json.dumps(
            {
                k: v
                for k, v in record.items()
                if k not in ("test_stdout", "test_stderr")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
