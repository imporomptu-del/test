"""Validate complete V8 development evidence without manufacturing accuracy labels."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tiny_target.visible_baseline import sha256
from tiny_target.visible_coverage import DetectionAvailability
from tiny_target.visible_regression import score


def load_run(path, spec, freeze):
    launch = json.loads((path / "launch.json").read_text())
    report = json.loads((path / "report.json").read_text())
    source = next(s for s in freeze["sources"] if s["clip_id"] == spec["clip"])
    if (
        not report["completed"]
        or report["source_sha256"] != source["sha256"]
        or launch["source_sha256"] != source["sha256"]
    ):
        raise ValueError("Incomplete run or source mismatch")
    expected = launch["expected_frames"] if spec["full_clip"] else spec["max_frames"]
    if (
        report["frames"] != expected
        or report["full_clip"] != spec["full_clip"]
        or launch["max_frames"] != spec.get("max_frames")
    ):
        raise ValueError("Full/prefix scope mismatch")
    package = {
        k.removeprefix("tiny_target/"): v
        for k, v in freeze["files_sha256"].items()
        if k.startswith("tiny_target/")
    }
    if launch["package_sha256"] != package:
        raise ValueError("Frozen package mismatch")
    cfg = (
        "configs/evaluation/phase20_visible_v7"
        + ("_pva" if spec["backend"] == "pva" else "")
        + ".json"
    )
    if launch["config_sha256"] != freeze["files_sha256"][cfg]:
        raise ValueError("Visible settings changed")
    if (
        spec["backend"] == "pva"
        and launch["motion_config_sha256"]
        != freeze["files_sha256"]["configs/evaluation/phase20_motion_v8.json"]
    ):
        raise ValueError("Motion settings changed")
    availability = DetectionAvailability()
    frames = 0
    with (path / "frames.jsonl").open() as f:
        for row in map(json.loads, f):
            if row["frame_index"] != frames:
                raise ValueError("Noncontiguous journal")
            availability.update(row["coverage"], row["motion"])
            frames += 1
    if frames != expected or availability.report() != report["availability"]:
        raise ValueError("Report/journal availability mismatch")
    return dict(
        clip=spec["clip"],
        backend=spec["backend"],
        run=str(path.resolve()),
        full_clip=spec["full_clip"],
        frames=frames,
        availability=availability.report(),
        qualified_proposals=report["qualified_track_count"],
        capacity_counts=report["counts"],
        processed_fps=report["processed_fps"],
        source_sha256=source["sha256"],
        journal_sha256=sha256(path / "frames.jsonl"),
        report_sha256=sha256(path / "report.json"),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    freeze = json.loads((a.root / "freeze.json").read_text())
    for name, digest in freeze["files_sha256"].items():
        if (
            sha256(a.root / "snapshot" / name) != digest
            or sha256(ROOT / name) != digest
        ):
            raise ValueError(f"Frozen implementation changed: {name}")
    reference = freeze.get("cpu_regression_reference")
    cpu_freeze = freeze
    if reference:
        cpu_root = Path(reference["root"])
        if sha256(cpu_root / "freeze.json") != reference["freeze_sha256"]:
            raise ValueError("CPU reference freeze changed")
        cpu_freeze = json.loads((cpu_root / "freeze.json").read_text())
        differences = {
            k
            for k in freeze["files_sha256"]
            if k.startswith("tiny_target/")
            and freeze["files_sha256"][k] != cpu_freeze["files_sha256"].get(k)
        }
        if differences != {"tiny_target/motion/translation_support.py"}:
            raise ValueError("CPU reuse is no longer justified")
    development = json.loads(
        (ROOT / "configs/evaluation/phase20_visible_development.json").read_text()
    )
    runs = []
    for spec in freeze["planned_runs"]:
        prefix = "pva" if spec["backend"] == "pva" else "cpu"
        path = Path(spec.get("reference_run", a.root / f'{prefix}_chunk{spec["clip"]}'))
        result = load_run(path, spec, freeze if prefix == "pva" else cpu_freeze)
        if prefix == "cpu":
            clip = next(
                c
                for c in development["clips"]
                if c["clip_id"] == "chunk" + spec["clip"]
            )
            result["anchor_regression"] = score(path, ROOT / clip["annotations"])
            result["v7_behavior_comparison"] = json.loads(
                (path / "v7_behavior_comparison.json").read_text()
            )
        elif spec["clip"] == "0126":
            result["prefix_anchor_diagnostics"] = json.loads(
                (path / "anchor_diagnostics.json").read_text()
            )
        runs.append(result)
    cpu_events = [
        e
        for r in runs
        if r["backend"] == "cpu_translation"
        for e in r["anchor_regression"]["events"]
    ]
    pva_events = next(r for r in runs if r["backend"] == "pva" and r["clip"] == "0126")[
        "prefix_anchor_diagnostics"
    ]["sparse_anchor_results"]
    pva027 = next(r for r in runs if r["backend"] == "pva" and r["clip"] == "0027")
    gates = dict(
        pva027_detection_coverage_restored=pva027["availability"][
            "usable_detection_coverage"
        ],
        no_pva_runtime_errors=all(
            r["availability"]["counts"]["pva_runtime_errors"] == 0
            for r in runs
            if r["backend"] == "pva"
        ),
        all_24_cpu_confident_anchors_retained=sum(
            e["required_anchor_count"] for e in cpu_events
        )
        == 24
        and all(
            e["dominant_track_anchor_hits"] == e["required_anchor_count"]
            for e in cpu_events
        ),
        all_6_pva_prefix_confident_anchors_retained=sum(
            e["required_anchor_count"] for e in pva_events
        )
        == 6
        and all(e["all_required_anchors_same_measured_id"] for e in pva_events),
        cpu_behavior_exactly_matches_v7=all(
            r["v7_behavior_comparison"]["exact_cpu_behavior_match"]
            for r in runs
            if r["backend"] == "cpu_translation"
        ),
    )
    result = dict(
        scope="Development motion-fix validation; 027 is unlabeled and 126 hardware evidence is a prefix, not a full clip",
        freeze_sha256=sha256(a.root / "freeze.json"),
        gates=gates,
        all_run_gates_pass=all(gates.values()),
        runs=runs,
        generalization_proven=False,
        precision=None,
        recall=None,
        false_positives_per_minute=None,
        cpu_reuse=reference,
        sealed_holdout_accessed=False,
    )
    extra_path = a.root / "additional_motion_pairs.json"
    if extra_path.exists():
        extra = json.loads(extra_path.read_text())
        for key, name in (
            ("config_sha256", "configs/evaluation/phase20_motion_v8.json"),
            ("global_motion_sha256", "tiny_target/motion/global_motion.py"),
            ("sparse_validator_sha256", "tiny_target/motion/translation_support.py"),
        ):
            if extra[key] != freeze["files_sha256"][name]:
                raise ValueError(
                    "Additional diagnostic did not use the frozen candidate"
                )
        result["additional_motion_only_checks"] = extra
    with a.output.open("x") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(dict(gates=gates, output=str(a.output)), indent=2))
    if not result["all_run_gates_pass"]:
        raise SystemExit(2)
