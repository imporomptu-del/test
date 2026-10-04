"""Verify completed learning-guard runs and compare frozen visible references."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tiny_target.visible_baseline import VisibleConfig
from score_phase20_accuracy import digest, score_rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--experiment-root", type=Path)
    p.add_argument("--shape-only", action="store_true")
    p.add_argument(
        "--cpu-only",
        action="store_true",
        help="Require both complete CPU runs; explicitly exclude hardware validation",
    )
    p.add_argument(
        "--mode",
        choices=("background_and_variance", "variance_only"),
        default="background_and_variance",
    )
    a = p.parse_args()
    root = a.root
    freeze = json.loads((root / "scoring_freeze.json").read_text())
    if (
        digest(root / "annotations.json") != freeze["labels_sha256"]
        or digest(root / "scoring_packet.json") != freeze["packet_sha256"]
    ):
        raise ValueError("Frozen scoring inputs changed")
    labels = json.loads((root / "annotations.json").read_text())
    packet = json.loads((root / "scoring_packet.json").read_text())
    base = json.loads((root / "baseline_summary.json").read_text())
    pilot = ROOT / "results/tiny_target/phase20/accuracy_baseline_v1_20260914"
    oldlabels = json.loads((pilot / "annotations.json").read_text())
    oldpacket = json.loads((pilot / "source_review/packet.json").read_text())
    specs = [
        ("0029", "cpu_translation", root / "learning_guard_experiment/cpu_0029"),
        ("0126", "cpu_translation", root / "learning_guard_experiment/cpu_0126"),
        ("0126", "pva", root / "pva_guard_0126"),
    ]
    if a.experiment_root:
        specs = [
            (
                cid,
                backend,
                a.experiment_root / (("pva_" if backend == "pva" else "cpu_") + cid),
            )
            for cid, backend, _ in specs
        ]
    results = []
    if a.cpu_only:
        specs = [spec for spec in specs if spec[1] != "pva"]
    for cid, backend, path in specs:
        baseline = next(
            r
            for r in base["runs"]
            if r["clip_id"] == cid and r["provenance"]["backend"] == backend
        )
        launch = json.loads((path / "launch.json").read_text())
        report = json.loads((path / "report.json").read_text())
        parent = Path(baseline["provenance"]["run"])
        previous = json.loads((parent / "launch.json").read_text())
        source = next(s for s in packet["plan"]["sources"] if s["clip_id"] == cid)
        oldcfg = asdict(VisibleConfig(**previous["configuration"]))
        newcfg = asdict(VisibleConfig(**launch["configuration"]))
        changed = {k for k in oldcfg if oldcfg[k] != newcfg[k]}
        required_changes = {"learning_exclusion_radius_px"}
        if a.mode != "background_and_variance":
            required_changes.add("learning_protection_mode")
        if a.shape_only:
            required_changes = {"shape_measurement_mode"}
        expected = 230 if backend == "pva" else source["frames"]
        if (
            changed != required_changes
            or newcfg["learning_protection_mode"] != a.mode
            or newcfg["learning_exclusion_radius_px"] != (0 if a.shape_only else 6)
            or (
                a.shape_only
                and newcfg["shape_measurement_mode"] != "mutual_half_height_r8"
            )
            or launch["annotations_supplied_to_detector"]
            or not report["completed"]
            or report["frames"] != expected
            or launch["source_sha256"] != source["sha256"]
            or report["source_sha256"] != source["sha256"]
            or report["full_clip"] != (backend != "pva")
            or launch["max_frames"] != (230 if backend == "pva" else None)
        ):
            raise ValueError("Source/configuration/completion mismatch")
        for name, sha in launch["package_sha256"].items():
            if digest(path / "implementation" / name) != sha:
                raise ValueError("Implementation snapshot changed")
        if (
            backend == "pva"
            and launch["motion_config_sha256"] != previous["motion_config_sha256"]
        ):
            raise ValueError("Motion configuration changed")
        count = 0
        pva_pairs = 0

        def rows():
            nonlocal count, pva_pairs
            with (path / "frames.jsonl").open() as f:
                for row in map(json.loads, f):
                    if row["frame_index"] != count:
                        raise ValueError("Non-contiguous run")
                    if backend == "pva" and count:
                        used = row["motion"]["motion_backends"]
                        if used.get("cpu_fallback") is not False or any(
                            used.get(k) != "PVA"
                            for k in (
                                "gaussian_pyramid",
                                "harris",
                                "optical_flow_pyrlk",
                            )
                        ):
                            raise ValueError("Missing PVA hardware evidence")
                        pva_pairs += 1
                    count += 1
                    yield row
            if count != expected:
                raise ValueError("Journal length mismatch")

        evaluation = score_rows(rows(), labels, packet, cid, source["fps"])
        with (path / "frames.jsonl").open() as f:
            pilot_score = score_rows(
                map(json.loads, f), oldlabels, oldpacket, cid, source["fps"]
            )
        comparison = []
        for before, after in zip(
            baseline["positive_windows"], evaluation["positive_windows"]
        ):
            comparison.append(
                dict(
                    window_id=before["window_id"],
                    visible_frames=before["visible_samples"],
                    before_hits=before["qualified_measured_hits"],
                    after_hits=after["qualified_measured_hits"],
                    before_misses=before["missed_visible_frames"],
                    after_misses=after["missed_visible_frames"],
                    newly_missed_frames=sorted(
                        set(after["missed_visible_frames"])
                        - set(before["missed_visible_frames"])
                    ),
                    before_ambiguity_frames=before["ambiguity_frames"],
                    after_ambiguity_frames=after["ambiguity_frames"],
                    after_observed_ids=after["observed_track_ids"],
                )
            )
        parent_report = json.loads((parent / "report.json").read_text())
        results.append(
            dict(
                clip_id=cid,
                backend=backend,
                comparison=comparison,
                evaluation=evaluation,
                pilot=pilot_score,
                actual_pva_pairs=pva_pairs,
                full_clip=report["full_clip"],
                frames=expected,
                configuration_changes=sorted(changed),
                counts=report["counts"],
                availability=report["availability"],
                qualified_proposal_workload_before=parent_report[
                    "qualified_track_count"
                ],
                qualified_proposal_workload_after=report["qualified_track_count"],
                processed_fps=report["processed_fps"],
                artifacts_sha256={
                    n: digest(path / n)
                    for n in ("launch.json", "report.json", "frames.jsonl")
                },
            )
        )
    result = dict(
        runs=results,
        labels_sha256=freeze["labels_sha256"],
        airborne_accuracy_verified=False,
        hardware_validation_included=not a.cpu_only,
        experiment_kind="shape_only" if a.shape_only else "learning_protection",
        decision="Experimental, off by default; no operational promotion without nuisance and identity validation",
        no_threshold_tuning=True,
        no_sealed_media_access=True,
    )
    with a.output.open("x") as f:
        json.dump(result, f, indent=2)
    print(
        json.dumps(
            [
                {
                    k: r[k]
                    for k in (
                        "clip_id",
                        "backend",
                        "comparison",
                        "qualified_proposal_workload_before",
                        "qualified_proposal_workload_after",
                    )
                }
                for r in results
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
