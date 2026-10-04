"""Immutable-reference, exact replay and acceptance audit; never relabel data."""
import argparse
from dataclasses import asdict
import itertools
import json
from pathlib import Path
import sys
from run_phase20_maturity import evaluate, write
from score_phase20_accuracy import digest, score_rows
from audit_phase20_encounter_results import inspect_run

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tiny_target.visible_baseline import VisibleConfig


def compare(old, new):
    result = []
    for before, after in zip(old["positive_windows"], new["positive_windows"], strict=True):
        if before["window_id"] != after["window_id"]:
            raise ValueError("Window mismatch")
        result.append(dict(window_id=before["window_id"], visible_samples=after["visible_samples"],
            before_hits=before["qualified_measured_hits"], after_hits=after["qualified_measured_hits"],
            remaining_misses=after["missed_visible_frames"],
            newly_missed=sorted(set(after["missed_visible_frames"]) - set(before["missed_visible_frames"])),
            recovered=sorted(set(before["missed_visible_frames"]) - set(after["missed_visible_frames"])),
            before_ambiguity=before["ambiguity_frames"], after_ambiguity=after["ambiguity_frames"],
            measured_ids=after["observed_track_ids"],
            dominant_id_visible_fraction=after["dominant_id_visible_fraction"]))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--runs", nargs="+", choices=("cpu_0029", "cpu_0126", "pva_0126", "cpu_0055", "cpu_0082"), required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    ref = ROOT / "results/tiny_target/phase20/encounter_accuracy_v2_20260914"
    baseline = json.loads((ref / "baseline_summary.json").read_text())
    review_packet = json.loads((ref / "source_review/packet.json").read_text())
    pilot = ROOT / "results/tiny_target/phase20/accuracy_baseline_v1_20260914"
    pilot_freeze = json.loads((pilot / "scoring_freeze.json").read_text())
    if (digest(pilot / "annotations.json") != pilot_freeze["labels_sha256"]
        or digest(pilot / "source_review/packet.json") != pilot_freeze["packet_sha256"]):
        raise ValueError("Original pilot reference changed")
    pilot_labels = json.loads((pilot / "annotations.json").read_text())
    pilot_packet = json.loads((pilot / "source_review/packet.json").read_text())
    shape = ROOT / "results/tiny_target/phase20/shape_features_v4b_20260914"
    results = []
    for name in a.runs:
        path = a.root / name
        cid = name[-4:]
        backend = "pva" if name.startswith("pva") else "cpu_translation"
        scored = evaluate(path, cid, ref)
        launch = json.loads((path / "launch.json").read_text())
        report = json.loads((path / "report.json").read_text())
        replay = launch.get("source_media_decoded_in_this_run") is False
        config_name = (("pva_config.json" if backend == "pva" else "cpu_config.json")
                       if replay else ("pva_config.json" if backend == "pva" else "config.json"))
        config_path = a.root / config_name
        if digest(config_path) != launch["config_sha256"] or asdict(VisibleConfig(**json.loads(config_path.read_text()))) != asdict(VisibleConfig(**launch["configuration"])):
            raise ValueError("Run differs from the frozen experiment configuration")
        manifest_key = "code_sha256" if replay and not launch.get("complete_replay_runtime_snapshot") else "package_sha256"
        for filename, sha in launch[manifest_key].items():
            if digest(path / "implementation" / filename) != sha:
                raise ValueError("Implementation snapshot changed")
        if report["configuration"] != launch["configuration"]:
            raise ValueError("Launch/report config mismatch")
        if not replay:
            experiment = json.loads((a.root / "freeze.json").read_text())
            package = {"tiny_target/" + k: v for k, v in launch["package_sha256"].items()}
            if package != experiment["code_sha256"]:
                raise ValueError("Run used a different frozen implementation")
        expected = launch["max_frames"] or launch["expected_frames"]
        if expected != report["frames"] or report["full_clip"] != (expected == launch["expected_frames"]):
            raise ValueError("Run length mismatch")
        old = next((r for r in baseline["runs"] if r["clip_id"] == cid and r["provenance"]["backend"] == backend), None)
        if old is None:
            # The additional CPU videos were already run in frozen V7 development.
            old_path = ROOT / "results/tiny_target/phase20/v7_frozen_evaluation_20260913" / ("cpu_chunk" + cid)
            old = evaluate(old_path, cid, ref)["evaluation"]
        else:
            old_path = Path(old["provenance"]["run"])
            for filename, sha in old["provenance"]["artifacts_sha256"].items():
                if digest(old_path / filename) != sha:
                    raise ValueError("Original frozen run changed")
        previous_cfg = asdict(VisibleConfig(**json.loads((old_path / "launch.json").read_text())["configuration"]))
        cfg = asdict(VisibleConfig(**launch["configuration"]))
        changes = {k: [previous_cfg[k], cfg[k]] for k in cfg if cfg[k] != previous_cfg[k]}
        allowed = {"shape_measurement_mode", "tracking_association_prior", "tracking_association_appearance", "learning_protection_mode", "learning_exclusion_radius_px", "learning_protection_geometry"}
        if set(changes) - allowed:
            raise ValueError(f"Unapproved configuration differences: {changes}")
        count, pva_pairs = 0, 0
        with (path / "frames.jsonl").open() as f:
            for row in map(json.loads, f):
                if row["frame_index"] != count:
                    raise ValueError("Non-contiguous journal")
                if backend == "pva" and count:
                    used = row["motion"]["motion_backends"]
                    if used.get("cpu_fallback") is not False or any(used.get(k) != "PVA" for k in ("gaussian_pyramid", "harris", "optical_flow_pyrlk")):
                        raise ValueError("PVA backend evidence missing")
                    pva_pairs += 1
                count += 1
        if count != expected:
            raise ValueError("Journal/report length mismatch")
        identical_replay_frames = None
        if replay:
            parent = Path(launch["parent_run"])
            if digest(parent / "frames.jsonl") != launch["parent_journal_sha256"]:
                raise ValueError("Replay parent changed")
            identical_replay_frames = 0
            with (parent / "frames.jsonl").open() as x, (path / "frames.jsonl").open() as y:
                for left, right in itertools.zip_longest(x, y):
                    if left is None or right is None:
                        raise ValueError("Replay length differs")
                    left, right = json.loads(left), json.loads(right)
                    for key in ("tracks", "tracking_metrics", "timings_ms"):
                        left.pop(key); right.pop(key)
                    if left != right:
                        raise ValueError("Replay changed original detection/motion evidence")
                    identical_replay_frames += 1
        shape_comparison = None
        if cid in ("0029", "0126"):
            previous = evaluate(shape / name, cid, ref)
            shape_comparison = compare(previous["evaluation"], scored["evaluation"])
        episode = next((e for e in review_packet["episodes"] if e["id"] == cid + "_lights"), None)
        diagnostic = inspect_run(path, scored["evaluation"]["positive_windows"], episode)
        baseline_diagnostic = inspect_run(old_path, old["positive_windows"], episode) if episode else None
        comparison = compare(old, scored["evaluation"])
        pilot_scores = []
        for target in (old_path, path):
            with (target / "frames.jsonl").open() as f:
                pilot_scores.append(score_rows(map(json.loads, f), pilot_labels, pilot_packet, cid, launch["fps"]))
        pilot_comparison = compare(*pilot_scores)
        anchors = scored["original_anchor_regression"]
        anchors_pass = all(e["dominant_track_anchor_hits"] == e["required_anchor_count"] for e in anchors["events"]) if anchors else None
        gate = all(not c["newly_missed"] and c["after_ambiguity"] <= c["before_ambiguity"] for c in comparison)
        if shape_comparison:
            gate &= all(not c["newly_missed"] and c["after_ambiguity"] <= c["before_ambiguity"] for c in shape_comparison)
        gate &= anchors_pass is not False
        gate &= all(not c["newly_missed"] for c in pilot_comparison)
        results.append(dict(**scored, configuration_changes=changes,
            comparison_to_original= comparison, comparison_to_shape=shape_comparison,
            original_pilot_comparison=pilot_comparison,
            full_clip_anchor_identity_pass=anchors_pass, reviewed_regression_gate_pass=gate if comparison else None,
            diagnostics=diagnostic, identical_replay_detection_motion_frames=identical_replay_frames,
            baseline_diagnostics=baseline_diagnostic,
            actual_pva_pairs=0 if replay else pva_pairs, inherited_pva_pairs=pva_pairs if replay else 0,
            original_qualified_workload=json.loads((old_path / "report.json").read_text())["qualified_track_count"],
            timings_are_replay_only=replay, processed_fps=report["processed_fps"]))
    frozen = ROOT / "results/tiny_target/phase20/v8c_motion_fix_20260913"
    manifest = json.loads((frozen / "freeze.json").read_text())
    for name, sha in manifest["files_sha256"].items():
        if "phase18" in name or digest(frozen / "snapshot" / name) != sha:
            raise ValueError("Forbidden or changed frozen file")
    record = dict(runs=results, historical_snapshot_files_intact=len(manifest["files_sha256"]),
        labels_sha256=digest(ref / "annotations.json"), promoted=False,
        pilot_labels_sha256=pilot_freeze["labels_sha256"],
        airborne_accuracy_verified=False, verified_negative_intervals=0,
        defaults_off=VisibleConfig().tracking_association_prior == "none")
    write(a.output, record)
    print(json.dumps([{k: r[k] for k in ("run", "comparison_to_original", "comparison_to_shape", "full_clip_anchor_identity_pass", "reviewed_regression_gate_pass", "qualified_proposal_workload", "actual_pva_pairs")} for r in results], indent=2))


if __name__ == "__main__":
    main()
