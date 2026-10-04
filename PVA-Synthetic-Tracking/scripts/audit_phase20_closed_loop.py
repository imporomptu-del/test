"""Post-run frozen-reference scoring; audited samples are not airborne truth."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tiny_target.visible_baseline import sha256
from run_phase20_maturity import evaluate
from score_phase20_accuracy import score_rows
from audit_phase20_encounter_results import inspect_run


def verify_run(path, frozen, cid):
    launch = json.loads((path / "launch.json").read_text())
    report = json.loads((path / "report.json").read_text())
    source = frozen["sources"][cid]
    if not report["completed"] or not report["full_clip"] or report["frames"] != source["frames"]:
        raise ValueError("Incomplete or wrong-scope run")
    if launch["max_frames"] is not None or launch["fps"] != 10 or launch["annotations_supplied_to_detector"]:
        raise ValueError("Cadence, scope or label boundary changed")
    if launch["source_sha256"] != source["sha256"] or report["source_sha256"] != source["sha256"]:
        raise ValueError("Source changed")
    if launch["config_sha256"] != frozen["config_sha256"]:
        raise ValueError("Configuration differs from freeze")
    package = {n.removeprefix("tiny_target/"): v for n, v in frozen["files_sha256"].items()
               if n.startswith("tiny_target/")}
    if launch["package_sha256"] != package:
        raise ValueError("Runtime differs from freeze")
    for name, digest in package.items():
        if sha256(path / "implementation" / name) != digest:
            raise ValueError("Implementation snapshot changed")
    if launch["motion_config_sha256"] != frozen["files_sha256"]["configs/evaluation/phase20_motion_v8.json"]:
        raise ValueError("Motion configuration changed")
    cuda = launch["exact_cuda_stabilization"]
    if cuda["library_sha256"] != frozen["compiled_library_sha256"]:
        raise ValueError("CUDA library changed")
    if not cuda["conformance"]["exact"] or cuda["conformance"]["warp_cases"] != 32 or cuda["conformance"]["gaussian_cases"] != 33:
        raise ValueError("CUDA startup numerical gate missing")
    counts = dict(frames=0, actual_pva_pairs=0, ready_frames=0, failures=0, resets=0)
    with (path / "frames.jsonl").open() as handle:
        for row in map(json.loads, handle):
            i = counts["frames"]
            if row["frame_index"] != i or row["timestamp_ns"] != i*100_000_000:
                raise ValueError("Noncontiguous journal or wrong cadence")
            coverage = row["coverage"]
            if coverage["full_shape_hw"] != [3190, 4784] or coverage["configured_crop"] is not None or coverage["native_pixel_sampling"] is not True:
                raise ValueError("Native full-frame coverage changed")
            motion = row["motion"]
            counts["failures"] += bool(motion["pva_failure"])
            counts["resets"] += bool(motion["reset"])
            counts["ready_frames"] += not coverage["warmup"]
            if i and not motion["pva_failure"]:
                backends = motion["motion_backends"]
                if backends.get("cpu_fallback") is not False or any(backends.get(k) != "PVA"
                        for k in ("gaussian_pyramid", "harris", "optical_flow_pyrlk")):
                    raise ValueError("Actual PVA provenance missing")
                counts["actual_pva_pairs"] += 1
            counts["frames"] += 1
    if counts["frames"] != source["frames"]:
        raise ValueError("Truncated journal")
    return launch, report, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clips", nargs="+", choices=("0029", "0126", "0055", "0082"),
                        default=["0029", "0126", "0055", "0082"])
    args = parser.parse_args()
    if args.output.exists() or len(args.clips) != len(set(args.clips)):
        raise ValueError("Existing output or duplicate clip")
    frozen = json.loads((args.root / "freeze.json").read_text())
    if set(frozen["sources"]) != {"0029", "0126", "0055", "0082"}:
        raise ValueError("Unexpected source scope")
    if sha256(args.root / "config.json") != frozen["config_sha256"]:
        raise ValueError("Frozen configuration changed")
    complete = len(args.clips) == 4
    if complete:
        status = json.loads((args.root / "status.json").read_text())
        if status["running"] or status["error"] or [r["name"] for r in status["completed"]] != [j["name"] for j in frozen["jobs"]]:
            raise ValueError("Batch not complete")
    reference = ROOT / "results/tiny_target/phase20/encounter_accuracy_v2_20260914"
    packet = json.loads((reference / "scoring_packet.json").read_text())
    pilot = ROOT / "results/tiny_target/phase20/accuracy_baseline_v1_20260914"
    pf = json.loads((pilot / "scoring_freeze.json").read_text())
    for name, key in (("annotations.json", "labels_sha256"), ("source_review/packet.json", "packet_sha256")):
        if sha256(pilot / name) != pf[key]:
            raise ValueError("Pilot reference changed")
    runs = []
    for cid in args.clips:
        path = args.root / ("pva_" + cid)
        launch, report, counts = verify_run(path, frozen, cid)
        scored = evaluate(path, cid, reference)
        with (path / "frames.jsonl").open() as handle:
            ps = score_rows(map(json.loads, handle), json.loads((pilot / "annotations.json").read_text()),
                json.loads((pilot / "source_review/packet.json").read_text()), cid, launch["fps"])
        windows = scored["evaluation"]["positive_windows"]
        if sum(w["visible_samples"] for w in windows) != {"0029": 146, "0126": 139, "0055": 0, "0082": 0}[cid]:
            raise ValueError("Frozen visible-reference denominator changed")
        light = next((w for w in packet["windows"] if w["clip_id"] == cid and w["id"].endswith("_lights")), None)
        region = None if light is None else dict(first=light["first"], last=light["last"], fixed_crop=light["crop_xywh"])
        runs.append(dict(clip_id=cid, frames=report["frames"], fps=report["processed_fps"],
            elapsed_seconds=report["elapsed_seconds"], timings_ms=report["timings_ms"],
            qualified_proposal_workload=report["qualified_track_count"], counts=counts,
            coverage_loss_counts=report["counts"], scored=scored, pilot=ps,
            diagnostic=inspect_run(path, windows, region)))
    all_samples = [w for r in runs for w in r["scored"]["evaluation"]["positive_windows"]]
    strict = complete and all(w["qualified_measured_hits"] == w["visible_samples"] and
        w["ambiguity_frames"] == 0 and len(w["observed_track_ids"]) == 1 for w in all_samples)
    result = dict(schema="seaqr.closed-loop-development-audit.v1", complete=complete, runs=runs,
        strict_dense_sample_continuity_gate=strict,
        timing_caveat="Exploratory sequential run, not controlled execution-only benchmark; completed-journal transfer overlapped remaining control clips.",
        generalization_proven=False, production_ready=False, holdouts_accessed=False,
        airborne_precision=None, airborne_recall=None, full_frame_false_alarm_rate=None,
        labels_used_only_after_processing=True, predictions_count_as_measurements=False,
        workload_is_unlabeled=True, freeze_sha256=sha256(args.root / "freeze.json"), analyzer_sha256=sha256(__file__))
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(dict(complete=complete, strict_dense_sample_continuity_gate=strict,
        runs=[dict(clip_id=r["clip_id"], workload=r["qualified_proposal_workload"],
            windows=[{k: w[k] for k in ("window_id", "visible_samples", "qualified_measured_hits",
                "missed_visible_frames", "ambiguity_frames", "observed_track_ids")} for w in r["scored"]["evaluation"]["positive_windows"]],
            pilot=[{k: w[k] for k in ("window_id", "visible_samples", "qualified_measured_hits", "missed_visible_frames", "ambiguity_frames")}
                   for w in r["pilot"]["positive_windows"]]) for r in runs]), indent=2))


if __name__ == "__main__":
    main()
