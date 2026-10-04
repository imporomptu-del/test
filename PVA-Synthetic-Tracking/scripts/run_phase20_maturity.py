"""One frozen maturity experiment, sequential replay and post-run scoring."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tiny_target.visible_baseline import VisibleConfig
from tiny_target.visible_regression import score
from replay_phase20_tracking import run
from score_phase20_accuracy import digest, score_rows


def write(path, value):
    with path.open("x") as f:
        json.dump(value, f, indent=2)


def evaluate(path, cid, reference):
    freeze = json.loads((reference / "scoring_freeze.json").read_text())
    for name, key in (("annotations.json", "labels_sha256"),
                      ("scoring_packet.json", "packet_sha256")):
        if digest(reference / name) != freeze[key]:
            raise ValueError("Frozen reference changed")
    launch = json.loads((path / "launch.json").read_text())
    report = json.loads((path / "report.json").read_text())
    packet = json.loads((reference / "scoring_packet.json").read_text())
    source = next(s for s in packet["plan"]["sources"] if s["clip_id"] == cid)
    if not report["completed"] or launch["source_sha256"] != source["sha256"]:
        raise ValueError("Incomplete or different source")
    labels = json.loads((reference / "annotations.json").read_text())
    with (path / "frames.jsonl").open() as f:
        evaluated = score_rows(map(json.loads, f), labels, packet, cid, source["fps"])
    legacy = None
    if report["full_clip"] and cid in ("0029", "0126"):
        dev = json.loads((ROOT / "configs/evaluation/phase20_visible_development.json").read_text())
        ref = next(c for c in dev["clips"] if c["clip_id"] == "chunk" + cid)
        legacy = score(path, ROOT / ref["annotations"])
    return dict(run=str(path.resolve()), clip_id=cid, evaluation=evaluated,
                full_clip=report["full_clip"], frames=report["frames"],
                execution_mode=launch.get("execution_mode", "source_decode_and_pipeline"),
                qualified_proposal_workload=report["qualified_track_count"],
                original_anchor_regression=legacy,
                artifacts_sha256={n: digest(path / n) for n in
                                  ("frames.jsonl", "launch.json", "report.json")},
                labels_sha256=freeze["labels_sha256"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    a = p.parse_args()
    parent_root = ROOT / "results/tiny_target/phase20/shape_features_v4b_20260914"
    reference = ROOT / "results/tiny_target/phase20/encounter_accuracy_v2_20260914"
    configs = {}
    for backend in ("cpu", "pva"):
        original = parent_root / ("config.json" if backend == "cpu" else "pva_config.json")
        cfg = asdict(VisibleConfig(**json.loads(original.read_text())))
        cfg["tracking_association_prior"] = "hit_maturity"
        configs[backend] = a.root / (backend + "_config.json")
        write(configs[backend], cfg)
    write(a.root / "freeze.json", dict(
        policy_sha256=digest(a.root / "policy.md"),
        config_sha256={k: digest(v) for k, v in configs.items()},
        implementation_sha256={str(p.relative_to(ROOT)): digest(p)
                              for p in (ROOT / "tiny_target").rglob("*.py")},
        reference_sha256=digest(reference / "annotations.json"),
        scope="Tracking-only replay; no new PVA hardware execution"))
    results = []
    for backend, cid in (("cpu", "0029"), ("cpu", "0126"), ("pva", "0126")):
        path = a.root / (backend + "_" + cid)
        run(parent_root / path.name, configs[backend], path, backend == "pva")
        result = evaluate(path, cid, reference)
        write(path / "evaluation.json", result)
        results.append(result)
        print(json.dumps({"run": path.name, "windows": [
            {k: w[k] for k in ("window_id", "qualified_measured_hits", "missed_visible_frames", "ambiguity_frames", "observed_track_ids")}
            for w in result["evaluation"]["positive_windows"]]}), flush=True)
    write(a.root / "summary.json", dict(runs=results, promoted=False))


if __name__ == "__main__":
    main()
