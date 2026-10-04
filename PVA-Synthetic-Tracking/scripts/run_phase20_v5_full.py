"""Fresh full-source V5 CPU evaluations; explicit development allowlist only."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from dataclasses import asdict
from run_phase20_maturity import evaluate, write
from score_phase20_accuracy import digest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tiny_target.visible_baseline import VisibleConfig


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--clips", nargs="+", choices=("0029", "0126", "0055", "0082"), required=True)
    p.add_argument("--initialize", action="store_true")
    p.add_argument("--variance-protection", action="store_true")
    p.add_argument("--shape-protection", action="store_true")
    p.add_argument("--appearance", action="store_true")
    a = p.parse_args()
    if (a.variance_protection or a.shape_protection or a.appearance) and not a.initialize:
        raise ValueError("Continuation must use the already frozen configuration")
    if a.variance_protection and a.shape_protection:
        raise ValueError("Choose one learning geometry")
    reference = ROOT / "results/tiny_target/phase20/encounter_accuracy_v2_20260914"
    packet = json.loads((reference / "scoring_packet.json").read_text())
    cp = a.root / "config.json"
    if a.initialize:
        a.root.mkdir(exist_ok=False)
        base = ROOT / "results/tiny_target/phase20/maturity_v5_20260914/cpu_config.json"
        cfg = json.loads(base.read_text())
        if a.appearance:
            cfg["tracking_association_appearance"] = "log_response"
        if a.variance_protection:
            cfg.update(learning_exclusion_radius_px=6, learning_protection_mode="variance_only")
        if a.shape_protection:
            cfg.update(learning_exclusion_radius_px=0, learning_protection_mode="variance_only",
                       learning_protection_geometry="observed_shape")
        write(cp, asdict(VisibleConfig(**cfg)))
        write(a.root / "freeze.json", dict(
            config_sha256=digest(cp),
            code_sha256={str(p.relative_to(ROOT)): digest(p)
                         for p in (ROOT / "tiny_target").rglob("*.py")},
            reference_sha256=digest(reference / "annotations.json"),
            policy="No tuning during evaluation; V4b shape + V5 maturity, optionally variance-only protection and unit-scale squared log raw-response association consistency. Circle uses six pixels; observed_shape transports previous measured connected pixels plus the existing position_sigma margin, with no missing-shape fallback. No threshold change.",
            allowed_clip_ids=["0029", "0126", "0055", "0082"],
            airborne_accuracy_verified=False))
    freeze = json.loads((a.root / "freeze.json").read_text())
    if digest(cp) != freeze["config_sha256"] or digest(reference / "annotations.json") != freeze["reference_sha256"]:
        raise ValueError("Frozen configuration/reference changed")
    for name, sha in freeze["code_sha256"].items():
        if digest(ROOT / name) != sha:
            raise ValueError("Frozen implementation changed")
    for cid in a.clips:
        source = next(s for s in packet["plan"]["sources"] if s["clip_id"] == cid)
        if digest(source["path"]) != source["sha256"]:
            raise ValueError("Source changed")
        dest = a.root / ("cpu_" + cid)
        subprocess.run([sys.executable, "-m", "tiny_target.visible_baseline",
                        "--source", source["path"], "--config", str(cp),
                        "--output", str(dest)], cwd=ROOT, check=True)
        result = evaluate(dest, cid, reference)
        write(dest / "evaluation.json", result)
        print(json.dumps({"clip": cid, "qualified_proposals": result["qualified_proposal_workload"],
            "windows": [{k: w[k] for k in ("window_id", "qualified_measured_hits", "missed_visible_frames", "ambiguity_frames", "observed_track_ids")}
            for w in result["evaluation"]["positive_windows"]]}), flush=True)


if __name__ == "__main__":
    main()
