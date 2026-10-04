"""Two known CPU development clips, one worker; frozen labels scored afterward."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from score_phase20_accuracy import digest, score_rows

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tiny_target.visible_regression import score


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path)
    p.add_argument(
        "--shape-only",
        action="store_true",
        help="Test shape measurements alone; learning protection stays disabled",
    )
    p.add_argument(
        "--mode",
        choices=("background_and_variance", "variance_only"),
        default="background_and_variance",
    )
    a = p.parse_args()
    root = a.root
    packet = json.loads((root / "scoring_packet.json").read_text())
    labels = json.loads((root / "annotations.json").read_text())
    freeze = json.loads((root / "scoring_freeze.json").read_text())
    if (
        digest(root / "annotations.json") != freeze["labels_sha256"]
        or digest(root / "scoring_packet.json") != freeze["packet_sha256"]
    ):
        raise ValueError("Changed reference")
    output = a.output or root / "learning_guard_experiment"
    output.mkdir(exist_ok=False)
    config = json.loads(
        (ROOT / "configs/evaluation/phase20_visible_v7.json").read_text()
    )
    config["learning_exclusion_radius_px"] = 6
    config["learning_protection_mode"] = a.mode
    if a.shape_only:
        config.pop("learning_protection_mode")
        config["learning_exclusion_radius_px"] = 0
        config["shape_measurement_mode"] = "mutual_half_height_r8"
    # No association cascade, no threshold change, no clip coordinates/times.
    cp = output / "config.json"
    with cp.open("x") as f:
        json.dump(config, f, indent=2)
    dev = json.loads(
        (ROOT / "configs/evaluation/phase20_visible_development.json").read_text()
    )
    results = []
    for cid in ("0029", "0126"):
        source = next(s for s in packet["plan"]["sources"] if s["clip_id"] == cid)
        if digest(source["path"]) != source["sha256"]:
            raise ValueError("Source changed")
        dest = output / ("cpu_" + cid)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "tiny_target.visible_baseline",
                "--source",
                source["path"],
                "--config",
                str(cp),
                "--output",
                str(dest),
            ],
            cwd=ROOT,
            check=True,
        )
        with (dest / "frames.jsonl").open() as f:
            scored = score_rows(map(json.loads, f), labels, packet, cid, source["fps"])
        ref = next(c for c in dev["clips"] if c["clip_id"] == "chunk" + cid)
        legacy = score(dest, ROOT / ref["annotations"])
        result = dict(
            **scored,
            run=str(dest.resolve()),
            original_anchor_regression=legacy,
            journal_sha256=digest(dest / "frames.jsonl")
        )
        with (dest / "evaluation.json").open("x") as f:
            json.dump(result, f, indent=2)
        results.append(result)
        print(
            json.dumps(
                dict(
                    clip=cid,
                    windows=[
                        {
                            k: w[k]
                            for k in (
                                "window_id",
                                "qualified_measured_hits",
                                "missed_visible_frames",
                                "ambiguity_frames",
                            )
                        }
                        for w in scored["positive_windows"]
                    ],
                )
            ),
            flush=True,
        )
    with (output / "summary.json").open("x") as f:
        json.dump(
            dict(runs=results, labels_sha256=freeze["labels_sha256"], promoted=False),
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
