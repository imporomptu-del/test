"""Single frozen association experiment on existing candidates; no media input."""
import argparse
import json
from pathlib import Path

from replay_phase20_tracking import run
from score_phase20_accuracy import score_rows, digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path)
    p.add_argument(
        "--assignment", choices=("greedy", "global_min_cost"), default="greedy"
    )
    a = p.parse_args()
    root = a.root
    specs = json.loads((root / "scoring_freeze.json").read_text())
    labels = json.loads((root / "annotations.json").read_text())
    packet = json.loads((root / "scoring_packet.json").read_text())
    if (
        digest(root / "annotations.json") != specs["labels_sha256"]
        or digest(root / "scoring_packet.json") != specs["packet_sha256"]
    ):
        raise ValueError("Frozen labels changed")
    output = a.output or root / "confirmed_first_experiment"
    output.mkdir(exist_ok=False)
    results = []
    for spec in specs["runs"]:
        parent = Path(spec["run"])
        launch = json.loads((parent / "launch.json").read_text())
        cfg = dict(
            launch["configuration"],
            tracking_association_cascade="confirmed_first"
            if a.assignment == "greedy"
            else "none",
            tracking_association_assignment=a.assignment,
        )
        name = spec["backend"] + "_" + spec["clip_id"]
        config_path = output / (name + ".json")
        with config_path.open("x") as f:
            json.dump(cfg, f, indent=2)
        destination = output / name
        report = run(
            parent, config_path, destination, allow_prefix=not spec["full_clip"]
        )
        with (destination / "frames.jsonl").open() as f:
            score = score_rows(
                map(json.loads, f), labels, packet, spec["clip_id"], launch["fps"]
            )
        results.append(
            dict(
                **score,
                backend=spec["backend"],
                run=str(destination.resolve()),
                journal_sha256=digest(destination / "frames.jsonl"),
                qualified_track_count=report["qualified_track_count"]
            )
        )
        print(
            json.dumps(
                dict(
                    clip=spec["clip_id"],
                    backend=spec["backend"],
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
                        for w in score["positive_windows"]
                    ],
                )
            ),
            flush=True,
        )
    with (output / "summary.json").open("x") as f:
        json.dump(
            dict(
                runs=results,
                labels_sha256=specs["labels_sha256"],
                scope="Tracking-only experiment, inherited detections/motion; no new hardware speed evidence",
                promoted=False,
            ),
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
