"""One-worker, two-clip development loop with detector/scorer process separation."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))
from tiny_target.visible_baseline import sha256
from tiny_target.visible_regression import score


def all_confident_anchors_pass(results):
    events = [event for result in results for event in result["events"]]
    return bool(events) and all(
        event["required_anchor_count"] >= 2
        and event["dominant_track_anchor_hits"] == event["required_anchor_count"]
        for event in events
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    args = p.parse_args()
    manifest = json.loads(
        (REPOSITORY / "configs/evaluation/phase20_visible_development.json").read_text()
    )
    cfg = json.loads(args.config.read_text())
    if cfg.get("motion_backend") != "cpu_translation":
        p.error(
            "This local development runner requires cpu_translation; run explicit PVA validation separately"
        )
    expected_code = {
        n: sha256(REPOSITORY / "tiny_target" / n)
        for n in ("visible_baseline.py", "tracking/kalman.py", "visible_quality.py")
    }
    expected_config = sha256(args.config)
    args.output_root.mkdir(parents=True, exist_ok=True)
    results = []
    for clip in manifest["clips"]:
        source = args.source_root / clip["source_name"]
        if sha256(source) != clip["source_sha256"]:
            raise ValueError(f"Source hash mismatch: {source}")
        destination = args.output_root / clip["clip_id"]
        if destination.exists():
            if not (destination / "report.json").exists():
                raise ValueError(
                    f"Incomplete existing run preserved: {destination}; use a new output root"
                )
            launch = json.loads((destination / "launch.json").read_text())
            if (
                launch["config_sha256"] != expected_config
                or launch["code_sha256"] != expected_code
                or launch["source_sha256"] != clip["source_sha256"]
            ):
                raise ValueError(
                    f"Existing run belongs to different source/config/code: {destination}"
                )
        else:
            # No annotation path, time interval, coordinate, or crop is passed here.
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tiny_target.visible_baseline",
                    "--source",
                    str(source.resolve()),
                    "--config",
                    str(args.config.resolve()),
                    "--output",
                    str(destination.resolve()),
                ],
                cwd=REPOSITORY,
                check=True,
            )
        result = score(destination, REPOSITORY / clip["annotations"])
        score_path = destination / "regression.json"
        if score_path.exists():
            if json.loads(score_path.read_text()) != result:
                raise ValueError("Existing regression differs; not overwriting")
        else:
            with score_path.open("x") as f:
                json.dump(result, f, indent=2)
        results.append(
            dict(clip_id=clip["clip_id"], run=str(destination.resolve()), **result)
        )
    aggregate = dict(
        scope="Two reviewed development recordings; not generalization or false-alarm validation",
        all_events_pass=all(r["all_events_pass"] for r in results),
        all_confident_anchors_pass=all_confident_anchors_pass(results),
        results=results,
    )
    summary = args.output_root / "summary.json"
    if summary.exists():
        if json.loads(summary.read_text()) != aggregate:
            raise ValueError("Existing aggregate differs; not overwriting")
    else:
        with summary.open("x") as f:
            json.dump(aggregate, f, indent=2)
    print(
        json.dumps(
            dict(
                all_events_pass=aggregate["all_events_pass"],
                all_confident_anchors_pass=aggregate["all_confident_anchors_pass"],
                summary=str(summary.resolve()),
            ),
            indent=2,
        )
    )
    if not aggregate["all_confident_anchors_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
