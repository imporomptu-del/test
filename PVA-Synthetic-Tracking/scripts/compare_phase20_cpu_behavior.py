"""Compare unchanged CPU detection/tracking across the motion-only development fix."""
import argparse
import itertools
import json
from pathlib import Path


def compare(first, second):
    launches = [json.loads((p / "launch.json").read_text()) for p in (first, second)]
    reports = [json.loads((p / "report.json").read_text()) for p in (first, second)]
    if launches[0]["source_sha256"] != launches[1]["source_sha256"]:
        raise ValueError("Source mismatch")
    if launches[0]["configuration"] != launches[1]["configuration"]:
        raise ValueError("Configuration mismatch")
    if launches[0]["configuration"]["motion_backend"] != "cpu_translation":
        raise ValueError("This exact comparison is CPU-only")
    if not all(r["completed"] and r["full_clip"] for r in reports):
        raise ValueError("Completed full clips required")
    differences = []
    count = 0
    with (first / "frames.jsonl").open() as a, (second / "frames.jsonl").open() as b:
        for left, right in itertools.zip_longest(a, b):
            if left is None or right is None:
                raise ValueError("Journal lengths differ")
            x, y = json.loads(left), json.loads(right)
            if x["frame_index"] != count or y["frame_index"] != count:
                raise ValueError("Noncontiguous journal")
            for key in (
                "frame_index",
                "timestamp_ns",
                "segment",
                "source_to_reference",
                "candidates",
                "tracks",
                "tracking_metrics",
                "motion",
            ):
                if x[key] != y[key]:
                    differences.append(dict(frame_index=count, field=key))
            cx, cy = dict(x["coverage"]), dict(y["coverage"])
            for c in (cx, cy):
                for key in ("detection_ms", "detection_ready", "unavailable_reason"):
                    c.pop(key, None)
            if cx != cy:
                differences.append(dict(frame_index=count, field="coverage"))
            count += 1
    if not all(r["frames"] == count for r in reports):
        raise ValueError("Report/journal frame mismatch")
    return dict(
        old=str(first.resolve()),
        new=str(second.resolve()),
        frames=count,
        exact_cpu_behavior_match=not differences,
        difference_count=len(differences),
        first_differences=differences[:20],
        ignored="Wall-clock timings and two additive availability annotations only; implementation hashes are expected to change and are separately frozen.",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--old", type=Path, required=True)
    p.add_argument("--new", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = compare(a.old, a.new)
    with a.output.open("x") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    if not result["exact_cpu_behavior_match"]:
        raise SystemExit(1)
