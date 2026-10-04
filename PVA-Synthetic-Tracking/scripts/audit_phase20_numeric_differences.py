"""Inspect cross-runtime journal differences without changing exact-test verdicts."""
import argparse
from collections import Counter
import itertools
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--left", type=Path, required=True)
    p.add_argument("--right", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    launches = [json.loads((d / "launch.json").read_text()) for d in (a.left, a.right)]
    reports = [json.loads((d / "report.json").read_text()) for d in (a.left, a.right)]
    for key in ("source_sha256", "configuration", "code_sha256"):
        if launches[0][key] != launches[1][key]:
            raise ValueError(f"Different launch {key}")
    if not all(r["completed"] for r in reports):
        raise ValueError("Completed reports required")
    counts = Counter()
    maximum = 0.0
    maximum_path = None
    samples = []

    def diff(x, y, path):
        nonlocal maximum, maximum_path
        if x == y:
            return
        if isinstance(x, dict) and isinstance(y, dict) and x.keys() == y.keys():
            for key in x:
                diff(x[key], y[key], f"{path}.{key}")
        elif isinstance(x, list) and isinstance(y, list) and len(x) == len(y):
            for i, (u, v) in enumerate(zip(x, y)):
                diff(u, v, f"{path}[{i}]")
        elif isinstance(x, float) and isinstance(y, float):
            counts["floating_value_differences"] += 1
            delta = abs(x - y)
            if delta > maximum:
                maximum, maximum_path = delta, path
        else:
            counts["discrete_or_structure_differences"] += 1
            if len(samples) < 20:
                samples.append(dict(path=path, left=x, right=y))

    count = 0
    with (a.left / "frames.jsonl").open() as left, (
        a.right / "frames.jsonl"
    ).open() as right:
        for x, y in itertools.zip_longest(left, right):
            if x is None or y is None:
                raise ValueError("Different journal lengths")
            x, y = json.loads(x), json.loads(y)
            for row in (x, y):
                if row["frame_index"] != count:
                    raise ValueError("Non-contiguous journal")
            for key in (
                "frame_index",
                "timestamp_ns",
                "segment",
                "source_to_reference",
                "candidates",
                "tracks",
                "tracking_metrics",
            ):
                diff(x[key], y[key], f"frame[{count}].{key}")
            cx, cy = dict(x["coverage"]), dict(y["coverage"])
            cx.pop("detection_ms", None)
            cy.pop("detection_ms", None)
            diff(cx, cy, f"frame[{count}].coverage")
            count += 1
    if any(r["frames"] != count for r in reports):
        raise ValueError("Report count mismatch")
    result = dict(
        left=str(a.left.resolve()),
        right=str(a.right.resolve()),
        frames=count,
        counts=dict(counts),
        maximum_absolute_float_difference=maximum,
        maximum_difference_path=maximum_path,
        discrete_difference_samples=samples,
        interpretation="Diagnostic only: original exact-comparison failure is preserved. No regression threshold is relaxed.",
    )
    with a.output.open("x") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
