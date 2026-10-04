"""Verify full-decode execution exactly reproduces a frozen tracking replay."""
import argparse
import itertools
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replay", type=Path, required=True)
    p.add_argument("--full", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--allow-prefix",
        action="store_true",
        help="Compare completed bounded prefixes without claiming full-clip validation",
    )
    a = p.parse_args()
    launches = [json.loads((d / "launch.json").read_text()) for d in (a.replay, a.full)]
    reports = [json.loads((d / "report.json").read_text()) for d in (a.replay, a.full)]
    for key in ("source_sha256", "configuration", "code_sha256"):
        if launches[0][key] != launches[1][key]:
            raise ValueError(f"Launches differ: {key}")
    if not all(r["completed"] for r in reports):
        raise ValueError("Completed data required")
    full_clip = all(r["full_clip"] for r in reports)
    if not full_clip:
        if not (
            a.allow_prefix
            and all(not r["full_clip"] for r in reports)
            and all(
                0 < r["frames"] == l.get("max_frames", -1) < l["expected_frames"]
                for l, r in zip(launches, reports)
            )
        ):
            raise ValueError(
                "Completed full clips or explicitly allowed matched prefixes required"
            )
    if (
        not launches[0]
        .get("execution_mode", "")
        .startswith("tracking_replay_of_frozen_")
    ):
        raise ValueError("Expected a replay source")
    if launches[1].get("execution_mode", "").startswith("tracking_replay_of_frozen_"):
        raise ValueError("Second source must decode media")
    differences = []
    count = 0
    with (a.replay / "frames.jsonl").open() as first, (
        a.full / "frames.jsonl"
    ).open() as second:
        for left, right in itertools.zip_longest(first, second):
            if left is None or right is None:
                raise ValueError("Journal lengths differ")
            x, y = json.loads(left), json.loads(right)
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
                if x[key] != y[key]:
                    differences.append(dict(frame_index=count, field=key))
            cx, cy = dict(x["coverage"]), dict(y["coverage"])
            cx.pop("detection_ms", None)
            cy.pop("detection_ms", None)
            if cx != cy:
                differences.append(dict(frame_index=count, field="coverage"))
            count += 1
    if any(r["frames"] != count for r in reports):
        raise ValueError("Report frame count differs from journal")
    result = dict(
        source_sha256=launches[0]["source_sha256"],
        replay=str(a.replay.resolve()),
        full=str(a.full.resolve()),
        frames=count,
        full_clip=full_clip,
        exact_candidate_tracking_coverage_match=not differences,
        difference_count=len(differences),
        first_differences=differences[:20],
        ignored_fields="Wall-clock stage timings only",
    )
    with a.output.open("x") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result))
    if differences:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
