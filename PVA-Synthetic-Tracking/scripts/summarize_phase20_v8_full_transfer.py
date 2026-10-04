"""Audit two complete frozen PVA runs; unlabeled workload is not accuracy."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tiny_target.visible_baseline import sha256
from prepare_phase20_v8_full_transfer import PARENT_SHA, SOURCES
from summarize_phase20_v8_validation import load_run


def audit_rows(rows, expected_frames, expected_shape=None):
    gaps, resets, backend_failures = [], [], []
    current_gap = None
    initial_warmup = 0
    seen_ready = False
    pva_pairs = 0
    frames = 0
    longest_ready = current_ready = 0
    ready_area_fractions = []
    for row in rows:
        index = row["frame_index"]
        if index != frames:
            raise ValueError("Noncontiguous journal")
        frames += 1
        coverage, motion = row["coverage"], row["motion"]
        height, width = coverage["full_shape_hw"]
        total = coverage["total_pixels"]
        if (
            total != height * width
            or total <= 0
            or not 0 <= coverage["searchable_pixels"] <= total
            or coverage["configured_crop"] is not None
            or coverage["native_pixel_sampling"] is not True
            or (expected_shape is not None and [height, width] != expected_shape)
        ):
            raise ValueError("Full-frame native coverage contract violated")
        ready = not coverage["warmup"] and coverage["searchable_pixels"] > 0
        reason = (
            "warmup"
            if coverage["warmup"]
            else "no_valid_search_support"
            if not ready
            else None
        )
        if (
            coverage.get("detection_ready") != ready
            or coverage.get("unavailable_reason") != reason
        ):
            raise ValueError("Stored frame availability disagrees with evidence")
        if ready:
            ready_area_fractions.append(coverage["searchable_pixels"] / total)
            seen_ready = True
            current_ready += 1
            longest_ready = max(longest_ready, current_ready)
            if current_gap is not None:
                gaps.append(current_gap)
                current_gap = None
        else:
            current_ready = 0
            if current_gap is None:
                current_gap = dict(
                    start_frame=index,
                    end_frame=index,
                    frames=0,
                    after_first_ready=seen_ready,
                    reasons=Counter(),
                )
            current_gap["end_frame"] = index
            current_gap["frames"] += 1
            current_gap["reasons"][reason] += 1
            if not seen_ready and coverage["warmup"]:
                initial_warmup += 1
        if motion["reset"]:
            resets.append(index)
        if "accepted" in motion:
            backends = motion.get("motion_backends", {})
            actual_pva = (
                all(
                    backends.get(k) == "PVA"
                    for k in ("optical_flow_pyrlk", "harris", "gaussian_pyramid")
                )
                and backends.get("cpu_fallback") is False
            )
            if actual_pva:
                pva_pairs += 1
            else:
                backend_failures.append(index)
        elif index and not motion.get("pva_failure", False):
            raise ValueError("Missing noninitial motion outcome")
    if current_gap is not None:
        gaps.append(current_gap)
    if frames != expected_frames:
        raise ValueError("Full-clip frame count mismatch")
    return dict(
        frames=frames,
        initial_warmup_frames=initial_warmup,
        unavailable_intervals=gaps,
        post_startup_unavailable_frames=sum(
            g["frames"] for g in gaps if g["after_first_ready"]
        ),
        longest_continuous_ready_frames=longest_ready,
        motion_reset_frame_indices=resets,
        actual_pva_pairs=pva_pairs,
        backend_mismatch_frame_indices=backend_failures,
        actual_pva_without_fallback=pva_pairs == expected_frames - 1
        and not backend_failures,
        detection_ever_ready=seen_ready,
        ready_frame_searchable_area_fraction=(
            dict(
                minimum=min(ready_area_fractions),
                mean=sum(ready_area_fractions) / len(ready_area_fractions),
                maximum=max(ready_area_fractions),
            )
            if ready_area_fractions
            else None
        ),
    )


def summarize(root, output):
    freeze = json.loads((root / "freeze.json").read_text())
    if (
        freeze["parent_freeze_sha256"] != PARENT_SHA
        or sha256(root / "parent_freeze.json") != PARENT_SHA
    ):
        raise ValueError("Wrong parent candidate")
    parent = json.loads((root / "parent_freeze.json").read_text())
    if freeze["files_sha256"] != parent["files_sha256"]:
        raise ValueError("Transfer candidate differs from parent")
    for name, digest in freeze["files_sha256"].items():
        if any(sha256(p) != digest for p in (ROOT / name, root / "snapshot" / name)):
            raise ValueError(f"Frozen candidate changed: {name}")
    if [
        (s["clip_id"], s["sha256"], s["probe"]["frames"]) for s in freeze["sources"]
    ] != [(c, h, n) for c, (h, n) in SOURCES.items()]:
        raise ValueError("Changed source declaration")
    if freeze["planned_runs"] != [
        dict(clip=c, backend="pva", full_clip=True, expected_frames=n)
        for c, (_, n) in SOURCES.items()
    ]:
        raise ValueError("Changed full-clip scope")
    runs = []
    for spec in freeze["planned_runs"]:
        path = root / f'pva_chunk{spec["clip"]}'
        result = load_run(path, spec, freeze)
        launch = json.loads((path / "launch.json").read_text())
        if (
            launch["expected_frames"] != spec["expected_frames"]
            or launch["annotations_supplied_to_detector"]
        ):
            raise ValueError("Wrong input scope or annotation leakage")
        for name, digest in launch["package_sha256"].items():
            if sha256(path / "implementation" / name) != digest:
                raise ValueError("Run implementation snapshot mismatch")
        with (path / "frames.jsonl").open() as f:
            source = next(s for s in freeze["sources"] if s["clip_id"] == spec["clip"])
            result["frame_audit"] = audit_rows(
                map(json.loads, f),
                spec["expected_frames"],
                [source["probe"]["height"], source["probe"]["width"]],
            )
        report = json.loads((path / "report.json").read_text())
        result["timings_ms"] = report["timings_ms"]
        result["launch_sha256"] = sha256(path / "launch.json")
        result["checks"] = dict(
            complete_frozen_full_clip=True,
            usable_detection_coverage=result["availability"][
                "usable_detection_coverage"
            ],
            actual_pva_without_fallback=result["frame_audit"][
                "actual_pva_without_fallback"
            ],
            no_pva_runtime_errors=result["availability"]["counts"]["pva_runtime_errors"]
            == 0,
            no_post_startup_detection_gaps=result["frame_audit"]["detection_ever_ready"]
            and result["frame_audit"]["post_startup_unavailable_frames"] == 0,
        )
        runs.append(result)
    result = dict(
        schema="seaqr.phase20.v8c-full-transfer-results.v1",
        scope="Two predeclared non-holdout full AVI clips, actual Jetson PVA; previously CPU-reviewed and PVA-pair-sampled development data",
        freeze_sha256=sha256(root / "freeze.json"),
        frozen_candidate_unchanged=True,
        runs=runs,
        all_execution_and_availability_checks_pass=all(
            all(r["checks"].values()) for r in runs
        ),
        precision=None,
        recall=None,
        false_positives_per_minute=None,
        generalization_proven=False,
        sealed_holdout_accessed=False,
        interpretation="Complete execution and detection availability are not target recall. Qualified proposals are unlabeled review workload, not object or false-positive counts. Gap failures are preserved; no tuning during evaluation.",
    )
    with output.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(
        json.dumps(
            dict(
                output=str(output),
                runs=[
                    dict(
                        clip=r["clip"],
                        checks=r["checks"],
                        availability=r["availability"],
                    )
                    for r in runs
                ],
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.root, args.output)
