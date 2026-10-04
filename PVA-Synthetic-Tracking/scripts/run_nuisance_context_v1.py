"""Frozen, local source-context diagnostic. No detector or acceptance rule."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parent / "outputs/seaqr_nuisance_origin_20261001"
SCHEMA = "seaqr.nuisance-context.v1"
LAG = 4
INPUTS = {
    "source": (ROOT.parent / "outputs/seaqr_discovery_pair_20260928/sources/chunk_0240.avi", "2f86f28785e302572a86e23688143edbd7f5f1f65e8a3434b86a427e79c6a585"),
    "journal": (ROOT.parent / "outputs/seaqr_feature_residual_trace_20260930/evidence/0240/run/frames.jsonl", "f669520c0af65315b5a1f106bf60a5b0ebffe7b8539d3dd973d9b2f2bdb7ee92"),
    "references": (ROOT / "configs/evaluation/discovery_pair_regression_20260929.json", "10d7ec68ba04b05784d4e19a56b0a6dbb561a9e44c7517513b7c1a3be5c881c3"),
    "persistent_plan": (OUT / "persistent_plan.json", "dfe5a761a68eca05a8265b0831334ebd8be8748a3c2368a4aa831ec4697ca737"),
}
CODE = [ROOT / p for p in (
    "scripts/nuisance_context_features_v1.py", "scripts/run_nuisance_context_v1.py",
    "tests/unit/test_nuisance_context_features_v1.py", "tests/unit/test_run_nuisance_context_v1.py")]
SELECTED = ("0/dark:571", "0/dark:35", "0/dark:54", "0/bright:447", "0/bright:764", "0/bright:1344")


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def read_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    def number(value):
        result = float(value)
        require(np.isfinite(result), "nonfinite JSON number")
        return result
    return json.loads(text, object_pairs_hook=pairs, parse_float=number,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def verify(path, digest):
    path = Path(path)
    require(path.is_absolute() and path.resolve() == path and path.is_file() and not path.is_symlink(), "canonical regular input required")
    require(sha(path) == digest, "input changed: " + str(path))


def pair_reason(rows, frame, previous):
    if previous is None:
        return "prior_actual_measurement_unavailable_in_frozen_scope"
    if frame < LAG:
        return "insufficient_history"
    interval = rows[frame-LAG:frame+1]
    if len(interval) != LAG+1 or [r["frame_index"] for r in interval] != list(range(frame-LAG, frame+1)):
        return "noncontiguous_history"
    if len({r["segment"] for r in interval}) != 1:
        return "segment_change"
    if any(r["motion"].get("reset") is not False for r in interval):
        return "reset_barrier"
    if any(r["motion"].get("accepted") is not True for r in interval):
        return "geometry_not_accepted"
    return None


def request(identity, group, polarity, frame, current, previous, rows):
    def xy(point):
        require(isinstance(point, list) and len(point) == 2 and all(type(v) in (float, int) and np.isfinite(v) for v in point), "finite raw coordinates required")
        return list(point)
    reason = pair_reason(rows, frame, previous)
    return dict(identity=identity, review_group=group, polarity=polarity, frame_index=frame,
                current_source_xy=xy(current), prior_frame_index=frame-LAG,
                previous_source_xy=xy(previous) if previous is not None else None,
                temporal_unavailable_reason=reason,
                current_source_to_reference=rows[frame]["source_to_reference"],
                prior_source_to_reference=rows[frame-LAG]["source_to_reference"] if frame >= LAG else None,
                geometry_is_independent_ground_truth=False)


def build_requests(persistent, references, rows):
    result = []
    selected = [s for s in persistent["selected"] if s["reviewable"]]
    require(tuple(s["identity"] for s in selected) == SELECTED, "frozen selected identity list differs")
    for item in selected:
        states = {s["frame_index"]: s for s in item["frames"]}
        for f, state in sorted(states.items()):
            if not (state["measured"] is True and state["qualified_moving"] is True):
                continue
            prior = states.get(f-LAG)
            previous = prior["raw_measurement_source_xy"] if prior and prior["measured"] is True else None
            group = "cloud_associated_review_hypothesis" if item["identity"] in ("0/bright:447", "0/bright:1344") else "source_visible_motion_class_unknown"
            result.append(request(item["identity"], group, item["polarity"], f,
                                  state["raw_measurement_source_xy"], previous, rows))
    points = {p["frame_index"]: p["measurement_source_xy"] for p in references}
    require(len(points) == len(references) and sorted(points) == list(range(430, 465)), "complete frozen positive pass required")
    for f, point in sorted(points.items()):
        result.append(request("known_0240_pass", "user_confirmed_pass_baseline_derived_positions", "dark", f, point, points.get(f-LAG), rows))
    require(len(result) == 267, "frozen request count differs")
    return sorted(result, key=lambda r: (r["frame_index"], r["identity"]))


def build_plan():
    for path, digest in INPUTS.values():
        verify(path, digest)
    rows = [read_json(line) for line in INPUTS["journal"][0].read_text().splitlines()]
    require(len(rows) == 673 and all(r["frame_index"] == i and r["timestamp_ns"] == i*100000000 for i, r in enumerate(rows)), "journal continuity/cadence differs")
    persistent = read_json(INPUTS["persistent_plan"][0].read_text())
    refs = read_json(INPUTS["references"][0].read_text())["positive_pass"]["baseline_measurements"]
    requests = build_requests(persistent, refs, rows)
    bindings = [{"path": str(p), "sha256": sha(p)} for p in CODE]
    plan = dict(schema=SCHEMA, clip="0240", input_bindings={k: dict(path=str(v[0]), sha256=v[1]) for k, v in INPUTS.items()},
                code_bindings=bindings, requests=requests, patch_size=65, sigmas=[1.5, 4.0, 8.0], lag_frames=LAG,
                decoded_frame_limit_inclusive=464, native_shape_hw=[3190, 4784], nominal_fps=10,
                sampling="float64 bilinear; current centered on raw source measurement; prior grids share inv(Hprior)*Hcurrent geometry, track grid translated to actual previous measurement",
                spatial="A=<I,G1.5-G4>; B=<I,G4-G8>; unit-sum kernels, DC-centered input; abs(A)/(abs(A)+abs(B)) is a scale-contrast ratio, not object probability",
                temporal="D=(abs(Acur-Abg)-abs(Acur-Atrack))/(abs(Acur-Abg)+abs(Acur-Atrack)); raw amplitudes/errors/denominator retained; numerical zero denominator is null",
                saturation="Aperture is censored if ANY positively weighted native gray contributor is0or255, including fractional interpolation; observed digital values retained, interpreted strata separate; no detector decision",
                geometry="Accepted archived geometry is not independently certified; no local refit, frame/lag search or use of predicted measurements",
                limits=["Single familiar development clip; no independent negatives or airborne recall/FPR estimate",
                        "Six previously reviewable identities cover232 of587 qualified measured burst records; edge-excluded IDs not replaced",
                        "Current positions are detector/reference conditioned: lagged data do not make this an independent causal prediction",
                        "267 descriptive current measurements; unavailable temporal history stays explicit; no retention scoring",
                        "No threshold fitting, feature sweep, veto, synthetic-to-real accuracy claim or production change",
                        "No RAW16, holdouts, SSH/GPU run, clocks or speed benchmark"],
                production_changed=False, classifier_trained=False, raw16_accessed=False, sealed_holdouts_accessed=False)
    for path, digest in INPUTS.values():
        verify(path, digest)
    for binding in bindings:
        verify(binding["path"], binding["sha256"])
    return plan


def numeric_leaves(value, prefix=""):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from numeric_leaves(child, prefix + "/" + key)
    elif type(value) in (float, int) and np.isfinite(value):
        yield prefix, float(value)


def summarize(records):
    groups = defaultdict(list)
    for record in records:
        groups[record["identity"]].append(record)
    output = {}
    for identity, group in groups.items():
        stats, uncensored, censored = defaultdict(list), defaultdict(list), defaultdict(list)
        for r in group:
            for branch in ("spatial", "pair"):
                feature = r[branch]
                if feature is None:
                    continue
                for field, value in numeric_leaves({branch: feature}):
                    stats[field].append(value)
                    destination = uncensored if feature.get("interpretation_available") is True else censored
                    destination[field].append(value)
        def describe(values):
            return {field: dict(n=len(v), minimum=min(v), median=float(np.median(v)), maximum=max(v)) for field, v in sorted(values.items())}
        output[identity] = dict(current_records=len(group),
            review_group=group[0]["review_group"],
            temporal_status_counts=dict(Counter(r["temporal_status"] for r in group)),
            spatial_interpretation_available_count=sum(r["spatial"] is not None and r["spatial"].get("interpretation_available") is True for r in group),
            temporal_interpretation_available_count=sum(r["pair"] is not None and r["pair"].get("interpretation_available") is True for r in group),
            feature_statistics_scope="features includes all observed digital values; interpretable and noninterpretable(censored or numerically degenerate) strata are separate; unsupported and absent pairs excluded, not zero",
            features=describe(stats), interpretable_features=describe(uncensored), noninterpretable_features=describe(censored))
    return output


def run(plan_path, output):
    import cv2
    plan_path, output = Path(plan_path), Path(output)
    require(not output.exists() and not output.is_symlink(), "fresh output required; no overwrite")
    plan_digest = sha(plan_path)
    plan = read_json(plan_path.read_text())
    require(plan == build_plan(), "plan differs from frozen metadata/code")
    import nuisance_context_features_v1 as core
    output.mkdir(parents=False)
    frozen = output / "implementation"
    frozen.mkdir()
    for b in plan["code_bindings"]:
        shutil.copyfile(b["path"], frozen / Path(b["path"]).name)
    shutil.copyfile(plan_path, output / "plan.json")
    require(sha(plan_path) == plan_digest, "plan changed before decode")
    for b in plan["code_bindings"]:
        verify(b["path"], b["sha256"])
        require(sha(frozen / Path(b["path"]).name) == b["sha256"], "copied implementation mismatch")
    generated = []
    generated_arrays = {}
    for index, case in enumerate(core.synthetic_cases()):
        generated.append(dict(name=case["name"], polarity=case["polarity"], expected_scope=case["expected_scope"],
                              features=core.measure_pair(case["current"], case["prior_background"], case["prior_actual"], case["polarity"])))
        for name in ("current", "prior_background", "prior_actual"):
            generated_arrays[f"case{index:03d}_{name}"] = case[name]
    write_json(output / "generated_controls.json", generated)
    np.savez_compressed(output / "generated_controls.npz", **generated_arrays)
    by_frame = defaultdict(list)
    for item in plan["requests"]:
        by_frame[item["frame_index"]].append(item)
    gray_history, records, arrays, gray_hashes = {}, [], {}, {}
    decode_count = 0
    started = time.monotonic()
    cap = cv2.VideoCapture(str(INPUTS["source"][0]))
    try:
        require(cap.isOpened(), "source did not open")
        for f in range(465):
            ok, bgr = cap.read()
            require(ok and bgr is not None and bgr.shape == (3190, 4784, 3) and bgr.dtype == np.uint8, "native frame decode failed")
            decode_count += 1
            # Retain at most five gray frames; do not save/decode unrelated clips.
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            gray_history[f] = gray
            if f-LAG-1 in gray_history:
                del gray_history[f-LAG-1]
            for item in by_frame.get(f, []):
                record = dict(item)
                index = len(records)
                y, x = np.mgrid[-32:33, -32:33]
                cx, cy = item["current_source_xy"]
                current = core.bilinear_sample(gray, x+cx, y+cy)
                current_censor = core.bilinear_saturation_mask(gray, x+cx, y+cy)
                arrays[f"record{index:03d}_current"] = current
                arrays[f"record{index:03d}_current_censor_mask"] = current_censor
                record["spatial"] = core.measure_patch(current, item["polarity"]) if np.isfinite(current).all() else None
                record["native_censor_counts"] = dict(current=int(current_censor.sum()), prior_background=None, prior_actual=None)
                if record["spatial"] is not None:
                    record["spatial"]["native_saturated_contributor_pixels"] = int(current_censor.sum())
                    record["spatial"]["interpretation_available"] = bool(record["spatial"]["interpretation_available"] and not current_censor.any())
                record["pair"] = None
                record["temporal_status"] = item["temporal_unavailable_reason"] or "available"
                record["residual_displacement_xy"] = None
                if record["spatial"] is None:
                    record["temporal_status"] = "current_patch_out_of_support"
                elif item["temporal_unavailable_reason"] is None:
                    try:
                        maps = core.transported_maps(item["current_source_xy"], item["previous_source_xy"],
                                                     item["current_source_to_reference"], item["prior_source_to_reference"])
                    except ValueError as error:
                        record["temporal_status"] = "invalid_geometry: " + str(error)
                    else:
                        prior = gray_history[f-LAG]
                        background = core.bilinear_sample(prior, maps["prior_background_x"], maps["prior_background_y"])
                        tracked = core.bilinear_sample(prior, maps["prior_track_x"], maps["prior_track_y"])
                        bg_censor = core.bilinear_saturation_mask(prior, maps["prior_background_x"], maps["prior_background_y"])
                        track_censor = core.bilinear_saturation_mask(prior, maps["prior_track_x"], maps["prior_track_y"])
                        arrays[f"record{index:03d}_prior_background"] = background
                        arrays[f"record{index:03d}_prior_actual"] = tracked
                        arrays[f"record{index:03d}_prior_background_censor_mask"] = bg_censor
                        arrays[f"record{index:03d}_prior_actual_censor_mask"] = track_censor
                        record["native_censor_counts"].update(prior_background=int(bg_censor.sum()), prior_actual=int(track_censor.sum()))
                        record["residual_displacement_xy"] = np.asarray(maps["residual_displacement_xy"]).tolist()
                        if np.isfinite(background).all() and np.isfinite(tracked).all():
                            record["pair"] = core.measure_pair(current, background, tracked, item["polarity"])
                            record["pair"]["native_censor_counts"] = dict(record["native_censor_counts"])
                            record["pair"]["interpretation_available"] = bool(record["pair"]["interpretation_available"] and not(current_censor.any() or bg_censor.any() or track_censor.any()))
                            record["pair"]["temporal"]["native_interpretation_available"] = record["pair"]["interpretation_available"]
                            if not record["pair"]["interpretation_available"]:
                                record["temporal_status"] = "observed_but_censored_or_uninformative"
                        else:
                            record["temporal_status"] = "prior_patch_out_of_support"
                        gray_hashes[str(f-LAG)] = hashlib.sha256(prior.tobytes()).hexdigest()
                gray_hashes[str(f)] = hashlib.sha256(gray.tobytes()).hexdigest()
                records.append(record)
    finally:
        cap.release()
    require(decode_count == 465 and len(records) == 267, "bounded decode/record count differs")
    for path, digest in INPUTS.values():
        verify(path, digest)
    for b in plan["code_bindings"]:
        verify(b["path"], b["sha256"])
    require(sha(plan_path) == plan_digest, "plan changed during execution")
    write_json(output / "measurements.json", records)
    np.savez_compressed(output / "sampled_patches.npz", **arrays)
    summary = dict(schema=SCHEMA + ".summary", passed=True, plan_sha256=plan_digest,
                   current_measurement_count=len(records), temporal_status_counts=dict(Counter(r["temporal_status"] for r in records)),
                   by_identity=summarize(records), decoded_frames=decode_count, last_decoded_frame=464,
                   gray_frame_sha256=gray_hashes, generated_control_count=len(generated),
                   diagnostic_wall_seconds=time.monotonic()-started, diagnostic_runtime_is_not_pipeline_fps=True,
                   physical_class_inferred=False, classifier_trained=False, production_changed=False,
                   limits=plan["limits"], artifacts={p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()})
    write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--plan-output", type=Path)
    action.add_argument("--run-plan", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.plan_output:
        require(args.output is None, "output only valid for run")
        write_json(args.plan_output, build_plan())
    else:
        require(args.output is not None, "fresh output required")
        summary = run(args.run_plan, args.output)
        print(json.dumps({k: summary[k] for k in ("passed", "current_measurement_count", "temporal_status_counts", "generated_control_count")}))


if __name__ == "__main__":
    main()
