"""Frozen U8 0029/0126 historical-assignment context descriptors; no classifier.

Planning reads only metadata/code. Execution hash-verifies both allowed sources
and decodes each sequentially once, retaining at most five gray frames.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import time
import types

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
SCHEMA = "seaqr.nuisance-context-legacy.v1"
CLIPS = ("0029", "0126")
LAG = 4
PINNED_CODE = {
    "nuisance_context_features_v1.py": "51cdfe412c5010db98117e0660fdfc69f40e5f415f4cceab88b77ba75e3bedda",
    "run_nuisance_context_v1.py": "d62b426ffdd69b5411d55634c101aecc5031ff91fe6359ba9e322168e0c0efd2",
    "score_feature_selection_references.py": "b8698675b8296c66a4923e625219a035a5c3082a0b9b711543830a7072febd20",
}
MANIFEST = ROOT / "configs/evaluation/phase20_accuracy_review_v1.json"
MANIFEST_SHA = "a19d5396b9e2e7047e39d56ff6758750f925470ee540034e6424ff5bc1efc284"
SOURCE_DIR = ROOT.parent / "outputs/jetson_review_clips_20260913"
EXPECTED = {"0029": dict(references=198, qualified_hits=196, requests=162, lag4_eligible=147, last_frame=352),
            "0126": dict(references=157, qualified_hits=156, requests=138, lag4_eligible=138, last_frame=218)}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def regular(path):
    path = Path(path)
    require(path.is_absolute() and path.resolve() == path and path.is_file()
            and not path.is_symlink(), "canonical regular input required")
    return path


def verify(path, digest):
    require(sha(regular(path)) == digest, "input changed: " + str(path))


def load_dependencies():
    modules = []
    for name, digest in PINNED_CODE.items():
        path = SCRIPT_DIR / name
        verify(path, digest)
        # Execute the verified bytes, never a potentially stale bytecode cache.
        content = path.read_bytes()
        require(hashlib.sha256(content).hexdigest() == digest, "code changed before import")
        module = types.ModuleType(name[:-3])
        module.__file__ = str(path)
        exec(compile(content, str(path), "exec"), module.__dict__)
        modules.append(module)
    return tuple(modules)


def test_path():
    nearby = SCRIPT_DIR / "test_nuisance_context_legacy_v1.py"
    return nearby if nearby.is_file() else ROOT / "tests/unit/test_nuisance_context_legacy_v1.py"


def code_bindings():
    bindings = [{"path": str(SCRIPT_DIR / name), "sha256": digest} for name, digest in PINNED_CODE.items()]
    for path in (Path(__file__).resolve(), test_path()):
        bindings.append(dict(path=str(regular(path)), sha256=sha(path)))
    for b in bindings:
        verify(b["path"], b["sha256"])
    return bindings


def read_metadata_journal(path, clip, samples, scorer):
    """Validate the complete journal, retaining only necessary raw track states."""
    wanted = defaultdict(list)
    for sample in samples:
        wanted[sample["frame_index"]].append(sample)
    state_frames = set(wanted) | {f-LAG for f in wanted if f >= LAG}
    rows, scored = [], []
    with regular(path).open() as stream:
        for index, line in enumerate(stream):
            require(bool(line.strip()), "blank journal row")
            row = scorer.decode(line)
            scorer.validate_row(row, index)
            scorer.observations(row)
            scored.extend(scorer.score_frame(row, wanted.get(index, [])))
            thin = {k: row[k] for k in ("frame_index", "segment", "motion", "source_to_reference")}
            thin["tracks"] = row["tracks"] if index in state_frames else []
            rows.append(thin)
    require(len(rows) == scorer.COUNTS[clip] and len(scored) == len(samples), "incomplete baseline journal/references")
    for record in scored:
        for stage in scorer.STAGES:
            actual, saved = record["stages"][stage], record["sample"]["saved"][stage]
            require(actual["hit"] == saved["hit"] and actual["assigned_id"] == saved["assigned_id"]
                    and set(actual["all_gated_ids"]) == set(saved["all_gated_ids"]), "original assignment changed")
    return rows, sorted(scored, key=lambda r: scorer.key(r["sample"]))


def actual_track(row, identity, *, qualified=False):
    found = [t for t in row["tracks"] if f'{t["segment"]}/{t["track_id"]}' == identity]
    if len(found) != 1:
        return None, "assigned_identity_missing" if not found else "assigned_identity_ambiguous"
    track = found[0]
    if track.get("measured") is not True or track.get("measurement_source_xy") is None:
        return None, "assigned_identity_not_actual"
    if qualified and track.get("qualified_moving") is not True:
        return None, "assigned_identity_not_qualified"
    return track, None


def build_requests(scored_by_clip, rows_by_clip, helper):
    requests, records = {}, []
    for clip in CLIPS:
        rows = rows_by_clip[clip]
        for source in scored_by_clip[clip]:
            sample = source["sample"]
            require(sample["clip_id"] == clip, "reference clip differs")
            f = sample["frame_index"]
            ref_id = json.dumps([sample[k] for k in ("panel", "clip_id", "window_id", "frame_index")], separators=(",", ":"))
            record = dict(reference_id=ref_id, **source, request_id=None, descriptor_unavailable_reason=None)
            stage = source["stages"]["qualified_measurement"]
            identity = stage["assigned_id"]
            if stage["hit"] is not True or identity is None:
                record["descriptor_unavailable_reason"] = "no_original_qualified_assignment"
                records.append(record)
                continue
            current, reason = actual_track(rows[f], identity, qualified=True)
            if reason:
                record["descriptor_unavailable_reason"] = reason
                records.append(record)
                continue
            require(identity.split("/")[1].split(":")[0] == sample["polarity"], "assigned polarity differs")
            key = (clip, f, identity)
            request_id = f"{clip}/{f}/{identity}"
            if key not in requests:
                prior, prior_reason = actual_track(rows[f-LAG], identity) if f >= LAG else (None, "insufficient_history")
                req = helper.request(f"{clip}/{identity}", "historical_reference_assigned_measurement_class_unknown",
                    sample["polarity"], f, current["measurement_source_xy"],
                    prior["measurement_source_xy"] if prior else None, rows)
                req.update(request_id=request_id, clip_id=clip, original_identity=identity, reference_ids=[],
                    current_measured=True, current_qualified=True, prior_measured=prior is not None,
                    prior_qualified=prior["qualified_moving"] if prior else None, prior_state_unavailable_reason=prior_reason,
                    assignment_ambiguity_is_not_resolved=True)
                requests[key] = req
            require(requests[key]["current_source_xy"] == current["measurement_source_xy"], "deduplicated coordinate conflict")
            requests[key]["reference_ids"].append(ref_id)
            record["request_id"] = request_id
            records.append(record)
    require(len({r["reference_id"] for r in records}) == len(records), "duplicate reference identity")
    return [requests[key] for key in sorted(requests)], records


def denominators(requests, records):
    result = {}
    for clip in CLIPS:
        refs = [r for r in records if r["sample"]["clip_id"] == clip]
        reqs = [r for r in requests if r["clip_id"] == clip]
        result[clip] = dict(references=len(refs), qualified_hits=sum(r["stages"]["qualified_measurement"]["hit"] for r in refs),
            requests=len(reqs), lag4_eligible=sum(r["temporal_unavailable_reason"] is None for r in reqs),
            last_frame=max((r["frame_index"] for r in reqs), default=-1),
            reference_panel_counts=dict(Counter(r["sample"]["panel"] for r in refs)),
            unavailable_reference_counts=dict(Counter(r["descriptor_unavailable_reason"] for r in refs if r["descriptor_unavailable_reason"])),
            prior_actual_unqualified=sum(r["prior_measured"] and r["prior_qualified"] is False for r in reqs),
            qualified_gated_ambiguity_reference_count=sum(r["stages"]["qualified_measurement"].get("multiple_gated_alternatives", False)
                or r["stages"]["qualified_measurement"].get("shared_gated_observation", False) for r in refs))
    return result


def build_plan():
    _, helper, scorer = load_dependencies()
    codes = code_bindings()
    bindings = {}
    manifest = scorer.read(scorer.bind(bindings, MANIFEST, MANIFEST_SHA))
    sources = {s["clip_id"]: s for s in manifest["sources"] if s["clip_id"] in CLIPS}
    require(set(sources) == set(CLIPS), "allowed source set differs")
    for clip, source in sources.items():
        require(source["path"] == str(SOURCE_DIR / f"chunk_{clip}.avi") and source["sha256"] == scorer.SOURCES[clip]
                and source["frames"] == scorer.COUNTS[clip] and source["shape_hw"] == [3190, 4784] and source["fps"] == 10,
                "frozen U8 source metadata differs")
    all_refs = scorer.references(bindings)
    rows_by_clip, scores = {}, {}
    for clip in CLIPS:
        paths = [scorer.JOURNALS / f"full_repeat0_{clip}" / n for n in ("frames.jsonl", "launch.json", "report.json")]
        for path, digest in zip(paths, scorer.BASELINE_HASHES[clip]):
            scorer.bind(bindings, path, digest)
        scorer.validate_run(clip, scorer.read(paths[1]), scorer.read(paths[2]))
        rows_by_clip[clip], scores[clip] = read_metadata_journal(paths[0], clip, [r for r in all_refs if r["clip_id"] == clip], scorer)
    requests, records = build_requests(scores, rows_by_clip, helper)
    counts = denominators(requests, records)
    for clip in CLIPS:
        require({k: counts[clip][k] for k in EXPECTED[clip]} == EXPECTED[clip], "frozen denominator differs")
    plan = dict(schema=SCHEMA, clips=list(CLIPS), source_bindings=sources,
        metadata_bindings=[dict(path=p, sha256=h) for p, h in sorted(bindings.items())], code_bindings=codes,
        requests=requests, reference_records=records, denominators=counts,
        historical_reference_inventory_count=len(all_refs), excluded_reference_count=sum(r["clip_id"] not in CLIPS for r in all_refs),
        patch_size=65, lag_frames=4, sigmas=[1.5, 4.0, 8.0],
        selection="Original frozen qualified assignment only; actual raw measurement centers, not manual reference centers; alternatives remain ambiguous, never replace the assigned identity; overlapping panels deduplicated only for identical clip/frame/identity",
        sampling="Identical float64 bilinear65x65 A/B/R and lag4 D to pinned0240 runner; full positive-corner native support, no clamp/padding; native positive-weight0/255 contributors censor interpretation",
        geometry="Prior same original segment/ID actual measurement (may be unqualified); every intervening archived geometry accepted, same segment, no reset; geometry is not independent truth",
        limits=["Familiar development references, not independent airborne recall/FPR or precision", "Three original qualified misses remain missing; no descriptor invented at reference coordinates",
            "Assignment alternatives and physical class uncertainty remain explicit", "No threshold fitting, acceptance rule, retention score, detector change, RAW16, holdout or network access",
            "Descriptive diagnostic wall time is not pipeline throughput"], classifier_trained=False, production_changed=False,
        raw16_accessed=False, sealed_holdouts_accessed=False)
    for binding in plan["metadata_bindings"] + codes:
        verify(binding["path"], binding["sha256"])
    return plan


def features_from_arrays(record, arrays, prefix, core):
    """Recomputable feature stage: current/prior patches and censor masks only."""
    current, mask = arrays[prefix+"current"], arrays[prefix+"current_censor_mask"]
    spatial = core.measure_patch(current, record["polarity"]) if np.isfinite(current).all() else None
    counts = dict(current=int(mask.sum()), prior_background=None, prior_actual=None)
    if spatial is not None:
        spatial["native_saturated_contributor_pixels"] = counts["current"]
        spatial["interpretation_available"] = bool(spatial["interpretation_available"] and not mask.any())
    pair = None
    if prefix+"prior_background" in arrays:
        bg, actual = (arrays[prefix+k] for k in ("prior_background", "prior_actual"))
        bm, tm = (arrays[prefix+k+"_censor_mask"] for k in ("prior_background", "prior_actual"))
        counts.update(prior_background=int(bm.sum()), prior_actual=int(tm.sum()))
        if spatial is not None and np.isfinite(bg).all() and np.isfinite(actual).all():
            pair = core.measure_pair(current, bg, actual, record["polarity"])
            pair["native_censor_counts"] = dict(counts)
            pair["interpretation_available"] = bool(pair["interpretation_available"] and not(mask.any() or bm.any() or tm.any()))
            pair["temporal"]["native_interpretation_available"] = pair["interpretation_available"]
    return dict(spatial=spatial, pair=pair, native_censor_counts=counts)


def sample_request(item, gray, prior_gray, index, core):
    prefix = f"record{index:03d}_"
    y, x = np.mgrid[-32:33, -32:33]
    cx, cy = item["current_source_xy"]
    arrays = {prefix+"current": core.bilinear_sample(gray, x+cx, y+cy),
              prefix+"current_censor_mask": core.bilinear_saturation_mask(gray, x+cx, y+cy)}
    record = dict(item, array_prefix=prefix, temporal_status=item["temporal_unavailable_reason"] or "available", residual_displacement_xy=None)
    if not np.isfinite(arrays[prefix+"current"]).all():
        record["temporal_status"] = "current_patch_out_of_support"
    elif item["temporal_unavailable_reason"] is None:
        require(prior_gray is not None, "eligible prior frame absent")
        try:
            maps = core.transported_maps(item["current_source_xy"], item["previous_source_xy"], item["current_source_to_reference"], item["prior_source_to_reference"])
        except ValueError as error:
            record["temporal_status"] = "invalid_geometry: " + str(error)
        else:
            record["residual_displacement_xy"] = maps["residual_displacement_xy"].tolist()
            for name, coord in (("prior_background", "prior_background"), ("prior_actual", "prior_track")):
                arrays[prefix+name] = core.bilinear_sample(prior_gray, maps[coord+"_x"], maps[coord+"_y"])
                arrays[prefix+name+"_censor_mask"] = core.bilinear_saturation_mask(prior_gray, maps[coord+"_x"], maps[coord+"_y"])
            if not all(np.isfinite(arrays[prefix+k]).all() for k in ("prior_background", "prior_actual")):
                record["temporal_status"] = "prior_patch_out_of_support"
    record.update(features_from_arrays(record, arrays, prefix, core))
    if record["pair"] is not None and not record["pair"]["interpretation_available"]:
        record["temporal_status"] = "observed_but_censored_or_uninformative"
    return record, arrays


def verify_array_parity(path, arrays, records, core):
    with np.load(path, allow_pickle=False) as saved:
        require(set(saved.files) == set(arrays), "NPZ key parity failed")
        for key, value in arrays.items():
            require(saved[key].dtype == value.dtype and saved[key].shape == (65, 65)
                    and saved[key].tobytes() == value.tobytes(), "NPZ exact array parity failed")
        for record in records:
            recomputed = features_from_arrays(record, saved, record["array_prefix"], core)
            require(recomputed == {k: record[k] for k in recomputed}, "saved-array feature parity failed")
    return dict(passed=True, arrays=len(arrays), records=len(records), comparison="exact dtype/shape/bytes and descriptor recomputation; not detector parity")


def run(plan_path, output):
    import cv2
    plan_path, output = Path(plan_path), Path(output)
    require(output.is_absolute() and output.resolve() == output and not output.exists() and not output.is_symlink(), "fresh canonical output required; no overwrite")
    own_digest, plan_digest = sha(Path(__file__).resolve()), sha(regular(plan_path))
    core, helper, _ = load_dependencies()
    plan = helper.read_json(plan_path.read_text())
    require(plan == build_plan(), "plan differs from frozen metadata/code")
    for source in plan["source_bindings"].values():
        verify(source["path"], source["sha256"])
    require(sha(Path(__file__).resolve()) == own_digest and sha(plan_path) == plan_digest, "code/plan changed before decode")
    output.mkdir(parents=False)
    frozen = output / "implementation"
    frozen.mkdir()
    for binding in plan["code_bindings"]:
        shutil.copyfile(binding["path"], frozen / Path(binding["path"]).name)
        verify(frozen / Path(binding["path"]).name, binding["sha256"])
    shutil.copyfile(plan_path, output / "plan.json")
    verify(output / "plan.json", plan_digest)
    records, arrays, gray_hashes, decoded = [], {}, {}, {}
    started = time.monotonic()
    for clip in CLIPS:
        by_frame = defaultdict(list)
        for item in plan["requests"]:
            if item["clip_id"] == clip:
                by_frame[item["frame_index"]].append(item)
        source, history = plan["source_bindings"][clip], {}
        # Recheck all bindings immediately before each clip; never open other media.
        for binding in plan["metadata_bindings"] + plan["code_bindings"]:
            verify(binding["path"], binding["sha256"])
        verify(plan_path, plan_digest)
        verify(source["path"], source["sha256"])
        cap = cv2.VideoCapture(source["path"])
        decoded[clip] = 0
        try:
            require(cap.isOpened(), "source did not open")
            for f in range(plan["denominators"][clip]["last_frame"]+1):
                ok, bgr = cap.read()
                require(ok and bgr is not None and bgr.shape == (3190, 4784, 3) and bgr.dtype == np.uint8, "native frame decode failed")
                decoded[clip] += 1
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                history[f] = gray
                history.pop(f-LAG-1, None)
                for item in by_frame.get(f, []):
                    record, patch_arrays = sample_request(item, gray, history.get(f-LAG), len(records), core)
                    records.append(record)
                    arrays.update(patch_arrays)
                    for frame in (f, f-LAG) if item["temporal_unavailable_reason"] is None else (f,):
                        key = f"{clip}/{frame}"
                        if key not in gray_hashes:
                            gray_hashes[key] = hashlib.sha256(history[frame].tobytes()).hexdigest()
        finally:
            cap.release()
        require(decoded[clip] == plan["denominators"][clip]["last_frame"]+1, "bounded decode count differs")
    require(len(records) == len(plan["requests"]), "measurement count differs")
    helper.write_json(output / "measurements.json", records)
    np.savez_compressed(output / "sampled_patches.npz", **arrays)
    audit = verify_array_parity(output / "sampled_patches.npz", arrays, records, core)
    require(helper.read_json((output / "measurements.json").read_text()) == records, "JSON roundtrip parity failed")
    for b in plan["metadata_bindings"] + plan["code_bindings"] + list(plan["source_bindings"].values()):
        verify(b["path"], b["sha256"])
    verify(plan_path, plan_digest)
    summary = dict(schema=SCHEMA+".summary", passed=True, plan_sha256=plan_digest, denominators=plan["denominators"],
        current_measurement_count=len(records), decoded_frames=decoded, gray_frame_sha256=gray_hashes,
        temporal_status_counts=dict(Counter(r["temporal_status"] for r in records)), by_identity=helper.summarize(records),
        saved_array_audit=audit, diagnostic_wall_seconds=time.monotonic()-started, diagnostic_runtime_is_not_pipeline_fps=True,
        physical_class_inferred=False, classifier_trained=False, production_changed=False, limits=plan["limits"],
        artifacts={p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()})
    helper.write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan-output", type=Path)
    mode.add_argument("--run-plan", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.plan_output:
        require(args.output is None and not args.plan_output.exists() and not args.plan_output.is_symlink(), "fresh plan required; no overwrite")
        _, helper, _ = load_dependencies()
        helper.write_json(args.plan_output, build_plan())
    else:
        require(args.output is not None, "fresh output required")
        result = run(args.run_plan, args.output)
        print(json.dumps({k: result[k] for k in ("passed", "current_measurement_count", "temporal_status_counts")}))


if __name__ == "__main__":
    main()
