"""Verify completed passive U8 detector-origin evidence, without decoding media.

Array arithmetic is independently reconstructed; full-journal detector parity is
the hash-bound completed-run attestation, not a fresh detector execution here.
All examples are selected development evidence, never precision/recall labels.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

SCHEMA = "seaqr.nuisance-origin.analysis.v1"
RUN_SCHEMA = "seaqr.nuisance-origin.run.v1"
CAPTURE_SCHEMA = "seaqr.nuisance-origin-capture.v1"
JOURNAL_SHA = "f669520c0af65315b5a1f106bf60a5b0ebffe7b8539d3dd973d9b2f2bdb7ee92"
SOURCE_SHA = "2f86f28785e302572a86e23688143edbd7f5f1f65e8a3434b86a427e79c6a585"
MANIFEST_SHA = "10d7ec68ba04b05784d4e19a56b0a6dbb561a9e44c7517513b7c1a3be5c881c3"
MEMBERS = {"run_nuisance_origin_v1.py", "nuisance_origin_capture_v1.py", "nuisance_origin_packet_v1.py",
    "packet_plan.json", "batch_discovery_pair.py", "test_nuisance_origin_capture_v1.py",
    "test_nuisance_origin_packet_v1.py", "test_nuisance_origin_runner_v1.py"}
CAPTURE_FRAMES = [i for anchor in (60, 75, 90, 435, 450) for i in range(anchor - 2, anchor + 3)]
FLOAT_PATCHES = ("warped_image", "warped_blur", "spatial", "temporal_pre_learning", "background_pre_finish",
    "variance_pre_finish", "background_post_finish", "variance_post_finish")
BOOL_PATCHES = ("support_pre_finish", "eligible_pre_finish", "support_post_finish", "learning_mask_post_finish")
TIMING_PATHS = [["timings_ms"], ["motion", "pva_timings_ms"], ["motion", "motion_fit", "timing_ms"],
               ["motion", "warp_timings_ms"], ["coverage", "detection_ms"]]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            result.update(block)
    return result.hexdigest()


def decode(text):
    def pairs(items):
        out = {}
        for k, v in items:
            require(k not in out, "duplicate JSON key")
            out[k] = v
        return out
    def number(value):
        out = float(value)
        require(math.isfinite(out), "nonfinite JSON number")
        return out
    return json.loads(text, object_pairs_hook=pairs, parse_float=number,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def read(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "missing/linked metadata")
    return decode(path.read_text())


def bound(path, digest, bindings):
    path = Path(path)
    require(isinstance(digest, str) and re.fullmatch("[a-f0-9]{64}", digest), "invalid SHA256")
    require(path.is_file() and not path.is_symlink() and sha(path) == digest, "hash mismatch: " + str(path))
    bindings[str(path.resolve())] = digest


def recheck(bindings):
    for name, digest in bindings.items():
        require(Path(name).is_file() and not Path(name).is_symlink() and sha(name) == digest,
                "input changed during analysis: " + name)


def array(desc, dtype, shape):
    require(isinstance(desc, dict) and set(desc) == {"dtype", "shape", "data_base64", "sha256"}, "array descriptor fields")
    require(desc["dtype"] == np.dtype(dtype).str and desc["shape"] == list(shape), "array dtype/shape differs")
    require(all(type(n) is int and n >= 0 for n in desc["shape"]), "invalid array shape")
    try:
        raw = base64.b64decode(desc["data_base64"], validate=True)
    except (ValueError, TypeError) as error:
        raise ValueError("invalid array base64") from error
    require(len(raw) == math.prod(shape) * np.dtype(dtype).itemsize
            and hashlib.sha256(raw).hexdigest() == desc["sha256"], "array bytes/hash differs")
    if np.dtype(dtype) == np.bool_:
        require(set(raw) <= {0, 1}, "noncanonical boolean bytes")
    result = np.frombuffer(raw, dtype=dtype).reshape(shape)
    require(result.dtype.kind != "f" or np.isfinite(result).all(), "nonfinite array")
    return result


def exact(a, b, message):
    a, b = np.asarray(a), np.asarray(b)
    require(a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(), message)


def f32_equal(a, b, message):
    require(type(a) in (int, float) and math.isfinite(a), message)
    # Scalar JSON must preserve the exact float32 value, not merely round to it.
    require(float(np.float32(a)) == a, message + " (not exact float32 JSON)")
    exact(np.asarray(a, np.float32), np.asarray(b, np.float32), message)


def update_state(spatial, background, variance, support, learn, parameters):
    """Explicit independent float32 round-after-each-operation; no fused FMA."""
    require(support.dtype == learn.dtype == np.bool_ and support.shape == learn.shape == spatial.shape,
            "state mask dtype/shape differs")
    require(not np.any(learn & ~support), "learning outside support")
    require(np.all(variance >= 0), "negative variance")
    t = np.subtract(spatial, background, dtype=np.float32)
    alpha, noise_alpha, clip2, floor2 = [np.float32(parameters[k]) for k in
        ("background_alpha", "pixel_noise_alpha", "pixel_noise_clip_sigma_squared", "noise_floor_squared")]
    require(0 < alpha <= 1 and 0 < noise_alpha <= 1 and clip2 > 0 and floor2 > 0
            and type(parameters["variance_only"]) is bool, "invalid finish parameters")
    background_updated = np.add(background, np.multiply(alpha, t, dtype=np.float32), dtype=np.float32)
    background_updated = np.where(support if parameters["variance_only"] else learn, background_updated, background)
    observed = np.minimum(np.multiply(t, t, dtype=np.float32), np.multiply(variance, clip2, dtype=np.float32))
    variance_updated = np.add(variance, np.multiply(noise_alpha, np.subtract(observed, variance, dtype=np.float32), dtype=np.float32), dtype=np.float32)
    variance_updated = np.maximum(np.where(learn, variance_updated, variance), floor2)
    return t, background_updated, variance_updated


def statistics(values, limit=None):
    v = np.asarray(values, np.float64)
    result = dict(minimum=float(v.min()), maximum=float(v.max()), mean=float(v.mean()),
                  std=float(v.std()), rms=float(np.sqrt(np.mean(v * v))))
    if limit is not None:
        result.update(display_below=int(np.count_nonzero(v < limit[0])),
                      display_above=int(np.count_nonzero(v > limit[1])), display_range=list(limit))
    return result


def verify_parity(result, parity):
    require(result.get("passed") is True and result.get("error") is None
            and result.get("preflight") is False and result.get("non_timing_journal_parity") is True
            and result.get("processed_frames") == result.get("decoded_frames_verified") == result.get("full_frames") == 673,
            "incomplete or failed run")
    require(parity.get("schema") == "seaqr.feature-residual-trace.v1.parity"
            and parity.get("passed") is True and parity.get("rows_compared") == parity.get("expected_frames") == 673
            and parity.get("original_journal_sha256") == JOURNAL_SHA
            and parity.get("diagnostic_journal_sha256") == result["journal_sha256"]
            and parity.get("mismatch_frames") == [] and parity.get("numeric_tolerance") is False
            and parity.get("excluded_paths") == TIMING_PATHS, "invalid complete parity attestation")


def load_verified(directory, bundle, digest):
    directory, bundle = Path(directory).resolve(), Path(bundle).resolve()
    bindings = {}
    bound(bundle / "freeze.json", digest, bindings)
    bound(directory / "freeze.json", digest, bindings)
    freeze = read(bundle / "freeze.json")
    require(freeze.get("schema") == RUN_SCHEMA + ".freeze" and set(freeze.get("files", {})) == MEMBERS,
            "freeze file inventory/schema differs")
    for name, member_sha in freeze["files"].items():
        bound(bundle / name, member_sha, bindings)
    plan = read(bundle / "packet_plan.json")
    require(plan.get("schema") == "seaqr.nuisance-origin.packet.v1" and plan.get("clip") == "0240"
            and plan.get("input_sha256") == dict(source=SOURCE_SHA, journal=JOURNAL_SHA, manifest=MANIFEST_SHA),
            "packet input scope differs")
    bound(directory / "result.json", sha(directory / "result.json"), bindings)
    result = read(directory / "result.json")
    workspace = result.get("workspace")
    require(isinstance(workspace, str) and re.fullmatch(r"/tmp/seaqr_nuisance_origin_20261001_[A-Za-z0-9]{6}", workspace),
            "result workspace differs from frozen scope")
    expected_source = dict(path="/home/serg/project/camera_reader_sky/srcsky/chunks/chunk_0240.avi", sha256=SOURCE_SHA,
        frames=673, width=4784, height=3190, fps=10, codec="mjpeg", pixel_format="yuvj420p")
    require(result.get("schema") == RUN_SCHEMA and result.get("freeze_sha256") == digest
            and result.get("source_clip") == "0240" and result.get("source") == expected_source
            and result.get("clocks_changed") is False
            and isinstance(result.get("clock_policy_before"), dict)
            and result.get("clock_policy_before") == result.get("clock_policy_after"), "run receipt binding differs")
    require(all(result.get(k) is False for k in ("algorithm_changed", "production_changed", "raw16_accessed",
        "sealed_holdouts_accessed", "source_class_inferred", "performance_benchmark")), "run scope differs")
    files = dict(capture="capture.json", parity="parity.json", journal="run/frames.jsonl", report="run/report.json",
                 launch="run/launch.json", preflight="preflight.json")
    for name, relative in files.items():
        bound(directory / relative, result.get(name + "_sha256"), bindings)
    bound(directory / "batch_status.json", sha(directory / "batch_status.json"), bindings)
    batch = read(directory / "batch_status.json")
    require(batch.get("schema") == RUN_SCHEMA + ".batch" and batch.get("complete") is True
            and batch.get("passed") is True and batch.get("error") is None and batch.get("current") is None
            and batch.get("freeze_sha256") == digest and batch.get("workers") == 1
            and batch.get("automatic_retries") == 0 and batch.get("production_changed") is False,
            "completed matching batch required")
    require([v.get("name") for v in batch.get("phases", [])] == ["tests", "preflight", "run"]
            and all(type(v.get("returncode")) is int and v["returncode"] == 0 for v in batch["phases"]),
            "batch child completion differs")
    artifact_map = {workspace + "/result.json": bindings[str((directory / "result.json").resolve())]}
    artifact_map.update({workspace + "/" + relative: result[name + "_sha256"] for name, relative in files.items()})
    require(batch.get("child_artifacts_sha256") == artifact_map, "batch completed artifact map differs")
    parity = read(directory / "parity.json")
    verify_parity(result, parity)
    preflight = read(directory / "preflight.json")
    require(preflight.get("schema") == RUN_SCHEMA and preflight.get("passed") is True and preflight.get("error") is None
            and preflight.get("preflight") is True and preflight.get("freeze_sha256") == digest
            and preflight.get("workspace") == workspace and preflight.get("source") == expected_source
            and preflight.get("source_clip") == "0240" and preflight.get("full_frames") == 673
            and preflight.get("inputs") == result.get("inputs") and isinstance(result.get("inputs"), dict)
            and preflight.get("identities") == result.get("identities") and isinstance(result.get("identities"), dict)
            and preflight.get("clocks_changed") is False
            and preflight.get("clock_policy_before") == preflight.get("clock_policy_after") == result["clock_policy_before"],
            "matching preflight missing")
    require(all(preflight.get(k) is False for k in ("algorithm_changed", "production_changed", "raw16_accessed",
        "sealed_holdouts_accessed", "source_class_inferred", "performance_benchmark")), "preflight scope differs")
    count = 0
    with (directory / "run/frames.jsonl").open() as stream:
        for count, line in enumerate(stream, 1):
            row = decode(line)
            require(type(row.get("frame_index")) is int and type(row.get("timestamp_ns")) is int
                    and row["frame_index"] == count - 1 and row["timestamp_ns"] == (count - 1) * 100000000,
                    "new journal frame/timestamp gap")
    require(count == 673, "new journal incomplete")
    capture = read(directory / "capture.json")
    launch = read(directory / "run/launch.json")
    require(launch.get("source") == expected_source["path"] and launch.get("source_sha256") == SOURCE_SHA
            and launch.get("expected_frames") == 673 and launch.get("fps") == 10, "launch source/count differs")
    configuration = launch["configuration"]
    recheck(bindings)
    return plan, capture, configuration, bindings


def analyze_capture(plan, capture, configuration):
    require(capture.get("schema") == CAPTURE_SCHEMA and capture.get("processed_frames") == capture.get("native_finish_calls") == 673
            and capture.get("capture_frames") == CAPTURE_FRAMES and capture.get("front_instances") == 1,
            "incomplete capture")
    require(capture.get("original_update_results_returned_unchanged") is True
            and capture.get("diagnostic_device_state_set") is False and capture.get("busy_flag_overridden") is False,
            "capture passivity scope differs")
    require(plan.get("clip") == "0240" and plan.get("patch_size") == 17
            and plan.get("source_shape_hw") == [3190, 4784]
            and sorted(map(int, plan["frame_points"])) == CAPTURE_FRAMES, "capture plan scope differs")
    episodes = plan["episodes"]
    expected_ids = [f"burst_{f:03d}_{p}" for f in (60, 75, 90) for p in ("bright", "dark")]
    expected_ids += [f"known_target_{f:03d}_dark" for f in (435, 450)]
    require([e["episode_id"] for e in episodes] == expected_ids and all(e["sample_status"] == "available" for e in episodes),
            "all eight frozen anchor samples required")
    tile = configuration["tile_size"]
    require(type(tile) is int and tile == 256 and configuration["input_bit_depth"] == 8
            and configuration["pixel_noise_model"] == "background_residual", "configuration differs")
    nx, ny = math.ceil(4784 / tile), math.ceil(3190 / tile)
    expected_parameters = dict(background_alpha=float(np.float32(configuration["background_alpha"])),
        pixel_noise_alpha=float(np.float32(configuration["pixel_noise_alpha"])),
        pixel_noise_clip_sigma_squared=float(np.float32(configuration["pixel_noise_clip_sigma"] ** 2)),
        noise_floor_squared=float(np.float32(configuration["noise_floor_dn"] ** 2)),
        variance_only=configuration["learning_protection_mode"] == "variance_only")
    trajectories, arrays, anchors = {}, {}, []
    records = capture["records"]
    require([r["frame_index"] for r in records] == CAPTURE_FRAMES, "missing or duplicated capture frame")
    require(capture.get("point_snapshots") == sum(len(p) for p in plan["frame_points"].values()), "point snapshot count differs")
    for row in records:
        f = row["frame_index"]
        require(row["timestamp_ns"] == f * 100000000 and row["shape_hw"] == [3190, 4784], "record frame geometry differs")
        for key, value in expected_parameters.items():
            require(row["finish_parameters"][key] == value, "finish parameter differs from launch")
        planned = plan["frame_points"][str(f)]
        require(len(row["points"]) == len(planned), "frame point count differs")
        stats = array(row["pre_tile_statistics"], "<f4", (nx * ny, 2))
        sigmas = array(row["pre_tile_sigmas_float64"], "<f8", (nx * ny,))
        exact(stats[:, 1], sigmas.astype(np.float32), "tile sigma float32 rounding differs")
        require(np.all(sigmas >= configuration["noise_floor_dn"]), "tile sigma below floor")
        for p, frozen in zip(row["points"], planned):
            require(all(p[key] == frozen[key] for key in ("point_id", "episode_id", "role", "reference_xy", "source_xy")),
                    "captured point differs from plan")
            x, y = p["reference_xy"]
            require(type(x) is type(y) is int and 8 <= x < 4784 - 8 and 8 <= y < 3190 - 8, "point patch outside native bounds")
            ti = (y // tile) * nx + x // tile
            require(p["tile_index"] == ti, "point tile index differs")
            a = {name: array(p[name], "<f4", (17, 17)) for name in FLOAT_PATCHES}
            a.update({name: array(p[name], "|b1", (17, 17)) for name in BOOL_PATCHES})
            exact(a["support_pre_finish"], a["support_post_finish"], "finish changed support")
            require(not np.any(a["eligible_pre_finish"] & ~a["support_pre_finish"]), "eligibility outside support")
            t, background_after, variance_after = update_state(a["spatial"], a["background_pre_finish"],
                a["variance_pre_finish"], a["support_pre_finish"], a["learning_mask_post_finish"], row["finish_parameters"])
            exact(t, a["temporal_pre_learning"], "float32 spatial-background temporal differs")
            exact(background_after, a["background_post_finish"], "exact background learning update differs")
            exact(variance_after, a["variance_post_finish"], "exact variance learning update differs")
            center, sigma = stats[ti]
            pixel_sigma = np.sqrt(a["variance_pre_finish"][8, 8], dtype=np.float32)
            effective = np.maximum(sigma, pixel_sigma)
            require(effective > 0, "effective sigma must be positive")
            matches = [v for v in row["raw_peaks"] if v["x"] == x and v["y"] == y]
            require(len(matches) <= 2, "duplicated original raw peak")
            for v in matches:
                require(v["polarity"] in {"bright", "dark"} and v["cell_index"] == 2 * ti + (v["polarity"] == "dark"), "raw peak cell/polarity differs")
                sign = np.float32(1 if v["polarity"] == "bright" else -1)
                score = np.divide(np.multiply(sign, np.subtract(t[8, 8], center, dtype=np.float32), dtype=np.float32), effective, dtype=np.float32)
                for key, value in (("score", score), ("response_dn", t[8, 8]), ("noise_sigma_dn", effective)):
                    f32_equal(v[key], value, "original raw peak " + key + " reconstruction differs")
            reported = p["center_values"]
            values = dict(spatial_dn=a["spatial"][8, 8], temporal_dn=t[8, 8], background_dn=a["background_pre_finish"][8, 8],
                variance_dn_squared=a["variance_pre_finish"][8, 8], tile_center_dn=center,
                tile_sigma_float32_dn=sigma, pixel_sigma_dn=pixel_sigma, effective_sigma_dn=effective,
                background_after_dn=background_after[8, 8], variance_after_dn_squared=variance_after[8, 8])
            for key, value in values.items():
                f32_equal(reported[key], value, "captured center field differs: " + key)
            supported, eligible, learn = [bool(a[k][8, 8]) for k in ("support_pre_finish", "eligible_pre_finish", "learning_mask_post_finish")]
            require(reported["supported"] is supported and reported["eligible"] is eligible
                and reported["learn_variance"] is learn and reported["variance_protected"] is (supported and not learn),
                "eligibility/learning phase semantics differ")
            require(reported["raw_peak_matches"] == matches and reported["exact_peak_reconstruction_count"] == len(matches), "reported raw peak matches differ")
            dominance = "tile" if sigma > pixel_sigma else "pixel_variance" if sigma < pixel_sigma else "equal"
            require(reported["tile_sigma_float64_dn"] == float(sigmas[ti])
                    and reported["tile_floor_or_variance_comparison"] == dominance,
                    "tile sigma/dominance center field differs")
            e = next(e for e in episodes if e["episode_id"] == p["episode_id"])
            if p["role"] == "sample" and f == e["anchor_frame"]:
                chosen = [v for v in matches if v["polarity"] == e["polarity"]]
                require(len(chosen) == 1 and p["reference_xy"] == e["sample_reference_xy"], "anchor original score peak absent")
                for plan_key, raw_key in (("candidate_score", "score"), ("candidate_response_dn", "response_dn"), ("candidate_noise_sigma_dn", "noise_sigma_dn")):
                    require(e[plan_key] == chosen[0][raw_key], "anchor value differs from archived baseline plan")
                anchors.append(dict(episode_id=e["episode_id"], frame_index=f, reference_xy=p["reference_xy"],
                    score=chosen[0]["score"], response_dn=chosen[0]["response_dn"], effective_sigma_dn=chosen[0]["noise_sigma_dn"], exact=True))
            trajectory = dict(frame_index=f, reference_xy=p["reference_xy"], source_xy=p["source_xy"],
                **{k: float(v) for k, v in values.items()}, tile_sigma_float64_dn=float(sigmas[ti]), supported=supported,
                eligible=eligible, learn_variance=learn, variance_protected=supported and not learn,
                sigma_dominance=dominance,
                original_peak_matches=matches, spatial_patch=statistics(a["spatial"], (-8, 8)),
                temporal_patch=statistics(t, (-8, 8)), warped_image_patch=statistics(a["warped_image"], (0, 255)),
                protected_patch_pixels=int(np.count_nonzero(a["support_pre_finish"] & ~a["learning_mask_post_finish"])))
            trajectories.setdefault(p["point_id"], []).append(trajectory)
            arrays[(p["point_id"], f)] = a
    require(len(anchors) == 8, "not all eight baseline anchors reconstructed")
    summaries = []
    for e in episodes:
        roles = {}
        for role in ("sample", "comparator"):
            key = e["episode_id"] + "/" + role
            history = trajectories.get(key)
            if history is None:
                require(role == "comparator" and e["comparator_status"] != "available", "missing declared trajectory")
                roles[role] = dict(status="missing", reason=e["comparator_status"])
                continue
            require([v["frame_index"] for v in history] == e["frames"], "incomplete five-frame trajectory")
            roles[role] = dict(status="available", frames=history,
                center_variance_protected_frames=[v["frame_index"] for v in history if v["variance_protected"]],
                sigma_dominance_counts={kind: sum(v["sigma_dominance"] == kind for v in history) for kind in ("tile", "pixel_variance", "equal")},
                center_signal_ranges={field: [min(v[field] for v in history), max(v[field] for v in history)]
                    for field in ("spatial_dn", "temporal_dn", "effective_sigma_dn", "variance_dn_squared")})
        summaries.append(dict(episode_id=e["episode_id"], anchor_frame=e["anchor_frame"], label=e["label"],
            roles=roles, independent_per_frame_ground_truth=False))
    result = dict(schema=SCHEMA, passed=True, clip="0240", capture_frames=25, point_snapshots=len(arrays),
        exact_anchor_reconstructions=anchors, exact_background_variance_updates_verified=len(arrays) * 17 * 17,
        episodes=summaries, production_changed=False, precision_or_recall_estimated=False, recommended_threshold_change=None,
        interpretation="Descriptive source-origin diagnostics only. Selection roles are not false-positive/negative truth; no algorithm candidate selected.",
        phase_semantics="Pre-finish phase mask is eligibility. Actual variance update uses post-finish learning permission, not eligibility.",
        display_policy="All warped patches use0..255; all spatial/temporal patches sharefixed−8..8DN. Clipping counts saved perpatch. No perpatch stretch.")
    return result, arrays


def display_pixels(values, bounds):
    low, high = bounds
    return np.rint((np.clip(np.asarray(values, np.float64), low, high) - low) * (255 / (high - low))).astype(np.uint8)


def render_contacts(plan, arrays, output):
    import cv2
    artifacts = []
    for e in plan["episodes"]:
        canvas = np.full((826, 650, 3), 22, np.uint8)
        cv2.putText(canvas, e["episode_id"] + " | passive detector arrays", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .46, (235, 235, 235), 1)
        cv2.putText(canvas, "Warped0..255 | spatial/temporal -8..8DN shared; clipping in JSON", (8, 44), cv2.FONT_HERSHEY_SIMPLEX, .40, (235, 235, 235), 1)
        rows = [(role, field, limits) for role in ("sample", "comparator") for field, limits in
                (("warped_image", (0, 255)), ("spatial", (-8, 8)), ("temporal_pre_learning", (-8, 8)))]
        for ri, (role, field, limits) in enumerate(rows):
            y = 77 + ri * 124
            for ci, frame in enumerate(e["frames"]):
                a = arrays.get((e["episode_id"] + "/" + role, frame))
                x = 5 + ci * 130
                label = role[:4] + "/" + ("warp" if field == "warped_image" else "spatial" if field == "spatial" else "temp") + f" f{frame}"
                cv2.putText(canvas, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, .32, (235, 235, 235), 1)
                if a is not None:
                    pixels = cv2.resize(display_pixels(a[field], limits), (102, 102), interpolation=cv2.INTER_NEAREST)
                    canvas[y:y + 102, x:x + 102] = pixels[..., None]
        name = e["episode_id"] + "_detector_arrays.png"
        path = Path(output) / name
        require(not path.exists() and cv2.imwrite(str(path), canvas), "contact sheet write failed")
        artifacts.append(dict(path=name, sha256=sha(path)))
    return artifacts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--freeze-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists() and not args.output.is_symlink(), "output exists; no overwrite")
    own_sha = sha(Path(__file__))
    plan, capture, configuration, bindings = load_verified(args.directory, args.bundle, args.freeze_sha256)
    result, arrays = analyze_capture(plan, capture, configuration)
    recheck(bindings)
    require(sha(Path(__file__)) == own_sha, "analyzer changed during execution")
    args.output.mkdir()
    result.update(analyzer_sha256=own_sha, freeze_sha256=args.freeze_sha256, input_sha256=bindings,
        full_journal_parity="Hash-bound completed-run exact673-frame attestation; not detector execution by this analyzer")
    result["contact_sheets"] = render_contacts(plan, arrays, args.output)
    recheck(bindings)
    require(sha(Path(__file__)) == own_sha, "analyzer changed during rendering")
    with (args.output / "summary.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(passed=True, anchors=len(result["exact_anchor_reconstructions"]), point_snapshots=result["point_snapshots"])))


if __name__ == "__main__":
    main()
