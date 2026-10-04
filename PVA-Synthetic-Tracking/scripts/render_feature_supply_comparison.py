#!/usr/bin/env python3
"""Two fixed local review windows; no inference, matching, or ID selection.

Imports presentation primitives only. Candidate execution must be explicitly
declared and hash-bound; baseline-shaped launch metadata alone is insufficient.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import importlib.util
from pathlib import Path
import re
import shutil
import subprocess

import cv2
import numpy as np

HELPER = Path(__file__).resolve().with_name("render_discovery_pair_review.py")
_spec = importlib.util.spec_from_file_location("feature_supply_presentation", HELPER)
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)
require, sha = base.require, base.sha

SOURCE_SHA = "2f86f28785e302572a86e23688143edbd7f5f1f65e8a3434b86a427e79c6a585"
SOURCE_REMOTE = "/home/serg/project/camera_reader_sky/srcsky/chunks/chunk_0240.avi"
FRAMES, WIDTH, HEIGHT, FPS = 673, 4784, 3190, Fraction(10)
CANDIDATE = dict(harris_gain=16, harris_capacity_policy="complete_grid", feature_image_scale=.5)
WINDOWS = (
    dict(name="burst_0050_0105", first=50, last=105, crop=None),
    dict(name="positive_0430_0464", first=430, last=464, crop=[2592, 2592, 640, 384]),
)
HEADER, FOOTER, GAP = 156, 80, 16


def read(path):
    path = Path(path)
    require(path.is_file() and path.stat().st_size <= 32 * 1024 * 1024, "Missing/oversize metadata")
    return base.decode(path.read_bytes())


def validate_contract(receipt, preflight, launch, report, arm):
    require(arm in ("baseline", "candidate"), "Unknown arm")
    schema = "seaqr.discovery-pair.baseline.v1" if arm == "baseline" else "seaqr.discovery-feature-supply.v1"
    expected_source = dict(path=SOURCE_REMOTE, sha256=SOURCE_SHA, frames=FRAMES,
        width=WIDTH, height=HEIGHT, fps=10, codec="mjpeg", pixel_format="yuvj420p")
    require(receipt.get("schema") == schema and receipt.get("passed") is True
            and receipt.get("error") is None and receipt.get("clip") == "0240"
            and receipt.get("source") == expected_source
            and type(receipt.get("processed_frames")) is int and receipt["processed_frames"] == FRAMES
            and receipt.get("decoded_frames_verified") == FRAMES, "Wrong, incomplete, or failed arm receipt")
    require(preflight.get("schema") == schema + ".preflight" and preflight.get("passed") is True
            and preflight.get("clip") == "0240" and preflight.get("source") == expected_source
            and preflight.get("detector_run") is False and preflight.get("probe_passed") is True
            and preflight.get("workspace") == receipt.get("workspace")
            and isinstance(receipt.get("input_sha256"), dict)
            and preflight.get("input_sha256") == receipt["input_sha256"], "Preflight does not bind successful execution")
    for key in ("detector_configuration_changed", "annotations_supplied_to_detector", "raw16_accessed", "sealed_holdouts_accessed"):
        require(receipt.get(key) is False, "Unexpected execution scope: " + key)
    if arm == "baseline":
        require(receipt.get("algorithm_changed") is False, "Baseline is not frozen baseline")
    else:
        require(receipt.get("algorithm_changed") is True and receipt.get("feature_algorithm_changed") is True
                and receipt.get("candidate") == CANDIDATE and preflight.get("candidate") == CANDIDATE,
                "Candidate declaration missing/different")
        for key in ("tracker_configuration_changed", "global_motion_gates_changed", "production_promotion"):
            require(receipt.get(key) is False, "Candidate expanded scope: " + key)
        adapter = receipt.get("feature_adapter", {})
        original, effective = adapter.get("original_motion_configuration", {}), adapter.get("effective_motion_configuration", {})
        require(original.get("harris_capacity_policy") == "legacy_default"
                and effective == dict(original, harris_capacity_policy="complete_grid")
                and effective.get("feature_image_scale") == .5, "Unexpected candidate motion configuration")
        transformation = adapter.get("source_transformation", {})
        require(adapter.get("candidate") == CANDIDATE and adapter.get("estimator_instances") == 1
                and transformation.get("exact_original_recovered") is True
                and transformation.get("only_statement_change") == "U8 to Harris S16 scale16 offset0"
                and transformation.get("heavy_capture_hooks") is False
                and preflight.get("feature_adapter", {}).get("source_transformation") == transformation
                and preflight.get("feature_adapter", {}).get("effective_motion_configuration") == effective,
                "Candidate feature adapter differs from successful preflight")
        controls = preflight.get("controls", [])
        require(preflight.get("conversion", {}).get("passed") is True
                and [v.get("name") for v in controls] == ["low_contrast_static", "low_contrast_translated", "flat_static"]
                and all(v.get("passed") is True and v.get("closed") is True for v in controls),
                "Candidate conversion/control preflight failed or incomplete")
    require(report.get("completed") is True and report.get("full_clip") is True
            and type(report.get("frames")) is int and report["frames"] == FRAMES
            and report.get("source_sha256") == launch.get("source_sha256") == SOURCE_SHA
            and launch.get("source") == SOURCE_REMOTE and launch.get("expected_frames") == FRAMES
            and launch.get("max_frames") is None and launch.get("fps") == 10
            and launch.get("annotations_supplied_to_detector") is False,
            "Wrong source, incomplete report, cadence, or annotation contract")
    expected_probe = dict(codec="mjpeg", pixel_format="yuvj420p", width=WIDTH,
        height=HEIGHT, declared_frame_count=FRAMES, frame_rate="10")
    require(all(launch.get("source_probe", {}).get(k) == v for k, v in expected_probe.items()), "Launch source probe differs")
    require(isinstance(launch.get("configuration"), dict) and launch["configuration"] == report.get("configuration")
            and isinstance(launch.get("package_sha256"), dict) and launch["package_sha256"], "Missing/changed configuration or package")
    lifecycle = report.get("frame_decode", {})
    require(lifecycle.get("decoded_frames") == lifecycle.get("consumed_frames") == FRAMES
            and lifecycle.get("dropped_frames") == 0 and lifecycle.get("worker_joined") is True
            and lifecycle.get("capture_released") is True
            and lifecycle.get("maximum_observed_frames_ahead", 2) <= 1
            and lifecycle.get("contract", {}).get("execution") == "prefetch_one",
            "Decode lifecycle mismatch")


def load_arm(directory, expected_receipt_sha, arm):
    directory = Path(directory).resolve()
    require(re.fullmatch(r"[0-9a-f]{64}", expected_receipt_sha) is not None, "Explicit lowercase receipt SHA256 required")
    paths = dict(receipt=directory / "execution_receipt.json", preflight=directory / "preflight.json",
        report=directory / "run/report.json", launch=directory / "run/launch.json", journal=directory / "run/frames.jsonl")
    hashes = {str(p): sha(p) for p in paths.values()}
    require(hashes[str(paths["receipt"])] == expected_receipt_sha, "Execution receipt SHA differs")
    receipt = read(paths["receipt"])
    for key, name in (("preflight_sha256", "preflight"), ("journal_sha256", "journal"), ("report_sha256", "report"), ("launch_sha256", "launch")):
        require(receipt.get(key) == hashes[str(paths[name])], "Receipt-bound artifact changed: " + name)
    preflight, launch, report = (read(paths[k]) for k in ("preflight", "launch", "report"))
    validate_contract(receipt, preflight, launch, report, arm)
    return dict(paths=paths, hashes=hashes, receipt=receipt, launch=launch, report=report)


def availability(row):
    coverage, motion = row.get("coverage", {}), row.get("motion", {})
    require(coverage.get("full_shape_hw") == [HEIGHT, WIDTH] and coverage.get("native_pixel_sampling") is True
            and coverage.get("configured_crop") is None, "Cropped/non-native inference")
    require(type(coverage.get("warmup")) is bool and type(coverage.get("detection_ready")) is bool
            and type(coverage.get("searchable_pixels")) is int
            and 0 <= coverage["searchable_pixels"] <= WIDTH * HEIGHT, "Invalid availability fields")
    ready = not coverage["warmup"] and coverage["searchable_pixels"] > 0
    reason = "warmup" if coverage["warmup"] else "no_valid_search_support" if not ready else None
    require(coverage["detection_ready"] == ready and coverage.get("unavailable_reason") == reason,
            "Availability fields disagree")
    require(isinstance(motion.get("status"), str) and motion["status"] and type(motion.get("reset")) is bool
            and isinstance(row.get("candidates"), list), "Missing display metadata")
    return dict(ready=ready, label="READY" if ready else "UNAVAILABLE: " + reason,
        unavailable_reason=reason, motion_status=motion["status"], motion_reset=motion["reset"],
        segment=row["segment"], raw_candidates=len(row["candidates"]))


def collect(path, arm, report):
    selected, ready_count = {}, 0
    for row, channels in base.journal_rows(path, FRAMES, FPS, "0240-" + arm):
        status = availability(row)
        ready_count += status["ready"]
        if any(w["first"] <= row["frame_index"] <= w["last"] for w in WINDOWS):
            selected[row["frame_index"]] = dict(channels=channels, status=status)
    require(len(selected) == 91 and ready_count == report.get("availability", {}).get("counts", {}).get("detection_ready_frames"),
            "Full journal readiness/count differs from report")
    return selected


def canvas_for(frame, baseline, candidate, window):
    require(any(window == declared for declared in WINDOWS), "Only fixed windows permitted")
    bc, cc = baseline["channels"], candidate["channels"]
    require(bc["frame_index"] == cc["frame_index"] and bc["timestamp_ns"] == cc["timestamp_ns"], "Arm timelines disagree")
    require(window["first"] <= bc["frame_index"] <= window["last"], "Frame outside fixed window")
    roi = window["crop"] or [0, 0, WIDTH, HEIGHT]
    x, y, width, height = roi
    require(frame.shape == (HEIGHT, WIDTH, 3) and frame.dtype == np.uint8, "Wrong native source pixels")
    raw = (cv2.resize(frame, (960, round(HEIGHT * 960 / WIDTH)), interpolation=cv2.INTER_AREA)
           if window["crop"] is None else frame[y:y + height, x:x + width].copy())
    panels = [raw, raw.copy(), raw.copy()]
    for panel, data in zip(panels[1:], (baseline, candidate)):
        # Never hide or infer states from readiness. Report exact saved output,
        # plus readiness separately; predicted-only states remain predictions.
        base.annotate(panel, data["channels"]["track_context"], roi)
    ph, pw = raw.shape[:2]
    ww, hh = 3 * pw + 2 * GAP, HEADER + ph + FOOTER
    hh += hh % 2
    canvas = np.full((hh, ww, 3), 23, np.uint8)
    labels = ("SOURCE / unmarked", "FROZEN BASELINE", "FEATURE-SUPPLY CANDIDATE / NOT PROMOTED")
    shown = {}
    for i, (panel, label) in enumerate(zip(panels, labels)):
        xx = i * (pw + GAP)
        canvas[HEADER:HEADER + ph, xx:xx + pw] = panel
        base.text(canvas, label, xx + 8, 82, pw - 16)
        if i:
            arm, data = ("baseline", baseline) if i == 1 else ("candidate", candidate)
            status, channels = data["status"], data["channels"]
            records = base.records_inside(channels["track_context"], roi)
            m = sum(v["measured"] for v in records)
            p = len(records) - m
            shown[arm] = dict(**status, current_measurements_in_view=m, predictions_in_view=p,
                full_frame_qualified_measurements=len(channels["observation_alerts"]),
                qualified_states_offscreen=sum(not base.inside(v["source_xy"], WIDTH, HEIGHT) for v in channels["track_context"]),
                in_view_records=records)
            base.text(canvas, status["label"], xx + 8, 106, pw - 16,
                color=base.GREEN if status["ready"] else base.AMBER)
            base.text(canvas, f"segment {status['segment']} | motion {status['motion_status']} | raw candidates {status['raw_candidates']}", xx + 8, 128, pw - 16)
            base.text(canvas, f"In view: M {m} measurements | P {p} predictions", xx + 8, 150, pw - 16)
    require(np.array_equal(canvas[HEADER:HEADER + ph, :pw], raw), "Raw panel modified")
    base.text(canvas, f"0240 | source frame {bc['frame_index']} | t={bc['timestamp_ns']/1e9:.1f}s | same source, same frame, 10FPS playback", 10, 25, ww - 20)
    base.text(canvas, "Green circle M = actual measurement; dashed amber P +age = prediction only. Track IDs are independent between runs.", 10, 51, ww - 20)
    scale = "Downscaled full field: tiny-object absence cannot be inferred." if window["crop"] is None else "Fixed native 1:1 crop; outside field hidden, all nearby qualified states shown."
    base.text(canvas, scale, 10, HEADER + ph + 23, ww - 20)
    base.text(canvas, "UNAVAILABLE is not zero detections. READY is not full-pixel coverage or accuracy. No new physical-class labels.", 10, HEADER + ph + 46, ww - 20)
    base.text(canvas, "No contrast adjustment, no temporal skipping, no outcome-selected IDs. H264 viewing copy; not detector input.", 10, HEADER + ph + 69, ww - 20)
    return canvas, shown


def verify_encoded(path, shape, window, output, original_qa):
    cap = cv2.VideoCapture(str(path))
    count, samples = 0, []
    try:
        require(cap.isOpened(), "Cannot decode rendered output")
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            require(frame.shape == shape and count <= window["last"] - window["first"], "Encoded frame geometry/count changed")
            index = window["first"] + count
            if index in original_qa:
                destination = output / "qa" / f"{window['name']}_encoded_f{index:04d}.png"
                require(cv2.imwrite(str(destination), frame), "Encoded QA write failed")
                original = cv2.imread(str(output / original_qa[index]["path"]))
                require(original is not None and original.shape == shape, "Unencoded QA unavailable")
                pw = window["crop"][2] if window["crop"] else 960
                ph = window["crop"][3] if window["crop"] else round(HEIGHT * 960 / WIDTH)
                error = np.abs(original[HEADER:HEADER + ph, :pw].astype(np.int16) - frame[HEADER:HEADER + ph, :pw].astype(np.int16))
                samples.append(dict(frame=index, path=str(destination.relative_to(output)), sha256=sha(destination),
                    raw_left_lossy_error_mean_dn=float(error.mean()), raw_left_lossy_error_max_dn=int(error.max())))
            count += 1
    finally:
        cap.release()
    require(count == window["last"] - window["first"] + 1 and len(samples) == 3, "Output decode/QA incomplete")
    return dict(decoded_frames=count, encoded_qa=samples)


def run(source, baseline_dir, candidate_dir, baseline_receipt_sha256, candidate_receipt_sha256, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    require(not output.exists(), "Fresh output directory required")
    arms = {arm: load_arm(directory, digest, arm) for arm, directory, digest in (
        ("baseline", baseline_dir, baseline_receipt_sha256), ("candidate", candidate_dir, candidate_receipt_sha256))}
    for key in ("configuration", "config_sha256", "motion_config_sha256", "package_sha256"):
        require(arms["baseline"]["launch"].get(key) == arms["candidate"]["launch"].get(key),
                "Unchanged detector/package metadata differs: " + key)
    bound = {str(Path(__file__).resolve()): sha(Path(__file__)), str(HELPER): sha(HELPER), str(base.POLICY_PATH): sha(base.POLICY_PATH)}
    for arm in arms.values():
        bound.update(arm["hashes"])
    bound[str(source)] = sha(source)
    require(bound[str(source)] == SOURCE_SHA, "Wrong local source")
    require(shutil.which("ffmpeg") and shutil.which("ffprobe"), "Local ffmpeg/ffprobe required")
    require("libx264" in subprocess.check_output(["ffmpeg", "-hide_banner", "-encoders"], text=True)
            and "fps_mode" in subprocess.check_output(["ffmpeg", "-hide_banner", "-h", "full"], text=True, stderr=subprocess.STDOUT),
            "Local encoder must support libx264 and fps_mode; old remote ffmpeg is unsupported")
    source_probe = base.probe(source)["streams"][0]
    require(source_probe["codec_name"] == "mjpeg" and source_probe["pix_fmt"] == "yuvj420p"
            and (source_probe["width"], source_probe["height"]) == (WIDTH, HEIGHT)
            and Fraction(source_probe["avg_frame_rate"]) == Fraction(source_probe["r_frame_rate"]) == FPS
            and int(source_probe["nb_frames"]) == FRAMES, "Wrong source stream")
    selected = {name: collect(arm["paths"]["journal"], name, arm["report"]) for name, arm in arms.items()}
    require(all(sha(Path(p)) == h for p, h in bound.items()), "Inputs changed during validation")
    output.mkdir(parents=True)
    (output / "qa").mkdir()
    base.write_json(output / "selection.json", dict(windows=WINDOWS, source_sha256=SOURCE_SHA,
        frozen_before_candidate_visual_review=True, candidate_selected=False, id_filter=False,
        positive_window_baseline_derived_not_independent_truth=True, burst_is_not_verified_negative=True))
    cap = cv2.VideoCapture(str(source))
    videos = []
    try:
        require(cap.isOpened(), "Cannot open source")
        for window in WINDOWS:
            require(cap.set(cv2.CAP_PROP_POS_FRAMES, window["first"])
                    and round(cap.get(cv2.CAP_PROP_POS_FRAMES)) == window["first"], "Source seek failed")
            encoder, original_qa, displayed = None, {}, []
            try:
                for index in range(window["first"], window["last"] + 1):
                    ok, frame = cap.read()
                    require(ok and round(cap.get(cv2.CAP_PROP_POS_FRAMES)) == index + 1, "Source window frame missing/misaligned")
                    canvas, shown = canvas_for(frame, selected["baseline"][index], selected["candidate"][index], window)
                    if encoder is None:
                        encoder = base.Encoder(output / (window["name"] + ".mp4"), canvas.shape, FPS, window["last"] - window["first"] + 1)
                    encoder.send(canvas)
                    displayed.append(dict(frame=index, **shown))
                    if index in (window["first"], (window["first"] + window["last"]) // 2, window["last"]):
                        path = output / "qa" / f"{window['name']}_original_f{index:04d}.png"
                        require(cv2.imwrite(str(path), canvas), "Original QA write failed")
                        original_qa[index] = dict(frame=index, path=str(path.relative_to(output)), sha256=sha(path), before_lossy_encoding=True)
                encoded = encoder.finish()
                require(len(original_qa) == 3, "Missing original QA")
                checked = verify_encoded(encoder.path, encoder.shape, window, output, original_qa)
                videos.append(dict(window=window, path=encoder.path.name, **encoded, **checked,
                    original_qa=list(original_qa.values()), frame_display_accounting=displayed))
            finally:
                if encoder is not None:
                    encoder.stop()
    finally:
        cap.release()
    require(all(sha(Path(p)) == h for p, h in bound.items()), "Inputs changed during render")
    receipt = dict(schema="seaqr.feature-supply.bounded-review.v1", completed=True, clip="0240",
        bound_input_sha256=bound, selection_sha256=sha(output / "selection.json"), source_probe=source_probe,
        videos=videos, source_frames_decoded=91, raw_left_exact_before_encoding=True,
        no_contrast_adjustment=True, no_temporal_downsampling=True, inference_rerun=False,
        candidate_production_promoted=False, matched_track_ids_assumed=False,
        interpretation="Bounded same-source presentation; availability and measurements are separate. Burst is not verified negative; baseline-derived positive window is not independent recall ground truth.",
        encoding="H264 CRF12 yuv420p,2threads; lossy viewing copies", versions=dict(opencv=cv2.__version__, numpy=np.__version__,
            ffmpeg=subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0]))
    base.write_json(output / "receipt.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "baseline-dir", "candidate-dir", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("baseline-receipt-sha256", "candidate-receipt-sha256"):
        parser.add_argument("--" + name, required=True)
    result = run(**vars(parser.parse_args()))
    print(base.json.dumps(dict(completed=result["completed"], videos=len(result["videos"]), source_frames=91)))
