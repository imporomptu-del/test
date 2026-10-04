#!/usr/bin/env python3
"""Bound, read-only presentation of completed discovery runs; no classification.

Run AFTER inference. The overview includes every source frame. Native crop
windows are detector-selected, bounded, nonexhaustive review aids, not truth.
Only this file and visible_output.py are required beside a remote invocation.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import subprocess

import cv2
import numpy as np

SIBLING = Path(__file__).resolve().with_name("visible_output.py")
POLICY_PATH = SIBLING if SIBLING.is_file() else Path(__file__).resolve().parents[1] / "tiny_target/visible_output.py"
_spec = importlib.util.spec_from_file_location("discovery_observation_policy", POLICY_PATH)
_policy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_policy)
ObservationOutput = _policy.ObservationOutput

GREEN, AMBER, WHITE = (75, 235, 105), (0, 190, 255), (238, 241, 245)
HEADER, FOOTER, GAP, CROP, PANEL_WIDTH = 112, 76, 16, 384, 960
ROW_LIMIT = 16 * 1024 * 1024


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def _pairs(values):
    out = {}
    for key, value in values:
        require(key not in out, "Duplicate JSON key: " + key)
        out[key] = value
    return out


def _float(value):
    number = float(value)
    require(math.isfinite(number), "Nonfinite JSON number")
    return number


def _invalid(value):
    raise ValueError("Nonfinite JSON constant: " + value)


def decode(raw):
    return json.loads(raw, object_pairs_hook=_pairs, parse_float=_float, parse_constant=_invalid)


def write_json(path, data):
    with Path(path).open("x") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)


def journal_rows(path, frames, fps, clip):
    """Validate complete timeline and all policy fields, including empty rows."""
    require(type(frames) is int and frames > 0, "Positive frame count required")
    policy = ObservationOutput(clip)
    count = 0
    with Path(path).open("rb") as stream:
        while True:
            raw = stream.readline(ROW_LIMIT + 1)
            if not raw:
                break
            require(len(raw) <= ROW_LIMIT, "Oversize journal row")
            row = decode(raw)
            require(isinstance(row, dict) and type(row.get("frame_index")) is int
                    and row["frame_index"] == count and count < frames,
                    "Missing, extra, reordered, or nonzero-start journal frame")
            require(type(row.get("timestamp_ns")) is int
                    and row["timestamp_ns"] == round(count / float(fps) * 1e9),
                    "Journal differs from source nominal timeline")
            channels = policy.update(row)
            yield row, channels
            count += 1
    require(count == frames, "Incomplete journal")


def inside(point, width, height):
    return 0 <= point[0] < width and 0 <= point[1] < height


def crop_at(point, width, height):
    require(width >= CROP and height >= CROP, "Source too small for native 384 crop")
    # Shift the VIEW to the native edge, never clamp an observation coordinate.
    x = min(max(math.floor(point[0]) - CROP // 2, 0), width - CROP)
    y = min(max(math.floor(point[1]) - CROP // 2, 0), height - CROP)
    return [x, y, CROP, CROP]


def select_windows(channels, frames, fps, width, height):
    """Six frame bins; displacement ranking uses actual qualified samples only."""
    bins = [dict(bin=i, first_frame=(i * frames + 5) // 6,
                 last_frame_exclusive=((i + 1) * frames + 5) // 6, candidates={})
            for i in range(6)]
    for row in channels:
        frame = row["frame_index"]
        bucket = bins[min(5, frame * 6 // frames)]["candidates"]
        for record in row["observation_alerts"]:
            require(record["measured"] and record["coordinate_kind"] == "measurement",
                    "Prediction cannot anchor a crop")
            identity, point = record["identity"], record["source_xy"]
            candidate = bucket.setdefault(identity, dict(identity=identity, track_id=record["track_id"],
                segment=record["segment"], samples=0, first_frame=frame,
                first_xy=point, last_frame=frame, last_xy=point, offscreen_samples=0))
            candidate["samples"] += 1
            candidate["last_frame"], candidate["last_xy"] = frame, point
            candidate["offscreen_samples"] += not inside(point, width, height)
    chosen_ids, windows = set(), []
    for bucket in bins:
        candidates = sorted(bucket.pop("candidates").values(), key=lambda v: (v["first_frame"], v["identity"]))
        for c in candidates:
            c["net_displacement_px"] = math.hypot(c["last_xy"][0] - c["first_xy"][0], c["last_xy"][1] - c["first_xy"][1])
            c["anchor_inside_source"] = inside(c["first_xy"], width, height)
            c["already_chosen"] = c["identity"] in chosen_ids
        eligible = [c for c in candidates if c["anchor_inside_source"] and not c["already_chosen"]]
        multiple = [c for c in eligible if c["samples"] >= 2]
        ranked = sorted(multiple, key=lambda c: (-c["net_displacement_px"], c["first_frame"], c["identity"])) if multiple else eligible
        winner = ranked[0] if ranked else None
        bucket.update(candidates=candidates, eligible_count=len(eligible), chosen_identity=winner["identity"] if winner else None,
                      ranking="largest first-to-last actual qualified displacement" if multiple else "earliest actual qualified fallback",
                      omitted_count=len(candidates) - bool(winner))
        for c in candidates:
            c["selected"] = c is winner
            c["omission_reason"] = (None if c is winner else "offscreen_first_measurement" if not c["anchor_inside_source"]
                else "identity_selected_in_earlier_bin" if c["already_chosen"] else "not_first_in_fixed_bin_ranking")
        if winner:
            chosen_ids.add(winner["identity"])
            anchor = winner["first_frame"]
            first = max(0, anchor - round(2 * float(fps)))
            # Exclusive end: a complete window has exactly six seconds of frames.
            stop = min(frames, anchor + round(4 * float(fps)))
            windows.append(dict(name=f"bin{bucket['bin'] + 1}_native", bin=bucket["bin"],
                identity=winner["identity"], anchor_frame=anchor, anchor_xy=winner["first_xy"],
                first_frame=first, last_frame_inclusive=stop - 1, frames=stop - first,
                crop_xywh=crop_at(winner["first_xy"], width, height)))
    return dict(schema="seaqr.discovery-review-selection.v1", detector_selected=True,
        physical_class="unknown", airborne_truth=False, exhaustive=False,
        criterion="Six equal frame bins; one unused identity per bin; >=2 actual qualified samples ranked by first-to-last net displacement; otherwise earliest actual qualified identity. Ties first frame then identity. No other-bin replacement. Offscreen first anchors retained but ineligible.",
        bins=bins, windows=windows, chosen_count=len(windows),
        candidate_identity_bin_count=sum(len(b["candidates"]) for b in bins),
        omitted_identity_bin_count=sum(b["omitted_count"] for b in bins))


def records_inside(records, roi):
    x, y, w, h = roi
    return [r for r in records if x <= r["source_xy"][0] < x + w and y <= r["source_xy"][1] < y + h]


def text(canvas, message, x, y, width, color=WHITE, scale=.58):
    size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0]
    scale *= min(1, width / max(1, size))
    cv2.putText(canvas, message, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def annotate(panel, records, roi):
    x, y, w, h = roi
    sx, sy = panel.shape[1] / w, panel.shape[0] / h
    for record in records_inside(records, roi):
        px = min(panel.shape[1] - 1, round((record["source_xy"][0] - x) * sx))
        py = min(panel.shape[0] - 1, round((record["source_xy"][1] - y) * sy))
        measured = record["measured"]
        color, radius = (GREEN if measured else AMBER), 9
        if measured:
            cv2.circle(panel, (px, py), radius, color, 1, cv2.LINE_AA)
            label = record["track_id"] + " M"
        else:
            for sign in (-1, 1):
                for a, b in ((-9, -4), (4, 9)):
                    cv2.line(panel, (px + sign * radius, py + a), (px + sign * radius, py + b), color, 1)
                    cv2.line(panel, (px + a, py + sign * radius), (px + b, py + sign * radius), color, 1)
            age = record["last_measurement_age_ns"]
            label = record["track_id"] + (f" P +{age / 1e9:.1f}s" if age is not None else " P age?")
        label_width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, .36, 1)[0][0]
        tx, ty = max(0, min(px + 12, panel.shape[1] - label_width - 1)), max(12, min(py - 10, panel.shape[0] - 2))
        cv2.putText(panel, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, .36, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(panel, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, .36, color, 1, cv2.LINE_AA)


def canvas_for(frame, channels, clip, fps, crop=None):
    height, width = frame.shape[:2]
    roi = crop or [0, 0, width, height]
    if crop is None:
        ph = max(1, round(height * PANEL_WIDTH / width))
        raw = cv2.resize(frame, (PANEL_WIDTH, ph), interpolation=cv2.INTER_AREA)
        marked = raw.copy()
        annotate(marked, channels["track_context"], roi)
        panels, labels = [raw, marked], ["SOURCE / unmarked full field", "QUALIFIED OUTPUT / M measurements, P predictions"]
    else:
        x, y, w, h = crop
        require(x >= 0 and y >= 0 and x + w <= width and y + h <= height, "Crop outside native source")
        raw = frame[y:y + h, x:x + w].copy()
        fresh, context = raw.copy(), raw.copy()
        annotate(fresh, channels["observation_alerts"], roi)
        annotate(context, channels["track_context"], roi)
        panels, labels = [raw, fresh, context], ["SOURCE / native 1:1 / unmarked", "MEASURED NOW / qualified observations", "TRACK CONTEXT / gaps retained"]
    ph, pw = raw.shape[:2]
    ww, hh = len(panels) * pw + (len(panels) - 1) * GAP, HEADER + ph + FOOTER
    ww += ww % 2
    hh += hh % 2
    canvas = np.full((hh, ww, 3), 23, np.uint8)
    for i, (panel, label) in enumerate(zip(panels, labels)):
        xx = i * (pw + GAP)
        canvas[HEADER:HEADER + ph, xx:xx + pw] = panel
        text(canvas, label, xx + 8, 98, pw - 16)
    require(np.array_equal(canvas[HEADER:HEADER + ph, :pw], raw), "Raw left panel changed")
    text(canvas, f"clip {clip} | source frame {channels['frame_index']} | t={channels['timestamp_ns'] / 1e9:.3f}s | playback {float(fps):g} FPS (not processing FPS)", 12, 27, ww - 24)
    text(canvas, "Green circle M = current measurement; dashed amber P = prediction only; +time = age since measurement.", 12, 53, ww - 24)
    text(canvas, "Physical class unknown. No verified airborne labels. Unqualified states are not drawn.", 12, 76, ww - 24)
    alerts, context = records_inside(channels["observation_alerts"], roi), records_inside(channels["track_context"], roi)
    predictions = sum(not r["measured"] for r in context)
    text(canvas, f"In view: {len(alerts)} current qualified measurements; {predictions} predictions. All nearby qualified states shown.", 12, HEADER + ph + 25, ww - 24)
    note = "Full clip / downscaled display can hide tiny targets; not detector input." if crop is None else "Detector-selected fixed native crop; nonexhaustive review, not accuracy or object-count evidence."
    text(canvas, note, 12, HEADER + ph + 47, ww - 24)
    text(canvas, "No contrast enhancement. H264 is a lossy viewing copy; original source remains unchanged.", 12, HEADER + ph + 67, ww - 24)
    return canvas


def probe(path, count=False):
    command = ["ffprobe", "-v", "error", "-select_streams", "v:0"]
    if count:
        command += ["-count_frames", "-show_frames"]
    command += ["-show_entries", "stream=codec_name,pix_fmt,width,height,avg_frame_rate,r_frame_rate,nb_frames,nb_read_frames,duration:frame=best_effort_timestamp_time", "-of", "json", str(path)]
    data = decode(subprocess.check_output(command))
    require(len(data.get("streams", [])) == 1, "One video stream required")
    return data


class Encoder:
    def __init__(self, path, shape, fps, expected):
        self.path, self.shape, self.fps, self.expected, self.count = path, shape, fps, expected, 0
        self.process = subprocess.Popen(["ffmpeg", "-nostdin", "-v", "error", "-n", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{shape[1]}x{shape[0]}",
            "-framerate", str(fps), "-i", "pipe:0", "-an", "-c:v", "libx264", "-threads", "2", "-preset", "fast", "-crf", "12", "-pix_fmt", "yuv420p", "-fps_mode", "passthrough", "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE)

    def send(self, canvas):
        require(canvas.shape == self.shape and canvas.dtype == np.uint8, "Changed encoder shape/type")
        self.process.stdin.write(canvas.tobytes())
        self.count += 1

    def finish(self):
        self.process.stdin.close()
        require(self.process.wait(timeout=60) == 0 and self.count == self.expected, "Incomplete encode")
        data = probe(self.path, count=True)
        stream = data["streams"][0]
        require(stream["codec_name"] == "h264" and stream["pix_fmt"] == "yuv420p"
                and (stream["height"], stream["width"]) == self.shape[:2]
                and int(stream["nb_read_frames"]) == self.expected
                and Fraction(stream["avg_frame_rate"]) == self.fps, "Encoded stream mismatch")
        timestamps = data.get("frames", [])
        require(len(timestamps) == self.expected and all(abs(float(t["best_effort_timestamp_time"]) - i / float(self.fps)) <= 1e-6 for i, t in enumerate(timestamps)), "Encoded frames retimed/resampled")
        return dict(ffprobe_stream=stream, verified_timestamp_count=len(timestamps), sha256=sha(self.path), bytes=self.path.stat().st_size)

    def stop(self):
        if self.process.poll() is None:
            self.process.terminate()
            self.process.wait(timeout=30)


def run(source, journal, report, output, clip):
    source, journal, report, output = map(lambda p: Path(p).resolve(), (source, journal, report, output))
    require(clip in ("0170", "0240"), "Only frozen discovery clips 0170/0240 allowed")
    require(not output.exists(), "Fresh output directory required")
    require(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg and ffprobe required")
    require("libx264" in subprocess.check_output(["ffmpeg", "-hide_banner", "-encoders"], text=True), "libx264 required")
    inputs = {"source": source, "journal": journal, "report": report, "renderer": Path(__file__).resolve(), "observation_policy": POLICY_PATH}
    hashes = {name: sha(path) for name, path in inputs.items()}
    saved = decode(report.read_bytes())
    require(saved.get("completed") is True and saved.get("full_clip") is True and saved.get("source_sha256") == hashes["source"], "Completed full-clip report/source binding required")
    frames = saved.get("frames")
    require(type(frames) is int and frames > 0, "Positive report frame count required")
    media = probe(source)["streams"][0]
    width, height, fps = media["width"], media["height"], Fraction(media["avg_frame_rate"])
    require(width >= CROP and height >= CROP and fps > 0 and Fraction(media["r_frame_rate"]) == fps, "Unexpected source geometry/cadence")
    require(media["pix_fmt"] in {"yuv420p", "yuvj420p", "gray", "gray8", "bgr24", "rgb24", "yuvj422p", "yuv422p"}, "Only calibrated 8-bit input formats allowed")
    require(int(media["nb_frames"]) == frames, "Source/report declared frame counts differ")
    selection = select_windows((channels for _, channels in journal_rows(journal, frames, fps, clip)), frames, fps, width, height)
    require(all(sha(path) == hashes[name] for name, path in inputs.items()), "Inputs changed during selection")
    output.mkdir(parents=True)
    (output / "videos").mkdir()
    (output / "qa").mkdir()
    selection.update(clip=clip, source_sha256=hashes["source"], journal_sha256=hashes["journal"], renderer_sha256=hashes["renderer"])
    write_json(output / "selection.json", selection)
    jobs = [dict(name="complete_overview", first_frame=0, last_frame_inclusive=frames - 1, frames=frames, crop_xywh=None)] + selection["windows"]
    encoders, receipts, qa = {}, [], []
    counts = dict(frames=0, qualified_measurements=0, qualified_predictions=0, offscreen_measurements=0, offscreen_predictions=0)
    cap = cv2.VideoCapture(str(source))
    try:
        require(cap.isOpened(), "Cannot decode bound source")
        for row, channels in journal_rows(journal, frames, fps, clip):
            ok, image = cap.read()
            require(ok and image.shape == (height, width, 3) and image.dtype == np.uint8, "Missing/wrong native frame")
            index = row["frame_index"]
            counts["frames"] += 1
            for record in channels["track_context"]:
                kind = "measurements" if record["measured"] else "predictions"
                counts["qualified_" + kind] += 1
                counts["offscreen_" + kind] += not inside(record["source_xy"], width, height)
            for job in jobs:
                if not job["first_frame"] <= index <= job["last_frame_inclusive"]:
                    continue
                name = job["name"]
                canvas = canvas_for(image, channels, clip, fps, job["crop_xywh"])
                if name not in encoders:
                    encoders[name] = Encoder(output / "videos" / (name + ".mp4"), canvas.shape, fps, job["frames"])
                encoders[name].send(canvas)
                qa_frames = [job["first_frame"], (job["first_frame"] + job["last_frame_inclusive"]) // 2, job["last_frame_inclusive"]]
                for slot, qa_frame in enumerate(qa_frames):
                    if index == qa_frame:
                        dest = output / "qa" / f"{name}_{slot}_f{index:06d}.png"
                        require(cv2.imwrite(str(dest), canvas), "QA PNG write failed")
                        qa.append(dict(video=name, slot=slot, source_frame=index, path=str(dest.relative_to(output)), sha256=sha(dest), before_lossy_encoding=True))
            if (index + 1) % 100 == 0:
                print(f"clip {clip}: rendered {index + 1}/{frames}", flush=True)
        require(not cap.read()[0], "Extra source frame")
        for job in jobs:
            require(job["name"] in encoders, "Unrendered video")
            encoded = encoders[job["name"]].finish()
            # Decode every emitted frame, and preserve three encoded QA samples
            # separately from the three exact pre-encode source/canvas samples.
            check = cv2.VideoCapture(str(encoders[job["name"]].path))
            decoded, encoded_qa = 0, []
            originals = [v for v in qa if v["video"] == job["name"]]
            try:
                require(check.isOpened(), "Encoded video cannot be opened")
                while True:
                    ok, pixels = check.read()
                    if not ok:
                        break
                    require(decoded < job["frames"] and pixels.shape == encoders[job["name"]].shape,
                            "Encoded decode count/geometry mismatch")
                    source_frame = job["first_frame"] + decoded
                    for original in originals:
                        if original["source_frame"] != source_frame:
                            continue
                        dest = output / "qa" / f"{job['name']}_encoded_{original['slot']}_f{source_frame:06d}.png"
                        require(cv2.imwrite(str(dest), pixels), "Encoded QA PNG write failed")
                        reference = cv2.imread(str(output / original["path"]))
                        require(reference is not None and reference.shape == pixels.shape, "Missing unencoded QA reference")
                        ph = CROP if job["crop_xywh"] else round(height * PANEL_WIDTH / width)
                        pw = CROP if job["crop_xywh"] else PANEL_WIDTH
                        difference = np.abs(pixels[HEADER:HEADER + ph, :pw].astype(np.int16)
                                            - reference[HEADER:HEADER + ph, :pw].astype(np.int16))
                        encoded_qa.append(dict(slot=original["slot"], source_frame=source_frame,
                            path=str(dest.relative_to(output)), sha256=sha(dest),
                            raw_left_lossy_error_mean_dn=float(difference.mean()),
                            raw_left_lossy_error_max_dn=int(difference.max())))
                    decoded += 1
            finally:
                check.release()
            require(decoded == job["frames"] and len(encoded_qa) == 3, "Incomplete encoded decode/QA inventory")
            receipts.append(dict(**job, path="videos/" + job["name"] + ".mp4", **encoded,
                                 independently_decoded_frames=decoded, encoded_qa=encoded_qa))
        require(len(qa) == 3 * len(jobs), "Three QA frames per video required")
        require(all(sha(path) == hashes[name] for name, path in inputs.items()), "Bound inputs changed during rendering")
        result = dict(schema="seaqr.discovery-pair-review.v1", completed=True, clip=clip,
            inputs={name: dict(path=str(path), sha256=hashes[name]) for name, path in inputs.items()},
            source_probe=media, counts=counts, selection_sha256=sha(output / "selection.json"), videos=receipts, qa=qa,
            raw_left_exact_before_encoding=True, no_temporal_resampling=True, contrast_enhancement=False,
            detector_or_tracker_rerun=False, physical_class="unknown", airborne_truth=False,
            all_qualified_states_in_view=True, unqualified_states_hidden=True,
            encoding="H264 CRF12 yuv420p, two CPU encoder threads; lossy viewing copies, not detector input",
            timestamp_basis="source nominal container playback; not physical acquisition verification or processing throughput",
            versions=dict(opencv=cv2.__version__, numpy=np.__version__, ffmpeg=subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0]))
        write_json(output / "receipt.json", result)
        return result
    finally:
        cap.release()
        for encoder in encoders.values():
            encoder.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "journal", "report", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--clip", choices=("0170", "0240"), required=True)
    args = parser.parse_args()
    receipt = run(**vars(args))
    print(json.dumps(dict(completed=receipt["completed"], videos=len(receipt["videos"]), counts=receipt["counts"])))


if __name__ == "__main__":
    main()
