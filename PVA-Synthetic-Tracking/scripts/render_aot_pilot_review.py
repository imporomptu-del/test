#!/usr/bin/env python3
"""Render the complete approved AOT pilot for review, never detector input.

The left panel is unmarked source, globally resized by exactly one half. The
right panel has dataset annotations and qualified CURRENT measurements only.
No matching, object classification, prediction, coast, selection, crop, or
detector/tracker rerun is performed. All captions live outside both images.
Requires the frozen manifest and its download-validation pixel-hash ledger.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

import cv2
import numpy as np


FRAME_COUNT, FPS = 300, 10
WIDTH, HEIGHT = 2448, 2048
BAR_HEIGHT = 80
EXPECTED_VIDEO_SHA256 = "869c37637b68de5eb2c65a6140caebcea58f01833b653a1f2991fec3b16e4d6f"
EXPECTED_MANIFEST_SHA256 = "425b8220417e9853a8fbf272a1119d4e1989678984c84f295c9aeb2507b72d8d"
JSON_LIMIT = 32 * 1024 * 1024
ROW_LIMIT = 16 * 1024 * 1024
SHA_RE = re.compile(r"[0-9a-f]{64}")
GT_COLOR, MEASUREMENT_COLOR = (0, 255, 255), (255, 255, 0)  # BGR


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON key: " + key)
        result[key] = value
    return result


def _constant(value):
    raise ValueError("Nonfinite JSON constant: " + value)


def decode(raw):
    return json.loads(raw, object_pairs_hook=_pairs, parse_constant=_constant)


def load_json(path):
    require(path.stat().st_size <= JSON_LIMIT, "Oversize metadata: " + str(path))
    with path.open("rb") as stream:
        return decode(stream.read(JSON_LIMIT + 1))


def finite_number(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def coordinate(value):
    require(isinstance(value, list) and len(value) == 2 and all(map(finite_number, value)),
            "Measurement coordinate must contain two finite numbers")
    return value


def inside_image(point):
    return 0 <= point[0] < WIDTH and 0 <= point[1] < HEIGHT


def validate_manifest(manifest, validation, manifest_sha):
    require(isinstance(manifest, dict) and isinstance(validation, dict),
            "Manifest and download validation must be objects")
    flight = manifest.get("flight_id")
    require(manifest.get("part") == "part1" and isinstance(flight, str)
            and re.fullmatch(r"[a-f0-9]{32}", flight), "Unexpected source identity")
    frames, images = manifest.get("frames"), validation.get("images")
    require(isinstance(frames, list) and isinstance(images, list)
            and len(frames) == len(images) == FRAME_COUNT,
            "Exactly 300 manifest and validated source frames required")
    require(validation.get("manifest_sha256") == manifest_sha
            and type(validation.get("frames")) is int and validation["frames"] == FRAME_COUNT
            and validation.get("resolution") == [WIDTH, HEIGHT]
            and validation.get("dtype") == "uint8" and validation.get("grayscale") is True
            and validation.get("detector_run") is False,
            "Download validation must bind this manifest and native gray8 inventory")
    previous_frame = previous_time = None
    for row, checked in zip(frames, images):
        require(isinstance(row, dict) and isinstance(checked, dict), "Invalid source frame record")
        source_frame, timestamp = row.get("source_frame"), row.get("timestamp_ns")
        require(type(source_frame) is int and source_frame >= 0
                and (previous_frame is None or source_frame == previous_frame + 1),
                "Manifest source frames must be contiguous")
        require(isinstance(timestamp, str) and re.fullmatch(r"[0-9]{19}", timestamp)
                and (previous_time is None or int(timestamp) > previous_time),
                "Acquisition timestamps must be exact increasing nanosecond strings")
        require(row.get("img_name") == f"{timestamp}{flight}.png", "Source image identity changed")
        require(all(type(checked.get(key)) is type(row[key]) and checked[key] == row[key]
                    for key in ("source_frame", "timestamp_ns", "img_name")),
                "Validation inventory differs from frozen manifest")
        require(isinstance(checked.get("pixel_sha256"), str)
                and SHA_RE.fullmatch(checked["pixel_sha256"]), "Missing source pixel hash")
        entities = row.get("entities")
        require(isinstance(entities, list) and len(entities) <= 10000, "Invalid annotation list")
        ids = set()
        box_count = 0
        for entity in entities:
            require(isinstance(entity, dict), "Annotation must be an object")
            require(entity.get("flight_id") == flight and entity.get("img_name") == row["img_name"]
                    and type(entity.get("time")) is int and entity["time"] == int(timestamp)
                    and isinstance(entity.get("blob"), dict)
                    and type(entity["blob"].get("frame")) is int
                    and entity["blob"]["frame"] == source_frame,
                    "Annotation does not belong to its source frame")
            require(("bb" in entity) == ("id" in entity), "Incomplete annotation box/identity")
            if "bb" not in entity:
                continue
            identity, box = entity["id"], entity["bb"]
            require(isinstance(identity, str) and bool(identity.strip()) and len(identity) <= 128
                    and identity not in ids, "Invalid/duplicate annotation identity")
            require(isinstance(box, list) and len(box) == 4 and all(map(finite_number, box))
                    and box[2] > 0 and box[3] > 0,
                    "Dataset bb must be finite [left, top, width, height] with positive extent")
            x, y, width, height = box
            require(all(map(finite_number, (x + width, y + height)))
                    and x < WIDTH and y < HEIGHT and x + width > 0 and y + height > 0,
                    "Dataset box must intersect the native field")
            ids.add(identity)
            box_count += 1
        require(type(row.get("airborne_label_count")) is int
                and row["airborne_label_count"] == box_count, "Manifest annotation count differs")
        previous_frame, previous_time = source_frame, int(timestamp)
    require(validation.get("source_frame_range")
            == [frames[0]["source_frame"], frames[-1]["source_frame"]],
            "Validation source extent differs")


def journal_rows(path):
    """Validate every field used for rendering; keep at most one journal row."""
    previous_segment = None
    count = 0
    with path.open("rb") as stream:
        while True:
            raw = stream.readline(ROW_LIMIT + 1)
            if not raw:
                break
            require(len(raw) <= ROW_LIMIT, "Oversize journal row")
            require(count < FRAME_COUNT, "Journal has more than 300 frames")
            row = decode(raw)
            require(isinstance(row, dict), "Journal row must be an object")
            require(type(row.get("frame_index")) is int and row["frame_index"] == count,
                    "Journal must contain ordered frame indices 0..299")
            require(type(row.get("timestamp_ns")) is int
                    and row["timestamp_ns"] == count * 100_000_000,
                    "Journal timestamps must be the original nominal 10 Hz timeline")
            segment = row.get("segment")
            require(type(segment) is int and segment >= 0
                    and (previous_segment is None or segment >= previous_segment),
                    "Journal segments must be nonnegative and nondecreasing")
            tracks = row.get("tracks")
            require(isinstance(tracks, list) and len(tracks) <= 10000, "Explicit bounded track list required")
            seen, points = set(), []
            excluded_predictions = excluded_unqualified = 0
            for track in tracks:
                require(isinstance(track, dict), "Track record must be an object")
                identity = track.get("track_id")
                require(isinstance(identity, str) and bool(identity.strip()) and len(identity) <= 128
                        and "/" not in identity and identity not in seen,
                        "Unique nonempty track identity required")
                require(type(track.get("segment")) is int and track["segment"] == segment,
                        "Track segment must match its frame")
                measured, qualified = track.get("measured"), track.get("qualified_moving")
                require(type(measured) is bool and type(qualified) is bool,
                        "Explicit measured and qualified booleans required")
                require("measurement_source_xy" in track, "Current measurement field required")
                if measured:
                    actual = coordinate(track["measurement_source_xy"])
                    if qualified:
                        points.append(tuple(actual))
                    else:
                        excluded_unqualified += 1
                else:
                    require(track["measurement_source_xy"] is None,
                            "Prediction/coast cannot carry a current measurement")
                    excluded_predictions += 1
                seen.add(identity)
            yield count, points, excluded_predictions, excluded_unqualified
            previous_segment = segment
            count += 1
    require(count == FRAME_COUNT, "Journal has fewer than 300 frames")


def probe(path, ffprobe, output=False):
    command = [ffprobe, "-v", "error", "-select_streams", "v:0"]
    if output:
        command += ["-count_frames", "-show_frames"]
    command += ["-show_entries", "stream=codec_name,pix_fmt,width,height,avg_frame_rate,r_frame_rate,nb_frames,nb_read_frames,duration:frame=best_effort_timestamp_time",
                "-of", "json", str(path)]
    result = decode(subprocess.check_output(command, timeout=120))
    streams = result.get("streams")
    require(isinstance(streams, list) and len(streams) == 1, "Exactly one selected video stream required")
    stream = streams[0]
    require(Fraction(stream["avg_frame_rate"]) == Fraction(FPS, 1)
            and Fraction(stream["r_frame_rate"]) == Fraction(FPS, 1), "Video cadence differs from 10 Hz")
    require(int(stream["nb_frames"]) == FRAME_COUNT, "Declared video frame count differs from 300")
    if output:
        require(stream["codec_name"] == "h264" and stream["pix_fmt"] == "yuv420p"
                and (stream["width"], stream["height"]) == (WIDTH, HEIGHT // 2 + BAR_HEIGHT)
                and int(stream["nb_read_frames"]) == FRAME_COUNT
                and Fraction(stream["duration"]) == Fraction(FRAME_COUNT, FPS),
                "Review video format/count/duration mismatch")
        frames = result.get("frames")
        require(isinstance(frames, list) and len(frames) == FRAME_COUNT,
                "Output decode does not contain exactly 300 frames")
        require(all(Fraction(frame["best_effort_timestamp_time"]) == Fraction(index, FPS)
                    for index, frame in enumerate(frames)), "Output frames were retimed/resampled")
    else:
        require(stream["codec_name"] == "ffv1" and stream["pix_fmt"] in {"gray", "gray8"}
                and (stream["width"], stream["height"]) == (WIDTH, HEIGHT),
                "Source must be native 2448x2048 gray8 FFV1")
    return result


def render_frame(gray, source_row, points, index):
    panel = cv2.cvtColor(cv2.resize(gray, (WIDTH // 2, HEIGHT // 2), interpolation=cv2.INTER_AREA),
                         cv2.COLOR_GRAY2BGR)
    canvas = np.zeros((HEIGHT // 2 + BAR_HEIGHT, WIDTH, 3), dtype=np.uint8)
    canvas[:HEIGHT // 2, :WIDTH // 2] = panel
    marked = canvas[:HEIGHT // 2, WIDTH // 2:]
    marked[:] = panel
    for entity in source_row["entities"]:
        if "bb" not in entity:
            continue
        x, y, width, height = entity["bb"]
        x1 = min(WIDTH // 2 - 1, int(round(max(0, x) / 2)))
        y1 = min(HEIGHT // 2 - 1, int(round(max(0, y) / 2)))
        x2 = min(WIDTH // 2 - 1, int(round(min(WIDTH, x + width) / 2)))
        y2 = min(HEIGHT // 2 - 1, int(round(min(HEIGHT, y + height) / 2)))
        cv2.rectangle(marked, (x1, y1), (x2, y2), GT_COLOR, 2)
    for x, y in points:
        if not inside_image((x, y)):
            continue  # Offscreen observations are counted, never clamped into edge markers.
        point = (min(WIDTH // 2 - 1, int(round(x / 2))), min(HEIGHT // 2 - 1, int(round(y / 2))))
        cv2.drawMarker(marked, point, MEASUREMENT_COLOR, cv2.MARKER_CROSS, 15, 2)
        cv2.circle(marked, point, 7, MEASUREMENT_COLOR, 1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    base = HEIGHT // 2
    cv2.putText(canvas, "SOURCE | full field, unmarked, 0.5x", (12, base + 20), font, .55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"ANNOTATIONS + MEASURED OUTPUT | frame {index:03d}/299 | t={index / FPS:04.1f}s",
                (WIDTH // 2 + 12, base + 20), font, .55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Yellow GT = dataset annotation, not a detector result", (12, base + 44), font, .50, GT_COLOR, 1, cv2.LINE_AA)
    cv2.putText(canvas, "Cyan = qualified actual measurement; enlarged marker, not object extent", (WIDTH // 2 + 12, base + 44), font, .50, MEASUREMENT_COLOR, 1, cv2.LINE_AA)
    cv2.putText(canvas, "DISPLAY ONLY - NOT DETECTOR INPUT | 0.5x downscaling can hide few-pixel targets | no predictions/coasts | no true/false-object labels", (12, base + 68), font, .50, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def stop_process(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def encode(video, output, manifest, validation, journal, ffmpeg):
    decode_command = [ffmpeg, "-nostdin", "-v", "error", "-threads", "1", "-i", str(video),
                      "-map", "0:v:0", "-an", "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"]
    encode_command = [ffmpeg, "-nostdin", "-n", "-v", "error", "-f", "rawvideo", "-pixel_format", "bgr24",
                      "-video_size", f"{WIDTH}x{HEIGHT // 2 + BAR_HEIGHT}", "-framerate", str(FPS), "-i", "pipe:0",
                      "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
                      "-threads", "1", "-fps_mode", "passthrough", "-movflags", "+faststart", str(output)]
    counts = {"frames": 0, "annotation_boxes": 0, "qualified_measured_observations": 0,
              "rendered_qualified_measurements": 0, "offscreen_qualified_measurements_not_drawn": 0,
              "excluded_prediction_or_coast_states": 0, "excluded_unqualified_measurements": 0}
    with tempfile.TemporaryFile() as decode_errors, tempfile.TemporaryFile() as encode_errors:
        decoder = subprocess.Popen(decode_command, stdout=subprocess.PIPE, stderr=decode_errors)
        encoder = None
        try:
            encoder = subprocess.Popen(encode_command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=encode_errors)
            for index, points, excluded_predictions, excluded_unqualified in journal_rows(journal):
                pixels = decoder.stdout.read(WIDTH * HEIGHT)
                require(len(pixels) == WIDTH * HEIGHT, f"Source ended early at frame {index}")
                require(hashlib.sha256(pixels).hexdigest() == validation["images"][index]["pixel_sha256"],
                        f"Native source pixel SHA-256 differs at frame {index}")
                gray = np.frombuffer(pixels, dtype=np.uint8).reshape(HEIGHT, WIDTH)
                row = manifest["frames"][index]
                encoder.stdin.write(render_frame(gray, row, points, index).tobytes(order="C"))
                counts["frames"] += 1
                counts["annotation_boxes"] += row["airborne_label_count"]
                counts["qualified_measured_observations"] += len(points)
                inside_count = sum(inside_image(point) for point in points)
                counts["rendered_qualified_measurements"] += inside_count
                counts["offscreen_qualified_measurements_not_drawn"] += len(points) - inside_count
                counts["excluded_prediction_or_coast_states"] += excluded_predictions
                counts["excluded_unqualified_measurements"] += excluded_unqualified
                if (index + 1) % 25 == 0:
                    print(f"Rendered and pixel-verified {index + 1}/{FRAME_COUNT} complete frames", flush=True)
            require(not decoder.stdout.read(1), "Source has more than 300 frames")
            require(decoder.wait(timeout=30) == 0, "Source decoding failed")
            encoder.stdin.close()
            require(encoder.wait(timeout=60) == 0, "Review encoding failed")
        except Exception:
            for label, errors in (("decoder", decode_errors), ("encoder", encode_errors)):
                errors.seek(0)
                detail = errors.read(8192).decode("utf-8", errors="replace").strip()
                if detail:
                    print(f"{label}: {detail}", file=sys.stderr)
            raise
        finally:
            stop_process(decoder)
            decoder.stdout.close()
            if encoder is not None:
                stop_process(encoder)
                if not encoder.stdin.closed:
                    encoder.stdin.close()
    return counts, decode_command, encode_command


def render(video, manifest, journal, validation, output, receipt):
    inputs = {name: Path(value).absolute() for name, value in
              (("video", video), ("manifest", manifest), ("journal", journal), ("validation", validation))}
    for name, path in inputs.items():
        require(path.is_file() and not path.is_symlink(), f"Missing/linked {name} input: {path}")
    inputs = {name: path.resolve(strict=True) for name, path in inputs.items()}
    output, receipt = Path(output).absolute(), Path(receipt).absolute()
    require(output.suffix.lower() == ".mp4" and receipt.suffix.lower() == ".json", "MP4 output and JSON receipt required")
    require(len(set(inputs.values()) | {output.resolve(), receipt.resolve()}) == 6, "Distinct inputs and outputs required")
    for path in (output, receipt):
        require(not path.exists() and not path.is_symlink(), "Refusing overwrite: " + str(path))
        require(path.parent.is_dir(), "Output parent must already exist: " + str(path.parent))
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    require(ffmpeg and ffprobe, "ffmpeg and ffprobe are required")
    identities = {name: sha256(path) for name, path in inputs.items()}
    require(identities["video"] == EXPECTED_VIDEO_SHA256, "Video is not the approved native pilot SHA-256")
    require(identities["manifest"] == EXPECTED_MANIFEST_SHA256, "Manifest is not the approved frozen pilot SHA-256")
    saved_manifest, saved_validation = load_json(inputs["manifest"]), load_json(inputs["validation"])
    validate_manifest(saved_manifest, saved_validation, identities["manifest"])
    for _ in journal_rows(inputs["journal"]):
        pass  # Validate the complete journal before starting any media output.
    source_probe = probe(inputs["video"], ffprobe)
    with tempfile.TemporaryDirectory(prefix=".aot_review_", dir=output.parent) as directory:
        temporary_video = Path(directory) / "review.mp4"
        counts, decoder_command, encoder_command = encode(inputs["video"], temporary_video, saved_manifest,
                                                          saved_validation, inputs["journal"], ffmpeg)
        output_probe = probe(temporary_video, ffprobe, output=True)
        require(all(sha256(path) == identities[name] for name, path in inputs.items()),
                "Input artifacts changed during rendering")
        document = {
            "schema": "seaqr.aot.pilot-review.v1", "passed": True,
            "inputs": {name: str(path) for name, path in inputs.items()}, "inputs_sha256": identities,
            "renderer": str(Path(__file__).resolve()), "renderer_sha256": sha256(__file__),
            "output": str(output), "output_sha256": sha256(temporary_video),
            "output_bytes": temporary_video.stat().st_size, "output_resolution": [WIDTH, HEIGHT // 2 + BAR_HEIGHT],
            "source_resolution": [WIDTH, HEIGHT], "panel_resolution": [WIDTH // 2, HEIGHT // 2],
            "external_caption_bar_height": BAR_HEIGHT, "fps": FPS, "duration_seconds": FRAME_COUNT / FPS,
            "frame_index_range": [0, FRAME_COUNT - 1], "pixel_hashes_verified": FRAME_COUNT,
            "source_frame_range": saved_validation["source_frame_range"], "counts": counts,
            "source_probe": source_probe, "output_probe": output_probe,
            "decoder_command": decoder_command, "encoder_command": encoder_command,
            "versions": {"python": sys.version, "opencv": cv2.__version__, "numpy": np.__version__,
                         "ffmpeg": subprocess.check_output([ffmpeg, "-version"], text=True, timeout=10).splitlines()[0],
                         "ffprobe": subprocess.check_output([ffprobe, "-version"], text=True, timeout=10).splitlines()[0]},
            "display_only_not_detector_input": True, "detector_or_tracker_rerun": False,
            "qualification_changed": False, "predictions_or_coasts_rendered": False,
            "frame_selection_crop_or_resampling": False,
            "measurement_coordinate": "measurement_source_xy, only measured=true AND qualified_moving=true",
            "left_panel": "Unmarked native source, globally resized exactly 0.5x with INTER_AREA; captions outside image",
            "right_panel": "Same resized source plus yellow dataset GT boxes and cyan qualified current measurement markers",
            "legend": {"yellow": "GT = dataset annotation, not detector output", "cyan": "Qualified actual measurement; enlarged marker, not object extent"},
            "limitations": ["Display downscaling and lossy H.264 can hide few-pixel targets; this MP4 is NOT detector input.",
                            "Observations are not object counts or airborne classifications; no true/false-object labels or GT association performed.",
                            "10 Hz nominal container/baseline timeline; exact irregular acquisition timestamps remain unchanged in the manifest.",
                            "Offscreen qualified measurements, if any, are counted but not drawn or clamped to image edges.",
                            "Annotation absence is not proof that no physical object is present."],
            "derived_data_notice": {"provider": "Amazon Airborne Object Tracking",
                                    "license": "https://cdla.dev/permissive-1-0/",
                                    "transformation": "Complete 300-frame display-only side-by-side review; 0.5x panels, overlays, captions, H.264 encoding."},
        }
        # Hard-link publication is exclusive even if another process creates the destination mid-render.
        os.link(temporary_video, output)
        with receipt.open("x", encoding="utf-8") as stream:
            json.dump(document, stream, indent=2, allow_nan=False)
            stream.write("\n")
    return document


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("video", "manifest", "journal", "validation", "output", "receipt"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    try:
        result = render(**vars(args))
    except (ValueError, OSError, KeyError, TypeError, subprocess.SubprocessError) as error:
        parser.exit(1, f"Review render failed: {error}\n")
    print(json.dumps({key: result[key] for key in ("passed", "output", "output_sha256", "counts")}, indent=2))


if __name__ == "__main__":
    main()
