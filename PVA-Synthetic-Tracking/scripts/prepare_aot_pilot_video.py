#!/usr/bin/env python3
"""Package the validated 300-frame AOT pilot locally; never run a detector.

Requires the frozen intake manifest and completed download validation. Native
gray8 PNG pixels are streamed without filters/rescaling into gray FFV1 AVI at
nominal 10 Hz, then checked through the actual prefetch-one visible reader.
Acquisition timestamps remain in the unchanged manifest; the container does NOT
use those potentially irregular timestamps. A fresh output directory is required.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tiny_target.frame_source import probe_video
from tiny_target.visible_decode import VisibleFrameReader


FRAME_CAP = 300
PNG_BYTES_CAP = 1024 ** 3
WIDTH, HEIGHT, FPS = 2448, 2048, 10
SHA_PATTERN = re.compile(r"[a-f0-9]{64}")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha256(path):
    """Bound memory even for the multi-hundred-MiB derived video."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path):
    require(path.is_file() and not path.is_symlink(), f"missing/linked input: {path}")
    require(path.stat().st_size <= 32 * 1024 * 1024, f"oversize metadata: {path}")
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def validate_inputs(source):
    manifest_path = source / "frozen_image_manifest.json"
    validation_path = source / "download_validation.json"
    manifest, validation = load_json(manifest_path), load_json(validation_path)
    identities = {p.name: file_sha256(p) for p in (manifest_path, validation_path)}
    require(validation.get("manifest_sha256") == identities[manifest_path.name],
            "download validation does not match frozen manifest")
    require(manifest.get("part") == "part1"
            and re.fullmatch(r"[a-f0-9]{32}", manifest.get("flight_id", "")),
            "unsupported pilot source identity")
    rows, images = manifest.get("frames"), validation.get("images")
    require(isinstance(rows, list) and isinstance(images, list)
            and len(rows) == len(images) == FRAME_CAP,
            "packaging requires exactly the frozen 300-frame pilot")
    require(type(validation.get("frames")) is int and validation["frames"] == FRAME_CAP
            and validation.get("resolution") == [WIDTH, HEIGHT]
            and validation.get("dtype") == "uint8" and validation.get("grayscale") is True
            and validation.get("detector_run") is False,
            "missing successful native-gray8 download validation")
    previous_index, previous_time, total = None, None, 0
    for row, checked in zip(rows, images):
        index, timestamp = row["source_frame"], row["timestamp_ns"]
        require(type(index) is int and index >= 0, "invalid source frame index")
        require(isinstance(timestamp, str) and re.fullmatch(r"[0-9]{19}", timestamp),
                "source timestamp must remain an exact nanosecond string")
        require(previous_index is None or index == previous_index + 1,
                "missing/reordered causal input frame")
        require(previous_time is None or int(timestamp) > previous_time,
                "non-increasing source timestamp")
        expected_name = f"{timestamp}{manifest['flight_id']}.png"
        require(row["img_name"] == expected_name, "unexpected source PNG name")
        require(all(checked.get(key) == row[key]
                    for key in ("source_frame", "timestamp_ns", "img_name")),
                "download validation frame inventory differs from manifest")
        obj = row["source_object"]
        require(obj["key"] == f"part1/Images/{manifest['flight_id']}/{expected_name}",
                "source object outside frozen sequence")
        require(type(obj["bytes"]) is int and 0 < obj["bytes"] <= PNG_BYTES_CAP
                and type(checked.get("bytes")) is int and checked["bytes"] == obj["bytes"],
                "invalid source PNG byte count")
        require(re.fullmatch(r'"[a-f0-9]{32}"', obj["etag"]),
                "unsupported source ETag checksum")
        require(all(isinstance(checked.get(key), str)
                    and SHA_PATTERN.fullmatch(checked[key])
                    for key in ("png_sha256", "pixel_sha256")),
                "missing validated PNG/pixel SHA-256")
        total += obj["bytes"]
        previous_index, previous_time = index, int(timestamp)
    require(total <= PNG_BYTES_CAP and total == manifest.get("source_png_bytes")
            == validation.get("source_png_bytes"), "source image byte cap/total changed")
    require(validation.get("source_frame_range")
            == [rows[0]["source_frame"], rows[-1]["source_frame"]],
            "download validation source extent changed")
    image_directory = source / "source_png"
    require(image_directory.is_dir() and not image_directory.is_symlink(),
            "missing/linked source PNG directory")
    return manifest, validation, identities


def validated_pixels(source, row, checked):
    path = source / "source_png" / row["img_name"]
    require(path.is_file() and not path.is_symlink(), f"missing/linked PNG: {path.name}")
    require(path.stat().st_size == checked["bytes"], f"PNG size changed: {path.name}")
    with path.open("rb") as stream:
        data = stream.read(checked["bytes"] + 1)
    require(len(data) == checked["bytes"], f"PNG read size changed: {path.name}")
    require(hashlib.sha256(data).hexdigest() == checked["png_sha256"],
            f"PNG SHA-256 changed: {path.name}")
    require(hashlib.md5(data).hexdigest() == row["source_object"]["etag"].strip('"'),
            f"PNG source ETag checksum changed: {path.name}")
    # Reject a non-native/decompression-bomb header before calling the decoder.
    require(len(data) >= 33 and data[:8] == b"\x89PNG\r\n\x1a\n"
            and data[8:16] == b"\x00\x00\x00\rIHDR"
            and struct.unpack(">II", data[16:24]) == (WIDTH, HEIGHT)
            and data[24:28] == bytes((8, 0, 0, 0)) and data[28] in (0, 1),
            f"PNG header is not native gray8: {path.name}")
    pixels = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    require(pixels is not None and pixels.shape == (HEIGHT, WIDTH)
            and pixels.dtype == np.uint8, f"decoded PNG is not native gray8: {path.name}")
    require(hashlib.sha256(pixels.tobytes(order="C")).hexdigest() == checked["pixel_sha256"],
            f"decoded PNG pixel hash changed: {path.name}")
    return pixels


def encode_video(source, output, manifest, validation, ffmpeg):
    video = output / "pilot_gray8_ffv1_10fps.avi"
    command = [ffmpeg, "-nostdin", "-n", "-v", "error",
               "-f", "rawvideo", "-pixel_format", "gray",
               "-video_size", f"{WIDTH}x{HEIGHT}", "-framerate", str(FPS), "-i", "pipe:0",
               "-an", "-c:v", "ffv1", "-level", "3", "-pix_fmt", "gray", "-threads", "1",
               str(video)]
    with (output / "ffmpeg.stderr.log").open("xb") as errors:
        process = subprocess.Popen(command, stdin=subprocess.PIPE,
                                   stdout=subprocess.DEVNULL, stderr=errors)
        try:
            for index, (row, checked) in enumerate(zip(manifest["frames"], validation["images"])):
                pixels = validated_pixels(source, row, checked)
                process.stdin.write(pixels.tobytes(order="C"))
                if (index + 1) % 25 == 0:
                    print(f"encoded {index + 1}/{FRAME_CAP} validated native gray8 frames", flush=True)
            process.stdin.close()
            require(process.wait(timeout=60) == 0,
                    "FFV1 encoding failed; see retained ffmpeg.stderr.log")
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            if not process.stdin.closed:
                process.stdin.close()
    return video, command


def verify_video(video, validation):
    probe = probe_video(video)
    require(probe.codec == "ffv1" and probe.pixel_format in {"gray", "gray8"}
            and (probe.width, probe.height) == (WIDTH, HEIGHT)
            and probe.frame_rate == Fraction(FPS, 1)
            and probe.declared_frame_count == FRAME_CAP,
            "derived video probe does not match lossless native-frame contract")
    count = 0
    with VisibleFrameReader(video, (HEIGHT, WIDTH), execution="prefetch_one") as reader:
        require(reader.fps == FPS and reader.expected == FRAME_CAP,
                "actual visible reader reports different cadence/frame count")
        while True:
            frame, _ = reader.read()
            if frame is None:
                break
            require(count < FRAME_CAP and frame.index == count, "extra/reordered decoded frame")
            require(frame.gray.shape == (HEIGHT, WIDTH) and frame.gray.dtype == np.uint8,
                    "visible reader changed image size/type")
            require(hashlib.sha256(frame.gray.tobytes(order="C")).hexdigest()
                    == validation["images"][count]["pixel_sha256"],
                    f"lossless round-trip mismatch at input frame {count}")
            count += 1
            if count % 25 == 0:
                print(f"round-trip verified {count}/{FRAME_CAP} native frames", flush=True)
    stats = reader.completed_stats()
    require(count == FRAME_CAP and stats["decoded_frames"] == stats["consumed_frames"] == FRAME_CAP
            and stats["dropped_frames"] == 0 and stats["maximum_observed_frames_ahead"] <= 1
            and stats["worker_joined"] and stats["capture_released"],
            "incomplete visible-reader lossless verification")
    return probe, stats, reader.contract


def prepare(source, output):
    source, output = Path(source).resolve(strict=True), Path(output).absolute()
    require(source.is_dir(), "input must be an existing pilot directory")
    require(not output.exists() and not output.is_symlink(),
            "packaging output already exists; refusing overwrite or unvalidated reuse")
    require(output.resolve() != source and source / "source_png" not in output.resolve().parents,
            "output must not replace source data")
    ffmpeg = shutil.which("ffmpeg")
    require(ffmpeg and shutil.which("ffprobe"), "ffmpeg and ffprobe are required")
    manifest, validation, identities = validate_inputs(source)
    output.mkdir(parents=True, exist_ok=False)
    video, command = encode_video(source, output, manifest, validation, ffmpeg)
    probe, stats, contract = verify_video(video, validation)
    for name, digest in identities.items():
        require(file_sha256(source / name) == digest, f"input metadata changed during packaging: {name}")
    times = [int(row["timestamp_ns"]) for row in manifest["frames"]]
    errors = [t - times[0] - index * 100_000_000 for index, t in enumerate(times)]
    receipt = {
        "schema": "seaqr.aot.lossless-packaging.v1", "passed": True,
        "input_directory": str(source), "inputs_sha256": identities,
        "flight_id": manifest["flight_id"], "frames": FRAME_CAP,
        "source_frame_range": validation["source_frame_range"],
        "source_png_bytes": validation["source_png_bytes"],
        "native_resolution": [WIDTH, HEIGHT], "native_dtype": "uint8",
        "video": str(video), "video_bytes": video.stat().st_size,
        "video_sha256": file_sha256(video), "probe": probe.to_dict(),
        "encoder_command": command,
        "ffmpeg_version": subprocess.check_output([ffmpeg, "-version"], text=True, timeout=10).splitlines()[0],
        "opencv_version": cv2.__version__, "numpy_version": np.__version__,
        "visible_reader": contract, "decode_stats": stats,
        "pixel_hashes_verified": FRAME_CAP,
        "source_acquisition_timestamps_preserved_in": str(source / "frozen_image_manifest.json"),
        "input_timing_adaptation": "nominal 10 Hz CFR; not exact source acquisition times",
        "source_timestamp_span_ns": str(times[-1] - times[0]),
        "nominal_frame_timestamp_span_ns": str((FRAME_CAP - 1) * 100_000_000),
        "maximum_abs_nominal_timestamp_error_ns": str(max(abs(value) for value in errors)),
        "derived_data_notice": {
            "provider": "Amazon Airborne Object Tracking",
            "license": "https://cdla.dev/permissive-1-0/",
            "transformation": "Pixel-exact gray FFV1 AVI packaging at nominal 10 Hz; source PNGs and annotation/timestamp manifest unchanged.",
        },
        "detector_run": False, "full_pipeline_validation": False,
        "accuracy_evaluation": False, "remote_transfer": False,
        "script_sha256": file_sha256(Path(__file__)),
    }
    with (output / "packaging_validation.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: receipt[key] for key in (
        "passed", "frames", "video_bytes", "video_sha256", "input_timing_adaptation", "detector_run"
    )}, indent=2))
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="completed AOT pilot intake directory")
    parser.add_argument("--output", type=Path, required=True, help="new, nonexistent packaging directory")
    args = parser.parse_args()
    prepare(args.input, args.output)


if __name__ == "__main__":
    main()
