"""Deterministic recorded-frame sources.

Video decoding uses FFmpeg directly so 16-bit grayscale data is not silently
converted to 8-bit BGR by a convenience video API.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from fractions import Fraction
import json
from pathlib import Path
import shutil
import statistics
import subprocess
from typing import Any, Iterator, Mapping, Protocol

import numpy as np

from .types import Frame, TimestampSource
from .validation import FrameSequenceValidator


class FrameSourceError(RuntimeError):
    pass


class FrameSource(Protocol):
    def __iter__(self) -> Iterator[Frame]: ...


@dataclass(frozen=True, slots=True)
class VideoProbe:
    path: Path
    codec: str
    pixel_format: str
    width: int
    height: int
    frame_rate: Fraction
    declared_frame_count: int | None
    duration_seconds: float | None

    @property
    def interval_ns(self) -> int:
        return round(1_000_000_000 * self.frame_rate.denominator / self.frame_rate.numerator)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "codec": self.codec,
            "pixel_format": self.pixel_format,
            "width": self.width,
            "height": self.height,
            "frame_rate": str(self.frame_rate),
            "declared_frame_count": self.declared_frame_count,
            "duration_seconds": self.duration_seconds,
        }


def _required_executable(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise FrameSourceError(f"Required executable is unavailable: {name}")
    return path


def _positive_fraction(value: str) -> Fraction:
    try:
        fraction = Fraction(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise FrameSourceError(f"Invalid frame rate: {value!r}") from exc
    if fraction <= 0:
        raise FrameSourceError(f"Frame rate must be positive: {value!r}")
    return fraction


def probe_video(path: str | Path) -> VideoProbe:
    video_path = Path(path).expanduser().resolve()
    if not video_path.is_file():
        raise FrameSourceError(f"Video does not exist: {video_path}")
    command = [
        _required_executable("ffprobe"),
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,pix_fmt,width,height,avg_frame_rate,r_frame_rate,nb_frames,duration",
        "-of",
        "json",
        str(video_path),
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise FrameSourceError(
            f"ffprobe failed for {video_path}: {completed.stderr.strip()}"
        )
    try:
        streams = json.loads(completed.stdout)["streams"]
        stream = streams[0]
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise FrameSourceError(f"ffprobe returned no usable video stream: {video_path}") from exc

    frame_rate_text = stream.get("avg_frame_rate") or stream.get("r_frame_rate")
    frame_rate = _positive_fraction(str(frame_rate_text))
    frame_count_text = stream.get("nb_frames")
    declared_frame_count = (
        int(frame_count_text)
        if frame_count_text not in {None, "", "N/A"}
        else None
    )
    duration_text = stream.get("duration")
    duration_seconds = (
        float(duration_text)
        if duration_text not in {None, "", "N/A"}
        else None
    )
    return VideoProbe(
        path=video_path,
        codec=str(stream.get("codec_name", "unknown")),
        pixel_format=str(stream.get("pix_fmt", "unknown")),
        width=int(stream["width"]),
        height=int(stream["height"]),
        frame_rate=frame_rate,
        declared_frame_count=declared_frame_count,
        duration_seconds=duration_seconds,
    )


def load_timestamp_csv(path: str | Path) -> list[int]:
    timestamp_path = Path(path).expanduser().resolve()
    if not timestamp_path.is_file():
        raise FrameSourceError(f"Timestamp CSV does not exist: {timestamp_path}")
    timestamps: list[int] = []
    with timestamp_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not {"frame_index", "unix_time_ns"}.issubset(
            reader.fieldnames
        ):
            raise FrameSourceError(
                "Timestamp CSV must contain frame_index and unix_time_ns columns"
            )
        for expected_index, row in enumerate(reader):
            try:
                frame_index = int(row["frame_index"])
                timestamp_ns = int(row["unix_time_ns"])
            except (TypeError, ValueError) as exc:
                raise FrameSourceError(
                    f"Invalid timestamp row {expected_index + 2} in {timestamp_path}"
                ) from exc
            if frame_index != expected_index:
                raise FrameSourceError(
                    f"Timestamp frame_index must be contiguous from zero; "
                    f"expected {expected_index}, got {frame_index}"
                )
            if timestamp_ns < 0:
                raise FrameSourceError("Timestamp values must be non-negative")
            timestamps.append(timestamp_ns)
    if not timestamps:
        raise FrameSourceError(f"Timestamp CSV is empty: {timestamp_path}")
    return timestamps


def sidecar_expected_interval_ns(timestamps: list[int]) -> int | None:
    """Return a robust nominal cadence from explicit acquisition timestamps."""

    positive_deltas = [
        current - previous
        for previous, current in zip(timestamps, timestamps[1:], strict=False)
        if current > previous
    ]
    if not positive_deltas:
        return None
    return round(statistics.median(positive_deltas))


def _read_exact(stream: Any, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


class FfmpegVideoSource:
    """Decode one video sequentially while preserving grayscale bit depth."""

    def __init__(
        self,
        path: str | Path,
        *,
        timestamp_csv: str | Path | None = None,
        timestamp_policy: str = "sidecar_or_container",
        bit_depth: int | None = None,
        start_frame: int = 0,
        max_frames: int | None = None,
        timestamp_gap_factor: float = 1.5,
    ) -> None:
        if start_frame < 0:
            raise ValueError("start_frame must be non-negative")
        if max_frames is not None and max_frames <= 0:
            raise ValueError("max_frames must be positive when supplied")
        if timestamp_policy not in {
            "sidecar_or_container",
            "require_sidecar",
            "container_rate",
        }:
            raise ValueError(f"Unknown timestamp_policy: {timestamp_policy}")

        self.probe = probe_video(path)
        self.start_frame = start_frame
        self.max_frames = max_frames
        self.timestamp_gap_factor = timestamp_gap_factor
        self._timestamps = (
            load_timestamp_csv(timestamp_csv)
            if timestamp_csv is not None and timestamp_policy != "container_rate"
            else None
        )
        if timestamp_policy == "require_sidecar" and self._timestamps is None:
            raise FrameSourceError("timestamp_policy=require_sidecar needs timestamp_csv")
        sidecar_interval_ns = (
            sidecar_expected_interval_ns(self._timestamps)
            if self._timestamps is not None
            else None
        )
        self.expected_interval_ns = (
            sidecar_interval_ns
            if sidecar_interval_ns is not None
            else self.probe.interval_ns
        )
        self.expected_interval_source = (
            "sidecar_median_positive_delta"
            if sidecar_interval_ns is not None
            else "container_frame_rate"
        )

        if "16" in self.probe.pixel_format:
            self.output_pixel_format = "gray16le"
            self.dtype = np.dtype("<u2")
            inferred_depth = 16
        else:
            self.output_pixel_format = "gray8"
            self.dtype = np.dtype("u1")
            inferred_depth = 8
        self.bit_depth = bit_depth if bit_depth is not None else inferred_depth
        if self.bit_depth <= 0 or self.bit_depth > self.dtype.itemsize * 8:
            raise ValueError(
                f"bit_depth {self.bit_depth} is incompatible with {self.dtype}"
            )
        if self._timestamps is not None and start_frame >= len(self._timestamps):
            raise FrameSourceError(
                f"start_frame {start_frame} is beyond {len(self._timestamps)} timestamps"
            )

    @property
    def has_explicit_timestamps(self) -> bool:
        return self._timestamps is not None

    def __iter__(self) -> Iterator[Frame]:
        frame_bytes = self.probe.width * self.probe.height * self.dtype.itemsize
        command = [
            _required_executable("ffmpeg"),
            "-nostdin",
            "-v",
            "error",
            "-i",
            str(self.probe.path),
            "-map",
            "0:v:0",
            "-f",
            "rawvideo",
            "-pix_fmt",
            self.output_pixel_format,
            "pipe:1",
        ]
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert process.stdout is not None
        validator = FrameSequenceValidator(
            expected_interval_ns=self.expected_interval_ns,
            gap_factor=self.timestamp_gap_factor,
        )
        yielded = 0
        decoded_index = 0
        stopped_early = False
        stderr_text = ""
        source_base_ns = self._timestamps[0] if self._timestamps is not None else 0
        try:
            while True:
                payload = _read_exact(process.stdout, frame_bytes)
                if not payload:
                    break
                if len(payload) != frame_bytes:
                    raise FrameSourceError(
                        f"Truncated decoded frame {decoded_index}: "
                        f"{len(payload)} of {frame_bytes} bytes"
                    )
                if self._timestamps is not None and decoded_index >= len(self._timestamps):
                    raise FrameSourceError(
                        "Video contains more decoded frames than timestamp rows"
                    )
                if decoded_index < self.start_frame:
                    decoded_index += 1
                    continue

                image = np.frombuffer(payload, dtype=self.dtype).reshape(
                    self.probe.height, self.probe.width
                )
                if self._timestamps is None:
                    timestamp_ns = decoded_index * self.probe.interval_ns
                    source_timestamp_ns = None
                    timestamp_source = TimestampSource.CONTAINER_RATE
                else:
                    source_timestamp_ns = self._timestamps[decoded_index]
                    timestamp_ns = source_timestamp_ns - source_base_ns
                    timestamp_source = TimestampSource.SIDECAR_UNIX_NS

                frame = Frame(
                    image=image,
                    timestamp_ns=timestamp_ns,
                    source_timestamp_ns=source_timestamp_ns,
                    timestamp_source=timestamp_source,
                    frame_index=decoded_index,
                    sequence=decoded_index,
                    source_id=str(self.probe.path),
                    bit_depth=self.bit_depth,
                )
                yield validator.observe(frame)
                yielded += 1
                decoded_index += 1
                if self.max_frames is not None and yielded >= self.max_frames:
                    stopped_early = True
                    break
        finally:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            process.stdout.close()
            if process.stderr is not None:
                stderr_text = process.stderr.read().decode(
                    "utf-8", errors="replace"
                ).strip()
                process.stderr.close()

        if not stopped_early and process.returncode not in {0, -15}:
            raise FrameSourceError(f"ffmpeg decode failed: {stderr_text}")
        if yielded == 0:
            raise FrameSourceError("Video source produced no frames")
        if (
            not stopped_early
            and self._timestamps is not None
            and decoded_index != len(self._timestamps)
        ):
            raise FrameSourceError(
                f"Decoded {decoded_index} frames but timestamp CSV has "
                f"{len(self._timestamps)} rows"
            )


class NpyManifestSource:
    """Small exact fixtures where each manifest entry references one NPY image."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        try:
            manifest = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise FrameSourceError(f"Cannot read NPY manifest {self.path}: {exc}") from exc
        if manifest.get("schema_version") != 1:
            raise FrameSourceError("NPY manifest schema_version must be 1")
        entries = manifest.get("frames")
        if not isinstance(entries, list) or not entries:
            raise FrameSourceError("NPY manifest frames must be a non-empty list")
        self.entries: list[Mapping[str, Any]] = []
        for index, entry in enumerate(entries):
            if not isinstance(entry, Mapping):
                raise FrameSourceError(f"Manifest frame {index} must be a mapping")
            self.entries.append(entry)
        expected_interval_ns = manifest.get("expected_interval_ns")
        self.expected_interval_ns = (
            int(expected_interval_ns) if expected_interval_ns is not None else None
        )

    def __iter__(self) -> Iterator[Frame]:
        validator = FrameSequenceValidator(expected_interval_ns=self.expected_interval_ns)
        for expected_index, entry in enumerate(self.entries):
            image_path = Path(str(entry["path"])).expanduser()
            if not image_path.is_absolute():
                image_path = (self.path.parent / image_path).resolve()
            image = np.load(image_path, allow_pickle=False)
            frame_index = int(entry.get("frame_index", expected_index))
            frame = Frame(
                image=image,
                timestamp_ns=int(entry["timestamp_ns"]),
                source_timestamp_ns=(
                    int(entry["source_timestamp_ns"])
                    if entry.get("source_timestamp_ns") is not None
                    else None
                ),
                timestamp_source=TimestampSource.MANIFEST,
                frame_index=frame_index,
                sequence=(
                    int(entry["sequence"])
                    if entry.get("sequence") is not None
                    else frame_index
                ),
                source_id=str(self.path),
                bit_depth=int(entry.get("bit_depth", image.dtype.itemsize * 8)),
                exposure_us=(
                    int(entry["exposure_us"])
                    if entry.get("exposure_us") is not None
                    else None
                ),
                gain=(float(entry["gain"]) if entry.get("gain") is not None else None),
                black_level=(
                    int(entry["black_level"])
                    if entry.get("black_level") is not None
                    else None
                ),
            )
            declared_hash = entry.get("pixel_sha256")
            if declared_hash is not None and frame.pixel_sha256() != declared_hash:
                raise FrameSourceError(f"Pixel hash mismatch for {image_path}")
            yield validator.observe(frame)


def source_from_config(input_config: Mapping[str, Any]) -> FrameSource:
    source_kind = input_config["source"]
    if source_kind == "npy_manifest":
        return NpyManifestSource(input_config["path"])
    if source_kind == "video":
        return FfmpegVideoSource(
            input_config["path"],
            timestamp_csv=input_config.get("timestamp_csv"),
            timestamp_policy=str(
                input_config.get("timestamp_policy", "sidecar_or_container")
            ),
            bit_depth=(
                int(input_config["bit_depth"])
                if input_config.get("bit_depth") is not None
                else None
            ),
            start_frame=int(input_config.get("start_frame", 0)),
            max_frames=(
                int(input_config["max_frames"])
                if input_config.get("max_frames") is not None
                else None
            ),
            timestamp_gap_factor=float(
                input_config.get("timestamp_gap_factor", 1.5)
            ),
        )
    raise FrameSourceError(f"Unsupported source kind: {source_kind}")
