"""Phase 1 CLI: inspect a recording through the production frame contract."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from .config import ConfigError, load_config
from .frame_source import (
    FfmpegVideoSource,
    FrameSourceError,
    source_from_config,
)
from .telemetry import StageTimings, file_identity, run_identity, write_json_exclusive


REPOSITORY = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "seaqr.tiny-target.inspect.v1"


def inspect_config(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    source = source_from_config(config.input)
    timings = StageTimings()
    frame_digest = hashlib.sha256(b"seaqr-frame-set-v1\0")
    frame_count = 0
    first_metadata: dict[str, Any] | None = None
    last_metadata: dict[str, Any] | None = None
    discontinuities: Counter[str] = Counter()
    dtypes: Counter[str] = Counter()
    shapes: Counter[str] = Counter()

    with timings.measure("recorded_input"):
        for frame in source:
            pixel_hash = frame.pixel_sha256()
            frame_digest.update(str(frame.frame_index).encode("ascii"))
            frame_digest.update(b"\0")
            frame_digest.update(str(frame.timestamp_ns).encode("ascii"))
            frame_digest.update(b"\0")
            frame_digest.update(pixel_hash.encode("ascii"))
            frame_digest.update(b"\n")
            metadata = frame.metadata_dict()
            metadata["pixel_sha256"] = pixel_hash
            if first_metadata is None:
                first_metadata = metadata
            last_metadata = metadata
            frame_count += 1
            dtypes[frame.image.dtype.str] += 1
            shapes[f"{frame.shape[1]}x{frame.shape[0]}"] += 1
            discontinuities.update(item.value for item in frame.discontinuities)

    if frame_count == 0 or first_metadata is None or last_metadata is None:
        raise FrameSourceError("Input inspection produced no frames")

    source_details: dict[str, Any] = {
        "kind": config.input["source"],
        "identity": file_identity(config.input["path"]),
    }
    warnings: list[str] = []
    if isinstance(source, FfmpegVideoSource):
        source_details["video_probe"] = source.probe.to_dict()
        source_details["timestamp_explicit"] = source.has_explicit_timestamps
        source_details["timestamp_semantics"] = config.input.get(
            "timestamp_semantics", "unspecified"
        )
        source_details["decoded_pixel_format"] = source.output_pixel_format
        if not source.has_explicit_timestamps:
            warnings.append(
                "Timestamps were reconstructed from the container frame rate; "
                "do not treat derived target velocities as sensor-authoritative."
            )
        elif config.input.get("timestamp_semantics") != "sensor_exposure":
            warnings.append(
                "Explicit sidecar timestamps are not identified as sensor exposure "
                "timestamps; their acquisition semantics must accompany velocity results."
            )
        if source.probe.codec == "mjpeg":
            warnings.append(
                "Motion JPEG is lossy and 8-bit decoded; use RAW16/FFV1 data for "
                "weak-signal sensitivity claims."
            )

    report = {
        "schema_version": SCHEMA_VERSION,
        "run": run_identity(REPOSITORY),
        "config": {
            "path": str(config.path),
            "sha256": config.sha256,
            "resolved": config.raw,
        },
        "source": source_details,
        "inspection": {
            "frame_count": frame_count,
            "frame_set_sha256": frame_digest.hexdigest(),
            "first_frame": first_metadata,
            "last_frame": last_metadata,
            "dtypes": dict(sorted(dtypes.items())),
            "shapes": dict(sorted(shapes.items())),
            "discontinuities": dict(sorted(discontinuities.items())),
        },
        "timings": timings.summary(),
        "warnings": warnings,
    }
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect recorded tiny-target input without processing pixels"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help="Exclusively create a JSON report; prints to stdout when omitted",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = inspect_config(args.config)
        if args.output is None:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            output = write_json_exclusive(args.output, report)
            print(f"Wrote {output}")
    except (ConfigError, FrameSourceError, OSError, ValueError) as exc:
        raise SystemExit(f"tiny-target inspection failed: {exc}") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
