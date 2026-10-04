"""Typed data passed between tiny-target pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
import hashlib
from typing import Any

import numpy as np


class TimestampSource(str, Enum):
    """Where a frame's timing came from."""

    CAMERA = "camera"
    SIDECAR_UNIX_NS = "sidecar_unix_ns"
    CONTAINER_RATE = "container_rate"
    MANIFEST = "manifest"


class Discontinuity(str, Enum):
    """Conditions that prevent a frame sequence from being silently uniform."""

    FRAME_INDEX_GAP = "frame_index_gap"
    SEQUENCE_GAP = "sequence_gap"
    DUPLICATE_TIMESTAMP = "duplicate_timestamp"
    NON_MONOTONIC_TIMESTAMP = "non_monotonic_timestamp"
    TIMESTAMP_GAP = "timestamp_gap"
    CHUNK_BOUNDARY = "chunk_boundary"


@dataclass(frozen=True, slots=True)
class Frame:
    """One radiometric image and the metadata needed to interpret it.

    ``timestamp_ns`` is monotonic within a source and is the timestamp used by
    motion models. ``source_timestamp_ns`` preserves an absolute or device
    timestamp when one exists. Arrays are marked read-only so downstream stages
    cannot accidentally mutate evidence shared with another stage.

    ``valid_mask=None`` means that every source pixel is valid. A concrete mask
    is introduced after geometric warping or when the source has invalid pixels.
    """

    image: np.ndarray
    timestamp_ns: int
    frame_index: int
    source_id: str
    bit_depth: int
    timestamp_source: TimestampSource
    source_timestamp_ns: int | None = None
    sequence: int | None = None
    exposure_us: int | None = None
    gain: float | None = None
    black_level: int | None = None
    valid_mask: np.ndarray | None = None
    discontinuities: tuple[Discontinuity, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        image = np.asarray(self.image)
        if image.ndim != 2:
            raise ValueError(f"Frame image must be 2-D grayscale, got {image.shape}")
        if image.size == 0:
            raise ValueError("Frame image cannot be empty")
        if image.dtype.kind not in "uif":
            raise TypeError(f"Unsupported frame dtype: {image.dtype}")
        if self.timestamp_ns < 0:
            raise ValueError("timestamp_ns must be non-negative")
        if self.frame_index < 0:
            raise ValueError("frame_index must be non-negative")
        if self.sequence is not None and self.sequence < 0:
            raise ValueError("sequence must be non-negative when supplied")
        if not self.source_id:
            raise ValueError("source_id cannot be empty")
        if self.bit_depth <= 0:
            raise ValueError("bit_depth must be positive")
        if image.dtype.kind in "ui" and self.bit_depth > image.dtype.itemsize * 8:
            raise ValueError(
                f"bit_depth {self.bit_depth} exceeds dtype capacity {image.dtype}"
            )
        if self.valid_mask is not None:
            valid_mask = np.asarray(self.valid_mask)
            if valid_mask.shape != image.shape:
                raise ValueError(
                    "valid_mask shape must match image shape: "
                    f"{valid_mask.shape} != {image.shape}"
                )
            if valid_mask.dtype != np.bool_:
                raise TypeError("valid_mask must have boolean dtype")
            valid_mask = np.ascontiguousarray(valid_mask)
            if valid_mask.flags.writeable:
                valid_mask = valid_mask.copy()
            valid_mask.setflags(write=False)
            object.__setattr__(self, "valid_mask", valid_mask)

        image = np.ascontiguousarray(image)
        if image.flags.writeable:
            image = image.copy()
        image.setflags(write=False)
        object.__setattr__(self, "image", image)
        object.__setattr__(self, "timestamp_source", TimestampSource(self.timestamp_source))
        object.__setattr__(
            self,
            "discontinuities",
            tuple(Discontinuity(item) for item in self.discontinuities),
        )

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.image.shape[0]), int(self.image.shape[1])

    def with_discontinuities(
        self, *items: Discontinuity
    ) -> "Frame":
        merged = tuple(dict.fromkeys((*self.discontinuities, *items)))
        return replace(self, discontinuities=merged)

    def pixel_sha256(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"seaqr-frame-pixels-v1\0")
        digest.update(self.image.dtype.str.encode("ascii"))
        digest.update(b"\0")
        digest.update(f"{self.shape[0]}x{self.shape[1]}".encode("ascii"))
        digest.update(b"\0")
        digest.update(memoryview(self.image).cast("B"))
        return digest.hexdigest()

    def metadata_dict(self) -> dict[str, Any]:
        return {
            "timestamp_ns": self.timestamp_ns,
            "source_timestamp_ns": self.source_timestamp_ns,
            "timestamp_source": self.timestamp_source.value,
            "frame_index": self.frame_index,
            "sequence": self.sequence,
            "source_id": self.source_id,
            "shape": list(self.shape),
            "dtype": self.image.dtype.str,
            "bit_depth": self.bit_depth,
            "exposure_us": self.exposure_us,
            "gain": self.gain,
            "black_level": self.black_level,
            "valid_mask_present": self.valid_mask is not None,
            "discontinuities": [item.value for item in self.discontinuities],
        }
