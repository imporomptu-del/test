"""Core contracts for honest, reproducible end-to-end evaluation."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..detection import CandidateRecord, integrated_gaussian_kernel
from ..types import Frame


class EvaluationError(RuntimeError):
    """An evaluation manifest, injection, or metric is invalid."""


_CATEGORIES = {
    "noise_only",
    "real_targets",
    "raw_injected",
    "stress_scene",
    "controlled_geometry",
}
_AVAILABILITY = {"available", "available_on_jetson", "missing"}
_LABEL_STATUS = {
    "verified_empty",
    "labeled",
    "synthetic_ground_truth",
    "unlabeled",
    "unavailable",
}


@dataclass(frozen=True, slots=True)
class DatasetEntry:
    dataset_id: str
    category: str
    availability: str
    label_status: str
    source_path: str | None
    scene_tags: tuple[str, ...]
    notes: str

    def __post_init__(self) -> None:
        if not self.dataset_id:
            raise EvaluationError("dataset_id cannot be empty")
        if self.category not in _CATEGORIES:
            raise EvaluationError(f"Unknown dataset category: {self.category}")
        if self.availability not in _AVAILABILITY:
            raise EvaluationError(f"Unknown dataset availability: {self.availability}")
        if self.label_status not in _LABEL_STATUS:
            raise EvaluationError(f"Unknown label status: {self.label_status}")
        if self.availability != "missing" and not self.source_path:
            raise EvaluationError("available datasets require source_path")
        if self.availability == "missing" and self.source_path is not None:
            raise EvaluationError("missing datasets must use a null source_path")

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "category": self.category,
            "availability": self.availability,
            "label_status": self.label_status,
            "source_path": self.source_path,
            "scene_tags": list(self.scene_tags),
            "notes": self.notes,
        }


@dataclass(frozen=True, slots=True)
class DatasetInventory:
    manifest_id: str
    path: Path
    sha256: str
    entries: tuple[DatasetEntry, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "manifest_id": self.manifest_id,
            "path": str(self.path),
            "sha256": self.sha256,
            "entries": [entry.to_dict() for entry in self.entries],
        }


def load_dataset_inventory(path: str | Path) -> DatasetInventory:
    manifest_path = Path(path).expanduser().resolve()
    try:
        raw_bytes = manifest_path.read_bytes()
        value = json.loads(raw_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"Cannot load dataset inventory {manifest_path}: {exc}") from exc
    if not isinstance(value, Mapping) or value.get("schema_version") != 1:
        raise EvaluationError("dataset inventory schema_version must be exactly 1")
    manifest_id = value.get("manifest_id")
    if not isinstance(manifest_id, str) or not manifest_id:
        raise EvaluationError("dataset inventory manifest_id must be non-empty")
    raw_entries = value.get("entries")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise EvaluationError("dataset inventory entries must be a non-empty list")
    entries = []
    ids: set[str] = set()
    for index, raw in enumerate(raw_entries):
        if not isinstance(raw, Mapping):
            raise EvaluationError(f"dataset entry {index} must be a mapping")
        unknown = sorted(
            set(raw)
            - {
                "dataset_id",
                "category",
                "availability",
                "label_status",
                "source_path",
                "scene_tags",
                "notes",
            }
        )
        if unknown:
            raise EvaluationError(f"Unknown dataset entry keys: {unknown}")
        source_path = raw.get("source_path")
        if source_path is not None:
            if not isinstance(source_path, str) or not source_path:
                raise EvaluationError("dataset source_path must be non-empty or null")
            resolved = Path(source_path).expanduser()
            if not resolved.is_absolute():
                resolved = (manifest_path.parent / resolved).resolve()
            source_path = str(resolved)
        scene_tags = raw.get("scene_tags", [])
        if not isinstance(scene_tags, list) or not all(
            isinstance(item, str) and item for item in scene_tags
        ):
            raise EvaluationError("dataset scene_tags must be non-empty strings")
        entry = DatasetEntry(
            dataset_id=str(raw.get("dataset_id", "")),
            category=str(raw.get("category", "")),
            availability=str(raw.get("availability", "")),
            label_status=str(raw.get("label_status", "")),
            source_path=source_path,
            scene_tags=tuple(scene_tags),
            notes=str(raw.get("notes", "")),
        )
        if entry.dataset_id in ids:
            raise EvaluationError(f"Duplicate dataset_id: {entry.dataset_id}")
        ids.add(entry.dataset_id)
        entries.append(entry)
    present_categories = {entry.category for entry in entries}
    missing_categories = sorted(_CATEGORIES - present_categories)
    if missing_categories:
        raise EvaluationError(
            f"dataset inventory omits required categories: {missing_categories}"
        )
    return DatasetInventory(
        manifest_id=manifest_id,
        path=manifest_path,
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
        entries=tuple(entries),
    )


@dataclass(frozen=True, slots=True)
class SyntheticTarget:
    target_id: str
    flux_dn: float
    reference_timestamp_ns: int
    reference_position_xy_px: tuple[float, float]
    velocity_xy_px_s: tuple[float, float]
    acceleration_xy_px_s2: tuple[float, float] = (0.0, 0.0)
    first_frame_index: int = 0
    last_frame_index: int | None = None

    def __post_init__(self) -> None:
        values = (
            self.flux_dn,
            *self.reference_position_xy_px,
            *self.velocity_xy_px_s,
            *self.acceleration_xy_px_s2,
        )
        if not self.target_id or any(not math.isfinite(float(value)) for value in values):
            raise EvaluationError("synthetic target values must be named and finite")
        if self.flux_dn <= 0:
            raise EvaluationError("synthetic target flux_dn must be positive")
        if self.reference_timestamp_ns < 0 or self.first_frame_index < 0:
            raise EvaluationError("synthetic target reference/frame values cannot be negative")
        if self.last_frame_index is not None and self.last_frame_index < self.first_frame_index:
            raise EvaluationError("synthetic target frame range is reversed")

    def active(self, frame_index: int) -> bool:
        return frame_index >= self.first_frame_index and (
            self.last_frame_index is None or frame_index <= self.last_frame_index
        )

    def position_at(self, timestamp_ns: int) -> tuple[float, float]:
        dt = (timestamp_ns - self.reference_timestamp_ns) / 1e9
        return (
            self.reference_position_xy_px[0]
            + self.velocity_xy_px_s[0] * dt
            + 0.5 * self.acceleration_xy_px_s2[0] * dt**2,
            self.reference_position_xy_px[1]
            + self.velocity_xy_px_s[1] * dt
            + 0.5 * self.acceleration_xy_px_s2[1] * dt**2,
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SyntheticTarget":
        data = dict(value)
        unknown = sorted(set(data) - set(cls.__dataclass_fields__))
        if unknown:
            raise EvaluationError(f"Unknown synthetic target keys: {unknown}")
        for name in (
            "reference_position_xy_px",
            "velocity_xy_px_s",
            "acceleration_xy_px_s2",
        ):
            if name in data:
                item = data[name]
                if not isinstance(item, (list, tuple)) or len(item) != 2:
                    raise EvaluationError(f"synthetic target {name} must have two values")
                data[name] = (float(item[0]), float(item[1]))
        return cls(**data)


def transformed_target_truth(
    target: SyntheticTarget,
    frame_indices: Sequence[int],
    reference_timestamp_ns: int,
    frame_metadata: Mapping[int, tuple[int, np.ndarray]],
) -> dict[str, Any] | None:
    """Fit one injected trajectory in the stabilized window coordinates."""

    if not all(target.active(index) for index in frame_indices):
        return None
    times = []
    positions = []
    for frame_index in frame_indices:
        try:
            timestamp_ns, matrix = frame_metadata[frame_index]
        except KeyError as exc:
            raise EvaluationError(
                f"missing stabilization metadata for frame {frame_index}"
            ) from exc
        x, y = target.position_at(timestamp_ns)
        mapped = np.asarray(matrix, np.float64) @ np.array([x, y, 1.0], np.float64)
        if abs(mapped[2]) < 1e-12:
            raise EvaluationError("injected truth maps to infinity")
        positions.append(mapped[:2] / mapped[2])
        times.append((timestamp_ns - reference_timestamp_ns) / 1e9)
    design = np.column_stack((np.ones(len(times)), np.asarray(times)))
    position_array = np.asarray(positions)
    coefficients, _, _, _ = np.linalg.lstsq(design, position_array, rcond=None)
    return {
        "target_id": target.target_id,
        "flux_dn": target.flux_dn,
        "position_xy_px": coefficients[0].tolist(),
        "velocity_xy_px_s": coefficients[1].tolist(),
        "stabilized_sample_positions_xy_px": position_array.tolist(),
        "fit_residual_rms_px": float(
            np.sqrt(np.mean((design @ coefficients - position_array) ** 2))
        ),
    }


@dataclass(frozen=True, slots=True)
class SyntheticInjectionSpec:
    random_seed: int
    psf_sigma_px: float
    psf_radius_px: int
    targets: tuple[SyntheticTarget, ...]

    def __post_init__(self) -> None:
        if isinstance(self.random_seed, bool) or not isinstance(self.random_seed, int):
            raise EvaluationError("injection random_seed must be an integer")
        if not math.isfinite(self.psf_sigma_px) or self.psf_sigma_px <= 0:
            raise EvaluationError("injection psf_sigma_px must be positive")
        if self.psf_radius_px <= 0:
            raise EvaluationError("injection psf_radius_px must be positive")
        ids = [target.target_id for target in self.targets]
        if len(ids) != len(set(ids)):
            raise EvaluationError("synthetic target IDs must be unique")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SyntheticInjectionSpec":
        data = dict(value)
        unknown = sorted(
            set(data) - {"random_seed", "psf_sigma_px", "psf_radius_px", "targets"}
        )
        if unknown:
            raise EvaluationError(f"Unknown synthetic injection keys: {unknown}")
        targets = data.get("targets")
        if not isinstance(targets, list):
            raise EvaluationError("synthetic injection targets must be a list")
        parsed_targets = []
        for item in targets:
            if not isinstance(item, Mapping):
                raise EvaluationError("synthetic injection target must be a mapping")
            parsed_targets.append(SyntheticTarget.from_mapping(item))
        data["targets"] = tuple(parsed_targets)
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return {
            "random_seed": self.random_seed,
            "psf_sigma_px": self.psf_sigma_px,
            "psf_radius_px": self.psf_radius_px,
            "targets": [
                {
                    "target_id": target.target_id,
                    "flux_dn": target.flux_dn,
                    "reference_timestamp_ns": target.reference_timestamp_ns,
                    "reference_position_xy_px": list(
                        target.reference_position_xy_px
                    ),
                    "velocity_xy_px_s": list(target.velocity_xy_px_s),
                    "acceleration_xy_px_s2": list(target.acceleration_xy_px_s2),
                    "first_frame_index": target.first_frame_index,
                    "last_frame_index": target.last_frame_index,
                }
                for target in self.targets
            ],
        }


def load_injection_spec(path: str | Path) -> tuple[SyntheticInjectionSpec, dict[str, str]]:
    resolved = Path(path).expanduser().resolve()
    try:
        raw = resolved.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"Cannot load injection spec {resolved}: {exc}") from exc
    if not isinstance(value, Mapping) or value.get("schema_version") != 1:
        raise EvaluationError("injection spec schema_version must be exactly 1")
    data = dict(value)
    data.pop("schema_version")
    return SyntheticInjectionSpec.from_mapping(data), {
        "path": str(resolved),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


class SyntheticInjector:
    """Inject a pixel-integrated PSF into source radiometry before processing."""

    def __init__(self, spec: SyntheticInjectionSpec) -> None:
        self.spec = spec
        self.records: list[dict[str, Any]] = []

    def inject(self, frame: Frame) -> Frame:
        image = frame.image.astype(np.float64, copy=True)
        original = image.copy()
        height, width = frame.shape
        events = []
        for target in self.spec.targets:
            if not target.active(frame.frame_index):
                continue
            x, y = target.position_at(frame.timestamp_ns)
            center_x, center_y = int(round(x)), int(round(y))
            kernel = integrated_gaussian_kernel(
                self.spec.psf_sigma_px,
                self.spec.psf_radius_px,
                x - center_x,
                y - center_y,
            ).astype(np.float64)
            radius = self.spec.psf_radius_px
            source_x0 = max(0, radius - center_x)
            source_y0 = max(0, radius - center_y)
            source_x1 = min(2 * radius + 1, width + radius - center_x)
            source_y1 = min(2 * radius + 1, height + radius - center_y)
            destination_x0 = max(0, center_x - radius)
            destination_y0 = max(0, center_y - radius)
            destination_x1 = destination_x0 + (source_x1 - source_x0)
            destination_y1 = destination_y0 + (source_y1 - source_y0)
            requested = target.flux_dn * kernel[source_y0:source_y1, source_x0:source_x1]
            if requested.size:
                image[destination_y0:destination_y1, destination_x0:destination_x1] += requested
            events.append(
                {
                    "target_id": target.target_id,
                    "position_xy_px": [x, y],
                    "requested_flux_dn": target.flux_dn,
                    "in_bounds_requested_flux_dn": float(np.sum(requested)),
                    "subpixel_phase_xy": [x - center_x, y - center_y],
                }
            )
        if frame.image.dtype.kind in "ui":
            dtype_limit = np.iinfo(frame.image.dtype).max
            sensor_limit = min(dtype_limit, (1 << frame.bit_depth) - 1)
            image = np.clip(np.rint(image), 0, sensor_limit).astype(frame.image.dtype)
        else:
            image = image.astype(frame.image.dtype)
        achieved_total = float(
            np.sum(image.astype(np.float64) - original, dtype=np.float64)
        )
        self.records.append(
            {
                "frame_index": frame.frame_index,
                "timestamp_ns": frame.timestamp_ns,
                "events": events,
                "achieved_total_flux_dn_after_quantization_and_clipping": achieved_total,
                "output_dtype": image.dtype.str,
                "injection_stage": "decoded_source_before_motion_stabilization_preprocessing",
            }
        )
        return replace(frame, image=image)

    def coverage_summary(self) -> dict[str, Any]:
        """Summarize whether each configured target reached processed image support."""

        targets = []
        for target in self.spec.targets:
            events = [
                event
                for record in self.records
                for event in record["events"]
                if event["target_id"] == target.target_id
            ]
            targets.append(
                {
                    "target_id": target.target_id,
                    "active_processed_frame_count": len(events),
                    "in_bounds_requested_flux_dn": float(
                        sum(
                            float(event["in_bounds_requested_flux_dn"])
                            for event in events
                        )
                    ),
                }
            )
        return {
            "processed_frame_count": len(self.records),
            "targets": targets,
        }

    def validate_coverage(self) -> dict[str, Any]:
        """Reject evaluation runs whose configured injection was never observable."""

        summary = self.coverage_summary()
        inactive = [
            item["target_id"]
            for item in summary["targets"]
            if item["active_processed_frame_count"] == 0
        ]
        out_of_bounds = [
            item["target_id"]
            for item in summary["targets"]
            if item["active_processed_frame_count"] > 0
            and item["in_bounds_requested_flux_dn"] <= 0
        ]
        problems = []
        if inactive:
            problems.append(f"not active in processed frames: {inactive}")
        if out_of_bounds:
            problems.append(f"never entered image bounds: {out_of_bounds}")
        if problems:
            raise EvaluationError(
                "synthetic injection has no observable coverage ("
                + "; ".join(problems)
                + ")"
            )
        return summary


def match_candidates(
    candidates: Sequence[CandidateRecord],
    truths: Sequence[Mapping[str, Any]],
    *,
    maximum_position_error_px: float,
    maximum_velocity_error_px_s: float,
) -> dict[str, Any]:
    if maximum_position_error_px <= 0 or maximum_velocity_error_px_s <= 0:
        raise EvaluationError("matching gates must be positive")
    options = []
    for truth_index, truth in enumerate(truths):
        position = np.asarray(truth["position_xy_px"], np.float64)
        velocity = np.asarray(truth["velocity_xy_px_s"], np.float64)
        for candidate_index, candidate in enumerate(candidates):
            position_error = float(
                np.linalg.norm(np.array([candidate.x_px, candidate.y_px]) - position)
            )
            velocity_error = float(
                np.linalg.norm(np.asarray(candidate.velocity_xy_px_s) - velocity)
            )
            if (
                position_error <= maximum_position_error_px
                and velocity_error <= maximum_velocity_error_px_s
            ):
                cost = (
                    (position_error / maximum_position_error_px) ** 2
                    + (velocity_error / maximum_velocity_error_px_s) ** 2
                )
                options.append(
                    (
                        cost,
                        str(truth["target_id"]),
                        candidate.candidate_index,
                        truth_index,
                        candidate_index,
                        position_error,
                        velocity_error,
                    )
                )
    options.sort(key=lambda item: (item[0], item[1], item[2]))
    assigned_truths: set[int] = set()
    assigned_candidates: set[int] = set()
    matches = []
    for _, _, _, truth_index, candidate_index, position_error, velocity_error in options:
        if truth_index in assigned_truths or candidate_index in assigned_candidates:
            continue
        assigned_truths.add(truth_index)
        assigned_candidates.add(candidate_index)
        matches.append(
            {
                "target_id": str(truths[truth_index]["target_id"]),
                "candidate_index": candidates[candidate_index].candidate_index,
                "position_error_px": position_error,
                "velocity_error_px_s": velocity_error,
                "detector_score_snr": candidates[candidate_index].normalized_score_snr,
            }
        )
    return {
        "matches": matches,
        "true_positive_count": len(matches),
        "false_negative_count": len(truths) - len(matches),
        "unmatched_candidate_count": len(candidates) - len(matches),
        "unmatched_truth_ids": [
            str(truth["target_id"])
            for index, truth in enumerate(truths)
            if index not in assigned_truths
        ],
        "unmatched_candidate_indices": [
            candidate.candidate_index
            for index, candidate in enumerate(candidates)
            if index not in assigned_candidates
        ],
    }


class ThresholdAccumulator:
    """Aggregate detection metrics without inventing labels for unknown objects."""

    def __init__(self, threshold_snr: float, *, unmatched_are_false: bool) -> None:
        self.threshold_snr = float(threshold_snr)
        self.unmatched_are_false = unmatched_are_false
        self.window_count = 0
        self.truth_count = 0
        self.true_positive_count = 0
        self.false_negative_count = 0
        self.unmatched_candidate_count = 0
        self.position_errors: list[float] = []
        self.velocity_errors: list[float] = []
        self.by_flux: dict[str, list[int]] = {}

    def add(self, truths: Sequence[Mapping[str, Any]], result: Mapping[str, Any]) -> None:
        self.window_count += 1
        self.truth_count += len(truths)
        self.true_positive_count += int(result["true_positive_count"])
        self.false_negative_count += int(result["false_negative_count"])
        self.unmatched_candidate_count += int(result["unmatched_candidate_count"])
        matched_ids = {str(item["target_id"]) for item in result["matches"]}
        self.position_errors.extend(float(item["position_error_px"]) for item in result["matches"])
        self.velocity_errors.extend(float(item["velocity_error_px_s"]) for item in result["matches"])
        for truth in truths:
            key = f"{float(truth['flux_dn']):g}"
            totals = self.by_flux.setdefault(key, [0, 0])
            totals[1] += 1
            totals[0] += int(str(truth["target_id"]) in matched_ids)

    def summary(
        self,
        *,
        evaluated_frame_count: int,
        duration_s: float,
        frame_shape: tuple[int, int],
    ) -> dict[str, Any]:
        false_count = self.unmatched_candidate_count if self.unmatched_are_false else None
        megapixel_frames = (
            evaluated_frame_count * frame_shape[0] * frame_shape[1] / 1e6
        )
        precision = (
            self.true_positive_count / (self.true_positive_count + false_count)
            if false_count is not None and self.true_positive_count + false_count > 0
            else None
        )
        return {
            "threshold_snr": self.threshold_snr,
            "window_count": self.window_count,
            "truth_opportunity_count": self.truth_count,
            "true_positive_count": self.true_positive_count,
            "false_negative_count": self.false_negative_count,
            "unmatched_candidate_count": self.unmatched_candidate_count,
            "unmatched_candidates_are_labeled_false_alarms": self.unmatched_are_false,
            "probability_of_detection": (
                self.true_positive_count / self.truth_count if self.truth_count else None
            ),
            "precision": precision,
            "recall": (
                self.true_positive_count / self.truth_count if self.truth_count else None
            ),
            "false_alarms": (
                {
                    "count": false_count,
                    "per_integration_window": false_count / self.window_count,
                    "per_input_frame": false_count / evaluated_frame_count,
                    "per_minute": false_count * 60 / duration_s,
                    "per_megapixel_frame": false_count / megapixel_frames,
                }
                if false_count is not None
                and self.window_count
                and evaluated_frame_count
                and duration_s > 0
                and megapixel_frames > 0
                else None
            ),
            "localization_error_px": _error_summary(self.position_errors),
            "velocity_error_px_s": _error_summary(self.velocity_errors),
            "probability_of_detection_by_flux_dn": {
                flux: detected / total for flux, (detected, total) in sorted(self.by_flux.items())
            },
        }


def _error_summary(values: Sequence[float]) -> dict[str, float] | None:
    if not values:
        return None
    array = np.asarray(values, np.float64)
    return {
        "rmse": float(np.sqrt(np.mean(array**2))),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "maximum": float(np.max(array)),
    }


def latency_summary(values_ms: Sequence[float]) -> dict[str, float | int] | None:
    if not values_ms:
        return None
    values = np.asarray(values_ms, np.float64)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise EvaluationError("latencies must be finite and non-negative")
    return {
        "count": int(values.size),
        "median_ms": float(np.median(values)),
        "p90_ms": float(np.percentile(values, 90)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "maximum_ms": float(np.max(values)),
    }
