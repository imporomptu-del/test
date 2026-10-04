"""Fail-closed readiness gate for camera PSF and real-data evaluation cohorts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .telemetry import run_identity, write_json_exclusive


REPOSITORY = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "seaqr.tiny-target.real-data-readiness.v1"
DEFAULT_MANIFEST = (
    REPOSITORY / "configs" / "evaluation" / "phase16_real_data_manifest.json"
)


class RealDataValidationError(RuntimeError):
    """The readiness manifest itself is malformed."""


def _read_json(path: Path) -> tuple[Mapping[str, Any], str]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise RealDataValidationError(f"Cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise RealDataValidationError(f"JSON document must be an object: {path}")
    return value, hashlib.sha256(raw).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _path(value: Any, base: Path, field: str) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise RealDataValidationError(f"{field} must be a non-empty path or null")
    result = Path(value).expanduser()
    if not result.is_absolute():
        result = base / result
    return result.resolve()


def _nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RealDataValidationError(f"{field} must be a non-empty string")
    return value


def _declared_sha(value: Any, field: str) -> str:
    result = _nonempty_string(value, field).lower()
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise RealDataValidationError(f"{field} must be a lowercase SHA-256 digest")
    return result


def _unknown_keys(value: Mapping[str, Any], allowed: set[str], context: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise RealDataValidationError(f"Unknown {context} keys: {unknown}")


def _validate_psf_metadata(
    path: Path,
    *,
    calibration_id: str,
    operating_condition_id: str,
    kernel_sha256: str,
) -> dict[str, Any]:
    value, metadata_sha = _read_json(path)
    required = {
        "schema_version",
        "calibration_id",
        "source",
        "kernel_sha256",
        "operating_condition_id",
        "capture_method",
        "reviewer",
        "reviewed_at_utc",
        "camera",
        "conditions",
    }
    _unknown_keys(value, required, "PSF metadata")
    missing = sorted(required - set(value))
    if missing:
        raise RealDataValidationError(f"PSF metadata omits keys: {missing}")
    if value["schema_version"] != 1:
        raise RealDataValidationError("PSF metadata schema_version must be exactly 1")
    if value["calibration_id"] != calibration_id:
        raise RealDataValidationError("PSF metadata calibration_id does not match manifest")
    if value["source"] != "measured_camera_point_sources":
        raise RealDataValidationError("PSF metadata source must be measured_camera_point_sources")
    if value["kernel_sha256"] != kernel_sha256:
        raise RealDataValidationError("PSF metadata kernel_sha256 does not match the kernel")
    if value["operating_condition_id"] != operating_condition_id:
        raise RealDataValidationError(
            "PSF metadata operating_condition_id does not match manifest"
        )
    for field in ("capture_method", "reviewer", "reviewed_at_utc"):
        _nonempty_string(value[field], f"PSF metadata {field}")
    camera = value["camera"]
    conditions = value["conditions"]
    if not isinstance(camera, Mapping) or not isinstance(conditions, Mapping):
        raise RealDataValidationError("PSF camera and conditions must be objects")
    for field in ("model", "camera_id"):
        _nonempty_string(camera.get(field), f"PSF camera.{field}")
    for field in (
        "focus_setting",
        "aperture",
        "wavelength_band",
        "exposure_us",
        "gain",
        "sensor_temperature_c",
        "field_region",
    ):
        item = conditions.get(field)
        if item is None or isinstance(item, (list, dict)) or (
            isinstance(item, str) and not item.strip()
        ):
            raise RealDataValidationError(
                f"PSF conditions.{field} must be an explicit scalar value"
            )
    return {"path": str(path), "sha256": metadata_sha}


def _validate_kernel(path: Path) -> dict[str, Any]:
    try:
        kernels = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
    except (OSError, ValueError, EOFError) as exc:
        raise RealDataValidationError(f"Cannot load measured PSF kernel {path}: {exc}") from exc
    if kernels.ndim == 2:
        kernels = kernels[None, ...]
    if kernels.ndim != 3:
        raise RealDataValidationError("Measured PSF must be 2-D or [phase, height, width]")
    if kernels.shape[1] % 2 != 1 or kernels.shape[2] % 2 != 1:
        raise RealDataValidationError("Measured PSF kernel dimensions must be odd")
    flux = np.sum(kernels, axis=(1, 2))
    if not np.all(np.isfinite(kernels)) or np.any(flux <= 0):
        raise RealDataValidationError("Measured PSF values must be finite with positive flux")
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "shape": list(kernels.shape),
        "phase_count": int(kernels.shape[0]),
    }


def _validate_real_target_evidence(
    value: Mapping[str, Any], path: Path
) -> dict[str, Any]:
    targets = value.get("targets")
    if value.get("assertion") != "authoritative_real_targets":
        raise RealDataValidationError(
            "real-target evidence assertion must be authoritative_real_targets"
        )
    if not isinstance(targets, list) or not targets:
        raise RealDataValidationError("real-target evidence requires at least one target")
    frame_count = 0
    target_ids: set[str] = set()
    for target_index, target in enumerate(targets):
        if not isinstance(target, Mapping):
            raise RealDataValidationError(f"target {target_index} must be an object")
        _unknown_keys(target, {"target_id", "object_type", "frames"}, "target")
        target_id = _nonempty_string(target.get("target_id"), "target_id")
        if target_id in target_ids:
            raise RealDataValidationError(f"duplicate target_id: {target_id}")
        target_ids.add(target_id)
        if target.get("object_type") != "point_source":
            raise RealDataValidationError("real targets must use object_type=point_source")
        frames = target.get("frames")
        if not isinstance(frames, list) or not frames:
            raise RealDataValidationError(f"target {target_id} requires labeled frames")
        seen_frames: set[int] = set()
        for annotation in frames:
            if not isinstance(annotation, Mapping):
                raise RealDataValidationError("target frame annotations must be objects")
            _unknown_keys(
                annotation,
                {"frame_index", "timestamp_ns", "x_px", "y_px", "visibility", "uncertainty_px"},
                "target frame annotation",
            )
            frame_index = annotation.get("frame_index")
            timestamp_ns = annotation.get("timestamp_ns")
            if isinstance(frame_index, bool) or not isinstance(frame_index, int) or frame_index < 0:
                raise RealDataValidationError("annotation frame_index must be a nonnegative integer")
            if frame_index in seen_frames:
                raise RealDataValidationError(f"target {target_id} repeats frame {frame_index}")
            seen_frames.add(frame_index)
            if isinstance(timestamp_ns, bool) or not isinstance(timestamp_ns, int) or timestamp_ns < 0:
                raise RealDataValidationError("annotation timestamp_ns must be a nonnegative integer")
            if annotation.get("visibility") not in {"visible", "partially_occluded"}:
                raise RealDataValidationError("annotation visibility must be visible or partially_occluded")
            for field in ("x_px", "y_px", "uncertainty_px"):
                number = annotation.get(field)
                if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(float(number)):
                    raise RealDataValidationError(f"annotation {field} must be finite")
            if float(annotation["uncertainty_px"]) < 0:
                raise RealDataValidationError("annotation uncertainty_px cannot be negative")
        frame_count += len(frames)
    return {
        "path": str(path),
        "target_count": len(targets),
        "labeled_target_frame_count": frame_count,
    }


def _validate_empty_evidence(value: Mapping[str, Any], path: Path) -> dict[str, Any]:
    if value.get("assertion") != "no_point_targets":
        raise RealDataValidationError(
            "empty-scene evidence assertion must be no_point_targets"
        )
    intervals = value.get("reviewed_intervals")
    if not isinstance(intervals, list) or not intervals:
        raise RealDataValidationError("empty-scene evidence requires reviewed_intervals")
    reviewed_frames = 0
    previous_last = -1
    for interval in intervals:
        if not isinstance(interval, Mapping):
            raise RealDataValidationError("reviewed intervals must be objects")
        _unknown_keys(interval, {"first_frame_index", "last_frame_index"}, "reviewed interval")
        first = interval.get("first_frame_index")
        last = interval.get("last_frame_index")
        if (
            isinstance(first, bool)
            or isinstance(last, bool)
            or not isinstance(first, int)
            or not isinstance(last, int)
            or first < 0
            or last < first
            or first <= previous_last
        ):
            raise RealDataValidationError(
                "reviewed intervals must be ordered, disjoint nonnegative inclusive ranges"
            )
        reviewed_frames += last - first + 1
        previous_last = last
    return {
        "path": str(path),
        "reviewed_interval_count": len(intervals),
        "reviewed_frame_count": reviewed_frames,
    }


def _validate_evidence(
    path: Path,
    *,
    purpose: str,
    source_sha256: str,
    operating_condition_id: str,
) -> dict[str, Any]:
    value, evidence_sha = _read_json(path)
    common = {
        "schema_version",
        "evidence_id",
        "evidence_type",
        "source_sha256",
        "operating_condition_id",
        "reviewer",
        "reviewed_at_utc",
        "method",
        "assertion",
    }
    specific = {"targets"} if purpose == "real_target_sensitivity" else {"reviewed_intervals"}
    allowed = common | specific
    _unknown_keys(value, allowed, "evidence")
    missing = sorted(allowed - set(value))
    if missing:
        raise RealDataValidationError(f"evidence omits keys: {missing}")
    if value["schema_version"] != 1:
        raise RealDataValidationError("evidence schema_version must be exactly 1")
    expected_type = (
        "real_target_annotations"
        if purpose == "real_target_sensitivity"
        else "verified_empty_review"
    )
    if value["evidence_type"] != expected_type:
        raise RealDataValidationError(f"evidence_type must be {expected_type}")
    if value["source_sha256"] != source_sha256:
        raise RealDataValidationError("evidence source_sha256 does not match recording")
    if value["operating_condition_id"] != operating_condition_id:
        raise RealDataValidationError(
            "evidence operating_condition_id does not match cohort"
        )
    for field in ("evidence_id", "reviewer", "reviewed_at_utc", "method"):
        _nonempty_string(value[field], f"evidence {field}")
    result = (
        _validate_real_target_evidence(value, path)
        if purpose == "real_target_sensitivity"
        else _validate_empty_evidence(value, path)
    )
    result["sha256"] = evidence_sha
    return result


def audit_manifest(
    manifest_path: str | Path,
    *,
    verify_source_hashes: bool = True,
) -> dict[str, Any]:
    """Audit claim readiness; missing/inaccessible evidence becomes a blocker."""

    path = Path(manifest_path).expanduser().resolve()
    value, manifest_sha = _read_json(path)
    allowed_top = {"schema_version", "manifest_id", "psf_calibrations", "cohorts"}
    _unknown_keys(value, allowed_top, "manifest")
    if value.get("schema_version") != 1:
        raise RealDataValidationError("manifest schema_version must be exactly 1")
    manifest_id = _nonempty_string(value.get("manifest_id"), "manifest_id")
    raw_psfs = value.get("psf_calibrations")
    raw_cohorts = value.get("cohorts")
    if not isinstance(raw_psfs, list) or not raw_psfs:
        raise RealDataValidationError("manifest requires psf_calibrations")
    if not isinstance(raw_cohorts, list) or not raw_cohorts:
        raise RealDataValidationError("manifest requires cohorts")

    psf_results: list[dict[str, Any]] = []
    psf_ids: set[str] = set()
    for raw in raw_psfs:
        if not isinstance(raw, Mapping):
            raise RealDataValidationError("PSF calibration entries must be objects")
        allowed = {
            "calibration_id",
            "availability",
            "kernel_path",
            "metadata_path",
            "operating_condition_id",
            "notes",
        }
        _unknown_keys(raw, allowed, "PSF calibration")
        calibration_id = _nonempty_string(raw.get("calibration_id"), "calibration_id")
        if calibration_id in psf_ids:
            raise RealDataValidationError(f"duplicate calibration_id: {calibration_id}")
        psf_ids.add(calibration_id)
        availability = raw.get("availability")
        if availability not in {"available", "available_on_jetson", "missing"}:
            raise RealDataValidationError("invalid PSF availability")
        condition = _nonempty_string(
            raw.get("operating_condition_id"), "PSF operating_condition_id"
        )
        kernel_path = _path(raw.get("kernel_path"), path.parent, "kernel_path")
        metadata_path = _path(raw.get("metadata_path"), path.parent, "metadata_path")
        blockers: list[str] = []
        artifacts: dict[str, Any] = {}
        if availability == "missing":
            if kernel_path is not None or metadata_path is not None:
                raise RealDataValidationError("missing PSF entries must use null paths")
            blockers.append("measured PSF calibration is missing")
        else:
            for label, artifact_path in (("kernel", kernel_path), ("metadata", metadata_path)):
                if artifact_path is None:
                    blockers.append(f"{label} path is not declared")
                elif not artifact_path.is_file():
                    blockers.append(f"{label} file is inaccessible on this host: {artifact_path}")
            if not blockers:
                try:
                    kernel = _validate_kernel(kernel_path)  # type: ignore[arg-type]
                    metadata = _validate_psf_metadata(
                        metadata_path,  # type: ignore[arg-type]
                        calibration_id=calibration_id,
                        operating_condition_id=condition,
                        kernel_sha256=kernel["sha256"],
                    )
                    artifacts = {"kernel": kernel, "metadata": metadata}
                except RealDataValidationError as exc:
                    blockers.append(str(exc))
        psf_results.append(
            {
                "calibration_id": calibration_id,
                "availability": availability,
                "operating_condition_id": condition,
                "claim_ready": not blockers,
                "blockers": blockers,
                "artifacts": artifacts,
                "notes": str(raw.get("notes", "")),
            }
        )

    psf_by_id = {item["calibration_id"]: item for item in psf_results}
    cohort_results: list[dict[str, Any]] = []
    cohort_ids: set[str] = set()
    for raw in raw_cohorts:
        if not isinstance(raw, Mapping):
            raise RealDataValidationError("cohort entries must be objects")
        allowed = {
            "cohort_id",
            "purpose",
            "availability",
            "source_path",
            "source_sha256",
            "timestamps_path",
            "timestamps_sha256",
            "evidence_path",
            "psf_calibration_id",
            "operating_condition_id",
            "notes",
        }
        _unknown_keys(raw, allowed, "cohort")
        cohort_id = _nonempty_string(raw.get("cohort_id"), "cohort_id")
        if cohort_id in cohort_ids:
            raise RealDataValidationError(f"duplicate cohort_id: {cohort_id}")
        cohort_ids.add(cohort_id)
        purpose = raw.get("purpose")
        if purpose not in {"real_target_sensitivity", "verified_empty_false_alarm"}:
            raise RealDataValidationError("invalid cohort purpose")
        availability = raw.get("availability")
        if availability not in {"available", "available_on_jetson", "missing"}:
            raise RealDataValidationError("invalid cohort availability")
        condition = _nonempty_string(
            raw.get("operating_condition_id"), "cohort operating_condition_id"
        )
        psf_id = _nonempty_string(raw.get("psf_calibration_id"), "psf_calibration_id")
        source_path = _path(raw.get("source_path"), path.parent, "source_path")
        timestamps_path = _path(raw.get("timestamps_path"), path.parent, "timestamps_path")
        evidence_path = _path(raw.get("evidence_path"), path.parent, "evidence_path")
        blockers: list[str] = []
        artifacts: dict[str, Any] = {}
        referenced_psf = psf_by_id.get(psf_id)
        if referenced_psf is None:
            raise RealDataValidationError(f"cohort references unknown PSF: {psf_id}")
        if not referenced_psf["claim_ready"]:
            blockers.append(f"referenced measured PSF is not ready: {psf_id}")
        elif referenced_psf["operating_condition_id"] != condition:
            blockers.append("cohort and measured PSF operating conditions do not match")
        if availability == "missing":
            if any(item is not None for item in (source_path, timestamps_path, evidence_path)):
                raise RealDataValidationError("missing cohort entries must use null paths")
            if raw.get("source_sha256") is not None or raw.get("timestamps_sha256") is not None:
                raise RealDataValidationError("missing cohort entries must use null hashes")
            blockers.append("recording and authoritative evidence are missing")
        else:
            source_sha = _declared_sha(raw.get("source_sha256"), "source_sha256")
            timestamps_sha = _declared_sha(
                raw.get("timestamps_sha256"), "timestamps_sha256"
            )
            for label, artifact_path in (
                ("recording", source_path),
                ("timestamp sidecar", timestamps_path),
                ("evidence", evidence_path),
            ):
                if artifact_path is None:
                    blockers.append(f"{label} path is not declared")
                elif not artifact_path.is_file():
                    blockers.append(f"{label} file is inaccessible on this host: {artifact_path}")
            if not blockers or all("referenced measured PSF" in item for item in blockers):
                if source_path is not None and source_path.is_file():
                    actual_source_sha = _sha256(source_path) if verify_source_hashes else None
                    artifacts["recording"] = {
                        "path": str(source_path),
                        "size_bytes": source_path.stat().st_size,
                        "declared_sha256": source_sha,
                        "verified_sha256": actual_source_sha,
                    }
                    if actual_source_sha is None:
                        blockers.append("recording SHA-256 was not verified")
                    elif actual_source_sha != source_sha:
                        blockers.append("recording SHA-256 does not match manifest")
                if timestamps_path is not None and timestamps_path.is_file():
                    actual_timestamps_sha = _sha256(timestamps_path)
                    artifacts["timestamps"] = {
                        "path": str(timestamps_path),
                        "sha256": actual_timestamps_sha,
                    }
                    if actual_timestamps_sha != timestamps_sha:
                        blockers.append("timestamp sidecar SHA-256 does not match manifest")
                if evidence_path is not None and evidence_path.is_file():
                    try:
                        artifacts["evidence"] = _validate_evidence(
                            evidence_path,
                            purpose=purpose,
                            source_sha256=source_sha,
                            operating_condition_id=condition,
                        )
                    except RealDataValidationError as exc:
                        blockers.append(str(exc))
        cohort_results.append(
            {
                "cohort_id": cohort_id,
                "purpose": purpose,
                "availability": availability,
                "operating_condition_id": condition,
                "psf_calibration_id": psf_id,
                "claim_ready": not blockers,
                "blockers": blockers,
                "artifacts": artifacts,
                "notes": str(raw.get("notes", "")),
            }
        )

    claims: dict[str, Any] = {}
    for purpose in ("real_target_sensitivity", "verified_empty_false_alarm"):
        eligible = [
            item["cohort_id"]
            for item in cohort_results
            if item["purpose"] == purpose and item["claim_ready"]
        ]
        relevant = [item for item in cohort_results if item["purpose"] == purpose]
        blockers = [
            {"cohort_id": item["cohort_id"], "reasons": item["blockers"]}
            for item in relevant
            if item["blockers"]
        ]
        claims[purpose] = {
            "ready": bool(eligible),
            "eligible_cohort_ids": eligible,
            "blockers": blockers,
        }
    overall_ready = all(item["ready"] for item in claims.values())
    return {
        "schema_version": SCHEMA_VERSION,
        "run": run_identity(REPOSITORY),
        "implementation_sha256": {
            str(Path(__file__).resolve().relative_to(REPOSITORY)): hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest()
        },
        "manifest": {
            "manifest_id": manifest_id,
            "path": str(path),
            "sha256": manifest_sha,
        },
        "source_hash_verification_enabled": verify_source_hashes,
        "psf_calibrations": psf_results,
        "cohorts": cohort_results,
        "claim_readiness": claims,
        "overall_ready": overall_ready,
        "verdict": (
            "ready_for_real-data_pipeline_evaluation"
            if overall_ready
            else "blocked_missing_or_invalid_authoritative_evidence"
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--skip-source-hash",
        action="store_true",
        help="Inventory-only mode; cohorts remain claim-ineligible.",
    )
    parser.add_argument(
        "--require-ready",
        action="store_true",
        help="Exit 2 unless both real-target and verified-empty claims are ready.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = audit_manifest(
        args.manifest,
        verify_source_hashes=not args.skip_source_hash,
    )
    if args.output is None:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        output = write_json_exclusive(args.output, report)
        print(f"Wrote {output}")
    return 2 if args.require_ready and not report["overall_ready"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
