"""Bounded, frozen causal temporal features; never a classifier or output gate.

The real-media entry point requires an explicit CLI flag. Importing this module
or running its synthetic unit tests does not read source video. Every selected
observation remains in the output, including unavailable temporal evidence.
"""
import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "results/tiny_target/accuracy_v36_20260924"
CONTEXT = PARENT / "context_01"
FULL = PARENT / "full_context_01"
SHADOW = PARENT / "shadow_01"
SOURCES = {
    "0029": ROOT.parent / "outputs/jetson_review_clips_20260913/chunk_0029.avi",
    "0126": ROOT.parent / "outputs/jetson_review_clips_20260913/chunk_0126.avi",
}
COUNTS = {"0029": 687, "0126": 674}
FRAME_NS = 100_000_000
IMPLEMENTATION = (
    "scripts/diagnose_accuracy_v37_temporal.py",
    "scripts/accuracy_v37_registration.py", "scripts/accuracy_v37_temporal.py",
    "scripts/accuracy_v36_context.py",
    "tests/unit/test_accuracy_v37_extraction.py",
    "tests/unit/test_accuracy_v37_registration.py", "tests/unit/test_accuracy_v37_temporal.py",
    "docs/accuracy_v37_plan.md",
)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def bind(files, path, expected=None):
    path = str(Path(path).resolve())
    digest = sha(path)
    if expected is not None and digest != expected:
        raise ValueError("Changed bound input: " + path)
    if files.setdefault(path, digest) != digest:
        raise ValueError("Input changed during preflight: " + path)
    return digest


def recheck(files):
    for path, digest in files.items():
        if sha(path) != digest:
            raise ValueError("Bound file changed: " + path)


def validate_selection(selection):
    observations = selection["observations"]
    by_key = {item["key"]: item for item in observations}
    if (len(observations) != 358 or len(by_key) != 358
            or Counter(item["clip"] for item in observations) != {"0029": 150, "0126": 208}):
        raise ValueError("The exact 358 bounded v36 observations are required")
    for item in observations:
        if (item["key"] != f'{item["clip"]}/{item["frame"]}/{item["identity"]}'
                or item["polarity"] not in ("bright", "dark")
                or not item["identity"].split("/", 1)[1].startswith(item["polarity"] + ":")):
            raise ValueError("Invalid selected identity")
    for kind, denominator, hits in (("dense", 285, 284), ("pilot", 28, 28), ("anchor", 24, 24)):
        groups = [group for group in selection["groups"] if group["kind"] == kind]
        if len(groups) != denominator or sum(bool(group["keys"]) for group in groups) != hits:
            raise ValueError("Changed frozen reference denominator or baseline matching")
    if any(key not in by_key for group in selection["groups"] for key in group["keys"]):
        raise ValueError("Reference group points outside selected observations")
    if len(selection["controls"]) != 7 or sum(len(o["provisional_control_indices"]) for o in observations) != 70:
        raise ValueError("Changed provisional-control denominator")
    if any(type(i) is not int or not 0 <= i < 7
           for o in observations for i in o["provisional_control_indices"]):
        raise ValueError("Invalid control membership")
    return by_key


def verify_parent(audit_path):
    """Verify all full-audit file pins, without opening any source video."""
    audit_path = Path(audit_path).resolve()
    audit = read(audit_path)
    freeze = read(FULL / "freeze.json")
    summary = read(FULL / "summary.json")
    if (audit.get("schema") != "seaqr.accuracy-v36-full-context-independent-audit.v1"
            or audit.get("verified") is not True or audit.get("experiment") != str(FULL)
            or audit.get("full_context_freeze_sha256") != sha(FULL / "freeze.json")
            or audit.get("complete_result_workload_and_summary_verified") is not True
            or audit.get("independent_causal_cache_reconstruction") is not True
            or audit.get("every_baseline_qualified_state_verified_once") is not True
            or audit.get("bounded_context_observation_features_crosschecked") != 358
            or audit.get("known_reference_retention_passed") is not True
            or audit.get("raw16_accessed") is not False
            or summary.get("completed") is not True
            or summary.get("freeze_sha256") != sha(FULL / "freeze.json")):
        raise ValueError("Completed independent v36 full-context audit required")
    for kind, samples, hits in (("dense", 285, 284), ("pilot", 28, 28), ("anchors", 24, 24)):
        if audit["reference_totals"]["baseline"][kind] != {"samples": samples, "hits": hits}:
            raise ValueError("Changed independently audited reference totals")
    checked = audit.get("checked_files_sha256")
    if not isinstance(checked, dict) or not checked:
        raise ValueError("Audit contains no file pins")
    required = {str(FULL / name) for name in ("freeze.json", "summary.json", "unit.log")}
    required.update(freeze["inputs_sha256"])
    required.update(str(FULL / name) for name in summary["outputs_sha256"])
    for name, digest in freeze["implementation_sha256"].items():
        required.update((str(ROOT / name), str(FULL / "implementation" / name)))
        if checked.get(str(ROOT / name)) != digest:
            raise ValueError("Full-context implementation missing from audit")
    required.update(str(CONTEXT / name) for name in ("selection.json", "observations.json", "freeze.json"))
    if not required <= checked.keys():
        raise ValueError("Full audit omits required evidence")
    auditor = ROOT / "scripts/audit_accuracy_v36_full_context.py"
    if sha(auditor) != audit.get("auditor_sha256"):
        raise ValueError("Independent auditor source changed")
    files = {}
    for path, digest in checked.items():
        # The parent independent audit intentionally pins no source media.
        if Path(path).suffix.lower() in (".avi", ".mp4", ".raw16"):
            raise ValueError("Unexpected media in independent-audit file pins")
        bind(files, path, digest)
    bind(files, audit_path)
    bind(files, auditor, audit["auditor_sha256"])
    selection = read(CONTEXT / "selection.json")
    validate_selection(selection)
    context_freeze = read(CONTEXT / "freeze.json")
    bind(files, CONTEXT / "selection.json", context_freeze["selection_sha256"])
    parent = read(SHADOW / "freeze.json")
    for cid in SOURCES:
        if freeze["source_videos"][cid] != {"path": str(SOURCES[cid]), "sha256": parent["inputs"][cid]["source_sha256"]}:
            raise ValueError("Source identity differs between frozen parents")
        journal = Path(parent["inputs"][cid]["path"]) / "frames.jsonl"
        bind(files, journal, parent["inputs"][cid]["files_sha256"][str(journal)])
    return selection, files, parent


def track_map(row):
    result = {}
    for item in row["tracks"]:
        identity = f'{item["segment"]}/{item["track_id"]}'
        if identity in result or item["segment"] != row["segment"]:
            raise ValueError("Duplicate identity or inconsistent track segment")
        result[identity] = item
    return result


def validate_row(row, frame):
    if (type(row.get("frame_index")) is not int or row["frame_index"] != frame
            or type(row.get("timestamp_ns")) is not int or row["timestamp_ns"] != frame * FRAME_NS
            or type(row.get("segment")) is not int or row["segment"] < 0
            or not isinstance(row.get("motion"), dict) or type(row["motion"].get("reset")) is not bool):
        raise ValueError("Exact contiguous 10fps journal chronology required")
    return track_map(row)


def matrix(row):
    value = np.asarray(row.get("source_to_reference"), dtype=np.float64)
    if value.shape != (3, 3) or not np.isfinite(value).all():
        raise ValueError("Finite 3x3 source-to-reference matrix required")
    return value


def transform_xy(transform, xy):
    value = transform @ np.array([xy[0], xy[1], 1.0], dtype=np.float64)
    if not np.isfinite(value).all() or value[2] == 0:
        raise ValueError("Invalid homogeneous point mapping")
    result = value[:2] / value[2]
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite mapped coordinate")
    return result


def crop_native(gray, center, radius):
    """Preserve native values; outside-source samples are NaN, never padding."""
    if gray.ndim != 2 or gray.dtype != np.uint8:
        raise ValueError("Native uint8 gray frame required")
    x, y = map(int, center)
    size = 2 * radius + 1
    patch = np.full((size, size), np.nan, dtype=np.float32)
    x0, y0, x1, y1 = max(0, x-radius), max(0, y-radius), min(gray.shape[1], x+radius+1), min(gray.shape[0], y+radius+1)
    if x1 > x0 and y1 > y0:
        patch[y0-y+radius:y1-y+radius, x0-x+radius:x1-x+radius] = gray[y0:y1, x0:x1]
    return patch


def resample_prior(previous_gray, current_center, current_to_previous):
    """W0 maps current source coordinates to prior source coordinates."""
    y, x = np.mgrid[-26:27, -26:27]
    points = np.stack((x + current_center[0], y + current_center[1], np.ones(x.shape)))
    mapped = np.einsum("ij,jkl->ikl", current_to_previous, points)
    valid = np.isfinite(mapped).all(axis=0) & (mapped[2] != 0)
    map_x = np.full(x.shape, -1000, dtype=np.float32)
    map_y = np.full(y.shape, -1000, dtype=np.float32)
    np.divide(mapped[0], mapped[2], out=map_x, where=valid)
    np.divide(mapped[1], mapped[2], out=map_y, where=valid)
    output = cv2.remap(np.asarray(previous_gray, dtype=np.float32), map_x, map_y,
                       interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                       borderValue=float("nan"))
    output[~valid] = np.nan
    return output


def array_record(arrays, prefix, name, value):
    value = np.ascontiguousarray(value)
    key = prefix + "_" + name
    if key in arrays:
        raise ValueError("Duplicate patch array key")
    arrays[key] = value.copy()
    return dict(array_key=key, shape=list(value.shape), dtype=str(value.dtype),
                sha256=hashlib.sha256(value.tobytes()).hexdigest(),
                finite_pixels=int(np.isfinite(value).sum()))


def frame_metadata(row):
    value = {name: copy.deepcopy(row.get(name)) for name in
             ("frame_index", "timestamp_ns", "segment", "source_to_reference")}
    value["reference_reset"] = row["motion"]["reset"]
    return value


def availability_metadata(value, stage):
    if (not isinstance(value, dict) or type(value.get("available")) is not bool
            or not isinstance(value.get("status"), str) or not value["status"]
            or not isinstance(value.get("reasons"), list)
            or any(not isinstance(reason, str) or not reason for reason in value["reasons"])):
        raise ValueError("Malformed " + stage + " availability metadata")


def process_observation(observation, row, previous_row, gray, previous_gray,
                        registration, temporal, arrays, prefix, original_patch_sha=None):
    """Extract features at actual measurements, with exactly one prior frame."""
    current = track_map(row).get(observation["identity"])
    if (current is None or current.get("measured") is not True
            or current.get("qualified_moving") is not True
            or current.get("measurement_source_xy") != observation["measurement_source_xy"]
            or row["frame_index"] != observation["frame"]):
        raise ValueError("Selected observation no longer matches parent measurement")
    xy = np.asarray(current["measurement_source_xy"], dtype=np.float64)
    if xy.shape != (2,) or not np.isfinite(xy).all():
        raise ValueError("Invalid actual measurement coordinate")
    center = np.floor(xy + .5).astype(np.int64)
    current49 = crop_native(gray, center, 24)
    current25 = current49[12:37, 12:37]
    if original_patch_sha is not None:
        if (not np.isfinite(current25).all()
                or hashlib.sha256(current25.astype(np.uint8).tobytes()).hexdigest() != original_patch_sha):
            raise ValueError("Current source pixels differ from audited v36 patch")
    result = dict(**copy.deepcopy(observation), available=False, status="unknown",
                  reason=None, unavailable_reasons=[], previous_actual_measurement_available=False,
                  current_frame=frame_metadata(row), previous_frame=frame_metadata(previous_row) if previous_row else None,
                  integer_center_xy=center.tolist(), current_xy=(xy-center).tolist(),
                  delta_time_ns=None, current_to_previous_matrix=None,
                  previous_actual_measurement_source_xy=None, previous_point_current_grid_xy=None,
                  previous_point_registered_grid_xy=None, registration=None, features=None,
                  patches={"current49": array_record(arrays, prefix, "current49", current49)},
                  v36_current_patch_bytes_crosschecked=original_patch_sha is not None)
    prior53 = np.full((53, 53), np.nan, dtype=np.float32)
    reason = None
    if previous_row is None or previous_gray is None:
        reason = "missing_previous_frame"
    elif (previous_row["frame_index"] != row["frame_index"]-1
          or row["timestamp_ns"]-previous_row["timestamp_ns"] != FRAME_NS):
        reason = "nonadjacent_or_stale_previous_frame"
    elif previous_row["segment"] != row["segment"] or row.get("motion", {}).get("reset") is True:
        reason = "different_reference_segment"
    else:
        result["delta_time_ns"] = row["timestamp_ns"]-previous_row["timestamp_ns"]
        try:
            warp = np.linalg.inv(matrix(previous_row)) @ matrix(row)
            # Require invertibility for previous measurement mapping as well.
            inverse = np.linalg.inv(warp)
            if not np.isfinite(warp).all() or not np.isfinite(inverse).all():
                raise ValueError("Nonfinite inverse")
        except (ValueError, np.linalg.LinAlgError, TypeError):
            reason = "invalid_source_to_reference_transform"
        else:
            result["current_to_previous_matrix"] = warp.tolist()
            prior53 = resample_prior(previous_gray, center, warp)
            previous = track_map(previous_row).get(observation["identity"])
            if previous is None or previous.get("measured") is not True:
                reason = "missing_previous_actual_measurement"
            else:
                prior_xy = np.asarray(previous.get("measurement_source_xy"), dtype=np.float64)
                if prior_xy.shape != (2,) or not np.isfinite(prior_xy).all():
                    raise ValueError("Invalid previous actual measurement coordinate")
                result["previous_actual_measurement_available"] = True
                result["previous_actual_measurement_source_xy"] = prior_xy.tolist()
                try:
                    prior_point = transform_xy(inverse, prior_xy) - center
                except ValueError:
                    reason = "invalid_previous_measurement_mapping"
                else:
                    result["previous_point_current_grid_xy"] = prior_point.tolist()
    result["patches"]["prior53"] = array_record(arrays, prefix, "prior53", prior53)
    if reason:
        result["reason"] = reason
        result["unavailable_reasons"] = [reason]
        return result
    fitted = registration.measure(current49.copy(), prior53.copy(), prior_point_xy=prior_point.tolist())
    availability_metadata(fitted, "registration")
    fitted = dict(fitted)
    current_patch = fitted.pop("current_patch")
    prior_patch = fitted.pop("registered_prior_patch")
    json.dumps(fitted, allow_nan=False)
    result["registration"] = fitted
    for name, value in (("current25", current_patch), ("registered_prior25_raw", prior_patch)):
        if value is not None:
            if np.shape(value) != (25, 25):
                raise ValueError("Registration returned incorrect patch shape")
            result["patches"][name] = array_record(arrays, prefix, name, value)
    if not fitted["available"]:
        result["reason"] = "registration_" + str(fitted.get("status", "unavailable"))
        result["unavailable_reasons"] = (["registration/" + reason for reason in fitted["reasons"]]
                                         or [result["reason"]])
        return result
    if fitted.get("registered_prior_patch_photometrically_corrected") is not False:
        raise ValueError("Prior patch must be raw before its one photometric correction")
    shift = np.asarray(fitted["shift_xy"], dtype=float)
    gain, offset = fitted["gain"], fitted["offset"]
    if (shift.shape != (2,) or not np.isfinite(shift).all() or current_patch is None or prior_patch is None
            or not math.isfinite(gain) or not math.isfinite(offset) or gain <= 0
            or not np.isfinite(current_patch).all() or not np.isfinite(prior_patch).all()):
        raise ValueError("Available registration requires finite patches and positive photometric fit")
    if not np.array_equal(np.asarray(current_patch), current25):
        raise ValueError("Registration changed the native current center patch")
    corrected = gain * np.asarray(prior_patch, dtype=np.float64) + offset
    result["patches"]["registered_prior25_corrected"] = array_record(arrays, prefix, "registered_prior25_corrected", corrected)
    result["previous_point_registered_grid_xy"] = (prior_point-shift).tolist()
    features = temporal.measure(np.asarray(current_patch).copy(), corrected.copy(),
                                previous_xy=(prior_point-shift).tolist(),
                                polarity=observation["polarity"], current_xy=(xy-center).tolist())
    availability_metadata(features, "temporal")
    json.dumps(features, allow_nan=False)
    result["features"] = features
    result["available"] = features["available"]
    result["status"] = "available" if features["available"] else "unknown"
    result["reason"] = "diagnostic_available" if features["available"] else "temporal_" + str(features.get("status", "unavailable"))
    if not features["available"]:
        result["unavailable_reasons"] = (["temporal/" + reason for reason in features["reasons"]]
                                         or [result["reason"]])
    return result


def extract_clip(cid, selection, journal, source, registration, temporal, original_patches,
                 results, arrays, *, capture_factory=cv2.VideoCapture, shape=(3190, 4784), frame_count=None,
                 source_sha256=None):
    selected = defaultdict(list)
    for observation in selection["observations"]:
        if observation["clip"] == cid:
            selected[observation["frame"]].append(observation)
    if not selected:
        raise ValueError("No selected observations for clip")
    last = max(selected)
    cap = capture_factory(str(source))
    previous_gray = previous_row = None
    try:
        if (not cap.isOpened() or abs(cap.get(cv2.CAP_PROP_FPS)-10) > 1e-6
                or (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))) != shape
                or int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) != frame_count):
            raise ValueError("Unexpected decoder metadata")
        decoder = dict(backend=cap.getBackendName(), reported_frame_count=frame_count,
                       decoded_frames=last+1, first_frame=0, last_frame=last,
                       fps=10, shape_hw=list(shape), eof_not_checked_beyond_bounded_interval=True)
        with Path(journal).open() as stream:
            for frame in range(last+1):
                line = stream.readline()
                if not line:
                    raise ValueError("Parent journal ended before selected observation")
                row = json.loads(line)
                validate_row(row, frame)
                ok, bgr = cap.read()
                if not ok or not isinstance(bgr, np.ndarray) or bgr.shape != (*shape, 3) or bgr.dtype != np.uint8:
                    raise ValueError("Sequential source decode failed or changed geometry/type")
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                for observation in selected.get(frame, ()):
                    record = process_observation(observation, row, previous_row, gray, previous_gray,
                        registration, temporal, arrays, f"observation_{len(results):04d}", original_patches[observation["key"]])
                    record["source_sha256"] = source_sha256
                    results.append(record)
                previous_gray, previous_row = gray, row
    finally:
        cap.release()
    return decoder


def distribution(values):
    if not values:
        return {"count": 0}
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite feature distribution")
    return dict(count=len(values), minimum=float(values.min()), p05=float(np.quantile(values, .05)),
                median=float(np.median(values)), p95=float(np.quantile(values, .95)), maximum=float(values.max()))


def numeric_features(value, prefix=""):
    for key, item in value.items():
        name = prefix + key
        if isinstance(item, dict):
            yield from numeric_features(item, name + ".")
        elif type(item) in (int, float):
            if not math.isfinite(item):
                raise ValueError("Nonfinite temporal feature")
            yield name, item


def unavailable_reason_counts(records):
    return dict(Counter(reason for record in records if not record["available"]
                        for reason in sorted(set(record.get("unavailable_reasons", [record["reason"]])))))


def summarize(selection, results):
    by_key = {result["key"]: result for result in results}
    if len(by_key) != len(results) or by_key.keys() != {o["key"] for o in selection["observations"]}:
        raise ValueError("Missing or duplicate selected result")
    scopes = defaultdict(set)
    groups = []
    for group in selection["groups"]:
        keys = group["keys"]
        scopes[group["kind"] + "/" + group["window"]].update(keys)
        groups.append(dict(**copy.deepcopy(group), baseline_has_match=bool(keys),
                           diagnostic_available_keys=[key for key in keys if by_key[key]["available"]]))
    controls = []
    for i, control in enumerate(selection["controls"]):
        keys = [r["key"] for r in results if i in r["provisional_control_indices"]]
        scopes[f'control/{i}/{control["label"]}'].update(keys)
        controls.append(dict(**copy.deepcopy(control), baseline_selected_measurements=len(keys),
                             diagnostic_available_measurements=sum(by_key[k]["available"] for k in keys),
                             primary_unavailable_reasons=dict(Counter(by_key[k]["reason"] for k in keys if not by_key[k]["available"])),
                             unavailable_reasons=unavailable_reason_counts([by_key[k] for k in keys])))
    scopes["all"].update(by_key)
    for cid in SOURCES:
        scopes["clip/" + cid].update(r["key"] for r in results if r["clip"] == cid)
    stats = {}
    for name, keys in sorted(scopes.items()):
        features = defaultdict(list)
        for key in sorted(keys):
            record = by_key[key]
            if record["available"]:
                for feature, value in numeric_features(record["features"]):
                    features[feature].append(value)
        stats[name] = dict(selected_observations=len(keys),
            previous_actual_measurements=sum(by_key[k]["previous_actual_measurement_available"] for k in keys),
            diagnostic_available=sum(by_key[k]["available"] for k in keys),
            primary_unavailable_reasons=dict(Counter(by_key[k]["reason"] for k in keys if not by_key[k]["available"])),
            unavailable_reasons=unavailable_reason_counts([by_key[k] for k in keys]),
            unavailable_reason_counts_may_overlap=True,
            feature_distributions={key: distribution(value) for key, value in sorted(features.items())})
    reference_provenance = {}
    for kind in ("dense", "pilot", "anchor"):
        relevant = [group for group in groups if group["kind"] == kind]
        reference_provenance[kind] = dict(samples=len(relevant),
            original_baseline_matched_samples=sum(group["baseline_has_match"] for group in relevant),
            samples_with_available_diagnostic=sum(bool(group["diagnostic_available_keys"]) for group in relevant),
            diagnostic_availability_is_not_detection_retention=True)
    return dict(selected_observations=len(results),
        previous_actual_measurements=sum(r["previous_actual_measurement_available"] for r in results),
        diagnostic_available=sum(r["available"] for r in results),
        reference_provenance=reference_provenance, groups=groups, provisional_controls=controls,
        feature_distributions=stats, classifier_promoted=False, output_gate_applied=False,
        detector_rerun=False, thresholds_tuned=False, raw16_accessed=False, sealed_holdouts_accessed=False,
        airborne_precision=None, airborne_recall=None, false_alarms_per_minute=None,
        accuracy_retention_claimed=False,
        limitations=["Bounded diagnostic features only; no acceptance threshold or eligibility changes",
            "Unknown temporal evidence is not a negative or a retained detection",
            "The 285/28/24 reference groups overlap and contain unknown physical classes",
            "Selected provisional controls are not authoritative or representative negatives",
            "Exactly f-1 actual measurement required; intermittent targets may be unavailable",
            "This extraction is not an independent re-decoding audit"])


def run(output, audit_path):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Fresh exclusive v37 output directory required")
    selection, files, parent = verify_parent(audit_path)
    source_digests = {}
    for cid, source in SOURCES.items():
        source_digests[cid] = bind(files, source, parent["inputs"][cid]["source_sha256"])
    implementation = {name: bind(files, ROOT / name) for name in IMPLEMENTATION}
    output.mkdir(parents=True)
    with (output / "unit.log").open("x") as log:
        subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests/unit",
                        "-p", "test_accuracy_v37*.py", "-v"], cwd=ROOT,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    recheck(files)
    shutil.copy2(CONTEXT / "selection.json", output / "selection.json")
    for name, digest in implementation.items():
        target = output / "implementation" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
        bind(files, ROOT / name, digest)
        bind(files, target, digest)
    bind(files, output / "unit.log")
    bind(files, output / "selection.json", sha(CONTEXT / "selection.json"))
    freeze = dict(schema="seaqr.accuracy-v37-temporal-freeze.v1", pre_extraction=True,
        created_ns=time.time_ns(), inputs_sha256=files.copy(), implementation_sha256=implementation,
        source_videos={cid: dict(path=str(SOURCES[cid]), sha256=digest) for cid, digest in source_digests.items()},
        selection_sha256=sha(output / "selection.json"), unit_log_sha256=sha(output / "unit.log"),
        opencv_version=cv2.__version__, numpy_version=np.__version__,
        opencv_build_information_sha256=hashlib.sha256(cv2.getBuildInformation().encode()).hexdigest(),
        temporal_lag_frames=1, temporal_lag_ns=FRAME_NS,
        previous_actual_measurement_required=True, current_patch_size=49, prior_patch_size=53,
        current_native_values="uint8 values stored exactly as float32; NaN beyond source border",
        current_rounding="floor(actual_source_coordinate+0.5)",
        current_to_previous_mapping="inverse(H_previous) @ H_current",
        previous_sampling="cv2.remap float32 INTER_LINEAR BORDER_CONSTANT NaN",
        prior_point_registration="inverse(W0) @ previous_actual_xy - current_integer_center - shift_xy",
        temporal_prior_photometric_correction="gain * raw_registered_prior_patch25 + offset, exactly once",
        classifier_promoted=False, output_gate_applied=False)
    write(output / "freeze.json", freeze)
    bind(files, output / "freeze.json")
    recheck(files)
    # Imports are intentionally delayed until the experiment is frozen.
    from accuracy_v37_registration import AnnulusRegistration
    from accuracy_v37_temporal import TemporalPointDiagnostic
    registration, temporal = AnnulusRegistration(), TemporalPointDiagnostic()
    original = {item["key"]: item["patch_sha256"] for item in read(CONTEXT / "observations.json")}
    if original.keys() != {item["key"] for item in selection["observations"]}:
        raise ValueError("Audited native patch inventory differs from selection")
    results, arrays, decoders = [], {}, {}
    for cid, source in SOURCES.items():
        print(f"Bounded sequential temporal extraction {cid}", flush=True)
        journal = Path(parent["inputs"][cid]["path"]) / "frames.jsonl"
        decoders[cid] = extract_clip(cid, selection, journal, source, registration, temporal,
                                    original, results, arrays, frame_count=COUNTS[cid], source_sha256=source_digests[cid])
    validate_selection(selection)
    np.savez_compressed(output / "selected_patches.npz", **arrays)
    write(output / "observations.json", results)
    output_hashes = {name: bind(files, output / name) for name in ("observations.json", "selected_patches.npz")}
    summary = summarize(selection, results)
    summary.update(schema="seaqr.accuracy-v37-temporal-summary.v1", completed=True,
                   capture_records=decoders, freeze_sha256=sha(output / "freeze.json"),
                   all_358_current_native_patch_bytes_crosschecked=all(r["v36_current_patch_bytes_crosschecked"] for r in results),
                   outputs_sha256=output_hashes)
    recheck(files)
    write(output / "summary.json", summary)
    print(json.dumps({key: summary[key] for key in ("selected_observations", "previous_actual_measurements", "diagnostic_available")}), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, default=PARENT / "full_context_independent_audit_01.json")
    parser.add_argument("--execute-extraction", action="store_true",
                        help="Explicitly execute the reviewed, frozen bounded source extraction")
    args = parser.parse_args()
    if not args.execute_extraction:
        parser.error("No pixels extracted: explicit --execute-extraction required after plan/code review")
    run(args.output, args.audit)
