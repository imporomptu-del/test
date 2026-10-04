"""Frozen V37 saved-evidence audit; never opens or decodes source media.

Selection, journal geometry, array integrity, summary, and selected constrained
least-squares fits are checked independently. Frozen feature implementations are
shared ONLY for explicitly labeled saved-array registration/temporal replay.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/tiny_target/accuracy_v37_20260924"
EXPERIMENT = BASE / "temporal_01"
V36 = ROOT / "results/tiny_target/accuracy_v36_20260924"
CONTEXT, SHADOW, FULL = (V36 / name for name in ("context_01", "shadow_01", "full_context_01"))
PARENT_AUDIT = V36 / "full_context_independent_audit_01.json"
FREEZE_SHA256 = "0a25d626ae5ca364ebe1201121c18c63db123d9e50e40e942a6c5c8773333f2a"
COUNTS = {"0029": 687, "0126": 674}
FRAME_NS = 100_000_000
IMPLEMENTATION = (
    "scripts/diagnose_accuracy_v37_temporal.py", "scripts/accuracy_v37_registration.py",
    "scripts/accuracy_v37_temporal.py", "scripts/accuracy_v36_context.py",
    "tests/unit/test_accuracy_v37_extraction.py", "tests/unit/test_accuracy_v37_registration.py",
    "tests/unit/test_accuracy_v37_temporal.py", "docs/accuracy_v37_plan.md",
)
ARRAY_SPECS = {
    "current49": ((49, 49), "float32"), "prior53": ((53, 53), "float32"),
    "current25": ((25, 25), "float64"), "registered_prior25_raw": ((25, 25), "float64"),
    "registered_prior25_corrected": ((25, 25), "float64"),
}
# Validation tolerances only, never model-selection or eligibility thresholds.
RTOL = ATOL = 1e-9


def require(condition, message):
    if not condition:
        raise ValueError(message)


def parse(text):
    def invalid(token):
        raise ValueError("Nonfinite JSON constant: " + token)
    def unique(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON key: " + key)
            result[key] = value
        return result
    return json.loads(text, parse_constant=invalid, object_pairs_hook=unique)


def read(path):
    return parse(Path(path).read_text())


def sha(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "Regular non-symlink file required: " + str(path))
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exact(actual, expected, label):
    options = dict(sort_keys=True, allow_nan=False, separators=(",", ":"))
    require(json.dumps(actual, **options) == json.dumps(expected, **options), "Exact mismatch: " + label)


def close_tree(actual, expected, label):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and actual.keys() == expected.keys(), "Fields differ: " + label)
        for key in expected:
            close_tree(actual[key], expected[key], label + "/" + key)
    elif isinstance(expected, (list, tuple)):
        require(isinstance(actual, (list, tuple)) and len(actual) == len(expected), "List differs: " + label)
        for i, (a, e) in enumerate(zip(actual, expected)):
            close_tree(a, e, label + "/" + str(i))
    elif type(expected) is float:
        require(type(actual) in (int, float) and math.isfinite(actual) and math.isfinite(expected)
                and math.isclose(actual, expected, rel_tol=RTOL, abs_tol=ATOL),
                "Numeric mismatch: " + label + " " + str((actual, expected)))
    else:
        require(type(actual) is type(expected) and actual == expected, "Value differs: " + label)


def close_array(actual, expected, label):
    require(actual.shape == expected.shape and np.allclose(actual, expected, rtol=RTOL, atol=ATOL,
                                                         equal_nan=True), "Array mismatch: " + label)


class Integrity:
    def __init__(self):
        self.files = {}

    def bind(self, path, expected=None):
        path = Path(path)
        require(path.is_absolute() and path.resolve().is_relative_to(ROOT.resolve()),
                "Audit cannot open outside repository: " + str(path))
        require(path.suffix in (".json", ".jsonl", ".py", ".md", ".log", ".npz"),
                "Media or unsupported audit input forbidden: " + str(path))
        value = sha(path)
        require(expected is None or value == expected, "Changed bound file: " + str(path))
        require(self.files.setdefault(str(path), value) == value, "File changed during audit: " + str(path))
        return value

    def recheck(self):
        for path, digest in self.files.items():
            require(sha(path) == digest, "File changed during audit: " + path)


def verify_inputs(integrity, experiment):
    integrity.bind(experiment / "freeze.json", FREEZE_SHA256)
    freeze = read(experiment / "freeze.json")
    require(freeze["schema"] == "seaqr.accuracy-v37-temporal-freeze.v1"
            and freeze["pre_extraction"] is True and freeze["classifier_promoted"] is False
            and freeze["output_gate_applied"] is False, "Invalid frozen scope")
    geometry = dict(temporal_lag_frames=1, temporal_lag_ns=FRAME_NS,
        previous_actual_measurement_required=True, current_patch_size=49, prior_patch_size=53,
        current_native_values="uint8 values stored exactly as float32; NaN beyond source border",
        current_rounding="floor(actual_source_coordinate+0.5)",
        current_to_previous_mapping="inverse(H_previous) @ H_current",
        previous_sampling="cv2.remap float32 INTER_LINEAR BORDER_CONSTANT NaN",
        prior_point_registration="inverse(W0) @ previous_actual_xy - current_integer_center - shift_xy",
        temporal_prior_photometric_correction="gain * raw_registered_prior_patch25 + offset, exactly once")
    exact({key: freeze[key] for key in geometry}, geometry, "frozen geometry")
    require(set(freeze["source_videos"]) == set(COUNTS), "Unexpected source scope")
    source_claims = {record["path"]: record["sha256"] for record in freeze["source_videos"].values()}
    skipped = {}
    for path, digest in freeze["inputs_sha256"].items():
        if path in source_claims:
            require(digest == source_claims[path], "Source claim differs within freeze")
            skipped[path] = digest
        else:
            integrity.bind(path, digest)
    exact(skipped, source_claims, "exact inherited source-byte claims")
    require(set(freeze["implementation_sha256"]) == set(IMPLEMENTATION), "Implementation inventory changed")
    for relative, digest in freeze["implementation_sha256"].items():
        for path in (ROOT / relative, experiment / "implementation" / relative):
            require(freeze["inputs_sha256"].get(str(path)) == digest, "Snapshot omitted from input pins")
            integrity.bind(path, digest)
    integrity.bind(experiment / "selection.json", freeze["selection_sha256"])
    integrity.bind(experiment / "unit.log", freeze["unit_log_sha256"])
    require(re.search(r"Ran 69 tests in [0-9.]+s\s+OK\s*$", (experiment / "unit.log").read_text()) is not None,
            "Successful frozen 69-test receipt required")
    require(str(PARENT_AUDIT) in freeze["inputs_sha256"], "Parent audit not bound")
    parent_audit = read(PARENT_AUDIT)
    parent_freeze = read(FULL / "freeze.json")
    require(parent_audit["schema"] == "seaqr.accuracy-v36-full-context-independent-audit.v1"
            and parent_audit["verified"] is True and parent_audit["experiment"] == str(FULL)
            and parent_audit["full_context_freeze_sha256"] == sha(FULL / "freeze.json")
            and parent_audit["complete_result_workload_and_summary_verified"] is True
            and parent_audit["independent_causal_cache_reconstruction"] is True
            and parent_audit["every_baseline_qualified_state_verified_once"] is True
            and parent_audit["bounded_context_observation_features_crosschecked"] == 358
            and parent_audit["known_reference_retention_passed"] is True,
            "Invalid completed independent parent audit")
    for path, digest in parent_audit["checked_files_sha256"].items():
        require(freeze["inputs_sha256"].get(path) == digest, "Parent evidence omitted or changed")
    integrity.bind(ROOT / "scripts/audit_accuracy_v36_full_context.py", parent_audit["auditor_sha256"])
    shadow = read(SHADOW / "freeze.json")
    for cid in COUNTS:
        expected_path = ROOT.parent / "outputs/jetson_review_clips_20260913" / ("chunk_" + cid + ".avi")
        metadata = dict(path=str(expected_path), sha256=shadow["inputs"][cid]["source_sha256"])
        exact(freeze["source_videos"][cid], metadata, "source identity claim " + cid)
        exact(parent_freeze["source_videos"][cid], metadata, "parent source identity " + cid)
    integrity.bind(experiment / "summary.json")
    summary = read(experiment / "summary.json")
    require(summary["completed"] is True and summary["freeze_sha256"] == FREEZE_SHA256
            and summary["schema"] == "seaqr.accuracy-v37-temporal-summary.v1"
            and summary["all_358_current_native_patch_bytes_crosschecked"] is True,
            "Incomplete or unbound result")
    require(set(summary["outputs_sha256"]) == {"observations.json", "selected_patches.npz"},
            "Unexpected output inventory")
    for name, digest in summary["outputs_sha256"].items():
        integrity.bind(experiment / name, digest)
    exact(read(experiment / "selection.json"), read(CONTEXT / "selection.json"), "verbatim parent selection")
    return freeze, shadow, summary, skipped


def load_journals(shadow):
    journals = {}
    for cid, expected_count in COUNTS.items():
        path = Path(shadow["inputs"][cid]["path"]) / "frames.jsonl"
        rows = []
        with path.open() as stream:
            for frame, line in enumerate(stream):
                row = parse(line)
                require(type(row["frame_index"]) is int and row["frame_index"] == frame
                        and type(row["timestamp_ns"]) is int and row["timestamp_ns"] == frame * FRAME_NS
                        and type(row["segment"]) is int and row["segment"] >= 0
                        and type(row["motion"]["reset"]) is bool, "Journal chronology changed")
                ids = [(track["segment"], track["track_id"]) for track in row["tracks"]]
                require(len(ids) == len(set(ids)) and all(segment == row["segment"] for segment, _ in ids),
                        "Duplicate or cross-segment journal identity")
                rows.append(row)
        require(len(rows) == expected_count, "Incomplete parent journal")
        journals[cid] = rows
    return journals


def rebuild_selection(shadow, journals):
    controls = read(shadow["references"]["controls"])["controls"]
    require(len(controls) == 7, "Control window denominator changed")
    groups, observations = [], []
    for cid in COUNTS:
        baseline = read(SHADOW / (cid + "_results.json"))["arms"]["baseline"]
        for kind in ("dense", "pilot"):
            for window in baseline["references"][kind]["positive_windows"]:
                for sample in window["evidence"]:
                    frame = sample["frame_index"]
                    groups.append(dict(kind=kind, clip=cid, window=window["window_id"], frame=frame,
                        keys=[f"{cid}/{frame}/{identity}" for identity in sample["all_gated_same_polarity_ids"]],
                        baseline_assigned_id=sample["assigned_track_id"]))
        for anchor in baseline["required_anchor_evidence"]:
            groups.append(dict(kind="anchor", clip=cid, window=anchor["event_id"], frame=anchor["frame"],
                keys=[f'{cid}/{anchor["frame"]}/{identity}' for identity in anchor["ids"]],
                baseline_assigned_id=None))
        reference_keys = {key for group in groups if group["clip"] == cid for key in group["keys"]}
        observed_keys = set()
        for row in journals[cid]:
            for track in row["tracks"]:
                require(type(track["measured"]) is bool and type(track["qualified_moving"]) is bool,
                        "Nonboolean baseline state")
                if not (track["measured"] and track["qualified_moving"]):
                    continue
                identity = f'{track["segment"]}/{track["track_id"]}'
                key = f'{cid}/{row["frame_index"]}/{identity}'
                xy = track["measurement_source_xy"]
                require(np.shape(xy) == (2,) and np.isfinite(xy).all(), "Invalid actual measurement")
                memberships = []
                if cid == "0126":
                    for index, control in enumerate(controls):
                        first, last = control["frames_inclusive"]
                        x, y, width, height = control["crop_xywh"]
                        if first <= row["frame_index"] <= last and x <= xy[0] < x + width and y <= xy[1] < y + height:
                            memberships.append(index)
                if key in reference_keys or memberships:
                    require(key not in observed_keys, "Duplicated selected state")
                    observed_keys.add(key)
                    observations.append(dict(key=key, clip=cid, frame=row["frame_index"], identity=identity,
                        measurement_source_xy=xy, polarity=track["track_id"].split(":")[0],
                        provisional_control_indices=memberships))
        require(reference_keys <= observed_keys, "Dropped original matched reference")
    require(len(observations) == len({o["key"] for o in observations}) == 358
            and Counter(o["clip"] for o in observations) == {"0029": 150, "0126": 208}, "Changed 358-state selection")
    for kind, denominator, matched in (("dense", 285, 284), ("pilot", 28, 28), ("anchor", 24, 24)):
        items = [g for g in groups if g["kind"] == kind]
        require(len(items) == denominator and sum(bool(g["keys"]) for g in items) == matched,
                "Changed reference denominator: " + kind)
    require(sum(len(o["provisional_control_indices"]) for o in observations) == 70, "Changed 70-control membership count")
    return dict(observations=observations, groups=groups, controls=controls)


def frame_metadata(row):
    return {**{key: row[key] for key in ("frame_index", "timestamp_ns", "segment", "source_to_reference")},
            "reference_reset": row["motion"]["reset"]}


def transform_point(matrix, xy):
    value = matrix @ np.asarray([xy[0], xy[1], 1.0], dtype=np.float64)
    require(np.isfinite(value).all() and value[2] != 0, "Invalid homogeneous geometry")
    return value[:2] / value[2]


class DirectLeastSquares:
    """Independent full-design active sets, not normalized-bank fit helpers."""
    def __init__(self):
        self.y, self.x = np.mgrid[-12:13, -12:13].astype(np.float64)
        u, v = self.x.ravel() / 12, self.y.ravel() / 12
        self.background = np.column_stack((np.ones(625), u, v, u*u, u*v, v*v))

    def gaussian(self, sigma, xy):
        return np.exp(-((self.x - xy[0])**2 + (self.y - xy[1])**2) / (2*sigma*sigma)).ravel()

    def solve(self, vector, columns=()):
        design = np.column_stack((self.background, *columns))
        coefficients = np.linalg.lstsq(design, vector, rcond=None)[0]
        residual = vector - design @ coefficients
        return float(residual @ residual), coefficients[6:]

    def pair(self, vector, prior, current, free_current):
        candidates = [(self.solve(vector)[0], 0.0, 0.0)]
        for index, column in enumerate((prior, current)):
            sse, coefficients = self.solve(vector, (column,))
            if coefficients[0] >= 0 or (index == 1 and free_current):
                candidates.append((sse, float(coefficients[0]) if index == 0 else 0.0,
                                   float(coefficients[0]) if index == 1 else 0.0))
        sse, coefficients = self.solve(vector, (prior, current))
        if coefficients[0] >= 0 and (free_current or coefficients[1] >= 0):
            candidates.append((sse, *map(float, coefficients)))
        return min(candidates, key=lambda item: item[0])

    def condition(self, a, b):
        projected = []
        for column in (a, b):
            residual = column - self.background @ np.linalg.lstsq(self.background, column, rcond=None)[0]
            projected.append(residual / np.linalg.norm(residual))
        rho = abs(float(projected[0] @ projected[1]))
        return (1 + rho) / (1 - rho)

    def check(self, current, corrected_prior, features, polarity, label):
        if "point_sse" not in features:
            return False
        vector = (current - corrected_prior).ravel().astype(np.float64)
        energy = self.solve(vector)[0]
        close_tree(features["background_residual_energy"], energy, label + "/background LS")
        sign = 1 if polarity == "bright" else -1
        previous_xy = features["previous_xy"]
        null_candidates = [energy]
        for sigma in (1, 2, 3):
            column = -sign * self.gaussian(sigma, previous_xy)
            sse, coefficients = self.solve(vector, (column,))
            if coefficients[0] >= 0:
                null_candidates.append(sse)
        null_sse = min(null_candidates)
        close_tree(features["null_sse"], null_sse, label + "/best three-sigma null LS")
        for model in ("point", "edge"):
            previous = -sign * self.gaussian(features[model + "_previous_sigma_px"], previous_xy)
            if model == "point":
                center = np.asarray(features["current_xy"]) + features["point_offset_xy"]
                column = sign * self.gaussian(features["point_sigma_px"], center)
                current_amplitude_key = "point_current_amplitude_dn"
            else:
                angle = features["edge_orientation_rad"]
                column = np.tanh((self.x * math.cos(angle) + self.y * math.sin(angle)
                                  - features["edge_offset_px"]) / features["edge_width_px"]).ravel()
                current_amplitude_key = "edge_amplitude_dn"
            sse, prior_amplitude, current_amplitude = self.pair(vector, previous, column, model == "edge")
            close_tree(features[model + "_sse"], sse, label + "/" + model + " constrained joint LS")
            close_tree(features[model + "_previous_amplitude_dn"], prior_amplitude, label + "/previous amplitude " + model)
            close_tree(features[current_amplitude_key], current_amplitude, label + "/current amplitude " + model)
            condition = self.condition(previous, column)
            close_tree(features[model + "_pair_condition"], condition, label + "/condition " + model)
            require(condition <= 10000, "Ill-conditioned selected model")
            gain = float(np.clip((null_sse - sse) / energy, 0, 1))
            close_tree(features[model + "_gain_fraction"], gain, label + "/gain " + model)
        close_tree(features["point_minus_edge_fraction"],
                   features["point_gain_fraction"] - features["edge_gain_fraction"], label + "/non-nested margin")
        return True


def load_frozen_module(experiment, filename, name):
    path = experiment / "implementation/scripts" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify_observations(experiment, selection, journals, freeze):
    records = read(experiment / "observations.json")
    exact([r["key"] for r in records], [o["key"] for o in selection["observations"]], "complete ordered observations")
    original = {o["key"]: o for o in read(CONTEXT / "observations.json")}
    require(len(original) == 358, "Incomplete original patch inventory")
    registration = load_frozen_module(experiment, "accuracy_v37_registration.py", "audited_v37_registration").AnnulusRegistration()
    temporal = load_frozen_module(experiment, "accuracy_v37_temporal.py", "audited_v37_temporal").TemporalPointDiagnostic()
    ls = DirectLeastSquares()
    array_keys, replay_count, temporal_count, direct_count = set(), 0, 0, 0
    with np.load(experiment / "selected_patches.npz", allow_pickle=False) as stored:
        require(len(stored.files) == len(set(stored.files)), "Duplicate NPZ members")
        for index, (record, selected) in enumerate(zip(records, selection["observations"])):
            label = record["key"]
            exact({key: record[key] for key in selected}, selected, label + "/selected provenance")
            cid, frame = selected["clip"], selected["frame"]
            current_row = journals[cid][frame]
            prior_row = journals[cid][frame - 1] if frame else None
            exact(record["current_frame"], frame_metadata(current_row), label + "/current frame")
            exact(record["previous_frame"], frame_metadata(prior_row) if prior_row else None, label + "/previous frame")
            exact(record["source_sha256"], freeze["source_videos"][cid]["sha256"], label + "/source claim")
            require(record["v36_current_patch_bytes_crosschecked"] is True, "Missing current native crosscheck")
            xy = np.asarray(selected["measurement_source_xy"], dtype=np.float64)
            center = np.floor(xy + 0.5).astype(np.int64)
            exact(record["integer_center_xy"], center.tolist(), label + "/nearest native pixel")
            close_tree(record["current_xy"], (xy - center).tolist(), label + "/fractional current coordinate")
            patches = {}
            for name, metadata in record["patches"].items():
                require(name in ARRAY_SPECS, "Unknown saved array")
                expected_key = f"observation_{index:04d}_{name}"
                require(metadata["array_key"] == expected_key and expected_key not in array_keys, "Array alias/identity mismatch")
                array_keys.add(expected_key)
                value = stored[expected_key]
                shape, dtype = ARRAY_SPECS[name]
                require(value.shape == shape and str(value.dtype) == dtype and not np.isinf(value).any(), "Saved array geometry/type changed")
                exact(metadata, dict(array_key=expected_key, shape=list(shape), dtype=dtype,
                    sha256=hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest(),
                    finite_pixels=int(np.isfinite(value).sum())), label + "/array metadata " + name)
                patches[name] = value
            require({"current49", "prior53"} <= patches.keys(), "Missing raw saved contexts")
            current49, prior53 = patches["current49"], patches["prior53"]
            gy, gx = np.mgrid[-24:25, -24:25]
            inside = ((gx + center[0] >= 0) & (gx + center[0] < 4784)
                      & (gy + center[1] >= 0) & (gy + center[1] < 3190))
            require(np.array_equal(np.isfinite(current49), inside), "Current native border support changed")
            values = current49[inside]
            require(np.all((values >= 0) & (values <= 255)) and np.array_equal(values, np.floor(values)),
                    "Current49 no longer contains native uint8 values")
            finite_prior = prior53[np.isfinite(prior53)]
            require(np.all((finite_prior >= 0) & (finite_prior <= 255)), "Interpolated raw prior outside grayscale bounds")
            current25 = current49[12:37, 12:37]
            require(np.isfinite(current25).all() and hashlib.sha256(current25.astype(np.uint8).tobytes()).hexdigest()
                    == original[label]["patch_sha256"], "Current25 differs from independently audited V36 native bytes")
            geometry = dict(delta_time_ns=None, current_to_previous_matrix=None,
                previous_actual_measurement_available=False, previous_actual_measurement_source_xy=None,
                previous_point_current_grid_xy=None, previous_point_registered_grid_xy=None)
            geometry_reason = None
            if prior_row is None:
                geometry_reason = "missing_previous_frame"
            elif prior_row["segment"] != current_row["segment"] or current_row["motion"]["reset"]:
                geometry_reason = "different_reference_segment"
            else:
                geometry["delta_time_ns"] = FRAME_NS
                hc = np.asarray(current_row["source_to_reference"], dtype=np.float64)
                hp = np.asarray(prior_row["source_to_reference"], dtype=np.float64)
                require(hc.shape == hp.shape == (3, 3) and np.isfinite(hc).all() and np.isfinite(hp).all(),
                        "Invalid journal frame transform")
                warp = np.linalg.solve(hp, hc)
                inverse = np.linalg.inv(warp)
                require(np.isfinite(warp).all() and np.isfinite(inverse).all(), "Noninvertible pair transform")
                geometry["current_to_previous_matrix"] = warp.tolist()
                prior_tracks = {f'{t["segment"]}/{t["track_id"]}': t for t in prior_row["tracks"]}
                previous = prior_tracks.get(selected["identity"])
                if previous is None or previous["measured"] is not True:
                    geometry_reason = "missing_previous_actual_measurement"
                else:
                    previous_xy = previous["measurement_source_xy"]
                    require(np.shape(previous_xy) == (2,) and np.isfinite(previous_xy).all(), "Invalid previous actual coordinate")
                    prior_point = transform_point(inverse, previous_xy) - center
                    geometry.update(previous_actual_measurement_available=True,
                        previous_actual_measurement_source_xy=previous_xy,
                        previous_point_current_grid_xy=prior_point.tolist())
            if geometry_reason:
                expected_patches = {"current49", "prior53"}
                expected_state = dict(available=False, status="unknown", reason=geometry_reason,
                                      unavailable_reasons=[geometry_reason], registration=None, features=None)
            else:
                reg = registration.measure(current49.copy(), prior53.copy(), prior_point_xy=prior_point.tolist())
                replay_count += 1
                current_returned = reg.pop("current_patch")
                prior_returned = reg.pop("registered_prior_patch")
                expected_patches = {"current49", "prior53", "current25"}
                require(current_returned is not None, "Registration omitted current patch")
                close_array(patches["current25"], current_returned, label + "/registration current25")
                close_array(patches["current25"], current25.astype(np.float64), label + "/native current25 unchanged")
                if prior_returned is not None:
                    expected_patches.add("registered_prior25_raw")
                    close_array(patches["registered_prior25_raw"], prior_returned, label + "/geometric registered prior")
                if not reg["available"]:
                    expected_state = dict(available=False, status="unknown", reason="registration_" + reg["status"],
                        unavailable_reasons=["registration/" + reason for reason in reg["reasons"]], registration=reg, features=None)
                else:
                    require(reg["registered_prior_patch_photometrically_corrected"] is False,
                            "Photometric correction could be applied twice")
                    corrected = reg["gain"] * np.asarray(prior_returned, dtype=np.float64) + reg["offset"]
                    expected_patches.add("registered_prior25_corrected")
                    close_array(patches["registered_prior25_corrected"], corrected, label + "/exactly once photometry")
                    mapped = prior_point - np.asarray(reg["shift_xy"])
                    geometry["previous_point_registered_grid_xy"] = mapped.tolist()
                    features = temporal.measure(current_returned.copy(), corrected.copy(), previous_xy=mapped.tolist(),
                                                polarity=selected["polarity"], current_xy=(xy - center).tolist())
                    temporal_count += 1
                    expected_state = dict(available=features["available"], status="available" if features["available"] else "unknown",
                        reason="diagnostic_available" if features["available"] else "temporal_" + features["status"],
                        unavailable_reasons=[] if features["available"] else ["temporal/" + reason for reason in features["reasons"]],
                        registration=reg, features=features)
                    if ls.check(patches["current25"], patches["registered_prior25_corrected"],
                                record["features"], selected["polarity"], label):
                        direct_count += 1
            require(patches.keys() == expected_patches, "Missing or surplus saved patches: " + label)
            close_tree({key: record[key] for key in geometry}, geometry, label + "/independent adjacent-frame geometry")
            close_tree({key: record[key] for key in expected_state}, expected_state, label + "/saved-array numerical replay")
            expected_keys = set(selected) | set(geometry) | set(expected_state) | {
                "current_frame", "previous_frame", "integer_center_xy", "current_xy", "patches",
                "v36_current_patch_bytes_crosschecked", "source_sha256"}
            require(record.keys() == expected_keys, "Unexpected observation fields: " + label)
        require(set(stored.files) == array_keys, "Unreferenced or missing archive arrays")
    return records, dict(array_count=len(array_keys), registration_replays=replay_count,
                         temporal_replays=temporal_count, independent_selected_joint_ls_pairs=direct_count)


def reason_counts(records):
    primary, all_reasons = Counter(), Counter()
    for record in records:
        if not record["available"]:
            primary[record["reason"]] += 1
            all_reasons.update(set(record["unavailable_reasons"]))
    return dict(primary), dict(all_reasons)


def numeric_leaves(value, prefix=""):
    for key, item in value.items():
        if type(item) is dict:
            yield from numeric_leaves(item, prefix + key + ".")
        elif type(item) in (int, float):
            require(math.isfinite(item), "Nonfinite feature statistic")
            yield prefix + key, item


def independent_summary(selection, records):
    by_key = {r["key"]: r for r in records}
    scopes = defaultdict(set)
    groups, controls = [], []
    for group in selection["groups"]:
        keys = group["keys"]
        scopes[group["kind"] + "/" + group["window"]].update(keys)
        groups.append({**group, "baseline_has_match": bool(keys),
                       "diagnostic_available_keys": [key for key in keys if by_key[key]["available"]]})
    for index, control in enumerate(selection["controls"]):
        selected = [r for r in records if index in r["provisional_control_indices"]]
        scopes[f'control/{index}/{control["label"]}'].update(r["key"] for r in selected)
        primary, reasons = reason_counts(selected)
        controls.append({**control, "baseline_selected_measurements": len(selected),
            "diagnostic_available_measurements": sum(r["available"] for r in selected),
            "primary_unavailable_reasons": primary, "unavailable_reasons": reasons})
    scopes["all"].update(by_key)
    for cid in COUNTS:
        scopes["clip/" + cid].update(r["key"] for r in records if r["clip"] == cid)
    stats = {}
    for name, keys in sorted(scopes.items()):
        selected = [by_key[key] for key in sorted(keys)]
        samples = defaultdict(list)
        for record in selected:
            if record["available"]:
                for key, value in numeric_leaves(record["features"]):
                    samples[key].append(value)
        distributions = {}
        for key, values in sorted(samples.items()):
            values = np.sort(np.asarray(values, dtype=np.float64))
            # Linear quantiles computed without the extraction summary helper.
            def quantile(fraction):
                coordinate = (len(values) - 1) * fraction
                low, high = math.floor(coordinate), math.ceil(coordinate)
                return float(values[low] + (coordinate - low) * (values[high] - values[low]))
            distributions[key] = dict(count=len(values), minimum=float(values[0]), p05=quantile(.05),
                                      median=quantile(.5), p95=quantile(.95), maximum=float(values[-1]))
        primary, reasons = reason_counts(selected)
        stats[name] = dict(selected_observations=len(selected),
            previous_actual_measurements=sum(r["previous_actual_measurement_available"] for r in selected),
            diagnostic_available=sum(r["available"] for r in selected), primary_unavailable_reasons=primary,
            unavailable_reasons=reasons, unavailable_reason_counts_may_overlap=True,
            feature_distributions=distributions)
    reference = {}
    for kind in ("dense", "pilot", "anchor"):
        items = [group for group in groups if group["kind"] == kind]
        reference[kind] = dict(samples=len(items), original_baseline_matched_samples=sum(bool(g["keys"]) for g in items),
            samples_with_available_diagnostic=sum(bool(g["diagnostic_available_keys"]) for g in items),
            diagnostic_availability_is_not_detection_retention=True)
    return dict(selected_observations=len(records),
        previous_actual_measurements=sum(r["previous_actual_measurement_available"] for r in records),
        diagnostic_available=sum(r["available"] for r in records), reference_provenance=reference,
        groups=groups, provisional_controls=controls, feature_distributions=stats)


def audit(experiment=EXPERIMENT):
    experiment = Path(experiment).resolve()
    require(experiment == EXPERIMENT, "Auditor pinned to the frozen temporal_01 experiment")
    integrity = Integrity()
    auditor_sha = integrity.bind(Path(__file__).resolve())
    freeze, shadow, summary, skipped = verify_inputs(integrity, experiment)
    journals = load_journals(shadow)
    selection = rebuild_selection(shadow, journals)
    exact(read(experiment / "selection.json"), selection, "independently reconstructed selection")
    records, numerical = verify_observations(experiment, selection, journals, freeze)
    rebuilt = independent_summary(selection, records)
    close_tree({key: summary[key] for key in rebuilt}, rebuilt, "complete independently reconstructed summary")
    for flag in ("classifier_promoted", "output_gate_applied", "detector_rerun", "thresholds_tuned",
                 "raw16_accessed", "sealed_holdouts_accessed", "accuracy_retention_claimed"):
        require(summary[flag] is False, "Unsupported scope or accuracy claim: " + flag)
    for metric in ("airborne_precision", "airborne_recall", "false_alarms_per_minute"):
        require(summary[metric] is None, "Unjustified population metric: " + metric)
    require(set(summary["capture_records"]) == set(COUNTS), "Capture clip scope changed")
    for cid, count in COUNTS.items():
        last = max(o["frame"] for o in selection["observations"] if o["clip"] == cid)
        exact(summary["capture_records"][cid], dict(backend="FFMPEG", reported_frame_count=count,
            decoded_frames=last+1, first_frame=0, last_frame=last, fps=10, shape_hw=[3190, 4784],
            eof_not_checked_beyond_bounded_interval=True), "bounded decoder receipt " + cid)
    integrity.recheck()
    return dict(schema="seaqr.accuracy-v37-temporal-saved-evidence-audit.v1", verified=True,
        experiment=str(experiment), temporal_freeze_sha256=FREEZE_SHA256, auditor_sha256=auditor_sha,
        created_ns=time.time_ns(), checked_file_count=len(integrity.files), checked_files_sha256=integrity.files,
        all_nonmedia_bound_files_rehashed_after_analysis=True,
        source_media_hash_claims_inherited_not_rehashed=skipped, source_video_files_opened=False,
        source_video_decoding_independently_validated=False,
        prior53_global_warp_source_pixels_independently_validated=False,
        current25_native_bytes_match_independently_audited_v36_patches=358,
        exact_selection_independently_reconstructed=True, selected_observations=len(records),
        adjacent_original_journal_geometry_independently_verified=True,
        raw_saved_array_hash_shape_dtype_and_inventory_verified=True,
        prior_photometric_correction_exactly_once_verified=True,
        frozen_registration_and_temporal_modules_shared_for_numerical_replay=True,
        extraction_runner_or_summarize_imported=False,
        selected_temporal_models_independent_full_design_constrained_least_squares=True,
        independently_recomputed_whole_template_bank_optimality=False,
        numerical_validation_tolerances=dict(relative=RTOL, absolute=ATOL),
        numerical_recomputation_counts=numerical, complete_summary_independently_reconstructed=True,
        reference_provenance=rebuilt["reference_provenance"],
        provisional_control_selected_measurements=sum(c["baseline_selected_measurements"] for c in rebuilt["provisional_controls"]),
        previous_actual_measurements=rebuilt["previous_actual_measurements"], diagnostic_available=rebuilt["diagnostic_available"],
        per_scope_availability={key: {name: value[name] for name in ("selected_observations", "previous_actual_measurements",
            "diagnostic_available", "primary_unavailable_reasons", "unavailable_reasons")}
            for key, value in rebuilt["feature_distributions"].items()},
        raw16_accessed=False, sealed_holdouts_accessed=False, remote_state_accessed=False,
        classifier_promoted=False, output_gate_applied=False, thresholds_tuned=False,
        airborne_accuracy_established=False,
        limitations=["Saved-array audit, not independent video decoding or prior global-warp pixel verification",
            "Original source hashes are extraction-receipt claims; this auditor never opens source media",
            "Registration and temporal bank search are replayed using frozen implementations",
            "Selected point/edge fits and three-sigma prior null separately checked with direct constrained least squares",
            "Overlapping development references have unknown physical class; selected controls are provisional",
            "Availability is not object classification, detection retention, or calibrated confidence"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=EXPERIMENT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Exclusive fresh audit output required")
    report = audit(args.experiment)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({key: report[key] for key in ("verified", "selected_observations", "diagnostic_available",
        "numerical_recomputation_counts", "source_video_files_opened")}, allow_nan=False))
