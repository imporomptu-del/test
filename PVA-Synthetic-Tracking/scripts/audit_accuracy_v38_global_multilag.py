"""Saved-evidence audit of V38, with no source-video decoding or sampling.

All frozen evidence is hash-bound. Geometry, membership, support, and summary
are reconstructed independently. Numerical replay uses the frozen feature
copies (a shared-code check); a separately implemented direct-LS check covers a
bounded, deterministic, diverse subset of point/edge/nested fits.
"""
import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/tiny_target/accuracy_v38_20260925"
EXPERIMENT = BASE / "global_multilag_01"
V36 = ROOT / "results/tiny_target/accuracy_v36_20260924"
CONTEXT = V36 / "context_01"
FULL = V36 / "full_context_01"
SHADOW = V36 / "shadow_01"
LAGS = (1, 2, 4, 8)
SHIFTS = tuple((dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1))
METRICS = ("point_gain_fraction", "edge_gain_fraction", "point_minus_edge_fraction",
           "point_amplitude_dn", "residual_rms_dn", "signed_point_amplitude_dn")
SOURCES = {cid: ROOT.parent / f"outputs/jetson_review_clips_20260913/chunk_{cid}.avi"
           for cid in ("0029", "0126")}
SOURCE_HASHES = {"0029": "0330bc3e390a793c2bf6afe7b16720ad3cd6bb8943ee162caf9ce6eff800f359",
                 "0126": "c5302b873656793da47f1da3c03f05df595f17c3f9bc407ce0bfd99b7e718344"}
COUNTS = {"0029": 687, "0126": 674}
RTOL, ATOL = 2e-10, 2e-8  # Numerical validation, never an eligibility tolerance.


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result
    def invalid(token):
        raise ValueError("Nonfinite JSON constant: " + token)
    return json.loads(Path(path).read_text(), object_pairs_hook=unique, parse_constant=invalid)


def sha(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "Regular non-symlink file required: " + str(path))
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            result.update(block)
    return result.hexdigest()


def exact(actual, expected, label):
    require(json.dumps(actual, sort_keys=True, allow_nan=False) == json.dumps(expected, sort_keys=True, allow_nan=False),
            "Exact mismatch: " + label)


def close(actual, expected, label):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and actual.keys() == expected.keys(), "Fields differ: " + label)
        for key, value in expected.items():
            close(actual[key], value, label+"/"+key)
    elif isinstance(expected, (list, tuple)):
        require(isinstance(actual, (list, tuple)) and len(actual) == len(expected), "List differs: " + label)
        for index, (a, b) in enumerate(zip(actual, expected)):
            close(a, b, label+f"/{index}")
    elif type(expected) is float:
        require(type(actual) in (float, int) and math.isfinite(actual)
                and math.isclose(actual, expected, rel_tol=RTOL, abs_tol=ATOL),
                f"Numeric mismatch {label}: {actual!r} versus {expected!r}")
    else:
        require(type(actual) is type(expected) and actual == expected, "Value differs: " + label)


class Integrity:
    def __init__(self):
        self.files = {}

    def bind(self, path, expected=None):
        path = Path(path).resolve()
        if path.suffix.lower() in (".avi", ".mp4", ".raw16"):
            require(path in SOURCES.values(), "Out-of-scope media path")
        else:
            require(path.is_relative_to(ROOT) or path.is_relative_to(ROOT.parent / "outputs/seaqr_accuracy_v37_review_20260924"),
                    "Out-of-scope bound input: " + str(path))
            require(path.suffix in (".json", ".jsonl", ".py", ".md", ".log", ".npz", ".png"),
                    "Unexpected bound input type")
        digest = sha(path)
        require(expected is None or digest == expected, "Digest mismatch: " + str(path))
        require(self.files.setdefault(str(path), digest) == digest, "File changed during binding")
        return digest

    def recheck(self):
        for path, digest in self.files.items():
            require(sha(path) == digest, "Bound evidence changed during audit: " + path)


def bind_experiment(integrity, experiment):
    freeze_sha = integrity.bind(experiment / "freeze.json")
    freeze, summary = read(experiment / "freeze.json"), read(experiment / "summary.json")
    require(freeze["schema"] == "seaqr.accuracy-v38-global-multilag-freeze.v1" and freeze["pre_extraction"] is True,
            "Wrong experiment freeze")
    require(summary["schema"] == "seaqr.accuracy-v38-global-multilag-summary.v1" and summary["completed"] is True
            and summary["freeze_sha256"] == freeze_sha, "Incomplete/unbound summary")
    exact(freeze["lags"], list(LAGS), "fixed lag bank")
    exact(freeze["shifts_xy"], [list(shift) for shift in SHIFTS], "fixed sensitivity probes")
    require(freeze["original_selection_count"] == 358 and freeze["source_frame_buffer_limit"] == 9
            and freeze["prior_association_required"] is False and freeze["local_registration_required"] is False
            and freeze["photometric_gain"] == 1 and freeze["diagnostic_only"] is True
            and freeze["classifier_promoted"] is False and freeze["output_gate_applied"] is False,
            "Frozen diagnostic scope differs")
    exact(freeze["source_videos"], {cid: {"path": str(path), "sha256": SOURCE_HASHES[cid]}
                                   for cid, path in SOURCES.items()}, "allowed source-video identities")
    for path, digest in freeze["inputs_sha256"].items():
        integrity.bind(path, digest)
    for name, digest in freeze["implementation_sha256"].items():
        integrity.bind(ROOT / name, digest)
        integrity.bind(experiment / "implementation" / name, digest)
        require(freeze["inputs_sha256"].get(str(ROOT/name)) == digest
                and freeze["inputs_sha256"].get(str(experiment/"implementation"/name)) == digest,
                "Implementation/live snapshot absent from freeze input pins")
    for name in ("selection.json", "unit.log"):
        require(str(experiment/name) in freeze["inputs_sha256"], "Unfrozen selection or test log")
    for cid, path in SOURCES.items():
        require(freeze["inputs_sha256"].get(str(path)) == SOURCE_HASHES[cid], "Source not pinned before extraction")
    for name, digest in summary["outputs_sha256"].items():
        require(name in ("observations.json", "source_patches.npz"), "Unexpected output inventory")
        integrity.bind(experiment/name, digest)
    exact(set_as_list(summary["outputs_sha256"]), ["observations.json", "source_patches.npz"], "complete output inventory")
    integrity.bind(experiment / "summary.json")
    test_log = (experiment/"unit.log").read_text()
    count = re.search(r"Ran (\d+) tests in", test_log)
    require(count and int(count[1]) >= 88 and re.search(r"\nOK\s*$", test_log), "Frozen test run failed or incomplete")
    # The known historical receipts pin original denominators and source bytes.
    integrity.bind(V36 / "full_context_independent_audit_01.json",
                   "61e2503d624cfce93786bbfd731b5a44dda152822b40035ab22b4f0a6801486d")
    integrity.bind(ROOT / "results/tiny_target/accuracy_v37_20260924/temporal_saved_evidence_audit_01.json",
                   "9076bb4f8b31f9eb50c320c17178f2e3cce339a05008974d0d315f715d7fbf78")
    return freeze, summary, int(count[1])


def set_as_list(value):
    return sorted(value)


def load_journals():
    frozen = read(SHADOW / "freeze.json")
    rows = {}
    for cid, expected in COUNTS.items():
        path = Path(frozen["inputs"][cid]["path"])/"frames.jsonl"
        with path.open() as stream:
            data = [json.loads(line) for line in stream]
        require(len(data) == expected, "Changed parent journal length")
        for index, row in enumerate(data):
            require(type(row["frame_index"]) is int and row["frame_index"] == index
                    and type(row["timestamp_ns"]) is int and row["timestamp_ns"] == index*100_000_000,
                    "Parent journal chronology differs")
            require(type(row["motion"]["reset"]) is bool, "Missing explicit reset provenance")
            identities = [f'{t["segment"]}/{t["track_id"]}' for t in row["tracks"]]
            require(len(set(identities)) == len(identities), "Duplicate original journal identity")
        rows[cid] = data
    return rows


def rebuild_selection(selection, rows, integrity):
    original = read(CONTEXT / "selection.json")
    require(len(original["observations"]) == 358, "Original selection cardinality changed")
    exact(selection["controls"], original["controls"], "unchanged seven controls")
    original_groups = [g for g in selection["groups"] if g["kind"] != "compact_light"]
    exact(original_groups, original["groups"], "all original reference groups")
    for kind, samples, hits in (("dense", 285, 284), ("pilot", 28, 28), ("anchor", 24, 24)):
        groups = [g for g in original_groups if g["kind"] == kind]
        require(len(groups) == samples and sum(bool(g["keys"]) for g in groups) == hits,
                "Dropped original positive or changed denominator")
    require(sum(len(o["provisional_control_indices"]) for o in original["observations"]) == 70,
            "Original provisional-control denominator changed")
    reference_path = BASE / "compact_light_reference_v1.json"
    approval = read(BASE / "compact_light_reference_v1_root_review.json")
    integrity.bind(reference_path, approval["annotation_sha256"])
    require(approval["approved_use"] == "provisional_class_unknown_image_feature_regression"
            and approval["before_v38_scoring"] is True and approval["airborne_truth"] is False,
            "Separate source reference was not approved for this use")
    reference = read(reference_path)
    require(reference["clip"] == "0029" and reference["physical_class"] == "unknown", "Reference scope changed")
    require([s["frame_index"] for s in reference["frames"]] == list(range(12, 25)), "Reference temporal review scope changed")
    visible = [s for s in reference["frames"] if s["visibility"] == "visible"]
    exact([s["frame_index"] for s in visible], [14, 15, 16, 17, 18, 19, 23, 24], "all eight new visible samples")
    with (FULL/"0029_decisions.jsonl").open() as stream:
        decisions = [json.loads(line) for line in stream]
    observations = {o["key"]: copy.deepcopy(o) for o in original["observations"]}
    extra_groups = []
    for sample in visible:
        frame = sample["frame_index"]
        xy = np.asarray(sample["source_xy"], dtype=float)
        radius = sample["position_uncertainty_radius_px"]+2.0
        matches = []
        for track in rows["0029"][frame]["tracks"]:
            if track["measured"] and track["qualified_moving"] and track["track_id"].startswith("bright:"):
                distance = math.hypot(*(np.asarray(track["measurement_source_xy"])-xy))
                if distance <= radius:
                    identity = f'{track["segment"]}/{track["track_id"]}'
                    key = f"0029/{frame}/{identity}"
                    observation = dict(key=key, clip="0029", frame=frame, identity=identity,
                        measurement_source_xy=track["measurement_source_xy"], polarity="bright", provisional_control_indices=[])
                    if key in observations:
                        exact(observations[key], observation, "existing extra-reference observation")
                    observations[key] = observation
                    matches.append((distance, identity, key))
        matches.sort()
        accepted = {f'{t["segment"]}/{t["track_id"]}' for t in decisions[frame]["tracks"] if t["accepted"]}
        extra_groups.append(dict(kind="compact_light", clip="0029", window="class_unknown_compact_light_v1", frame=frame,
            keys=[key for _, _, key in matches], baseline_assigned_id=matches[0][1] if matches else None,
            source_reference_xy=xy.tolist(), uncertainty_px=sample["position_uncertainty_radius_px"], matching_radius_px=radius,
            v36_retained_keys=[key for _, identity, key in matches if identity in accepted]))
    exact([g for g in selection["groups"] if g["kind"] == "compact_light"], extra_groups,
          "independent new-reference matching, including unmatched visible frames")
    ordered = sorted(observations.values(), key=lambda o: (o["clip"], o["frame"], o["identity"]))
    exact(selection["observations"], ordered, "preserved original observations plus exact reference-derived additions")
    exact(selection["additional_reference"], dict(clip="0029", reviewed_frames=13, physical_class="unknown",
        visibility_counts=dict(Counter(s["visibility"] for s in reference["frames"])), source_selected_from_v36_review=True,
        independently_held_out=False, authoritative_airborne_truth=False, matching_extra_radius_px=2.0), "separate reference caveats")
    return {o["key"] for o in original["observations"]}


def metadata(row):
    result = {key: copy.deepcopy(row[key]) for key in ("frame_index", "timestamp_ns", "segment", "source_to_reference", "motion")}
    # Strict parent JSON/hash validation ensures there are no nonfinite scalars.
    json.dumps(result, allow_nan=False)
    result["nonfinite_metadata_fields_replaced_with_null"] = []
    return result


def original_track(row, identity):
    return next((t for t in row["tracks"] if f'{t["segment"]}/{t["track_id"]}' == identity), None)


def matrix_issue(row):
    try:
        value = np.asarray(row["source_to_reference"], dtype=float)
        if value.shape != (3, 3):
            return None, "invalid_transform_shape"
        if not np.isfinite(value).all():
            return None, "nonfinite_transform"
        condition = np.linalg.cond(value)
        if not np.isfinite(condition) or condition > 1e12:
            return None, "singular_or_ill_conditioned_transform"
        return value, None
    except (ValueError, TypeError, np.linalg.LinAlgError):
        return None, "invalid_transform"


def check_array(descriptor, arrays, used, expected_shape, label):
    require(isinstance(descriptor, dict), "Missing array descriptor: " + label)
    key = descriptor["array_key"]
    require(key in arrays and key not in used, "Missing or multiply referenced array: " + key)
    used.add(key)
    value = arrays[key]
    require(value.dtype == np.dtype("float64") and value.shape == expected_shape and not np.isinf(value).any(),
            "Wrong saved-array shape, dtype or infinity: " + label)
    finite = value[np.isfinite(value)]
    require(np.all((finite >= -1e-9) & (finite <= 255+1e-9)), "Saved source pixels outside 8-bit convex range")
    exact(descriptor, dict(array_key=key, shape=list(value.shape), dtype=str(value.dtype),
                          sha256=hashlib.sha256(value.tobytes()).hexdigest(), finite_pixels=int(np.isfinite(value).sum())),
          "array metadata/digest " + label)
    return value


def check_geometry(record, rows, current, priors):
    frame, cid, identity = record["frame"], record["clip"], record["identity"]
    row = rows[cid][frame]
    actual = original_track(row, identity)
    require(actual and actual["measured"] and actual["qualified_moving"], "Current selected track not qualified/measured")
    exact(record["measurement_source_xy"], actual["measurement_source_xy"], "actual source coordinates")
    center = [math.floor(value+.5) for value in actual["measurement_source_xy"]]
    fractional = [value-center[i] for i, value in enumerate(actual["measurement_source_xy"])]
    exact(record["current_integer_center_xy"], center, "current native rounding")
    exact(record["current_xy"], fractional, "current fractional center")
    exact(record["actual_current_source_xy"], actual["measurement_source_xy"], "source-pair actual current xy")
    exact(record["current_frame"], metadata(row), "current frame and full motion provenance")
    exact(record["source_shape_hw"], [3190, 4784], "native source shape")
    size = min(frame+1, 9)
    exact(record["buffer_scope"], dict(frames=size, maximum_frames=9, max_lag=8, oldest_frame=max(0, frame-8),
        current_frame=frame, owned_uint8_bytes=size*3190*4784), "bounded causal buffer")
    cy, cx = np.mgrid[-12:13, -12:13]
    current_mask = (cx+center[0] >= 0) & (cx+center[0] < 4784) & (cy+center[1] >= 0) & (cy+center[1] < 3190)
    require(np.array_equal(np.isfinite(current), current_mask), "Native crop support differs from source geometry")
    require(np.all(current[current_mask] == np.round(current[current_mask])), "Current pixels are not original uint8 samples")
    exact(record["current_finite_pixels"], int(current_mask.sum()), "current source-support count")
    require(len(record["lags"]) == 4, "Dropped fixed lag")
    for lag, item, prior_patch in zip(LAGS, record["lags"], priors):
        first = frame-lag
        prior = rows[cid][first] if first >= 0 else None
        interval = rows[cid][max(0, first+1):frame+1]
        for key, value in dict(lag=lag, current_frame_index=frame, requested_prior_frame_index=first,
            requested_prior_timestamp_ns=first*100_000_000, delta_time_ns=lag*100_000_000,
            optional_identity=identity, geometry_availability_is_not_motion_confidence=True,
            no_local_registration=True, no_photometric_correction=True).items():
            exact(item[key], value, "lag provenance " + key)
        exact(item["intervening_frames"], [metadata(r) for r in interval], "every causal intervening camera row")
        if prior is None:
            require(not item["available"] and item["reasons"] == ["prior_frame_not_yet_available"] and prior_patch is None,
                    "Missing causal source frame silently substituted")
            continue
        exact(item["prior_frame"], metadata(prior), "exact prior source frame")
        previous_track = original_track(prior, identity)
        previous_xy = previous_track["measurement_source_xy"] if previous_track and previous_track["measured"] else None
        expected_status = "actual_measurement" if previous_xy is not None else (
            "identity_present_without_actual_measurement" if previous_track else "identity_not_present")
        exact(item["previous_actual_measurement_available"], previous_xy is not None, "actual prior availability")
        exact(item["previous_actual_measurement_source_xy"], previous_xy, "never-predicted prior location")
        issues = []
        if any(r["segment"] != prior["segment"] for r in interval):
            issues.append("intervening_reference_segment_change")
        if any(r["motion"]["reset"] for r in interval):
            issues.append("intervening_reference_reset")
        hp, pe = matrix_issue(prior)
        hc, ce = matrix_issue(row)
        if pe:
            issues.append("prior_"+pe)
        if ce:
            issues.append("current_"+ce)
        for r in interval[:-1]:
            _, error = matrix_issue(r)
            if error:
                issues.append(f'intervening_frame_{r["frame_index"]}_{error}')
        if issues:
            exact(item["reasons"], issues, "geometry abstention reasons")
            require(item["available"] is False and prior_patch is None and item["evidence"] is None,
                    "Unavailable geometry supplied evidence")
            continue
        # Explicit inverse multiplication is separate from the producer's solve.
        warp = np.linalg.inv(hp) @ hc
        close(item["current_to_prior_matrix"], warp.tolist(), "independent camera composition")
        require(np.isfinite(warp).all() and np.linalg.cond(warp) <= 1e12, "Invalid composed camera transform")
        require(prior_patch is not None, "Geometrically available lag lost its saved source patch")
        yy, xx = np.mgrid[-13:14, -13:14]
        coordinates = np.stack((xx+center[0], yy+center[1], np.ones(xx.shape)))
        mapped = (warp @ coordinates.reshape(3, -1)).reshape(3, 27, 27)
        valid = np.isfinite(mapped).all(axis=0) & (mapped[2] != 0)
        sx = np.divide(mapped[0], mapped[2], out=np.full(xx.shape, np.nan), where=valid)
        sy = np.divide(mapped[1], mapped[2], out=np.full(yy.shape, np.nan), where=valid)
        supported = valid & np.isfinite(sx) & np.isfinite(sy) & (sx >= 0) & (sx <= 4783) & (sy >= 0) & (sy <= 3189)
        require(np.array_equal(supported, np.isfinite(prior_patch)), "Saved prior NaN mask disagrees with projective/source support")
        den = mapped[2][np.isfinite(mapped[2])]
        horizon = bool(den.size and den.min() <= 0 <= den.max())
        support = dict(interpolation="exact_float64_bilinear_positive_weight_support",
            mapping_finite_nonzero_denominator_pixels=int(valid.sum()), prior_finite_pixels=int(supported.sum()),
            projective_horizon_crosses_patch=horizon, denominator_min=float(den.min()) if den.size else None,
            denominator_max=float(den.max()) if den.size else None)
        close(item["support"], support, "independent projective support")
        if horizon:
            issues.append("projective_horizon_crosses_patch")
        if not valid.any():
            issues.append("no_finite_projective_mapping_support")
        exact(item["reasons"], issues, "projective availability reasons")
        exact(item["available"], not issues, "geometry availability only")
        mapped_point = None
        if previous_xy is not None:
            point = np.linalg.solve(warp, [*previous_xy, 1.0])
            if point[2] and np.isfinite(point).all():
                mapped_point = (point[:2]/point[2]-center).tolist()
            else:
                expected_status = "actual_measurement_mapping_unavailable"
        close(item["previous_point_current_grid_xy"], mapped_point, "actual prior point in current grid")
        exact(item["previous_measurement_status"], expected_status, "optional measurement provenance")
        shifts = []
        for dx, dy in SHIFTS:
            common = current_mask & np.isfinite(prior_patch[1+dy:26+dy, 1+dx:26+dx])
            shifts.append(dict(shift_xy=[dx, dy], common_finite_pixels=int(common.sum()), complete=bool(common.all()),
                               previous_point_xy=None if mapped_point is None else [mapped_point[0]-dx, mapped_point[1]-dy]))
        close(item["shift_support"], shifts, "all fixed shifts and actual-point shift sign")


def load_frozen_diagnostic(experiment):
    names = ("accuracy_v36_context", "accuracy_v37_temporal", "accuracy_v38_evidence")
    previous = {name: sys.modules.get(name) for name in names}
    try:
        for name in names:
            path = experiment/"implementation/scripts"/(name+".py")
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        result = sys.modules["accuracy_v38_evidence"].SourceEvidenceDiagnostic()
    finally:
        for name, old in previous.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old
    return result


def flags(item):
    evidence = item["evidence"]
    result = dict(geometry=item["available"], nominal_contrast_available=False, envelope_available=False,
                  nominal_source_supported=False, all_nine_source_supported=False,
                  nominal_joint_available=False, all_nine_joint_available=False)
    if evidence is not None:
        probes = evidence["probes"]
        require([p["shift_xy"] for p in probes] == [list(s) for s in SHIFTS], "Missing or reordered fixed probe")
        informative = [p["source_supported"] and p["contrast_informative"] for p in probes]
        exact(evidence["source_supported_probes"], sum(p["source_supported"] for p in probes), "supported probe count")
        exact(evidence["informative_probes"], sum(informative), "informative probe count")
        exact(evidence["nominal_contrast_available"], informative[4], "nominal availability")
        exact(evidence["envelope_available"], all(informative), "complete-envelope unknown propagation")
        require((evidence["envelope"] is not None) == all(informative), "Incomplete envelope must remain null")
        result.update(nominal_contrast_available=informative[4], envelope_available=all(informative),
                      nominal_source_supported=probes[4]["source_supported"],
                      all_nine_source_supported=all(p["source_supported"] for p in probes),
                      nominal_joint_available=probes[4]["conditional_pair"]["available"],
                      all_nine_joint_available=all(p["conditional_pair"]["available"] for p in probes))
        if all(informative):
            expected = {}
            for metric in METRICS:
                values = [p["difference_features"][metric] for p in probes]
                expected[metric] = dict(minimum=min(values), median=float(np.median(values)), maximum=max(values), span=max(values)-min(values))
            close(evidence["envelope"], expected, "independent nine-probe envelope")
    return result


def scope_summary(records):
    result = dict(observations=len(records), per_lag={})
    derived = {record["key"]: [flags(item) for item in record["lags"]] for record in records}
    names = dict(geometry_available="geometry", nominal_contrast_available="nominal_contrast_available",
                 all_nine_envelope_available="envelope_available", nominal_source_supported="nominal_source_supported",
                 all_nine_source_supported="all_nine_source_supported", nominal_joint_available="nominal_joint_available",
                 all_nine_joint_available="all_nine_joint_available")
    for index, lag in enumerate(LAGS):
        items = [record["lags"][index] for record in records]
        counts = {name: sum(derived[r["key"]][index][metric] for r in records) for name, metric in names.items()}
        counts["previous_actual_measurement_available"] = sum(item["previous_actual_measurement_available"] for item in items)
        counts["geometry_unknown_reasons"] = dict(Counter(reason for item in items for reason in item["reasons"]))
        probes = [probe for item in items if item["evidence"] for probe in item["evidence"]["probes"]]
        counts["probe_unknown_reasons"] = dict(Counter(reason for probe in probes for reason in probe["reasons"]))
        counts["joint_unknown_reasons"] = dict(Counter(reason for probe in probes for reason in probe["conditional_pair"]["reasons"]))
        counts["envelope_distributions"] = {}
        envelopes = [item["evidence"]["envelope"] for item in items if item["evidence"] and item["evidence"]["envelope_available"]]
        for metric in METRICS:
            counts["envelope_distributions"][metric] = {
                field: dict(count=len(envelopes), minimum=min(v[metric][field] for v in envelopes),
                    median=float(np.median([v[metric][field] for v in envelopes])), maximum=max(v[metric][field] for v in envelopes))
                for field in ("minimum", "median", "maximum", "span")} if envelopes else {"count": 0}
        result["per_lag"][str(lag)] = counts
    for metric in ("geometry", "nominal_contrast_available", "envelope_available"):
        result["all_four_lags_"+metric] = sum(all(f[metric] for f in derived[r["key"]]) for r in records)
    return result


def rebuild_summary(selection, records):
    by_key = {r["key"]: r for r in records}
    scopes, groups = defaultdict(set), []
    derived = {key: [flags(lag) for lag in record["lags"]] for key, record in by_key.items()}
    for original in selection["groups"]:
        item = copy.deepcopy(original)
        keys = original["keys"]
        require(len(keys) == len(set(keys)) and set(keys) <= by_key.keys(), "Invalid group key inventory")
        scopes[original["kind"]+"/"+original["window"]].update(keys)
        item["baseline_has_match"] = bool(keys)
        item["diagnostic_available_keys"] = {
            metric: {str(lag): [key for key in keys if derived[key][index][metric]] for index, lag in enumerate(LAGS)}
            for metric in ("geometry", "nominal_contrast_available", "envelope_available")}
        item["all_four_lag_envelope_keys"] = [key for key in keys if all(flag["envelope_available"] for flag in derived[key])]
        groups.append(item)
    for index, control in enumerate(selection["controls"]):
        scopes[f'control/{index}/{control["label"]}'].update(r["key"] for r in records if index in r["provisional_control_indices"])
    scopes["all"].update(by_key)
    for cid in COUNTS:
        scopes["clip/"+cid].update(r["key"] for r in records if r["clip"] == cid)
    references = {}
    for kind in ("dense", "pilot", "anchor", "compact_light"):
        chosen = [g for g in groups if g["kind"] == kind]
        references[kind] = dict(samples=len(chosen), baseline_matched_samples=sum(bool(g["keys"]) for g in chosen),
            per_lag={str(lag): {metric: sum(bool(g["diagnostic_available_keys"][metric][str(lag)]) for g in chosen)
                for metric in ("geometry", "nominal_contrast_available", "envelope_available")} for lag in LAGS},
            all_four_lag_envelope_samples=sum(bool(g["all_four_lag_envelope_keys"]) for g in chosen), detection_retention_claimed=False)
        if kind == "compact_light":
            references[kind]["v36_matched_samples"] = sum(bool(g["v36_retained_keys"]) for g in chosen)
    return dict(selected_observations=len(records), reference_provenance=references, groups=groups,
                scopes={name: scope_summary([by_key[key] for key in sorted(keys)]) for name, keys in sorted(scopes.items())})


class DirectLS:
    """Independent six/seven/eight-column fits with explicit polarity bounds."""
    def __init__(self, center):
        y, x = np.mgrid[-12:13, -12:13].astype(float)
        u, v = x.ravel()/12, y.ravel()/12
        self.background = np.column_stack((np.ones(625), u, v, u*u, u*v, v*v))
        self.points, self.point_info, self.edges, self.edge_info = [], [], [], []
        for sigma in (1., 2., 3.):
            for dx in (-1., 0., 1.):
                for dy in (-1., 0., 1.):
                    self.points.append(np.exp(-((x-center[0]-dx)**2+(y-center[1]-dy)**2)/(2*sigma*sigma)).ravel())
                    self.point_info.append((sigma, dx, dy))
        for width in (1., 2., 4.):
            for k in range(8):
                angle = k*math.pi/8
                for offset in (-2., 0., 2.):
                    self.edges.append(np.tanh((x*math.cos(angle)+y*math.sin(angle)-offset)/width).ravel())
                    self.edge_info.append((width, angle, offset))

    @staticmethod
    def fit(design, value):
        coefficient = np.linalg.lstsq(design, value, rcond=None)[0]
        residual = value-design@coefficient
        return float(residual@residual), float(coefficient[-1])

    def best(self, base, templates, value, energy, sign=None):
        candidates = []
        for column in templates:
            sse, amplitude = self.fit(np.column_stack((base, column)), value)
            if sign is not None and amplitude*sign < 0:
                sse, amplitude = energy, 0.0
            candidates.append((max(0., min(energy, energy-sse)), amplitude, sse))
        winner = int(np.argmax([c[0] for c in candidates]))
        return winner, candidates[winner]

    def measure(self, patch, polarity):
        value = np.asarray(patch, dtype=float).ravel()
        value = value-value.mean()
        energy, _ = self.fit(self.background, value)
        guard = 1e-24*max(1., float(value@value))
        sign = 1 if polarity == "bright" else -1
        pi, (pg, pa, _) = self.best(self.background, self.points, value, energy, sign)
        ei, (eg, ea, edge_energy) = self.best(self.background, self.edges, value, energy)
        nested = np.column_stack((self.background, self.edges[ei]))
        ci, (cg, ca, _) = self.best(nested, self.points, value, edge_energy, sign)
        informative, conditional = energy > guard, edge_energy > guard
        pg, eg, cg = pg/energy if informative else 0., eg/energy if informative else 0., cg/edge_energy if conditional else 0.
        ps, px, py = self.point_info[pi]
        ew, et, eo = self.edge_info[ei]
        cs, cx, cy = self.point_info[ci]
        return dict(informative=bool(informative), background_residual_energy=energy,
            point_gain_fraction=pg, edge_gain_fraction=eg, point_minus_edge_fraction=pg-eg,
            point_amplitude_dn=sign*pa if informative else 0., point_sigma_px=ps, point_offset_xy=[px, py],
            edge_width_px=ew, edge_orientation_rad=et, edge_offset_px=eo, edge_amplitude_dn=ea if informative else 0.,
            residual_rms_dn=math.sqrt(energy/625), point_absolute_gain=pg*energy, edge_absolute_gain=eg*energy,
            edge_residual_energy=edge_energy, conditional_informative=bool(conditional), point_gain_after_edge_fraction=cg,
            point_after_edge_absolute_gain=cg*edge_energy, point_after_edge_amplitude_dn=sign*ca if conditional else 0.,
            point_after_edge_sigma_px=cs, point_after_edge_offset_xy=[cx, cy])


def audit(experiment, output):
    experiment, output = Path(experiment).resolve(), Path(output).resolve()
    require(experiment == EXPERIMENT, "Only the frozen global_multilag_01 run is in scope")
    require(not output.exists() and output.parent == BASE, "Fresh JSON receipt directly under V38 base required")
    integrity = Integrity()
    auditor_sha = integrity.bind(Path(__file__).resolve())
    freeze, summary, tests = bind_experiment(integrity, experiment)
    selection, records, rows = read(experiment/"selection.json"), read(experiment/"observations.json"), load_journals()
    originals = rebuild_selection(selection, rows, integrity)
    expected = {o["key"]: o for o in selection["observations"]}
    require(len(records) == len(expected) == freeze["selected_observations"], "Observation count changed")
    exact([r["key"] for r in records], [o["key"] for o in selection["observations"]], "complete ordered selected observation inventory")
    original_patches = {r["key"]: r["patch_sha256"] for r in read(CONTEXT/"observations.json")}
    model = load_frozen_diagnostic(experiment)
    used, replayed, probe_count = set(), 0, 0
    examples, categories = [], set()
    memberships = defaultdict(set)
    for group in selection["groups"]:
        for key in group["keys"]:
            memberships[key].add(group["kind"]+"/"+group["window"])
    with np.load(experiment/"source_patches.npz", allow_pickle=False) as arrays:
        require(len(arrays.files) == len(set(arrays.files)), "Duplicate archive array keys")
        for index, record in enumerate(records):
            selected = expected[record["key"]]
            exact({key: record[key] for key in selected}, selected, "immutable selected observation")
            exact(record["source_sha256"], SOURCE_HASHES[record["clip"]], "source identity")
            exact(record["v36_native_patch_crosschecked"], record["key"] in originals, "original patch membership")
            current = check_array(record["current25_array"], arrays, used, (25, 25), record["key"]+"/current")
            if record["key"] in originals:
                require(np.isfinite(current).all() and hashlib.sha256(current.astype(np.uint8).tobytes()).hexdigest() == original_patches[record["key"]],
                        "Original V36 native patch bytes changed")
            priors = [check_array(item["prior27_array"], arrays, used, (27, 27), record["key"]+f'/lag{item["lag"]}')
                      if item["prior27_array"] is not None else None for item in record["lags"]]
            check_geometry(record, rows, current, priors)
            for item, prior in zip(record["lags"], priors):
                if not item["available"]:
                    require(item["evidence"] is None, "Unknown geometry cannot produce diagnostic evidence")
                    continue
                regenerated = model.measure(current, prior, polarity=record["polarity"], current_xy=record["current_xy"],
                                            previous_xy=item["previous_point_current_grid_xy"])
                close(item["evidence"], regenerated, record["key"]+f'/lag{item["lag"]}/frozen replay')
                flags(item)
                replayed += 1
                probe_count += len(regenerated["probes"])
                # Deterministic bounded sample: first fitted probe per clip,
                # polarity and lag, plus first fit in each reference/control
                # category. Not selected by how well LS happens to agree.
                strata = {(record["clip"], record["polarity"], item["lag"])}
                strata.update(("reference", name) for name in memberships[record["key"]])
                if record["provisional_control_indices"]:
                    strata.add(("control", record["clip"]))
                unseen = strata-categories
                candidate = next((p for p in regenerated["probes"] if p["contrast_informative"]), None)
                if unseen and candidate is not None:
                    require(len(examples) < 32, "Direct-LS sample exceeded frozen bounded audit scope")
                    dx, dy = candidate["shift_xy"]
                    diagnostic = DirectLS(record["current_xy"])
                    direct = diagnostic.measure(current-prior[1+dy:26+dy, 1+dx:26+dx], record["polarity"])
                    saved = {key: value for key, value in candidate["difference_features"].items() if key != "signed_point_amplitude_dn"}
                    close(saved, direct, record["key"]+"/independent direct LS")
                    sign = 1 if record["polarity"] == "bright" else -1
                    close(candidate["difference_features"]["signed_point_amplitude_dn"], sign*direct["point_amplitude_dn"], "signed DN amplitude")
                    examples.append(dict(key=record["key"], lag=item["lag"], shift_xy=[dx, dy],
                                         strata=[list(s) for s in sorted(unseen, key=str)],
                                         maximum_numeric_error=max(abs(saved[k]-v) for k, v in direct.items() if type(v) is float)))
                    categories.update(unseen)
            if (index+1) % 50 == 0:
                print(f"Audited {index+1}/{len(records)} observations", flush=True)
        require(used == set(arrays.files), "Unreferenced or missing saved patch arrays")
    rebuilt = rebuild_summary(selection, records)
    for name, value in rebuilt.items():
        close(summary[name], value, "independently reconstructed summary/"+name)
    for key, value in dict(diagnostic_only=True, detector_rerun=False, classifier_promoted=False,
        output_gate_applied=False, airborne_accuracy_established=False, thresholds_tuned=False,
        raw16_sources_accessed=False, sealed_holdouts_accessed=False, original_native_patches_crosschecked=358).items():
        exact(summary[key], value, "scope claim " + key)
    for cid, record in summary["capture_records"].items():
        require(cid in COUNTS and isinstance(record["backend"], str) and record["backend"], "Invalid decoder identity")
        last = max(r["frame"] for r in records if r["clip"] == cid)
        exact({k: v for k, v in record.items() if k != "backend"}, dict(first_frame=0, last_frame=last,
            decoded_frames=last+1, source_frames=COUNTS[cid], nominal_fps=10, max_buffer_frames=9,
            source_shape_hw=[3190, 4784]), "bounded decode provenance")
    integrity.recheck()
    receipt = dict(schema="seaqr.accuracy-v38-saved-evidence-audit.v1", verified=True, experiment=str(experiment),
        experiment_freeze_sha256=sha(experiment/"freeze.json"), auditor_sha256=auditor_sha,
        selected_observations=len(records), original_observations_preserved=358,
        extra_reference_visible_samples=rebuilt["reference_provenance"]["compact_light"]["samples"],
        frozen_unit_tests_passed=tests, array_count=len(used), every_array_digest_verified=True,
        source_geometry_and_nan_support_independently_reconstructed=True,
        every_intervening_motion_record_verified=True, optional_actual_prior_coordinates_verified=True,
        frozen_feature_lag_replays=replayed, frozen_feature_probe_replays=probe_count,
        complete_summary_independently_reconstructed=True, original_reference_denominators_preserved=True,
        reference_provenance=rebuilt["reference_provenance"],
        independent_direct_least_squares=dict(examples=examples, count=len(examples),
            coverage="Fractional current-point, edge, and point-after-best-edge full 6/7/8-column fits",
            optional_prior_current_joint_model_independently_refitted=False, rtol=RTOL, atol=ATOL),
        source_video_files_opened_for_hashing=True, source_video_paths_hashed=[str(p) for p in SOURCES.values()],
        source_video_decoding_repeated=False, source_video_to_sampled_array_values_independently_validated=False,
        shared_frozen_feature_code_used_for_complete_replay=True,
        producer_summary_or_sampler_imported=False, raw16_accessed=False, sealed_holdouts_accessed=False,
        classifier_promoted=False, accuracy_established=False,
        boundary=["Saved-array numerical/provenance audit, not an independent source decoder or sampler replay",
                  "Complete feature replay shares frozen implementation; direct LS is bounded and does not refit the optional joint model",
                  "Reference availability is not detection retention, and physical airborne class remains unknown"],
        checked_files_sha256=integrity.files, checked_file_count=len(integrity.files), all_bound_files_rehashed_after_analysis=True)
    with output.open("x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(verified=True, observations=len(records), lag_replays=replayed, direct_ls_examples=len(examples))), flush=True)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=EXPERIMENT)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    audit(arguments.experiment, arguments.output)
