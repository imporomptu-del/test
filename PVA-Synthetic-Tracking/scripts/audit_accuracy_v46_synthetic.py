"""Independent synthetic V46 geometry/provenance and paired-accounting audit.

The forecast check uses exact-rational normal-equation intercept weights and
independent relative-coordinate least squares, not the generator helper. No
real media, journals, cached real patches or remote systems are accessed.
"""
from __future__ import annotations

import hashlib
import argparse
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import json
from pathlib import Path
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
V45 = ROOT/"results/tiny_target/accuracy_v45_20260925/synthetic_01"
V45_RECEIPT = "a01c91b7b223986e4bb90f6eb6a1237ff6799c6d7d057a78e44a632b1ab9a5de"
EXPECTED_IDS = (
    "ordinary_prior_forecast", "current_absent", "prior_bias_halfpx", "prior_bias_twopx",
    "prior_jitter_halfpx", "prior_jitter_twopx", "current_departure_halfpx",
    "current_departure_twopx", "current_departure_sixpx", "combined_measurement_and_current_error",
    "current_psf_wide", "current_psf_elliptic", "psf_broadens_over_time", "current_dim_quarter",
    "brightness_ramp", "intermittent_source_history", "missing_three_measurements",
    "missing_four_measurements", "incorrect_last_two_measurements", "reversed_measurement_order",
    "persistent_fixed_at_forecast", "blinking_fixed_at_forecast", "persistent_fixed_near_forecast",
    "blinking_fixed_near_forecast", "moving_plus_fixed_at_forecast", "moving_plus_fixed_near_forecast",
    "observational_twin_moving", "observational_twin_sequential_fixed",
)
CURRENT_ONLY_IDS = (EXPECTED_IDS[0], EXPECTED_IDS[1], *EXPECTED_IDS[6:9],
                    EXPECTED_IDS[10], EXPECTED_IDS[11], EXPECTED_IDS[13], *EXPECTED_IDS[-2:])
ARRAY_KEYS = ("current129", "history129", "prior_centers_xy", "predicted_offset_xy")
ADAPTER_KEYS = (*ARRAY_KEYS, "polarity")
ARMS = {"amplitude_old_bounds": ("amplitude", "v43"), "presence_old_bounds": ("numerator", "v43"),
        "amplitude_box_bounds": ("amplitude", "v45"), "presence_box_bounds": ("numerator", "v45")}
NOMINAL_FIELDS = ("learned_design_sha256", "common_support_sha256", "common_support_count",
                  "components", "prior_context", "ambiguity_reasons", "conditional_on", "uncertainty_excludes")
RENDER_ATOL = 2e-12


def require(condition, message):
    if not condition:
        raise ValueError(message)


def closed_form_forecast(prior_times, reported_world_centers):
    """Predict time zero from the last four available reported past positions.

    Current truth is intentionally not an argument. The return includes the
    exact original history indices, so missing observations are not retimed.
    """
    times = np.asarray(prior_times, dtype=float)
    require(times.ndim == 1 and len(times) == len(reported_world_centers), "Matched one-dimensional prior times required")
    require(np.isfinite(times).all() and np.all(times < 0) and np.all(np.diff(times) > 0),
            "Forecast timestamps must be finite, strictly increasing and strictly prior")
    selected = [index for index, center in enumerate(reported_world_centers) if center is not None][-4:]
    require(len(selected) == 4, "Four reported prior positions required; no current-truth fallback allowed")
    points = np.asarray([reported_world_centers[index] for index in selected], dtype=float)
    require(points.shape == (4, 2) and np.isfinite(points).all(), "Reported positions must be finite XY pairs")
    t = times[selected]
    centered_t = t-float(np.mean(t))
    denominator = float(np.dot(centered_t, centered_t))
    require(denominator > 0, "Distinct prior timestamps required")
    point_mean = np.mean(points, axis=0)
    slope = np.sum(centered_t[:, None]*(points-point_mean), axis=0)/denominator
    forecast = point_mean-slope*float(np.mean(t))
    return forecast, selected


def expected_patch_geometry(predicted_world_xy, reported_world_centers):
    """World-to-local conversion around a prior-derived integer crop center."""
    forecast = np.asarray(predicted_world_xy, dtype=float)
    require(forecast.shape == (2,) and np.isfinite(forecast).all(), "Finite forecast XY required")
    rounded_center = np.floor(forecast+.5)
    origin = rounded_center-64.
    offset = forecast-rounded_center
    local = [None if center is None else (np.asarray(center, dtype=float)-origin).tolist()
             for center in reported_world_centers]
    return dict(crop_center_world_xy=rounded_center.tolist(), crop_origin_world_xy=origin.tolist(),
                predicted_offset_xy=offset.tolist(), prior_centers_xy=local)


def array_fingerprint(value):
    array = np.asarray(value)
    digest = hashlib.sha256()
    digest.update(str((array.shape, array.dtype.str)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def rational_normal_equation_forecast(times, centers):
    """Independent exact-rational intercept weights from 2x2 normal equations.

    Unlike the generator's centered dot product, this uses integer timestamp
    sums and exact representations of the supplied binary-float coordinates.
    It also avoids rounding a half-integer forecast across a crop boundary.
    """
    _, selected = closed_form_forecast(times, centers)
    t = [Fraction(float(times[i])) for i in selected]
    count, first, second = len(t), sum(t), sum(x*x for x in t)
    denominator = count*second-first*first
    weights = [(second-first*x)/denominator for x in t]
    predicted = [float(sum(w*Fraction(float(centers[i][axis])) for w, i in zip(weights, selected)))
                 for axis in (0, 1)]
    # A second, floating, independently solved least-squares cross-check.
    design = np.column_stack((np.asarray(t, dtype=float), np.ones(count)))
    points = np.asarray([centers[i] for i in selected], dtype=float)
    relative_intercept = np.linalg.lstsq(design, points-points[0], rcond=None)[0][1]
    require(np.allclose(points[0]+relative_intercept, predicted, rtol=0, atol=1e-11),
            "Independent rational and least-squares forecasts disagree")
    return predicted, selected, [float(w) for w in weights]


def near(actual, expected, message, atol=1e-12):
    require(np.shape(actual) == np.shape(expected) and
            np.allclose(actual, expected, rtol=0, atol=atol, equal_nan=False), message)


def independent_render(truth, origin):
    """Render the declared world, not the reported detector measurements.

    Uses an expanded elliptical quadratic instead of the generator's rotated
    coordinate formula. The small reported comparison tolerance is arithmetic
    only; it is not the V45 image uncertainty contract.
    """
    xx, yy = np.meshgrid(np.arange(129, dtype=float)+origin[0],
                         np.arange(129, dtype=float)+origin[1])
    background = 50.+np.where(xx >= 64., 20., 0.)+8.*np.sin(yy/12.)

    def point(center, amplitude, sigma, angle):
        cosine, sine = np.cos(angle), np.sin(angle)
        inv_x, inv_y = 1./float(sigma[0])**2, 1./float(sigma[1])**2
        a = cosine*cosine*inv_x+sine*sine*inv_y
        b = cosine*sine*(inv_x-inv_y)
        c = sine*sine*inv_x+cosine*cosine*inv_y
        dx, dy = xx-center[0], yy-center[1]
        return amplitude*np.exp(-.5*(a*dx*dx+2*b*dx*dy+c*dy*dy))

    history = np.stack([background+point(center, truth["prior_peak_dn"][i],
        truth["prior_psf_sigma_xy"][i], truth["prior_psf_angle_radians"][i])
        for i, center in enumerate(truth["apparent_point_prior_world_centers_xy"])])
    current = background+point(truth["apparent_point_current_world_center_xy"], truth["current_peak_dn"],
                               truth["current_psf_sigma_xy"], truth["current_psf_angle_radians"])
    fixed = truth["fixed_emitter"]
    if fixed is not None:
        for i in range(8):
            history[i] += point(fixed["center_world_xy"], fixed["prior_peak_dn"][i],
                                fixed["psf_sigma_xy"], fixed["psf_angle_radians"])
        current += point(fixed["center_world_xy"], fixed["current_peak_dn"],
                         fixed["psf_sigma_xy"], fixed["psf_angle_radians"])
    return history, current


def audit_geometry(cases, manifest=None):
    """Check all predeclared inputs without evaluating any numerical solver."""
    require(tuple(c["case_id"] for c in cases) == EXPECTED_IDS, "All 28 declared cases must remain in order")
    specifications = None if manifest is None else {c["case_id"]: c for c in manifest["scenarios"]}
    if manifest is not None:
        require(manifest["case_count"] == 28 and tuple(specifications) == EXPECTED_IDS,
                "Manifest denominator or case membership changed")
        require(tuple(manifest["current_only_same_history_geometry_group"]) == CURRENT_ONLY_IDS,
                "Current-only manifest group changed")
        require(manifest["observational_twins"] == list(EXPECTED_IDS[-2:]), "Twin membership changed")
    true_prior = np.column_stack((32.+4.*np.arange(8), np.full(8, 64.)))
    jitter = np.asarray(((1, 0), (-1, 1), (0, -1), (1, 1), (-1, 0), (0, 1), (1, -1), (-1, -1)))
    array_hashes, records, max_error = {}, [], 0.
    for case in cases:
        case_id = case["case_id"]
        require(set(case) == {"case_id", "family", "adapter_inputs", "generator_truth", "provenance"},
                case_id+": undeclared case fields")
        inputs, truth, provenance = case["adapter_inputs"], case["generator_truth"], case["provenance"]
        require(set(inputs) == set(ADAPTER_KEYS), case_id+": adapter contains truth or undeclared inputs")
        require(inputs["polarity"] == "bright", case_id+": polarity changed")
        require(provenance["prior_times"] == list(range(-8, 0)) and provenance["current_time"] == 0,
                case_id+": original timestamps changed")
        for field in ("forecast_uses_current_image", "forecast_uses_current_truth", "crop_uses_current_image",
                      "crop_uses_current_truth", "real_media_read"):
            require(provenance[field] is False, case_id+": current/real-image provenance violation")
        require(provenance["synthetic_only"] is True and
                provenance["forecast_is_synthetic_linear_test_not_real_quadratic_tracker"] is True,
                case_id+": synthetic forecast limitation omitted")
        require(truth["physical_identity_certified_from_images"] is False and
                truth["v45_fixed_geometry_bound_covers_deliberate_mismatch"] is False,
                case_id+": unsupported certification claimed")
        parameters = provenance["generation_parameters"]
        if specifications is not None:
            spec = specifications[case_id]
            require(parameters == spec["generation_parameters"] and case["family"] == spec["family"] and
                    truth["deliberate_model_mismatch_axes"] == spec["deliberate_model_mismatch_axes"],
                    case_id+": predeclared scenario changed")
        reported = true_prior.copy()
        if parameters["measurement_order"] == "reversed":
            reported = reported[::-1].copy()
        else:
            require(parameters["measurement_order"] == "chronological", "Unknown measurement order")
        reported += np.asarray(parameters["prior_measurement_bias_xy"])
        reported += float(parameters["prior_jitter_scale"])*jitter
        for index, displacement in parameters["incorrect_measurement_offsets"].items():
            reported[int(index)] += np.asarray(displacement)
        reported = [None if i in parameters["missing_measurement_indices"] else row.tolist()
                    for i, row in enumerate(reported)]
        require(reported == provenance["reported_prior_world_centers_xy"], case_id+": reported history was altered")
        forecast, selected, weights = rational_normal_equation_forecast(provenance["prior_times"], reported)
        require(selected == provenance["ols_history_indices"], case_id+": forecast selected wrong prior frames")
        near(provenance["ols_times"], np.asarray(provenance["prior_times"])[selected], case_id+": forecast retimed measurements", 0)
        near(provenance["ols_weights"], weights, case_id+": forecast weights changed")
        near(provenance["predicted_world_xy"], forecast, case_id+": forecast uses information beyond prior measurements")
        expected = expected_patch_geometry(forecast, reported)
        for field, expected_key in (("integer_crop_center_world_xy", "crop_center_world_xy"),
                                    ("crop_origin_world_xy", "crop_origin_world_xy"),
                                    ("predicted_offset_xy", "predicted_offset_xy")):
            near(provenance[field], expected[expected_key], case_id+": crop/forecast geometry changed", 1e-12)
        require(provenance["local_reported_prior_centers_xy"] == expected["prior_centers_xy"] and
                inputs["prior_centers_xy"] == expected["prior_centers_xy"], case_id+": adapter prior coordinates changed")
        near(inputs["predicted_offset_xy"], expected["predicted_offset_xy"], case_id+": adapter forecast offset changed")
        near(truth["apparent_point_prior_world_centers_xy"], true_prior, case_id+": measured errors leaked into world truth", 0)
        near(truth["apparent_point_current_world_center_xy"], np.asarray([64., 64.])+parameters["current_departure_xy"],
             case_id+": current truth followed the reported forecast", 0)
        for field in ("prior_peak_dn", "current_peak_dn", "prior_psf_sigma_xy", "prior_psf_angle_radians",
                      "current_psf_sigma_xy", "current_psf_angle_radians", "fixed_emitter"):
            require(truth[field] == parameters[field], case_id+": truth/image parameter mismatch")
        require(truth["latent_world_interpretation"] == parameters["identity_world"], case_id+": latent interpretation changed")
        sequential = parameters["identity_world"] == "sequential_fixed_emitters"
        moving = parameters["current_peak_dn"] > 0 and not sequential
        fixed = bool((parameters["fixed_emitter"] is not None and parameters["fixed_emitter"]["current_peak_dn"] > 0)
                     or (sequential and parameters["current_peak_dn"] > 0))
        require(truth["current_moving_source_present"] == moving and truth["current_fixed_emitter_present"] == fixed and
                truth["any_current_emitter_present"] == (moving or fixed), case_id+": emitter truth flags disagree")
        require(truth["physical_identity_ambiguous"] == (parameters["fixed_emitter"] is not None or case["family"] == "observational_identity"),
                case_id+": physical ambiguity metadata lost")
        rendered_history, rendered_current = independent_render(truth, expected["crop_origin_world_xy"])
        for key, shape, rendered in (("history129", (8, 129, 129), rendered_history), ("current129", (129, 129), rendered_current)):
            array = inputs[key]
            require(array.shape == shape and array.dtype == np.float64 and np.isfinite(array).all(), case_id+": invalid image array")
            error = float(np.max(np.abs(array-rendered)))
            max_error = max(max_error, error)
            require(error <= RENDER_ATOL, case_id+": independent world rendering mismatch")
            array_hashes[case_id+":"+key] = array_fingerprint(array)
        group = case_id in CURRENT_ONLY_IDS
        require(provenance["prior_equivalence_group"] == ("base_prior_world" if group else "unique_"+case_id),
                case_id+": prior-equivalence group changed")
        require(provenance["current_only_same_history_geometry_group"] == ("ordinary_current_variants" if group else None),
                case_id+": current-only group changed")
        records.append(dict(case_id=case_id, actual_prior_measurements=sum(p is not None for p in reported),
                            ols_indices=selected, independently_predicted_world_xy=forecast,
                            current_truth_minus_forecast_xy=(np.asarray(truth["apparent_point_current_world_center_xy"])-forecast).tolist(),
                            render_max_abs_error=error))
    mapping = {case["case_id"]: case for case in cases}
    ordinary = mapping[CURRENT_ONLY_IDS[0]]["adapter_inputs"]
    for case_id in CURRENT_ONLY_IDS:
        inputs = mapping[case_id]["adapter_inputs"]
        require(array_fingerprint(inputs["history129"]) == array_fingerprint(ordinary["history129"]),
                "Current-only variants changed historical pixels")
        for key in ("prior_centers_xy", "predicted_offset_xy", "polarity"):
            require(inputs[key] == ordinary[key], "Current-only variants changed prior geometry")
    for case_id in EXPECTED_IDS[-2:]:
        inputs = mapping[case_id]["adapter_inputs"]
        require(array_fingerprint(inputs["current129"]) == array_fingerprint(ordinary["current129"]),
                "Observational twins are not byte-identical")
    require(mapping[EXPECTED_IDS[-2]]["generator_truth"]["latent_world_interpretation"] !=
            mapping[EXPECTED_IDS[-1]]["generator_truth"]["latent_world_interpretation"], "Twins lost independent latent worlds")
    return dict(passed=True, case_count=28, images_reconstructed=28*9, pixel_values_compared=28*9*129*129,
                render_arithmetic_atol=RENDER_ATOL, maximum_render_abs_error=max_error,
                current_only_group_count=1, current_only_group_cases=len(CURRENT_ONLY_IDS),
                history_equivalence_group_count=19, observational_twins_and_ordinary_inputs_identical=True,
                forecast_crosschecks="Exact rational normal-equation intercept plus independent relative-position least squares",
                per_case=records, image_array_sha256=array_hashes,
                scope="Synthetic input-flow and rendering audit; no detector, association, camera-motion or physical-label validation")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def verify_bindings(bindings, run):
    """Reject any binding outside source/tests/plans and synthetic result trees."""
    synthetic_roots = (ROOT/"results/tiny_target/accuracy_v44_20260925/synthetic_01", V45, run)
    for name, expected in bindings.items():
        path = Path(name).resolve()
        safe = (ROOT/"scripts" in path.parents or ROOT/"tests/unit" in path.parents or
                path in [ROOT/"docs"/f"accuracy_v{version}_plan.md" for version in (44, 45, 46)] or
                any(parent in path.parents for parent in synthetic_roots))
        require(safe, "Binding outside synthetic audit allowlist: "+name)
        require(sha(path) == expected, "Frozen hash mismatch: "+name)


def evidence_counts(records, arm):
    result = dict(states=0, unavailable=0, available=0, positive=0, negative=0, unresolved=0)
    reasons = Counter()
    for record in records:
        value = record["arms"][arm]["raw_adapter_result"]
        result["states"] += 1
        if value["available"]:
            result["available"] += 1
            result[value["numerical_contrast"]["coefficient_sign"]] += 1
        else:
            result["unavailable"] += 1
            reasons.update(value["reasons"])
    return dict(counts=result, unavailable_reasons=dict(reasons))


def audit_evidence(records):
    # This immutable independent V45 validator is reused for the same unchanged
    # quantities. No numerical fit is rerun and no result-dependent arm is chosen.
    from audit_accuracy_v45_accounting import validate_evidence, finite_json
    require(finite_json(records), "Nonfinite saved result")
    details = []
    for record in records:
        require(set(record["arms"]) == set(ARMS), "A mathematical arm was omitted")
        reference = record["arms"]["amplitude_old_bounds"]["raw_adapter_result"]
        detail = dict(case_id=record["case_id"], arms={}, ambiguity_reasons=reference["ambiguity_reasons"],
                      common_support_count=reference["common_support_count"])
        for arm, (quantity, version) in ARMS.items():
            wrapped = record["arms"][arm]
            require((wrapped["arm"], wrapped["quantity"], wrapped["bound_version"]) == (arm, quantity, version),
                    "Mathematical arm/version/units mislabeled")
            require(wrapped["legacy_numerical_contrast_field_contains"] == quantity, "Numerator mislabeled as amplitude")
            value = wrapped["raw_adapter_result"]
            for layer in (wrapped, value):
                require(layer["motion_status"] == layer["physical_class"] == "unknown" and
                        layer["is_motion_or_classification_gate"] is False,
                        "Numerical evidence promoted to a motion/class decision")
            require(value["synthetic_only"] is True and value["production_changed"] is False,
                    "Synthetic adapter origin or production scope changed")
            for field in NOMINAL_FIELDS:
                require(value[field] == reference[field], "Arm altered nominal design/support/ambiguity: "+field)
            require("physical_motion_and_class_not_certified_by_source_contrast" in value["ambiguity_reasons"],
                    "Physical identity ambiguity omitted")
            require(value["prior_context"]["templates_use_prior_images_only"] is True and
                    value["prior_context"]["minimum_actual_prior_centers"] == 5 and
                    value["prior_context"]["supplied_forecast_offset_independently_verified"] is False,
                    "Adapter historical/provenance contract changed")
            score = value["numerical_contrast"]
            require(value["available"] is (score is not None and score["available"]), "Availability changed meaning")
            require(value["reasons"] == [] if value["available"] else bool(value["reasons"]),
                    "Availability/reasons inconsistent")
            if score is not None:
                validate_evidence(score, quantity)
            bounds = value["component_bounds"]
            if bounds is not None:
                metadata = bounds["metadata"]
                require(metadata["aligned_value_error_bound_dn"] == .5 and
                        metadata["moving_residual_stamp_error_bound_dn"] == 1. and
                        metadata["fixed_highpass_stamp_error_bound_dn"] == 1., "Image error contract changed")
                if version == "v45":
                    require(metadata["nominal_components_unchanged"] is True and
                            metadata["moving"]["uncertain_used_stamp_omitted"] is False,
                            "Bounds altered nominal components or removed used stamps")
                original = reference["component_bounds"]
                if original is not None:
                    original_meta = original["metadata"]
                    require(metadata["moving"]["used_history_indices"] == original_meta["moving"]["used_history_indices"],
                            "Moving-template used stamp membership changed")
                    require([(x["centre_xy"], x["used_history_indices"]) for x in metadata["fixed"]] ==
                            [(x["centre_xy"], x["used_history_indices"]) for x in original_meta["fixed"]],
                            "Fixed alternative identities or used stamps changed")
            detail["arms"][arm] = dict(available=value["available"], reasons=value["reasons"], quantity=quantity,
                interval=None if score is None else score["interval"],
                coefficient_sign=None if score is None else score["coefficient_sign"])
        for pair in (("amplitude_old_bounds", "presence_old_bounds"), ("amplitude_box_bounds", "presence_box_bounds")):
            require(record["arms"][pair[0]]["raw_adapter_result"]["component_bounds"] ==
                    record["arms"][pair[1]]["raw_adapter_result"]["component_bounds"], "Solver changed component bounds")
        details.append(detail)
    return details


def expected_arrays(adapter):
    return dict(current129=np.asarray(adapter["current129"]), history129=np.asarray(adapter["history129"]),
                prior_centers_xy=np.asarray([[np.nan, np.nan] if point is None else point
                                           for point in adapter["prior_centers_xy"]]),
                predicted_offset_xy=np.asarray(adapter["predicted_offset_xy"]))


def audit(run):
    """Audit completed saved synthetics only; this function never scores cases."""
    from accuracy_v46_synthetic_cases import build_cases, scenario_manifest
    from run_accuracy_v44_synthetic import causal_cases
    run = Path(run).resolve()
    require(run.parent == ROOT/"results/tiny_target/accuracy_v46_20260925", "Unexpected synthetic audit output tree")
    receipt = load(run/"completion_receipt.json")
    require(receipt["completed"] is True and receipt["synthetic_only"] is True and
            receipt["production_changed"] is False and receipt["real_packet_data_read"] is False, "Run not completed synthetics")
    require(sha(V45/"completion_receipt.json") == V45_RECEIPT, "Pinned V45 receipt changed")
    old_receipt = load(V45/"completion_receipt.json")
    verify_bindings(receipt["files_sha256"], run)
    verify_bindings(old_receipt["files_sha256"], run)
    freeze, saved, start, metadata = [load(run/name) for name in
        ("freeze.json", "inputs_complete.json", "score_start.json", "input_manifest.json")]
    baseline, stress, summary = [load(run/name) for name in ("baseline_results.json", "stress_results.json", "summary.json")]
    require(freeze["synthetic_only"] is True and freeze["v45_receipt_sha256"] == V45_RECEIPT and
            freeze["arms"] == {key: list(value) for key, value in ARMS.items()}, "Frozen baseline or arms changed")
    require(all(freeze["dependency_sha256"].get(name) == digest for name, digest in old_receipt["files_sha256"].items()) and
            freeze["dependency_sha256"].get(str(V45/"completion_receipt.json")) == V45_RECEIPT,
            "V45 frozen dependencies no longer bound")
    expected_receipt_files = set(freeze["dependency_sha256"]) | set(saved["inputs_sha256"]) | {
        str(run/name) for name in ("freeze.json", "inputs_complete.json", "score_start.json", "input_manifest.json",
                                   "baseline_results.json", "stress_results.json", "summary.json")}
    require(set(receipt["files_sha256"]) == expected_receipt_files, "Receipt accounting denominator changed")
    for bindings in (freeze["dependency_sha256"], saved["inputs_sha256"]):
        require(all(receipt["files_sha256"].get(name) == digest for name, digest in bindings.items()), "Nested hash binding changed")
    require(start["freeze_sha256"] == sha(run/"freeze.json") and
            start["inputs_complete_sha256"] == sha(run/"inputs_complete.json") and
            start["input_manifest_sha256"] == sha(run/"input_manifest.json"), "Score-start hash binding mismatch")
    require(saved["scores_started"] is False, "All inputs were not declared saved before scoring")
    chronology = [datetime.fromisoformat(item["created_at_utc"]) for item in (freeze, saved, start, receipt)]
    require(chronology == sorted(chronology), "Producer chronology out of order")
    fresh_stress, fresh_baseline, manifest = build_cases(), causal_cases(), scenario_manifest()
    require(freeze["scenario_manifest"] == manifest, "Predeclared manifest changed")
    geometry = audit_geometry(fresh_stress, manifest)
    require(baseline == load(V45/"causal_results.json"), "V45 six-case/four-arm baseline changed")
    require(len(baseline) == len(fresh_baseline) == 6 and
            [record["case_id"] for record in baseline] == [case["case_id"] for case in fresh_baseline], "Baseline denominator/order changed")
    require(tuple(record["case_id"] for record in stress) == EXPECTED_IDS, "Stress denominator/order changed")
    expected_metadata = {"baseline": [], "stress": []}
    hashes, paths = {}, set()
    for kind, cases in (("baseline", fresh_baseline), ("stress", fresh_stress)):
        for case in cases:
            case_id = case["case_id"]
            adapter = {key: case[key] for key in ADAPTER_KEYS} if kind == "baseline" else case["adapter_inputs"]
            expected = expected_arrays(adapter)
            archive_name = kind+"_"+case_id
            path = run/"inputs"/(archive_name+".npz")
            paths.add(str(path))
            with np.load(path, allow_pickle=False) as stored:
                require(set(stored.files) == set(ARRAY_KEYS), "Saved adapter inputs contain extra/missing arrays")
                for key in ARRAY_KEYS:
                    actual = stored[key]
                    require(actual.shape == expected[key].shape and actual.dtype == expected[key].dtype and
                            np.array_equal(actual, expected[key], equal_nan=True), "Saved/generator input mismatch: "+archive_name+":"+key)
                    hashes[archive_name+":"+key] = array_fingerprint(actual)
                if kind == "baseline":
                    with np.load(V45/"inputs"/("causal_"+case_id+".npz"), allow_pickle=False) as old:
                        require(set(old.files) == set(stored.files), "V45 baseline archive fields changed")
                        for key in ARRAY_KEYS:
                            require(array_fingerprint(stored[key]) == array_fingerprint(old[key]), "Baseline input bytes changed")
            item = dict(case_id=case_id, polarity=adapter["polarity"])
            if kind == "baseline":
                item["forecast_origin"] = "unchanged_legacy_caller_supplied_assumption"
            else:
                item.update({key: case[key] for key in ("family", "generator_truth", "provenance")})
            expected_metadata[kind].append(item)
    require(set(saved["inputs_sha256"]) == paths == {str(path.resolve()) for path in (run/"inputs").iterdir()},
            "Input archive count or membership changed")
    require(hashes == freeze["input_memory_sha256"], "Saved arrays differ from pre-score fingerprints")
    require(metadata == expected_metadata and freeze["input_metadata_sha256"] == json_hash(expected_metadata),
            "Frozen input metadata/truth changed")
    for record, fresh in zip(stress, fresh_stress):
        require(set(record) == {"case_id", "family", "generator_truth", "provenance", "arms"}, "Stress result schema changed")
        for key in ("case_id", "family", "generator_truth", "provenance"):
            require(record[key] == fresh[key], "Saved result truth/provenance changed")
    baseline_details, stress_details = audit_evidence(baseline), audit_evidence(stress)
    by_id = {record["case_id"]: record for record in stress}
    first = by_id[CURRENT_ONLY_IDS[0]]
    for case_id in CURRENT_ONLY_IDS:
        for arm in ARMS:
            reference = first["arms"][arm]["raw_adapter_result"]
            actual = by_id[case_id]["arms"][arm]["raw_adapter_result"]
            for field in NOMINAL_FIELDS:
                require(actual[field] == reference[field], "Current-only variation changed prior nominal field: "+field)
            require(actual["component_bounds"] == reference["component_bounds"], "Current-only pixels changed learned-template bounds")
    require(by_id[EXPECTED_IDS[-2]]["arms"] == by_id[EXPECTED_IDS[-1]]["arms"] == first["arms"],
            "Identical observational inputs produce different evidence")
    baseline_counts = {arm: evidence_counts(baseline, arm) for arm in ARMS}
    stress_counts = {arm: evidence_counts(stress, arm) for arm in ARMS}
    groups = {}
    for record in stress:
        groups.setdefault(record["family"], []).append(record)
    group_counts = {group: {arm: evidence_counts(records, arm) for arm in ARMS} for group, records in groups.items()}
    require(summary["baseline_counts"] == baseline_counts and summary["stress_counts"] == stress_counts and
            summary["stress_family_counts"] == group_counts, "Reported group counts disagree")
    require((summary["baseline_cases"], summary["stress_cases"], summary["arms"]) == (6, 28, 4), "Summary denominator changed")
    require(all(value is True for value in summary["invariants"].values()) and
            summary["geometry_shape_association_mismatch_not_covered_by_declared_dn_bound"] is True and
            summary["stress_numerical_signs_are_not_recall_or_physical_class"] is True and
            summary["production_changed"] is False and summary["real_packet_data_read"] is False,
            "Result scope or invariant assertions changed")
    verify_bindings(receipt["files_sha256"], run)
    return dict(schema="seaqr.accuracy-v46-independent-synthetic-audit.v1", completed=True, passed=True, issues=[],
        created_at_utc=datetime.now(timezone.utc).isoformat(), stress_completion_receipt_sha256=sha(run/"completion_receipt.json"),
        v45_completion_receipt_sha256=V45_RECEIPT, checked_files_sha256=receipt["files_sha256"],
        counts=dict(completion_bound_files=len(receipt["files_sha256"]), baseline_cases=6, stress_cases=28,
                    input_archives=34, input_arrays=len(hashes), arms=4, retained_case_arm_entries=136),
        geometry_audit=geometry, baseline_counts=baseline_counts, stress_counts=stress_counts,
        stress_family_counts=group_counts, baseline_details=baseline_details, stress_details=stress_details,
        v45_baseline_inputs_and_results_exact=True, frozen_nominal_design_support_and_ambiguity_preserved=True,
        current_only_design_support_and_bounds_identical=True, twins_and_ordinary_numerical_outputs_identical=True,
        all_motion_and_physical_class_unknown=True, original_error_contract_unchanged=True,
        synthetic_only=True, real_data_accessed=False, production_changed=False,
        producer_chronology=[time.isoformat() for time in chronology],
        limitations=[
            "UTC ordering is producer-declared chronology with hash binding, not external clock attestation",
            "This audit does not independently recompute every numerical fit; the V45 mathematical dependencies remain hash-pinned",
            "All 28 stress images are unquantized floating native-DN synthetics, not camera recordings",
            "Geometry, shape, temporal association and physical identity are outside fixed-geometry +/-0.5 DN interval coverage",
            "Positive source evidence is not motion/airborne classification; unavailable and zero-containing evidence are not negative detections",
            "Complete accounting permits diagnostic interpretation, not a required stress recall fraction or production promotion",
        ])


def write_new_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run, output = args.run.resolve(), args.output.resolve()
    base = ROOT/"results/tiny_target/accuracy_v46_20260925"
    require(run.parent == base and output.parent == base, "Audit output must be outside the frozen run in V46 root")
    require(not output.exists(), "Never overwrite a valid audit")
    receipt_path = run/"completion_receipt.json"
    require(load(receipt_path)["completed"] is True, "Wait for a completed run before freezing audit inputs")
    own_bindings = {str(path): sha(path) for path in (Path(__file__).resolve(),
        ROOT/"tests/unit/test_accuracy_v46_audit.py", receipt_path)}
    audit_freeze = output.with_name(output.stem+"_freeze.json")
    write_new_json(audit_freeze, dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
                                    files_sha256=own_bindings, scoring_performed=False))
    result = audit(run)
    require(all(sha(path) == digest for path, digest in own_bindings.items()), "Audit code/test/receipt changed during audit")
    result["audit_freeze_sha256"] = sha(audit_freeze)
    result["audit_files_sha256"] = own_bindings
    write_new_json(output, result)
    print(json.dumps({key: result[key] for key in ("passed", "issues", "counts", "stress_completion_receipt_sha256")}, indent=2))


if __name__ == "__main__":
    main()
