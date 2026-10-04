"""Independent scalar audit of one allowlisted generated-only V51 run.

No experiment, benchmark, or diagnostic implementation is imported. This
program may read only the 98 declared generated cases, compact run receipts,
and explicitly allowlisted V51 source files for hash verification. It does not
open media, prior image caches, real residual artifacts, or other datasets.
"""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "results/tiny_target/accuracy_v51_20260926"
FRAMES = tuple(range(8, 64))
ARMS = ("median8", "median3")
FAMILIES = ("step", "ramp", "pulse1", "pulse2", "pulse4", "pulse8", "pulse12",
            "localized_step", "localized_pulse2", "moving_stripe",
            "registration_shift", "missing_window")
VECTORS = ("slow8", "fast3", "early5", "scale", "fast_slow_delta",
           "fast_slow_delta_normalized", "early_recent_delta",
           "early_recent_delta_normalized", "departure_envelope",
           "departure_envelope_normalized", "return_fraction",
           "recent_center_suffix_length")
MATRICES = ("recent_departures", "recent_departures_normalized",
            "recent_center_margins", "recent_center_margins_normalized")
MASKS = ("point_available", "return_fraction_defined", "recent_center_defined")
INPUT_FIELDS = ("values", "event_active", "response_phase", "history_has_prior_event",
                "signal_delta", "affected_points_mask", "missing_mask", "points_xy")
POINTS = tuple((x, y) for y in range(8, 121, 8) for x in range(8, 121, 8)
               if 40 <= max(abs(x - 64), abs(y - 64)) <= 56)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def plain(value):
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def fingerprint(diagnostic):
    payload = {key: value for key, value in diagnostic.items()
               if key != "diagnostic_sha256"}
    return hashlib.sha256(json.dumps(plain(payload), sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def compare(expected, actual, context="root"):
    """Exact structure/counts; tight rounding tolerance only for reductions."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and expected.keys() == actual.keys(), context + ": keys differ")
        for key in expected:
            compare(expected[key], actual[key], context + "." + str(key))
    elif isinstance(expected, (list, tuple)):
        require(isinstance(actual, (list, tuple)) and len(expected) == len(actual), context + ": sequence differs")
        for index, (first, second) in enumerate(zip(expected, actual)):
            compare(first, second, f"{context}[{index}]")
    elif isinstance(expected, float):
        require(type(actual) in (int, float) and math.isfinite(actual)
                and math.isclose(expected, actual, rel_tol=1e-12, abs_tol=1e-10), context + ": numeric reduction differs")
    else:
        require(type(expected) is type(actual) and expected == actual, context + ": value/type differs")


def median(values):
    ordered = sorted(float(value) for value in values)
    count = len(ordered)
    require(count > 0, "Empty order statistic")
    middle = count // 2
    return ordered[middle] if count % 2 else (ordered[middle - 1] + ordered[middle]) / 2.0


def diagnostic_metadata():
    return dict(prior_count=8, prior_order="oldest_to_newest",
        samples_are_fixed_points_supplied_by_caller=True,
        per_point_support_selected_or_dropped=False,
        current_argument_accepted=False, labels_argument_accepted=False,
        geometry_argument_accepted=False, geometry_causality_certified=False,
        full_camera_online_causality_certified=False,
        forecast_values_selected_or_modified=False,
        scale_formula="max(1 DN, median(abs(history - median(history))))",
        scale_floor_dn=1.0, scale_is_descriptor_not_physical_noise_bound=True,
        normalization_divides_dn_descriptors_by_v50_scale=True,
        early_and_recent_windows_disjoint=True,
        recent_center_suffix_uses_latest_three_strict_positive_margins=True,
        return_fraction_is_observed_magnitude_return_not_future_probability=True,
        equal_prior_prefix_cannot_identify_different_future_responses=True,
        support_purity_or_guard_to_core_transfer_certified=False,
        classification_threshold_or_model_selection_performed=False,
        intervals_or_coverage_guarantees_produced=False,
        source_object_or_production_decision=False,
        unavailable_indices_preserved_without_imputation=True,
        nonfinite_json_representation="null")


def scalar_diagnostic(history):
    array = np.asarray(history)
    require(array.ndim == 2 and array.shape[0] == 8 and array.dtype.kind in "iuf",
            "Expected eight real prior vectors")
    count = array.shape[1]
    fields = {name: np.full(count, np.nan, dtype=np.float64) for name in VECTORS}
    fields.update({name: np.full((3, count), np.nan, dtype=np.float64) for name in MATRICES})
    fields.update({name: np.zeros(count, dtype=bool) for name in MASKS})
    reasons = []
    for index in range(count):
        values = [float(array[t, index]) for t in range(8)]
        if not all(math.isfinite(value) for value in values):
            reasons.append("nonfinite_input_after_float64_conversion")
            continue
        slow, fast, early = median(values), median(values[-3:]), median(values[:5])
        scale = max(1.0, median([abs(value - slow) for value in values]))
        departures = [value - early for value in values[-3:]]
        envelope = max(abs(value) for value in departures)
        margins = [abs(value - early) - abs(value - fast) for value in values[-3:]]
        numbers = dict(slow8=slow, fast3=fast, early5=early, scale=scale,
            fast_slow_delta=fast - slow, fast_slow_delta_normalized=(fast - slow) / scale,
            early_recent_delta=fast - early, early_recent_delta_normalized=(fast - early) / scale,
            departure_envelope=envelope, departure_envelope_normalized=envelope / scale)
        triples = dict(recent_departures=departures,
            recent_departures_normalized=[value / scale for value in departures],
            recent_center_margins=margins,
            recent_center_margins_normalized=[value / scale for value in margins])
        valid = all(math.isfinite(value) for value in numbers.values()) and all(
            math.isfinite(value) for triple in triples.values() for value in triple)
        returned = 1.0 - abs(departures[-1]) / envelope if valid and envelope > 0 else math.nan
        valid = valid and (not envelope > 0 or math.isfinite(returned))
        if not valid:
            reasons.append("nonfinite_descriptor_arithmetic")
            continue
        reasons.append(None)
        fields["point_available"][index] = True
        fields["return_fraction_defined"][index] = envelope > 0
        fields["recent_center_defined"][index] = fast != early
        for name, value in numbers.items():
            fields[name][index] = value
        for name, values3 in triples.items():
            fields[name][:, index] = values3
        fields["return_fraction"][index] = returned
        if fast != early:
            suffix = 0
            for margin in reversed(margins):
                if margin <= 0:
                    break
                suffix += 1
            fields["recent_center_suffix_length"][index] = float(suffix)
    result = dict(schema_version=1, total_count=count,
        available_count=int(fields["point_available"].sum()),
        point_unavailable_reasons=tuple(reasons), metadata=diagnostic_metadata(), **fields)
    result["diagnostic_sha256"] = fingerprint(result)
    return result


def case_id(family, amplitude, level, seed):
    if family == "stable":
        return f"stable_noise{level}_seed{seed}"
    sign = "neg" if amplitude < 0 else "pos"
    return f"{family}_a_{sign}{abs(amplitude)}_noise{level}_seed{seed}"


def specifications():
    design = [("stable", 0, level, seed) for level, seed in ((0, 71), (1, 991))]
    design += [(family, amplitude, level, seed) for family in FAMILIES
               for amplitude in (-32, -8, 8, 32) for level, seed in ((0, 71), (1, 991))]
    result = []
    for family, amplitude, level, seed in design:
        stop = None if family == "stable" else (
            20 + int(family[5:]) if family.startswith("pulse") else (
            22 if family == "localized_pulse2" else (
            44 if family in ("moving_stripe", "registration_shift") else 64)))
        result.append(dict(schema="accuracy_v51_generated_benchmark_v1",
            case_id=case_id(family, amplitude, level, seed), family=family,
            amplitude=amplitude, noise_level=level, seed=seed, frame_count=64,
            point_count=144, history_length=8, response_frame_indices=list(FRAMES),
            event_window=dict(start_inclusive=None if family == "stable" else 20,
                              stop_exclusive=stop),
            missing_window=dict(start_inclusive=18, stop_exclusive=21,
                                point_indices=list(range(8))) if family == "missing_window" else None,
            analytic_simulation_only=True, physical_sensor_model=False))
    return result


def generated_input(spec):
    """Independent analytic reconstruction; labels never enter scalar_diagnostic."""
    require(spec in specifications(), "Unknown generated specification")
    family, amplitude = spec["family"], spec["amplitude"]
    points = np.asarray(POINTS, dtype=np.int64)
    x, y = points.T
    base = 96.0 + 0.05 * (x - 64.0) + 0.03 * (y - 64.0)
    delta = np.zeros((64, 144), dtype=np.float64)
    mask = np.zeros((64, 144), dtype=bool)
    active = np.zeros(64, dtype=bool)
    phase = np.full(64, "baseline", dtype="<U10")
    if family != "stable":
        stop = spec["event_window"]["stop_exclusive"]
        for t in range(20, stop):
            active[t] = True
            phase[t] = "onset" if t == 20 else "event"
            selected = np.ones(144, dtype=bool)
            if family.startswith("localized_"):
                selected = (x >= 64) & (y >= 64)
            elif family == "moving_stripe":
                selected = abs(x - (8 + ((t - 20) % 15) * 8)) <= 8
            mask[t] = selected
            if family == "registration_shift":
                shift = (0, 1, -1, 2, -2, 1, 0, -1)[(t - 20) % 8]
                delta[t] = amplitude * (np.sin((x - 64 + shift) / 8.0) - np.sin((x - 64) / 8.0))
            else:
                amount = amplitude * min((t - 20) / 16.0, 1.0) if family == "ramp" else amplitude
                delta[t, selected] = amount
        phase[stop:] = "post_event"
    noise = np.random.default_rng(spec["seed"]).uniform(-spec["noise_level"], spec["noise_level"], (64, 144))
    values = base[None, :] + delta + noise
    missing = np.zeros((64, 144), dtype=bool)
    if family == "missing_window":
        missing[18:21, :8] = True
        values[missing] = np.nan
    history_event = np.asarray([any(active[max(0, t - 8):t]) for t in range(64)], dtype=bool)
    return dict(values=values, event_active=active, response_phase=phase,
        history_has_prior_event=history_event, signal_delta=delta,
        affected_points_mask=mask, missing_mask=missing, points_xy=points)


def diagnostic_summary(d):
    available = [i for i, good in enumerate(d["point_available"]) if good]
    centers = [i for i, good in enumerate(d["recent_center_defined"]) if good]
    returns = [i for i, good in enumerate(d["return_fraction_defined"]) if good]
    mean = lambda values: math.fsum(values) / len(values) if values else None
    return dict(prior_available_points=len(available), recent_center_defined_points=len(centers),
        recent_center_suffix_counts={str(k): sum(d["recent_center_suffix_length"][i] == k for i in centers)
                                     for k in range(4)},
        mean_absolute_fast_slow_delta_dn=mean([abs(float(d["fast_slow_delta"][i])) for i in available]),
        mean_absolute_early_recent_delta_dn=mean([abs(float(d["early_recent_delta"][i])) for i in available]),
        mean_absolute_recent_center_margin_dn=mean([abs(float(d["recent_center_margins"][j, i]))
                                                    for j in range(3) for i in centers]),
        observed_return_defined_points=len(returns),
        observed_return_fraction_sum=math.fsum(float(d["return_fraction"][i]) for i in returns),
        observed_return_fraction_mean=mean([float(d["return_fraction"][i]) for i in returns]),
        no_regime_or_object_label=True)


def measurement(current, d):
    current = np.asarray(current)
    require(current.ndim == 1 and current.shape == d["point_available"].shape
            and current.dtype.kind in "iuf", "Unexpected current schema")
    valid = [i for i, value in enumerate(current)
             if d["point_available"][i] and math.isfinite(float(value))]
    count = len(current)
    arms = {}
    for arm, field in (("median8", "slow8"), ("median3", "fast3")):
        errors = [float(current[i]) - float(d[field][i]) for i in valid]
        require(all(math.isfinite(value) for value in errors), "Nonfinite independent response errors")
        absolute = math.fsum(abs(value) for value in errors)
        squared = math.fsum(value * value for value in errors)
        require(math.isfinite(absolute) and math.isfinite(squared), "Nonfinite independent error reduction")
        arms[arm] = dict(point_count=len(valid), absolute_error_sum_dn=absolute,
            squared_error_sum_dn2=squared,
            mae_dn=absolute / len(valid) if valid else None,
            max_absolute_error_dn=max(abs(value) for value in errors) if valid else None)
    return dict(total_points=count,
        prior_unavailable_points=sum(not bool(good) for good in d["point_available"]),
        current_nonfinite_points=sum(not math.isfinite(float(value)) for value in current),
        scorable_points=len(valid), unscorable_points=count - len(valid),
        complete_window=bool(count and len(valid) == count), arms=arms)


def aggregate(rows):
    total = sum(row["measurement"]["total_points"] for row in rows)
    scored = sum(row["measurement"]["scorable_points"] for row in rows)
    complete = [row for row in rows if row["measurement"]["complete_window"]]
    arms = {}
    for arm in ARMS:
        absolute = math.fsum(row["measurement"]["arms"][arm]["absolute_error_sum_dn"] for row in rows)
        squared = math.fsum(row["measurement"]["arms"][arm]["squared_error_sum_dn2"] for row in rows)
        maxima = [row["measurement"]["arms"][arm]["max_absolute_error_dn"] for row in rows
                  if row["measurement"]["arms"][arm]["point_count"]]
        arms[arm] = dict(conditional_point_mae_dn=absolute / scored if scored else None,
            conditional_point_rmse_dn=math.sqrt(squared / scored) if scored else None,
            max_absolute_error_dn=max(maxima) if maxima else None,
            mean_complete_window_mae_dn=math.fsum(row["measurement"]["arms"][arm]["mae_dn"]
                for row in complete) / len(complete) if complete else None)
    returned = sum(row["diagnostic"]["observed_return_defined_points"] for row in rows)
    return dict(response_windows=len(rows), complete_windows=len(complete),
        total_point_opportunities=total, scorable_points=scored, unscorable_points=total - scored,
        prior_unavailable_points=sum(row["measurement"]["prior_unavailable_points"] for row in rows),
        current_nonfinite_points=sum(row["measurement"]["current_nonfinite_points"] for row in rows),
        missing_counts_overlap=True, arms=arms,
        recent_center_defined_points=sum(row["diagnostic"]["recent_center_defined_points"] for row in rows),
        recent_center_suffix_counts={str(k): sum(row["diagnostic"]["recent_center_suffix_counts"][str(k)]
                                                  for row in rows) for k in range(4)},
        observed_return_defined_points=returned,
        observed_return_fraction_mean=math.fsum(row["diagnostic"]["observed_return_fraction_sum"]
            for row in rows) / returned if returned else None,
        frames_points_and_cases_not_independent=True, descriptors_are_not_regime_classifications=True)


def canonical_file(path):
    path = Path(path)
    require(path.is_absolute() and path.resolve() == path and path.is_file(), "Noncanonical regular file: " + str(path))
    return path


def hash_file(path):
    return hashlib.sha256(canonical_file(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(canonical_file(path).read_text())


def allow_source(path):
    path = Path(path)
    allowed = path == ROOT / "docs/accuracy_v51_plan.md" or (
        path.parent == ROOT / "scripts" and re.fullmatch(r"[a-z0-9_]*accuracy_v51[a-z0-9_]*\.py", path.name)
    ) or (path.parent == ROOT / "tests/unit" and re.fullmatch(r"test_accuracy_v51[a-z0-9_]*\.py", path.name))
    require(bool(allowed), "Source outside literal V51 allowlist: " + str(path))
    return canonical_file(path)


def allow_case_file(run, path, filename):
    expected = Path(run) / filename
    require(Path(path) == expected, "Generated case path escaped its literal allowlist")
    return canonical_file(expected)


def read_arrays(path):
    with np.load(canonical_file(path), allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def exact_array(expected, actual, context):
    require(isinstance(actual, np.ndarray) and expected.dtype == actual.dtype
            and expected.shape == actual.shape, context + ": array dtype/shape differs")
    equal = np.array_equal(expected, actual, equal_nan=True) if expected.dtype.kind in "fci" else np.array_equal(expected, actual)
    require(equal, context + ": array values differ")


def chronology(freeze, manifest, summary, receipt):
    times = [datetime.fromisoformat(item["created_at_utc"]) for item in (freeze, manifest, summary, receipt)]
    require(all(value.tzinfo is not None for value in times), "Stage timestamps lack timezone")
    require(all(first <= second for first, second in zip(times, times[1:])), "Stage chronology is reversed")
    require(freeze["before_generation_and_scoring"] is True and manifest["response_scoring_started"] is False,
            "Freeze/scoring stage assertions differ")
    require(manifest["completed"] is True and summary["completed"] is True and receipt["completed"] is True,
            "Incomplete stage")


def expected_twins():
    return [dict(step_case_id=case_id("step", amplitude, level, seed),
        pulse_case_id=case_id(f"pulse{duration}", amplitude, level, seed), duration=duration,
        amplitude=amplitude, noise_level=level, seed=seed,
        history_start_inclusive=20 + duration - 8, history_stop_exclusive=20 + duration,
        response_frame=20 + duration, identical_prior_values=True, different_response_values=True,
        response_step_minus_pulse_analytic=amplitude)
        for duration in (1, 2, 4, 8, 12) for amplitude in (-32, -8, 8, 32)
        for level, seed in ((0, 71), (1, 991))]


def audit_run(run):
    run = Path(run).absolute()
    require(run.parent == OUTPUT_ROOT and run.resolve() == run and run.is_dir(), "Expected a canonical immediate V51 output child")
    compact_names = ("freeze.json", "forecasts_frozen.json", "responses.jsonl",
                     "identical_prefix_pairs.json", "summary.json")
    specs = specifications()
    data_names = {spec["case_id"] + suffix for spec in specs
                  for suffix in ("_input.npz", "_diagnostics.npz", "_context.json")}
    allowed_output = {str(run / name) for name in (*compact_names, *sorted(data_names))}
    freeze = read_json(run / "freeze.json")
    manifest = read_json(run / "forecasts_frozen.json")
    summary = read_json(run / "summary.json")
    receipt = read_json(run / "completion_receipt.json")
    compare(specs, freeze["specifications"], "freeze.specifications")
    compare(specs, freeze["benchmark"]["specifications"], "benchmark.specifications")
    compare(expected_twins(), freeze["benchmark"]["twin_pairs"], "benchmark.twins")
    compare([list(p) for p in POINTS], freeze["benchmark"]["points_xy"], "benchmark.points")
    for field, expected in (("case_count", 98), ("frame_count", 64), ("point_count", 144),
                             ("history_length", 8), ("clipped", False), ("quantized", False),
                             ("physical_sensor_model", False), ("analytic_simulation_only", True)):
        compare(expected, freeze["benchmark"][field], "benchmark." + field)
    chronology(freeze, manifest, summary, receipt)
    sources = freeze["source_files_sha256"]
    require(isinstance(sources, dict) and sources, "Missing source bindings")
    required = {str(ROOT / "scripts" / name) for name in (
        "accuracy_v51_benchmark.py", "accuracy_v51_history_diagnostic.py",
        "run_accuracy_v51_history.py", "audit_accuracy_v51_history.py")}
    require(required <= sources.keys(), "Core V51 sources not frozen")
    for path, digest in sources.items():
        require(hash_file(allow_source(path)) == digest, "Frozen source hash differs: " + path)
    require(set(receipt["files_sha256"]) == set(sources) | allowed_output, "Receipt file allowlist differs")
    require(set(manifest["files_sha256"]) == {str(run / name) for name in data_names}, "Forecast file allowlist differs")
    compare(sources, {path: receipt["files_sha256"][path] for path in sources}, "receipt.sources")
    for path in sorted(allowed_output):
        require(hash_file(path) == receipt["files_sha256"][path], "Run output hash differs: " + path)
        if path in manifest["files_sha256"]:
            require(receipt["files_sha256"][path] == manifest["files_sha256"][path], "Forecast/receipt binding differs")
    require(manifest["case_count"] == 98 and manifest["response_window_count"] == 5488,
            "Manifest denominators differ")
    require(manifest["adaptive_forecast_selection"] is False, "Adaptive model selection claimed")
    compare(specs, [entry["spec"] for entry in manifest["cases"]], "manifest.case_membership")
    observed_rows = [json.loads(line) for line in canonical_file(run / "responses.jsonl").read_text().splitlines()]
    require(len(observed_rows) == 5488, "Response row count differs")
    expected_rows = []
    twin_windows = {}
    wanted = {(pair[key], pair["response_frame"]) for pair in expected_twins()
              for key in ("step_case_id", "pulse_case_id")}
    array_names = set(VECTORS + MATRICES + MASKS)
    for entry in manifest["cases"]:
        spec = entry["spec"]
        name = spec["case_id"]
        input_path = allow_case_file(run, entry["input_path"], name + "_input.npz")
        diagnostic_path = allow_case_file(run, entry["diagnostic_path"], name + "_diagnostics.npz")
        context_path = allow_case_file(run, entry["context_path"], name + "_context.json")
        case = read_arrays(input_path)
        require(set(case) == set(INPUT_FIELDS), "Input array membership differs")
        expected_case = generated_input(spec)
        for key in INPUT_FIELDS:
            exact_array(expected_case[key], case[key], name + ".input." + key)
        arrays, contexts = read_arrays(diagnostic_path), read_json(context_path)
        require(set(arrays) == array_names and entry["array_keys"] == sorted(array_names), "Diagnostic array membership differs")
        require([item["frame_index"] for item in contexts] == list(FRAMES), "Context frame membership differs")
        for key, array in arrays.items():
            expected_shape = (56, 3, 144) if key in MATRICES else (56, 144)
            expected_dtype = np.dtype(bool if key in MASKS else np.float64)
            require(array.shape == expected_shape and array.dtype == expected_dtype, "Stored diagnostic schema differs")
        for index, frame in enumerate(FRAMES):
            reference = scalar_diagnostic(case["values"][frame - 8:frame])
            saved = dict(contexts[index]["diagnostic"])
            require(not (array_names & saved.keys()), "Context duplicates diagnostic arrays")
            saved.update({key: value[index] for key, value in arrays.items()})
            require(saved.keys() == reference.keys(), "Diagnostic field membership differs")
            for key in array_names:
                exact_array(reference[key], saved[key], f"{name}.frame{frame}.{key}")
            for key in reference.keys() - array_names - {"diagnostic_sha256"}:
                compare(plain(reference[key]), saved[key], f"{name}.frame{frame}.{key}")
            require(fingerprint(saved) == saved["diagnostic_sha256"] == reference["diagnostic_sha256"],
                    "Scalar or saved diagnostic fingerprint differs")
            row = dict(spec, frame_index=frame, scheduled_phase=str(case["response_phase"][frame]),
                history_has_prior_event=bool(case["history_has_prior_event"][frame]),
                measurement=measurement(case["values"][frame], reference), diagnostic=diagnostic_summary(reference))
            compare(plain(row), observed_rows[len(expected_rows)], f"response{len(expected_rows)}")
            expected_rows.append(row)
            if (name, frame) in wanted:
                twin_windows[name, frame] = (case["values"][frame - 8:frame].copy(),
                                             case["values"][frame].copy(), reference)
        print(f"V51 independent audit: {len(expected_rows) // 56}/98 generated cases", flush=True)
    pairs_file = read_json(run / "identical_prefix_pairs.json")
    require(pairs_file["all_pairs_passed"] is True and len(pairs_file["pairs"]) == 40, "Twin denominator differs")
    for pair, stored in zip(expected_twins(), pairs_file["pairs"]):
        frame = pair["response_frame"]
        first_h, first_y, first_d = twin_windows[pair["step_case_id"], frame]
        second_h, second_y, second_d = twin_windows[pair["pulse_case_id"], frame]
        require(np.array_equal(first_h, second_h) and first_d["diagnostic_sha256"] == second_d["diagnostic_sha256"], "Twin priors/descriptors differ")
        separation = abs(first_y - second_y)
        require(np.allclose(first_y - second_y, pair["amplitude"], rtol=0, atol=6e-14), "Twin response analytic difference fails")
        arms = {}
        for arm, key in (("median8", "slow8"), ("median3", "fast3")):
            require(np.array_equal(first_d[key], second_d[key]), "Twin forecasts differ")
            worst = [max(abs(float(pred) - float(a)), abs(float(pred) - float(b)))
                     for pred, a, b in zip(first_d[key], first_y, second_y)]
            require(all(value + 1e-12 >= delta / 2 for value, delta in zip(worst, separation)), "Twin minimax inequality fails")
            arms[arm] = dict(mean_worst_world_absolute_error_dn=math.fsum(worst) / 144,
                             max_worst_world_absolute_error_dn=max(worst))
        limits = dict(min=float(separation.min()), max=float(separation.max()))
        expected = dict(pair, identical_priors=True, identical_diagnostics=True, identical_forecasts=True,
            point_count=144, possible_response_separation_dn=limits,
            necessary_worst_world_error_dn={key: value / 2 for key, value in limits.items()},
            minimum_interval_width_to_cover_both_dn=limits, arms=arms,
            shared_history_diagnostics=diagnostic_summary(first_d), numerical_check_tolerance_dn=1e-12,
            analytic_response_difference_check_tolerance_dn=6e-14,
            not_a_physical_bound_or_predicted_class=True)
        compare(plain(expected), stored, "twin." + pair["pulse_case_id"])
    aggregates = dict(overall=aggregate(expected_rows))
    for label, key in (("by_case", "case_id"), ("by_family", "family"), ("by_noise_level", "noise_level")):
        groups = defaultdict(list)
        for row in expected_rows:
            groups[str(row[key])].append(row)
        aggregates[label] = {group: aggregate(rows) for group, rows in groups.items()}
    family_phase = defaultdict(lambda: defaultdict(list))
    for row in expected_rows:
        family_phase[row["family"]][row["scheduled_phase"]].append(row)
    aggregates["by_family_phase"] = {family: {phase: aggregate(rows) for phase, rows in phases.items()}
                                     for family, phases in family_phase.items()}
    for key, value in aggregates.items():
        compare(plain(value), summary[key], "summary." + key)
    overall = aggregates["overall"]
    require(overall["response_windows"] == 5488 and overall["total_point_opportunities"] == 790272,
            "Independent denominator differs")
    for field, expected in (("twin_pair_count", 40), ("all_identical_prefix_checks_passed", True),
        ("no_new_predictor_or_model_selection", True), ("no_production_or_object_decision", True),
        ("generated_only_not_real_camera_or_airborne_validation", True)):
        compare(expected, summary[field], "summary." + field)
    for artifact in (freeze, receipt):
        require(artifact["production_changed"] is False, "Production change claimed")
    require(receipt["real_data_accessed"] is False, "Real data claimed")
    # Recheck all bindings after reconstruction to catch concurrent mutations.
    for path, digest in receipt["files_sha256"].items():
        require(hash_file(path) == digest, "Bound file changed during audit: " + path)
    return dict(completed=True, created_at_utc=datetime.now(timezone.utc).isoformat(),
        audited_run=str(run), completion_receipt_sha256=hash_file(run / "completion_receipt.json"),
        audit_source_sha256=hash_file(Path(__file__).resolve()),
        case_count=98, response_windows_checked=5488, point_opportunities_checked=790272,
        identical_prefix_pairs_checked=40, prior_descriptor_array_fields_checked=len(array_names),
        source_bindings_checked=len(sources), generated_file_bindings_checked=len(manifest["files_sha256"]),
        stage_chronology_verified=True, independent_scalar_order_statistics=True,
        generated_inputs_reconstructed=True, all_descriptor_arrays_masks_reasons_fingerprints_checked=True,
        all_response_errors_and_aggregates_checked=True, all_checks_passed=True,
        runtime_imports_exclude_experiment_benchmark_and_diagnostic=True,
        production_changed=False, real_data_accessed=False,
        physical_sensor_or_airborne_accuracy_claim=False,
        independent_overall=plain(overall), independent_by_family=plain(aggregates["by_family"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.absolute()
    require(output.parent == OUTPUT_ROOT and output.resolve() == output and output.suffix == ".json"
            and not output.exists(), "Audit output must be a fresh sibling JSON")
    result = audit_run(args.run)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: result[key] for key in ("completed", "all_checks_passed",
          "case_count", "response_windows_checked", "point_opportunities_checked")}), flush=True)


if __name__ == "__main__":
    main()
