"""Reconstruct two bounded saved measurement histories without media or tuning.

Independent QR fits use accepted reference-coordinate detector peaks, never
filtered track states, predictions, reference annotations, or future frames.
Leave-one-out values explain leverage; they are not a sample-removal policy.
Only a fresh diagnostic JSON is written. No production module is imported.
"""

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
JOURNAL = ROOT / "results/tiny_target/visible_validation_v34_20260923/audit_20260924/evidence/run/full_repeat0_0029/frames.jsonl"
JOURNAL_SHA256 = "e97064888d5901c98ced2be81bf7982f6f21fdd571e6a5197e59aaee4b4fbcf8"
OUTPUT_DIRECTORY = ROOT / "results/tiny_target/accuracy_v39_20260925"
IDENTITIES = ("bright:80", "bright:138")
LAST_FRAME = 24
WINDOW_HITS, MINIMUM_HITS, MAXIMUM_RMSE = 8, 5, 3.0
INPUTS = (
    JOURNAL,
    ROOT / "docs/phase20_airborne_accuracy_protocol.md",
    ROOT / "docs/accuracy_v39_next_steps.md",
    ROOT / "tiny_target/visible_quality.py",
    ROOT / "tiny_target/visible_baseline.py",
    ROOT / "tiny_target/tracking/kalman.py",
    Path(__file__).resolve(),
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fit_qr(history):
    """Independent unscaled-time QR solution of the two quadratic coordinates."""
    require(len(history) >= 3, "Three distinct timestamps required for a quadratic fit")
    end = history[-1]["timestamp_ns"]
    time = np.asarray([(m["timestamp_ns"]-end)/1e9 for m in history], dtype=float)
    values = np.asarray([m["raw_reference_xy"] for m in history], dtype=float)
    design = np.column_stack((np.ones_like(time), time, time*time))
    require(np.linalg.matrix_rank(design) == 3, "Rank-deficient time design")
    q, r = np.linalg.qr(design, mode="reduced")
    coefficient = np.linalg.solve(r, q.T@values)
    fitted = design@coefficient
    error = values-fitted
    per_sample_sse = np.sum(error*error, axis=1)
    sse = float(per_sample_sse.sum())
    return dict(time_origin_ns=end, coefficient=coefficient, fitted=fitted, error=error,
                per_sample_sse=per_sample_sse, sse=sse,
                rmse=float(math.sqrt(sse/len(history))),
                leverage=np.sum(q*q, axis=1), design_condition=float(np.linalg.cond(design)))


def describe_fit(history):
    fitted = fit_qr(history)
    rows = []
    for index, measurement in enumerate(history):
        omitted = history[:index]+history[index+1:]
        reduced = fit_qr(omitted) if len(omitted) >= 3 else None
        h = float(fitted["leverage"][index])
        rows.append(dict(frame=measurement["frame"], raw_reference_xy=measurement["raw_reference_xy"],
            fitted_reference_xy=fitted["fitted"][index].tolist(),
            observed_minus_fitted_xy=fitted["error"][index].tolist(),
            residual_norm_px=float(np.linalg.norm(fitted["error"][index])),
            residual_squared_px=float(fitted["per_sample_sse"][index]),
            fraction_of_total_squared_error=float(fitted["per_sample_sse"][index]/fitted["sse"]) if fitted["sse"] else 0.,
            leverage=h,
            leave_one_out_prediction_error_xy=(fitted["error"][index]/(1-h)).tolist() if h < 1-1e-12 else None,
            leave_one_out_refit_rmse_px=reduced["rmse"] if reduced else None,
            leave_one_out_remaining_history_count=len(omitted),
            leave_one_out_remaining_history_ready=len(omitted) >= MINIMUM_HITS))
    axis_sse = np.sum(fitted["error"]**2, axis=0)
    return dict(history_frames=[m["frame"] for m in history], sample_count=len(history),
        first_timestamp_ns=history[0]["timestamp_ns"], last_timestamp_ns=history[-1]["timestamp_ns"],
        time_span_seconds=(history[-1]["timestamp_ns"]-history[0]["timestamp_ns"])/1e9,
        qr_design_condition=fitted["design_condition"],
        quadratic_coefficients_xy_by_constant_seconds_seconds_squared=fitted["coefficient"].tolist(),
        sse_px_squared=fitted["sse"], rmse_px=fitted["rmse"], axis_sse_xy_px_squared=axis_sse.tolist(),
        fraction_sse_x=float(axis_sse[0]/fitted["sse"]) if fitted["sse"] else 0.,
        rows=rows, leave_one_out_is_diagnostic_only=True,
        no_measurement_removed_from_observed_fit=True)


def prior_only_prediction(history, timestamp_ns, observed):
    if len(history) < MINIMUM_HITS:
        return None
    fitted = fit_qr(history)
    dt = (timestamp_ns-fitted["time_origin_ns"])/1e9
    predicted = np.asarray((1., dt, dt*dt))@fitted["coefficient"]
    difference = np.asarray(observed)-predicted
    return dict(history_frames=[m["frame"] for m in history], maximum_used_frame=history[-1]["frame"],
        extrapolation_seconds=dt, predicted_reference_xy=predicted.tolist(),
        observed_minus_prediction_xy=difference.tolist(), innovation_norm_px=float(np.linalg.norm(difference)),
        prediction_is_not_a_measurement=True, uncertainty_or_outlier_probability_estimated=False,
        used_to_change_association_or_qualification=False)


def candidate_record(candidate, global_index, bright_index):
    shape = candidate.get("shape")
    return dict(global_candidate_index=global_index, bright_candidate_index=bright_index,
                raw_reference_xy=[candidate["x"], candidate["y"]], source_xy=candidate["source_xy"],
                detector_score=candidate["score"], response_dn=candidate.get("response_dn"),
                shape=copy.deepcopy(shape), score_is_not_existence_confidence=True)


def find_measurement(row, track):
    bright = [(index, c) for index, c in enumerate(row["candidates"]) if c["polarity"] == "bright"]
    matched = [(index, ci, c) for ci, (index, c) in enumerate(bright)
               if math.dist(c["source_xy"], track["measurement_source_xy"]) < 1e-7]
    require(len(matched) == 1, "Measurement must identify one exact bright detector candidate")
    global_index, bright_index, candidate = matched[0]
    raw = np.asarray([candidate["x"], candidate["y"]], dtype=float)
    require(np.isfinite(raw).all(), "Nonfinite accepted reference peak")
    matrix = np.asarray(row["source_to_reference"], dtype=float)
    projected = matrix@np.asarray([*track["measurement_source_xy"], 1.])
    require(np.isfinite(projected).all() and projected[2] != 0
            and np.allclose(projected[:2]/projected[2], raw, rtol=0, atol=1e-7),
            "Source measurement does not map back to accepted raw reference peak")
    matching_audits = [a for a in row["tracking_metrics"]["bright"]["association_audit"]
                       if a["track_id"] == int(track["track_id"].split(":")[1])]
    require(len(matching_audits) <= 1, "Duplicate association audit for track")
    audit = copy.deepcopy(matching_audits[0]) if matching_audits else None
    if audit:
        require(audit["candidate_index"] == bright_index, "Association index is not the bright-only candidate index")
        alternate = audit.get("alternate_candidate_index")
        if alternate is not None:
            require(0 <= alternate < len(bright), "Invalid alternate bright candidate index")
            alternate_global, alternate_candidate = bright[alternate]
            audit["alternate_candidate"] = candidate_record(alternate_candidate, alternate_global, alternate)
        audit["likelihood_gap_is_not_regularized_assignment_gap"] = True
        audit["negative_alternate_gap_does_not_prove_assignment_error"] = True
    else:
        require(track["hits"] == 1, "Nonbirth measurement lacks expected association audit")
    return dict(frame=row["frame_index"], timestamp_ns=row["timestamp_ns"],
                track_id=track["track_id"], **candidate_record(candidate, global_index, bright_index),
                association_audit=audit, association_audit_absent_because_logged_birth=not bool(audit))


def replay(rows):
    histories = {identity:[] for identity in IDENTITIES}
    states, measurements, births, comparisons = [], [], {}, []
    for row in rows:
        frame = row["frame_index"]
        target_tracks = {t["track_id"]:t for t in row["tracks"] if t["track_id"] in IDENTITIES}
        current_measurements = {}
        for identity in IDENTITIES:
            if identity not in target_tracks:
                require(identity not in births, "Target track disappeared before bounded audit end")
                continue
            track = target_tracks[identity]
            require(track["segment"] == 0, "Unexpected reference segment in bounded histories")
            history = histories[identity]
            if identity not in births:
                require(track["measured"] and track["hits"] == 1, "Complete history must start at first logged birth")
                births[identity] = frame
            prediction = None
            if track["measured"]:
                measurement = find_measurement(row, track)
                prediction = prior_only_prediction(history[-WINDOW_HITS:], row["timestamp_ns"], measurement["raw_reference_xy"])
                if history:
                    previous = history[-1]
                    seconds = (measurement["timestamp_ns"]-previous["timestamp_ns"])/1e9
                    require(seconds > 0, "Noncausal measurement chronology")
                    delta = np.asarray(measurement["raw_reference_xy"])-previous["raw_reference_xy"]
                    measurement["previous_actual_frame"] = previous["frame"]
                    measurement["actual_to_actual_interval_seconds"] = seconds
                    measurement["raw_reference_displacement_xy"] = delta.tolist()
                    measurement["finite_difference_velocity_xy_px_s"] = (delta/seconds).tolist()
                history.append(measurement)
                measurements.append(measurement)
                current_measurements[identity] = measurement
            require(len(history) == track["hits"], "Saved history is incomplete or hit count changed")
            recent = history[-WINDOW_HITS:]
            ready = len(recent) >= MINIMUM_HITS
            fit = describe_fit(recent) if ready else None
            passed = bool(ready and fit["rmse_px"] <= MAXIMUM_RMSE)
            quality = track["motion_quality"]
            require(quality["ready"] is ready and quality["passed"] is passed
                    and quality["measured_history_count"] == len(recent)
                    and quality["maximum_rmse_px"] == MAXIMUM_RMSE,
                    "Independent reconstructed quality state differs")
            difference = None
            if ready:
                difference = abs(fit["rmse_px"]-quality["quadratic_fit_rmse_px"])
                require(difference <= 1e-8, "Independent QR RMSE differs from frozen journal")
            else:
                require(quality["quadratic_fit_rmse_px"] is None, "Not-ready state unexpectedly has RMSE")
            states.append(dict(frame=frame, timestamp_ns=row["timestamp_ns"], track_id=identity,
                measured=track["measured"], lifecycle=track["lifecycle"],
                qualified_moving=track["qualified_moving"], total_hits=track["hits"],
                independent_hits=track["independent_hits"], excursion_px=track["excursion_px"],
                confirmation_timestamp_ns=track["confirmation_timestamp_ns"],
                last_actual_measurement_frame=history[-1]["frame"],
                logged_motion_quality=copy.deepcopy(quality), independently_recomputed_ready=ready,
                independently_recomputed_passed=passed, rmse_absolute_difference_px=difference,
                fit=fit, prior_only_prediction=prediction,
                qualification_not_redefined=True))
        if all(identity in current_measurements for identity in IDENTITIES):
            a, b = (current_measurements[identity] for identity in IDENTITIES)
            delta = np.asarray(b["raw_reference_xy"])-a["raw_reference_xy"]
            comparisons.append(dict(frame=frame, track_ids=list(IDENTITIES),
                reference_xy_80=a["raw_reference_xy"], reference_xy_138=b["raw_reference_xy"],
                delta_138_minus_80_xy=delta.tolist(), separation_px=float(np.linalg.norm(delta)),
                both_actual_measured=True, physical_identity_or_object_count_inferred=False))
    require(births == {"bright:80":9, "bright:138":14}, "Unexpected bounded history births")
    return dict(birth_frames=births, measured_counts={k:len(v) for k,v in histories.items()},
                measurements=measurements, states=states, simultaneous_measurement_geometry=comparisons,
                focus_states=[s for s in states if s["frame"] in (16,17)],
                recomputed_rmse_count=sum(s["fit"] is not None for s in states),
                maximum_rmse_absolute_difference_px=max(s["rmse_absolute_difference_px"] or 0 for s in states))


def self_test():
    history = [dict(frame=i, timestamp_ns=i*100_000_000,
                    raw_reference_xy=[30+2*i+.5*i*i, 40-3*i+.25*i*i]) for i in range(8)]
    fit = fit_qr(history)
    require(fit["rmse"] < 1e-10, "Exact quadratic QR self-test failed")
    prediction = prior_only_prediction(history[:-1], history[-1]["timestamp_ns"], history[-1]["raw_reference_xy"])
    require(prediction["maximum_used_frame"] == 6 and prediction["innovation_norm_px"] < 1e-10,
            "Causal prior-only prediction self-test failed")
    perturbed = copy.deepcopy(history)
    perturbed[4]["raw_reference_xy"][0] += 4
    result = describe_fit(perturbed)
    require(abs(sum(r["residual_squared_px"] for r in result["rows"])-result["sse_px_squared"]) < 1e-10,
            "Residual decomposition self-test failed")
    return dict(exact_quadratic=True, prior_only_causal_prediction=True, residual_decomposition=True)


def run(output):
    output = Path(output).resolve()
    require(output.parent == OUTPUT_DIRECTORY and output.suffix == ".json", "Fresh JSON must stay in the bounded V39 result directory")
    if output.exists():
        raise FileExistsError("Refusing to overwrite an existing diagnostic")
    hashes = {str(path):sha(path) for path in INPUTS}
    require(hashes[str(JOURNAL)] == JOURNAL_SHA256, "Original journal bytes changed")
    tests = self_test()
    rows = []
    with JOURNAL.open() as stream:
        for index in range(LAST_FRAME+1):
            line = stream.readline()
            require(bool(line), "Truncated original journal prefix")
            row = json.loads(line)
            require(row["frame_index"] == index and row["timestamp_ns"] == index*100_000_000,
                    "Expected original contiguous 10 Hz frame grid")
            rows.append(row)
    replayed = replay(rows)
    result = dict(schema="seaqr.accuracy-v39-bounded-motion-history-diagnostic.v1", completed=True,
        clip="0029", inspected_journal_frames=[0,LAST_FRAME], analyzed_identities=list(IDENTITIES),
        parameters=dict(window_hits=WINDOW_HITS, minimum_hits=MINIMUM_HITS, maximum_rmse_px=MAXIMUM_RMSE),
        method="Unscaled-seconds QR least squares, independent of production's normalized-time lstsq",
        raw_measurement_source="Exactly matched accepted bright candidate x,y in reference pixels; verified via source_to_reference",
        filtered_track_reference_xy_used_as_measurement=False, source_annotations_read_or_used=False,
        leave_one_out_is_explanatory_not_an_outlier_removal_or_gate=True,
        prior_only_predictions_never_use_current_or_future_measurements=True,
        full_association_objective_or_optimum_recomputed=False,
        protocol="Airborne class unknown; image-feature diagnostics are not operational airborne accuracy",
        source_media_opened=False, raw16_accessed=False, sealed_holdouts_accessed=False, remote_accessed=False,
        thresholds_tuned=False, measurements_reassigned=False, production_changed=False,
        synthetic_self_tests=tests, inputs_sha256=hashes, **replayed)
    for path,digest in hashes.items():
        require(sha(path) == digest, "Input changed during read-only analysis")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(output=str(output), measured_counts=result["measured_counts"],
        recomputed_rmse_count=result["recomputed_rmse_count"],
        maximum_rmse_absolute_difference_px=result["maximum_rmse_absolute_difference_px"])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.output)
