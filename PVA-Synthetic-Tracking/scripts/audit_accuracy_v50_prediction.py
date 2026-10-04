"""Independent, explicit-execution audit of the completed V50 diagnostic.

No V50 producer module is imported.  The auditor reconstructs geometry and
arithmetic independently and opens only the literal non-embargo cache paths
jointly bound by the pinned V48 metadata and the fresh V50 freeze.  It never
recursively follows old receipts.  This validates computation, not physical
noise assumptions, airborne labels, statistical independence, or causality of
the inherited camera registration.
"""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import re

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/tiny_target/accuracy_v50_20260926"
BASE = ROOT / "results/tiny_target/accuracy_v48_20260926/shadow_01"
CACHE = ROOT / "results/tiny_target/accuracy_v43_20260925/stability_01"
BASE_RECEIPT_SHA = "b21753dfdfc6fb253f800c6a6fc3ad1ee393200058e488af6e423ffd3128a251"
ARMS = ("median8_unit_scale", "median8_temporal_scale", "median3_temporal_scale")
POLICIES = ("disjoint_anchor_frames", "all_calibration_frames")
PARTITIONS = ("calibration", "embargo", "evaluation")
RUN_FILES = ("freeze.json", "calibration_forecasts.jsonl", "calibration_forecasts_frozen.json",
             "calibration_measurements.jsonl", "calibration.json", "evaluation_forecasts.jsonl",
             "evaluation_forecasts_frozen.json", "evaluation_measurements.jsonl",
             "evaluation_intervals.jsonl", "state_results.jsonl", "reference_context.json",
             "nine_frame_bins.json", "summary.json")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def exact_file(path):
    path = Path(path)
    require(path.is_absolute() and path.resolve() == path and path.is_file(),
            "Expected canonical regular file: " + str(path))
    return path


def digest(path):
    return hashlib.sha256(exact_file(path).read_bytes()).hexdigest()


def _reject_constant(value):
    raise ValueError("Nonfinite JSON constant: " + value)


def read_json(path):
    return json.loads(exact_file(path).read_text(), parse_constant=_reject_constant)


def read_lines(path):
    return [json.loads(x, parse_constant=_reject_constant)
            for x in exact_file(path).read_text().splitlines()]


def content_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def same(actual, expected, where="value"):
    """Structural equality with explicit tolerance only for floating summaries."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), where + ": keys differ")
        for key in expected:
            same(actual[key], expected[key], where + "/" + str(key))
    elif isinstance(expected, (tuple, list)):
        require(isinstance(actual, (list, tuple)) and len(actual) == len(expected),
                where + ": sequence length differs")
        for i, (a, b) in enumerate(zip(actual, expected)):
            same(a, b, where + "/" + str(i))
    elif isinstance(expected, float):
        require(type(actual) in (float, int) and math.isfinite(actual)
                and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12),
                where + ": floating value differs")
    else:
        require(type(actual) is type(expected) and actual == expected, where + ": value differs")


def state_key(row):
    return row["clip"], row["frame_index"], row["segment"], row["track_id"]


def greedy(frames):
    chosen = []
    for frame in sorted(set(frames)):
        if not chosen or frame - chosen[-1] >= 9:
            chosen.append(frame)
    return chosen


def metadata_scope(rows):
    require(len(rows) == len({state_key(r) for r in rows}), "Duplicate original state")
    groups = defaultdict(set)
    for row in rows:
        require(row["clip"] in ("0029", "0126"), "Unexpected clip")
        require(type(row["frame_index"]) is int and row["frame_index"] >= 0,
                "Invalid response frame")
        require(type(row["segment"]) is int and row["segment"] >= 0, "Invalid segment")
        groups[(row["clip"], row["segment"])].add(row["frame_index"])
        if row["archive"] is not None:
            same(row["geometry"]["geometry"]["prior_frame_indices"],
                 list(range(row["frame_index"]-8, row["frame_index"])), "prior chronology")
    cuts = {g: sorted(fs)[(len(fs)-1)//2] for g, fs in groups.items()}
    parts = {}
    for row in rows:
        key = state_key(row); cutoff = cuts[(key[0], key[2])]
        parts[key] = ("calibration" if key[1] <= cutoff else "embargo"
                      if key[1] <= cutoff+8 else "evaluation")
        if parts[key] == "evaluation":
            require(key[1]-8 > cutoff, "Evaluation/calibration input span overlap")
    counts = {}
    for name in PARTITIONS:
        selected = [r for r in rows if parts[state_key(r)] == name]
        fs = {(r["clip"], r["segment"], r["frame_index"]) for r in selected}
        ar = [r for r in selected if r["archive"] is not None]
        af = {(r["clip"], r["segment"], r["frame_index"]) for r in ar}
        counts[name] = dict(states=len(selected), archived_states=len(ar),
            history_unknown_states=len(selected)-len(ar), unique_response_frames=len(fs),
            archived_response_frames=len(af), frames_without_archives=len(fs-af))
    return cuts, parts, counts


def validate_scope(saved, rows):
    cuts, parts, counts = metadata_scope(rows)
    for name, expected in (("history_frames", 8), ("embargo_frames", 8), ("anchor_spacing_frames", 9),
                           ("scope_only_no_scores_or_image_values_read", True)):
        same(saved[name], expected, "scope invariant " + name)
    same(saved["quantile_policies"], dict(primary="disjoint_input_span_calibration_frame_anchors",
        sensitivity="all_archived_calibration_response_frames_dependent", automatic_fallback=False,
        statistical_independence_established=False, production_policy=False), "scope policy limitations")
    same(saved["cutoffs"], [dict(clip=c, segment=s, cutoff_frame_index=f)
                           for (c, s), f in sorted(cuts.items())], "scope cutoffs")
    ordered = sorted(rows, key=lambda r: (r["clip"], r["segment"], r["frame_index"], r["track_id"]))
    same(saved["assignments"], [dict(state_key=list(state_key(r)), partition=parts[state_key(r)],
                                    archived=r["archive"] is not None) for r in ordered], "assignments")
    same(saved["counts"], counts, "scope counts")
    for part in PARTITIONS:
        selected = [r for r in ordered if parts[state_key(r)] == part]
        same(saved["partitions"][part], [list(state_key(r)) for r in selected], "partition rows")
        groups = defaultdict(list)
        for row in selected:
            groups[(row["clip"], row["segment"], row["frame_index"])].append(row)
        units, missing = [], []
        for (c, s, f), rs in sorted(groups.items()):
            good = [list(state_key(r)) for r in rs if r["archive"] is not None]
            bad = [list(state_key(r)) for r in rs if r["archive"] is None]
            if good:
                units.append(dict(clip=c, segment=s, frame_index=f, input_span=[f-8, f],
                                  state_keys=good, history_unknown_state_keys=bad))
            else:
                missing.append(dict(clip=c, segment=s, frame_index=f, state_keys=bad))
        same(saved["frames_without_archives"][part], missing, "missing-frame accounting")
        if part != "embargo":
            chosen = []
            for group in sorted(cuts):
                local = [u for u in units if (u["clip"], u["segment"]) == group]
                chosen.extend(u for u in local if u["frame_index"] in greedy([x["frame_index"] for x in local]))
            same(saved["anchors"][part], chosen, "anchor units")
        if part == "calibration":
            same(saved["calibration_all_archived_frames"], units, "sensitivity units")
    return cuts, parts, counts


def _middle(values):
    values = sorted(values)
    i = len(values)//2
    return values[i] if len(values)%2 else (values[i-1]+values[i])/2.0


def reconstruct_forecast(history, centers):
    """Separate scalar/order-statistic implementation; no producer calls."""
    require(history.shape == (8, 129, 129), "Invalid history shape")
    candidates = [(x, y) for y in range(8, 121, 8) for x in range(8, 121, 8)
                  if 40 <= max(abs(x-64), abs(y-64)) <= 56]
    keep = []; rejected_near = rejected_nonfinite = 0
    values = {}
    for x, y in candidates:
        v = [float(history[t, y, x]) for t in range(8)]
        near = any(c is not None and max(abs(x-c[0]), abs(y-c[1])) <= 12 for c in centers)
        finite = all(math.isfinite(z) for z in v)
        rejected_near += near; rejected_nonfinite += not finite
        if not near and finite:
            keep.append((x, y)); values[(x, y)] = v
    stencils = []
    for axis in ("x", "y"):
        for x, y in candidates:
            triplet = [(x-8, y), (x, y), (x+8, y)] if axis == "x" else [(x, y-8), (x, y), (x, y+8)]
            if all(point in values for point in triplet):
                stencils.append(dict(axis=axis, center_xy=[x, y],
                                     pixels_xy=[list(p) for p in triplet], weights=[1, -2, 1]))
    used = sorted({tuple(p) for s in stencils for p in s["pixels_xy"]}, key=lambda p: (p[1], p[0]))
    first = [_middle(values[p]) for p in used]
    latest = [_middle(values[p][-3:]) for p in used]
    scales = [max(1.0, _middle([abs(z-m) for z in values[p]])) for p, m in zip(used, first)]
    available = bool(used) and all(math.isfinite(x) for x in first+latest+scales)
    result = dict(available=available, reasons=[] if available else
                  ["no_prior_selected_guard_contrasts" if not used else "nonfinite_prior_forecast_arithmetic"],
        candidate_count=len(candidates), eligible_count=len(keep), used_count=len(used), stencil_count=len(stencils),
        candidate_support_sha256=content_hash(candidates), eligible_support_sha256=content_hash(keep),
        used_support_sha256=content_hash(used), stencil_sha256=content_hash(stencils),
        used_points_xy=[list(p) for p in used], stencils=stencils,
        prior_rejection_counts_nonexclusive=dict(prior_foreground_footprint=int(rejected_near),
                                                 nonfinite_prior_history=int(rejected_nonfinite)),
        missing_prior_center_indices=[i for i, c in enumerate(centers) if c is None], arms={})
    if available:
        result["arms"] = {ARMS[0]: dict(prediction=first, scale=[1.0]*len(used)),
                          ARMS[1]: dict(prediction=first, scale=scales),
                          ARMS[2]: dict(prediction=latest, scale=scales)}
    return result


def reconstruct_measurement(current, prediction, saved_forecast):
    result = dict(available=False, reasons=[], forecast_sha256=saved_forecast["forecast_sha256"],
                  used_support_sha256=prediction["used_support_sha256"], used_count=prediction["used_count"],
                  current_nonfinite_used_point_count=None, arms={})
    if not prediction["available"]:
        result["reasons"] = ["prior_forecast_unavailable"]
        return result
    observed = [float(current[y, x]) for x, y in prediction["used_points_xy"]]
    result["current_nonfinite_used_point_count"] = sum(not math.isfinite(v) for v in observed)
    if result["current_nonfinite_used_point_count"]:
        result["reasons"] = ["nonfinite_current_on_fixed_used_guard_support"]
        return result
    arms = {}
    for arm in ARMS:
        p = prediction["arms"][arm]
        residual = [v-m for v, m in zip(observed, p["prediction"])]
        normalized = [abs(r)/s for r, s in zip(residual, p["scale"])]
        peak = max(map(abs, residual))
        mae = 0.0 if not peak else peak*math.fsum(abs(r)/peak for r in residual)/len(residual)
        rmse = 0.0 if not peak else peak*math.sqrt(math.fsum((r/peak)**2 for r in residual)/len(residual))
        if not all(math.isfinite(v) for v in residual+normalized+[mae, rmse]):
            result["reasons"] = ["nonfinite_current_residual_or_summary"]
            return result
        arms[arm] = dict(residuals=residual, normalized_absolute_errors=normalized,
                         max_score=max(normalized), mae_dn=mae, rmse_dn=rmse)
    result.update(available=True, arms=arms)
    return result


def frame_units(records):
    grouped = defaultdict(list)
    for r in records:
        grouped[(r["clip"], r["segment"], r["frame_index"])].append(r)
    units = []
    for (c, s, f), rs in sorted(grouped.items()):
        missing = sum(not r["measurement"]["available"] for r in rs)
        scores = {arm: None if missing else max(r["measurement"]["arms"][arm]["max_score"] for r in rs)
                  for arm in ARMS}
        units.append(dict(clip=c, segment=s, frame_index=f, state_keys=[r["state_key"] for r in rs],
                          scores=scores, unavailable_packet_count=missing))
    return units


def quantile(scores):
    finite = sorted(x for x in scores if x is not None)
    rank = (9*(len(finite)+1)+9)//10
    reasons = ([] if finite else ["no_finite_calibration_units"])
    if rank > len(finite):
        reasons.append("requested_rank_exceeds_finite_units")
    return dict(available=not reasons, reasons=reasons, total_units=len(scores), finite_units=len(finite),
                missing_units=len(scores)-len(finite), rank=rank, q=finite[rank-1] if not reasons else None,
                target_numerator=9, target_denominator=10)


def calibration_policies(records):
    units = frame_units(records)
    groups = sorted({(u["clip"], u["segment"]) for u in units})
    result = {}
    for policy in POLICIES:
        entries = []
        for c, s in groups:
            frames = [u for u in units if (u["clip"], u["segment"]) == (c, s)]
            indices = [u["frame_index"] for u in frames]
            if policy == POLICIES[0]:
                indices = greedy(indices)
            chosen = [f for f in frames if f["frame_index"] in indices]
            entries.append(dict(clip=c, segment=s, selected_frame_indices=indices, frames=chosen,
                arms={a: quantile([f["scores"][a] for f in chosen]) for a in ARMS}))
        result[policy] = entries
    return result


def distribution(values):
    if not len(values):
        return dict(count=0, mean=None, median=None, p90=None, p95=None, max=None)
    a = np.asarray(values, dtype=float)
    return dict(count=len(a), mean=float(a.mean()), median=float(np.median(a)),
                p90=float(np.quantile(a, .9)), p95=float(np.quantile(a, .95)), max=float(a.max()))


def reconstruct_intervals(records, forecasts, policies):
    result = []
    for r in records:
        p = forecasts[tuple(r["state_key"])]; m = r["measurement"]
        item = {k: r[k] for k in ("state_key", "clip", "segment", "frame_index")}
        item["policies"] = {}
        for policy in POLICIES:
            qgroup = next(q for q in policies[policy] if (q["clip"], q["segment"]) == (r["clip"], r["segment"]))
            arms = {}
            for arm in ARMS:
                calibration = qgroup["arms"][arm]
                reason = ("forecast_unavailable" if not p["available"] else
                          "current_support_unavailable" if not m["available"] else
                          "calibration_unavailable" if not calibration["available"] else None)
                if reason:
                    arms[arm] = dict(available=False, reason=reason)
                else:
                    q = calibration["q"]; pa = p["arms"][arm]
                    half = [q*s for s in pa["scale"]]
                    covered = [e <= q for e in m["arms"][arm]["normalized_absolute_errors"]]
                    arms[arm] = dict(available=True, point_count=len(half), covered_point_count=sum(covered),
                        whole_packet_covered=all(covered), half_width_dn=distribution(half), half_width_values=half,
                        full_8bit_range_included_points=sum(c-h <= 0 and c+h >= 255 for c, h in zip(pa["prediction"], half)),
                        q=q, status="within_empirical_band" if all(covered) else "outside_empirical_band",
                        not_an_object_or_model_validity_decision=True)
            item["policies"][policy] = arms
        result.append(item)
    return result


def interval_summary(items, policy, arm):
    entries = [i["policies"][policy][arm] for i in items]
    good = [e for e in entries if e["available"]]
    grouped = defaultdict(list)
    for item, entry in zip(items, entries):
        grouped[(item["clip"], item["segment"], item["frame_index"])].append(entry)
    whole = [g for g in grouped.values() if all(e["available"] for e in g)]
    packets = sum(e["whole_packet_covered"] for e in good)
    points = sum(e["point_count"] for e in good)
    covered = sum(e["covered_point_count"] for e in good)
    frames = sum(all(e["whole_packet_covered"] for e in g) for g in whole)
    return dict(archived_packets=len(items), interval_available_packets=len(good),
        unavailable_reasons=dict(Counter(e["reason"] for e in entries if not e["available"])),
        covered_packets=packets, conditional_packet_coverage=packets/len(good) if good else None,
        guard_state_point_pairs=points, covered_guard_state_point_pairs=covered,
        conditional_point_coverage=covered/points if points else None,
        archived_frames=len(grouped), interval_available_whole_frames=len(whole),
        covered_whole_frames=frames, conditional_whole_frame_coverage=frames/len(whole) if whole else None,
        half_width_dn=distribution([h for e in good for h in e["half_width_values"]]),
        full_8bit_range_included_point_pairs=sum(e["full_8bit_range_included_points"] for e in good),
        frames_and_pixels_not_independent=True)


def reconstruct_metrics(records, intervals):
    output = {}
    for label in ["combined"] + sorted({r["clip"] for r in records}):
        rs = [r for r in records if label == "combined" or r["clip"] == label]
        items = [i for i in intervals if label == "combined" or i["clip"] == label]
        good = [r for r in rs if r["measurement"]["available"]]
        grouped = defaultdict(list)
        for r in rs:
            grouped[(r["clip"], r["segment"], r["frame_index"])].append(r)
        whole = [g for g in grouped.values() if all(r["measurement"]["available"] for r in g)]
        arms = {}
        for arm in ARMS:
            arms[arm] = dict(
                packet_mae_dn=distribution([r["measurement"]["arms"][arm]["mae_dn"] for r in good]),
                packet_rmse_dn=distribution([r["measurement"]["arms"][arm]["rmse_dn"] for r in good]),
                packet_maximum_absolute_error_dn=distribution([max(map(abs, r["measurement"]["arms"][arm]["residuals"])) for r in good]),
                packet_maximum_normalized_error=distribution([r["measurement"]["arms"][arm]["max_score"] for r in good]),
                complete_frame_macro_mae_dn=distribution([float(np.mean([r["measurement"]["arms"][arm]["mae_dn"] for r in g])) for g in whole]),
                policies={p: interval_summary(items, p, arm) for p in POLICIES})
        pairs = [r["measurement"]["arms"][ARMS[2]]["mae_dn"]-
                 r["measurement"]["arms"][ARMS[1]]["mae_dn"] for r in good]
        output[label] = dict(archived_packets=len(rs), scorable_packets=len(good),
            packet_unavailability_reasons=dict(Counter(reason for r in rs if not r["measurement"]["available"]
                                                       for reason in r["measurement"]["reasons"])),
            archived_frames=len(grouped), complete_scorable_frames=len(whole), arms=arms,
            paired_candidate_minus_median8_mae_dn=distribution(pairs),
            paired_candidate_better=sum(x < 0 for x in pairs), paired_equal=sum(x == 0 for x in pairs),
            paired_candidate_worse=sum(x > 0 for x in pairs))
    return output


def validate_packet_bindings(rows, parts, freeze, baseline_receipt, expected_count=493):
    """Closed literal allowlist; validate all paths before opening any packet."""
    expected = {}
    for row in rows:
        if row["archive"] is None or parts[state_key(row)] == "embargo":
            continue
        require(re.fullmatch(r"(?:bright|dark):[0-9]+", row["track_id"]) is not None,
                "Invalid packet track identity")
        relative = (f'inputs/{row["clip"]}_{row["frame_index"]:04d}_{row["segment"]}_'
                    f'{row["track_id"].replace(":", "_")}.npz')
        require(row["archive"]["path"] == relative, "Packet path differs from literal state identity")
        path = str(CACHE / relative)
        sha = row["archive"]["sha256"]
        require(re.fullmatch(r"[0-9a-f]{64}", sha) is not None, "Invalid packet digest")
        require(baseline_receipt["files_sha256"].get(path) == sha, "Packet not bound by pinned V48 receipt")
        require(path not in expected, "Duplicate packet path")
        expected[path] = sha
    same(freeze["packet_files_sha256"], expected, "closed packet allowlist")
    require(len(expected) == expected_count, "Unexpected packet denominator")
    return expected


def _audit_completed(run_dir):
    receipt = read_json(run_dir / "completion_receipt.json")
    require(receipt.get("completed") is True, "V50 is not complete")
    freeze = read_json(run_dir / "freeze.json")
    require(freeze.get("completed_before_packet_access") is True, "Missing pre-access freeze")
    require(freeze.get("production_changed") is False, "Production change declared")
    same(freeze["predictors"], list(ARMS), "frozen arms")
    same(freeze["policies"], list(POLICIES), "frozen policies")
    old_receipt_path = BASE / "completion_receipt.json"
    require(digest(old_receipt_path) == BASE_RECEIPT_SHA, "Pinned V48 receipt changed")
    old_receipt = read_json(old_receipt_path)
    inputs = freeze["input_files_sha256"]
    base_files = {str(BASE / n) for n in ("completion_receipt.json", "states.jsonl", "selected_ledger.json", "reference_evidence.json")}
    require(base_files <= set(inputs), "Original metadata omitted from freeze")
    for name, sha in inputs.items():
        p = Path(name)
        allowed_source = any(p.parent == ROOT / directory and p.suffix == extension
                             for directory, extension in (("scripts", ".py"), ("tests/unit", ".py"), ("docs", ".md")))
        require(name in base_files or allowed_source, "Frozen input outside closed metadata/source allowlist")
        require(digest(p) == sha, "Frozen input changed: " + name)
        if name in base_files:
            expected = BASE_RECEIPT_SHA if p == old_receipt_path else old_receipt["files_sha256"].get(name)
            require(sha == expected, "Original metadata differs from pinned receipt")
    rows = read_lines(BASE / "states.jsonl")
    refs = read_json(BASE / "reference_evidence.json")
    ledger = read_json(BASE / "selected_ledger.json")
    same(ledger["states"], [{k: v for k, v in r.items() if k != "arms"} for r in rows], "original ledger")
    require(len(rows) == 1211 and sum(r["archive"] is not None for r in rows) == 509, "Original denominator changed")
    require(len(refs["samples"]) == 355, "Reference denominator changed")
    cuts, parts, counts = validate_scope(freeze["scope"], rows)
    packets = validate_packet_bindings(rows, parts, freeze, old_receipt)
    bindings = dict(inputs, **packets)
    for name in RUN_FILES:
        path = run_dir / name
        bindings[str(path)] = digest(path)
    same(receipt["files_sha256"], bindings, "completion receipt binding set")
    predictions = {}; measured = {}; by_partition = {}
    packet_reads = []
    shapes = {"history129": (8, 129, 129), "current129": (129, 129),
              "prior_centers_xy": (8, 2), "predicted_offset_xy": (2,)}
    for partition in ("calibration", "evaluation"):
        subset = [r for r in rows if parts[state_key(r)] == partition and r["archive"] is not None]
        saved_predictions = read_lines(run_dir / (partition + "_forecasts.jsonl"))
        saved_measurements = read_lines(run_dir / (partition + "_measurements.jsonl"))
        expected_keys = [list(state_key(r)) for r in subset]
        same([p["state_key"] for p in saved_predictions], expected_keys, "forecast membership")
        same([m["state_key"] for m in saved_measurements], expected_keys, "measurement membership")
        for row, sp, sm in zip(subset, saved_predictions, saved_measurements):
            path = CACHE / row["archive"]["path"]
            require(str(path) in packets, "Attempt to open unapproved packet")
            payload = exact_file(path).read_bytes()
            require(hashlib.sha256(payload).hexdigest() == packets[str(path)], "Packet changed before decode")
            packet_reads.append(str(path))
            with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
                require(set(archive.files) == set(shapes), "Packet member schema differs")
                arrays = {n: archive[n] for n in shapes}
            for name, a in arrays.items():
                require(a.dtype == np.float64 and a.shape == shapes[name], "Packet array schema differs")
            ca = arrays["prior_centers_xy"]
            require((np.isfinite(ca).all(axis=1) | np.isnan(ca).all(axis=1)).all(), "Invalid center metadata")
            centers = [None if np.isnan(c).all() else c.tolist() for c in ca]
            same(centers, row["geometry"]["geometry"]["prior_centers_xy"], "prior center identity")
            same(arrays["predicted_offset_xy"].tolist(), row["geometry"]["geometry"]["predicted_offset_xy"], "forecast offset identity")
            oracle = reconstruct_forecast(arrays["history129"], centers)
            saved = sp["forecast"]
            for name, value in oracle.items():
                same(saved[name], value, "forecast " + str(state_key(row)) + "/" + name)
            require(content_hash({k: v for k, v in saved.items() if k != "forecast_sha256"}) == saved["forecast_sha256"],
                    "Saved forecast self-hash differs")
            result = reconstruct_measurement(arrays["current129"], oracle, saved)
            for name, value in result.items():
                same(sm["measurement"][name], value, "measurement " + str(state_key(row)) + "/" + name)
            expected_record = dict(state_key=list(state_key(row)), clip=row["clip"], segment=row["segment"],
                                   frame_index=row["frame_index"], forecast_available=oracle["available"])
            same({k: sm[k] for k in expected_record}, expected_record, "measurement identity")
            predictions[state_key(row)] = oracle
            # Use audited saved arithmetic for inclusive ties and exact paired
            # sign accounting; independent arithmetic above already checked it.
            measured[state_key(row)] = sm
        by_partition[partition] = saved_measurements
        frozen = read_json(run_dir / (partition + "_forecasts_frozen.json"))
        require(frozen["forecasts_sha256"] == digest(run_dir / (partition + "_forecasts.jsonl")), "Forecast freeze hash differs")
        require(frozen["current129_members_decoded"] is False and frozen["completed"] is True
                and frozen["packet_count"] == len(subset), "Forecast freeze claims differ")
    require(len(packet_reads) == len(set(packet_reads)) == 493, "Audit packet read denominator differs")
    require(set(packet_reads) == set(packets), "Not all and only authorized packets reconstructed")
    calibration = read_json(run_dir / "calibration.json")
    oracle_policies = calibration_policies(by_partition["calibration"])
    same(calibration["policies"], oracle_policies, "calibration policies")
    for flag in ("completed", "no_iid_or_exchangeability_coverage_guarantee", "unavailable_units_not_replaced",
                 "no_automatic_policy_fallback", "held_later_current_arrays_not_decoded"):
        require(calibration.get(flag) is True, "Calibration safety flag missing: " + flag)
    intervals = reconstruct_intervals(by_partition["evaluation"], predictions, oracle_policies)
    same(read_lines(run_dir / "evaluation_intervals.jsonl"), intervals, "evaluation coverage and widths")
    states = read_lines(run_dir / "state_results.jsonl")
    expected_states = []
    for row in rows:
        key = state_key(row); partition = parts[key]; m = measured.get(key)
        status = ("history_unknown" if row["archive"] is None else "embargo_not_scored" if partition == "embargo"
                  else "forecast_unavailable" if not m["forecast_available"] else
                  "response_unavailable" if not m["measurement"]["available"] else "background_measured")
        expected_states.append(dict(state_key=list(key), partition=partition, v50_status=status,
            original_qualified_moving=row["qualified_moving"], original_reference_samples=row["reference_samples"],
            source_scores_and_original_detections_unchanged=True))
    same(states, expected_states, "all state results including unknown/embargo")
    context = read_json(run_dir / "reference_context.json")
    stripped = deepcopy(context)
    states_by_key = {tuple(s["state_key"]): s for s in states}
    for sample in stripped["samples"]:
        detail = sample.pop("v50_background_context")
        original = sample["original"]; c, f = original["clip_id"], original["frame_index"]
        groups = [g for g in cuts if g[0] == c]
        require(len(groups) == 1, "Ambiguous reference segment")
        cutoff = cuts[groups[0]]
        partition = "calibration" if f <= cutoff else "embargo" if f <= cutoff+8 else "evaluation"
        identity = sample["original_strict_assigned_identity"]
        k = None if identity is None else (c, f, int(identity.split("/", 1)[0]), identity.split("/", 1)[1])
        same(detail, dict(partition=partition, original_assigned_state_key=None if k is None else list(k),
            original_assigned_state_status=None if k is None else states_by_key[k]["v50_status"],
            original_unassigned_stays_unassigned=k is None, not_detection_recall_or_airborne_identity=True), "reference context")
    same(stripped, refs, "all original reference assignments and alternatives unchanged")
    alternatives = sum(len(s["measured_alternatives"]) for s in refs["samples"])
    assigned = {(s["original"]["clip_id"], s["original"]["frame_index"], s["original_strict_assigned_identity"])
                for s in refs["samples"] if s["original_strict_assigned_identity"] is not None}
    misses = [dict(sample_index=s["sample_index"], clip=s["original"]["clip_id"], frame=s["original"]["frame_index"])
              for s in refs["samples"] if s["original_strict_assigned_identity"] is None]
    require((alternatives, len(assigned), len(misses)) == (424, 300, 3), "Reference accounting differs")
    summary = read_json(run_dir / "summary.json")
    same(summary["scope_counts"], counts, "summary scope")
    same(summary["partitions"], {p: dict(Counter(s["v50_status"] for s in states if s["partition"] == p)) for p in PARTITIONS}, "summary statuses")
    same(summary["evaluation"], reconstruct_metrics(by_partition["evaluation"], intervals), "all evaluation summaries")
    same(summary["original_unassigned_references"], misses, "summary misses")
    same(summary["reference_partition_counts"], dict(Counter(s["v50_background_context"]["partition"] for s in context["samples"])), "reference partition counts")
    for name, expected in (("original_states", 1211), ("original_archives", 509), ("opened_packets", 493),
                           ("embargo_archives_unread", 16), ("original_reference_samples", 355),
                           ("original_reference_alternatives", 424)):
        same(summary[name], expected, "summary denominator " + name)
    for flag in ("completed", "source_solver_or_detector_not_called", "no_untouched_test_airborne_accuracy_false_alarm_or_generalization_claim",
                 "current_registered_geometry_is_not_a_fully_causal_camera_pipeline", "forecasts_and_intervals_are_empirical_not_certified_physical_bounds",
                 "no_automatic_arm_selection_or_policy_fallback"):
        require(summary.get(flag) is True, "Summary safety flag missing: " + flag)
    require(summary.get("production_changed") is False, "Production change declared")
    anchor_keys = {tuple(k) for u in freeze["scope"]["anchors"]["evaluation"] for k in u["state_keys"]}
    anchor_items = [i for i in intervals if tuple(i["state_key"]) in anchor_keys]
    same(summary["evaluation_disjoint_anchor_intervals"],
         {p: {a: interval_summary(anchor_items, p, a) for a in ARMS} for p in POLICIES}, "evaluation anchor summaries")
    bins = read_json(run_dir / "nine_frame_bins.json")
    expected_bins = []
    grouped_states = defaultdict(list)
    for state in states:
        if state["partition"] == "evaluation":
            c, f, s, _ = state["state_key"]; grouped_states[(c, s, f//9)].append(state)
    for (c, s, b), local in sorted(grouped_states.items()):
        items = [i for i in intervals if (i["clip"], i["segment"], i["frame_index"]//9) == (c, s, b)]
        frames = {v["state_key"][1] for v in local}; archived_frames = {i["frame_index"] for i in items}
        expected_bins.append(dict(clip=c, segment=s, nine_frame_bin=b, selected_states=len(local),
            selected_response_frames=len(frames), state_status_counts=dict(Counter(v["v50_status"] for v in local)),
            response_frames_without_archives=sorted(frames-archived_frames),
            frames_with_history_unknown_states=sorted({v["state_key"][1] for v in local if v["v50_status"] == "history_unknown"}),
            policies={p: {a: interval_summary(items, p, a) for a in ARMS} for p in POLICIES}))
    same(bins, dict(bins=expected_bins, adjacent_bins_can_share_prior_frames=True,
                   not_independent_trials=True, history_unknown_states_are_not_covered=True), "all bins including unknown-only bins")
    times = [freeze["created_at_utc"], read_json(run_dir / "calibration_forecasts_frozen.json")["created_at_utc"],
             calibration["created_at_utc"], read_json(run_dir / "evaluation_forecasts_frozen.json")["created_at_utc"],
             summary["created_at_utc"], receipt["created_at_utc"]]
    parsed = [datetime.fromisoformat(t) for t in times]
    require(parsed == sorted(parsed), "Persisted freeze/calibration/evaluation chronology differs")
    for name, sha in bindings.items():
        require(digest(Path(name)) == sha, "Bound input/output changed during audit: " + name)
    return dict(original_states=1211, original_archives=509, audited_packets=493,
        embargo_packets_accessed=0, calibration_packets=190, evaluation_packets=303,
        original_history_unknown_states=702, later_history_unknown_states=470,
        references=355, alternatives=424, unique_original_assigned_states=300,
        unchanged_unassigned_references=misses, scope_counts=counts,
        reconstructed_forecasts=len(predictions), reconstructed_measurements=len(measured),
        reconstructed_evaluation_intervals=len(intervals), reconstructed_nine_frame_bins=len(expected_bins),
        forecast_guard_state_point_pairs=sum(v["used_count"] for v in predictions.values()),
        quantile_ranks={p: [dict(clip=e["clip"], segment=e["segment"], arms=e["arms"]) for e in oracle_policies[p]] for p in POLICIES},
        completion_receipt_sha256=digest(run_dir / "completion_receipt.json"),
        rechecked_bound_files=len(bindings), frozen_packet_path_set_sha256=content_hash(sorted(packets)),
        independent_arithmetic_float_comparison=dict(relative_tolerance=1e-12, absolute_tolerance=1e-12),
        inclusive_tie_and_sign_summaries_recomputed_from_independently_checked_saved_arithmetic=True,
        limitations=["Not an airborne-detection, false-alarm, or untouched-test evaluation.",
                     "Numerical recomputation is not an IEEE interval-arithmetic certificate.",
                     "Frame spacing does not prove independence or future coverage.",
                     "Inherited geometry uses current-frame registration.",
                     "Stored ordering records and frozen source support, but cannot independently prove, historical access order."])


def audit(run_dir, output, *, execute=False):
    require(execute is True, "Explicit post-completion execution confirmation required")
    run_dir = Path(run_dir).absolute(); output = Path(output).absolute()
    require(run_dir.parent == OUTPUT and run_dir.resolve() == run_dir and run_dir.is_dir(),
            "Expected an immediate canonical V50 run directory")
    require(output.parent == OUTPUT and output.resolve() == output and output.suffix == ".json" and not output.exists(),
            "Expected a fresh sibling V50 audit JSON")
    report = dict(completed=False, passed=False, issues=[], created_at_utc=datetime.now(timezone.utc).isoformat(),
                  run_directory=str(run_dir), producer_modules_imported=False, production_changed=False)
    try:
        report["checks"] = _audit_completed(run_dir)
        report.update(completed=True, passed=True)
    except Exception as exc:
        report.update(completed=True, passed=False, issues=[type(exc).__name__ + ": " + str(exc)])
    report["audit_source_sha256"] = digest(Path(__file__).resolve())
    test = ROOT / "tests/unit/test_accuracy_v50_prediction_audit.py"
    report["audit_tests_sha256"] = digest(test)
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    result = audit(args.run, args.output, execute=args.execute)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
