"""Bounded saved-result arithmetic / eight-fit audit, without video decoding.

Does not replay all new shape-bank searches or independently resample source
videos. Reads only current25 NPZ members and performs eight full-design fits.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/tiny_target/accuracy_v39_20260925"
RUN = BASE / "shapes_01"
PARENT = ROOT / "results/tiny_target/accuracy_v38_20260925/global_multilag_01"
AUDIT = PARENT.parent / "global_multilag_saved_evidence_audit_01.json"
AUDIT_SHA = "ec5f682215f5561939673fe7ad6dd7810128aa0a0cb301624e92bd4c05c0bd5d"
FAMILIES = ("centered_isotropic", "localized_isotropic", "localized_elongated", "localized_equal_pair")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    def unique(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result
    def nonfinite(value):
        raise ValueError("Nonfinite JSON constant: "+value)
    return json.loads(Path(path).read_text(), object_pairs_hook=unique, parse_constant=nonfinite)


def sha(path):
    require(Path(path).suffix.lower() not in (".avi", ".mp4", ".raw16", ".raw"),
            "Source media outside this saved-result audit")
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def close(a, b):
    if isinstance(b, dict):
        require(isinstance(a, dict) and a.keys() == b.keys(), "Dictionary fields changed")
        for key in b:
            close(a[key], b[key])
    elif isinstance(b, list):
        require(isinstance(a, list) and len(a) == len(b), "List dimensions changed")
        for x, y in zip(a, b):
            close(x, y)
    elif type(b) is float:
        require(type(a) in (float, int) and math.isfinite(a)
                and math.isclose(a, b, rel_tol=2e-10, abs_tol=2e-8), "Numeric disagreement")
    else:
        require(type(a) is type(b) and a == b, "Scalar disagreement")


def fit(columns, patch):
    y, x = np.mgrid[-12:13, -12:13].astype(float)
    design = np.column_stack([z.ravel() for z in (np.ones_like(x), x, y, x*x, x*y, y*y)]
                            + [c.ravel() for c in columns])
    values = patch.ravel()
    coefficients = np.linalg.lstsq(design, values, rcond=None)[0]
    if len(columns) == 2 and coefficients[-1] < 0:
        coefficients = np.r_[np.linalg.lstsq(design[:, :-1], values, rcond=None)[0], 0.]
    residual = values-design@coefficients
    return float(residual@residual), coefficients


def edge(params):
    y, x = np.mgrid[-12:13, -12:13].astype(float)
    return np.tanh((x*math.cos(params["orientation_rad"])+y*math.sin(params["orientation_rad"])
                    -params["offset_px"])/params["width_px"])


def compact(family, parameters):
    y, x = np.mgrid[-12:13, -12:13].astype(float)
    def gaussian(center, sx, sy, angle):
        dx, dy = x-center[0], y-center[1]
        u, v = dx*math.cos(angle)+dy*math.sin(angle), -dx*math.sin(angle)+dy*math.cos(angle)
        return np.exp(-.5*((u/sx)**2+(v/sy)**2))
    xy = np.asarray(parameters["center_xy"])
    if family.endswith("isotropic"):
        return gaussian(xy, parameters["sigma_px"], parameters["sigma_px"], 0.)
    angle = parameters["orientation_rad"]
    if family.endswith("elongated"):
        return gaussian(xy, parameters["sigma_u_px"], parameters["sigma_v_px"], angle)
    delta = .5*parameters["separation_px"]*np.array([math.cos(angle), math.sin(angle)])
    sigma = parameters["sigma_px"]
    return gaussian(xy-delta, sigma, sigma, 0.)+gaussian(xy+delta, sigma, sigma, 0.)


def arithmetic(records):
    answer = dict(observations=len(records), available=sum(r["diagnostic"]["available"] for r in records), families={})
    for name in FAMILIES:
        best = [r["diagnostic"]["families"][name]["best"] for r in records if r["diagnostic"]["available"]]
        gains = [b["gain_over_best_edge_fraction"] for b in best if b["gain_over_best_edge_fraction"] is not None]
        median = lambda values: float(np.median(values)) if values else None
        answer["families"][name] = dict(fitted_states=len(best),
            search_boundary_winners=sum(b["search_boundary_winner"] for b in best),
            compact_coefficient_zero=sum(b["compact_coefficient_on_zero_boundary"] for b in best),
            conditional_gain_informative_states=len(gains), median_conditional_gain=median(gains),
            median_fitted_center_offset_px=median([math.hypot(*b["compact"]["center_offset_xy"]) for b in best]),
            median_compact_amplitude_dn=median([b["compact_amplitude_dn"] for b in best]))
    return answer


def audit():
    freeze, completion, summary = [read(RUN/name) for name in ("freeze.json", "completion_receipt.json", "summary.json")]
    require(freeze["pre_real_patch_scoring"] is True and freeze["all_selected_current_patches"] == 364, "Freeze scope")
    require(freeze["families"] == list(FAMILIES), "Family order changed")
    require(completion["completed"] is True and summary["completed"] is True, "Incomplete study")
    checked = completion["checked_files_sha256"].copy()
    require(checked[str(AUDIT)] == AUDIT_SHA, "Historical receipt pin changed")
    for path, digest in freeze["inputs_sha256"].items():
        require(checked.get(path) == digest, "Incomplete end-of-run binding")
    for name in ("freeze.json", "summary.json", "observations.json", "synthetic_study.json", "unit.log"):
        require(str(RUN/name) in checked, "Missing result binding")
    for name in ("scripts/evaluate_accuracy_v39_shapes.py", "scripts/accuracy_v39_shape_study.py",
                 "tests/unit/test_accuracy_v39_shape_study.py", "docs/accuracy_v39_plan.md"):
        require(checked[str(ROOT/name)] == checked[str(RUN/"implementation"/name)], "Snapshot drift")
    checked[str(RUN/"completion_receipt.json")] = sha(RUN/"completion_receipt.json")
    checked[str(Path(__file__).resolve())] = sha(__file__)
    for path, digest in checked.items():
        require(sha(path) == digest, "Changed bound file: "+path)
    require(summary["freeze_sha256"] == checked[str(RUN/"freeze.json")]
            and summary["observations_sha256"] == checked[str(RUN/"observations.json")], "Summary pointers")
    require("Ran 19 tests" in (RUN/"unit.log").read_text() and (RUN/"unit.log").read_text().rstrip().endswith("OK"), "Frozen tests did not pass")
    records, parent, selection = read(RUN/"observations.json"), read(PARENT/"observations.json"), read(PARENT/"selection.json")
    require(len(records) == len(parent) == len(selection["observations"]) == 364, "Changed observation denominator")
    require([r["key"] for r in records] == [r["key"] for r in parent] == [r["key"] for r in selection["observations"]], "Selection or ordering changed")
    require(len({r["key"] for r in records}) == 364, "Duplicate observations")
    direct, strata, array_keys = [], set(), set()
    with np.load(PARENT/"source_patches.npz", allow_pickle=False) as archive:
        for record, old in zip(records, parent):
            for name in ("key", "clip", "frame", "identity"):
                require(record[name] == old[name], "Copied identity changed")
            close(record["original_measurement_source_xy"], old["measurement_source_xy"])
            diagnostic = record["diagnostic"]
            close(diagnostic["current_xy"], old["current_xy"])
            require(diagnostic["polarity"] == old["polarity"], "Polarity changed")
            for lag in old["lags"]:
                close(diagnostic["baseline"], lag["evidence"]["current_source"]["features"])
            for field in ("classifier_promoted", "family_selection_applied", "lag_selection_applied", "source_localization_replaces_measurement"):
                require(diagnostic[field] is False, "Unexpected promotion or replacement")
            require(diagnostic["physical_class"] == "unknown" and diagnostic["physical_object_count"] is None
                    and diagnostic["presence_classification"] is None, "Unjustified object claim")
            desc = old["current25_array"]
            require(desc["array_key"] not in array_keys, "Repeated current patch")
            array_keys.add(desc["array_key"])
            patch = archive[desc["array_key"]]  # No prior-array members are accessed.
            require(patch.shape == (25, 25) and patch.dtype == np.float64, "Current array shape/dtype")
            require(hashlib.sha256(patch.tobytes()).hexdigest() == desc["sha256"] == record["source_patch_sha256"], "Current pixel digest")
            require(diagnostic["available"] is True and tuple(diagnostic["families"]) == FAMILIES, "Unexpected shape availability/family omission")
            for family, count in zip(FAMILIES, (27, 75, 200, 400)):
                values = diagnostic["families"][family]
                require(values["compact_template_count"] == count and values["joint_template_pair_count"] == count*72, "Nonuniform bank")
                require(values["location_alternatives_include_best"] is True and values["location_alternatives"][0] == values["best"], "Alternative winner lost")
                require(len({tuple(v["compact"]["center_xy"]) for v in values["location_alternatives"]}) == 3, "Repeated alternatives")
                for best in values["location_alternatives"]:
                    for field in ("residual_energy", "compact_amplitude_dn", "gain_over_best_edge", "background_gain_fraction"):
                        require(math.isfinite(best[field]) and best[field] >= 0, "Invalid fit scalar")
                    require(0 <= best["background_gain_fraction"] <= 1, "Fit fraction out of range")
                    fraction = best["gain_over_best_edge_fraction"]
                    require(fraction is None or (math.isfinite(fraction) and 0 <= fraction <= 1), "Conditional fraction")
                    require((fraction is not None) == diagnostic["edge_residual_informative"], "False informative fraction")
                    offset = best["compact"]["center_offset_xy"]
                    require(all(v in values["center_grid_px"] for v in offset), "Off-bank fitted center")
                    close(best["compact"]["center_xy"], [c+d for c, d in zip(old["current_xy"], offset)])
                    require(best["search_boundary_winner"] == any(abs(v) == max(values["center_grid_px"]) for v in offset), "Boundary disclosure changed")
                stratum = (old["polarity"], family)
                if stratum in strata:
                    continue
                strata.add(stratum)
                best, sign = values["best"], 1 if old["polarity"] == "bright" else -1
                point, e = compact(family, best["compact"]), edge(best["edge"])
                sse, coefficients = fit([e, sign*point], patch)
                for actual, expected in ((best["residual_energy"], sse),
                                         (best["compact_amplitude_dn"], float(coefficients[-1])),
                                         (best["edge"]["signed_amplitude_dn"], float(coefficients[-2]))):
                    close(actual, expected)
                old_edge = diagnostic["baseline"]
                edge_sse, _ = fit([edge(dict(width_px=old_edge["edge_width_px"], orientation_rad=old_edge["edge_orientation_rad"], offset_px=old_edge["edge_offset_px"]))], patch)
                close(diagnostic["best_edge_residual_energy"], edge_sse)
                close(best["gain_over_best_edge"], max(0., edge_sse-sse))
                direct.append(dict(key=record["key"], family=family, polarity=old["polarity"],
                    residual_energy_error=abs(best["residual_energy"]-sse),
                    compact_amplitude_error=abs(best["compact_amplitude_dn"]-float(coefficients[-1]))))
    require(len(direct) == 8, "Eight direct polarity/family fits required")
    by_key = {r["key"]: r for r in records}
    scopes, groups = defaultdict(set), []
    for group in selection["groups"]:
        scopes[group["kind"]+"/"+group["window"]].update(group["keys"])
        groups.append(dict(kind=group["kind"], window=group["window"], frame=group["frame"],
            baseline_selected_keys=group["keys"], available_shape_keys=[key for key in group["keys"] if by_key[key]["diagnostic"]["available"]]))
    close(summary["reference_groups"], groups)
    references = {}
    for kind, denominator, matches in (("dense", 285, 284), ("pilot", 28, 28), ("anchor", 24, 24), ("compact_light", 8, 6)):
        selected = [g for g in groups if g["kind"] == kind]
        require(len(selected) == denominator and sum(bool(g["baseline_selected_keys"]) for g in selected) == matches, "Frozen reference denominator/matching changed")
        references[kind] = dict(samples=denominator, baseline_matched_samples=matches,
            shape_available_samples=sum(bool(g["available_shape_keys"]) for g in selected), missing_samples_recovered=False, airborne_truth=False)
    close(summary["references"], references)
    for item in selection["observations"]:
        scopes["all"].add(item["key"])
        scopes["clip/"+item["clip"]].add(item["key"])
        for control in item["provisional_control_indices"]:
            scopes["control/"+str(control)].add(item["key"])
    require(sum(len(keys) for name, keys in scopes.items() if name.startswith("control/")) == 70, "Changed provisional control scope")
    independently_summarized = {name: arithmetic([by_key[key] for key in sorted(keys)]) for name, keys in sorted(scopes.items())}
    close(summary["scopes"], independently_summarized)
    require(summary["selected_observations"] == summary["unchanged_baseline_features_crosschecked"] == 364, "Summary counts")
    for field in ("family_selection_applied", "classifier_promoted", "measurement_coordinates_replaced", "airborne_accuracy_established", "raw16_accessed", "sealed_holdouts_accessed"):
        require(summary[field] is False, "Summary promotion claim")
    require(summary["real_video_frames_decoded"] == 0, "Unexpected video access")
    for path, digest in checked.items():
        require(sha(path) == digest, "Input changed during audit: "+path)
    return dict(schema="seaqr.accuracy-v39-bounded-shape-audit.v1", verified=True,
        experiment=str(RUN), checked_files_sha256=checked, original_current_arrays_verified=len(array_keys),
        every_original_baseline_compared_to_all_parent_lags=True, summary_arithmetic_independently_verified=True,
        reference_denominators=references, unmatched_groups=[g for g in groups if not g["baseline_selected_keys"]],
        direct_full_design_fits=direct, direct_fit_count=8, current_npz_members_only=True,
        all_shape_bank_searches_independently_replayed=False, source_video_sampling_validated=False,
        source_videos_opened=False, classifier_promoted=False,
        limits="Eight selected fits, not exhaustive validation of every new bank winner; no physical classification or accuracy conclusion")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), "Fresh audit receipt required")
    result = audit()
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(verified=True, observations=result["original_current_arrays_verified"],
                         direct_fits=8, bound_files=len(result["checked_files_sha256"]))))
