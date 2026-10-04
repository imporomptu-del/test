"""Freeze the unchanged V44 synthetic matrix and run four V45 mathematical arms."""
from collections import Counter
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import platform

import numpy as np

from accuracy_v44_contrast import source_contrast
from accuracy_v44_synthetic_cases import build_cases, scenario_manifest
from accuracy_v45_presence import source_presence
from accuracy_v45_causal_probe import ARMS, evaluate_causal_probe
from run_accuracy_v44_synthetic import ARRAY_KEYS, causal_cases


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "results/tiny_target/accuracy_v45_20260925"
BASELINE = ROOT / "results/tiny_target/accuracy_v44_20260925/synthetic_01"
BASELINE_RECEIPT_SHA256 = "782949f426b9538922e580175ec91241cf7569cffc4cc8568a81109d1b44d76c"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def require_hashes(bindings):
    changed = [name for name, digest in bindings.items() if sha(name) != digest]
    if changed:
        raise ValueError("Frozen file changed: " + repr(changed))


def dependency_paths():
    paths = [ROOT/"docs/accuracy_v45_plan.md", ROOT/"scripts/accuracy_v42_localized.py",
             ROOT/"scripts/accuracy_v43_components.py", ROOT/"scripts/accuracy_v43_bounds.py"]
    for version in (44, 45):
        paths += list((ROOT/"scripts").glob(f"*accuracy_v{version}*.py"))
        paths += list((ROOT/"tests/unit").glob(f"test_accuracy_v{version}*.py"))
    return sorted(set(paths))


def input_arrays(cases, probes):
    arrays = {}
    for c in cases:
        arrays[c["case_id"]] = dict({k: c[k] for k in ARRAY_KEYS},
            history_images=c["synthetic_observations"]["history_images"],
            current_image=c["synthetic_observations"]["current_image"])
    for p in probes:
        arrays["causal_"+p["case_id"]] = dict(
            current129=p["current129"], history129=p["history129"],
            prior_centers_xy=np.asarray([[np.nan, np.nan] if c is None else c for c in p["prior_centers_xy"]]),
            predicted_offset_xy=np.asarray(p["predicted_offset_xy"]))
    return arrays


def memory_fingerprints(arrays):
    """Bind shape, dtype and values, including the supplied geometry arrays."""
    result = {}
    for name, values in arrays.items():
        for key, value in values.items():
            value = np.asarray(value)
            digest = hashlib.sha256()
            digest.update(str((value.shape, value.dtype.str)).encode("ascii"))
            digest.update(value.tobytes(order="C"))
            result[name+":"+key] = digest.hexdigest()
    return result


def checked_baseline(arrays):
    receipt = BASELINE/"completion_receipt.json"
    if sha(receipt) != BASELINE_RECEIPT_SHA256:
        raise ValueError("V44 completion receipt differs from the pinned checkpoint")
    bound = json.loads(receipt.read_text())["files_sha256"]
    # Only explicit synthetic artifacts and source files bound by that receipt;
    # do not follow any older experiment/result chain.
    require_hashes(bound)
    bindings = {str(receipt): sha(receipt)}
    for name, expected in arrays.items():
        path = BASELINE/"inputs"/(name+".npz")
        if str(path) not in bound:
            raise ValueError("Synthetic input missing from V44 receipt")
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(expected):
                raise ValueError("Synthetic input keys changed")
            for key, value in expected.items():
                if not np.array_equal(archive[key], value, equal_nan=True):
                    raise ValueError("Synthetic input changed: " + name + ":" + key)
        bindings[str(path)] = sha(path)
    result = {}
    for name in ("oracle_results", "causal_results"):
        path = BASELINE/(name+".json")
        if str(path) not in bound:
            raise ValueError("Baseline result missing from receipt")
        bindings[str(path)] = sha(path)
        result[name] = json.loads(path.read_text())
    return bindings, result


def count_evidence(values):
    counts = Counter(states=0, available=0, unavailable=0, excludes_zero=0,
                     includes_zero=0, positive=0, negative=0, unresolved=0)
    for v in values:
        counts["states"] += 1
        if v is None or not v["available"]:
            counts["unavailable"] += 1
            continue
        counts["available"] += 1
        counts["excludes_zero" if v["interval_excludes_zero"] else "includes_zero"] += 1
        counts[v["coefficient_sign"]] += 1
    return dict(counts)


def summarize(records, causal, baseline):
    if len(records) != 16 or len(causal) != 6:
        raise ValueError("All unchanged 16 oracle and six causal cases required")
    if [r["case_id"] for r in records] != [r["case_id"] for r in baseline["oracle_results"]]:
        raise ValueError("Oracle order/membership changed")
    if [r["case_id"] for r in causal] != [r["case_id"] for r in baseline["causal_results"]]:
        raise ValueError("Causal order/membership changed")
    for new, old in zip(records, baseline["oracle_results"]):
        if new["arms"]["amplitude_old_bounds"] != old["contrast"]:
            raise ValueError("V44 oracle baseline changed: " + new["case_id"])
        for key in ("truth", "integration", "temporal_provenance"):
            if new[key] != old[key]:
                raise ValueError("Case provenance changed")
        if (new["arms"]["amplitude_old_bounds"] != new["arms"]["amplitude_box_bounds"]
                or new["arms"]["presence_old_bounds"] != new["arms"]["presence_box_bounds"]):
            raise ValueError("Oracle supplied bounds cannot differ across bound arms")
        for v in new["arms"].values():
            if v["motion_status"] != "unknown" or v["physical_class"] != "unknown":
                raise ValueError("Numerical evidence became physical classification")
    for new, old in zip(causal, baseline["causal_results"]):
        if new["arms"]["amplitude_old_bounds"]["raw_adapter_result"] != old["result"]:
            raise ValueError("V44 causal baseline changed: " + new["case_id"])
        original = new["arms"]["amplitude_old_bounds"]["raw_adapter_result"]
        for arm in new["arms"].values():
            value = arm["raw_adapter_result"]
            for key in ("learned_design_sha256", "common_support_sha256", "common_support_count",
                        "ambiguity_reasons", "prior_context", "components"):
                if value[key] != original[key]:
                    raise ValueError("Mathematical arm changed nominal evidence: " + key)
            if (value["motion_status"] != "unknown" or value["physical_class"] != "unknown"
                    or value["is_motion_or_classification_gate"]):
                raise ValueError("Causal probe became physical classification")
    twins = [r["arms"] for r in records if r["case_id"].startswith("identifiability_")]
    if len(twins) != 2 or twins[0] != twins[1]:
        raise ValueError("Observationally identical twins differ")
    return dict(
        oracle_counts_by_arm={arm: count_evidence(r["arms"][arm] for r in records) for arm in ARMS},
        causal_counts_by_arm={arm: count_evidence(r["arms"][arm]["raw_adapter_result"]["numerical_contrast"]
                                                for r in causal) for arm in ARMS},
        oracle_box_arms_duplicate_supplied_bounds=True, baseline_exactly_reproduced=True,
        nominal_design_support_and_ambiguity_identical_across_arms=True,
        observational_twins_identical=True, numerator_and_amplitude_are_different_quantities=True,
        synthetic_only=True, production_changed=False, real_media_read=False,
        no_motion_or_airborne_classification=True, no_real_accuracy_gain_claimed=True)


def run(output):
    output = Path(output).resolve()
    if OUTPUT_ROOT not in output.parents:
        raise ValueError("Output must be a fresh child of the dedicated V45 directory")
    if output.exists():
        raise FileExistsError("Frozen output cannot be reused")
    bindings = {str(path): sha(path) for path in dependency_paths()}
    cases, probes = build_cases(), causal_cases()
    manifest = scenario_manifest()
    if [c["case_id"] for c in cases] != [c["case_id"] for c in manifest["scenarios"]]:
        raise ValueError("Case manifest changed")
    if (len(cases) != 16 or len(probes) != 6
            or len({c["case_id"] for c in cases}) != 16
            or len({p["case_id"] for p in probes}) != 6):
        raise ValueError("Require 16 unique oracle and six unique causal cases before scoring")
    arrays = input_arrays(cases, probes)
    if len(arrays) != 22:
        raise ValueError("All 22 unique input archives required before scoring")
    original_memory = memory_fingerprints(arrays)
    for values in arrays.values():
        for value in values.values():
            np.asarray(value).flags.writeable = False
    baseline_bindings, baseline = checked_baseline(arrays)
    bindings.update(baseline_bindings)
    output.mkdir(parents=True)
    (output/"inputs").mkdir()
    write_json(output/"freeze.json", dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        dependency_sha256=bindings, arms={k:list(v) for k,v in ARMS.items()},
        oracle_manifest=manifest, causal_cases=[p["case_id"] for p in probes],
        baseline_receipt_sha256=BASELINE_RECEIPT_SHA256,
        all_22_inputs_array_equal_to_v44=True, synthetic_only=True,
        in_memory_input_sha256=original_memory,
        runtime=dict(python=platform.python_version(), numpy=np.__version__)))
    inputs = {}
    for name, values in arrays.items():
        path = output/"inputs"/(name+".npz")
        np.savez_compressed(path, **values)
        inputs[str(path)] = sha(path)
    write_json(output/"inputs_complete.json", dict(inputs_sha256=inputs, scores_started=False,
        created_at_utc=datetime.now(timezone.utc).isoformat()))
    require_hashes(bindings); require_hashes(inputs)
    write_json(output/"score_start.json", dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        freeze_sha256=sha(output/"freeze.json"), inputs_complete_sha256=sha(output/"inputs_complete.json")))
    records = []
    for c in cases:
        args = {key:c[key] for key in ARRAY_KEYS}
        # No learned stamps exist in these oracle cases: bound-arm pairs are
        # explicitly repeated references, not four independent observations.
        amplitude, numerator = source_contrast(**args), source_presence(**args)
        arms = {arm: amplitude if quantity == "amplitude" else numerator
                for arm,(quantity,_) in ARMS.items()}
        records.append(dict(case_id=c["case_id"], arms=arms, truth=c["truth"],
            integration=c["integration"], temporal_provenance=c["temporal_provenance"],
            temporal_metadata_used_by_numerical_solver=False,
            motion_status="unknown", physical_class="unknown"))
    causal = [dict(case_id=p["case_id"], arms={arm:evaluate_causal_probe(
        **{k:v for k,v in p.items() if k != "case_id"}, arm=arm) for arm in ARMS}) for p in probes]
    if memory_fingerprints(input_arrays(cases, probes)) != original_memory:
        raise ValueError("In-memory inputs or supplied geometry mutated while scoring")
    summary = summarize(records, causal, baseline)
    write_json(output/"oracle_results.json", records)
    write_json(output/"causal_results.json", causal)
    write_json(output/"summary.json", summary)
    require_hashes(bindings); require_hashes(inputs)
    bound = dict(bindings, **inputs)
    bound.update({str(p):sha(p) for p in output.glob("*.json")})
    write_json(output/"completion_receipt.json", dict(completed=True, files_sha256=bound,
        created_at_utc=datetime.now(timezone.utc).isoformat(), production_changed=False, real_media_read=False))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        parser.error("--execute is required for the frozen synthetic-only matrix")
    print(json.dumps(run(args.output), indent=2))
