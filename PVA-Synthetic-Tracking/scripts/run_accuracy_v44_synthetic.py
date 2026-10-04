"""Freeze and run V44 synthetic controls only; no real image inputs accepted."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import numpy as np

from accuracy_v44_causal_probe import evaluate_causal_probe
from accuracy_v44_contrast import source_contrast
from accuracy_v44_synthetic_cases import build_cases, scenario_manifest


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "results/tiny_target/accuracy_v44_20260925"
ARRAY_KEYS = ("y", "Z", "m", "P", "response_bound", "nuisance_bound", "source_bound")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def causal_cases():
    """Explicit predeclared variations of the ordinary V43 synthetic fixture."""
    y, x = np.indices((129, 129))
    background = 50 + 20*(x >= 64) + 8*np.sin(y/12)
    centers = [[29+4*i, 64] for i in range(8)]
    point = lambda cx, cy: 30*np.exp(-((x-cx)**2+(y-cy)**2)/2)
    history = np.stack([background+point(*center) for center in centers])
    current = background + point(64, 64)
    plane = 1.5 + 3*(x-64)/64 - 2*(y-64)/64
    variants = (
        ("ordinary_prior_learned", current, centers),
        ("shared_affine_change", current+plane, centers),
        ("gain_and_affine_change", 1.2*current+plane, centers),
        ("current_source_absent", background, centers),
        ("current_source_six_pixels_off_forecast", background+point(70, 64), centers),
        ("insufficient_prior_centers", current, [None]*4 + centers[4:]),
    )
    return [dict(case_id=name, current129=image.copy(), history129=history.copy(),
                 prior_centers_xy=[None if c is None else list(c) for c in points],
                 predicted_offset_xy=[0., 0.], polarity="bright") for name, image, points in variants]


def dependency_paths():
    explicit = [ROOT/"docs/accuracy_v44_plan.md",
                ROOT/"scripts/accuracy_v42_localized.py", ROOT/"scripts/accuracy_v43_components.py",
                ROOT/"scripts/accuracy_v43_bounds.py"]
    return sorted(set(explicit + list((ROOT/"scripts").glob("*accuracy_v44*.py"))
                      + list((ROOT/"tests/unit").glob("test_accuracy_v44*.py"))))


def require_hashes(expected):
    mismatches = [name for name, digest in expected.items() if sha(name) != digest]
    if mismatches:
        raise ValueError("Frozen dependencies changed: " + repr(mismatches))


def summarize(records, causal):
    counts = Counter(states=len(records), interval_available=0, interval_excludes_zero=0,
                     interval_includes_zero=0, unavailable=0)
    for record in records:
        value = record["contrast"]
        counts["interval_available"] += int(value["available"])
        counts["unavailable"] += int(not value["available"])
        if value["available"]:
            counts["interval_excludes_zero" if value["interval_excludes_zero"] else "interval_includes_zero"] += 1
    twins = [r["contrast"] for r in records if r["case_id"].startswith("identifiability_")]
    if len(twins) != 2 or twins[0] != twins[1]:
        raise ValueError("Observationally identical inputs must have identical contrast outputs")
    return dict(oracle_vector_counts=dict(counts), observationally_identical_outputs_equal=True,
                causal_probe_count=len(causal), causal_intervals_available=sum(c["result"]["available"] for c in causal),
                causal_intervals_excluding_zero=sum(bool((c["result"]["numerical_contrast"] or {}).get("interval_excludes_zero")) for c in causal),
                synthetic_only=True, production_changed=False, real_media_read=False,
                no_physical_motion_or_airborne_decision=True,
                no_accuracy_gain_claimed=True)


def run(output):
    output = Path(output).resolve()
    if OUTPUT_ROOT not in output.parents:
        raise ValueError("Output must be a fresh child of the dedicated V44 result directory")
    if output.exists():
        raise FileExistsError("Frozen outputs cannot be reused or overwritten")
    dependencies = dependency_paths()
    bindings = {str(path): sha(path) for path in dependencies}
    manifest = scenario_manifest()
    probes = causal_cases()
    output.mkdir(parents=True)
    (output/"inputs").mkdir()
    freeze = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), synthetic_only=True,
                  dependency_sha256=bindings, oracle_manifest=manifest,
                  causal_cases=[p["case_id"] for p in probes],
                  causal_template_origin="V43 prior-image learning on ideal supplied synthetic prior centers",
                  causal_source_peak_dn=30, causal_current_center_xy=[64, 64],
                  causal_offset_is_predeclared_not_estimated_from_current=True,
                  runtime=dict(python=platform.python_version(), numpy=np.__version__))
    write_json(output/"freeze.json", freeze)
    cases = build_cases()
    if [c["case_id"] for c in cases] != [c["case_id"] for c in manifest["scenarios"]]:
        raise ValueError("Synthetic matrix differs from predeclared manifest")
    input_hashes = {}
    for case in cases:
        path = output/"inputs"/(case["case_id"]+".npz")
        np.savez_compressed(path, **{k: case[k] for k in ARRAY_KEYS},
                            history_images=case["synthetic_observations"]["history_images"],
                            current_image=case["synthetic_observations"]["current_image"])
        input_hashes[str(path)] = sha(path)
    for probe in probes:
        path = output/"inputs"/("causal_"+probe["case_id"]+".npz")
        np.savez_compressed(path, current129=probe["current129"], history129=probe["history129"],
                            prior_centers_xy=np.asarray([[np.nan, np.nan] if c is None else c for c in probe["prior_centers_xy"]]),
                            predicted_offset_xy=probe["predicted_offset_xy"])
        input_hashes[str(path)] = sha(path)
    write_json(output/"inputs_complete.json", dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
                                                   inputs_sha256=input_hashes, scores_started=False))
    require_hashes(bindings)
    require_hashes(input_hashes)
    write_json(output/"score_start.json", dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
                    inputs_complete_sha256=sha(output/"inputs_complete.json"),
                    freeze_sha256=sha(output/"freeze.json"), all_inputs_rehashed=True))
    records = []
    for case in cases:
        result = source_contrast(**{key: case[key] for key in ARRAY_KEYS})
        records.append(dict(case_id=case["case_id"], contrast=result, truth=case["truth"],
                            integration=case["integration"], temporal_provenance=case["temporal_provenance"],
                            temporal_metadata_used_by_numerical_solver=False,
                            motion_status="unknown", physical_class="unknown"))
    causal = [dict(case_id=p["case_id"], result=evaluate_causal_probe(
                    **{key: value for key, value in p.items() if key != "case_id"})) for p in probes]
    summary = summarize(records, causal)
    write_json(output/"oracle_results.json", records)
    write_json(output/"causal_results.json", causal)
    write_json(output/"summary.json", summary)
    require_hashes(bindings)
    require_hashes(input_hashes)
    bound = dict(bindings, **input_hashes)
    bound.update({str(path): sha(path) for path in output.glob("*.json")})
    write_json(output/"completion_receipt.json", dict(completed=True, created_at_utc=datetime.now(timezone.utc).isoformat(),
                                                      files_sha256=bound, production_changed=False, real_media_read=False))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    arguments = parser.parse_args()
    if not arguments.execute:
        parser.error("--execute is required to freeze and run the synthetic-only matrix")
    print(json.dumps(run(arguments.output), indent=2))
