"""Read-only, cache-only two-clip shadow evidence; never a production gate.

The explicit synthetic-readiness receipt is required before any real packet is
hashed or loaded. Old completion receipts are metadata, NOT recursive authority
to open videos, journals, other clips or arbitrary files. Only the predeclared
0029/0126 packet names in the pinned V43 manifest may be read. This runner does
not decode media, read journals, relabel references or modify frozen experiments.
"""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import platform
import re
import threading

import numpy as np

from accuracy_v45_causal_probe import ARMS, evaluate_causal_probe


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "results/tiny_target/accuracy_v46_20260925"
CACHE = ROOT / "results/tiny_target/accuracy_v43_20260925/stability_01"
INVENTORY = ROOT / "results/tiny_target/accuracy_v42_20260925/reference_inventory.json"
PINNED = {
    "receipt": "b3ae0fa68296f12e2493bc87419da861e06f5c45f7d4349f1e8396bdabad5042",
    "manifest": "619997859738ab81eecc1aa4031292a39e83dacc2cb2eb066ac89cbf3f0e9f80",
    "inventory": "444df960282569ef4c60a80a15bc6a50756fe0eadba421969061f6fb5aab86df",
}
CLIPS = ("0029", "0126")
EXPECTED_COUNTS = {"0029": (741, 350), "0126": (470, 159)}
EXPECTED_PANELS = {"dense": 285, "pilot": 28, "anchor": 24, "compact_light": 8, "grid": 10}
ARRAY_SHAPES = {"current129": (129, 129), "history129": (8, 129, 129),
                "prior_centers_xy": (8, 2), "predicted_offset_xy": (2,)}
SAMPLE_COLUMNS = ["panel", "clip_id", "window_id", "frame_index", "source_xy",
                  "position_uncertainty_px", "polarity", "saved_evidence_index", "stages"]
MATCHING_FIELDS = ("learned_design_sha256", "common_support_sha256", "common_support_count",
                   "ambiguity_reasons", "prior_context", "components")


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    with Path(path).open() as stream:
        return json.load(stream, parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def require_hashes(bindings):
    for path, expected in bindings.items():
        if sha(path) != expected:
            raise ValueError("Bound file changed: " + str(path))


def merge_bindings(bindings, additions):
    for path, digest in additions.items():
        if path in bindings and bindings[path] != digest:
            raise ValueError("Conflicting frozen dependency digest: " + path)
        bindings[path] = digest


def state_key(state):
    return state["clip"], state["frame_index"], state["segment"], state["track_id"]


def _regular_exact(path):
    """Do not let symlink resolution broaden any explicit read allowlist."""
    path = Path(path)
    if not path.is_absolute() or path.resolve() != path or not path.is_file():
        raise ValueError("Expected an existing nonsymlink canonical file: " + str(path))
    return path


def stress_binding_allowed(path, stress_run):
    path = Path(path)
    if not path.is_absolute() or path.resolve() != path:
        return False
    for folder, suffix in ((ROOT/"scripts", ".py"), (ROOT/"tests/unit", ".py"), (ROOT/"docs", ".md")):
        if path.parent == folder and path.suffix == suffix:
            return True
    synthetic_runs = (ROOT/"results/tiny_target/accuracy_v44_20260925/synthetic_01",
                      ROOT/"results/tiny_target/accuracy_v45_20260925/synthetic_01", stress_run)
    return any(folder in path.parents for folder in synthetic_runs)


def require_stress_readiness(stress_run):
    stress_run = Path(stress_run).resolve()
    if stress_run != OUTPUT_ROOT/"synthetic_01":
        raise ValueError("Only the explicit V46 synthetic_01 stress run is permitted")
    readiness_path = _regular_exact(OUTPUT_ROOT/"shadow_readiness_01.json")
    readiness = read_json(readiness_path)
    for flag in ("completed", "audited", "real_shadow_readiness_passed", "diagnostic_shadow_allowed"):
        if readiness.get(flag) is not True:
            raise ValueError("Synthetic readiness did not authorize shadow: " + flag)
    if (readiness.get("stress_run") != str(stress_run)
            or readiness.get("permitted_clips") != list(CLIPS)
            or readiness.get("production_changed") is not False):
        raise ValueError("Synthetic readiness scope differs")
    receipt_path = _regular_exact(stress_run/"completion_receipt.json")
    audit_path = _regular_exact(OUTPUT_ROOT/"synthetic_independent_audit_01.json")
    if readiness.get("synthetic_audit_path") != str(audit_path):
        raise ValueError("Unexpected synthetic audit path")
    bindings = {str(readiness_path): sha(readiness_path),
                str(receipt_path): readiness["stress_completion_receipt_sha256"],
                str(audit_path): readiness["synthetic_audit_sha256"]}
    require_hashes(bindings)
    receipt, audit = read_json(receipt_path), read_json(audit_path)
    if receipt.get("completed") is not True or audit.get("passed") is not True or audit.get("issues") != []:
        raise ValueError("Stress completion/audit is not clean")
    if audit.get("stress_completion_receipt_sha256") != bindings[str(receipt_path)]:
        raise ValueError("Synthetic audit does not bind the authorized stress receipt")
    dependencies = receipt.get("files_sha256")
    if not isinstance(dependencies, dict) or not dependencies:
        raise ValueError("Stress receipt must bind its inputs/code/results")
    for name, digest in dependencies.items():
        if not stress_binding_allowed(name, stress_run):
            raise ValueError("Stress receipt path outside synthetic/code allowlist: " + name)
        _regular_exact(name)
        merge_bindings(bindings, {name:digest})
    require_hashes(bindings)
    return bindings


def packet_path(state, cache=CACHE):
    """Validate literal identity-derived packet name before touching its bytes."""
    if state.get("clip") not in CLIPS:
        raise ValueError("Packet clip outside explicit allowlist")
    for field in ("frame_index", "segment"):
        if type(state.get(field)) is not int or state[field] < 0:
            raise ValueError("Invalid state integer")
    if not re.fullmatch(r"(?:bright|dark):[0-9]+", state.get("track_id", "")):
        raise ValueError("Invalid original track identity")
    archive = state["archive"]
    expected = (f'inputs/{state["clip"]}_{state["frame_index"]:04d}_{state["segment"]}_'
                f'{state["track_id"].replace(":", "_")}.npz')
    if not isinstance(archive, dict) or archive.get("path") != expected:
        raise ValueError("Packet path must exactly match allowed state identity")
    if not re.fullmatch(r"[0-9a-f]{64}", archive.get("sha256", "")):
        raise ValueError("Invalid packet digest")
    path = Path(cache)/PurePosixPath(expected)
    return _regular_exact(path)


def validate_scope(manifest, inventory):
    selected = [deepcopy(s) for s in manifest["states"] if s["clip"] in CLIPS]
    keys = [state_key(s) for s in selected]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate selected state")
    for clip, expected in EXPECTED_COUNTS.items():
        states = [s for s in selected if s["clip"] == clip]
        if (len(states), sum(s["archive"] is not None for s in states)) != expected:
            raise ValueError("Frozen state/packet denominator changed: " + clip)
    for s in selected:
        geometry = s["geometry"]
        if type(s["qualified_moving"]) is not bool:
            raise ValueError("Malformed original strict status")
        if (geometry.get("available") is True) != (s["archive"] is not None):
            raise ValueError("Geometry/cache availability differs")
        if s["archive"] is None:
            if geometry["reasons"] != ["fewer_than_five_prior_actual_same_id_measurements"]:
                raise ValueError("Unexpected frozen geometry reason")
        else:
            g = geometry["geometry"]
            if (g["current_target_fields_accessed"] is not False
                    or g["current_global_transform_uses_current_whole_frame"] is not True
                    or g["measured_prior_count"] < 5
                    or g["prior_frame_indices"] != list(range(s["frame_index"]-8, s["frame_index"]))):
                raise ValueError("Frozen causal geometry contract differs")
    if inventory["sample_columns"] != SAMPLE_COLUMNS:
        raise ValueError("Reference schema changed")
    samples = [dict(sample_index=i, original=dict(zip(SAMPLE_COLUMNS, deepcopy(row))))
               for i, row in enumerate(inventory["samples"]) if row[1] in CLIPS]
    if dict(Counter(s["original"]["panel"] for s in samples)) != EXPECTED_PANELS:
        raise ValueError("Reference denominators changed")
    lookup = dict(zip(keys, selected))
    expected_membership = {key: [] for key in keys}
    for sample in samples:
        original = sample["original"]
        stages = original["stages"]
        identities = stages["actual_measurement"][2]
        if len(identities) != len(set(identities)):
            raise ValueError("Duplicated reference alternative")
        for identity in identities:
            segment, track = identity.split("/", 1)
            key = (original["clip_id"], original["frame_index"], int(segment), track)
            if key not in lookup:
                raise ValueError("Original reference alternative lost")
            expected_membership[key].append(sample["sample_index"])
        strict = stages[inventory["panels"][original["panel"]]["original_strict_stage"]]
        if strict[1] is not None and strict[1] not in identities:
            raise ValueError("Strict assignment absent from original alternatives")
        if bool(strict[2]) != strict[0] or any(identity not in identities for identity in strict[2]):
            raise ValueError("Original strict alternatives differ from measured alternatives")
        for identity in strict[2]:
            segment, track = identity.split("/", 1)
            key = (original["clip_id"], original["frame_index"], int(segment), track)
            if lookup[key]["qualified_moving"] is not True:
                raise ValueError("Strict reference alternative lost original qualification")
        if bool(identities) != stages["actual_measurement"][0]:
            raise ValueError("Actual hit status disagrees with original alternatives")
        actual_assignment = stages["actual_measurement"][1]
        if (actual_assignment is None) != (not identities) or (identities and actual_assignment not in identities):
            raise ValueError("Original actual assignment differs from measured alternatives")
    for key, state in lookup.items():
        if state["reference_samples"] != expected_membership[key]:
            raise ValueError("Reference-to-state membership differs")
    miss = [s for s in samples if s["original"]["panel"] == "dense"
            and s["original"]["clip_id"] == "0126" and s["original"]["frame_index"] == 216]
    if len(miss) != 1 or miss[0]["original"]["stages"]["actual_measurement"][0] is not False:
        raise ValueError("Original 0126 frame216 miss must remain")
    return selected, samples


def load_scope_metadata():
    receipt_path, manifest_path = CACHE/"completion_receipt.json", CACHE/"cache_manifest.json"
    bindings = {str(_regular_exact(receipt_path)): PINNED["receipt"],
                str(_regular_exact(manifest_path)): PINNED["manifest"],
                str(_regular_exact(INVENTORY)): PINNED["inventory"]}
    require_hashes(bindings)
    receipt, manifest, inventory = (read_json(p) for p in (receipt_path, manifest_path, INVENTORY))
    if receipt.get("completed") is not True:
        raise ValueError("V43 cache did not complete")
    for path in (manifest_path, INVENTORY):
        if receipt["files_sha256"].get(str(path)) != bindings[str(path)]:
            raise ValueError("Compact provenance hash chain differs")
    for field in ("completed_before_any_arm_score", "all_geometry_exactly_reproduced",
                  "all_seven_v42_saved_inputs_exactly_reproduced"):
        if manifest.get(field) is not True:
            raise ValueError("Missing frozen cache provenance: " + field)
    states, samples = validate_scope(manifest, inventory)
    # Do NOT traverse/re-hash the V43 receipt: it also binds videos and journals.
    return states, samples, inventory["panels"], bindings, receipt["files_sha256"]


def load_packet(path, state):
    # Decode precisely the bytes whose digest was checked, avoiding a separate
    # hash/open race. This does not introduce a second real-data access route:
    # run calls it only after readiness and literal packet-path validation.
    payload = _regular_exact(path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != state["archive"]["sha256"]:
        raise ValueError("Packet bytes changed before decode")
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        if set(archive.files) != set(ARRAY_SHAPES):
            raise ValueError("Unexpected packet members")
        packet = {key: archive[key] for key in ARRAY_SHAPES}
    for key, shape in ARRAY_SHAPES.items():
        value = packet[key]
        if value.shape != shape or value.dtype != np.float64 or np.isinf(value).any():
            raise ValueError("Malformed frozen packet: " + key)
        value.setflags(write=False)
    centers = packet["prior_centers_xy"]
    if np.any(np.isfinite(centers).any(axis=1) != np.isfinite(centers).all(axis=1)):
        raise ValueError("Partial prior coordinate")
    if not np.isfinite(packet["predicted_offset_xy"]).all():
        raise ValueError("Nonfinite forecast offset")
    g = state["geometry"]["geometry"]
    expected_centers = np.asarray([[np.nan, np.nan] if p is None else p for p in g["prior_centers_xy"]])
    if (not np.array_equal(centers, expected_centers, equal_nan=True)
            or not np.array_equal(packet["predicted_offset_xy"], g["predicted_offset_xy"])):
        raise ValueError("Packet geometry differs from frozen metadata")
    return packet


def packet_fingerprint(packet):
    return {key: hashlib.sha256(value.tobytes()).hexdigest() for key, value in packet.items()}


def real_origin_copy(value):
    """Correct only inherited origin metadata; preserve every evidence field."""
    result = deepcopy(value)
    raw = result["raw_adapter_result"]
    if raw.get("synthetic_only") is not True:
        raise ValueError("Frozen adapter origin schema differs")
    raw["synthetic_only"] = False
    result.update(synthetic_only=False, input_origin="frozen_local_8bit_avi_derived_v43_packet",
                  legacy_adapter_reuse={
                      "unchanged_implementation": "accuracy_v45_causal_probe.evaluate_causal_probe",
                      "origin_only_overrides": [{"path": "raw_adapter_result.synthetic_only", "from": True, "to": False}],
                      "numeric_design_support_and_ambiguity_fields_preserved": True})
    return result


def validate_evidence(value):
    """Fail closed on malformed reporting, without changing numerical evidence."""
    raw = value["raw_adapter_result"]
    for layer, record in (("wrapper", value), ("adapter", raw)):
        if (record.get("motion_status") != "unknown" or record.get("physical_class") != "unknown"
                or record.get("is_motion_or_classification_gate") is not False
                or record.get("production_changed", False) is not False):
            raise ValueError("Shadow " + layer + " became production/classification")
    if raw.get("production_changed") is not False or type(raw.get("available")) is not bool:
        raise ValueError("Invalid adapter availability/production status")
    contrast = raw["numerical_contrast"]
    if contrast is None:
        if raw["available"]:
            raise ValueError("Available adapter lacks operative numerical evidence")
        return
    if (contrast.get("motion_status") != "unknown" or contrast.get("physical_class") != "unknown"
            or contrast.get("diagnostics", {}).get("no_production_gate") is not True
            or contrast.get("is_motion_or_classification_gate", False) is not False
            or contrast.get("production_changed", False) is not False):
        raise ValueError("Shadow numerical subrecord became production/classification")
    if type(contrast.get("available")) is not bool or contrast["available"] != raw["available"]:
        raise ValueError("Adapter/numerical availability disagrees")
    if not contrast["available"]:
        if any(contrast.get(key) is not None for key in (
                "interval", "coefficient_sign", "interval_excludes_zero", "estimate", "numerator", "error_bound")):
            raise ValueError("Unavailable numerical evidence contains an operative interval or sign")
        return
    interval = contrast.get("interval")
    if (not isinstance(interval, (list, tuple)) or len(interval) != 2
            or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.integer, np.floating))
                   or not np.isfinite(v) for v in interval)):
        raise ValueError("Available numerical interval must contain two finite real endpoints")
    low, high = interval
    if low > high:
        raise ValueError("Available numerical interval endpoints are reversed")
    sign = "positive" if low > 0 else "negative" if high < 0 else "unresolved"
    if (contrast.get("coefficient_sign") != sign
            or contrast.get("interval_excludes_zero") is not (sign != "unresolved")):
        raise ValueError("Numerical interval, sign and zero-exclusion status disagree")


def evaluate_packet(packet, state):
    before = packet_fingerprint(packet)
    centers = [p.tolist() if np.isfinite(p).all() else None for p in packet["prior_centers_xy"]]
    values = {}
    for arm in ARMS:
        # Deliberately no actual_source_xy, reference, clip, frame or truth argument.
        values[arm] = real_origin_copy(evaluate_causal_probe(
            packet["current129"], packet["history129"], centers,
            packet["predicted_offset_xy"], state["track_id"].split(":")[0], arm=arm))
        if packet_fingerprint(packet) != before:
            raise ValueError("Adapter mutated frozen in-memory input")
        validate_evidence(values[arm])
    original = values["amplitude_old_bounds"]["raw_adapter_result"]
    for value in values.values():
        raw = value["raw_adapter_result"]
        for key in MATCHING_FIELDS:
            if raw[key] != original[key]:
                raise ValueError("Arm changed nominal design/support/ambiguity: " + key)
    return values


def evidence_summary(value):
    if value is None:
        return dict(available=False, geometry_unavailable=True, interval=None, coefficient_sign=None)
    validate_evidence(value)
    raw = value["raw_adapter_result"]
    contrast = raw["numerical_contrast"]
    return dict(available=raw["available"], geometry_unavailable=False,
                reasons=raw["reasons"], ambiguity_reasons=raw["ambiguity_reasons"],
                interval=None if contrast is None else contrast["interval"],
                coefficient_sign=None if contrast is None else contrast["coefficient_sign"])


def reference_report(samples, panels, records):
    lookup = {state_key(r): r for r in records}
    report = []
    for item in samples:
        sample = deepcopy(item)
        original = sample["original"]
        alternatives = []
        for identity in original["stages"]["actual_measurement"][2]:
            segment, track = identity.split("/", 1)
            key = (original["clip_id"], original["frame_index"], int(segment), track)
            record = lookup[key]
            alternatives.append(dict(identity=identity, state_key=list(key),
                original_qualified_moving=record["qualified_moving"],
                geometry_available=record["geometry"]["available"],
                arm_evidence={arm:evidence_summary(record["arms"][arm]) for arm in ARMS}))
        strict_stage = panels[original["panel"]]["original_strict_stage"]
        assignment = original["stages"][strict_stage][1]
        assigned = [a for a in alternatives if a["identity"] == assignment]
        if assignment is not None and len(assigned) != 1:
            raise ValueError("Original strict assignment lost")
        sample.update(measured_alternatives=alternatives, original_strict_stage=strict_stage,
                      original_strict_assigned_identity=assignment,
                      original_strict_assigned_evidence=None if not assigned else assigned[0]["arm_evidence"])
        report.append(sample)
    return dict(samples=report, overlapping_samples_not_independent=True,
                original_assignments_alternatives_and_misses_preserved=True,
                no_new_labels=True, no_best_alternative_selection=True,
                references_establish_visible_features_not_airborne_class=True)


def summarize(records, states, references):
    if len(records) != len(states) or [state_key(r) for r in records] != [state_key(s) for s in states]:
        raise ValueError("State denominator/order changed")
    for record, state in zip(records, states):
        if {k:record[k] for k in state} != state:
            raise ValueError("Frozen state metadata changed")
        if set(record["arms"]) != set(ARMS):
            raise ValueError("An arm was omitted")
    by_arm = {}
    for arm in ARMS:
        counts = Counter(states=len(records), geometry_available=0, available=0,
                         unavailable=0, positive=0, negative=0, unresolved=0)
        for record in records:
            counts["geometry_available"] += int(record["geometry"]["available"])
            value = evidence_summary(record["arms"][arm])
            if not value["available"]:
                counts["unavailable"] += 1
            else:
                counts["available"] += 1
                if value["coefficient_sign"] not in ("positive", "negative", "unresolved"):
                    raise ValueError("Unknown available interval sign")
                counts[value["coefficient_sign"]] += 1
        by_arm[arm] = dict(counts)
    return dict(completed=True, synthetic_only=False, states=len(records),
        original_strict_states=sum(r["qualified_moving"] for r in records),
        packets=sum(r["archive"] is not None for r in records),
        reference_samples=len(references["samples"]), arms=by_arm,
        production_changed=False, detections_added=0, detections_removed=0,
        no_filter_applied=True, no_new_labels=True, no_historical_v45_real_baseline_claim=True,
        original_production_measurements_and_assignments_preserved=True,
        numerator_and_amplitude_are_different_quantities=True,
        no_airborne_accuracy_or_false_alarm_claim=True,
        current_global_transform_uses_current_whole_frame=True,
        forecast_causality_inherited_from_frozen_provenance_not_reverified_from_journals=True,
        packet_only_no_media_decode_or_journal_read=True)


def dependency_paths():
    paths = [ROOT/"docs/accuracy_v46_plan.md", ROOT/"scripts/accuracy_v42_localized.py",
             ROOT/"scripts/accuracy_v43_components.py", ROOT/"scripts/accuracy_v43_bounds.py"]
    for version in (44, 45, 46):
        paths += list((ROOT/"scripts").glob(f"*accuracy_v{version}*.py"))
        paths += list((ROOT/"tests/unit").glob(f"test_accuracy_v{version}*.py"))
    return sorted(set(paths))


class Progress:
    """A bounded diagnostic may be slow; emit status without blocking its work."""
    def __init__(self):
        self.message = "checking synthetic readiness"
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.stop.wait(30):
            print("V46 shadow: " + self.message, flush=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *unused):
        self.stop.set()
        self.thread.join(timeout=1)


def run(output, stress_run, *, execute=False):
    if not execute:
        raise ValueError("Explicit --execute and successful synthetic readiness are required")
    output = Path(output).resolve()
    if output.parent != OUTPUT_ROOT or output == OUTPUT_ROOT/"synthetic_01":
        raise ValueError("Output must be a fresh dedicated V46 child, not synthetic_01")
    if output.exists():
        raise FileExistsError("Frozen output cannot be reused")
    with Progress() as progress:
        bindings = require_stress_readiness(stress_run)
        states, samples, panels, provenance, cache_receipt = load_scope_metadata()
        merge_bindings(bindings, provenance)
        merge_bindings(bindings, {str(p):sha(_regular_exact(p)) for p in dependency_paths()})
        packets = {}
        for state in states:
            if state["archive"] is not None:
                path = packet_path(state, CACHE)
                digest = state["archive"]["sha256"]
                if cache_receipt.get(str(path)) != digest:
                    raise ValueError("Packet digest not bound by pinned cache receipt")
                packets[str(path)] = digest
        if len(packets) != sum(count[1] for count in EXPECTED_COUNTS.values()):
            raise ValueError("Exact packet denominator changed")
        # First real packet byte access is here, AFTER the strict readiness gate.
        progress.message = "hashing all allowlisted packets before any score"
        require_hashes(packets)
        require_hashes(bindings)
        output.mkdir(parents=True, exist_ok=False)
        write_json(output/"selected_ledger.json", dict(states=states, reference_samples=samples,
                                                       inherited_reference_panels=panels))
        write_json(output/"freeze.json", dict(created_at_utc=now(),
            compact_and_code_dependencies_sha256=bindings, packet_sha256=packets,
            selected_ledger_sha256=sha(output/"selected_ledger.json"),
            allowed_clips=list(CLIPS), arms={k:list(v) for k,v in ARMS.items()},
            all_selected_packets_hashed_before_any_score=True, synthetic_only=False,
            production_changed=False, no_historical_v45_real_baseline_claim=True,
            current_global_transform_uses_current_whole_frame=True,
            runtime=dict(python=platform.python_version(), numpy=np.__version__)))
        write_json(output/"score_start.json", dict(started_at_utc=now(),
            freeze_sha256=sha(output/"freeze.json"), selected_ledger_sha256=sha(output/"selected_ledger.json")))
        records = []
        current_key = None
        try:
            with (output/"states.jsonl").open("x") as stream:
                for index, state in enumerate(states):
                    current_key = state_key(state)
                    progress.message = f"state {index+1}/{len(states)}; key {current_key}"
                    record = deepcopy(state)
                    record["arms"] = {arm:None for arm in ARMS}
                    if state["archive"] is not None:
                        path = packet_path(state, CACHE)
                        require_hashes({str(path):packets[str(path)]})
                        record["arms"] = evaluate_packet(load_packet(path, state), state)
                    stream.write(json.dumps(record, allow_nan=False)+"\n")
                    stream.flush()
                    records.append(record)
            references = reference_report(samples, panels, records)
            summary = summarize(records, states, references)
            write_json(output/"reference_evidence.json", references)
            write_json(output/"summary.json", summary)
            progress.message = "rehashing all bound files and packets after scoring"
            require_hashes(bindings)
            require_hashes(packets)
            final = dict(bindings, **packets)
            for name in ("selected_ledger.json", "freeze.json", "score_start.json", "states.jsonl",
                         "reference_evidence.json", "summary.json"):
                final[str(output/name)] = sha(output/name)
            write_json(output/"completion_receipt.json", dict(completed=True, files_sha256=final,
                completed_at_utc=now(), production_changed=False, synthetic_only=False,
                packet_only_no_media_decode_or_journal_read=True))
        except Exception as exc:
            write_json(output/"failure.json", dict(completed=False, state_key=current_key,
                completed_records=len(records), exception_type=type(exc).__name__, message=str(exc),
                partial_records_are_not_a_completed_evaluation=True))
            raise
        print(f'V46 shadow complete: {len(records)} states; {len(packets)} packets; no filter applied', flush=True)
        return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stress-run", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        parser.error("--execute is required; synthetic readiness must already be approved")
    print(json.dumps(run(args.output, args.stress_run, execute=args.execute), indent=2))
