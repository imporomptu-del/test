#!/usr/bin/env python3
"""Metadata-only decision for the bounded reverse-status ownership experiment.

Reuses one hash-pinned, accelerator-free analysis module. Never imports VPI,
opens media, estimates motion, refits a model, or tunes a quality threshold.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import re

import numpy as np

SCHEMA = "seaqr.pva-reverse-ownership.summary.v1"
HELPER_SHA = "c9a1ca5414b41b92176d69deaf5817550a3eaac63b0f6f048bb398342fba9708"
CASES = ("texture__shift_two", "texture__shift_fractional", "texture__shift_half",
         "out_of_range_shift4", "bridge__shift_4_m2", "flat")
ROLES = (("plain", "base"), ("shared", "base"), ("shared", "trace"),
         ("copied", "base"), ("copied", "trace"))
FAILED = CASES[:2]
POSITIVE_CONTROLS = CASES[2:5]
SOURCES = {"probe_pva_reverse_ownership.py", "batch_pva_reverse_ownership.py", "summarize_pva_reverse_ownership.py",
           "test_pva_reverse_ownership.py", "test_batch_pva_reverse_ownership.py", "test_summarize_pva_reverse_ownership.py",
           "batch_discovery_pair.py"}
SAFETY_SHA = "a8c5ebe468f638af7ae39cf876e816a462c0fab1f3673add0eb81e037a8504bf"
REFERENCE = dict(workspace="/tmp/seaqr_pva_robustness_20261001_sDW19P",
    probe_sha256="9ccc3bb0c0d664a12421c348f06bfa810e1ea9f32a19590f514101e2a8446e79",
    freeze_sha256="6cad8a44b9aebf8ac0c335d436727a4a008bc683f82174a2bd69ee2495a322ad")
REFERENCE_SOURCES = {"batch_discovery_pair.py": SAFETY_SHA,
    "batch_pva_robustness.py": "432a92d868e9f121059bc9bfc0be6bf254573d89ae5938c051d8ccb08d1f3de0",
    "probe_pva_robustness.py": REFERENCE["probe_sha256"], "summarize_pva_robustness.py": HELPER_SHA,
    "test_batch_pva_robustness.py": "b6986b834e0e64c5a9f9a65976ad0c470169a8517a3aab0fe4370dcf48ed2769",
    "test_pva_robustness.py": "42da2bf0d4e0d917f01cd8df971c6fdc2b61956324689711f43f42407f108b67",
    "test_summarize_pva_robustness.py": "6c179839c16f87126cf0f1fd4e81553020542214f5d3e3e60df505083e371ff5"}
EXPECTED_MOTION = dict(feature_intensity_mapping="bit_shift", optical_flow_backend="PVA",
    harris_capacity_policy="complete_grid", flow_status_policy="legacy_default", feature_cpu_policy="batched_exact_v1",
    feature_image_scale=.5, max_features=384, grid_rows=6, grid_cols=8, max_features_per_cell=8,
    feature_border_px=16., saturation_fraction=.995, saturated_neighborhood_radius_px=2, exclusion_regions_xyxy=[],
    harris_strength=.5, harris_sensitivity=.0625, harris_gradient_size=3, harris_block_size=3, harris_min_nms_distance=8,
    pyramid_levels=2, pyramid_scale=.5, pyramid_backend="auto", window_size=11, max_iterations=6,
    forward_backward_check=True, max_forward_backward_error_px=3., max_displacement_px=120.,
    minimum_accepted_features=30, minimum_grid_coverage=.2)


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def load_helper(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and sha(path) == HELPER_SHA,
            "Pinned metadata helper differs")
    spec = importlib.util.spec_from_file_location("reverse_immutable_metadata_helper", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inventory(helper):
    original = {case["id"]: case for case in helper.inventory()}
    return [original[name] for name in CASES]


def mask_counterfactual(helper, row):
    """Replace only post-backward forward flags with saved immediate flags.

    All actual reverse endpoints/statuses and original numeric tests remain.
    This tests mask attribution only, not whether another reverse run succeeds.
    """
    capture = row["capture"]
    if capture.get("partial"):
        return dict(available=False, recovered_count=None, removed_count=None,
                    reason="No returned correspondence; no counterfactual denominator", fits_performed=0)
    data = capture["data"]
    final = data["final_filter"]
    p, q, back = (helper.array(final[key]) for key in
                  ("previous_full_points", "current_full_points", "backward_full_points"))
    status = helper.array(data["after_forward"]["forward_status"]).reshape(-1)
    back_status = helper.array(data["after_backward"]["backward_status_vpi"]).reshape(-1)
    old = helper.array(final["accepted_mask"])
    require(p.dtype == q.dtype == back.dtype == np.dtype("float32") and p.shape == q.shape == back.shape
            and status.shape == back_status.shape == old.shape == (len(p),), "Counterfactual arrays differ")
    config = row["canonical_nontiming"]["effective_motion_configuration"]
    finite = np.isfinite(p).all(axis=1) & np.isfinite(q).all(axis=1)
    bounds = (q[:, 0] >= 0) & (q[:, 0] < 640) & (q[:, 1] >= 0) & (q[:, 1] < 512)
    displacement = np.linalg.norm(q-p, axis=1)
    fb = np.linalg.norm(back-p, axis=1).astype(np.float32)
    eligible = finite & bounds & (displacement <= config["max_displacement_px"])
    eligible &= np.isfinite(back).all(axis=1) & np.isfinite(fb) & (back_status == 0) & (fb <= config["max_forward_backward_error_px"])
    actual_forward = helper.array(data["after_backward"]["forward_status"]).reshape(-1)
    require(actual_forward.shape == status.shape and old.dtype == np.dtype("bool")
            and np.array_equal(old, eligible & (actual_forward == 0)), "Saved acceptance differs from original safe formula")
    new = eligible & (status == 0)
    return dict(available=True, recovered_count=int((new & ~old).sum()), removed_count=int((old & ~new).sum()),
        same_accepted_mask=bool(np.array_equal(old, new)), backward_valid_but_forward_lost_count=int(((back_status == 0) & (status != 0)).sum()),
        recovered_selected_slots=np.flatnonzero(new & ~old).tolist(), removed_selected_slots=np.flatnonzero(old & ~new).tolist(),
        original_backward_outputs_held_fixed=True, original_thresholds_held_fixed=True, fits_performed=0,
        does_not_claim_alternative_reverse_flow_recovery=True)


def preintervention_identity(helper, shared, copied):
    for field in ("case_metadata", "input_pair", "frame_pixel_sha256", "effective_motion_configuration", "global_configuration"):
        require(shared["canonical_nontiming"][field] == copied["canonical_nontiming"][field],
                "Preintervention canonical identity differs: "+field)
    require(shared["canonical_nontiming"]["effective_motion_configuration"]["pyramid_levels"] == 2,
            "Frozen depth2 required")
    a, b = shared["capture"]["data"], copied["capture"]["data"]
    for direction in ("previous", "current"):
        left, right = a["pyramids"][direction], b["pyramids"][direction]
        require(len(left) == len(right) == 2, "Two full prepared pyramid levels required")
        for index in range(2):
            require(helper.same_array(left[index]["array"], right[index]["array"]), "Full pyramid bytes differ")
            actual = helper.array(left[index]["array"])
            require(actual.dtype == np.dtype("uint8") and actual.shape == ((256,320),(128,160))[index],
                    "Prepared pyramid representation differs")
        require(helper.same_array(a["pyramids"]["proxy"][direction]["array"], b["pyramids"]["proxy"][direction]["array"]),
                "Motion proxy bytes differ")
    partial = shared["capture"].get("partial", False)
    require(partial == copied["capture"].get("partial", False), "Feature availability differs before intervention")
    if partial:
        return dict(passed=True, pixels_configuration_pyramids_exact=True, features_available=False,
                    selected_seeds_exact=None, immediate_forward_points_and_status_exact=None)
    for field in ("selected_points", "selected_scores", "selected_indices"):
        require(helper.same_array(a["before_forward"][field], b["before_forward"][field]), "Selected seed differs: "+field)
    for field in ("tracked_points", "forward_status"):
        require(helper.same_array(a["after_forward"][field], b["after_forward"][field]), "Immediate forward output differs: "+field)
    return dict(passed=True, pixels_configuration_pyramids_exact=True, features_available=True,
                selected_seeds_exact=True, immediate_forward_points_and_status_exact=True)


def changes(helper, shared, copied):
    """Separate changed rejection labels from changed actual measurements/model."""
    a, b = shared["canonical_nontiming"], copied["canonical_nontiming"]
    ca, cb = a["result"]["correspondence"], b["result"]["correspondence"]
    if ca is None or cb is None:
        require(ca is None and cb is None, "Availability differs before reverse constructor")
        return dict(returned_correspondences_exact=True, original_global_result_exact=True,
                    reverse_outputs_exact=None, only_rejection_classification_changed=False, scientific_measurements_changed=False)
    returned = all(helper.same_array(ca["arrays"][field], cb["arrays"][field]) for field in
                   ("previous_points", "current_points", "harris_scores", "forward_backward_error_px"))
    global_same = all(a["result"][field] == b["result"][field] for field in ("fit", "inlier_mask", "residuals_px"))
    da, db = shared["capture"]["data"], copied["capture"]["data"]
    reverse_same = all(helper.same_array(da["after_backward"][field], db["after_backward"][field])
                       for field in ("backward_points_vpi", "backward_status_vpi"))
    stripped_a, stripped_b = copy.deepcopy(a), copy.deepcopy(b)
    for value in (stripped_a, stripped_b):
        value["result"]["correspondence"]["metrics"].pop("rejections", None)
    only_labels = a != b and stripped_a == stripped_b
    return dict(returned_correspondences_exact=returned, original_global_result_exact=global_same,
        reverse_outputs_exact=reverse_same, only_rejection_classification_changed=only_labels,
        scientific_measurements_changed=not (returned and global_same), canonical_exact=a == b)


def case_gates(case, analysis):
    fit = analysis["original_fit"]
    if case["id"] == "flat":
        return dict(passed=fit["accepted"] is not True, kind="negative_no_accepted_fit",
                    fit_ran=fit.get("fit_ran"), global_accepted=fit["accepted"])
    good = fit["accepted"] is True and fit["translation_error_px"] is not None and fit["translation_error_px"] <= .25
    return dict(passed=bool(good and analysis["point_guard_passed"]), kind="recovery" if case["id"] in FAILED else "positive_control",
                global_accepted=fit["accepted"], translation_error_px=fit["translation_error_px"],
                global_truth_passed=bool(good), nonvacuous_interior_point_guard_passed=analysis["point_guard_passed"])


def packet(helper, item):
    value = helper.array(item["array"])
    meta = item["metadata"]
    require(isinstance(meta.get("type"), str) and type(meta.get("id")) is int and meta["id"] > 0
            and type(meta.get("size")) is int and meta["size"] == len(value)
            and type(meta.get("capacity")) is int and meta["capacity"] >= meta["size"], "Invalid active VPI packet metadata")
    return value, meta


def constructor_check(helper, row):
    audit = row["constructor_audit"]
    arm = row["arm"]
    require(audit.get("arm") == arm and audit.get("enabled") is (arm != "plain"), "Constructor facade role differs")
    returned = row["canonical_nontiming"]["result"]["correspondence"] is not None
    expected_calls = 2 if returned and arm != "plain" else 0
    require(audit.get("constructor_calls") == expected_calls, "Unexpected constructor-call inventory")
    if expected_calls == 0:
        require(audit.get("reverse") is None, "Unexpected reverse constructor for no-flow/plain case")
        return dict(executed=False, expected_constructor_calls=0)
    forward, reverse, after = audit["forward"], audit["reverse"], audit["after"]
    require(forward.get("explicit_kptstatus") is False and forward.get("keyword_names") == ["backend"],
            "Forward initialization changed")
    require(reverse.get("keyword_names") == ["backend", "kptstatus"] and str(reverse.get("backend")).endswith("PVA"),
            "Reverse constructor keywords/backend differ")
    for field in ("active_status_bytes_equal", "size_capacity_type_equal", "cloned_in_both_arms", "nonzero_flags_preserved"):
        require(reverse.get(field) is True, "Clone identity assertion failed: "+field)
    require(reverse.get("constructor_context_changed") is False and after.get("clone_retained") is True
            and audit.get("clone_retained_until_estimator_close") is True, "Clone lifetime or stream context differs")
    source, sm = packet(helper, reverse["source_status_before"])
    clone, cm = packet(helper, reverse["clone_status_before"])
    require(source.dtype == clone.dtype == np.dtype("uint8") and source.shape == clone.shape
            and helper.same_array(reverse["source_status_before"]["array"], reverse["clone_status_before"]["array"])
            and sm["id"] != cm["id"] and all(sm[key] == cm[key] for key in ("size", "capacity", "type")),
            "Clone must be separately owned with equal active bytes and representation")
    _, km = packet(helper, reverse["keypoints_before"])
    require(km["size"] == sm["size"], "Reverse point/status size differs")
    choice = "clone" if arm == "copied" else "source"
    require(reverse.get("chosen_status") == choice and reverse.get("actual_argument_is_forward_status") is (arm == "shared")
            and reverse.get("actual_argument_is_clone") is (arm == "copied")
            and reverse.get("actual_argument_metadata") == (cm if arm == "copied" else sm), "Actual reverse status argument differs")
    _, sa = packet(helper, after["source_status"])
    _, ca = packet(helper, after["clone_status"])
    require(sa == sm and ca == cm, "Retained status size/type/identity changed")
    unused = "source_status" if arm == "copied" else "clone_status"
    before_unused = reverse[unused+"_before"]
    require(helper.same_array(after[unused]["array"], before_unused["array"]), "Unused separately owned status was mutated")
    if row["mode"] == "trace":
        data = row["capture"]["data"]
        require(data["after_backward"].get("estimator_local_backward_initial_status_is_forward_status") is True
            and data["after_backward"].get("actual_backward_argument_is_forward_status") is (arm == "shared")
            and data["after_backward"].get("actual_backward_argument_is_clone") is (arm == "copied"),
            "Actual versus estimator-local reverse argument identity differs")
        require(helper.same_array(reverse["source_status_before"]["array"], data["after_forward"]["forward_status"])
                and helper.same_array(reverse["keypoints_before"]["array"], data["after_forward"]["tracked_points"]),
                "Constructor inputs differ from immediate forward outputs")
        require(helper.same_array(after["source_status"]["array"], data["after_backward"]["forward_status"])
                and helper.same_array(after[choice+"_status"]["array"], data["after_backward"]["backward_status_vpi"]),
                "Actual selected constructor status differs from returned status")
        require(helper.same_array(after["keypoints"]["array"], data["after_backward"]["tracked_points"]),
                "Forward endpoint lifetime differs")
        for direction, recorded in (("previous", forward["pyramid"]), ("current", reverse["pyramid"])):
            require(len(recorded) == 2 and all(helper.same_array(item["array"], data["pyramids"][direction][i]["array"])
                    for i, item in enumerate(recorded)), "Constructor pyramid differs from prepared trace")
    return dict(executed=True, expected_constructor_calls=2, independent_clone_identity=True,
                same_initial_active_flags=True, same_size_capacity_type=True, same_allocation_schedule=True,
                chosen_status=choice, forward_initialization_and_constructor_context_unchanged=True,
                original_nonzero_status_count=int(np.count_nonzero(source)))


def matched_constructor_schedule(helper, shared, copied):
    a, b = shared["constructor_audit"], copied["constructor_audit"]
    require(a["constructor_calls"] == b["constructor_calls"], "Matched constructor counts differ")
    if a["constructor_calls"] == 0:
        return dict(executed=False, compared=False)
    for field in ("keyword_names", "backend", "constructor_context_changed"):
        require(a["reverse"][field] == b["reverse"][field], "Reverse constructor nonidentity option differs")
    for field in ("source_status_before", "clone_status_before", "keypoints_before"):
        left, right = a["reverse"][field], b["reverse"][field]
        require(helper.same_array(left["array"], right["array"])
                and all(left["metadata"][key] == right["metadata"][key] for key in ("size", "capacity", "type")),
                "Preintervention constructor bytes/representation differ: "+field)
    require(helper.same_array(a["forward"]["keypoints"]["array"], b["forward"]["keypoints"]["array"]),
            "Forward constructor seeds differ")
    for call in ("forward", "reverse"):
        require(len(a[call]["pyramid"]) == len(b[call]["pyramid"]) == 2 and all(
            helper.same_array(x["array"], y["array"]) for x, y in zip(a[call]["pyramid"], b[call]["pyramid"])),
            "Constructor pyramid bytes differ across ownership arms")
    return dict(executed=True, compared=True, initial_flags_points_representation_and_pyramids_exact=True,
                sole_declared_reverse_argument_difference="source status identity versus separately owned clone")


def validate_freeze(helper, frozen):
    expected = dict(schema="pva_reverse_ownership.v1", scope="generated_only_no_production_change",
        cases=inventory(helper), depth=2, roles=[dict(arm=a, mode=m) for a,m in ROLES], shape_hw=[512,640],
        safety=dict(start_c=65,stop_c=75,phase_seconds=900,batch_seconds=3600,workers=1,automatic_retries=0),
        reference=REFERENCE, gates=dict(fit_translation_error_px=.25, point_error_px=.25, truth_interior_margin_px=128),
        protocol=dict(fresh_processes=30,focal_estimate_calls_per_child=1,preflight_pva_calls=0,
            original_estimator_source_unchanged=True,forward_status_initialization_unchanged=True,
            constructor_stream_context_unchanged=True,all_motion_quality_gates_unchanged=True,
            shared_copied_difference="actual reverse kptstatus identity only; both allocate/write/read/retain same-sized copy",
            status_copy="preserve every active U8 byte, size and capacity; no zero reset; inactive capacity bytes not claimed equal",
            matched_shared_must_equal_fresh_plain_canonical=True,shared_copied_each_base_trace_exact=True,
            before_reverse_input_equality_required=True, recover_cases=list(FAILED), preserve_cases=list(POSITIVE_CONTROLS),
            negative_case="flat",positive_guard="original fit accepted, translation error<=.25, >=1 fixed-interior accepted point and max error<=.25",
            unsuccessful_branch="stop without retries or tuning",production_promotion=False))
    require(set(frozen) == set(expected)|{"source_sha256"}
            and helper.canonical_sha({key:frozen[key] for key in expected}) == helper.canonical_sha(expected),
            "Frozen reverse ownership experiment differs")
    sources = frozen["source_sha256"]
    require(set(sources) == SOURCES and sources["batch_discovery_pair.py"] == SAFETY_SHA
            and all(isinstance(value,str) and re.fullmatch(r"[a-f0-9]{64}",value) for value in sources.values()),
            "Frozen source inventory differs")


def validate_child(helper, row, case, arm, mode, digest, frozen):
    require(row.get("schema") == "seaqr.pva-reverse-ownership.v1" and row.get("case") == case["id"]
        and row.get("case_metadata") == case and row.get("arm") == arm and row.get("mode") == mode
        and row.get("depth") == 2, "Child role differs")
    for key,expected in dict(completed=True,passed_integrity=True,generated_only=True,source_media_accessed=False,
        detector_run=False,production_changes=False,production_promotion=False,preflight_pva_calls=0,
        focal_estimate_calls=1,closed=True,clocks_changed=False).items():
        require(type(row.get(key)) is type(expected) and row[key] == expected, "Child scope/lifecycle differs: "+key)
    require(row.get("error") is None and row.get("cleanup_error") is None, "Child execution error")
    inputs = row.get("input_sha256", {})
    require(inputs.get("freeze_sha256") == digest and inputs.get("source_sha256") == frozen["source_sha256"]
        and inputs.get("reference") == REFERENCE and inputs.get("reference_source_sha256") == REFERENCE_SOURCES,
        "Child frozen dependencies differ")
    canonical = row.get("canonical_nontiming", {})
    require(canonical.get("case_metadata") == case and helper.canonical_sha(canonical) == row.get("canonical_nontiming_sha256")
        and canonical.get("effective_motion_configuration") == EXPECTED_MOTION
        and canonical.get("global_configuration") == helper.EXPECTED_GLOBAL, "Canonical identity or original gates differ")
    require(row.get("global_fits") == (0 if canonical["result"]["correspondence"] is None else 1),
            "Original fit call count differs")
    observation = dict(method_sha256="bfe680841a214acdddb16c3f12bc166921760613c3446a0a6c2ca8ac37e7286b",
        estimator_source_changed=False,sys_settrace=mode=="trace",forward_initializer_changed=False,
        constructor_context_changed=False,reverse_status_object_substitution=arm=="copied",
        matched_constructor_readbacks_and_copy=arm!="plain",extra_pva_compute_calls=0)
    require(row.get("observation_contract") == observation, "Observation/intervention contract differs")
    pair = canonical.get("input_pair", {})
    require(pair.get("native_shape_hw") == [512,640] and pair.get("source_media_accessed") is False,
            "Generated native input differs")
    for field in ("previous_pixel_sha256","current_pixel_sha256","original_current_pixel_sha256"):
        require(isinstance(pair.get(field),str) and re.fullmatch(r"[a-f0-9]{64}",pair[field]), "Invalid generated pixel identity")
    quant = pair.get("quantization", {})
    require(quant.get("clipped_pixels") == 0 and "current_maximum_abs_rounding_error_dn" in quant
        and all(type(value) in (int,float) and np.isfinite(value) and 0 <= value <= .5
                for key,value in quant.items() if key.endswith("rounding_error_dn")), "Quantization evidence differs")
    frame = canonical.get("frame_pixel_sha256", {})
    require(set(frame) == {"previous","current"} and all(isinstance(v,str) and re.fullmatch(r"[a-f0-9]{64}",v)
            for v in frame.values()), "Frame pixel identities differ")
    require((row.get("capture") is not None) is (mode == "trace"), "Trace mode/capture differs")
    return constructor_check(helper,row)


def phase_name(case, arm, mode):
    return case+"_"+arm+"_"+mode


def decision(cases, identity_failures):
    failures = [x["case"]["id"] for x in cases if not x["copied"]["gates"]["passed"]]
    changed = [x["case"]["id"] for x in cases if x.get("changes") and x["changes"]["scientific_measurements_changed"]]
    if identity_failures:
        outcome = "inconclusive_stop"
    elif not changed:
        outcome = "no_effect_stop"
    elif failures:
        outcome = "insufficient_recovery_stop"
    else:
        outcome = "candidate_supported_for_full_generated_regression_only"
    return dict(outcome=outcome, stop_ownership_branch=outcome != "candidate_supported_for_full_generated_regression_only",
        copied_failed_case_ids=failures, changed_measurement_case_ids=changed,
        ready_for_full_generated_regression=outcome == "candidate_supported_for_full_generated_regression_only",
        production_promotion=False, tuning_authorized=False)


def summarize(directory, freeze_sha256, bundle, robustness_summary):
    directory,bundle = Path(directory).absolute(),Path(bundle).absolute()
    helper_path = Path(robustness_summary).absolute()
    helper = load_helper(helper_path)
    require(directory.is_dir() and bundle.is_dir() and not directory.is_symlink() and not bundle.is_symlink(),
            "Real metadata and bundle directories required")
    require(isinstance(freeze_sha256,str) and re.fullmatch(r"[a-f0-9]{64}",freeze_sha256), "Caller-bound freeze required")
    bindings = {str(helper_path):HELPER_SHA}
    def bound(path, expected=None):
        digest = sha(path)
        require(expected is None or digest == expected, "Evidence hash differs: "+path.name)
        value = helper.read(path)
        require(sha(path) == digest, "Evidence changed during read: "+path.name)
        bindings[str(path)] = digest
        return value
    frozen = bound(directory/"freeze.json",freeze_sha256)
    validate_freeze(helper,frozen)
    for name,digest in frozen["source_sha256"].items():
        path=bundle/name
        require(path.is_file() and not path.is_symlink() and sha(path)==digest,"Frozen source differs: "+name)
        bindings[str(path)]=digest
    require(sha(Path(__file__)) == frozen["source_sha256"]["summarize_pva_reverse_ownership.py"], "Running summary differs from freeze")
    batch = bound(directory/"batch_status.json")
    require(batch.get("schema") == "seaqr.pva-reverse-ownership.batch.v1" and batch.get("complete") is True
        and batch.get("execution_passed") is True and batch.get("generated_only") is True
        and batch.get("camera_media_accessed") is False and batch.get("clock_writes") is False
        and batch.get("error") is None and batch.get("not_run") == [] and batch.get("current") is None
        and batch.get("freeze_sha256") == freeze_sha256 and batch.get("source_sha256") == frozen["source_sha256"],
        "Complete generated batch required; no partial summary")
    phases = batch.get("phases",[])
    require([(p["case"],p["arm"],p["mode"]) for p in phases] == [(case,a,m) for case in CASES for a,m in ROLES]
        and len(phases)==30 and len({p["pid"] for p in phases})==30
        and all(type(p["pid"]) is int and p["pid"]>0 and p["returncode"]==0 for p in phases),
        "30 fresh completed processes required")
    require(batch.get("execution") == dict(workers=1,phases=30,phase_deadline_seconds=900,batch_deadline_seconds=3600,
        start_below_celsius=65,stop_at_celsius=75,automatic_retries=0)
        and 0 <= batch["elapsed_seconds"] < 3600 and all(0 <= p["elapsed_seconds"] < 900 for p in phases),
        "Frozen execution bounds differ")
    parity = bound(directory/"parity.json",batch["parity_sha256"])
    require(parity.get("schema") == "seaqr.pva-reverse-ownership.batch.v1.parity"
        and [(x["case"],x["arm"]) for x in parity.get("cases",[])] == [(c,a) for c in CASES for a in ("shared","copied")]
        and [x["case"] for x in parity.get("controls",[])] == list(CASES), "12 pairs and 6 plain controls required")
    rows,constructors = {},{}
    definitions={c["id"]:c for c in inventory(helper)}
    workspaces=set()
    for phase in phases:
        key=(phase["case"],phase["arm"],phase["mode"])
        require(phase["name"] == phase_name(*key),"Phase name differs")
        command=phase.get("command",[])
        require(isinstance(command,list) and len(command) in (13,14)
            and isinstance(command[4],str) and re.fullmatch(r"/tmp/seaqr_pva_reverse_20261001_[A-Za-z0-9]{6}",command[4]),
            "Scoped isolated command required")
        workspace=command[4];workspaces.add(workspace)
        expected=["/usr/bin/python3","-I",workspace+"/probe_pva_reverse_ownership.py","--workspace",workspace,
            "--freeze",workspace+"/freeze.json","--freeze-sha256",freeze_sha256,"--case",key[0],"--arm",key[1]]
        if key[2]=="trace":expected.append("--trace")
        require(command==expected,"Recorded command/role/freeze differs")
        row=bound(directory/(phase["name"]+".json"),phase["result_sha256"])
        constructors[key]=validate_child(helper,row,definitions[key[0]],key[1],key[2],freeze_sha256,frozen)
        rows[key]=row
    require(len(workspaces)==1,"Multiple execution workspaces")
    pairs,controls={},{}
    for item in parity["cases"]:
        key=item["case"],item["arm"]
        base,trace=(rows[key+(mode,)] for mode in ("base","trace"))
        same=base["canonical_nontiming"]==trace["canonical_nontiming"]
        require(same == (base["canonical_nontiming_sha256"]==trace["canonical_nontiming_sha256"])
            and item.get("passed") is same and item.get("exact_nontiming_parity") is same
            and item.get("baseline_sha256")==base["canonical_nontiming_sha256"] and item.get("trace_sha256")==trace["canonical_nontiming_sha256"]
            and item.get("timings_excluded") is True and item.get("trace_may_explain_baseline_only_if_passed") is same,
            "Saved pair parity differs")
        pairs[key]=same
    for item in parity["controls"]:
        case=item["case"]
        plain,shared=rows[case,"plain","base"],rows[case,"shared","base"]
        same=plain["canonical_nontiming"]==shared["canonical_nontiming"]
        require(same == (plain["canonical_nontiming_sha256"]==shared["canonical_nontiming_sha256"])
            and item.get("passed") is same and item.get("exact_nontiming_parity") is same
            and item.get("plain_sha256")==plain["canonical_nontiming_sha256"] and item.get("shared_sha256")==shared["canonical_nontiming_sha256"],
            "Saved plain/shared control differs")
        controls[case]=same
    require(batch.get("parity_passed") is all(pairs.values()) and parity.get("passed") is all(pairs.values())
        and batch.get("control_parity_passed") is all(controls.values()) and parity.get("controls_passed") is all(controls.values()),
        "Overall recorded parity differs")
    cases,failures=[],[]
    for case in inventory(helper):
        name=case["id"]
        item=dict(case=case,plain_shared_exact=controls[name],preintervention_identity=None,matched_constructor_schedule=None,
                  changes=None,matched_populations=None,shared_mask_counterfactual=None,copied_mask_counterfactual=None)
        internals={}
        for arm in ("shared","copied"):
            row=rows[name,arm,"base"];trace=rows[name,arm,"trace"]
            analysis,accepted=helper.accepted_analysis(row["canonical_nontiming"],case)
            analysis.update(gates=case_gates(case,analysis),exact_base_trace_parity=pairs[name,arm],availability=None)
            if pairs[name,arm]:
                analysis["availability"],internals[arm]=helper.capture_analysis(trace,case,accepted)
            else:failures.append(dict(case=name,reason=arm+" base/trace canonical mismatch"))
            item[arm]=analysis
        if not controls[name]:failures.append(dict(case=name,reason="Matched shared allocation/readback schedule changed fresh plain output"))
        if controls[name] and all(pairs[name,arm] for arm in ("shared","copied")):
            try:
                shared,copied=rows[name,"shared","trace"],rows[name,"copied","trace"]
                item["preintervention_identity"]=preintervention_identity(helper,shared,copied)
                item["matched_constructor_schedule"]=matched_constructor_schedule(helper,shared,copied)
                item["matched_populations"]=helper.matched_populations(internals["shared"],internals["copied"])
                item["changes"]=changes(helper,shared,copied)
                item["shared_mask_counterfactual"]=mask_counterfactual(helper,shared)
                item["copied_mask_counterfactual"]=mask_counterfactual(helper,copied)
            except ValueError as exc:failures.append(dict(case=name,reason=str(exc)))
        if name in FAILED and item["shared"]["gates"]["passed"]:
            failures.append(dict(case=name,reason="Frozen failed baseline did not reproduce; no recovery attribution"))
        if name in POSITIVE_CONTROLS+("flat",) and not item["shared"]["gates"]["passed"]:
            failures.append(dict(case=name,reason="Frozen positive/negative baseline control did not reproduce; no recovery attribution"))
        cases.append(item)
    require(all(sha(Path(path))==digest for path,digest in bindings.items()),"Inputs changed during metadata analysis")
    return dict(schema=SCHEMA,completed=True,metadata_only=True,source_media_accessed=False,numerical_refits=0,
        production_changes=False,production_promotion=False,freeze_sha256=freeze_sha256,input_sha256=bindings,
        children_completed=30,cases_completed=6,base_trace_pairs_checked=12,plain_shared_controls_checked=6,
        all_base_trace_parity_passed=all(pairs.values()),all_plain_shared_parity_passed=all(controls.values()),
        identity_failures=failures,decision=decision(cases,failures),cases=cases,
        limitations=["One generated run per role is not an estimate of reliability, video accuracy, FPR or throughput.",
            "Active CPU flags and exposed representations are observed; inactive capacity bytes and internal VPI/device state are not.",
            "Only a changed result with both recoveries and all controls passing justifies the full generated regression; never production promotion.",
            "The status-mask counterfactual holds original backward outputs fixed and performs no model fit.",
            "Unknown/lost endpoints are unavailable, not zero-error measurements; common survivors are a selected population.",
            "Earlier failed controls remain immutable evidence; this report does not supersede them."])


def save_summary(directory, freeze_sha256, bundle, robustness_summary, output):
    output=Path(output).absolute()
    require(output.suffix==".json" and output.parent.is_dir() and output.parent.resolve()==output.parent
        and not output.exists() and not output.is_symlink(),"Fresh JSON output required; no overwrite")
    value=summarize(directory,freeze_sha256,bundle,robustness_summary)
    with output.open("x") as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write("\n")
    return value


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory",required=True,type=Path);parser.add_argument("--freeze-sha256",required=True)
    parser.add_argument("--bundle",required=True,type=Path);parser.add_argument("--robustness-summary",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args()
    result=save_summary(args.directory,args.freeze_sha256,args.bundle,args.robustness_summary,args.output)
    print(json.dumps({key:result[key] for key in ("completed","children_completed","identity_failures","decision")}))
