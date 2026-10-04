#!/usr/bin/env python3
"""Summarize hash-bound generated-only evidence; never open media or rerun VPI."""
import argparse
import base64
import hashlib
import json
from pathlib import Path

import numpy as np


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def array(record):
    raw = base64.b64decode(record["data_base64"], validate=True)
    require(hashlib.sha256(raw).hexdigest() == record["sha256"], "Capture bytes differ")
    dtype = np.dtype(record["dtype"])
    require(dtype.kind in "uifb" and len(raw) <= 4*512*640, "Unexpected numerical capture")
    return np.frombuffer(raw, dtype=dtype).reshape(record["shape"])


def status(record):
    value = array(record).reshape(-1)
    require(value.dtype == np.uint8, "Status is not uint8")
    values, counts = np.unique(value, return_counts=True)
    return {str(int(v)): int(n) for v, n in zip(values, counts)}


def summarize(directory):
    directory = Path(directory)
    batch = read(directory/"batch_status.json")
    require(batch["schema"] == "seaqr.static-pva-texture.batch.v1"
            and batch["complete"] is True and batch["execution_passed"] is True
            and batch["generated_only"] is True and batch["camera_media_accessed"] is False
            and batch["clock_writes"] is False, "Incomplete or unexpected batch")
    require(sha(directory/"freeze.json") == batch["freeze_sha256"], "Freeze differs")
    freeze = read(directory/"freeze.json")
    require(freeze["schema"] == "static_pva_texture.v1" and freeze["cases"] == ["bridge", "texture"]
            and freeze["modes"] == ["base", "trace"] and freeze["files"] == batch["files"], "Freeze scope differs")
    phases = ["bridge_base", "bridge_trace", "texture_base", "texture_trace"]
    require([p["name"] for p in batch["phases"]] == phases and len({p["pid"] for p in batch["phases"]}) == 4,
            "Not four exact fresh processes")
    require(sha(directory/"parity.json") == batch["parity_sha256"], "Parity evidence differs")
    parity = read(directory/"parity.json")
    results = {}
    require(parity["schema"] == "seaqr.static-pva-texture.batch.v1.parity", "Parity schema differs")
    bindings = {"batch_status.json": sha(directory/"batch_status.json"), "parity.json": batch["parity_sha256"],
                "freeze.json": batch["freeze_sha256"]}
    for phase in batch["phases"]:
        filename = phase["name"]+".json"
        require(filename in {c+"_"+m+".json" for c in ("bridge", "texture") for m in ("base", "trace")}, "Unexpected phase")
        require(sha(directory/filename) == phase["result_sha256"], "Child evidence differs")
        bindings[filename] = phase["result_sha256"]
        result = read(directory/filename)
        case, mode = phase["name"].split("_")
        require(result["schema"] == "seaqr.static-pva-texture.v1" and result["completed"] is True
                and result["passed_integrity"] is True and result["case"] == case and result["mode"] == mode
                and result["generated_only"] is True and result["source_media_accessed"] is False
                and result["production_changes"] is False and result["detector_run"] is False
                and result["input_sha256"]["freeze_sha256"] == batch["freeze_sha256"]
                and result["input_sha256"]["files"] == freeze["files"], "Child contract differs")
        results[phase["name"]] = result
    cases = []
    for pair in parity["cases"]:
        name = pair["case"]
        require(name in ("bridge", "texture"), "Unexpected case")
        base, trace = results[name+"_base"], results[name+"_trace"]
        same = base["canonical_nontiming_sha256"] == trace["canonical_nontiming_sha256"]
        for result in (base, trace):
            digest = hashlib.sha256(json.dumps(result["canonical_nontiming"], sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
            require(digest == result["canonical_nontiming_sha256"], "Canonical bytes differ")
        require(same == pair["passed"], "Parity gate differs")
        captured = same and trace.get("capture") and not trace["capture"].get("partial", False)
        row = dict(case=name, exact_base_trace_parity=same, trace_interpretation_allowed=bool(captured),
                   baseline_accepted_points=base.get("accepted_points"), baseline_fit_accepted=base.get("scientific_fit_accepted"))
        canonical = base["canonical_nontiming"]
        if canonical.get("result", {}).get("correspondence") is not None:
            row["baseline_result"] = {"metrics": canonical["result"]["correspondence"]["metrics"], "fit": canonical["result"]["fit"]}
        else:
            row["baseline_unavailable_reason"] = canonical.get("result", {}).get("reason")
        if captured:
            capture = trace["capture"]
            data = capture["data"]
            count = data["before_forward"]["selected_count"]
            require(all(array(data[stage][field]).size == count for stage,field in (
                ("after_forward","forward_status"), ("after_backward","forward_status"),
                ("after_backward","backward_status_vpi"))), "Status count differs")
            row["observations"] = dict(
                selected_points=data["before_forward"]["selected_count"],
                forward_status_immediately_after_forward=status(data["after_forward"]["forward_status"]),
                forward_status_after_backward=status(data["after_backward"]["forward_status"]),
                backward_status=status(data["after_backward"]["backward_status_vpi"]),
                forward_status_changed_by_backward=capture["forward_status_bytes_changed_by_backward"],
                forward_points_changed_by_backward=capture["forward_point_bytes_changed_by_backward"],
                identities=data["after_backward"]["identities"],
                status_same_python_object=data["after_backward"]["status_same_python_object"],
                points_same_python_object=data["after_backward"]["points_same_python_object"],
                final_rejections=data["final_filter"]["rejection_counts"])
            levels = []
            require(len(data["pyramids"]["previous"]) == len(data["pyramids"]["current"]) == 4,
                    "Pyramid level count differs")
            for index, (p, q) in enumerate(zip(data["pyramids"]["previous"], data["pyramids"]["current"])):
                left, right = array(p["array"]), array(q["array"])
                require(left.dtype == right.dtype == np.uint8 and left.shape == right.shape
                        == ((256,320),(128,160),(64,80),(32,40))[index], "Pyramid representation differs")
                levels.append(dict(level=index, shape_hw=list(left.shape), identical_previous_current=np.array_equal(left,right),
                                   minimum=int(left.min()), maximum=int(left.max()), std_dn=float(left.std()), unique_values=int(len(np.unique(left))),
                                   horizontal_nonzero_differences=int(np.count_nonzero(np.diff(left.astype(np.int16),axis=1))),
                                   vertical_nonzero_differences=int(np.count_nonzero(np.diff(left.astype(np.int16),axis=0)))))
            row["observations"]["pyramids"] = levels
        cases.append(row)
    require(sorted(row["case"] for row in cases) == ["bridge", "texture"], "Case inventory differs")
    require(all(sha(directory/name) == expected for name,expected in bindings.items()), "Evidence changed during summary")
    return dict(schema="seaqr.static-pva-texture.summary.v1", completed=True, generated_only=True,
                production_changed=False, input_sha256=bindings, cases=cases,
                limitations=["Generated static controls do not establish real-video detection accuracy or throughput.",
                             "Initial legacy forward status is implicit and unobserved.",
                             "Readback parity covers these final outputs, not all possible allocator or scheduling states.",
                             "Pyramid gradients and failure flags are observations, not proof of an SDK defect."])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = summarize(args.directory)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, indent=2, allow_nan=False))
