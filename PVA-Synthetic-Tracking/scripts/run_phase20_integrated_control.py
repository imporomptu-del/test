"""Isolate CUDA execution from PVA motion on the authorized chunk0029.

This diagnostic changes only the stabilization execution backend to its CPU
cubic reference. It never changes detection/tracking policy or reads labels.
"""
import itertools
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tiny_target.visible_baseline import VisibleConfig, sha256


def write_new(path, value):
    with path.open("x") as handle:
        json.dump(value, handle, indent=2)


def main():
    frozen = json.loads((ROOT / "freeze.json").read_text())
    source = frozen["sources"]["0029"]
    if source["path"] != "/home/serg/project/camera_reader_sky/srcsky/chunks/chunk_0029.avi":
        raise ValueError("Control source outside authorized scope")
    for name, expected in frozen["files_sha256"].items():
        if sha256(ROOT / name) != expected:
            raise ValueError("Frozen runtime changed: " + name)
    if sha256(source["path"]) != source["sha256"]:
        raise ValueError("Control source changed")
    status = json.loads((ROOT / "status.json").read_text())
    if status["running"] or status["error"] or len(status["completed"]) != 6:
        raise ValueError("Original batch must complete before the control")
    library_hash = sha256(ROOT / "libseaqr_integrated.so")
    if library_hash != status["library_sha256"]:
        raise ValueError("Compiled library changed")
    baseline = ROOT / "full_0029"
    report = json.loads((baseline / "report.json").read_text())
    if not report["completed"] or not report["full_clip"] or report["frames"] != source["frames"]:
        raise ValueError("Integrated reference incomplete")
    output = ROOT / "control_0029"
    if output.exists():
        raise ValueError("Never overwrite a control")
    config = json.loads((ROOT / "resident_config.json").read_text())
    if config["stabilization_execution"] != "cuda_cubic_resident":
        raise ValueError("Unexpected integrated mode")
    config["stabilization_execution"] = "reference"
    VisibleConfig(**config)
    config_path = ROOT / "control_config.json"
    write_new(config_path, config)
    write_new(ROOT / "control_freeze.json", dict(
        source=source,
        config_sha256=sha256(config_path),
        parent_freeze_sha256=sha256(ROOT / "freeze.json"),
        integrated_journal_sha256=sha256(baseline / "frames.jsonl"),
        library_sha256=library_hash,
        script_sha256=sha256(__file__),
        only_configuration_change={"stabilization_execution": ["cuda_cubic_resident", "reference"]},
        labels_supplied=False,
        purpose="Separate PVA camera-motion/tracking behavior from the new CUDA execution; no tuning."))
    started = time.time()
    command = [sys.executable, "-m", "tiny_target.visible_baseline", "--source", source["path"],
               "--config", str(config_path), "--motion-config", str(ROOT / "configs/evaluation/phase20_motion_v8.json"),
               "--output", str(output)]
    with (ROOT / "control_0029.log").open("x") as log:
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=10800)
    control = json.loads((output / "report.json").read_text())
    if not control["completed"] or not control["full_clip"] or control["frames"] != source["frames"]:
        raise ValueError("Control incomplete")
    differences = []
    compared = 0
    with (baseline / "frames.jsonl").open() as first, (output / "frames.jsonl").open() as second:
        for index, (left, right) in enumerate(itertools.zip_longest(first, second)):
            if left is None or right is None:
                raise ValueError("Different journal lengths")
            left, right = json.loads(left), json.loads(right)
            for key in ("frame_index", "timestamp_ns", "segment", "source_to_reference", "candidates", "tracks", "tracking_metrics"):
                if left[key] != right[key]:
                    differences.append([index, key])
            left_coverage, right_coverage = dict(left["coverage"]), dict(right["coverage"])
            left_coverage.pop("detection_ms")
            right_coverage.pop("detection_ms")
            if left_coverage != right_coverage:
                differences.append([index, "coverage"])
            for row in (left, right):
                if row["motion"]["pva_failure"] or row["motion"]["reset"]:
                    raise ValueError("PVA failure/reset in control comparison")
                if index:
                    backends = row["motion"]["motion_backends"]
                    if backends.get("cpu_fallback") is not False or any(
                        backends.get(k) != "PVA" for k in ("gaussian_pyramid", "harris", "optical_flow_pyrlk")
                    ):
                        raise ValueError("Actual PVA provenance missing")
            compared += 1
    result = dict(exact=not differences, frames=compared, difference_count=len(differences),
                  first_differences=differences[:20], actual_pva_pairs_per_run=compared - 1,
                  control_fps=control["processed_fps"], integrated_fps=report["processed_fps"],
                  reference_journal_sha256=sha256(output / "frames.jsonl"),
                  integrated_journal_sha256=sha256(baseline / "frames.jsonl"),
                  control_freeze_sha256=sha256(ROOT / "control_freeze.json"),
                  started_unix=started, finished_unix=time.time(),
                  accuracy_policy_changed=False, generalization_proven=False)
    write_new(ROOT / "control_comparison.json", result)
    print(json.dumps(result, indent=2), flush=True)
    if differences:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
