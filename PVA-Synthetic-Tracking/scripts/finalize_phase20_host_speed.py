"""Verify frozen execution-only evidence and preserve separate accuracy limits."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tiny_target.visible_baseline import sha256
from compare_phase20_exact_runs import compare, shape_accelerator, validate_decode, transition_for_pair


def verify_package(root, frozen, job):
    package = {n.removeprefix("tiny_target/"): v for n, v in frozen["files_sha256"].items()
               if n.startswith("tiny_target/")}
    after = root / job["name"]
    launch = json.loads((after / "launch.json").read_text())
    validate_decode(launch, json.loads((after / 'report.json').read_text()))
    if launch["package_sha256"] != package or launch["config_sha256"] != frozen["config_sha256"]:
        raise ValueError("Runtime does not match freeze")
    if launch["source_sha256"] != frozen["sources"][job["clip_id"]]["sha256"]:
        raise ValueError("Source changed")
    if launch["exact_cuda_stabilization"]["library_sha256"] != frozen["compiled_library_sha256"]:
        raise ValueError("GPU library changed")
    if 'gpu_transition' in frozen:
        transition=frozen['gpu_transition'];build_path=root/'libseaqr_integrated.so.build.json'
        gpu_build=json.loads(build_path.read_text())
        if (sha256(root/'libseaqr_integrated.so')!=transition['after_library_sha256']
                or sha256(build_path)!=transition['candidate_build_sha256']
                or gpu_build['library_sha256']!=transition['after_library_sha256']
                or gpu_build['reference_library_sha256']!=transition['before_library_sha256']
                or transition['after_library_sha256']!=frozen['compiled_library_sha256']
                or gpu_build['schema']!='seaqr.cuda-peak-gate-build.v1'
                or gpu_build['diagnostic_only'] is not False
                or gpu_build['variant']!='threshold_before_local_max'
                or gpu_build['algorithm_policy_changed'] is not False):
            raise ValueError('GPU transition binary/build mismatch')
        for name,expected in gpu_build['sources_sha256'].items():
            if expected!=frozen['files_sha256'].get('scripts/'+name):
                raise ValueError('GPU build source not frozen')
    native = shape_accelerator(launch)
    build = frozen.get('native_shape_build')
    if (native is None) != (build is None):
        raise ValueError('Native shape freeze/provenance mismatch')
    if native is not None:
        if (native['library_sha256'] != build['library_sha256']
                or sha256(root/'libseaqr_shapes.so') != build['library_sha256']
                or sha256(root/'libseaqr_shapes.so.build.json') != frozen['native_shape_build_sha256']
                or build != json.loads((root/'libseaqr_shapes.so.build.json').read_text())
                or build['source_sha256'] != frozen['files_sha256']['scripts/phase20_native_shapes.cpp']
                or build['builder_sha256'] != frozen['files_sha256']['scripts/build_phase20_native_shapes.py']):
            raise ValueError('Native shape binary/build changed')
    for relative, digest in package.items():
        if sha256(after / "implementation" / relative) != digest:
            raise ValueError("Implementation snapshot changed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--timing-reference", type=Path,
        help="Optional completed execution-only predecessor, separate from the fixed accuracy reference")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    frozen = json.loads((args.root / "freeze.json").read_text())
    status = json.loads((args.root / "status.json").read_text())
    if status["running"] or status["error"] or [r["name"] for r in status["completed"]] != [j["name"] for j in frozen["jobs"]]:
        raise ValueError("Speed batch incomplete")
    if set(frozen["sources"]) != {"0029", "0126", "0055", "0082"}:
        raise ValueError("Unexpected cohort")
    if sha256(args.root / "config.json") != frozen["config_sha256"]:
        raise ValueError("Configuration changed")
    if sha256(args.reference / "freeze.json") != frozen["parent_freeze_sha256"]:
        raise ValueError("Reference freeze changed")
    accuracy = json.loads((args.reference / "full_audit.json").read_text())
    if not accuracy["complete"] or accuracy["freeze_sha256"] != frozen["parent_freeze_sha256"]:
        raise ValueError("Missing reference accuracy audit")
    timing_reference = args.timing_reference or args.reference
    timing_freeze = json.loads((timing_reference / "freeze.json").read_text())
    timing_status = json.loads((timing_reference / "status.json").read_text())
    if timing_status["running"] or timing_status["error"] or timing_freeze["sources"] != frozen["sources"]:
        raise ValueError("Timing reference incomplete or source cohort changed")
    if [r["name"] for r in timing_status["completed"]] != [j["name"] for j in frozen["jobs"]]:
        raise ValueError("Timing reference jobs changed")
    if sha256(timing_reference / "config.json") != timing_freeze["config_sha256"]:
        raise ValueError("Timing reference configuration changed")
    comparisons, accuracy_comparisons = [], []
    for job in frozen["jobs"]:
        name, cid = job["name"], job["clip_id"]
        before, after = args.reference / name, args.root / name
        verify_package(args.root, frozen, job)
        verify_package(timing_reference, timing_freeze, job)
        for filename, expected in frozen["execution_only_reference"]["artifacts_sha256"][name].items():
            if sha256(before / filename) != expected:
                raise ValueError("Frozen reference artifact changed")
        remote = json.loads((after / "comparison.json").read_text())
        if not remote["exact"] or remote["output_sha256"] != sha256(after / "frames.jsonl"):
            raise ValueError("Remote numerical gate missing or stale")
        if args.timing_reference:
            accuracy_checked = compare(before, after, args.output / (name + "_accuracy_comparison.json"),
                gpu_transition=frozen.get('gpu_transition'))
            if accuracy_checked["frames"] != frozen["sources"][cid]["frames"]:
                raise ValueError("Wrong accuracy comparison frame count")
            accuracy_comparisons.append(dict(clip_id=cid, **accuracy_checked))
        checked = compare(timing_reference / name, after, args.output / (name + "_local_comparison.json"),
            gpu_transition=transition_for_pair(
                json.loads((timing_reference / name / 'launch.json').read_text()),
                json.loads((after / 'launch.json').read_text()), frozen.get('gpu_transition')))
        if checked["frames"] != frozen["sources"][cid]["frames"]:
            raise ValueError("Wrong frame count")
        comparisons.append(dict(clip_id=cid, **checked))
    total_frames = sum(v["frames"] for v in comparisons)
    before_seconds = sum(v["frames"] / v["before_fps"] for v in comparisons)
    after_seconds = sum(v["frames"] / v["after_fps"] for v in comparisons)
    result = dict(passed_execution_equivalence=True, full_clip_frames=total_frames,
        comparisons=comparisons, aggregate_before_fps=total_frames / before_seconds,
        aggregate_after_fps=total_frames / after_seconds, aggregate_speedup=before_seconds / after_seconds,
        accuracy_comparisons=accuracy_comparisons or comparisons,
        timing_reference=dict(path=str(timing_reference), freeze_sha256=sha256(timing_reference / "freeze.json")),
        accuracy=dict(unchanged_from_reference_by_exact_journal_comparison=True,
            reference_audit_sha256=sha256(args.reference / "full_audit.json"),
            strict_dense_sample_continuity_gate=accuracy["strict_dense_sample_continuity_gate"],
            source="Frozen development samples, not a verified airborne population",
            labels_used_during_processing=False, airborne_precision=None, airborne_recall=None,
            full_frame_false_alarm_rate=None),
        production_ready=False, holdouts_accessed=False,
        timing_caveat="One matching sequential full-clip run per implementation; not a repeated thermal/load-controlled distribution. Timing excludes input hashing and startup, includes per-frame decode and journaling.",
        freeze_sha256=sha256(args.root / "freeze.json"), script_sha256=sha256(__file__))
    with (args.output / "summary.json").open("x") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k not in ("comparisons", "accuracy_comparisons")}, indent=2))


if __name__ == "__main__":
    main()
