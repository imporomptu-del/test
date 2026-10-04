"""Freeze host execution improvements with an unchanged four-clip policy."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import tarfile
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tiny_target.visible_baseline import VisibleConfig, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--reference-workspace", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument('--native-shape-build', type=Path,
        help='Explicit build record from the target machine; enables only compiled bookkeeping')
    parser.add_argument('--gpu-build', type=Path,
        help='Explicit tested kernel build; only an exact frozen binary transition is permitted')
    parser.add_argument('--frame-decode-execution', choices=['sequential', 'prefetch_one'],
        default='sequential', help='Execution only: bounded ordered decode overlap')
    args = parser.parse_args()
    old = json.loads((args.reference / "freeze.json").read_text())
    status = json.loads((args.reference / "status.json").read_text())
    if status["running"] or status["error"] or len(status["completed"]) != 4:
        raise ValueError("Reference batch must finish before benchmark freeze")
    if set(old["sources"]) != {"0029", "0126", "0055", "0082"}:
        raise ValueError("Unexpected source scope")
    cfg = json.loads((args.reference / "config.json").read_text())
    if sha256(args.reference / "config.json") != old["config_sha256"]:
        raise ValueError("Reference policy changed")
    cfg = asdict(VisibleConfig(**dict(cfg, cuda_median_library=args.workspace + "/libseaqr_integrated.so",
        frame_decode_execution=args.frame_decode_execution)))
    build = None
    gpu_build = None
    if args.gpu_build:
        gpu_build=json.loads(args.gpu_build.read_text())
        if (gpu_build['schema']!='seaqr.cuda-peak-gate-build.v1'
                or gpu_build['variant']!='threshold_before_local_max'
                or gpu_build['diagnostic_only'] is not False
                or gpu_build['algorithm_policy_changed'] is not False
                or gpu_build['reference_library_sha256']!=old['compiled_library_sha256']):
            raise ValueError('Unexpected GPU build/reference transition')
        for name,expected in gpu_build['sources_sha256'].items():
            if sha256(ROOT/'scripts'/name)!=expected:
                raise ValueError('GPU build source changed: '+name)
    if args.native_shape_build:
        build = json.loads(args.native_shape_build.read_text())
        if (build['source_sha256'] != sha256(ROOT/'scripts/phase20_native_shapes.cpp')
                or build['builder_sha256'] != sha256(ROOT/'scripts/build_phase20_native_shapes.py')
                or build['abi'] != 1 or build['no_fast_math'] is not True):
            raise ValueError('Native shape build does not match current sources/contract')
        cfg = asdict(VisibleConfig(**dict(cfg, native_shape_library=args.workspace+'/libseaqr_shapes.so',
            native_shape_library_sha256=build['library_sha256'])))
    args.output.mkdir(parents=True, exist_ok=False)
    config_path = args.output / "config.json"
    with config_path.open("x") as handle:
        json.dump(cfg, handle, indent=2)
    paths = sorted((ROOT / "tiny_target").rglob("*.py")) + [ROOT / name for name in (
        "configs/evaluation/phase20_motion_v8.json", "scripts/run_phase20_closed_loop_batch.py",
        "scripts/compare_phase20_exact_runs.py", "tests/unit/test_exact_quadratic.py",
        "tests/unit/test_kalman_tracking.py", "scripts/profile_phase20_host_speed.py",
        "tests/unit/test_detection_batching.py", "tests/unit/test_visible_learning.py",
        "scripts/verify_phase20_host_efficiency.py", "scripts/verify_phase20_detection_batch.py",
        "scripts/phase20_native_shapes.cpp", "scripts/build_phase20_native_shapes.py",
        "tests/unit/test_native_shapes.py", "scripts/verify_phase20_native_shapes.py",
        "tests/unit/test_visible_decode.py", "tests/unit/test_exact_run_comparison.py",
        "tests/unit/test_speed_finalization.py", "scripts/verify_phase20_decode_evidence.py",
        "tests/unit/test_decode_evidence.py")]
    if gpu_build is not None:
        paths += [ROOT/'scripts'/name for name in gpu_build['sources_sha256']]
        paths += [ROOT/'scripts'/name for name in ('repeat_phase20_kernel_speed.py','finalize_phase20_host_speed.py')]
        paths = sorted(set(paths))
    frozen = dict(schema="seaqr.frozen-closed-loop-development.v1", sources=old["sources"], jobs=old["jobs"],
        config_sha256=sha256(config_path), compiled_library_sha256=old["compiled_library_sha256"],
        files_sha256={str(p.relative_to(ROOT)): sha256(p) for p in paths},
        parent_freeze_sha256=sha256(args.reference / "freeze.json"), labels_used_during_processing=False,
        algorithm_change="None. Optional compiled bounded shape/patch bookkeeping; centroid reductions stay in NumPy reference order. Existing host optimizations retained.",
        settings_not_changed="All detection/tracking policy, sampling, cadence, coverage and caps.",
        execution_only_reference=dict(workspace=args.reference_workspace,
            artifacts_sha256={job["name"]: {name: sha256(args.reference / job["name"] / name)
                for name in ("launch.json", "report.json", "frames.jsonl")} for job in old["jobs"]}))
    if build is not None:
        frozen.update(native_shape_build=build, native_shape_build_sha256=sha256(args.native_shape_build))
    if gpu_build is not None:
        frozen.update(compiled_library_sha256=gpu_build['library_sha256'],
            gpu_transition=dict(schema='seaqr.exact-gpu-transition.v1',
                before_library_sha256=old['compiled_library_sha256'],
                after_library_sha256=gpu_build['library_sha256'],candidate_build_sha256=sha256(args.gpu_build)),
            algorithm_change='None. Only unchanged threshold predicates move before read-only local-maximum search. Native shape bookkeeping and all host policy remain fixed.')
    if args.frame_decode_execution == 'prefetch_one':
        frozen['algorithm_change'] = ('None. Ordered one-frame decode/grayscale prefetch only; '
            'motion/detection/tracking remain single-consumer and causal. '
            'Previously verified native-shape and peak-gate binaries retained unchanged.')
    freeze_path = args.output / "freeze.json"
    with freeze_path.open("x") as handle:
        json.dump(frozen, handle, indent=2)
    with tarfile.open(args.output / "runtime.tar.gz", "w:gz") as archive:
        for path in paths:
            archive.add(path, arcname=str(path.relative_to(ROOT)))
        for path in (config_path, freeze_path):
            archive.add(path, arcname=path.name)
    print(json.dumps(dict(workspace=args.workspace, archive_sha256=sha256(args.output / "runtime.tar.gz")), indent=2))


if __name__ == "__main__":
    main()
