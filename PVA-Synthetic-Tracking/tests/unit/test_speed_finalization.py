"""A faster timing reference must not replace or weaken the accuracy gate."""
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from tiny_target.visible_baseline import VisibleConfig, sha256

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("speed_finalize_test", ROOT / "scripts/finalize_phase20_host_speed.py")
finalizer = importlib.util.module_from_spec(spec)
with patch.object(sys, "path", [str(ROOT / "scripts"), *sys.path]):
    spec.loader.exec_module(finalizer)


def write(path, value):
    path.write_text(json.dumps(value))


def make_run(root, fps, parent=None):
    root.mkdir()
    config = asdict(VisibleConfig(spatial_background='median5', spatial_filter_backend='cuda_median5',
        cuda_median_library='/explicit/gpu.so', state_update_backend='cuda_resident',
        pixel_noise_enabled=True, pixel_noise_model='background_residual', shape_measurement_mode='mutual_half_height_r8'))
    write(root / "config.json", config)
    code = "# test fixture, not a media-derived implementation\n"
    code_hash = hashlib.sha256(code.encode()).hexdigest()
    sources = {cid: dict(path="synthetic_" + cid, sha256="source_" + cid, frames=1)
               for cid in ("0029", "0126", "0055", "0082")}
    jobs = [dict(name="pva_" + cid, clip_id=cid, max_frames=None) for cid in sources]
    frozen = dict(sources=sources, jobs=jobs, files_sha256={"tiny_target/example.py": code_hash},
        config_sha256=sha256(root / "config.json"), compiled_library_sha256="test_library")
    for job in jobs:
        target = root / job["name"]
        (target / "implementation").mkdir(parents=True)
        (target / "implementation/example.py").write_text(code)
        write(target / "launch.json", dict(source_sha256=sources[job["clip_id"]]["sha256"],
            fps=10, configuration=config, config_sha256=frozen["config_sha256"],
            package_sha256={"example.py": code_hash},
            exact_cuda_stabilization=dict(library_sha256="test_library")))
        write(target / "report.json", dict(completed=True, full_clip=True, frames=1,
            processed_fps=fps, timings_ms={}))
        (target / "frames.jsonl").write_text(json.dumps(dict(frame_index=0,
            candidates=[dict(x=20, y=30)], timings_ms=dict(detection=1000/fps))) + "\n")
        write(target / "comparison.json", dict(exact=True, output_sha256=sha256(target / "frames.jsonl")))
    write(root / "status.json", dict(running=False, error=None,
        completed=[dict(name=job["name"]) for job in jobs]))
    if parent:
        frozen["parent_freeze_sha256"] = sha256(parent / "freeze.json")
        frozen["execution_only_reference"] = dict(artifacts_sha256={job["name"]: {
            filename: sha256(parent / job["name"] / filename) for filename in (
                "launch.json", "report.json", "frames.jsonl")} for job in jobs})
    write(root / "freeze.json", frozen)
    return root


class SpeedFinalizationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.accuracy = make_run(self.root / "accuracy", 1.)
        write(self.accuracy / "full_audit.json", dict(complete=True,
            freeze_sha256=sha256(self.accuracy / "freeze.json"), strict_dense_sample_continuity_gate=False))
        self.timing = make_run(self.root / "timing", 2., self.accuracy)
        self.after = make_run(self.root / "after", 4., self.accuracy)
        self.output = self.root / "verified"

    def finalize(self):
        arguments = ["finalize", "--root", str(self.after), "--reference", str(self.accuracy),
            "--timing-reference", str(self.timing), "--output", str(self.output)]
        with patch("sys.argv", arguments), patch("builtins.print"):
            finalizer.main()

    def enable_native_fixture(self):
        # A provenance fixture only: this file is never loaded as native code.
        library = self.after/'libseaqr_shapes.so'
        library.write_bytes(b'native shape test fixture')
        build = dict(library_sha256=sha256(library), source_sha256='test_source', builder_sha256='test_builder',
            abi=1, no_fast_math=True)
        write(self.after/'libseaqr_shapes.so.build.json', build)
        cfg = json.loads((self.after/'config.json').read_text())
        cfg.update(native_shape_library=str(library), native_shape_library_sha256=sha256(library))
        write(self.after/'config.json', cfg)
        frozen = json.loads((self.after/'freeze.json').read_text())
        frozen.update(config_sha256=sha256(self.after/'config.json'), native_shape_build=build,
            native_shape_build_sha256=sha256(self.after/'libseaqr_shapes.so.build.json'))
        frozen['files_sha256'].update({'scripts/phase20_native_shapes.cpp':'test_source',
            'scripts/build_phase20_native_shapes.py':'test_builder'})
        for job in frozen['jobs']:
            path = self.after/job['name']/'launch.json'
            launch = json.loads(path.read_text())
            launch.update(configuration=cfg, config_sha256=frozen['config_sha256'], external_accelerators=dict(shape=dict(
                backend='native_cpu_bookkeeping', abi=1, library_path=str(library), library_sha256=sha256(library),
                fallback=False, centroid_reductions='NumPy float64 reference order', radius_px=8)))
            write(path, launch)
        write(self.after/'freeze.json', frozen)

    def test_native_library_is_verified_with_exact_journals(self):
        self.enable_native_fixture()
        self.finalize()
        result = json.loads((self.output/'summary.json').read_text())
        self.assertTrue(result['passed_execution_equivalence'])
        self.assertEqual(result['comparisons'][0]['native_shape_accelerators']['after']['abi'], 1)

    def test_changed_native_binary_is_rejected(self):
        self.enable_native_fixture()
        (self.after/'libseaqr_shapes.so').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'Native shape binary/build changed'):
            self.finalize()

    def test_changed_native_build_is_rejected(self):
        self.enable_native_fixture()
        write(self.after/'libseaqr_shapes.so.build.json', dict(abi=2))
        with self.assertRaisesRegex(ValueError, 'Native shape binary/build changed'):
            self.finalize()

    def test_missing_native_provenance_is_rejected(self):
        self.enable_native_fixture()
        path = self.after/'pva_0029/launch.json'
        launch = json.loads(path.read_text())
        del launch['external_accelerators']['shape']
        write(path, launch)
        with self.assertRaisesRegex(ValueError, 'native shape provenance'):
            self.finalize()

    def test_incremental_speed_and_failed_accuracy_stay_separate(self):
        self.finalize()
        result = json.loads((self.output / "summary.json").read_text())
        self.assertEqual(result["aggregate_speedup"], 2.)
        self.assertEqual(result["aggregate_before_fps"], 2.)
        self.assertEqual(result["aggregate_after_fps"], 4.)
        self.assertEqual(result["full_clip_frames"], 4)
        self.assertEqual(len(result["accuracy_comparisons"]), 4)
        self.assertEqual(result["accuracy_comparisons"][0]["before_fps"], 1.)
        self.assertFalse(result["accuracy"]["strict_dense_sample_continuity_gate"])
        self.assertFalse(result["production_ready"])

    def test_timing_reference_snapshot_cannot_change(self):
        (self.timing / "pva_0029/implementation/example.py").write_text("changed")
        with self.assertRaisesRegex(ValueError, "snapshot changed"):
            self.finalize()

    def test_changed_output_cannot_pass_even_with_refreshed_remote_hash(self):
        target = self.after / "pva_0029"
        (target / "frames.jsonl").write_text(json.dumps(dict(frame_index=0, candidates=[dict(x=21, y=30)])) + "\n")
        write(target / "comparison.json", dict(exact=True, output_sha256=sha256(target / "frames.jsonl")))
        with self.assertRaisesRegex(AssertionError, "Non-timing output changed"):
            self.finalize()

    def test_incomplete_timing_reference_is_rejected(self):
        write(self.timing / "status.json", dict(running=True, error=None, completed=[]))
        with self.assertRaisesRegex(ValueError, "Timing reference incomplete"):
            self.finalize()

    def enable_gpu_fixture(self):
        library=self.after/'libseaqr_integrated.so'
        library.write_bytes(b'GPU fixture, never loaded')
        build=dict(schema='seaqr.cuda-peak-gate-build.v1',diagnostic_only=False,
            variant='threshold_before_local_max',algorithm_policy_changed=False,
            library_sha256=sha256(library),reference_library_sha256='a'*64,
            sources_sha256={'fixture.cu':'b'*64})
        write(self.after/'libseaqr_integrated.so.build.json',build)
        frozen=json.loads((self.after/'freeze.json').read_text())
        frozen['compiled_library_sha256']=sha256(library)
        frozen['files_sha256']['scripts/fixture.cu']='b'*64
        frozen['gpu_transition']=dict(schema='seaqr.exact-gpu-transition.v1',
            before_library_sha256='a'*64,after_library_sha256=sha256(library),
            candidate_build_sha256=sha256(self.after/'libseaqr_integrated.so.build.json'))
        for job in frozen['jobs']:
            path=self.after/job['name']/'launch.json'
            launch=json.loads(path.read_text())
            launch['exact_cuda_stabilization']['library_sha256']=sha256(library)
            write(path,launch)
        return frozen,frozen['jobs'][0]

    def test_gpu_transition_package_verified(self):
        frozen,job=self.enable_gpu_fixture()
        finalizer.verify_package(self.after,frozen,job)

    def test_changed_gpu_binary_rejected(self):
        frozen,job=self.enable_gpu_fixture()
        (self.after/'libseaqr_integrated.so').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'GPU transition'):
            finalizer.verify_package(self.after,frozen,job)

    def test_changed_gpu_build_rejected(self):
        frozen,job=self.enable_gpu_fixture()
        path=self.after/'libseaqr_integrated.so.build.json'
        build=json.loads(path.read_text());build['diagnostic_only']=True;write(path,build)
        frozen['gpu_transition']['candidate_build_sha256']=sha256(path)
        with self.assertRaisesRegex(ValueError,'GPU transition'):
            finalizer.verify_package(self.after,frozen,job)

    def test_unfrozen_gpu_source_rejected(self):
        frozen,job=self.enable_gpu_fixture()
        frozen['files_sha256']['scripts/fixture.cu']='c'*64
        with self.assertRaisesRegex(ValueError,'GPU build source not frozen'):
            finalizer.verify_package(self.after,frozen,job)


if __name__ == "__main__":
    unittest.main()
