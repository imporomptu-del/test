"""Speed validation may ignore timing, never geometry or a missing journal row."""
import copy
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from tiny_target.visible_baseline import VisibleConfig

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/compare_phase20_exact_runs.py"
SPEC = importlib.util.spec_from_file_location("exact_run_comparison_test", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ExactRunComparisonTests(unittest.TestCase):
    def test_prefetch_requires_bounded_lossless_provenance_and_cleanup(self):
        launch = dict(configuration=asdict(VisibleConfig(frame_decode_execution='prefetch_one')))
        report = dict(frames=2)
        with self.assertRaisesRegex(ValueError, 'provenance'):
            MODULE.validate_decode(launch, report)
        launch['frame_decode'] = MODULE.decode_contract('prefetch_one')
        report['frame_decode'] = dict(contract=launch['frame_decode'], decoded_frames=2,
            consumed_frames=2, read_calls=2, maximum_observed_frames_ahead=1,
            worker_joined=True, capture_released=True, dropped_frames=0)
        MODULE.validate_decode(launch, report)
        for key, value in [('decoded_frames', 3), ('consumed_frames', 1), ('worker_joined', False),
                ('capture_released', False), ('dropped_frames', 1), ('maximum_observed_frames_ahead', 2), ('read_calls', 4)]:
            bad = copy.deepcopy(report); bad['frame_decode'][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'drain'):
                MODULE.validate_decode(launch, bad)
        launch['frame_decode']['drop_frames'] = True
        with self.assertRaisesRegex(ValueError, 'provenance'):
            MODULE.validate_decode(launch, report)

    def test_same_gpu_timing_pair_does_not_apply_inherited_accuracy_transition(self):
        left = dict(exact_cuda_stabilization=dict(library_sha256='b'*64))
        right = copy.deepcopy(left)
        transition = dict(schema='seaqr.exact-gpu-transition.v1', before_library_sha256='a'*64,
            after_library_sha256='b'*64, candidate_build_sha256='c'*64)
        self.assertIsNone(MODULE.transition_for_pair(left, right, transition))
        left['exact_cuda_stabilization']['library_sha256'] = 'a'*64
        self.assertEqual(MODULE.transition_for_pair(left, right, transition), transition)
        left['exact_cuda_stabilization']['library_sha256'] = 'd'*64
        with self.assertRaises(ValueError):
            MODULE.transition_for_pair(left, right, transition)

    def create(self, root, name, rows):
        path = root / name
        path.mkdir()
        launch = dict(configuration=asdict(VisibleConfig()), fps=10, source_sha256="fixture",
            exact_cuda_stabilization=dict(library_sha256="fixture-library"))
        report = dict(completed=True, full_clip=True, frames=2, processed_fps=2., timings_ms={})
        for filename, value in (("launch.json", launch), ("report.json", report)):
            (path / filename).write_text(json.dumps(value))
        (path / "frames.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        return path

    def test_only_explicit_timing_fields_are_ignored(self):
        rows = [dict(frame_index=i, motion=dict(pva_timings_ms=dict(total=50),
            motion_fit=dict(timing_ms=2, parameters=dict(x=1.)), warp_timings_ms=dict(total=10)),
            coverage=dict(detection_ms=5, searchable_pixels=100), tracks=[], timings_ms={}) for i in range(2)]
        right = copy.deepcopy(rows)
        right[1]["motion"]["pva_timings_ms"]["total"] = 100
        right[1]["motion"]["motion_fit"]["timing_ms"] = 4
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a, b = self.create(root, "a", rows), self.create(root, "b", right)
            self.assertTrue(MODULE.compare(a, b, root / "exact.json")["exact"])
            right[1]["motion"]["motion_fit"]["parameters"]["x"] += .001
            c = self.create(root, "c", right)
            with self.assertRaisesRegex(AssertionError, "Non-timing"):
                MODULE.compare(a, c, root / "different.json")
            d = self.create(root, "d", right[:1])
            with self.assertRaisesRegex(ValueError, "length"):
                MODULE.compare(a, d, root / "truncated.json")
        self.assertEqual(MODULE.without_timing(dict(unknown_ms=10)), dict(unknown_ms=10))

    def test_native_execution_requires_provenance_without_excluding_shapes(self):
        cfg = asdict(VisibleConfig(spatial_background='median5', spatial_filter_backend='cuda_median5',
            cuda_median_library='/explicit/gpu.so', state_update_backend='cuda_resident',
            pixel_noise_enabled=True, pixel_noise_model='background_residual',
            shape_measurement_mode='mutual_half_height_r8'))
        rows = [dict(frame_index=i, candidates=[dict(x=10, y=20, shape=dict(support_pixels=4))]) for i in range(2)]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a, b = self.create(root, 'a', rows), self.create(root, 'b', rows)
            launches = [json.loads((p/'launch.json').read_text()) for p in (a, b)]
            launches[0]['configuration'] = cfg
            launches[1]['configuration'] = dict(cfg, native_shape_library='/explicit/native.so', native_shape_library_sha256='a'*64)
            for p, launch in zip((a, b), launches):
                (p/'launch.json').write_text(json.dumps(launch))
            with self.assertRaisesRegex(ValueError, 'provenance'):
                MODULE.compare(a, b, root/'missing.json')
            native = dict(abi=1, backend='native_cpu_bookkeeping', fallback=False, radius_px=8,
                library_path='/explicit/native.so', library_sha256='a'*64,
                centroid_reductions='NumPy float64 reference order')
            launches[1]['external_accelerators'] = dict(shape=native)
            (b/'launch.json').write_text(json.dumps(launches[1]))
            self.assertTrue(MODULE.compare(a, b, root/'valid.json')['exact'])
            rows[1]['candidates'][0]['shape']['support_pixels'] += 1
            (b/'frames.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            with self.assertRaisesRegex(AssertionError, 'Non-timing'):
                MODULE.compare(a, b, root/'different.json')
            launches[1]['configuration']['spatial_threshold_sigma'] += .1
            (b/'launch.json').write_text(json.dumps(launches[1]))
            with self.assertRaisesRegex(ValueError, 'policy'):
                MODULE.compare(a, b, root/'policy.json')

    def test_gpu_change_requires_explicit_frozen_pair_without_ignoring_outputs(self):
        rows=[dict(frame_index=i,candidates=[dict(x=3.,score=2.)]) for i in range(2)]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);a=self.create(root,'a',rows);b=self.create(root,'b',rows)
            for path,digest in ((a,'a'*64),(b,'b'*64)):
                launch=json.loads((path/'launch.json').read_text())
                launch['exact_cuda_stabilization']['library_sha256']=digest
                (path/'launch.json').write_text(json.dumps(launch))
            with self.assertRaisesRegex(ValueError,'GPU library changed'):
                MODULE.compare(a,b,root/'unauthorized.json')
            transition=dict(schema='seaqr.exact-gpu-transition.v1',before_library_sha256='a'*64,
                after_library_sha256='b'*64,candidate_build_sha256='c'*64)
            self.assertTrue(MODULE.compare(a,b,root/'allowed.json',gpu_transition=transition)['exact'])
            for bad in (dict(transition,after_library_sha256='d'*64),
                    dict(transition,candidate_build_sha256='short'),dict(transition,ignore_scores=True)):
                with self.assertRaises(ValueError):MODULE.compare(a,b,root/'bad.json',gpu_transition=bad)
            changed=copy.deepcopy(rows);changed[1]['candidates'][0]['score']=2.000001
            (b/'frames.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in changed))
            with self.assertRaisesRegex(AssertionError,'Non-timing'):
                MODULE.compare(a,b,root/'different.json',gpu_transition=transition)


if __name__ == "__main__":
    unittest.main()
