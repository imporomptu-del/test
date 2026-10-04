"""Generated runner guard tests; no runtime imports, media or network access."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[2]/'scripts/run_accuracy_v56_diagnostic.py'
spec = importlib.util.spec_from_file_location('v56_runner_tests', SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def runtime():
    return dict(blas=[dict(threads=12, sha256='generated')], affinity=list(range(12)),
                numpy='generated', opencv='generated', clock_ticks=100, opencv_threads=12,
                thread_environment=dict.fromkeys(('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS',
                                                   'MKL_NUM_THREADS', 'GOTO_NUM_THREADS')))


class RunnerGuards(unittest.TestCase):
    def test_original_runtime_policy(self):
        runner.runtime_check(runtime(), runtime())
        after = runtime()
        after['opencv_threads'] = 2
        runner.runtime_check(after, runtime(), after=True)

    def test_runtime_identity_changes_fail(self):
        for key in ('blas', 'affinity', 'numpy', 'opencv', 'thread_environment', 'clock_ticks'):
            with self.subTest(key=key):
                actual = runtime()
                actual[key] = None
                with self.assertRaises(ValueError):
                    runner.runtime_check(actual, runtime())

    def test_wrong_cv_workers_fail_after(self):
        with self.assertRaises(ValueError):
            runner.runtime_check(runtime(), runtime(), after=True)

    def test_equal_but_changed_thread_policy_fails(self):
        value = runtime()
        value['blas'][0]['threads'] = 1
        with self.assertRaises(ValueError):
            runner.runtime_check(value, value)

    def test_equal_but_set_thread_environment_fails(self):
        value = runtime()
        value['thread_environment']['OMP_NUM_THREADS'] = '12'
        with self.assertRaises(ValueError):
            runner.runtime_check(value, value)

    def test_unknown_arm_rejected_before_hardware_access(self):
        with self.assertRaises(ValueError):
            runner.run('../outside')

    def test_write_is_nonoverwriting(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'record.json'
            runner.write(path, {'generated': True})
            with self.assertRaises(FileExistsError):
                runner.write(path, {'generated': False})
            self.assertEqual(runner.read(path), {'generated': True})
            self.assertEqual(runner.sha(path), hashlib.sha256(path.read_bytes()).hexdigest())

    def test_manifest_frame_scope(self):
        root = SCRIPT.parents[1]
        manifest = json.loads((root/'configs/evaluation/accuracy_v56_diagnostic_probes.json').read_text())
        self.assertEqual([p['frame_index'] for p in manifest['reference_probes']], runner.FRAMES)
        self.assertEqual(manifest['source_media']['expected_frames'], 674)
        self.assertEqual(manifest['clip'], '0126')
        self.assertEqual(len(manifest['provisional_controls']), 7)


if __name__ == '__main__':
    unittest.main()
