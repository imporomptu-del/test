"""Full-validation wrapper tests; fake runtimes only, never media or sysfs."""
import copy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

ROOT=Path(__file__).resolve().parents[2]
if (ROOT/'scripts').is_dir():sys.path.insert(0,str(ROOT/'scripts'))
import run_visible_validation_v34 as r
import visible_validation_v34 as v


def runtime():
    return dict(blas=[dict(threads=12,path='/fake/lib.so',sha256='test')],affinity=list(range(12)),
        numpy='test',opencv='test',opencv_threads=2,clock_ticks=100,
        thread_environment=dict.fromkeys(v.THREAD_KEYS))


class WrapperTests(unittest.TestCase):
    def test_scope_rejects_before_dependencies(self):
        with patch.object(r,'dependencies') as deps:
            for name in ('full_0001','../x','',None):
                with self.assertRaises(ValueError):r.run(name,Path('/tmp/x'))
            deps.assert_not_called()

    def test_output_must_match_frozen_trial_and_be_fresh(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(r,'HERE',Path(tmp).resolve()):
            trial=v.schedule()[2]['name']; output=Path(tmp).resolve()/'run'/trial
            self.assertEqual(r.request(trial,output)['kind'],'full')
            with self.assertRaises(ValueError):r.request(trial,Path(tmp)/'other')
            output.mkdir(parents=True)
            with self.assertRaises(FileExistsError):r.request(trial,output)

    def test_runtime_identity_threads_environment_and_opencv(self):
        expected=runtime(); r.validate_runtime(expected,expected,after=True)
        for key,value in (('affinity',[0]),('numpy','changed'),('opencv_threads',1)):
            actual=copy.deepcopy(expected); actual[key]=value
            with self.assertRaises(ValueError):r.validate_runtime(actual,expected,after=True)
        one=runtime(); one['blas'][0]['threads']=1
        with self.assertRaises(ValueError):r.validate_runtime(one,one)
        env=runtime(); env['thread_environment']['OPENBLAS_NUM_THREADS']='12'
        with self.assertRaises(ValueError):r.validate_runtime(env,env)

    def test_dependencies_refuse_root_before_runtime_import(self):
        with patch.object(r.os,'geteuid',return_value=0):
            with self.assertRaises(PermissionError):r.dependencies()

    def exercise(self, spec, passed=True, count=None):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve(); (root/'run').mkdir(); output=root/'run'/spec['name']
            info=runtime(); old=dict(passed=passed,error=None,processed_frames=count or v.expected_count(spec),fps=5,wall_s=100)
            baseline=SimpleNamespace(run=Mock()); profiler=SimpleNamespace(run=Mock())
            with patch.object(r,'HERE',root), patch.object(v,'verify_freeze',return_value=dict(runtime_reference=info)), \
                    patch.object(r,'dependencies',return_value=(baseline,profiler,lambda:info,{})), \
                    patch.object(v,'read',return_value=old),patch.object(v,'sha',return_value='hash'):
                if passed and (count is None or count==v.expected_count(spec)):
                    r.run(spec['name'],output)
                else:
                    with self.assertRaises(AssertionError):r.run(spec['name'],output)
            result=v.read(output.with_suffix('.v34.json'))
            return baseline,profiler,result

    def test_full_clip_does_not_inherit_a_prefix_cap(self):
        s=v.schedule()[2]; baseline,profiler,result=self.exercise(s)
        call=baseline.run.call_args.args[0]
        self.assertIsNone(call.frames); self.assertEqual(call.arm,'combined'); self.assertFalse(call.state_audit)
        self.assertEqual(result['count'],687); self.assertFalse(result['traced'])
        profiler.run.assert_not_called()

    def test_smoke_checks_private_state_before_full_trials(self):
        baseline,profiler,result=self.exercise(v.schedule()[0])
        call=baseline.run.call_args.args[0]
        self.assertTrue(call.state_audit); self.assertEqual(call.frames,128)
        self.assertTrue(result['audit']); profiler.run.assert_not_called()

    def test_profiles_are_separate_and_labeled(self):
        baseline,profiler,result=self.exercise(v.schedule()[-1])
        baseline.run.assert_not_called(); profiler.run.assert_called_once()
        self.assertTrue(result['traced']); self.assertEqual(result['trace_sha256'],'hash')

    def test_failed_output_or_incomplete_frames_cannot_pass(self):
        for options in (dict(passed=False),dict(count=128)):
            _,_,result=self.exercise(v.schedule()[2],**options)
            self.assertFalse(result['passed']); self.assertIsNotNone(result['error'])


if __name__=='__main__':unittest.main()
