"""Scoring-only wiring, fail-closed metadata and performance gate tests."""
import ast
import copy
from contextlib import ExitStack
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

ROOT=Path(__file__).resolve().parents[2]
if (ROOT/'scripts').is_dir():sys.path.insert(0,str(ROOT/'scripts'))
import run_visible_native_serial_v35 as r
import visible_native_serial_v35 as v


def runtime():
    return dict(blas=[dict(threads=12,path='/fake/lib.so',sha256='test')],affinity=list(range(12)),
        numpy='test',opencv='test',opencv_threads=2,clock_ticks=100,
        thread_environment=dict.fromkeys(v.THREAD_KEYS))


def base_receipt(count=128):
    return dict(schema='seaqr.visible-combined-v29.v1',passed=True,error=None,
        processed_frames=count,fps=5,wall_s=count/5,native_motion_v25_enabled=False,
        staged_v24_enabled=False,execution_policy='serial_reference',comparison={'exact':True})


class WrapperTests(unittest.TestCase):
    def test_scope_rejects_before_dependencies(self):
        with patch.object(r,'dependencies') as deps:
            for name in ('full_0001','../x','',None):
                with self.assertRaises(ValueError):r.run(name,Path('/tmp/x'))
            deps.assert_not_called()

    def test_output_fresh_and_exact_scope(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(r,'HERE',Path(tmp).resolve()):
            trial=v.schedule()[2]['name']; output=Path(tmp).resolve()/'run'/trial
            self.assertEqual(r.request(trial,output)['kind'],'prefix')
            with self.assertRaises(ValueError):r.request(trial,Path(tmp)/'other')
            output.parent.mkdir()
            v.write(output.with_suffix('.v35base.json'),{})
            with self.assertRaises(FileExistsError):r.request(trial,output)

    def test_runtime_identity(self):
        expected=runtime(); r.validate_runtime(expected,expected,after=True)
        for key,value in (('affinity',[0]),('numpy','changed'),('opencv_threads',1)):
            actual=copy.deepcopy(expected); actual[key]=value
            with self.assertRaises(ValueError):r.validate_runtime(actual,expected,after=True)
        one=runtime(); one['blas'][0]['threads']=1
        with self.assertRaises(ValueError):r.validate_runtime(one,one)
        env=runtime(); env['thread_environment']['OPENBLAS_NUM_THREADS']='12'
        with self.assertRaises(ValueError):r.validate_runtime(env,env)

    def test_root_refused_before_runtime_import(self):
        with patch.object(r.os,'geteuid',return_value=0):
            with self.assertRaises(PermissionError):r.dependencies()

    def test_adapter_truthful_single_serialization_without_mutating_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'trial'; original=base_receipt(); saved=copy.deepcopy(original)
            adapter=r.ReceiptAdapter(p,True,{'source':'frozen'})
            adapter(p.with_suffix('.v29.json'),original)
            self.assertEqual(original,saved); self.assertFalse(p.with_suffix('.v29.json').exists())
            new=v.read(p.with_suffix('.v35base.json'))
            self.assertEqual(new['schema'],'seaqr.visible-native-serial-base-v35.v1')
            self.assertTrue(new['native_motion_v25_enabled']); self.assertEqual(new['comparison'],original['comparison'])
            with self.assertRaises(ValueError):adapter(p.with_suffix('.v29.json'),original)

    def test_adapter_rejects_wrong_source_path_schema_flags_or_overwrite(self):
        for key,value in (('schema','unexpected'),('native_motion_v25_enabled',True),
                          ('staged_v24_enabled',True),('execution_policy','staged')):
            with tempfile.TemporaryDirectory() as tmp:
                p=Path(tmp)/'trial'; original=base_receipt(); original[key]=value
                with self.assertRaises(ValueError):r.ReceiptAdapter(p,True,{})(p.with_suffix('.v29.json'),original)
                self.assertFalse(p.with_suffix('.v35base.json').exists())
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'trial'; adapter=r.ReceiptAdapter(p,False,{})
            with self.assertRaises(ValueError):adapter(Path(tmp)/'wrong.json',base_receipt())
            v.write(p.with_suffix('.v35base.json'),{'keep':True})
            with self.assertRaises(FileExistsError):adapter(p.with_suffix('.v29.json'),base_receipt())
            self.assertEqual(v.read(p.with_suffix('.v35base.json')),{'keep':True})

    def test_call_coverage_fallback_and_early_reference_rejection(self):
        h=SimpleNamespace(calls=120,fallbacks=0,passthroughs=0)
        r.validate_calls({'arm':'native'},128,127,h)
        for options in ({'calls':0},{'calls':128},{'fallbacks':1},{'passthroughs':1}):
            bad=SimpleNamespace(**dict(vars(h),**options))
            with self.assertRaises(AssertionError):r.validate_calls({'arm':'native'},128,127,bad)
        with self.assertRaises(AssertionError):r.validate_calls({'arm':'native'},128,126,h)
        with self.assertRaises(AssertionError):r.validate_calls({'arm':'reference'},128,127,h)
        h.calls=0; r.validate_calls({'arm':'reference'},128,127,h)

    def exercise(self,spec,fail=False,incomplete=False):
        with tempfile.TemporaryDirectory() as tmp,ExitStack() as stack:
            root=Path(tmp).resolve(); (root/'run').mkdir(); output=root/'run'/spec['name']
            info=runtime(); count=v.expected_count(spec); calls=[]
            helper=SimpleNamespace(calls=0,fallbacks=0,passthroughs=0,transformed_sha256='transform')
            def original(*args,**kwargs):calls.append('reference')
            def adapted(*args,**kwargs):helper.calls+=1; calls.append('native')
            motion=SimpleNamespace(fit_global_motion=original)
            baseline=SimpleNamespace(write=v.write,__file__=__file__)
            old_write=baseline.write
            def child(args):
                self.assertEqual(args.frames,spec['frames']); self.assertEqual(args.arm,'combined')
                self.assertEqual(args.state_audit,spec['audit'])
                for _ in range(count-1):motion.fit_global_motion(None)
                value=base_receipt(count-1 if incomplete else count)
                if fail:value.update(passed=False,error='injected')
                baseline.write(output.with_suffix('.v29.json'),value)
                if fail:raise RuntimeError('injected')
            baseline.run=child
            stack.enter_context(patch.object(r,'HERE',root))
            stack.enter_context(patch.object(v,'verify_freeze',return_value=dict(runtime_reference=info)))
            stack.enter_context(patch.object(r,'dependencies',return_value=(baseline,lambda:info,{},motion,helper,adapted)))
            stack.enter_context(patch.object(v,'sha',return_value='hash'))
            if spec['kind']=='full':
                v.write(root/'run/performance_gate.json',{'passed':True})
                stack.enter_context(patch.object(v,'performance_gate',return_value={'passed':True}))
            if fail or incomplete:
                with self.assertRaises((RuntimeError,AssertionError)):r.run(spec['name'],output)
            else:r.run(spec['name'],output)
            result=v.read(output.with_suffix('.v35.json'))
            self.assertIs(motion.fit_global_motion,original); self.assertIs(baseline.write,old_write)
            self.assertTrue(result['bindings_restored']); self.assertEqual(result['fit_calls'],count-1)
            self.assertFalse(output.with_suffix('.v29.json').exists())
            self.assertEqual(set(calls),{spec['arm']})
            return result

    def test_native_and_reference_share_serial_parent_but_only_native_scores(self):
        for s in v.schedule()[:4]:
            out=self.exercise(s); self.assertTrue(out['passed'])
            self.assertEqual(out['native_calls'],127 if s['arm']=='native' else 0)

    def test_full_has_no_prefix_cap(self):
        out=self.exercise(v.schedule()[-4]); self.assertEqual(out['count'],687)

    def test_bindings_restored_after_parent_failure_and_incomplete_output(self):
        for opts in ({'fail':True},{'incomplete':True}):
            out=self.exercise(v.schedule()[0],**opts); self.assertFalse(out['passed']); self.assertIsNotNone(out['error'])

    def test_failed_or_changed_speed_gate_prevents_full_before_runtime(self):
        for saved,recomputed in (({'passed':False},{'passed':False}),({'passed':True},{'passed':False})):
            with tempfile.TemporaryDirectory() as tmp,patch.object(r,'HERE',Path(tmp).resolve()), \
                    patch.object(v,'verify_freeze',return_value={}),patch.object(r,'dependencies') as deps, \
                    patch.object(v,'performance_gate',return_value=recomputed):
                root=Path(tmp).resolve(); (root/'run').mkdir(); v.write(root/'run/performance_gate.json',saved)
                s=v.schedule()[-1]
                with self.assertRaises(ValueError):r.run(s['name'],root/'run'/s['name'])
                deps.assert_not_called()

    def test_failed_generated_gate_prevents_any_video_dependency(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(r,'HERE',Path(tmp).resolve()), \
                patch.object(v,'verify_freeze',side_effect=ValueError('generated gate')),patch.object(r,'dependencies') as deps:
            s=v.schedule()[0]
            with self.assertRaises(ValueError):r.run(s['name'],Path(tmp).resolve()/'run'/s['name'])
            deps.assert_not_called()

    def test_hardware_guard_asts_unchanged(self):
        parent=ROOT/'scripts/visible_validation_v34.py'
        if not parent.exists():parent=v.V34/'visible_validation_v34.py'
        old={x.name:ast.dump(x,include_attributes=False) for x in ast.parse(parent.read_text()).body if isinstance(x,(ast.FunctionDef,ast.ClassDef))}
        new={x.name:ast.dump(x,include_attributes=False) for x in ast.parse(Path(v.__file__).read_text()).body if isinstance(x,(ast.FunctionDef,ast.ClassDef))}
        names=('read_node','write_min','snapshot','validate_original','expected_policy','check_policy',
            'PolicyTransitionError','unchanged_limits','wait_policy','set_policy','restore','transition',
            'hardware_preflight','sensors','thermal_check','child_environment','proc_identity','stop_owned',
            'guard_reason','watchdog','Guard','competing_experiments','exclusive_lock')
        for n in names:self.assertEqual(old[n],new[n],n)


class PerformanceTests(unittest.TestCase):
    def fixtures(self,directory,speedups=None,latency=None):
        for spec in (s for s in v.schedule() if s['kind']=='prefix'):
            native=spec['arm']=='native'; i=spec['repeat']; c=spec['clip']
            ratio=(speedups or {}).get(c,[1.2]*3)[i] if native else 1
            timing=(latency or {}).get(c,[80]*3)[i] if native else 100
            base=base_receipt(); base.update(schema='seaqr.visible-native-serial-base-v35.v1',
                native_motion_v25_enabled=native,wall_s=25.6/ratio,fps=5*ratio,
                consumer_frame_ms=[timing]*128,
                execution={'frames':[dict(request_ns=1,consumer_complete_ns=1+timing*1000000)]*128})
            path=directory/spec['name']; v.write(path.with_suffix('.v35base.json'),base)
            outer=dict(**spec,passed=True,error=None,count=128,bindings_restored=True,
                receipt_sha256=v.sha(path.with_suffix('.v35base.json')),fps=base['fps'],wall_s=base['wall_s'])
            v.write(path.with_suffix('.v35.json'),outer)

    def test_passes_meaningful_gain_and_excludes_smokes_and_full(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp); self.fixtures(d); out=v.performance_gate(d)
            self.assertTrue(out['passed']); self.assertEqual(len(out['evidence_sha256']),12)
            self.assertAlmostEqual(out['clips']['0126']['pooled_speedup'],1.2)
            self.assertTrue(out['full_regression_permitted'])

    def test_small_gains_or_any_slower_pair_do_not_advance(self):
        for ratios in ({'0126':[1.04]*3},{'0126':[1.07]*3,'0082':[1.07]*3},
                       {'0126':[1.5,1.5,.999]}):
            with tempfile.TemporaryDirectory() as tmp:
                d=Path(tmp); self.fixtures(d,speedups=ratios)
                self.assertFalse(v.performance_gate(d)['passed'])

    def test_consistent_tail_regression_rejects_even_with_better_fps(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp); self.fixtures(d,latency={'0082':[110,110,80]})
            self.assertFalse(v.performance_gate(d)['passed'])

    def test_single_tail_outlier_is_reported_but_not_consistent(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp); self.fixtures(d,latency={'0082':[110,80,80]})
            out=v.performance_gate(d); self.assertTrue(out['passed'])
            self.assertEqual(out['clips']['0082']['latency']['consumer_cadence']['worse_pairs'],1)

    def test_missing_changed_or_nan_evidence_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp)
            with self.assertRaises(FileNotFoundError):v.performance_gate(d)
            self.fixtures(d)
            orig=v.read
            for mode in ('hash','frames','native_flag','restore','nan'):
                def corrupt(path):
                    value=orig(path)
                    if Path(path).name==v.schedule()[2]['name']+'.v35.json':
                        if mode=='hash':value['receipt_sha256']='changed'
                        if mode=='restore':value['bindings_restored']=False
                        if mode=='nan':value['wall_s']=float('nan')
                    if Path(path).name==v.schedule()[2]['name']+'.v35base.json':
                        if mode=='frames':value['execution']['frames'].pop()
                        if mode=='native_flag':value['native_motion_v25_enabled']=True
                    return value
                with patch.object(v,'read',side_effect=corrupt),self.assertRaises(ValueError):v.performance_gate(d)

    def test_percentiles_and_invalid_samples(self):
        self.assertAlmostEqual(v.percentile95([1,2,3,4]),3.85)
        for data in ([],[0],[float('nan')],[float('inf')],[-1],[True]):
            with self.assertRaises(ValueError):v.percentile95(data)


if __name__=='__main__':unittest.main()
