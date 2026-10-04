from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from verify_visible_native_v25 import performance,attribution,schedules,WORKLOADS,MODES,distribution


def timings():
    trials,samples={},{}
    for c in WORKLOADS:
        for i in range(3):
            for m in MODES:
                name=f'{c}_repeat{i}_{m}';fps=1.3 if m=='native_staged' else 1.
                samples[name]={k:[1.]*128 for k in ('queue_aware','consumer_cadence','request_to_complete')}
                trials[name]=dict(count=128,fps=fps,wall_s=128/fps,traced=False,attributed=False,
                    metrics={k:distribution(v) for k,v in samples[name].items()})
    return trials,samples


def events():
    frames=[];rows=[]
    for i in range(128):
        start=i*1000
        frames.append(dict(frame=i,prepare_start_ns=start,prepare_end_ns=start+500,
            consumer_received_ns=start+500,consumer_complete_ns=start+1000))
        names=['correspondence','global_fit','composition','warp','detector','tracking'] if i else ['warp','detector','tracking']
        for n in names:
            consumer=n in ('detector','tracking');offset=600 if consumer else 100
            rows.append(dict(name=n,frame=i,thread=2 if consumer else 1,start_ns=start+offset,
                end_ns=start+offset+50,thread_cpu_ns=25,error=None))
    return rows,dict(policy='staged',frames=frames)


class NativeAuditTests(unittest.TestCase):
    def test_bounded_alternating_schedule(self):
        prefix,full,diagnostics=schedules()
        self.assertEqual((len(prefix),len(full),len(diagnostics)),(14,4,2))
        self.assertEqual([r['mode'] for r in prefix[2:8]],
            ['reference','native_staged','native_staged','reference','reference','native_staged'])
        self.assertTrue(all(not r['attributed'] for r in prefix))
        self.assertTrue(all(r['attributed'] for r in diagnostics))

    def test_missing_workload_cannot_pass(self):
        trials,samples=timings();self.assertTrue(performance(trials,samples)['passed'])
        del trials['0082_repeat1_reference']
        self.assertFalse(performance(trials,samples)['passed'])

    def test_gain_threshold_and_each_pair_are_enforced(self):
        trials,samples=timings()
        t=trials['0126_repeat0_native_staged'];t.update(fps=.99,wall_s=128/.99)
        self.assertFalse(performance(trials,samples)['passed'])
        trials,samples=timings()
        for n,t in trials.items():
            if n.endswith('native_staged'):t.update(fps=1.199,wall_s=128/1.199)
        self.assertFalse(performance(trials,samples)['passed'])

    def test_pre_admission_wait_is_not_hidden_by_throughput(self):
        trials,samples=timings()
        for i in (0,1):
            name=f'0126_repeat{i}_native_staged';samples[name]['request_to_complete']=[2.]*128
            trials[name]['metrics']['request_to_complete']=distribution([2.]*128)
        gate=performance(trials,samples)
        self.assertFalse(gate['passed'])
        self.assertTrue(gate['clips']['0126']['latency']['request_to_complete']['consistent_regression'])

    def test_diagnostic_timings_cannot_enter_performance_gate(self):
        trials,samples=timings();trials['0126_repeat0_reference']['attributed']=True
        with self.assertRaisesRegex(ValueError,'performance samples'):performance(trials,samples)

    def test_attribution_scope_thread_and_error_checks(self):
        rows,snapshot=events();result=attribution(rows,snapshot)
        self.assertEqual(result['stages']['steady_32_127']['global_fit']['wall']['count'],96)
        self.assertFalse(result['proves_gil_causality'])
        for key,value in (('error','test failure'),('thread',2),('end_ns',2000),('thread_cpu_ns',1000)):
            changed=deepcopy(rows);changed[0][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):attribution(changed,snapshot)
        with self.assertRaisesRegex(ValueError,'Incomplete attribution'):attribution(rows[1:],snapshot)

    def test_reference_thread_ownership_must_be_serial(self):
        rows,snapshot=events();snapshot['policy']='reference'
        with self.assertRaisesRegex(ValueError,'ownership'):attribution(rows,snapshot)
        for e in rows:e['thread']=1
        self.assertTrue(attribution(rows,snapshot)['instrumented_not_clean_fps'])


if __name__=='__main__':unittest.main()
