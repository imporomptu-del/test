import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from visible_threads_v31 import policy,environment,scope,schedule,traces,full,KEYS,gate
from run_visible_threads_v31 import validate_runtime


class ThreadsTest(unittest.TestCase):
    def test_arms(self):
        self.assertEqual(policy('combined'),('combined','one'))
        self.assertEqual(policy('v26_default'),('v26','inherited'))
        self.assertEqual(policy('v20'),('v20','inherited'))
        with self.assertRaises(ValueError):policy('v20_default')

    def test_environment_is_child_only(self):
        original={'PATH':'fake'}
        self.assertEqual(environment(original,'combined'),dict(PATH='fake',OPENBLAS_NUM_THREADS='1'))
        self.assertEqual(original,{'PATH':'fake'})
        self.assertEqual(environment(original,'v26_default'),original)
        for key in KEYS:
            with self.assertRaises(ValueError):environment({key:'2'},'combined')

    def test_schedule_and_scope(self):
        self.assertEqual(len(schedule()),38)
        self.assertEqual(len(traces()),4)
        self.assertEqual(len(full()),4)
        rows=schedule()+traces()+full()
        self.assertEqual(len({r['name'] for r in rows}),46)
        for r in rows:scope(r['clip'],r['mode'],r['frames'],r['audit'],r['traced'])
        for c,m,f,a,t in [('0029','combined',128,False,False),('0082','v26',None,False,False),
            ('0126','combined',128.0,False,False),('0126','combined',128,True,True),('0126','v26',128,False,True)]:
            with self.assertRaises(ValueError):scope(c,m,f,a,t)

    def test_runtime_must_match_policy(self):
        with patch.dict(os.environ,{},clear=True):
            validate_runtime(dict(blas=[dict(threads=12)]),'combined_default')
            with self.assertRaises(ValueError):validate_runtime(dict(blas=[dict(threads=1)]),'combined')
        with patch.dict(os.environ,{'OPENBLAS_NUM_THREADS':'1'},clear=True):
            validate_runtime(dict(blas=[dict(threads=1)]),'combined')
            with self.assertRaises(ValueError):validate_runtime(dict(blas=[dict(threads=12)]),'combined')
            with self.assertRaises(ValueError):validate_runtime(dict(blas=[]),'combined')

    def timing_receipts(self):
        result={}
        for s in schedule():
            if s['audit']:continue
            fps={'v20':1.,'v26_default':1.2,'v26':1.3,'v28':1.1,'combined_default':1.3,'combined':1.5}[s['mode']]
            result[s['name']]=dict(passed=True,error=None,clip=s['clip'],arm=policy(s['mode'])[0],
                frames=128,processed_frames=128,state_audit=False,wall_s=128/fps,fps=fps,
                consumer_frame_ms=[256/fps]*128,execution=dict(frames=[dict(request_ns=1,ready_ns=1000001,
                    consumer_complete_ns=int(256e6/fps)+2000001) for _ in range(128)]))
        return result

    def test_original_gate_and_additional_default_controls(self):
        rows=self.timing_receipts()
        self.assertTrue(gate(rows)['passed'])
        for i in range(3):rows[f'0082_repeat{i}_combined_default'].update(fps=2.,wall_s=64.)
        g=gate(rows)
        self.assertTrue(g['original_gate']['passed'])
        self.assertFalse(g['passed'])

    def test_invalid_or_missing_control_rejected(self):
        rows=self.timing_receipts(); rows.pop('0082_repeat0_v26_default')
        with self.assertRaises(ValueError):gate(rows)
        for key,value in [('wall_s',float('nan')),('fps',0.),('state_audit',True),('processed_frames',127)]:
            rows=self.timing_receipts(); rows['0082_repeat0_v26_default'][key]=value
            with self.assertRaises(ValueError):gate(rows)


if __name__=='__main__':unittest.main()
