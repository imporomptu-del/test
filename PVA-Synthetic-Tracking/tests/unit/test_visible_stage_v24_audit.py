"""Synthetic report/trace checks. No recorded media or holdouts are opened."""
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
import nsys_stage_v24 as timeline
from verify_visible_stage_v24 import performance,distribution
from test_visible_stage_v24 import consume
import test_nsys_overlap_v23 as fixtures
from test_nsys_overlap_v23 import WORKER,MAIN


def trials(speedup=1.3,request_age=600):
    rows,samples={},{}
    for c in ('0126','0082'):
        for i in range(3):
            for m in ('reference','staged'):
                name=f'{c}_repeat{i}_{m}';fps=2.5*(speedup if m=='staged' else 1)
                values=dict(consumer_cadence=[1000/fps]*128,queue_aware=[500]*128,
                            request_to_complete=[request_age if m=='staged' else 700]*128)
                rows[name]=dict(name=name,count=128,wall_s=128/fps,fps=fps,traced=False,
                               metrics={k:distribution(v) for k,v in values.items()})
                samples[name]=values
    return rows,samples


class StageAuditTests(unittest.TestCase):
    def test_speed_and_all_latency_boundaries(self):
        self.assertTrue(performance(*trials())['passed'])
        self.assertFalse(performance(*trials(speedup=1.19))['passed'])
        value=performance(*trials(request_age=900))
        self.assertFalse(value['passed'])
        self.assertTrue(value['clips']['0126']['latency']['request_to_complete']['consistent_regression'])
        self.assertFalse(value['clips']['0126']['latency']['queue_aware']['consistent_regression'])

    def test_one_slow_pair_or_missing_workload_blocks(self):
        rows,samples=trials(speedup=2)
        rows['0126_repeat0_staged'].update(fps=2,wall_s=64)
        self.assertFalse(performance(rows,samples)['passed'])
        rows,samples=trials();del rows['0082_repeat2_staged']
        self.assertFalse(performance(rows,samples)['passed'])

    def fixture(self,root):
        sql,old=fixtures.OverlapAnalysisTests().fixture(root)
        db=sqlite3.connect(sql)
        for i in range(128):
            for stage,start,end in (('cpu_prepare',1,5),('warp_wait',6,11),('warp_gpu',12,19)):
                db.execute('INSERT INTO NVTX_EVENTS VALUES (?,?,?,?,?)',
                           (i*100+start,i*100+end,f'seaqr24|{i}|{stage}',WORKER,None))
            for offset in (1,2,3):
                correlation=i*8+offset
                db.execute('UPDATE CUPTI_ACTIVITY_KIND_RUNTIME SET start=?,end=? WHERE correlationId=?',
                           (i*100+13,i*100+14,correlation))
                db.execute('UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET start=?,end=? WHERE correlationId=?',
                           (i*100+14,i*100+18,correlation))
        db.commit();db.close()
        r=json.loads(old.read_text());_,snap,_=consume('staged',count=128)
        r.update(schema='seaqr.visible-stage-v24.v1',mode='staged',execution=snap,nvtx_push_pop_counts=[768,768])
        receipt=root/'trace.v24.json';receipt.write_text(json.dumps(r))
        return sql,receipt

    def test_full_trace_analysis_read_only_and_actual_cuda_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            paths=self.fixture(Path(directory));before=[timeline.sha(p) for p in paths]
            result=timeline.analyze(*paths)
            self.assertTrue(result['passed']);self.assertEqual(result['stage_range_count'],768)
            steady=result['stage_windows']['32_126']
            self.assertEqual(steady['warp_cuda_with_previous_detector_ms_per_interval'],0)
            self.assertGreater(steady['warp_cuda_with_previous_tracking_ms_per_interval'],0)
            self.assertEqual(before,[timeline.sha(p) for p in paths])

    def test_extra_stage_and_kernel_failures_are_not_silently_accepted(self):
        for failure in ('missing','early_warp','cross_thread','missing_kernel','gaussian_outside_tail','receipt'):
            with self.subTest(failure=failure),tempfile.TemporaryDirectory() as directory:
                sql,receipt=self.fixture(Path(directory));db=sqlite3.connect(sql)
                if failure=='missing':db.execute("DELETE FROM NVTX_EVENTS WHERE text='seaqr24|1|warp_wait'")
                elif failure=='early_warp':
                    db.execute("UPDATE NVTX_EVENTS SET end=107 WHERE text='seaqr24|1|warp_wait'")
                    db.execute("UPDATE NVTX_EVENTS SET start=108 WHERE text='seaqr24|1|warp_gpu'")
                elif failure=='cross_thread':
                    db.execute("UPDATE NVTX_EVENTS SET endGlobalTid=? WHERE text='seaqr24|1|warp_gpu'",(MAIN,))
                elif failure=='missing_kernel':db.execute('DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE correlationId=1')
                elif failure=='gaussian_outside_tail':
                    db.execute('UPDATE CUPTI_ACTIVITY_KIND_RUNTIME SET start=31,end=32,globalTid=? WHERE correlationId=2',(MAIN,))
                else:
                    r=json.loads(receipt.read_text());r['nvtx_push_pop_counts']=[384,384];receipt.write_text(json.dumps(r))
                db.commit();db.close()
                with self.assertRaises(AssertionError):timeline.analyze(sql,receipt)


if __name__=='__main__':unittest.main()
