import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts'))
from analyze_visible_interaction_v30 import analyze, statistics
from profile_visible_interaction_v30 import MAJOR


class AnalysisTest(unittest.TestCase):
    def fixture(self, root):
        database, receipt = root/'trace.sqlite', root/'trace.json'
        db = sqlite3.connect(database)
        db.executescript('''
            CREATE TABLE StringIds(id INTEGER,value TEXT);
            CREATE TABLE NVTX_EVENTS(start INTEGER,end INTEGER,text TEXT,globalTid INTEGER);
            CREATE TABLE PROCESSES(globalPid INTEGER,pid INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,shortName INTEGER,globalPid INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY(start INTEGER,end INTEGER,copyKind INTEGER,bytes INTEGER,globalPid INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_MEMSET(start INTEGER,end INTEGER,globalPid INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER,end INTEGER,nameId INTEGER,globalTid INTEGER);
            CREATE TABLE DIAGNOSTIC_EVENT(severity INTEGER,globalPid INTEGER,text TEXT);
            CREATE TABLE SCHED_EVENTS(start INTEGER,globalTid INTEGER,isSchedIn INTEGER);
            CREATE TABLE ENUM_CUDA_MEMCPY_OPER(id INTEGER,label TEXT);
        ''')
        pid, tid = 1<<24, (1<<24)+19
        db.execute('INSERT INTO PROCESSES VALUES(?,?)', (pid, 19))
        db.execute('INSERT INTO ENUM_CUDA_MEMCPY_OPER VALUES(1,"HtoD")')
        db.executemany('INSERT INTO SCHED_EVENTS VALUES(?,?,?)', [(0,tid,1),(128000000,tid,0)])
        kernels = ['gaussian5', 'gaussian5', 'median5', 'residual_prepare', 'front_erode',
            'front_erode', 'front_eligible', 'front_noise', 'select_peaks', 'front_learn', 'finish_state', 'warp_cubic']
        names = {name:i+1 for i,name in enumerate(sorted(set(kernels)))}
        db.executemany('INSERT INTO StringIds VALUES(?,?)', [(i,n) for n,i in names.items()])
        events, telemetry = [], []
        for f in range(128):
            start = f*1000000+1
            for stage in MAJOR:
                db.execute('INSERT INTO NVTX_EVENTS VALUES(?,?,?,?)', (start,start+900000,f'seaqr30|{f}|{stage}',tid))
                events.append(dict(frame=f, name=stage, start_ns=start, end_ns=start+900000,
                                   thread_cpu_ns=200000, process_cpu_ns=400000))
            for k, name in enumerate(kernels):
                db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(?,?,?,?)', (start+k*1000,start+k*1000+500,names[name],pid))
            telemetry.append(dict(monotonic_ns=start, sensors={'fake_temperature':40000}))
        db.commit()
        db.close()
        receipt.write_text(json.dumps(dict(passed=True, traced=True, error=None, telemetry_errors=[], pid=19,
            push_pop_counts=[768,768], events=events, telemetry=telemetry,
            runtime_before={}, runtime_after={}, sensor_types={})))
        return database, receipt

    def test_complete_windows_and_unions(self):
        with tempfile.TemporaryDirectory() as directory:
            db, receipt = self.fixture(Path(directory))
            result = analyze(db, receipt)
            self.assertTrue(result['complete'])
            w = result['windows']['32_126']
            self.assertEqual(w['frames'], 95)
            self.assertEqual(w['frame_interval_ms'], 1)
            self.assertEqual(w['kernels']['gaussian5']['calls'],190)
            self.assertAlmostEqual(w['cuda_active_union_ms_per_frame'], .006)
            self.assertEqual(w['thread_scheduled_ms_per_frame']['19'], 1)

    def test_rejects_missing_kernel(self):
        with tempfile.TemporaryDirectory() as directory:
            db, receipt = self.fixture(Path(directory))
            connection = sqlite3.connect(db)
            connection.execute('DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE rowid=1')
            connection.commit(); connection.close()
            with self.assertRaisesRegex(AssertionError, 'kernel coverage'):
                analyze(db,receipt)

    def test_rejects_diagnostic_loss(self):
        with tempfile.TemporaryDirectory() as directory:
            db, receipt = self.fixture(Path(directory))
            connection = sqlite3.connect(db)
            connection.execute('INSERT INTO DIAGNOSTIC_EVENT VALUES(1,NULL,"dropped events")')
            connection.commit(); connection.close()
            with self.assertRaisesRegex(AssertionError, 'Trace error/loss'):
                analyze(db,receipt)

    def test_statistics_and_nonfinite(self):
        self.assertEqual(statistics([4,1,2,3])['median'],2.5)
        self.assertAlmostEqual(statistics([1,2,3,4])['p95'],3.85)
        for values in ([], [float('nan')], [float('inf')]):
            with self.assertRaises(AssertionError):
                statistics(values)


if __name__ == '__main__':
    unittest.main()
