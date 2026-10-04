import importlib.util
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[2] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('nsys_overlap_v23', SCRIPTS / 'nsys_overlap_v23.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

PID = (1 << 48) + (100 << 24)
MAIN, WORKER = PID + 100, PID + 101


def ranges():
    result = []
    # One useful next-frame overlap: motion(i+1) is [100i+100,100i+120],
    # coinciding with detector(i) [100i+30,100i+110] and tracking(i) [115,125].
    for i in range(128):
        for stage, start, end, tid in (
                ('motion_worker', i*100, i*100+20, WORKER),
                ('detector', i*100+30, i*100+110, MAIN),
                ('tracking', i*100+115, i*100+125, MAIN)):
            result.append(dict(start=start, end=end, text=f'seaqr23|{i}|{stage}',
                               globalTid=tid, endGlobalTid=None))
    return result


def kernels_and_runtime(rows):
    names = {i+1: name for i, name in enumerate(module.EXPECTED)}
    ids = {name: i for i, name in names.items()}
    kernels, runtime = [], []
    for row in rows:
        stage = row['text'].split('|')[-1]
        if stage == 'tracking':
            continue
        for name, count in module.EXPECTED.items():
            if (stage == 'motion_worker') != (name in ('gaussian5', 'warp_cubic')):
                continue
            for _ in range(count):
                correlation = len(kernels) + 1
                runtime.append(dict(start=row['start']+1, end=row['start']+2,
                                    globalTid=row['globalTid'], correlationId=correlation))
                kernels.append(dict(start=row['start']+2, end=row['start']+5,
                                    globalPid=PID, correlationId=correlation, shortName=ids[name]))
    return kernels, runtime, names


class OverlapAnalysisTests(unittest.TestCase):
    def test_ranges_complete_and_threads_separate(self):
        frames, tids = module.validate_ranges(ranges())
        self.assertEqual(len(frames), 128)
        self.assertNotEqual(tids['motion_worker'], tids['tracking'])

    def test_range_failures(self):
        for mutation in ('missing', 'duplicate', 'same_thread', 'nested', 'out_of_order', 'open', 'far_ahead'):
            with self.subTest(mutation=mutation):
                rows = ranges()
                if mutation == 'missing':
                    rows.pop()
                elif mutation == 'duplicate':
                    rows.append(dict(rows[0]))
                elif mutation == 'same_thread':
                    for row in rows:
                        row['globalTid'] = MAIN
                elif mutation == 'nested':
                    rows[2]['start'] = rows[1]['start']+1
                elif mutation == 'out_of_order':
                    rows[3]['start'], rows[3]['end'] = 0, 1
                elif mutation == 'open':
                    rows[0]['end'] = None
                elif mutation == 'far_ahead':
                    rows[6]['start'], rows[6]['end'] = 121, 122
                with self.assertRaises(AssertionError):
                    module.validate_ranges(rows)

    def test_kernel_correlation_not_temporal_guess(self):
        rows = ranges()
        module.validate_ranges(rows)
        kernels, runtime, names = kernels_and_runtime(rows)
        # Device execution can be deferred beyond its host stage; retain exact
        # launching frame via correlation rather than assign to a concurrent frame.
        kernels[0]['start'] += 100
        kernels[0]['end'] += 100
        result = module.kernel_coverage(kernels, runtime, rows, names, PID)
        self.assertTrue(result['passed'])
        self.assertEqual(result['per_frame']['0']['gaussian5'], 2)

    def test_missing_kernel_and_ambiguous_correlation_fail(self):
        rows = ranges()
        module.validate_ranges(rows)
        kernels, runtime, names = kernels_and_runtime(rows)
        with self.assertRaisesRegex(AssertionError, 'per-frame'):
            module.kernel_coverage(kernels[1:], runtime, rows, names, PID)
        runtime.append(dict(runtime[0]))
        with self.assertRaisesRegex(AssertionError, 'Ambiguous'):
            module.kernel_coverage(kernels, runtime, rows, names, PID)

    def test_window_union_and_previous_frame_overlap(self):
        rows = ranges()
        frames, _ = module.validate_ranges(rows)
        # Both copies overlap each other and part of the kernel; all duration
        # unions must count each instant once. Place them across the boundary.
        start = frames[32]['detector']['start']
        data = dict(KERNEL=[dict(start=start-5, end=start+10)],
                    MEMCPY=[dict(start=start+5, end=start+15), dict(start=start+7, end=start+12)],
                    MEMSET=[])
        value = module.window_metrics(data, rows, frames, 32)
        self.assertEqual(value['intervals'], 95)
        self.assertAlmostEqual(value['cuda_active_union_ms_per_interval'], 15/(95*1e6))
        self.assertAlmostEqual(value['kernel_copy_overlap_ms_per_interval'], 5/(95*1e6))
        self.assertEqual(value['cuda_rows_crossing_boundary']['KERNEL'], 1)
        self.assertTrue(all(p['host_overlap_ns'] == 15 for p in value['worker_previous_frame_pairs']))

    def test_scheduler_carry_in_union_and_unknown_tail(self):
        events = []
        for tid, cpu, sequence in ((MAIN, 1, [(0,1),(8,0),(12,1),(20,0)]),
                                   (WORKER, 2, [(2,1),(15,0),(19,1)])):
            events.extend(dict(globalTid=tid, cpu=cpu, start=t, isSchedIn=state) for t,state in sequence)
        value = module.scheduling_metrics(events, dict(detector=MAIN,motion_worker=WORKER),
                                          4,18,[(6,13)],1)
        self.assertEqual(value['threads']['main']['scheduled_ms_per_interval'], 10)
        self.assertEqual(value['threads']['worker']['scheduled_ms_per_interval'], 11)
        self.assertEqual(value['both_threads_scheduled_ms_per_interval'], 7)
        self.assertEqual(value['both_threads_scheduled_and_process_cuda_active_ms_per_interval'], 3)
        self.assertTrue(value['threads']['worker']['unmatched_in_at_trace_end'])
        self.assertTrue(value['threads']['worker']['events_enclose_window'])

    def test_scheduler_malformed_intervals_not_invented(self):
        events = [dict(globalTid=MAIN,cpu=1,start=t,isSchedIn=s) for t,s in [(0,1),(5,1),(8,0)]]
        value = module.scheduling_metrics(events,dict(detector=MAIN,motion_worker=WORKER),0,10,[],1)
        self.assertEqual(value['threads']['main']['scheduled_ms_per_interval'],3)
        self.assertEqual(value['threads']['main']['malformed_transition_count'],1)
        self.assertEqual(value['threads']['worker']['events'],0)
        self.assertEqual(value['both_threads_scheduled_ms_per_interval'],0)

    def test_kernel_duration_clipping_sum_union_and_counts(self):
        rows = [dict(start=0,end=8,shortName=1),dict(start=5,end=12,shortName=1),
                dict(start=20,end=30,shortName=2)]
        values = module.kernel_durations(rows,{1:'gaussian5',2:'median5'},4,10,1)
        self.assertEqual(set(values),{'gaussian5'})
        self.assertEqual(values['gaussian5']['calls_intersecting_window'],2)
        self.assertEqual(values['gaussian5']['calls_starting_in_window'],1)
        self.assertEqual(values['gaussian5']['calls_crossing_boundary'],2)
        self.assertEqual(values['gaussian5']['clipped_sum_ms_per_interval'],9)
        self.assertEqual(values['gaussian5']['clipped_union_ms_per_interval'],6)

    def test_runtime_api_thread_separation_nested_wait_and_device_overlap(self):
        rows = [dict(start=0,end=8,nameId=1,globalTid=MAIN,returnValue=0),
                dict(start=5,end=12,nameId=1,globalTid=MAIN,returnValue=1),
                dict(start=5,end=6,nameId=2,globalTid=MAIN,returnValue=0),
                dict(start=5,end=8,nameId=2,globalTid=WORKER,returnValue=0),
                dict(start=5,end=8,nameId=1,globalTid=WORKER+99,returnValue=0)]
        value = module.runtime_durations(rows,{1:'cudaMemcpy',2:'cudaDeviceSynchronize'},
                dict(detector=MAIN,motion_worker=WORKER),4,10,[(7,9)],1)
        main, worker = value['threads']['main'],value['threads']['worker']
        self.assertEqual(main['all_apis']['clipped_sum_ms_per_interval'],10)
        self.assertEqual(main['all_apis']['clipped_union_ms_per_interval'],6)
        self.assertEqual(main['by_api']['cudaMemcpy']['nonzero_return_value_calls'],1)
        self.assertEqual(main['by_api']['cudaMemcpy']['coincident_process_cuda_ms_per_interval'],2)
        self.assertEqual(worker['all_apis']['calls_intersecting_window'],1)
        self.assertEqual(worker['all_apis']['clipped_union_ms_per_interval'],3)
        self.assertEqual(worker['all_apis_coincident_process_cuda_ms_per_interval'],1)

    def fixture(self, root):
        rows = ranges()
        kernels, runtime, names = kernels_and_runtime(rows)
        dbpath = root / 'trace.sqlite'
        db = sqlite3.connect(dbpath)
        db.executescript('''
            CREATE TABLE StringIds(id INTEGER, value TEXT);
            CREATE TABLE PROCESSES(globalPid INTEGER,pid INTEGER,name TEXT);
            CREATE TABLE NVTX_EVENTS(start INTEGER,end INTEGER,text TEXT,globalTid INTEGER,endGlobalTid INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,globalPid INTEGER,correlationId INTEGER,shortName INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY(start INTEGER,end INTEGER,globalPid INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER,end INTEGER,globalTid INTEGER,correlationId INTEGER,nameId INTEGER,returnValue INTEGER);
            CREATE TABLE DIAGNOSTIC_EVENT(severity INTEGER,text TEXT,globalPid INTEGER);
            CREATE TABLE ENUM_DIAGNOSTIC_SEVERITY_LEVEL(id INTEGER,name TEXT);
        ''')
        db.executemany('INSERT INTO StringIds VALUES (?,?)', names.items())
        db.execute('INSERT INTO StringIds VALUES (?,?)', (100,'cudaLaunchKernel'))
        db.execute('INSERT INTO PROCESSES VALUES (?,?,?)', (PID, 100, 'python3'))
        db.executemany('INSERT INTO NVTX_EVENTS VALUES (?,?,?,?,?)',
                       [(r['start'],r['end'],r['text'],r['globalTid'],None) for r in rows])
        db.executemany('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?)',
                       [(r['start'],r['end'],r['globalPid'],r['correlationId'],r['shortName']) for r in kernels])
        db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (?,?,?)', (50,60,PID))
        db.executemany('INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?,?,?,?,?,?)',
                       [(r['start'],r['end'],r['globalTid'],r['correlationId'],100,0) for r in runtime])
        db.executemany('INSERT INTO ENUM_DIAGNOSTIC_SEVERITY_LEVEL VALUES (?,?)', [(3,'Error'),(4,'Verbose')])
        db.execute('INSERT INTO DIAGNOSTIC_EVENT VALUES (?,?,?)', (4,'Not an error',PID))
        db.commit()
        db.close()
        (root/'trace').mkdir()
        (root/'trace'/'launch.json').write_text(json.dumps(dict(
            configuration=dict(input_bit_depth=8),source='/unused/chunk_0126.avi',
            external_accelerators=dict(median=dict(library_sha256=module.CUDA_SHA256)),
            exact_cuda_stabilization=dict(library_sha256=module.CUDA_SHA256))))
        (root/'trace.v20.json').write_text(json.dumps(dict(passed=True,error=None,processed_frames=128)))
        receipt = dict(schema='seaqr.visible-overlap-v23.v1', passed=True,error=None,traced=True,
            mode='overlap',frames=128,processed_frames=128,clip='0126',gpu_changed=False,
            algorithm_changed=False,raw16_accessed=False,defaults_changed=False,
            nvtx_push_pop_counts=[384,384],baseline_receipt_sha256=module.sha(root/'trace.v20.json'),
            execution=dict(mode='overlap',frames=[dict(frame=i) for i in range(128)],engine=dict(
                closed=True,joined=True,discarded_on_cancel=0,prepared=128,delivered=128,released=128,max_owned=2)))
        receipt_path = root/'trace.v23.json'
        receipt_path.write_text(json.dumps(receipt))
        return dbpath, receipt_path

    def test_end_to_end_read_only_and_diagnostic_severity(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = self.fixture(Path(temporary))
            before = [module.sha(p) for p in paths]
            result = module.analyze(*paths)
            self.assertTrue(result['passed'])
            self.assertEqual(result['stage_range_count'], 384)
            self.assertFalse(result['performance_comparison'])
            self.assertEqual(before, [module.sha(p) for p in paths])

    def test_trace_loss_or_original_cuda_change_fails(self):
        for failure in ('loss', 'cuda', 'baseline'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                paths = self.fixture(root)
                if failure == 'loss':
                    db = sqlite3.connect(paths[0])
                    db.execute('INSERT INTO DIAGNOSTIC_EVENT VALUES (?,?,?)', (2,'Dropped trace events',PID))
                    db.commit()
                    db.close()
                elif failure == 'cuda':
                    path = root/'trace'/'launch.json'
                    value = json.loads(path.read_text())
                    value['exact_cuda_stabilization']['library_sha256'] = 'different'
                    path.write_text(json.dumps(value))
                else:
                    (root/'trace.v20.json').write_text('{}')
                with self.assertRaises(AssertionError):
                    module.analyze(*paths)


if __name__ == '__main__':
    unittest.main()
