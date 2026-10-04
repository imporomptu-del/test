"""Read-only common-clock analysis; traced timings never become clean speedups."""
import argparse
from collections import defaultdict
import math
from pathlib import Path
import sqlite3

from profile_visible_interaction_v30 import read, write, sha, MAJOR
from nsys_timeline_v22 import clip, intersection, length, merge


def require(value, message):
    if not value:
        raise AssertionError(message)


def statistics(values):
    values = sorted(values)
    require(bool(values) and all(math.isfinite(v) for v in values), 'Missing/invalid samples')
    def percentile(q):
        index = (len(values)-1)*q
        lo, hi = math.floor(index), math.ceil(index)
        return values[lo]+(values[hi]-values[lo])*(index-lo)
    return dict(count=len(values), mean=sum(values)/len(values), min=values[0],
                median=percentile(.5), p95=percentile(.95), max=values[-1])


def analyze(path, receipt):
    trace = read(receipt)
    require(trace['passed'] and trace['traced'] and trace['error'] is None
            and not trace['telemetry_errors'], 'Incomplete diagnostic receipt')
    db = sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    try:
        require(db.execute('PRAGMA quick_check').fetchone()[0] == 'ok', 'Corrupt trace database')
        names = dict(db.execute('SELECT id,value FROM StringIds'))
        ranges = [dict(r) for r in db.execute(
            "SELECT start,end,text,globalTid FROM NVTX_EVENTS WHERE text LIKE 'seaqr30|%' ORDER BY start")]
        require(len(ranges) == trace['push_pop_counts'][0] == trace['push_pop_counts'][1], 'Lost ranges')
        for r in ranges:
            _, index, r['stage'] = r['text'].split('|')
            r['frame'] = None if index == 'None' else int(index)
            require(r['end'] is not None and r['end'] >= r['start'], 'Incomplete range')
        major = {s: [r for r in ranges if r['stage'] == s] for s in MAJOR}
        for name, rows in major.items():
            require([r['frame'] for r in rows] == list(range(128)), 'Wrong frame coverage '+name)
        tids = {r['globalTid'] for rows in major.values() for r in rows}
        require(len(tids) == 1, 'Major stages must remain serial on one consumer')
        tid = next(iter(tids))
        process = [dict(r) for r in db.execute('SELECT * FROM PROCESSES WHERE pid=?', (trace['pid'],))]
        require(len(process) == 1 and process[0]['globalPid']+trace['pid'] == tid, 'Wrong process')
        pid = process[0]['globalPid']
        data = {k: [dict(r) for r in db.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_'+k+' WHERE globalPid=?', (pid,))]
                for k in ('KERNEL', 'MEMCPY', 'MEMSET')}
        runtime = [dict(r) for r in db.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE globalTid=?', (tid,))]
        diagnostics = [dict(r) for r in db.execute('SELECT * FROM DIAGNOSTIC_EVENT')]
        bad = [r for r in diagnostics if (r['severity'] >= 3 and r['globalPid'] in (None, pid))
               or any(w in r['text'].lower() for w in ('dropped', 'overflow', 'lost events', 'incomplete trace'))]
        require(not bad, 'Trace error/loss: '+repr(bad))
        scheduling = defaultdict(list)
        for r in db.execute('SELECT * FROM SCHED_EVENTS WHERE globalTid>=? AND globalTid<? ORDER BY start', (pid, pid+(1<<24))):
            scheduling[r['globalTid']].append(dict(r))
        kinds = dict(db.execute('SELECT id,label FROM ENUM_CUDA_MEMCPY_OPER'))
        windows = {}
        for first in (0, 32):
            start, end = major['motion_reference'][first]['start'], major['motion_reference'][127]['start']
            frames, scale = 127-first, (127-first)*1e6
            intervals = {k: clip([(r['start'], r['end']) for r in rows], start, end) for k, rows in data.items()}
            active = merge([p for rows in intervals.values() for p in rows])
            kernels, copies = defaultdict(list), defaultdict(list)
            for r in data['KERNEL']:
                if start <= r['start'] < end:
                    require(r['end'] <= end, 'Kernel crosses complete frame boundary')
                    kernels[names[r['shortName']]].append(r)
            expected = dict(gaussian5=2, median5=1, residual_prepare=1, front_erode=2,
                front_eligible=1, front_noise=1, select_peaks=1, front_learn=1, finish_state=1, warp_cubic=1)
            for name, multiplier in expected.items():
                require(len(kernels[name]) == multiplier*frames, 'Missing kernel coverage '+name)
            for r in data['MEMCPY']:
                if start <= r['start'] < end:
                    require(r['end'] <= end, 'Copy crosses complete frame boundary')
                    copies[kinds[r['copyKind']]].append(r)
            stages = defaultdict(list)
            for r in ranges:
                if r['frame'] is not None and first <= r['frame'] < 127:
                    stages[r['stage']].append((r['start'], r['end']))
            host_rows = defaultdict(list)
            for r in trace['events']:
                if r['frame'] is not None and first <= r['frame'] < 127:
                    host_rows[r['name']].append(r)
            api = defaultdict(list)
            for r in runtime:
                if r['start'] < end and r['end'] > start:
                    api[names[r['nameId']]].append((max(start, r['start']), min(end, r['end'])))
            scheduled = {t: clip([(a['start'], b['start']) for a, b in zip(rows, rows[1:]) if a['isSchedIn']], start, end)
                         for t, rows in scheduling.items()}
            stages_result = {}
            for name, rows in stages.items():
                stages_result[name] = dict(calls=len(rows), host_ms_per_frame=length(rows)/scale,
                    coincident_cuda_ms_per_frame=length(intersection(rows, active))/scale,
                    thread_cpu_ms_per_frame=sum(r['thread_cpu_ns'] for r in host_rows[name])/scale,
                    process_cpu_ms_per_frame=sum(r['process_cpu_ns'] for r in host_rows[name])/scale,
                    runtime_api_ms_per_frame={n: length(intersection(rows, v))/scale for n, v in api.items()})
            # Telemetry uses perf_counter_ns rather than Nsight timestamps. Match
            # by explicit corresponding motion frame boundaries on its own clock.
            own_motion = sorted([r for r in trace['events'] if r['name'] == 'motion_reference'], key=lambda r: r['frame'])
            own_start, own_end = own_motion[first]['start_ns'], own_motion[127]['start_ns']
            samples = [r for r in trace['telemetry'] if own_start <= r['monotonic_ns'] <= own_end]
            sensors = defaultdict(list)
            for r in samples:
                for name, value in r['sensors'].items():
                    if isinstance(value, int):
                        sensors[name].append(value)
            windows[f'{first}_126'] = dict(frames=frames, frame_interval_ms=(end-start)/scale,
                cuda_active_union_ms_per_frame=length(active)/scale, cuda_active_fraction=length(active)/(end-start),
                no_observed_cuda_ms_per_frame=((end-start)-length(active))/scale,
                device_union_ms_per_frame={k: length(v)/scale for k, v in intervals.items()},
                kernels={k: dict(calls=len(v), ms_per_frame=sum(r['end']-r['start'] for r in v)/scale) for k, v in kernels.items()},
                copies={k: dict(calls=len(v), ms_per_frame=sum(r['end']-r['start'] for r in v)/scale,
                               bytes_per_frame=sum(r['bytes'] for r in v)/frames) for k, v in copies.items()},
                stages=stages_result, cuda_api={k: dict(calls=len(v), ms_per_frame=length(v)/scale) for k, v in api.items()},
                thread_scheduled_ms_per_frame={str(t-pid): length(v)/scale for t, v in scheduled.items()},
                consumer_tid=trace['pid'], telemetry_samples=len(samples),
                sensor_statistics={k: statistics(v) for k, v in sensors.items()})
        return dict(windows=windows, runtime_before=trace['runtime_before'], runtime_after=trace['runtime_after'],
            sensor_types=trace['sensor_types'], diagnostics=diagnostics, complete=True,
            expected_kernel_counts_passed=True, sqlite_sha256=sha(path), receipt_sha256=sha(receipt),
            caveat='Instrumented windows only. CUDA activity union is not SM utilization; scheduled CPU time may include spinning. '
                   'Host API waits overlap device execution. Nested stages and process CPU measurements overlap; never add them.')
    finally:
        db.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sqlite', type=Path, required=True)
    p.add_argument('--receipt', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    write(args.output, analyze(args.sqlite, args.receipt))
