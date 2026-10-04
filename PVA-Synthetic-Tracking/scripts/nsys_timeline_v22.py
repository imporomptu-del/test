"""Read-only analysis of Nsight SQLite events; interval unions avoid double counting."""
from collections import defaultdict
import sqlite3
from pathlib import Path


def merge(intervals):
    result = []
    for start, end in sorted(intervals):
        if end < start:
            raise ValueError('Reversed interval')
        if end == start:
            continue
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return result


def clip(intervals, start, end):
    if end <= start:
        raise ValueError('Invalid window')
    return merge((max(a, start), min(b, end)) for a, b in intervals if b > start and a < end)


def length(intervals):
    return sum(b-a for a, b in merge(intervals))


def intersection(left, right):
    a, b = merge(left), merge(right)
    i = j = 0
    result = []
    while i < len(a) and j < len(b):
        start, end = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if end > start:
            result.append((start, end))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return result


def analyze(path):
    db = sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    try:
        if db.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
            raise AssertionError('Corrupt SQLite trace')
        names = dict(db.execute('SELECT id,value FROM StringIds'))
        ranges = [dict(r) for r in db.execute(
            "SELECT start,end,text,globalTid FROM NVTX_EVENTS WHERE text LIKE 'seaqr|%' ORDER BY start")]
        if any(r['end'] is None or r['end'] < r['start'] for r in ranges):
            raise AssertionError('Incomplete NVTX range')
        for r in ranges:
            _, frame, r['stage'] = r['text'].split('|')
            r['frame'] = None if frame == 'None' else int(frame)
        motion = [r for r in ranges if r['stage'] == 'motion_and_warp']
        if [r['frame'] for r in motion] != list(range(128)):
            raise AssertionError('Incomplete NVTX frame coverage')
        tids = {r['globalTid'] for r in ranges}
        if len(tids) != 1:
            raise AssertionError('Stage ranges must be on the one consumer thread')
        tid = next(iter(tids))
        processes = [dict(r) for r in db.execute('SELECT * FROM PROCESSES')]
        targets = [p for p in processes if p['globalPid']+p['pid'] == tid]
        if len(targets) != 1:
            raise AssertionError('Cannot identify traced pipeline process')
        process = targets[0]
        pid = process['globalPid']
        data = {}
        for kind in ('KERNEL', 'MEMCPY', 'MEMSET'):
            data[kind] = [dict(r) for r in db.execute(
                'SELECT * FROM CUPTI_ACTIVITY_KIND_'+kind+' WHERE globalPid=?', (pid,))]
        if not data['KERNEL'] or not data['MEMCPY']:
            raise AssertionError('CUDA activity missing for the pipeline process')
        runtime = [dict(r) for r in db.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_RUNTIME')]
        diagnostics = [dict(r) for r in db.execute('SELECT * FROM DIAGNOSTIC_EVENT')]
        loss = [r for r in diagnostics if any(w in r['text'].lower() for w in
                                            ('dropped', 'overflow', 'lost events', 'incomplete trace'))]
        errors = [r for r in diagnostics if r['severity'] >= 3 and r['globalPid'] in (None, pid)]
        if errors or loss:
            raise AssertionError('Trace error/loss requires investigation: '+repr(errors+loss))
        copy_kinds = {r['id']:r['label'] for r in db.execute('SELECT * FROM ENUM_CUDA_MEMCPY_OPER')}
        scopes = {}
        for first in (0, 32):
            # Exclude frame 127 because there is no next motion-entry boundary.
            start, end, frames = motion[first]['start'], motion[127]['start'], 127-first
            window = [(start, end)]
            scale = frames*1e6
            intervals = {k:clip([(r['start'],r['end']) for r in rows], start, end)
                         for k, rows in data.items()}
            activity = merge([v for items in intervals.values() for v in items])
            kernels = defaultdict(list)
            for r in data['KERNEL']:
                if start <= r['start'] < end:
                    if r['end'] > end:
                        raise AssertionError('Kernel crossed the selected frame boundary')
                    kernels[names[r['shortName']]].append(r)
            copies = defaultdict(list)
            for r in data['MEMCPY']:
                if start <= r['start'] < end:
                    if r['end'] > end:
                        raise AssertionError('Copy crossed the selected frame boundary')
                    copies[copy_kinds[r['copyKind']]].append(r)
            stages = defaultdict(list)
            for r in ranges:
                if r['frame'] is not None and first <= r['frame'] < 127:
                    stages[r['stage']].append((r['start'], r['end']))
            # Scheduler records are process-tree scope. Reconstruct ONLY the
            # consumer thread, retaining the last pre-window state.
            scheduling = [dict(r) for r in db.execute(
                'SELECT * FROM SCHED_EVENTS WHERE globalTid=? ORDER BY start', (tid,))]
            running = [(a['start'], b['start']) for a,b in zip(scheduling,scheduling[1:]) if a['isSchedIn']]
            running = clip(running, start, end)
            api = defaultdict(list)
            for r in runtime:
                if r['globalTid'] == tid and r['start'] < end and r['end'] > start:
                    api[names[r['nameId']]].append((max(start,r['start']), min(end,r['end'])))
            # Count expected ordinary per-frame kernels. These checks detect
            # missing activities even when no diagnostic loss message exists.
            expected = {'gaussian5':2*frames, 'median5':frames, 'residual_prepare':frames,
                        'gather_samples':frames, 'select_peaks':frames,
                        'finish_state':frames, 'warp_cubic':frames}
            for name,count in expected.items():
                if len(kernels[name]) != count:
                    raise AssertionError('Unexpected kernel coverage: '+name)
            scopes[f'{first}_126'] = dict(frames=frames, start_ns=start, end_ns=end,
                frame_interval_ms=(end-start)/scale,
                cuda_active_union_ms_per_frame=length(activity)/scale,
                cuda_active_fraction=length(activity)/(end-start),
                no_observed_cuda_activity_ms_per_frame=((end-start)-length(activity))/scale,
                device_union_ms_per_frame={k:length(v)/scale for k,v in intervals.items()},
                kernel_copy_overlap_ms_per_frame=length(intersection(intervals['KERNEL'],intervals['MEMCPY']))/scale,
                consumer_scheduled_ms_per_frame=length(running)/scale,
                consumer_scheduled_and_cuda_active_ms_per_frame=length(intersection(running,activity))/scale,
                cpu_schedule_events=len(scheduling),
                kernels={n:dict(calls=len(v), sum_ms_per_frame=sum(r['end']-r['start'] for r in v)/scale)
                         for n,v in kernels.items()},
                copies={n:dict(calls=len(v), bytes_per_frame=sum(r['bytes'] for r in v)/frames,
                              sum_ms_per_frame=sum(r['end']-r['start'] for r in v)/scale)
                        for n,v in copies.items()},
                stages={n:dict(calls=len(v), host_ms_per_frame=length(v)/scale,
                               coincident_cuda_ms_per_frame=length(intersection(v,activity))/scale)
                        for n,v in stages.items()},
                consumer_cuda_api={n:dict(calls=len(v), union_ms_per_frame=length(v)/scale)
                                   for n,v in api.items()})
        return dict(windows=scopes, process=process, stage_range_count=len(ranges),
                    table_counts={k:len(v) for k,v in data.items()}, diagnostics=diagnostics,
                    reported_trace_loss_events=loss, expected_kernel_counts_passed=True,
                    warning='CUDA-active is a union of this process kernel/copy/memset intervals, '
                            'not SM utilization or whole-device busy time. PVA execution, other '
                            'processes and bandwidth counters are not captured. CPU scheduled time '
                            'is not exclusive instruction execution. Stage overlap is temporal, '
                            'not causal attribution. API waits overlap device work; do not add them.')
    finally:
        db.close()
