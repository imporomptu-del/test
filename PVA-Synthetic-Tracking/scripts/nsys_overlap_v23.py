"""Read-only, common-clock diagnosis of bounded v23 two-thread Nsight traces.

Usage: --sqlite trace.sqlite --receipt trace.v23.json --output analysis.json
The adjacent trace/launch.json must accompany the receipt. No media is opened.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sqlite3

from nsys_timeline_v22 import clip, intersection, length, merge


CUDA_SHA256 = '0877b81e2332c329c3bbae2d07a0db5b615c821aa11943c6a68c797f37bdc197'
EXPECTED = {'gaussian5': 2, 'warp_cubic': 1, 'median5': 1,
            'residual_prepare': 1, 'gather_samples': 1,
            'select_peaks': 1, 'finish_state': 1}
STAGES = ('motion_worker', 'detector', 'tracking')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def validate_receipt(path):
    path = Path(path)
    receipt = json.loads(path.read_text())
    _require(path.name.endswith('.v23.json'), 'Expected adjacent .v23.json receipt')
    _require(receipt.get('schema') == 'seaqr.visible-overlap-v23.v1'
             and receipt.get('passed') is True and receipt.get('error') is None
             and receipt.get('traced') is True and receipt.get('mode') == 'overlap'
             and receipt.get('frames') == receipt.get('processed_frames') == 128
             and receipt.get('clip') in ('0126', '0082'), 'Not a successful bounded overlap trace')
    _require(all(receipt.get(k) is False for k in ('gpu_changed', 'algorithm_changed',
                 'raw16_accessed', 'defaults_changed')), 'Frozen scope flags changed/missing')
    _require(receipt.get('nvtx_push_pop_counts') == [384, 384], 'Receipt NVTX counts are incomplete')
    snap = receipt['execution']
    _require(snap['mode'] == 'overlap' and [r['frame'] for r in snap['frames']] == list(range(128)),
             'Receipt frame coverage differs')
    engine = snap['engine']
    _require(engine['closed'] and engine['joined'] and engine['discarded_on_cancel'] == 0
             and engine['prepared'] == engine['delivered'] == engine['released'] == 128
             and 0 < engine['max_owned'] <= 2, 'Incomplete ownership lifecycle')
    stem = path.name[:-len('.v23.json')]
    launch_path = path.parent / stem / 'launch.json'
    launch = json.loads(launch_path.read_text())
    # These are receipt identities, not a new inspection of the remote .so.
    _require(launch['configuration']['input_bit_depth'] == 8
             and launch['source'].endswith('/chunk_' + receipt['clip'] + '.avi'),
             'Unexpected source contract in launch receipt')
    _require(launch['external_accelerators']['median']['library_sha256'] == CUDA_SHA256
             and launch['exact_cuda_stabilization']['library_sha256'] == CUDA_SHA256,
             'Original CUDA detector/warp library identities changed')
    baseline = path.parent / (stem + '.v20.json')
    _require(sha(baseline) == receipt['baseline_receipt_sha256'], 'Linked v20 receipt changed')
    old = json.loads(baseline.read_text())
    _require(old['passed'] and old['error'] is None and old['processed_frames'] == 128,
             'Linked baseline parity receipt failed')
    return receipt, dict(receipt_sha256=sha(path), launch_sha256=sha(launch_path),
                        baseline_receipt_sha256=sha(baseline),
                        original_cuda_library_sha256=CUDA_SHA256,
                        evidence_scope='Receipt chain and launch identity checked; independent '
                        'full journal/motion parity verification remains separate.')


def validate_ranges(rows, mode='overlap'):
    """Reject wrong coverage, nesting, cross-thread endings and unsafe ordering."""
    _require(mode == 'overlap', 'Only candidate overlap traces are accepted')
    frames = defaultdict(dict)
    for row in rows:
        parts = row['text'].split('|')
        _require(len(parts) == 3 and parts[0] == 'seaqr23', 'Unexpected annotation label')
        frame, stage = int(parts[1]), parts[2]
        _require(0 <= frame < 128 and stage in STAGES, 'Unexpected frame/stage')
        _require(row['end'] is not None and row['end'] > row['start'], 'Incomplete/reversed NVTX range')
        _require(row.get('endGlobalTid') in (None, 0, row['globalTid']), 'Cross-thread NVTX ending')
        _require(stage not in frames[frame], 'Duplicate annotation')
        row.update(frame=frame, stage=stage)
        frames[frame][stage] = row
    _require(set(frames) == set(range(128)) and all(set(v) == set(STAGES) for v in frames.values()),
             'Missing NVTX frame/stage coverage')
    tids = {stage: {frames[i][stage]['globalTid'] for i in range(128)} for stage in STAGES}
    _require(all(len(v) == 1 for v in tids.values()), 'Stage migrated between OS threads')
    tids = {stage: next(iter(values)) for stage, values in tids.items()}
    _require(tids['detector'] == tids['tracking'] != tids['motion_worker'],
             'Expected one distinct motion worker and one main consumer')
    by_thread = defaultdict(list)
    for row in rows:
        by_thread[row['globalTid']].append(row)
    for thread_rows in by_thread.values():
        thread_rows.sort(key=lambda r: r['start'])
        _require(all(a['end'] <= b['start'] for a, b in zip(thread_rows, thread_rows[1:])),
                 'Stages on one thread overlap/nest')
    for i in range(128):
        motion, detector, tracking = (frames[i][s] for s in STAGES)
        _require(motion['end'] <= detector['start'] and detector['end'] <= tracking['start'],
                 'Motion/detection/tracking dependency order violated')
        if i:
            _require(frames[i-1]['tracking']['end'] <= detector['start']
                     and frames[i-1]['motion_worker']['end'] <= motion['start'],
                     'Frames processed out of order')
        if i >= 2:
            _require(frames[i-2]['tracking']['end'] <= motion['start'],
                     'Motion ran more than one frame ahead of leased consumer')
    return dict(frames), tids


def _pid_for_tid(tid):
    # Nsight serialized GlobalId stores the native TID in the low 24 bits.
    return tid & ~((1 << 24) - 1)


def _rows(db, table, tables):
    return [dict(row) for row in db.execute('SELECT * FROM ' + table)] if table in tables else []


def kernel_coverage(kernels, runtime, ranges, names, pid):
    """Associate launches by correlation AND launching thread, not GPU clock proximity."""
    correlations = defaultdict(list)
    for row in runtime:
        if row['globalTid'] is not None and _pid_for_tid(row['globalTid']) == pid:
            correlations[row['correlationId']].append(row)
    by_tid = defaultdict(list)
    for row in ranges:
        by_tid[row['globalTid']].append(row)
    counts = {i: Counter() for i in range(128)}
    stages = defaultdict(Counter)
    unattributed = Counter()
    for kernel in kernels:
        name = names[kernel['shortName']]
        matches = []
        for api in correlations[kernel['correlationId']]:
            for stage in by_tid[api['globalTid']]:
                if stage['start'] <= api['start'] and api['end'] <= stage['end']:
                    matches.append(stage)
        _require(len(matches) <= 1, 'Ambiguous launch correlation/NVTX attribution')
        if matches:
            stage = matches[0]
            counts[stage['frame']][name] += 1
            stages[stage['stage']][name] += 1
        else:
            unattributed[name] += 1
    mismatches = [{"frame": i, "kernel": name, "expected": expected,
                   "observed": counts[i][name]}
                  for i in range(128) for name, expected in EXPECTED.items()
                  if counts[i][name] != expected]
    _require(not mismatches, 'Missing/unexpected per-frame CUDA kernels: ' + repr(mismatches[:12]))
    return dict(expected_per_frame=EXPECTED, expected_frames=128, passed=True,
                per_frame={str(i): dict(v) for i, v in counts.items()},
                by_stage={k: dict(v) for k, v in stages.items()},
                unattributed_to_frame=dict(unattributed),
                attribution='CUDA runtime correlation ID and launching OS thread contained in '
                'NVTX range. Unattributed events may include initialization/conformance/VPI helpers; '
                'they are retained in process CUDA interval unions, not assigned by proximity.')


def scheduling_metrics(events, tids, start, end, activity, scale):
    """Conservative IN→OUT reconstruction; unknown trace edges are not extended."""
    running, details = {}, {}
    for label, tid in (('main', tids['detector']), ('worker', tids['motion_worker'])):
        rows = sorted((r for r in events if r['globalTid'] == tid), key=lambda r: r['start'])
        valid, malformed = [], []
        for a, b in zip(rows, rows[1:]):
            if a['isSchedIn'] and not b['isSchedIn'] and a['cpu'] == b['cpu']:
                valid.append((a['start'], b['start']))
            elif bool(a['isSchedIn']) == bool(b['isSchedIn']) or (a['isSchedIn'] and a['cpu'] != b['cpu']):
                malformed.append(dict(start_ns=a['start'], end_ns=b['start']))
        running[label] = clip(valid, start, end)
        details[label] = dict(global_tid=tid, events=len(rows),
            first_event_ns=rows[0]['start'] if rows else None,
            last_event_ns=rows[-1]['start'] if rows else None,
            events_enclose_window=bool(rows and rows[0]['start'] <= start and rows[-1]['start'] >= end),
            unmatched_in_at_trace_end=bool(rows and rows[-1]['isSchedIn']),
            malformed_transition_count=len(malformed), malformed_transitions=malformed[:16],
            scheduled_ms_per_interval=length(running[label])/scale,
            scheduled_and_process_cuda_active_ms_per_interval=length(intersection(running[label], activity))/scale)
    both = intersection(running['main'], running['worker'])
    return dict(threads=details,
        both_threads_scheduled_ms_per_interval=length(both)/scale,
        both_threads_scheduled_and_process_cuda_active_ms_per_interval=length(intersection(both, activity))/scale,
        interpretation='Reconstructed adjacent same-CPU scheduled-IN→OUT pairs only. Unknown trace '
        'edges and malformed transitions are not extrapolated. Simultaneous scheduling supports '
        'CPU concurrency but does not prove useful instruction execution, exclude IRQ/preemption '
        'overhead, or identify GIL ownership/contention.')


def duration_summary(rows, start, end, scale):
    """Boundary-clipped sums AND unions; counts are explicit at the boundary."""
    selected = [r for r in rows if r['start'] < end and r['end'] > start]
    _require(all(r['end'] >= r['start'] for r in selected), 'Reversed measured interval')
    bounded = [(max(start,r['start']), min(end,r['end'])) for r in selected]
    return dict(calls_intersecting_window=len(selected),
        calls_starting_in_window=sum(start <= r['start'] < end for r in selected),
        calls_crossing_boundary=sum(r['start'] < start or r['end'] > end for r in selected),
        clipped_sum_ms_per_interval=sum(b-a for a,b in bounded)/scale,
        clipped_union_ms_per_interval=length(bounded)/scale)


def kernel_durations(kernels, names, start, end, scale):
    by_name = defaultdict(list)
    for row in kernels:
        if row['start'] < end and row['end'] > start:
            by_name[names[row['shortName']]].append(row)
    return {name: duration_summary(rows,start,end,scale) for name,rows in sorted(by_name.items())}


def runtime_durations(runtime, names, tids, start, end, activity, scale):
    result = {}
    for label, tid in (('main',tids['detector']), ('worker',tids['motion_worker'])):
        by_name = defaultdict(list)
        selected = [r for r in runtime if r['globalTid'] == tid and r['start'] < end and r['end'] > start]
        for row in selected:
            by_name[names[row['nameId']]].append(row)
        functions = {}
        for name, rows in sorted(by_name.items()):
            value = duration_summary(rows,start,end,scale)
            bounded = clip([(r['start'],r['end']) for r in rows],start,end)
            value.update(coincident_process_cuda_ms_per_interval=length(intersection(bounded,activity))/scale,
                nonzero_return_value_calls=sum(r.get('returnValue',0) != 0 for r in rows))
            functions[name] = value
        all_bounded = clip([(r['start'],r['end']) for r in selected],start,end)
        result[label] = dict(global_tid=tid, all_apis=duration_summary(selected,start,end,scale),
            all_apis_coincident_process_cuda_ms_per_interval=length(intersection(all_bounded,activity))/scale,
            by_api=functions)
    return dict(threads=result, interpretation='CUDA runtime API durations are host elapsed intervals, '
        'including waits, scheduling, synchronization and possible driver serialization. cudaMemcpy '
        'API time is not device transfer time. Sums may include nested/overlapping calls; use unions '
        'for wall coverage. Main/worker API intervals may overlap each other, GPU work and NVTX '
        'stages, so never add them to those durations. Coincident CUDA can belong to another frame '
        'or thread; neither coincidence nor inflation alone establishes a synchronization or GIL cause.')


def window_metrics(data, ranges, frames, first, scheduling=None, tids=None, runtime=None, names=None):
    last = 127
    start, end = frames[first]['detector']['start'], frames[last]['detector']['start']
    count, scale = last-first, (last-first)*1e6
    intervals = {k: clip([(r['start'], r['end']) for r in rows], start, end)
                 for k, rows in data.items()}
    activity = merge([item for rows in intervals.values() for item in rows])
    stage_intervals = {s: clip([(r['start'], r['end']) for r in ranges if r['stage'] == s], start, end)
                       for s in STAGES}
    pairs = []
    # Pair current main frame i with lookahead worker i+1. Both intervals are
    # clipped to the same clock window; journal gaps are not labeled useful work.
    for i in range(first, last):
        worker = frames[i+1]['motion_worker']
        left = clip([(worker['start'], worker['end'])], start, end)
        main = clip([(frames[i][s]['start'], frames[i][s]['end'])
                     for s in ('detector', 'tracking')], start, end)
        common = intersection(left, main)
        pairs.append(dict(main_frame=i, worker_frame=i+1, host_overlap_ns=length(common),
                          worker_host_ns=length(left)))
    tracking = stage_intervals['tracking']
    result = dict(first_main_frame=first, last_main_frame_exclusive=last, intervals=count,
        start_ns=start, end_ns=end, interval_ms=(end-start)/scale,
        clock='Nsight common timestamp clock only; perf_counter timestamps are not mixed in.',
        boundary='Main detector entry first→127; CUDA operations crossing boundaries are clipped. '
        'All-window means first=0, not startup/initial motion/final drain.',
        cuda_active_union_ms_per_interval=length(activity)/scale,
        cuda_active_fraction=length(activity)/(end-start),
        no_observed_process_cuda_ms_per_interval=((end-start)-length(activity))/scale,
        device_union_ms_per_interval={k: length(v)/scale for k, v in intervals.items()},
        kernel_copy_overlap_ms_per_interval=length(intersection(intervals['KERNEL'], intervals['MEMCPY']))/scale,
        cuda_overlap_with_main_tracking_ms_per_interval=length(intersection(activity, tracking))/scale,
        kernel_overlap_with_main_tracking_ms_per_interval=length(intersection(intervals['KERNEL'], tracking))/scale,
        worker_host_overlap_with_previous_detector_tracking_ms_per_interval=sum(p['host_overlap_ns'] for p in pairs)/scale,
        worker_previous_frame_pairs=pairs,
        stages={s: dict(host_union_ms_per_interval=length(v)/scale,
                        coincident_process_cuda_ms_per_interval=length(intersection(v, activity))/scale)
                for s, v in stage_intervals.items()},
        cuda_rows_intersecting_window={k: sum(r['start'] < end and r['end'] > start for r in rows)
                                      for k, rows in data.items()},
        cuda_rows_crossing_boundary={k: sum((r['start'] < start < r['end']) or
                                           (r['start'] < end < r['end']) for r in rows)
                                    for k, rows in data.items()})
    result['cpu_scheduling'] = None if scheduling is None else scheduling_metrics(
        scheduling, tids, start, end, activity, scale)
    result['kernels'] = None if names is None else kernel_durations(data['KERNEL'],names,start,end,scale)
    result['runtime_cuda_api'] = None if runtime is None else runtime_durations(
        runtime,names,tids,start,end,activity,scale)
    return result


def analyze(sqlite_path, receipt_path):
    receipt, provenance = validate_receipt(receipt_path)
    db = sqlite3.connect(Path(sqlite_path).resolve().as_uri() + '?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    try:
        _require(db.execute('PRAGMA quick_check').fetchone()[0] == 'ok', 'Corrupt SQLite trace')
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        _require({'StringIds', 'NVTX_EVENTS', 'PROCESSES', 'CUPTI_ACTIVITY_KIND_KERNEL',
                  'CUPTI_ACTIVITY_KIND_MEMCPY', 'CUPTI_ACTIVITY_KIND_RUNTIME'} <= tables,
                 'Required CUDA/NVTX/process tables missing')
        names = dict(db.execute('SELECT id,value FROM StringIds'))
        ranges = []
        for row in _rows(db, 'NVTX_EVENTS', tables):
            label = row['text'] if row['text'] is not None else names.get(row.get('textId'), '')
            if label.startswith('seaqr23|'):
                row['text'] = label
                ranges.append(row)
        frames, tids = validate_ranges(ranges)
        processes = _rows(db, 'PROCESSES', tables)
        targets = [p for p in processes if p['globalPid']+p['pid'] == tids['detector']]
        _require(len(targets) == 1, 'Cannot uniquely identify main traced process')
        process, = targets
        pid = process['globalPid']
        _require(_pid_for_tid(tids['motion_worker']) == pid, 'Motion worker belongs to different process')
        data = {k: [r for r in _rows(db, 'CUPTI_ACTIVITY_KIND_'+k, tables) if r['globalPid'] == pid]
                for k in ('KERNEL', 'MEMCPY', 'MEMSET')}
        _require(data['KERNEL'] and data['MEMCPY'], 'No pipeline CUDA activity')
        _require(all(r['end'] >= r['start'] for rows in data.values() for r in rows), 'Reversed CUDA interval')
        diagnostics = _rows(db, 'DIAGNOSTIC_EVENT', tables)
        levels = {r['id']: r['name'].lower() for r in _rows(db, 'ENUM_DIAGNOSTIC_SEVERITY_LEVEL', tables)}
        loss_words = ('dropped', 'overflow', 'lost events', 'incomplete trace', 'trace data loss')
        loss = [r for r in diagnostics if any(w in r['text'].lower() for w in loss_words)]
        errors = [r for r in diagnostics if r.get('globalPid') in (None, pid)
                  and (levels.get(r['severity'], '') in ('error', 'fatal')
                       or (not levels and r['severity'] == 3))]
        _require(not loss and not errors, 'Trace loss/error needs investigation: '+repr(loss+errors))
        runtime = _rows(db, 'CUPTI_ACTIVITY_KIND_RUNTIME', tables)
        coverage = kernel_coverage(data['KERNEL'], runtime, ranges, names, pid)
        scheduling = _rows(db, 'SCHED_EVENTS', tables) if 'SCHED_EVENTS' in tables else None
        return dict(schema='seaqr.nsys-overlap-v23.v1', passed=True, clip=receipt['clip'],
            sqlite_sha256=sha(sqlite_path), analyzer_sha256=sha(__file__), provenance=provenance,
            interval_helpers_sha256=sha(Path(__file__).with_name('nsys_timeline_v22.py')),
            process=process, global_tids=tids, stage_range_count=len(ranges),
            stage_order_and_thread_separation_passed=True, kernel_coverage=coverage,
            process_cuda_table_counts={k: len(v) for k, v in data.items()},
            windows={'all_0_126': window_metrics(data, ranges, frames, 0, scheduling, tids, runtime, names),
                     'steady_32_126': window_metrics(data, ranges, frames, 32, scheduling, tids, runtime, names)},
            default_window='steady_32_126',
            scheduling_table_present='SCHED_EVENTS' in tables,
            diagnostics=diagnostics, diagnostic_table_present='DIAGNOSTIC_EVENT' in tables,
            reported_trace_loss_events=loss,
            absence_of_loss_proves_completeness=False,
            performance_comparison=False,
            limitations=[
                'Host interval overlap is temporal overlap, not proof of simultaneous useful CPU '
                'execution or GIL release. No GIL sampling was collected.',
                'CUDA-active is the union of this process kernels/copies/memsets, not GPU utilization, '
                'SM occupancy, bandwidth utilization, or whole-device busy/idle time.',
                'No PVA hardware trace is available; motion_worker includes host/PVA waits and CUDA work.',
                'Kernel launch attribution is not attribution of all device work to a host interval; '
                'coincident CUDA activity can belong to another frame/stage.',
                'A traced run is diagnostic only. Do not use its FPS or interval times as the '
                'unprofiled reference/candidate speed or queue-latency acceptance comparison.',
                'Expected per-frame kernels and balanced NVTX improve completeness confidence but '
                'cannot prove that every optional/background trace event was captured.'])
    finally:
        db.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sqlite', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = analyze(args.sqlite, args.receipt)
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
