"""96-frame exact prefix check; transfer profiling only on frames72-95."""
import argparse
import ctypes as C
from dataclasses import asdict, replace
from itertools import islice
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tiny_target import visible_baseline as visible
from compare_phase20_exact_runs import without_timing, shape_accelerator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'config', 'motion-config', 'reference', 'library', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Never overwrite a profiling experiment')
    if (str(args.source) != '/home/serg/project/camera_reader_sky/srcsky/chunks/chunk_0126.avi'
            or visible.sha256(args.source) != 'c5302b873656793da47f1da3c03f05df595f17c3f9bc407ce0bfd99b7e718344'):
        raise ValueError('Only the authorized development126 source is allowed')
    build_path = args.library.with_suffix('.so.build.json')
    build = json.loads(build_path.read_text())
    if not build['diagnostic_only'] or visible.sha256(args.library) != build['library_sha256']:
        raise ValueError('Probe binary/build mismatch')
    for name, expected in build['sources_sha256'].items():
        if visible.sha256(ROOT/'scripts'/name) != expected:
            raise ValueError('Probe source changed: '+name)
    launch = json.loads((args.reference/'launch.json').read_text())
    report = json.loads((args.reference/'report.json').read_text())
    if not report['completed'] or not report['full_clip'] or report['frames'] < 96:
        raise ValueError('A complete full-clip reference is required')
    if (visible.sha256(args.config) != launch['config_sha256']
            or visible.sha256(args.source) != launch['source_sha256']
            or visible.sha256(args.motion_config) != launch['motion_config_sha256']):
        raise ValueError('Reference configuration/source changed')
    for name, expected in launch['package_sha256'].items():
        if visible.sha256(ROOT/'tiny_target'/name) != expected:
            raise ValueError('Reference package changed')
    cfg = visible.VisibleConfig(**json.loads(args.config.read_text()))
    if cfg.stabilization_execution != 'cuda_cubic_resident' or cfg.motion_backend != 'pva':
        raise ValueError('Expected integrated PVA/GPU path')
    if visible.sha256(cfg.cuda_median_library) != launch['exact_cuda_stabilization']['library_sha256']:
        raise ValueError('Reference CUDA binary changed')
    shape_accelerator(launch)
    probe = C.CDLL(str(args.library.resolve()))
    probe.seaqr_transfer_probe_enable.argtypes = [C.c_int]
    probe.seaqr_transfer_probe_enable.restype = None
    probe.seaqr_transfer_probe_read.argtypes = [C.c_void_p]*5
    probe.seaqr_transfer_probe_read.restype = C.c_int
    args.output.mkdir()
    config_path = args.output/'probe_config.json'
    with config_path.open('x') as f:
        json.dump(asdict(replace(cfg, cuda_median_library=str(args.library.resolve()))), f, indent=2)
    record = dict(passed=False, diagnostic_only=True, full_clip_fps_claim=False,
        profiled_frames_inclusive=[72,95], source_sha256=launch['source_sha256'],
        reference_journal_sha256=visible.sha256(args.reference/'frames.jsonl'),
        reference_cuda_library_sha256=launch['exact_cuda_stabilization']['library_sha256'],
        probe_build_sha256=visible.sha256(build_path), probe_build=build,
        script_sha256=visible.sha256(__file__),
        caveat='Instrumentation adds events/synchronization. Host cudaMemcpy time includes waiting for queued GPU work; event-bracketed copy intervals also include API/queue gaps and are not isolated DMA bandwidth. Sync events are not measured. Not GPU utilization, kernel occupancy, or throughput.')
    original = visible.PvaMotion.update
    def update(self, gray, frame_index, timestamp_ns):
        if frame_index == 72:
            probe.seaqr_transfer_probe_enable(1)
        return original(self, gray, frame_index, timestamp_ns)
    try:
        with patch.object(visible.PvaMotion, 'update', update):
            visible.run(args.source, config_path, args.output/'run', args.motion_config, max_frames=96)
        probe.seaqr_transfer_probe_enable(0)
        meta = np.empty((64,3), np.int32)
        calls = np.empty(64, np.uint64); counts = np.empty(64, np.uint64)
        host = np.empty(64, np.float64); events = np.empty(64, np.float64)
        n = probe.seaqr_transfer_probe_read(*(a.ctypes.data for a in (meta,calls,counts,host,events)))
        if not 0 < n <= 64:
            raise ValueError('Missing/invalid probe records')
        with (args.reference/'frames.jsonl').open() as a, (args.output/'run/frames.jsonl').open() as b:
            before = [json.loads(r) for r in islice(a,96)]
            after = [json.loads(r) for r in b]
        if len(before) != 96 or len(after) != 96:
            raise ValueError('Truncated probe/reference prefix')
        differences = [i for i,(a,b) in enumerate(zip(before,after)) if without_timing(a) != without_timing(b)]
        if differences:
            record['differing_frames'] = differences
            raise AssertionError('Probe changed non-timing output')
        source_names = {1:'phase20_cuda_warp_exact.cu', 2:'phase20_cuda_resident.cu',
            3:'phase20_cuda_integrated.cu', 4:'phase20_cuda_median.cu'}
        kinds = {-1:'device_synchronize', 1:'host_to_device', 2:'device_to_host', 3:'device_to_device'}
        rows = []
        for i in range(n):
            source,line,kind = map(int,meta[i])
            rows.append(dict(source=source_names[source], line=line, operation=kinds[kind], calls=int(calls[i]),
                total_bytes=int(counts[i]), bytes_per_profiled_frame=int(counts[i])/24,
                host_api_ms_per_frame=float(host[i])/24,
                event_copy_interval_ms_per_frame=None if kind == -1 else float(events[i])/24))
        record.update(passed=True, exact_prefix_frames=96, exact_fields='All non-timing journal fields',
            calls=rows, profiled_frame_count=24,
            copy_bytes_per_frame=sum(r['bytes_per_profiled_frame'] for r in rows),
            copy_event_intervals_ms_per_frame=sum(r['event_copy_interval_ms_per_frame'] or 0 for r in rows),
            host_copy_api_ms_per_frame=sum(r['host_api_ms_per_frame'] for r in rows if r['operation'] != 'device_synchronize'),
            host_device_synchronize_ms_per_frame=sum(r['host_api_ms_per_frame'] for r in rows if r['operation'] == 'device_synchronize'))
    except Exception as exc:
        record['error'] = repr(exc)
        raise
    finally:
        probe.seaqr_transfer_probe_enable(0)
        with (args.output/'transfer_profile.json').open('x') as f:
            json.dump(record, f, indent=2)
    print(json.dumps({k:v for k,v in record.items() if k not in ('calls','probe_build')}, indent=2))


if __name__ == '__main__':
    main()
