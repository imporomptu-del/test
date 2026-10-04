"""Bounded real-pipeline prefix with individual GPU events and exact-output gate."""
import argparse
import ctypes as C
from dataclasses import asdict, replace
from itertools import islice
import json
from pathlib import Path
import subprocess
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
    generated = args.library.parent/(args.library.stem+'_sources')
    for name, expected in build['generated_sha256'].items():
        if visible.sha256(generated/name) != expected:
            raise ValueError('Generated probe source changed')
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
    probe.seaqr_kernel_probe_enable.argtypes = [C.c_int]
    probe.seaqr_kernel_probe_enable.restype = C.c_int
    probe.seaqr_kernel_probe_read.argtypes = [C.c_void_p,C.c_void_p,C.c_int]
    probe.seaqr_kernel_probe_read.restype = C.c_int
    probe.seaqr_kernel_probe_close.argtypes = []
    probe.seaqr_kernel_probe_close.restype = None
    # Allocate events before any video processing; no per-frame allocations.
    if probe.seaqr_kernel_probe_enable(1) or probe.seaqr_kernel_probe_enable(0):
        raise RuntimeError('Cannot initialize GPU timing events')
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
        caveat='Events bracket individual default-stream launches without per-kernel synchronization. Instrumented slice, not throughput or utilization/occupancy. Event/launch scheduling overhead may affect short kernels. Native detection/tracking and causal feedback are unchanged.')
    original = visible.PvaMotion.update
    def update(self, gray, frame_index, timestamp_ns):
        if frame_index == 72 and probe.seaqr_kernel_probe_enable(1):
            raise RuntimeError('Cannot enable GPU timing')
        return original(self, gray, frame_index, timestamp_ns)
    telemetry = None
    try:
        with (args.output/'tegrastats.log').open('x') as log:
            telemetry = subprocess.Popen(['/usr/bin/tegrastats','--interval','1000'], stdout=log, stderr=subprocess.STDOUT)
            with patch.object(visible.PvaMotion, 'update', update):
                visible.run(args.source, config_path, args.output/'run', args.motion_config, max_frames=96)
        if probe.seaqr_kernel_probe_enable(0):
            raise RuntimeError('Cannot disable GPU timing')
        ids = np.empty(1024, np.int32)
        durations = np.empty(1024, np.float32)
        n = probe.seaqr_kernel_probe_read(ids.ctypes.data, durations.ctypes.data, len(ids))
        if not 0 < n <= len(ids):
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
        rows = []
        for key in sorted(set(map(int, ids[:n]))):
            values = durations[:n][ids[:n]==key].astype(float)
            rows.append(dict(kernel=build['sites'][str(key)], calls=len(values),
                total_ms=float(values.sum()), ms_per_profiled_frame=float(values.sum())/24,
                median_ms=float(np.median(values)), p95_ms=float(np.percentile(values,95)),
                samples_ms=values.tolist()))
        required = {'warp_u8','gaussian_horizontal','gaussian_vertical','median5',
            'residual_prepare','gather_samples','select_peaks','finish_state'}
        if any(r['calls']!=24 for r in rows if r['kernel'] in required) or not required <= {r['kernel'] for r in rows}:
            raise ValueError('Unexpected profiled kernel call counts')
        record.update(passed=True, exact_prefix_frames=96, exact_fields='All non-timing journal fields',
            kernels=rows, profiled_frame_count=24, total_kernel_ms_per_frame=sum(r['ms_per_profiled_frame'] for r in rows),
            no_gpu_power_clock_or_service_changes=True)
    except Exception as exc:
        record['error'] = repr(exc)
        raise
    finally:
        if telemetry is not None:
            telemetry.terminate()
            telemetry.wait(timeout=10)
        probe.seaqr_kernel_probe_enable(0)
        probe.seaqr_kernel_probe_close()
        with (args.output/'kernel_profile.json').open('x') as f:
            json.dump(record, f, indent=2)
    print(json.dumps({k:v for k,v in record.items() if k != 'probe_build'}, indent=2))


if __name__ == '__main__':
    main()
