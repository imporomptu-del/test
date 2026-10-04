"""Bounded RAW16 development profiling; never opens a split or scans media folders.

Profile/audit modes instrument production methods and are NOT throughput runs.
Timed mode calls the uninstrumented production entry point. All modes preserve
the old Phase 19 policy: this is execution evidence, not an accuracy evaluation.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import cProfile
from dataclasses import fields, is_dataclass
import hashlib
import json
import math
from pathlib import Path
import platform
import pstats
import sys
import time
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tiny_target import dense_screen as dense
from tiny_target.frame_source import FfmpegVideoSource

MEDIA_ROOT = Path('/home/serg/project/camera_reader_sky/srcsky/chunks_raw16_test')
ALLOWED_CLIPS = ('0029', '0040')
CONFIG = ROOT / 'configs/evaluation/phase19_dense_screen_v1.json'
MOTION = ROOT / 'configs/tiny_target_phase12_cfar_test.yaml'
LIBRARY = ROOT / 'build/cuda/libtiny_target_cuda.so'
FROZEN_HASHES = {
    CONFIG: 'e9eb5d86e64beb8bcaf3ffb77967120e1745b16838eff9722aa49657e940a8ed',
    MOTION: '473b19b76f9a25035bf7b5d7f02b899144f3e3cd8369706712a012df350de4fe',
    LIBRARY: 'e29dc8bae949e41497aff82cfa2fe07d52e1c039b5c337187bdb8c2e87b1fc65',
}
# Only known wall-time/environment fields. Timestamps, window duration, numerical
# statistics, launch geometry and ALL detection/tracking decisions stay included.
NON_SEMANTIC = frozenset({
    'timings_ms', 'timing_ms', 'synthetic_tracking_timings_ms', 'candidate_timings_ms',
    'window_timing_ms', 'candidate_timing_ms', 'median_pva_total_ms',
    'median_stabilization_total_ms', 'effective_memory_bandwidth_gib_s',
    'free_device_bytes_before', 'free_device_bytes_after_allocations',
    'velocity_hypotheses_per_second', 'effective_pixel_samples_per_second',
})


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_paths(clip):
    if clip not in ALLOWED_CLIPS:
        raise ValueError('Only the two explicitly authorized development IDs are allowed')
    return (MEDIA_ROOT / f'chunk_{clip}.mkv',
            MEDIA_ROOT / f'chunk_{clip}_timestamps.csv')


def compact(value):
    """Lossless identity of arrays plus all non-timing scalar state."""
    if isinstance(value, np.ndarray):
        contiguous = np.ascontiguousarray(value)
        return dict(shape=list(value.shape), dtype=value.dtype.str,
                    sha256=hashlib.sha256(memoryview(contiguous).cast('B')
                                          if contiguous.size else b'').hexdigest())
    if isinstance(value, np.generic):
        return compact(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return {'nonfinite_float': repr(value)}
    if is_dataclass(value):
        return {f.name: compact(getattr(value, f.name)) for f in fields(value)
                if f.name not in NON_SEMANTIC}
    if isinstance(value, dict):
        return {str(k): compact(v) for k, v in value.items() if k not in NON_SEMANTIC}
    if isinstance(value, (list, tuple)):
        return [compact(v) for v in value]
    return value


def semantic_report(report):
    effective = dict(report['configuration']['effective'])
    effective.pop('background_execution', None)
    return compact(dict(screening=report['screening'], source=report['source'],
                        configuration=effective, injection=report['injection']))


class Instrumentation:
    def __init__(self, mode, events):
        self.mode = mode
        self.events = events
        self.stages = {}
        self.stack = []
        self.frames = 0
        self.event_count = 0

    def record(self, name, value):
        if self.mode == 'audit':
            self.events.write(json.dumps(dict(stage=name, value=compact(value)),
                                         sort_keys=True, allow_nan=False) + '\n')
            self.event_count += 1

    def call(self, name, fn, *args, **kwargs):
        if self.mode != 'profile':
            return fn(*args, **kwargs)
        entry = [time.perf_counter_ns(), 0]
        self.stack.append(entry)
        try:
            return fn(*args, **kwargs)
        finally:
            elapsed = time.perf_counter_ns() - entry[0]
            self.stack.pop()
            if self.stack:
                self.stack[-1][1] += elapsed
            stage = self.stages.setdefault(name, dict(calls=0, inclusive_ms=0., exclusive_ms=0.))
            stage['calls'] += 1
            stage['inclusive_ms'] += elapsed / 1e6
            stage['exclusive_ms'] += (elapsed - entry[1]) / 1e6

    def install(self, context):
        def wrap(owner, method, stage, after=None):
            original = getattr(owner, method)

            def wrapped(*args, **kwargs):
                result = self.call(stage, original, *args, **kwargs)
                if self.mode == 'audit' and after is not None:
                    self.record(stage, after(args, result))
                return result

            context.enter_context(patch.object(owner, method, wrapped))

        original_iter = FfmpegVideoSource.__iter__

        def source_iterator(source):
            iterator = original_iter(source)
            try:
                while True:
                    try:
                        frame = self.call('decode_and_frame_contract', next, iterator)
                    except StopIteration:
                        break
                    if frame.image.dtype != np.dtype('<u2') or frame.bit_depth != 16:
                        raise ValueError('Native uint16 input contract was violated')
                    if frame.image.shape != (3190, 4784):
                        raise ValueError('Unexpected source geometry')
                    if frame.source_timestamp_ns is None:
                        raise ValueError('Recorded source timestamp is required')
                    self.frames += 1
                    self.record('source_frame', frame)
                    yield frame
                    if self.frames % 16 == 0:
                        print(json.dumps(dict(completed_frames=self.frames, mode=self.mode)), flush=True)
            finally:
                iterator.close()

        context.enter_context(patch.object(FfmpegVideoSource, '__iter__', source_iterator))
        wrap(dense.PvaPyrLkMotionEstimator, 'estimate', 'pva_motion', lambda a, r: r)
        wrap(dense, 'fit_global_motion', 'global_fit', lambda a, r: r)
        wrap(dense.FullResolutionStabilizer, 'stabilize', 'full_resolution_warp', lambda a, r: r)
        wrap(dense.StabilizedCropSource, '_crop', 'crop', lambda a, r: r)
        wrap(dense.DensePointScreener, '_events_for_frame', 'background_and_filter',
             lambda a, r: dict(events=r, location=a[0]._background_location,
                 variance=a[0]._background_variance, support=a[0]._background_support,
                 matched=a[0]._last_synthetic_frame))
        wrap(dense.CudaShiftAndStack, 'integrate', 'cuda_shift_stack', lambda a, r: r)
        wrap(dense.CandidateExtractor, 'ranking_surface', 'candidate_ranking', lambda a, r: r)
        wrap(dense.CandidateExtractor, 'extract', 'candidate_extract', lambda a, r: r)
        wrap(dense.DensePointScreener, '_associate_synthetic_candidates', 'synthetic_association',
             lambda a, r: dict(active=a[0]._synthetic_active_tracks,
                              qualified=a[0]._synthetic_qualified_tracks))
        wrap(dense.DensePointScreener, '_update_synthetic_tracking', 'synthetic_orchestration')
        wrap(dense.DensePointScreener, 'process', 'screen_orchestration')
        wrap(dense.DensePointScreener, 'finalize', 'finalize', lambda a, r: r)


def write_json(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')


def run(args):
    if not 32 <= args.frames <= 128:
        raise ValueError('Bounded trial must contain 32..128 consecutive frames')
    video, sidecar = source_paths(args.clip)
    for path, expected in FROZEN_HASHES.items():
        if sha(path) != expected:
            raise ValueError(f'Frozen configuration/library changed: {path}')
    if video.is_symlink() or sidecar.is_symlink():
        raise ValueError('Development source may not be redirected by a symlink')
    observation = dict(mode=args.mode, clip=args.clip, requested_frames=args.frames,
        background_execution=args.background_execution,
        source=dict(path=str(video), size_bytes=video.stat().st_size,
                    mtime_ns=video.stat().st_mtime_ns, sidecar_sha256=sha(sidecar)),
        frozen_sha256={str(p.relative_to(ROOT)): sha(p) for p in FROZEN_HASHES},
        package_sha256={str(p.relative_to(ROOT)): sha(p)
                        for p in sorted((ROOT/'tiny_target').rglob('*.py'))
                        if not p.name.startswith('._')},
        script_sha256=sha(__file__), python=platform.python_version(), numpy=np.__version__,
        passed=False, warning='Unlabeled legacy discovery policy, not a real-object accuracy test. '
        'Profile/audit durations include instrumentation; only timed mode measures throughput.')
    args.output.mkdir(parents=True, exist_ok=False)
    execution_config = json.loads(CONFIG.read_text())
    # The original frozen policy and CUDA binary remain byte-identical. This
    # explicit execution switch is the sole additional configuration field.
    execution_config['background_execution'] = args.background_execution
    execution_path = args.output/'execution_config.json'
    write_json(execution_path, execution_config)
    observation['execution_config_sha256'] = sha(execution_path)
    profiler = cProfile.Profile()
    started = time.perf_counter()
    try:
        with ExitStack() as context:
            events = context.enter_context((args.output/'audit.jsonl').open('x')) if args.mode == 'audit' else None
            instrumentation = Instrumentation(args.mode, events)
            if args.mode != 'timed':
                instrumentation.install(context)
            if args.mode == 'profile':
                profiler.enable()
            report = dense.screen_video(execution_path, video, timestamp_csv=sidecar,
                motion_config_path=MOTION, max_frames=args.frames, bit_depth=16)
            profiler.disable()
            if report['screening']['frames_seen'] != args.frames:
                raise ValueError('Unexpected frame count')
            instrumentation.record('semantic_report', semantic_report(report))
        write_json(args.output/'report.json', report)
        observation.update(passed=True, wall_seconds=time.perf_counter()-started,
            instrumented_stages=instrumentation.stages, audit_events=instrumentation.event_count,
            frames_seen=report['screening']['frames_seen'], performance=report['performance'],
            pva=report['source']['pva_stabilization']['metrics'])
        if args.mode == 'profile':
            stats = pstats.Stats(profiler)
            entries = [dict(file=k[0], line=k[1], function=k[2], calls=v[1],
                            self_ms=v[2]*1000, cumulative_ms=v[3]*1000)
                       for k, v in stats.stats.items()]
            observation['python_profile'] = sorted(entries, key=lambda row: -row['self_ms'])
        if args.mode == 'audit':
            observation['audit_sha256'] = sha(args.output/'audit.jsonl')
    except BaseException as exc:
        observation.update(error=repr(exc), wall_seconds=time.perf_counter()-started)
        raise
    finally:
        profiler.disable()
        write_json(args.output/'observation.json', observation)
    print(json.dumps({k: v for k, v in observation.items()
                      if k not in ('package_sha256', 'python_profile')}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clip', choices=ALLOWED_CLIPS, required=True)
    parser.add_argument('--frames', type=int, default=64)
    parser.add_argument('--mode', choices=('profile', 'audit', 'timed'), required=True)
    parser.add_argument('--background-execution', choices=('indexed_reference', 'masked_ufunc'),
                        default='indexed_reference')
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
