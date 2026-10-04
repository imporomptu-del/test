"""Read-only runtime profiling of the frozen v6 64-frame development prefixes.

No detector/runtime changes, new media IDs, split reads, or injected controls.
Nested wall spans are additive only through their exclusive times. CUDA event
times are reported separately and must not be added to enclosing host times.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import ExitStack
import cProfile
from functools import wraps
import hashlib
import json
from pathlib import Path
import pstats
import sys
import tarfile
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
import run_raw16_cpu_v6 as v6
from profile_raw16_efficiency import sha, source_paths, write_json
from summarize_raw16_cpu_v6 import compare, compare_source_motion
from tiny_target import dense_screen as dense
from tiny_target.detection import candidates
from tiny_target.frame_source import FfmpegVideoSource
from tiny_target.types import Frame

RUNTIME_SHA = '880de3831c17a2f6bae92b2a8f32b54c2575608f5e2a467890c2c2373b7717aa'
BASELINE_HASHES = {
    '0040': {
        'report.json': 'bd1376a1fac2da6f2b559db8df7d43f09f4c0a23362ae274b193153bc7a70536',
        'source_frames.json': 'b1c4670167eb3fa6880ea28c4ea4091a48989cb2aa724cb767de29aa88f5e577',
        'motion_profile.json': '73062b5c887026db2e1c8497f62800a4d8a83d0c1a1e0623ca6f5defbf765767',
        'checks.json': '2749541405531a0cee6a790471b569623682e8c50d7201f9acdbab4d52c0c6a8',
    },
    '0029': {
        'report.json': '20b6bb93c0462afa1e4f5b3b582580514c707499c6f19ec049bce5dadb4f2e34',
        'source_frames.json': 'd51922a27c322ead130f821d6337da7e35bb6723d7f62a05555e54a67deedb3d',
        'motion_profile.json': '4e8ee147f5a16bb193296f6a07a8e7c52b9f7ae1597fbd782b826b105311a220',
        'checks.json': '4fa6766383773a048349c1173af00b99e5d3af22b473d2a4acee99392aeb1ed3',
    },
}


class Spans:
    """Single-thread host spans; suspended generators are never timed as work."""

    def __init__(self, clock=time.perf_counter_ns):
        self.clock = clock
        self.stack = []
        self.nodes = {}

    def call(self, name, group, fn, *args, **kwargs):
        parent = self.stack[-1] if self.stack else None
        group = group or (parent['group'] if parent else 'other')
        path = (*parent['path'], name) if parent else (name,)
        entry = dict(start=self.clock(), children=0, path=path, group=group)
        self.stack.append(entry)
        try:
            return fn(*args, **kwargs)
        finally:
            elapsed = self.clock() - entry['start']
            self.stack.pop()
            if parent is not None:
                parent['children'] += elapsed
            node = self.nodes.setdefault(path, dict(group=group, calls=0,
                inclusive_ns=0, exclusive_ns=0, samples_ns=[]))
            node['calls'] += 1
            node['inclusive_ns'] += elapsed
            node['exclusive_ns'] += elapsed - entry['children']
            node['samples_ns'].append(elapsed)

    def iterator(self, name, group, iterator):
        try:
            while True:
                try:
                    value = self.call(name, group, next, iterator)
                except StopIteration:
                    return
                yield value
        finally:
            self.call(name + '_close', group, iterator.close)

    def wrap(self, context, owner, name, group=None, label=None):
        original = getattr(owner, name)

        @wraps(original)
        def measured(*args, **kwargs):
            return self.call(label or name, group, original, *args, **kwargs)

        context.enter_context(patch.object(owner, name, measured))

    def summary(self):
        if self.stack:
            raise ValueError('Cannot summarize unfinished spans')
        roots = [node for path, node in self.nodes.items() if len(path) == 1]
        if len(roots) != 1 or roots[0]['calls'] != 1:
            raise ValueError('Exactly one complete root call is required')
        total = roots[0]['inclusive_ns']
        groups = defaultdict(int)
        rows = []
        for path, node in sorted(self.nodes.items()):
            if node['exclusive_ns'] < 0:
                raise ValueError('Negative exclusive span')
            groups[node['group']] += node['exclusive_ns']
            samples = sorted(node['samples_ns'])
            rows.append(dict(path='/'.join(path), group=node['group'], calls=node['calls'],
                inclusive_s=node['inclusive_ns']/1e9, exclusive_s=node['exclusive_ns']/1e9,
                minimum_call_s=samples[0]/1e9, maximum_call_s=samples[-1]/1e9))
        summed = sum(groups.values())
        if total <= 0 or summed != total:
            raise ValueError('Exclusive accounting does not match root wall time')
        return dict(wall_s=total/1e9, exclusive_sum_s=summed/1e9,
            accounting_error_ns=summed-total,
            groups={key:dict(exclusive_s=value/1e9, fraction=value/total)
                    for key, value in sorted(groups.items(), key=lambda row:-row[1])},
            spans=rows)

    def install(self, context):
        original_iter = FfmpegVideoSource.__iter__

        def measured_source(source):
            # Enter/exit around next(), not around yield: consumer processing
            # time must not appear in decode time.
            return self.iterator('decode_next', 'decode_delivery', original_iter(source))

        context.enter_context(patch.object(FfmpegVideoSource, '__iter__', measured_source))
        for owner, name, group in (
            (dense, 'screen_video', 'setup_and_orchestration'),
            (dense.PvaPyrLkMotionEstimator, 'estimate', 'motion_correspondences'),
            (dense, 'fit_global_motion', 'global_motion_fit'),
            (dense.FullResolutionStabilizer, 'stabilize', 'stabilization'),
            (dense.FullResolutionStabilizer, '_warp_cpu', 'stabilization'),
            (dense.StabilizedCropSource, '_crop', 'stabilization'),
            (dense.DensePointScreener, 'process', 'setup_and_orchestration'),
            (dense.DensePointScreener, '_events_for_frame', 'background_and_filter'),
            (dense.DensePointScreener, '_record_availability', 'availability_diagnostics'),
            (dense.DensePointScreener, '_update_synthetic_tracking', 'setup_and_orchestration'),
            (dense.ReferenceSyntheticWindow, 'update', 'synthetic_window_management'),
            (dense.CudaShiftAndStack, 'integrate', 'synthetic_integration'),
            (dense.CandidateExtractor, 'ranking_surface', 'candidate_ranking'),
            (dense.CandidateExtractor, 'extract', 'candidate_extraction'),
            (dense.DensePointScreener, '_associate_synthetic_candidates', 'track_association'),
            (dense.DensePointScreener, 'finalize', 'finalization'),
            (Frame, 'pixel_sha256', 'evidence_hashing_and_writing'),
            (v6.sequence, 'identity', 'evidence_hashing_and_writing'),
            (v6.full, 'write_json', 'evidence_hashing_and_writing'),
            (v6, 'write_json', 'evidence_hashing_and_writing'),
        ):
            self.wrap(context, owner, name, group)
        # Inherit the enclosing category to avoid charging a warp-time OpenCV
        # call to background filtering, or candidate work to motion.
        for name in ('_tile_robust_cfar', '_local_maxima', '_balanced_top_indices', '_invalid_distance'):
            self.wrap(context, candidates, name)
        cv2 = dense._load_cv2()
        for name in ('filter2D', 'boxFilter', 'warpPerspective', 'erode', 'dilate'):
            self.wrap(context, cv2, name, label='opencv.' + name)


def verify_runtime(archive):
    if sha(archive) != RUNTIME_SHA:
        raise ValueError('Frozen v6 source archive changed')
    verified = {}
    with tarfile.open(archive) as source:
        for member in source:
            if not member.isfile():
                continue
            path = Path(member.name)
            if path.is_absolute() or '..' in path.parts:
                raise ValueError('Unsafe archive member')
            expected = hashlib.sha256(source.extractfile(member).read()).hexdigest()
            if sha(ROOT/path) != expected:
                raise ValueError(f'Runtime/config/harness differs from frozen v6: {path}')
            verified[str(path)] = expected
    package = {str(path.relative_to(ROOT)) for path in (ROOT/'tiny_target').rglob('*.py')
               if not path.name.startswith('._')}
    if package != {name for name in verified if name.startswith('tiny_target/') and name.endswith('.py')}:
        raise ValueError('Runtime module inventory changed')
    return verified


def run(args):
    # Allowlist rejection precedes every media access; the reused v6 harness
    # hard-codes exactly 64 frames and refuses existing output directories.
    source_paths(args.clip)
    if args.output.exists():
        raise FileExistsError(args.output)
    baseline = args.evidence/f'full_frame_{args.clip}_v6'
    for name, expected in BASELINE_HASHES[args.clip].items():
        if sha(baseline/name) != expected:
            raise ValueError(f'Frozen v6 reference changed: {name}')
    frozen = verify_runtime(args.runtime_archive)
    profiler = cProfile.Profile() if args.cprofile else None
    spans = Spans()
    status = None
    error = None
    try:
        with ExitStack() as context:
            spans.install(context)
            if profiler is not None:
                profiler.enable()
            try:
                status = spans.call('validation_run', 'validation_and_orchestration', v6.run,
                    argparse.Namespace(action='full', output=args.output, evidence=args.evidence,
                                       clip=args.clip, injected=False, reference=False))
            finally:
                if profiler is not None:
                    profiler.disable()
    except BaseException as exc:
        error = repr(exc)
        raise
    finally:
        if args.output.is_dir() and spans.nodes:
            timing = spans.summary()
            write_json(args.output/'stage_profile.json', dict(
                schema_version='seaqr.raw16-frozen-v6-profile.v1', clip=args.clip,
                cprofile_enabled=profiler is not None, runtime_archive_sha256=RUNTIME_SHA,
                frozen_files=frozen, script_sha256=sha(__file__), error=error, timing=timing,
                warning='Host wall spans, not device utilization. Decoder runs ahead in a child process; '
                        'decode delivery is blocked wait/copy/validation, not isolated decode compute. '
                        'Inclusive spans overlap. Only exclusive groups add to wall time. '
                        'All times include instrumentation; no sustained throughput claim.'))
            if profiler is not None:
                rows = [dict(file=key[0], line=key[1], function=key[2], calls=value[1],
                             self_s=value[2], cumulative_s=value[3])
                        for key, value in pstats.Stats(profiler).stats.items()]
                write_json(args.output/'python_profile.json', sorted(rows, key=lambda row:-row['self_s']))
    parity = compare(baseline, args.output)
    parity.update(compare_source_motion(baseline, args.output))
    checks = json.loads((args.output/'checks.json').read_text())['checks']
    passed = (status == 0 and parity['exact_semantics'] and parity['source_frames_exact']
              and parity['motion_points_exact'] and checks['processing_integrity_passed']
              and checks['detection_availability_passed'])
    write_json(args.output/'profile_parity.json', dict(passed=passed, comparison=parity,
        baseline_hashes=BASELINE_HASHES[args.clip], real_airborne_accuracy_validated=False))
    print(json.dumps(dict(passed=passed, clip=args.clip, timing=timing['groups']), indent=2), flush=True)
    return 0 if passed else 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clip', choices=('0029', '0040'), required=True)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--runtime-archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cprofile', action='store_true')
    raise SystemExit(run(parser.parse_args()))
