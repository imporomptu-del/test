"""Bounded full-image RAW16 development checks, with explicitly synthetic truth.

Only the previously authorized 0029/0040 sources are accepted. Never scans a
media directory or opens a split. Settings and control locations are frozen
before running. This is correctness evidence, not a throughput benchmark.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import platform
import resource
import sys
import time
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from profile_raw16_efficiency import source_paths, sha, write_json
from tiny_target import dense_screen as dense
from tiny_target.frame_source import FfmpegVideoSource

CONFIG = ROOT / 'configs/evaluation/raw16_full_frame_v2.json'
MOTION = ROOT / 'configs/evaluation/phase20_motion_v8.json'
CONTROLS = ROOT / 'configs/evaluation/raw16_full_frame_controls_v2.json'
LIBRARY = ROOT / 'build/cuda/libtiny_target_cuda.so'
FROZEN_HASHES = {
    CONFIG: 'ff0af1af9da5320f46566bb65fbe85fcc6c5112d250a4b08f91526dfc80517d9',
    CONTROLS: '8095010645c253e928169a3aeb80426ff70fe6b23563f6f21ae4044102acd5cb',
    MOTION: 'fe450546af91f01a0fb090d76df3ba4db24081b0077c6a220d5194990fdda5b1',
    LIBRARY: 'e29dc8bae949e41497aff82cfa2fe07d52e1c039b5c337187bdb8c2e87b1fc65',
}


def assess(report):
    """A completed/no-error run is not sufficient evidence of availability."""
    source = report['source']
    screen = report['screening']
    availability = screen['availability']
    windows = screen['synthetic_tracking']['windows']
    metrics = source['pva_stabilization']['metrics']
    integrity = {
        'all_64_frames_accounted': screen['frames_seen'] == len(availability['frames']) == 64,
        'native_full_frame_geometry': source['crop_xywh'] == [0, 0, 4784, 3190],
        'full_requested_area': source['requested_image_area_fraction'] == 1.,
        'no_pva_runtime_failures': metrics['pva_failures'] == 0,
    }
    supported = [w for w in windows if all(n > 0 for n in w['availability']['valid_ranking_pixels_3x3_row_major'])]
    coverage = {
        'at_least_48_frames_with_filter_support': availability['frames_with_valid_filter_support'] >= 48,
        'at_least_3_windows_with_support_in_all_nine_regions': len(supported) >= 3,
    }
    controls = None
    if report['injection'] is not None:
        expected = sorted(t['target_id'] for t in report['injection']['specification']['targets'])
        observed = report['injection']['synthetic_track_pool_evaluation']['detected_target_ids']
        controls = {'all_injected_targets_reach_track_pool': bool(expected) and sorted(observed) == expected}
    return dict(processing_integrity=integrity, processing_integrity_passed=all(integrity.values()),
        detection_availability=coverage, detection_availability_passed=all(coverage.values()),
        synthetic_controls=controls, synthetic_controls_passed=None if controls is None else all(controls.values()),
        real_target_accuracy_validated=False)


def run(args):
    video, sidecar = source_paths(args.clip)
    if args.injected and args.clip != '0040':
        raise ValueError('Frozen positive-control experiment is limited to development clip0040')
    for path, expected in FROZEN_HASHES.items():
        if sha(path) != expected:
            raise ValueError(f'Frozen experiment file changed: {path}')
    if video.is_symlink() or sidecar.is_symlink():
        raise ValueError('Development sources may not be redirected by symlinks')
    args.output.mkdir(parents=True, exist_ok=False)
    observation = dict(clip=args.clip, injected=args.injected, frames_requested=64,
        source=dict(path=str(video), size_bytes=video.stat().st_size,
                    mtime_ns=video.stat().st_mtime_ns, timestamp_sha256=sha(sidecar)),
        frozen_sha256={str(p.relative_to(ROOT)): sha(p) for p in FROZEN_HASHES},
        package_sha256={str(p.relative_to(ROOT)): sha(p)
                        for p in sorted((ROOT / 'tiny_target').rglob('*.py')) if not p.name.startswith('._')},
        script_sha256=sha(__file__), helper_sha256=sha(ROOT / 'scripts/profile_raw16_efficiency.py'),
        python=platform.python_version(), numpy=np.__version__,
        warning='Unlabeled development sources; injected controls are downstream of stabilization. '
                'No real-object recall/FAR, airborne identity, exposure timing or speed-generalization claim.')
    write_json(args.output / 'provenance.json', observation)
    decoded = []
    original_iter = FfmpegVideoSource.__iter__
    original_process = dense.DensePointScreener.process

    def checked_source(source):
        iterator = original_iter(source)
        try:
            for frame in iterator:
                if frame.image.dtype != np.dtype('<u2') or frame.bit_depth != 16 or frame.shape != (3190, 4784):
                    raise ValueError('Native uint16 source contract violated')
                if frame.source_timestamp_ns is None:
                    raise ValueError('Recorded timestamps are required')
                decoded.append(dict(frame_index=frame.frame_index, pixel_sha256=frame.pixel_sha256(),
                    timestamp_ns=frame.timestamp_ns, source_timestamp_ns=frame.source_timestamp_ns))
                yield frame
        finally:
            iterator.close()

    def progress(screener, frame, **kwargs):
        result = original_process(screener, frame, **kwargs)
        if (frame.frame_index + 1) % 8 == 0:
            print(json.dumps(dict(clip=args.clip, injected=args.injected, frames=frame.frame_index + 1,
                windows=len(screener._synthetic_window_summaries),
                filter_state=screener._availability[-1]['filter_state'])), flush=True)
        return result

    start = time.perf_counter()
    with ExitStack() as stack:
        stack.enter_context(patch.object(FfmpegVideoSource, '__iter__', checked_source))
        stack.enter_context(patch.object(dense.DensePointScreener, 'process', progress))
        report = dense.screen_video(CONFIG, video, timestamp_csv=sidecar, motion_config_path=MOTION,
            max_frames=64, bit_depth=16, injection_spec_path=CONTROLS if args.injected else None)
    checks = assess(report)
    write_json(args.output / 'report.json', report)
    write_json(args.output / 'source_frames.json', decoded)
    write_json(args.output / 'checks.json', dict(checks=checks, elapsed_wall_s=time.perf_counter() - start,
        peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        instrumentation='source pixel hashing and progress logging; not a clean throughput benchmark'))
    print(json.dumps(checks), flush=True)
    # Unavailable motion on 0029 is retained as a diagnostic result, not hidden
    # by failing to write the report. Accuracy/availability booleans stay false.
    return 0 if checks['processing_integrity_passed'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clip', choices=('0029', '0040'), required=True)
    parser.add_argument('--injected', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    raise SystemExit(run(parser.parse_args()))
