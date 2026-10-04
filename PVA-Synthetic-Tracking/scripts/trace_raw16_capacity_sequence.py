"""Trace point counts through a 64-frame development motion sequence.

Read only the allowlisted source; no detector tuning, no new clip selection.
Trace source integrity and feature support with or without stabilization.
"""
from __future__ import annotations

import argparse
from contextlib import closing
import hashlib
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
from profile_raw16_efficiency import source_paths, sha, write_json
from tiny_target.frame_source import FfmpegVideoSource
from tiny_target.config import load_config
from tiny_target.motion import PvaPyrLkMotionEstimator
from tiny_target.motion.pva_pyrlk import _raw_affine_parameters
from tiny_target import dense_screen as dense
from validate_raw16_motion_v4 import CANDIDATE, verify_candidate_config, verify_generated_controls


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    verify_candidate_config()
    verify_generated_controls(args.generated_controls)
    video, timestamps = source_paths(args.clip)
    if video.is_symlink() or timestamps.is_symlink():
        raise ValueError('No redirected source')
    rows = []
    original = PvaPyrLkMotionEstimator.estimate

    def recorded(estimator, previous, current):
        before = [f.pixel_sha256() for f in (previous, current)]
        c = original(estimator, previous, current)
        after = [f.pixel_sha256() for f in (previous, current)]
        if after != before:
            raise ValueError('Original source pixels changed')
        rows.append(dict(current_frame=current.frame_index, source_hashes=before,
            bit_depth=[previous.bit_depth, current.bit_depth],
            mask_present=[previous.valid_mask is not None, current.valid_mask is not None],
            affine_parameters=[_raw_affine_parameters(f) for f in (previous, current)],
            previous_points_sha256=hashlib.sha256(c.previous_points.tobytes()).hexdigest(),
            current_points_sha256=hashlib.sha256(c.current_points.tobytes()).hexdigest(),
            metrics=c.metrics))
        if current.frame_index % 8 == 0:
            print(args.mode, current.frame_index, c.count, flush=True)
        return c

    with patch.object(PvaPyrLkMotionEstimator, 'estimate', recorded):
        if args.mode == 'raw':
            source = FfmpegVideoSource(video, timestamp_csv=timestamps,
                timestamp_policy='require_sidecar', bit_depth=16, max_frames=64)
            estimator = PvaPyrLkMotionEstimator(load_config(CANDIDATE).raw['motion'])
            previous = None
            with closing(iter(source)) as frames:
                for current in frames:
                    if previous is not None:
                        estimator.estimate(previous, current)
                    previous = current
        else:
            cfg, _ = dense.load_dense_screen_config(ROOT/'configs/evaluation/raw16_full_frame_v2.json')
            source = dense.StabilizedCropSource(video, cfg, CANDIDATE,
                timestamp_csv=timestamps, bit_depth=16, max_frames=64, injector=None)
            with closing(iter(source)) as frames:
                for _ in frames:
                    pass
    write_json(args.output, dict(clip=args.clip, mode=args.mode, pairs=rows,
        script_sha256=sha(__file__), config_sha256=sha(CANDIDATE),
        warning='Diagnostic development sequence, not camera-motion or object ground truth.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clip', choices=('0029', '0040'), required=True)
    parser.add_argument('--mode', choices=('raw', 'stabilized'), required=True)
    parser.add_argument('--generated-controls', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
