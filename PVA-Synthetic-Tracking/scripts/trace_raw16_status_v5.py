"""Sequential versus reverse-order point identity on 64 approved RAW frames."""
from __future__ import annotations

import argparse
from contextlib import closing
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
from profile_raw16_efficiency import source_paths, sha, write_json
from tiny_target.config import load_config
from tiny_target.frame_source import FfmpegVideoSource
from tiny_target.motion import PvaPyrLkMotionEstimator, GlobalMotionConfig, fit_global_motion
from validate_raw16_motion_v5 import CANDIDATE, verify_candidate_config, verify_generated_controls

SELECTED = (1, 2, 20, 24, 43, 50, 56, 63)


def identity(c):
    return {name: hashlib.sha256(getattr(c, name).tobytes()).hexdigest() for name in
            ('previous_points', 'current_points', 'harris_scores', 'forward_backward_error_px')}


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    verify_candidate_config()
    verify_generated_controls(args.generated_controls)
    video, stamps = source_paths(args.clip)
    if video.is_symlink() or stamps.is_symlink():
        raise ValueError('No redirected media')
    cfg = load_config(CANDIDATE).raw
    estimator = PvaPyrLkMotionEstimator(cfg['motion'])
    fitter = GlobalMotionConfig.from_mapping(cfg['global_motion'])
    source = FfmpegVideoSource(video, timestamp_csv=stamps,
        timestamp_policy='require_sidecar', bit_depth=16, max_frames=64)
    rows, saved, frames_seen = [], {}, []
    previous = None
    with closing(iter(source)) as frames:
        for current in frames:
            frames_seen.append(dict(frame_index=current.frame_index, pixel_sha256=current.pixel_sha256()))
            if previous is not None:
                c = estimator.estimate(previous, current)
                fit = fit_global_motion(c, fitter)
                rows.append(dict(current_frame=current.frame_index, identity=identity(c),
                    metrics=c.metrics, fit=fit.to_dict(include_inlier_indices=False)))
                if current.frame_index in SELECTED:
                    saved[current.frame_index] = (previous, current)
                if current.frame_index % 8 == 0:
                    print('sequential', current.frame_index, c.count, fit.quality_status, flush=True)
            previous = current
    repeats = []
    for index in reversed(SELECTED):
        before = [f.pixel_sha256() for f in saved[index]]
        c = estimator.estimate(*saved[index])
        same = rows[index-1]['identity'] == identity(c)
        if before != [f.pixel_sha256() for f in saved[index]]:
            raise ValueError('Source mutated')
        repeats.append(dict(current_frame=index, identical_points=same, identity=identity(c)))
        print('reverse replay', index, c.count, 'exact', same, flush=True)
    passed = len(rows) == 63 and all(r['fit']['quality_status'] == 'accepted' for r in rows) and all(r['identical_points'] for r in repeats)
    write_json(args.output, dict(schema_version='seaqr.raw16-status-sequence.v5', clip=args.clip,
        passed=passed, frames=frames_seen, pairs=rows, reverse_replays=repeats,
        script_sha256=sha(__file__), config_sha256=sha(CANDIDATE),
        runtime_sha256={str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'tiny_target/motion').glob('*.py'))},
        warning='Order-invariance and development availability test only, not physical motion or airborne ground truth.'))
    return 0 if passed else 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clip', choices=('0029', '0040'), required=True)
    parser.add_argument('--generated-controls', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    raise SystemExit(run(parser.parse_args()))
