"""Read-only, bounded point-quality diagnosis on the two development prefixes.

No image directories or split manifests are opened. The failed v3 frontend and
historical frontend are compared without any threshold/configuration changes.
The selected pairs include v3's known0040 failures; they are diagnostic data,
not a fresh validation cohort.
"""
from __future__ import annotations

import argparse
from contextlib import closing
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'scripts'))
from profile_raw16_efficiency import source_paths, sha, write_json
from tiny_target.config import load_config
from tiny_target.frame_source import FfmpegVideoSource
from tiny_target.motion import PvaMotionConfig, PvaPyrLkMotionEstimator, GlobalMotionConfig, fit_global_motion
from tiny_target.motion.pva_pyrlk import _raw_affine_parameters

CONFIGS = {
    'historical': ('configs/evaluation/phase20_motion_v8.json', 'fe450546af91f01a0fb090d76df3ba4db24081b0077c6a220d5194990fdda5b1'),
    'raw_v3': ('configs/evaluation/raw16_motion_v3.json', '5c35fd8239e105d0a4e3b76133adad1cfd7f11d7610fb89b9843387052f7a74b'),
}
SELECTED_PAIRS = (1, 2, 20, 24, 43, 50, 56, 63)


def run(clip, output, include_v4=False, repeats=1):
    if output.exists():
        raise FileExistsError(output)
    video, timestamps = source_paths(clip)
    if video.is_symlink() or timestamps.is_symlink():
        raise ValueError('Development source cannot be redirected')
    configs = {}
    for mode, (name, expected) in CONFIGS.items():
        if sha(ROOT/name) != expected:
            raise ValueError('Frozen diagnostic configuration changed')
        configs[mode] = load_config(ROOT/name).raw
    if include_v4:
        candidate = load_config(ROOT/'configs/evaluation/raw16_motion_v4.json').raw
        from copy import deepcopy
        checked = deepcopy(candidate)
        if checked['motion'].pop('harris_capacity_policy') != 'complete_grid' or checked != configs['raw_v3']:
            raise ValueError('Only the capacity correction is allowed in this diagnostic')
        configs['raw_v4'] = candidate
    cv2.setNumThreads(2)
    estimators = {mode: PvaPyrLkMotionEstimator(PvaMotionConfig.from_mapping(cfg['motion']))
                  for mode, cfg in configs.items()}
    fitter = GlobalMotionConfig.from_mapping(configs['historical']['global_motion'])
    source = FfmpegVideoSource(video, timestamp_csv=timestamps,
        timestamp_policy='require_sidecar', bit_depth=16, max_frames=64)
    rows = []
    previous = None
    with closing(iter(source)) as frames:
        for current in frames:
            if current.frame_index in SELECTED_PAIRS:
                hashes = [f.pixel_sha256() for f in (previous, current)]
                modes = {}
                for mode, estimator in estimators.items():
                    attempts = [estimator.estimate(previous, current) for _ in range(repeats)]
                    pairs = attempts[0]
                    repeatable = all(np.array_equal(pairs.previous_points, p.previous_points)
                        and np.array_equal(pairs.current_points, p.current_points)
                        and np.array_equal(pairs.harris_scores, p.harris_scores) for p in attempts[1:])
                    fit = fit_global_motion(pairs, fitter)
                    modes[mode] = dict(correspondences=pairs.to_dict(), fit=fit.to_dict(),
                        residuals_px=[float(v) if np.isfinite(v) else None for v in fit.residuals_px],
                        repeats=repeats, repeated_points_exact=repeatable)
                    print(clip, current.frame_index, mode, pairs.count,
                          fit.quality_status, fit.metrics.get('p90_reprojection_error_px'), flush=True)
                if hashes != [f.pixel_sha256() for f in (previous, current)]:
                    raise ValueError('Source pixels were mutated')
                rows.append(dict(current_frame=current.frame_index, pixel_sha256=hashes,
                    affine_parameters=[_raw_affine_parameters(f) for f in (previous, current)], modes=modes))
            previous = current
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, dict(schema_version='seaqr.raw16-point-quality-diagnostic.v4',
        clip=clip, frames_read=64, selected_pairs=SELECTED_PAIRS, pairs=rows,
        configurations=configs, configuration_sha256={name: expected for name, expected in CONFIGS.values()},
        runtime_sha256={str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'tiny_target').rglob('*.py'))},
        script_sha256=sha(__file__), source=dict(path=str(video), size_bytes=video.stat().st_size,
            mtime_ns=video.stat().st_mtime_ns, timestamp_sha256=sha(timestamps)),
        warning='Diagnostic pairs selected from known development failures. No ground-truth camera-motion or airborne-object labels.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clip', choices=('0029', '0040'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--include-v4', action='store_true')
    parser.add_argument('--repeats', type=int, choices=(1, 3), default=1)
    args = parser.parse_args()
    run(args.clip, args.output, args.include_v4, args.repeats)
