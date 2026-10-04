"""Compare half-size features against the largest all-PVA image on development pairs.

No intensity conversion, motion acceptance thresholds or RAW pixels are changed.
The larger scale is derived from image geometry and the existing PVA pyramid cap,
not selected by searching for settings that accept a particular clip.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from profile_raw16_efficiency import source_paths, sha, write_json, compact
from tiny_target.config import load_config
from tiny_target.frame_source import FfmpegVideoSource
from tiny_target.motion import PvaMotionConfig, PvaPyrLkMotionEstimator, GlobalMotionConfig, fit_global_motion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clip', choices=('0029', '0040'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Never overwrite diagnostic evidence')
    video, timestamps = source_paths(args.clip)
    cfg = load_config(ROOT / 'configs/evaluation/phase20_motion_v8.json').raw
    base = PvaMotionConfig.from_mapping(cfg['motion'])
    source = FfmpegVideoSource(video, timestamp_csv=timestamps,
                               timestamp_policy='require_sidecar', bit_depth=16, max_frames=64)
    cap_w, cap_h = PvaPyrLkMotionEstimator.PVA_PYRAMID_MAX_SIZE
    scale = min(1.0, cap_w / source.probe.width, cap_h / source.probe.height)
    estimators = {'half_size': PvaPyrLkMotionEstimator(base),
                  'geometry_capped': PvaPyrLkMotionEstimator(replace(base, feature_image_scale=scale))}
    fitter = GlobalMotionConfig.from_mapping(cfg['global_motion'])
    selected = {1, 16, 32, 48, 63}
    results = []
    previous = None
    for current in source:
        if current.frame_index in selected:
            hashes = [frame.pixel_sha256() for frame in (previous, current)]
            variants = {}
            for name, estimator in estimators.items():
                pairs = estimator.estimate(previous, current)
                fit = fit_global_motion(pairs, fitter)
                variants[name] = dict(correspondences=compact(pairs.to_dict()), fit=compact(fit.to_dict()))
            assert hashes == [frame.pixel_sha256() for frame in (previous, current)]
            results.append(dict(current_frame=current.frame_index, raw_hashes=hashes, variants=variants))
            print(json.dumps(dict(clip=args.clip, frame=current.frame_index,
                results={k: dict(features=v['correspondences']['accepted_count'],
                    status=v['fit']['quality_status'], reasons=v['fit']['rejection_reasons'])
                    for k, v in variants.items()})), flush=True)
        previous = current
    write_json(args.output, dict(clip=args.clip, frames_read=64, diagnostic_pairs=results,
        geometry_scale=scale, pva_size_cap=[cap_w, cap_h],
        source=dict(path=str(video), size_bytes=video.stat().st_size,
                    mtime_ns=video.stat().st_mtime_ns, timestamp_sha256=sha(timestamps)),
        motion_config=cfg, script_sha256=sha(__file__),
        warning='Development motion diagnostic only; no RAW real-target labels or accuracy claim'))


if __name__ == '__main__':
    main()
