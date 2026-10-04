"""Bounded development diagnostic: same raw pairs, shared feature-image contrast.

Only the feature image changes; original uint16 frames, saturation/validity
exclusions and motion-fit quality thresholds are unchanged. This is diagnostic
evidence, not production acceptance or real-target accuracy.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'scripts'))
from profile_raw16_efficiency import source_paths, sha, write_json, compact
from tiny_target.config import load_config
from tiny_target.frame_source import FfmpegVideoSource
from tiny_target.motion import PvaMotionConfig, PvaPyrLkMotionEstimator, GlobalMotionConfig, fit_global_motion
from tiny_target.motion import pva_pyrlk


def feature_pair(previous, current):
    samples = np.concatenate([f.image[::8, ::8].ravel() for f in (previous, current)])
    lo, hi = np.percentile(samples, [.5, 99.5])
    span = max(float(hi-lo), 1024.)
    arrays = []
    for frame in (previous, current):
        values = (frame.image.astype(np.float32)-float(lo)) * (255./span)
        arrays.append(np.clip(np.rint(values), 0, 255).astype(np.uint8))
    return arrays, dict(low_dn=float(lo), high_dn=float(hi), effective_span_dn=span,
                       percentile_bounds=[.5,99.5], sample_stride=8, minimum_span_dn=1024.)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--clip', choices=('0029','0040'), required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError('Never overwrite diagnostic evidence')
    video, timestamps = source_paths(a.clip)
    cfg = load_config(ROOT/'configs/evaluation/phase20_motion_v8.json').raw
    estimator = PvaPyrLkMotionEstimator(PvaMotionConfig.from_mapping(cfg['motion']))
    fitter = GlobalMotionConfig.from_mapping(cfg['global_motion'])
    source = FfmpegVideoSource(video, timestamp_csv=timestamps, timestamp_policy='require_sidecar',
                               bit_depth=16, max_frames=64)
    selected = {1, 16, 32, 48, 63}
    results = []
    previous = None
    for current in source:
        if current.frame_index in selected:
            original_hashes = [f.pixel_sha256() for f in (previous, current)]
            variants = {}
            prepared, mapping = feature_pair(previous, current)
            for mode in ('native_shift', 'pair_robust_contrast'):
                if mode == 'native_shift':
                    correspondences = estimator.estimate(previous, current)
                else:
                    iterator = iter(prepared)
                    with patch.object(pva_pyrlk, '_motion_u8', lambda frame: next(iterator)):
                        correspondences = estimator.estimate(previous, current)
                fit = fit_global_motion(correspondences, fitter)
                variants[mode] = dict(correspondences=compact(correspondences.to_dict()),
                    fit=compact(fit.to_dict()), exact_raw_hashes=original_hashes)
            if original_hashes != [f.pixel_sha256() for f in (previous, current)]:
                raise ValueError('Diagnostic mutated original RAW16 data')
            sample = previous.image[::8,::8]
            results.append(dict(current_frame=current.frame_index, mapping=mapping,
                raw_percentiles=np.percentile(sample,[0,.5,50,99.5,100]).tolist(),
                shifted_distinct_sample_values=int(np.unique(sample>>8).size), variants=variants))
            print(json.dumps(dict(clip=a.clip, frame=current.frame_index,
                results={k:dict(features=v['correspondences']['accepted_count'],
                                accepted=v['fit']['quality_status'],
                                reasons=v['fit']['rejection_reasons']) for k,v in variants.items()})), flush=True)
        previous = current
    write_json(a.output, dict(clip=a.clip, frames_read=64, diagnostic_pairs=results,
        source=dict(path=str(video), size_bytes=video.stat().st_size,
                    mtime_ns=video.stat().st_mtime_ns, timestamp_sha256=sha(timestamps)),
        motion_config=cfg, script_sha256=sha(__file__),
        warning='Development diagnostic, no object labels, no acceptance threshold changes; not accuracy validation'))


if __name__ == '__main__':
    main()
