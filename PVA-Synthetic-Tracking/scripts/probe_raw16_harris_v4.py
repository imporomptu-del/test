"""Bounded implementation diagnosis, not an accuracy/configuration search.

Compare VPI Harris input representations and capacity with an independent
floating-point gradient calculation. Only generated inputs or three frames
of one explicitly allowlisted development clip can be opened.
"""
from __future__ import annotations

import argparse
from contextlib import closing
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from check_raw16_motion_controls import frame, texture, translated
from profile_raw16_efficiency import source_paths, sha, write_json
from tiny_target.frame_source import FfmpegVideoSource
from tiny_target.motion.pva_pyrlk import _feature_pixels


def describe(points, scores, motion, signed):
    xy = np.rint(points).astype(int)
    local = cv2.cornerHarris(motion.astype(np.float32) / 65535., 3, 3, .0625)
    rows = []
    for low, high in ((0, 1000), (1000, 1e5), (1e5, 1e7), (1e7, 5e8), (5e8, 2**32)):
        keep = (scores >= low) & (scores < high)
        if keep.any():
            samples = xy[keep]
            rows.append(dict(score_range=[low, high], count=int(keep.sum()),
                float_harris_quantiles=np.quantile(local[samples[:, 1], samples[:, 0]], [0, .1, .5, .9, 1]).tolist(),
                input_codes_quantiles=np.quantile(motion[samples[:, 1], samples[:, 0]], [0, .1, .5, .9, 1]).tolist(),
                signed_codes_quantiles=np.quantile(signed[samples[:, 1], samples[:, 0]], [0, .1, .5, .9, 1]).tolist()))
    return dict(count=len(points), bands=rows,
        xy_range=None if not len(points) else [points.min(axis=0).tolist(), points.max(axis=0).tolist()],
        row_counts=np.histogram(points[:, 1], np.linspace(0, motion.shape[0], 7))[0].tolist())


def diagnose(f):
    import vpi
    stream = vpi.Stream(vpi.Backend.CUDA | vpi.Backend.PVA)
    pixels = _feature_pixels(f, 'raw_robust_u16_v1')
    image = vpi.asimage(pixels, vpi.Format.U16)
    size = (round(f.shape[1] / 2), round(f.shape[0] / 2))
    motion_vpi = image.rescale(size, backend=vpi.Backend.CUDA, stream=stream)
    stream.sync()
    with motion_vpi.rlock_cpu() as data:
        motion = np.array(data, copy=True)
    result = []
    for name, scale, offset in (('signed_full', 1., -32768.), ('positive_half', .5, 0.)):
        signed_vpi = motion_vpi.convert(vpi.Format.S16, scale=scale, offset=offset,
                                       backend=vpi.Backend.CUDA, stream=stream)
        stream.sync()
        with signed_vpi.rlock_cpu() as data:
            signed = np.array(data, copy=True)
        expected = np.clip(np.rint(motion.astype(np.float64) * scale + offset), -32768, 32767)
        for capacity in (8192, ((size[0] + 7) // 8 + 1) * ((size[1] + 7) // 8 + 1)):
            for backend in ('PVA', 'CUDA'):
                features = vpi.Array(capacity, vpi.Type.KEYPOINT_F32)
                scores = vpi.Array(capacity, vpi.Type.U32)
                features, scores = signed_vpi.harriscorners(backend=getattr(vpi.Backend, backend),
                    out_features=features, out_scores=scores, gradient_size=3, block_size=3,
                    strength=1., sensitivity=.0625, min_nms_distance=8., stream=stream)
                stream.sync()
                with features.rlock_cpu() as data:
                    p = np.array(data, dtype=np.float32, copy=True)
                with scores.rlock_cpu() as data:
                    s = np.array(data, dtype=np.float64, copy=True).reshape(-1)
                row = dict(mapping=name, backend=backend, capacity=capacity,
                    conversion_max_error=float(np.max(np.abs(expected - signed))),
                    **describe(p, s, motion, signed))
                result.append(row)
                print(name, backend, capacity, len(p), row['row_counts'], flush=True)
    return result


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    cv2.setNumThreads(2)
    if args.clip is None:
        f = frame(texture(153159), 0, 'harris-representation')
    else:
        video, sidecar = source_paths(args.clip)
        if video.is_symlink() or sidecar.is_symlink():
            raise ValueError('No redirected media')
        source = FfmpegVideoSource(video, timestamp_csv=sidecar, timestamp_policy='require_sidecar',
                                  bit_depth=16, max_frames=3)
        with closing(iter(source)) as frames:
            f = next(frames)
    before = f.pixel_sha256()
    result = diagnose(f)
    if before != f.pixel_sha256():
        raise ValueError('Source pixels mutated')
    write_json(args.output, dict(clip=args.clip, source_frame_index=f.frame_index,
        source_pixel_sha256=before, results=result, script_sha256=sha(__file__),
        warning='Diagnostic ablations only, no camera-motion or object ground truth.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clip', choices=('0029', '0040'))
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
