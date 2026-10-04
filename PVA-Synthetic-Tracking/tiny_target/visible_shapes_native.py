"""Explicit bounded native bookkeeping; exact NumPy centroid arithmetic.

The reference implementation remains available and is the conformance oracle.
No implicit compiler, library discovery, fallback, or algorithmic thresholds.
"""
import ctypes as C
import hashlib
import math
from pathlib import Path

import numpy as np


class NativeShapes:
    def __init__(self, library, expected_sha256):
        path = Path(library).resolve(strict=True)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
            raise ValueError("Native shape library hash changed; no fallback")
        self.lib = C.CDLL(str(path))
        self.lib.seaqr_shapes_abi.argtypes = []
        self.lib.seaqr_shapes_abi.restype = C.c_int
        if self.lib.seaqr_shapes_abi() != 1:
            raise RuntimeError("Unsupported native shapes ABI; no fallback")
        self.lib.seaqr_shapes_v1.argtypes = [C.c_int] * 3 + [C.c_void_p] * 11
        self.lib.seaqr_shapes_v1.restype = C.c_int

    def consolidate(self, proposals, shape, seeds, patches, eligible, *, include_support=False):
        n = len(proposals)
        if (len(shape) != 2 or min(shape) < 1 or max(shape) > np.iinfo(np.int32).max
                or eligible.shape != tuple(shape) or eligible.dtype != np.bool_
                or seeds.shape != (n, 2) or seeds.dtype != np.int32
                or patches.shape != (n, 17, 17) or patches.dtype != np.float32 or n > 512):
            raise ValueError("Native shapes require <=512 int32 seeds, float32 17x17 patches and a matching bool mask")
        h, w = shape
        # Integer coordinates are the resident peak ABI, not quantized here.
        polarity = np.empty(n, np.int32)
        for i, p in enumerate(proposals):
            if (p['polarity'] not in ('bright', 'dark') or p['x'] != seeds[i, 0]
                    or p['y'] != seeds[i, 1] or not math.isfinite(p['score'])):
                raise ValueError("Native shape proposals must match finite resident peaks")
            polarity[i] = p['polarity'] == 'dark'
        seeds, patches, eligible = [np.ascontiguousarray(a) for a in (seeds, patches, eligible)]
        go = np.empty(n+1, np.int32); members = np.empty(n, np.int32)
        states = np.empty(n, np.int32); ro = np.empty(n+1, np.int32)
        pixels = np.empty(n*289, np.int64); values = np.empty(n*289, np.float32)
        summary = np.empty(2, np.int32)
        buffers = (seeds, polarity, patches, eligible, go, members, states, ro, pixels, values, summary)
        # Eleven pointers; counts and dimensions are supplied separately.
        code = self.lib.seaqr_shapes_v1(h, w, n, *(a.ctypes.data for a in buffers))
        if code:
            raise RuntimeError(f"Native shapes failed ({code}); no fallback")
        groups, rejected = map(int, summary)
        if not 0 <= groups <= n or go[0] != 0 or go[groups] != n or ro[0] != 0 or not 0 <= ro[groups] <= len(pixels):
            raise RuntimeError("Invalid native shape output bounds")
        output, merged = [], 0
        for g in range(groups):
            group = members[go[g]:go[g+1]].tolist()
            strongest = min(group, key=lambda i: (-proposals[i]['score'], proposals[i]['y'], proposals[i]['x']))
            p = dict(proposals[strongest])
            state = states[g]
            if state == 0:
                output.append(p)
                continue
            if state == 1:
                output.extend(dict(proposals[i]) for i in sorted(group))
                rejected += len(group)
                continue
            if state != 2:
                raise RuntimeError("Invalid native shape state")
            region = pixels[ro[g]:ro[g+1]]
            yy, xx = np.divmod(region, w)
            sign = 1 if p['polarity'] == 'bright' else -1
            weight = sign * values[ro[g]:ro[g+1]].astype(np.float64)
            total_weight = weight.sum()
            if total_weight == 0:
                raise ZeroDivisionError("Weights sum to zero, can't be normalized")
            cx = float((xx * weight).sum() / total_weight)
            cy = float((yy * weight).sum() / total_weight)
            x, y = round(cx), round(cy)
            key = y*w+x
            position = np.searchsorted(region, key)
            if position == len(region) or region[position] != key or not eligible[y, x]:
                output.extend(dict(proposals[i]) for i in sorted(group))
                rejected += len(group)
                continue
            xmin, xmax, ymin, ymax = int(xx.min()), int(xx.max()), int(yy[0]), int(yy[-1])
            p['shape'] = dict(method='mutual_half_height_r8', centroid_reference_xy=[cx, cy],
                peak_reference_xy=[p['x'], p['y']],
                member_peak_reference_xy=[[proposals[i]['x'], proposals[i]['y']] for i in sorted(group)],
                support_pixels=len(region), bbox_reference_xywh=[xmin, ymin, xmax-xmin+1, ymax-ymin+1],
                score_location='original accepted peak; not the centroid', position_quantization='nearest native pixel')
            p['x'], p['y'] = x, y
            if include_support:
                p['shape']['support_reference_xy'] = np.column_stack((xx, yy)).tolist()
            output.append(p)
            merged += len(group)-1
        return output, dict(method='mutual_half_height_r8', input_peaks=n, output_features=len(output),
            merged_peak_count=merged, unmodified_unbounded_or_unsupported_peaks=rejected,
            radius_px=8, half_height_fraction=0.5, no_new_threshold_seeds=True)
