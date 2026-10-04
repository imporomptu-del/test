"""Exact resident-detector noise statistics, batched by valid sample count.

Samples retain their original tile membership. Only private scratch arrays are
partitioned in place; no padding, subsampling, quantile approximation or change
to float32 median arithmetic is introduced.
"""
import numpy as np


def tile_noise_statistics(samples, support, layout, stride, noise_floor):
    """Return float32 device statistics and the original float64 sigma median."""
    stats = np.empty((len(layout), 2), np.float32)
    noise_values = np.empty(len(layout), np.float64)
    groups = {}
    for j, (ys, xs, offset, count) in enumerate(layout):
        mask = support[ys, xs][::stride, ::stride].ravel()
        sample = samples[offset:offset + count][mask]
        if not sample.size:
            stats[j] = 0.0, max(noise_floor, 0.0)
            noise_values[j] = max(noise_floor, 0.0)
        else:
            groups.setdefault(sample.size, []).append((j, sample))

    for group in groups.values():
        indices = [j for j, _ in group]
        values = np.stack([sample for _, sample in group])
        # np.stack owns this scratch space. A median depends on the multiset,
        # not sample order; repartitioning its absolute deviations is likewise
        # exact and leaves the device download/source mask untouched.
        centers = np.median(values, axis=1, overwrite_input=True)
        np.subtract(values, centers[:, None], out=values)
        np.abs(values, out=values)
        # The former loop converted each MAD to a Python float before applying
        # the scale/floor. Preserve that float64 arithmetic for telemetry even
        # though the device consumes rounded float32 center/sigma pairs.
        # fmax also preserves max(finite_floor, nan)'s floor-first behavior.
        sigmas = np.fmax(noise_floor,
            1.4826 * np.median(values, axis=1, overwrite_input=True).astype(np.float64))
        stats[indices, 0] = centers
        stats[indices, 1] = sigmas
        noise_values[indices] = sigmas
    return stats, float(np.median(noise_values))
