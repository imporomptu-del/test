"""Annotation-free, causal visible-track quality and candidate deduplication."""
from collections import deque
import math

import numpy as np


def suppress_nearby(proposals, radius):
    """Keep strongest same-polarity peak inside a declared resolution radius.

    Preserve original ordering and all rejection provenance. This cannot resolve
    two real points closer than the configured radius and is not identity proof.
    """
    if radius <= 0:
        return proposals, []
    kept = []
    suppressed = []
    cells = {}
    for idx in sorted(range(len(proposals)), key=lambda i: (-proposals[i]["score"], i)):
        p = proposals[idx]
        gx, gy = math.floor(p["x"] / radius), math.floor(p["y"] / radius)
        nearby = [
            j
            for y in range(gy - 1, gy + 2)
            for x in range(gx - 1, gx + 2)
            for j in cells.get((p["polarity"], x, y), [])
            if math.hypot(p["x"] - proposals[j]["x"], p["y"] - proposals[j]["y"])
            <= radius
        ]
        if nearby:
            winner = min(nearby, key=lambda j: (-proposals[j]["score"], j))
            suppressed.append(
                dict(
                    candidate_index=idx,
                    retained_candidate_index=winner,
                    reason="same_polarity_resolution_nms",
                )
            )
        else:
            kept.append(idx)
            cells.setdefault((p["polarity"], gx, gy), []).append(idx)
    return [proposals[i] for i in sorted(kept)], suppressed


class CausalMotionQuality:
    """Fit short measured-position history, allowing acceleration/turns.

    This is a consistency gate, not an existence probability. No filtered states,
    predictions, future observations, or truth coordinates enter the fit.
    """

    def __init__(self, window_hits=8, minimum_hits=5, maximum_rmse_px=3.0):
        if (
            not 5 <= minimum_hits <= window_hits
            or not math.isfinite(maximum_rmse_px)
            or maximum_rmse_px <= 0
        ):
            raise ValueError("Invalid causal quality controls")
        self.history = deque(maxlen=window_hits)
        self.minimum_hits = minimum_hits
        self.maximum_rmse_px = maximum_rmse_px
        self.latest = dict(
            ready=False,
            passed=False,
            measured_history_count=0,
            quadratic_fit_rmse_px=None,
        )

    def observe(self, timestamp_ns, x, y):
        if not np.isfinite((x, y)).all() or (
            self.history and timestamp_ns <= self.history[-1][0]
        ):
            raise ValueError(
                "Finite positions and strictly increasing timestamps required"
            )
        self.history.append((timestamp_ns, x, y))
        ready = len(self.history) >= self.minimum_hits
        rmse = None
        if ready:
            data = np.asarray(self.history, dtype=np.float64)
            times = (data[:, 0] - data[-1, 0]) / 1e9
            times /= max(abs(times[0]), 1e-9)
            design = np.column_stack((np.ones_like(times), times, times * times))
            fitted = design @ np.linalg.lstsq(design, data[:, 1:], rcond=None)[0]
            rmse = float(np.sqrt(np.mean(np.sum((fitted - data[:, 1:]) ** 2, axis=1))))
        self.latest = dict(
            ready=ready,
            passed=ready and rmse <= self.maximum_rmse_px,
            measured_history_count=len(self.history),
            quadratic_fit_rmse_px=rmse,
            maximum_rmse_px=self.maximum_rmse_px,
            input="recent raw measured reference positions; no predictions",
        )
        return self.latest
