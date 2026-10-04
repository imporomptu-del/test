"""Small empirical calibration helpers for the diagnostic-only V50 experiment.

One unit is a clip/segment/response-frame, scored by the maximum error across
*all* its archived state packets. Pixels, tracks, and packets are not separate
calibration units. Callers group by clip and segment before selecting frames.

The primary policy selects earliest disjoint nine-frame history/response blocks
using archived frame metadata only. A missing score never creates a replacement
anchor. The all-frame policy is a sensitivity analysis, not an independent-unit
claim. Neither policy establishes exchangeability, confidence, calibrated
physical error bounds, or production readiness.

The returned empirical threshold is inclusive: a normalized error equal to q
is inside. With the V50 prior-only scale max(1 DN, MAD of eight priors), callers
use half-width q * scale without clipping intervals to image-value bounds.
"""

from numbers import Integral, Real
import math
from typing import Sequence


DISJOINT_ANCHOR_FRAMES = "disjoint_anchor_frames"
ALL_CALIBRATION_FRAMES = "all_calibration_frames"
POLICIES = (DISJOINT_ANCHOR_FRAMES, ALL_CALIBRATION_FRAMES)


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _scores(values):
    """Validate every entry, including entries after an unavailable score."""
    result = []
    for value in values:
        if value is None:
            result.append(None)
            continue
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError("Scores must be finite nonnegative real numbers or None")
        try:
            converted = float(value)
        except (OverflowError, ValueError) as error:
            raise ValueError("Scores must be finite nonnegative real numbers or None") from error
        if not math.isfinite(converted) or converted < 0:
            raise ValueError("Scores must be finite nonnegative real numbers or None")
        result.append(converted)
    return result


def _frames(frames):
    values = []
    for frame in frames:
        if isinstance(frame, bool) or not isinstance(frame, Integral) or frame < 0:
            raise ValueError("Frames must be nonnegative integers")
        values.append(int(frame))
    return sorted(set(values))


def greedyanchors(frames: Sequence[int], gap: int = 9) -> list[int]:
    """Earliest metadata-only anchors; sort/deduplicate before greedy selection.

    Frames belong to one clip/segment. No score argument is accepted, so an
    unavailable selected response cannot be replaced by another scored frame.
    """
    gap = _positive_integer(gap, "gap")
    selected = []
    for frame in _frames(frames):
        if not selected or frame >= selected[-1] + gap:
            selected.append(frame)
    return selected


def select_calibration_frames(
    frames: Sequence[int], policy: str, gap: int = 9
) -> list[int]:
    """Select metadata frames under one of the two explicitly named policies."""
    gap = _positive_integer(gap, "gap")
    if policy == DISJOINT_ANCHOR_FRAMES:
        return greedyanchors(frames, gap=gap)
    if policy == ALL_CALIBRATION_FRAMES:
        return _frames(frames)
    raise ValueError(f"policy must be one of {POLICIES!r}")


def frame_score(packet_scores: Sequence[float | None]) -> float | None:
    """Maximum over every archived packet in a frame; any missing means unknown.

    The caller must supply an entry for every archived state packet, using None
    for an unscorable or missing packet. It must not pass only successful scores.
    """
    values = _scores(packet_scores)
    if not values:
        raise ValueError("A frame must contain at least one archived packet score")
    if any(value is None for value in values):
        return None
    return max(values)


def empirical_quantile(
    scores: Sequence[float | None], targetnumerator: int = 9, targetdenominator: int = 10
) -> dict:
    """Return order statistic ceil((m + 1) * target) of m finite frame units.

    Missing selected units remain in total/missing counts, not in m, and do not
    receive substitute frames. When the requested rank exceeds m, q is unknown;
    the rank is never clamped. Availability is computational, not a statistical
    coverage guarantee. Repeated values and zero scores are retained as units.
    """
    numerator = _positive_integer(targetnumerator, "targetnumerator")
    denominator = _positive_integer(targetdenominator, "targetdenominator")
    if numerator > denominator:
        raise ValueError("targetnumerator cannot exceed targetdenominator")
    values = _scores(scores)
    finite = sorted(value for value in values if value is not None)
    count = len(finite)
    rank = ((count + 1) * numerator + denominator - 1) // denominator
    reasons = []
    if count == 0:
        reasons.append("no_finite_calibration_units")
    if rank > count:
        reasons.append("requested_rank_exceeds_finite_units")
    return {
        "available": not reasons,
        "reasons": reasons,
        "total_units": len(values),
        "finite_units": count,
        "missing_units": len(values) - count,
        "rank": rank,
        "q": finite[rank - 1] if not reasons else None,
        "target_numerator": numerator,
        "target_denominator": denominator,
    }
