"""Execution success is separate from detector availability and target accuracy."""
from collections import Counter
import math


def finite_json(value):
    if isinstance(value, dict):
        return {k: finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class DetectionAvailability:
    def __init__(self):
        self.counts = Counter()
        self.reasons = Counter()
        self.paths = Counter()
        self.unavailable_streak = 0
        self.longest_unavailable_streak = 0

    def update(self, coverage, motion):
        ready = not coverage["warmup"] and coverage["searchable_pixels"] > 0
        coverage["detection_ready"] = ready
        coverage["unavailable_reason"] = (
            "warmup"
            if coverage["warmup"]
            else "no_valid_search_support"
            if not ready
            else None
        )
        self.counts["frames"] += 1
        self.counts["detection_ready_frames"] += ready
        self.counts["warmup_frames"] += coverage["warmup"]
        self.counts["no_spatial_support_frames"] += coverage["searchable_pixels"] == 0
        self.counts["motion_resets"] += motion["reset"]
        self.counts["pva_runtime_errors"] += motion.get("pva_failure", False)
        if "accepted" in motion:
            self.counts["motion_fits_attempted"] += 1
            self.counts["motion_fits_accepted"] += motion["accepted"]
        self.reasons.update(motion.get("rejection_reasons", []))
        path = (
            motion.get("motion_fit", {})
            .get("metrics", {})
            .get("coverage_acceptance_path")
        )
        if path:
            self.paths[path] += 1
        self.unavailable_streak = 0 if ready else self.unavailable_streak + 1
        self.longest_unavailable_streak = max(
            self.longest_unavailable_streak, self.unavailable_streak
        )

    def report(self):
        ready = self.counts["detection_ready_frames"]
        return dict(
            detection_status="available_unlabeled" if ready else "unavailable",
            usable_detection_coverage=ready > 0,
            detection_ready_fraction=ready / max(1, self.counts["frames"]),
            counts=dict(self.counts),
            longest_unavailable_streak_frames=self.longest_unavailable_streak,
            motion_rejection_reason_counts=dict(self.reasons),
            motion_coverage_path_counts=dict(self.paths),
            interpretation="Availability is not recall or full-pixel coverage. Zero proposals do not establish an empty scene.",
        )
