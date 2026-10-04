"""Sequence validation shared by recorded and live frame sources."""

from __future__ import annotations

from dataclasses import dataclass, field

from .types import Discontinuity, Frame


@dataclass(slots=True)
class FrameSequenceValidator:
    """Annotate timestamp and sequence discontinuities without hiding frames."""

    expected_interval_ns: int | None = None
    gap_factor: float = 1.5
    _previous: Frame | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.expected_interval_ns is not None and self.expected_interval_ns <= 0:
            raise ValueError("expected_interval_ns must be positive")
        if self.gap_factor <= 1.0:
            raise ValueError("gap_factor must be greater than 1")

    def observe(self, frame: Frame) -> Frame:
        previous = self._previous
        events: list[Discontinuity] = []
        if previous is not None:
            if frame.frame_index != previous.frame_index + 1:
                events.append(Discontinuity.FRAME_INDEX_GAP)

            if frame.sequence is not None and previous.sequence is not None:
                if frame.sequence != previous.sequence + 1:
                    events.append(Discontinuity.SEQUENCE_GAP)

            delta_ns = frame.timestamp_ns - previous.timestamp_ns
            if delta_ns == 0:
                events.append(Discontinuity.DUPLICATE_TIMESTAMP)
            elif delta_ns < 0:
                events.append(Discontinuity.NON_MONOTONIC_TIMESTAMP)
            elif (
                self.expected_interval_ns is not None
                and delta_ns > self.expected_interval_ns * self.gap_factor
            ):
                events.append(Discontinuity.TIMESTAMP_GAP)

        result = frame.with_discontinuities(*events) if events else frame
        self._previous = result
        return result

    def reset(self) -> None:
        self._previous = None

