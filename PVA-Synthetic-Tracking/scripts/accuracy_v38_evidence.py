"""Global-source multi-lag evidence building block; never an output gate.

Nine fixed translation probes measure sensitivity, not independent votes or
camera confidence. Flat local texture and missing prior association do not
prevent extraction. Cancellation/unsupported pixels remain explicit unknowns.
"""
import math

import numpy as np

from accuracy_v36_context import PointEdgeDiagnostic
from accuracy_v37_temporal import TemporalPointDiagnostic


SHIFTS = tuple((dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1))
METRICS = ("point_gain_fraction", "edge_gain_fraction", "point_minus_edge_fraction",
           "point_amplitude_dn", "residual_rms_dn", "signed_point_amplitude_dn")


def pixels(value, shape):
    if np.iscomplexobj(value):
        raise ValueError("Real source pixels required")
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Numeric source pixels required") from exc
    if result.shape != shape or np.isinf(result).any():
        raise ValueError("Fixed source shape required; NaN borders allowed, infinity forbidden")
    finite = result[np.isfinite(result)]
    # Convex bilinear interpolation can exceed an endpoint by float round-off.
    # This is a numeric bound tolerance, not clipping or a scene threshold.
    if np.any((finite < -1e-9) | (finite > 255+1e-9)):
        raise ValueError("Uncorrected 8-bit source samples must be in [0,255]")
    return result


def coordinate(value, *, fractional=False):
    if np.iscomplexobj(value):
        raise ValueError("Real source coordinate required")
    try:
        xy = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Finite xy required") from exc
    if xy.shape != (2,) or not np.isfinite(xy).all():
        raise ValueError("Finite xy required")
    if fractional and np.any(np.abs(xy) > .5):
        raise ValueError("Fractional current center must be within nearest pixel")
    return xy


class FractionalPointEdge(PointEdgeDiagnostic):
    """V36 background/edge model, with the point bank at actual fractional xy."""
    def __init__(self, current_xy):
        center = coordinate(current_xy, fractional=True)
        super().__init__()
        y, x = np.mgrid[-12:13, -12:13].astype(np.float64)
        point = []
        for sigma, dx, dy in self._point_info:
            point.append(np.exp(-((x-center[0]-dx)**2+(y-center[1]-dy)**2)/(2*sigma*sigma)))
        self._point, self._point_norm = self._templates(point)


def envelope(values):
    data = np.asarray(values, dtype=np.float64)
    if data.shape != (9,) or not np.isfinite(data).all():
        raise ValueError("All nine finite informative values required for envelope")
    lo, hi = float(data.min()), float(data.max())
    return dict(minimum=lo, median=float(np.median(data)), maximum=hi, span=hi-lo)


class SourceEvidenceDiagnostic:
    def __init__(self):
        self._pair = TemporalPointDiagnostic()

    def measure(self, current25, prior27, *, polarity, current_xy=(0., 0.), previous_xy=None):
        if polarity not in ("bright", "dark"):
            raise ValueError("Polarity must be bright or dark")
        current = pixels(current25, (25, 25))
        previous = pixels(prior27, (27, 27))
        center = coordinate(current_xy, fractional=True)
        previous_point = None if previous_xy is None else coordinate(previous_xy)
        model = FractionalPointEdge(center)
        sign = 1 if polarity == "bright" else -1
        current_finite = bool(np.isfinite(current).all())
        current_features = model.measure(current, polarity) if current_finite else None
        snapshot = dict(source_supported=current_finite,
            informative=bool(current_features and current_features["informative"]),
            features=current_features,
            physical_class="unknown", presence_classification=None)
        probes = []
        for dx, dy in SHIFTS:
            # prior27 grid is q=-13..13; shifted current25 support is -12..12.
            prior = previous[1+dy:26+dy, 1+dx:26+dx]
            supported = current_finite and bool(np.isfinite(prior).all())
            features = model.measure(current-prior, polarity) if supported else None
            informative = bool(features and features["informative"])
            if features:
                features["signed_point_amplitude_dn"] = sign*features["point_amplitude_dn"]
            mapped = None if previous_point is None else previous_point-(dx, dy)
            conditional = dict(available=False, status="unknown", reasons=[
                "unsupported_source_patch" if not supported else "missing_previous_actual_measurement"])
            if supported and mapped is not None:
                conditional = self._pair.measure(current, prior, previous_xy=mapped.tolist(),
                                                  polarity=polarity, current_xy=center.tolist())
            probes.append(dict(shift_xy=[dx, dy], source_supported=supported,
                contrast_informative=informative,
                reasons=[] if informative else ["uninformative_difference" if supported else "unsupported_source_patch"],
                difference_features=features,
                previous_point_xy=None if mapped is None else mapped.tolist(),
                conditional_pair=conditional,
                saturation_counts=dict(current_zero=int(np.count_nonzero(current == 0)),
                    current_255=int(np.count_nonzero(current == 255)),
                    interpolated_prior_zero=int(np.count_nonzero(prior == 0)),
                    interpolated_prior_255=int(np.count_nonzero(prior == 255)))))
        nominal = probes[SHIFTS.index((0, 0))]
        all_informative = all(probe["contrast_informative"] for probe in probes)
        return dict(diagnostic_only=True, classifier_promoted=False,
            current_xy=center.tolist(), previous_xy=None if previous_point is None else previous_point.tolist(),
            previous_actual_measurement_available=previous_point is not None,
            current_source=snapshot, probes=probes,
            source_supported_probes=sum(p["source_supported"] for p in probes),
            informative_probes=sum(p["contrast_informative"] for p in probes),
            nominal_contrast_available=nominal["contrast_informative"],
            envelope_available=all_informative,
            envelope={key: envelope([p["difference_features"][key] for p in probes]) for key in METRICS}
                     if all_informative else None,
            reasons=[] if all_informative else ["not_all_nine_probes_supported_and_informative"],
            missing_prior_point_may_contaminate_difference=previous_point is None,
            photometric_gain=1.0, photometric_offset=0.0,
            uncertainty_envelope_is_calibrated=False,
            independent_votes=False, lag_or_shift_selection_applied=False,
            warning="Source contrast and registration sensitivity only, not verified motion, presence, or airborne class")
