"""Fixed annulus-only local registration for the v37 temporal diagnostic.

Inputs already share the current-camera coordinate grid. This module does not
know clip IDs, timestamps, truth labels, detector thresholds or track velocity.
Availability conditions are explicit diagnostic assumptions, not confidence.
"""
import hashlib
import math

import numpy as np


class AnnulusRegistration:
    CURRENT_SIZE = 49
    PRIOR_SIZE = 53
    ANNULUS_INNER = 14
    ANNULUS_OUTER = 22
    PRIOR_POINT_RADIUS = 9
    SEARCH_RADIUS = 2.0
    SEARCH_STEP = 0.5
    MINIMUM_PIXELS = 128
    MINIMUM_VARIANCE = 0.25
    GAIN_RANGE = (0.5, 2.0)
    MAXIMUM_RESIDUAL_VARIANCE_FRACTION = 0.25
    MINIMUM_NONLOCAL_DISTANCE = 1.0
    MINIMUM_ALTERNATE_MSE_GAP = 1.0 / 12.0

    def __init__(self):
        axis = np.arange(-24, 25, dtype=np.float64)
        self._y, self._x = np.meshgrid(axis, axis, indexing="ij")
        radius = np.maximum(np.abs(self._x), np.abs(self._y))
        self._annulus = (radius >= self.ANNULUS_INNER) & (radius <= self.ANNULUS_OUTER)
        shifts = np.arange(-self.SEARCH_RADIUS, self.SEARCH_RADIUS + self.SEARCH_STEP / 2,
                           self.SEARCH_STEP)
        self._shifts = tuple((float(dx), float(dy)) for dx in shifts for dy in shifts)

    @staticmethod
    def _input(value, shape):
        if np.iscomplexobj(value):
            raise ValueError("Real-valued registration pixels required")
        try:
            array = np.asarray(value, dtype=np.float64)
        except (ValueError, TypeError) as exc:
            raise ValueError("Numeric registration pixels required") from exc
        if array.shape != shape or np.isinf(array).any():
            raise ValueError("Expected fixed input shape without infinite pixels")
        # NaNs are allowed as missing warp/border support, not replaced with zero.
        return array

    def _sample(self, prior, dx, dy):
        """Bilinear prior(q+shift); zero-weight corners are not observations."""
        x = self._x + 26 + dx
        y = self._y + 26 + dy
        x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
        x1, y1 = np.minimum(x0 + 1, 52), np.minimum(y0 + 1, 52)
        fx, fy = x - x0, y - y0
        result = np.zeros_like(x)
        finite = np.ones_like(x, dtype=bool)
        nonsaturated = np.ones_like(x, dtype=bool)
        for yy, xx, weight in (
                (y0, x0, (1-fx)*(1-fy)), (y0, x1, fx*(1-fy)),
                (y1, x0, (1-fx)*fy), (y1, x1, fx*fy)):
            values = prior[yy, xx]
            used = weight > 0
            observed = np.isfinite(values)
            finite &= ~used | observed
            nonsaturated &= ~used | (observed & (values > 0) & (values < 255))
            result += np.where(used & observed, values, 0.0) * weight
        result[~finite] = np.nan
        return result, nonsaturated

    def measure(self, current49, prior53, prior_point_xy=None):
        current = self._input(current49, (49, 49))
        prior = self._input(prior53, (53, 53))
        mask = self._annulus.copy()
        point = None
        if prior_point_xy is not None:
            if (not isinstance(prior_point_xy, (list, tuple, np.ndarray))
                    or np.ndim(prior_point_xy) != 1
                    or len(prior_point_xy) != 2
                    or np.iscomplexobj(prior_point_xy)
                    or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.number))
                           or not math.isfinite(float(v)) for v in prior_point_xy)):
                raise ValueError("Finite previous measured point coordinates required")
            point = [float(v) for v in prior_point_xy]
            # Union of the point's radius9 footprint across the FULL shift box.
            # A radius11 circle would not cover the diagonal corner shift.
            ex = np.maximum(np.abs(self._x - point[0]) - self.SEARCH_RADIUS, 0)
            ey = np.maximum(np.abs(self._y - point[1]) - self.SEARCH_RADIUS, 0)
            mask &= np.hypot(ex, ey) > self.PRIOR_POINT_RADIUS
        mask &= np.isfinite(current) & (current > 0) & (current < 255)
        before_prior = int(mask.sum())
        sampled, valid_masks, individual_support = [], [], []
        for dx, dy in self._shifts:
            sample, valid = self._sample(prior, dx, dy)
            sampled.append(sample)
            valid_masks.append(valid)
            individual_support.append(int(np.count_nonzero(mask & valid)))
        # Form this once; every hypothesis must use exactly these same pixels.
        for valid in valid_masks:
            mask &= valid
        count = int(mask.sum())
        current_patch = current[12:37, 12:37].copy()
        answer = dict(
            available=False, status="unavailable", reasons=[], shift_xy=None,
            gain=None, offset=None, mse=None, source_variance=None,
            current_annulus_variance=None, mask_count=count,
            positivefit=False, boundarywinner=None,
            alternate_nonlocal_mse_gap=None, alternate_nonlocal_mse=None,
            alternate_nonlocal_shift_xy=None,
            current_patch=current_patch, registered_prior_patch=None,
            registered_prior_patch_photometrically_corrected=False,
            prior_point_xy=point,
            hypotheses=[],
            mask_sha256=hashlib.sha256(mask.astype(np.uint8).tobytes()).hexdigest(),
            common_mask_for_all_hypotheses=True,
            geometry=dict(current_size=49, prior_size=53, evaluation_size=25,
                annulus_chebyshev_radii=[self.ANNULUS_INNER, self.ANNULUS_OUTER],
                prior_exclusion_radius=self.PRIOR_POINT_RADIUS,
                prior_exclusion_dilated_by_search_box=True,
                residual_shift_radius=self.SEARCH_RADIUS, residual_shift_step=self.SEARCH_STEP,
                prior_sampling="bilinear at current q + shift; raw geometric values"),
            annulus_pixels_before_prior_validity=before_prior,
            individual_hypothesis_support_counts=individual_support,
            texture_diagnostics={},
            ambiguity_diagnostics=dict(minimum_nonlocal_distance=self.MINIMUM_NONLOCAL_DISTANCE,
                minimum_mse_gap_dn_squared=self.MINIMUM_ALTERNATE_MSE_GAP,
                tie_policy="minimum MSE, then shift squared norm, then dx, then dy"),
            availability_assumptions=dict(minimum_mask_pixels=self.MINIMUM_PIXELS,
                minimum_variance_dn_squared=self.MINIMUM_VARIANCE,
                minimum_gain=self.GAIN_RANGE[0], maximum_gain=self.GAIN_RANGE[1],
                residual_mse_must_be_below_current_variance_fraction=self.MAXIMUM_RESIDUAL_VARIANCE_FRACTION,
                nonboundary_winner_required=True, calibrated_confidence=False),
        )
        if count < self.MINIMUM_PIXELS:
            answer.update(status="insufficient_common_support", reasons=["insufficient_common_support"])
            return answer
        observed = current[mask]
        centered = observed - observed.mean()
        current_variance = float(np.mean(centered*centered))
        answer["current_annulus_variance"] = current_variance
        gy, gx = np.gradient(current)
        gradient_mask = mask & np.isfinite(gx) & np.isfinite(gy)
        gradients = np.column_stack((gx[gradient_mask], gy[gradient_mask]))
        eigenvalues = (np.linalg.eigvalsh(gradients.T @ gradients / len(gradients)).tolist()
                       if len(gradients) else None)
        answer["texture_diagnostics"] = dict(
            gradient_sample_count=int(len(gradients)),
            current_gradient_second_moment_eigenvalues=eigenvalues,
            current_annulus_variance_dn_squared=current_variance,
            note="Gradient spectrum is diagnostic only; availability uses declared variance and ambiguity checks.",
        )
        fits = []
        for index, ((dx, dy), sample) in enumerate(zip(self._shifts, sampled)):
            values = sample[mask]
            values_centered = values - values.mean()
            source_variance = float(np.mean(values_centered*values_centered))
            fit = dict(shift_xy=[dx, dy], mask_count=count, source_variance=source_variance,
                       gain=None, offset=None, mse=None, valid_fit=False, reason=None)
            if source_variance < self.MINIMUM_VARIANCE:
                fit["reason"] = "insufficient_source_texture"
            else:
                gain = float(np.dot(values_centered, centered) / np.dot(values_centered, values_centered))
                offset = float(observed.mean() - gain*values.mean())
                residual = observed - (gain*values + offset)
                mse = float(np.mean(residual*residual))
                fit.update(gain=gain, offset=offset, mse=mse)
                if not all(math.isfinite(v) for v in (gain, offset, mse)):
                    fit["reason"] = "nonfinite_fit"
                elif not self.GAIN_RANGE[0] <= gain <= self.GAIN_RANGE[1]:
                    fit["reason"] = "gain_outside_declared_range"
                else:
                    fit.update(valid_fit=True, reason="admissible")
                    fits.append((mse, dx*dx+dy*dy, dx, dy, index))
            answer["hypotheses"].append(fit)
        if not fits:
            answer.update(status="no_admissible_fit", reasons=["no_admissible_fit"])
            return answer
        best = min(fits)
        mse, _, dx, dy, index = best
        best_record = answer["hypotheses"][index]
        registered = sampled[index][12:37, 12:37].copy()
        answer.update(shift_xy=[dx, dy], gain=best_record["gain"], offset=best_record["offset"],
            mse=mse, source_variance=best_record["source_variance"], positivefit=True,
            boundarywinner=abs(dx) == self.SEARCH_RADIUS or abs(dy) == self.SEARCH_RADIUS,
            registered_prior_patch=registered)
        alternatives = [fit for fit in fits
                        if math.hypot(fit[2]-dx, fit[3]-dy) >= self.MINIMUM_NONLOCAL_DISTANCE]
        if alternatives:
            alternate = min(alternatives)
            answer.update(alternate_nonlocal_mse=alternate[0],
                alternate_nonlocal_mse_gap=alternate[0]-mse,
                alternate_nonlocal_shift_xy=[alternate[2], alternate[3]])
        answer["ambiguity_diagnostics"].update(valid_fit_count=len(fits),
            alternate_nonlocal_fit_count=len(alternatives))
        reasons = []
        if current_variance < self.MINIMUM_VARIANCE:
            reasons.append("insufficient_current_texture")
        if answer["boundarywinner"]:
            reasons.append("search_boundary_winner")
        if not mse < self.MAXIMUM_RESIDUAL_VARIANCE_FRACTION * current_variance:
            reasons.append("annulus_residual_too_large")
        if not alternatives or answer["alternate_nonlocal_mse_gap"] < self.MINIMUM_ALTERNATE_MSE_GAP:
            reasons.append("ambiguous_nonlocal_registration")
        if not np.isfinite(current_patch).all() or not np.isfinite(registered).all():
            reasons.append("invalid_evaluation_patch")
        answer.update(available=not reasons, status=reasons[0] if reasons else "available", reasons=reasons)
        return answer
