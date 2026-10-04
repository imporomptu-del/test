"""Constrained two-frame source diagnostics, never an object classifier.

The caller supplies an independently background-registered, photometrically
corrected prior patch. A displaced previous point and a current point have
independent nonnegative contrast coefficients. Edges are a non-nested
alternative, not a negative class. Bank maxima/gains are not probabilities.
"""

import math

import numpy as np


class TemporalPointDiagnostic:
    SIZE = 25
    SIGMAS = (1.0, 2.0, 3.0)
    OFFSETS = (-1.0, 0.0, 1.0)
    EDGE_WIDTHS = (1.0, 2.0, 4.0)
    EDGE_OFFSETS = (-2.0, 0.0, 2.0)
    EDGE_ANGLES = tuple(k * math.pi / 8 for k in range(8))
    MIN_DISPLACEMENT_PX = 1.0
    MAX_PAIR_CONDITION = 10000.0

    def __init__(self):
        self._y, self._x = np.mgrid[-12:13, -12:13].astype(np.float64)
        x, y = self._x / 12, self._y / 12
        design = np.stack((np.ones_like(x), x, y, x*x, x*y, y*y), axis=-1)
        self._q = np.linalg.qr(design.reshape(-1, 6), mode="reduced")[0]
        edge, self._edge_info = [], []
        for width in self.EDGE_WIDTHS:
            for theta in self.EDGE_ANGLES:
                for offset in self.EDGE_OFFSETS:
                    edge.append(np.tanh((self._x*math.cos(theta) +
                                         self._y*math.sin(theta)-offset)/width))
                    self._edge_info.append((width, theta, offset))
        self._edge, self._edge_norm = self._templates(edge)

    def _templates(self, patches):
        t = np.asarray(patches, dtype=np.float64).reshape(len(patches), -1)
        residual = t - (t @ self._q) @ self._q.T
        norms = np.linalg.norm(residual, axis=1)
        if not np.isfinite(norms).all() or np.any(norms <= 1e-12):
            raise ValueError("Degenerate temporal template")
        return residual / norms[:, None], norms

    def _gaussian(self, sigma, xy):
        return np.exp(-((self._x-xy[0])**2 + (self._y-xy[1])**2)/(2*sigma*sigma))

    @staticmethod
    def _xy(value, name):
        if np.iscomplexobj(value):
            raise ValueError(f"Real {name} xy required")
        try:
            result = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Finite {name} xy required") from exc
        if result.shape != (2,) or not np.isfinite(result).all():
            raise ValueError(f"Finite {name} xy required")
        return result

    @staticmethod
    def _patch(value):
        if np.iscomplexobj(value):
            raise ValueError("Real 25x25 patch required")
        try:
            result = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("Finite 25x25 patch required") from exc
        if result.shape != (25, 25) or not np.isfinite(result).all():
            raise ValueError("Finite 25x25 patch required")
        return result

    def _fit_pair_bank(self, residual, prior, current, nonnegative_current):
        """Solve all normalized two-column fits, including active boundaries.

        Condition is that of the two-column normal matrix, not its square
        root. Coefficients multiply normalized background-projected columns.
        Both the null and zero are included for every admissible pair.
        """
        rho = prior @ current.T
        pu = prior @ residual
        cu = current @ residual
        condition = (1 + np.abs(rho)) / np.maximum(1 - np.abs(rho), 1e-300)
        valid = condition <= self.MAX_PAIR_CONDITION
        # Do not form enormous cancelling coefficients for excluded pairs.
        denom = np.where(valid, 1 - rho*rho, 1.0)
        pa = (pu[:, None] - rho*cu[None, :])/denom
        ca = (cu[None, :] - rho*pu[:, None])/denom
        interior = (pa >= 0) & valid
        if nonnegative_current:
            interior &= ca >= 0
        gain = np.where(interior, pa*pu[:, None] + ca*cu[None, :], -np.inf)
        best_pa = np.where(interior, pa, 0.0)
        best_ca = np.where(interior, ca, 0.0)
        # Prior-only, current-only, and zero boundary solutions. Strict >
        # leaves deterministic first-bank/interior ties without a tolerance.
        prior_only = np.broadcast_to(np.maximum(pu, 0)[:, None], rho.shape)
        current_only = np.broadcast_to(
            (np.maximum(cu, 0) if nonnegative_current else cu)[None, :], rho.shape)
        for a, b in ((prior_only, np.zeros_like(rho)),
                     (np.zeros_like(rho), current_only),
                     (np.zeros_like(rho), np.zeros_like(rho))):
            g = 2*(a*pu[:, None]+b*cu[None, :])-(a*a+b*b+2*rho*a*b)
            take = valid & (g > gain)
            gain = np.where(take, g, gain)
            best_pa = np.where(take, a, best_pa)
            best_ca = np.where(take, b, best_ca)
        if not valid.any():
            return None
        p, c = np.unravel_index(int(np.argmax(gain)), gain.shape)
        a, b = float(best_pa[p, c]), float(best_ca[p, c])
        remainder = residual - a*prior[p] - b*current[c]
        return dict(prior_index=int(p), current_index=int(c), prior_coefficient=a,
                    current_coefficient=b, sse=float(remainder @ remainder),
                    condition=float(condition[p, c]))

    def measure(self, current_patch, registered_prior_patch, *, previous_xy,
                polarity, current_xy=(0.0, 0.0)):
        if polarity not in ("bright", "dark"):
            raise ValueError("Polarity must be bright or dark")
        current = self._patch(current_patch)
        previous = self._patch(registered_prior_patch)
        pxy = self._xy(previous_xy, "previous")
        cxy = self._xy(current_xy, "current")
        if np.any(np.abs(cxy) > 0.5):
            raise ValueError("Current xy must be fractional offset from nearest pixel")
        distance = math.hypot(float(cxy[0]-pxy[0]), float(cxy[1]-pxy[1]))
        if not math.isfinite(distance):
            raise ValueError("Relative displacement exceeds finite numeric range")
        result = dict(available=False, status="unknown", reasons=[],
                      diagnostic_only=True, displacement_px=distance,
                      previous_xy=pxy.tolist(), current_xy=cxy.tolist())
        if np.any(np.abs(pxy) > self.SIZE//2):
            result["reasons"].append("previous_center_outside_patch")
            return result
        if distance < self.MIN_DISPLACEMENT_PX:
            result["reasons"].append("subpixel_relative_displacement")
        flat = (current-previous).reshape(-1)
        centered = flat-flat.mean()
        residual = centered-self._q @ (self._q.T @ centered)
        energy = float(residual @ residual)
        if not math.isfinite(energy):
            raise ValueError("Patch residual exceeds finite numeric range")
        result["background_residual_energy"] = energy
        result["residual_rms_dn"] = math.sqrt(energy/flat.size)
        informative = energy > 1e-24*max(1.0, float(centered @ centered))
        if not informative:
            result["reasons"].append("uninformative_difference")
            return result
        sign = 1.0 if polarity == "bright" else -1.0
        prior, prior_norm = self._templates([self._gaussian(s, pxy) for s in self.SIGMAS])
        prior *= -sign
        point, point_info = [], []
        for sigma in self.SIGMAS:
            for dx in self.OFFSETS:
                for dy in self.OFFSETS:
                    point.append(self._gaussian(sigma, cxy + (dx, dy)))
                    point_info.append((sigma, dx, dy))
        point, point_norm = self._templates(point)
        point *= sign
        null_dots = np.maximum(prior @ residual, 0.0)
        ni = int(np.argmax(null_dots*null_dots))
        null_residual = residual-null_dots[ni]*prior[ni]
        null_sse = float(null_residual @ null_residual)
        pf = self._fit_pair_bank(residual, prior, point, True)
        ef = self._fit_pair_bank(residual, prior, self._edge, False)
        if pf is None or ef is None:
            result["reasons"].append("ill_conditioned_model_bank")
            return result
        # Floating-point clamping only; never turn nonnegative nested gain
        # into a decision. Banks are unequal and gains are uncalibrated.
        pg = float(np.clip((null_sse-pf["sse"])/energy, 0.0, 1.0))
        eg = float(np.clip((null_sse-ef["sse"])/energy, 0.0, 1.0))
        sigma, dx, dy = point_info[pf["current_index"]]
        width, angle, offset = self._edge_info[ef["current_index"]]
        result.update(
            null_sse=null_sse, point_sse=pf["sse"], edge_sse=ef["sse"],
            null_previous_sigma_px=self.SIGMAS[ni],
            null_previous_amplitude_dn=float(null_dots[ni]/prior_norm[ni]),
            point_gain_fraction=pg, edge_gain_fraction=eg,
            point_minus_edge_fraction=pg-eg,
            point_current_amplitude_dn=pf["current_coefficient"]/float(point_norm[pf["current_index"]]),
            point_previous_amplitude_dn=pf["prior_coefficient"]/float(prior_norm[pf["prior_index"]]),
            point_sigma_px=sigma, point_offset_xy=[dx, dy],
            point_previous_sigma_px=self.SIGMAS[pf["prior_index"]],
            point_pair_condition=pf["condition"],
            edge_amplitude_dn=ef["current_coefficient"]/float(self._edge_norm[ef["current_index"]]),
            edge_previous_amplitude_dn=ef["prior_coefficient"]/float(prior_norm[ef["prior_index"]]),
            edge_width_px=width, edge_orientation_rad=angle, edge_offset_px=offset,
            edge_previous_sigma_px=self.SIGMAS[ef["prior_index"]],
            edge_pair_condition=ef["condition"],
        )
        result["available"] = not result["reasons"]
        result["status"] = "available" if result["available"] else "unknown"
        return result
