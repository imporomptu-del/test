"""Source-pixel point-versus-edge evidence, not an object classifier.

Fixed, interpretable template banks: every fit has a quadratic background plus
one contrast coefficient. Bank sizes differ, so their maximum improvements are
not calibrated probabilities, Bayes factors, or significance tests. Inference
uses one current native-resolution patch; labels and track history are absent.
"""
import math

import numpy as np


class PointEdgeDiagnostic:
    SIZE = 25
    POINT_SIGMAS = (1.0, 2.0, 3.0)
    POINT_OFFSETS = (-1.0, 0.0, 1.0)
    EDGE_WIDTHS = (1.0, 2.0, 4.0)
    EDGE_OFFSETS = (-2.0, 0.0, 2.0)
    EDGE_ORIENTATIONS = tuple(k * math.pi / 8 for k in range(8))

    def __init__(self):
        axis = np.arange(self.SIZE, dtype=np.float64) - self.SIZE // 2
        y, x = np.meshgrid(axis, axis, indexing="ij")
        # Scaling coordinates improves conditioning without changing the space.
        u, v = x / 12, y / 12
        basis = np.stack((np.ones_like(x), u, v, u*u, u*v, v*v), axis=-1)
        self._q = np.linalg.qr(basis.reshape(-1, 6), mode="reduced")[0]
        point, self._point_info = [], []
        for sigma in self.POINT_SIGMAS:
            for dx in self.POINT_OFFSETS:
                for dy in self.POINT_OFFSETS:
                    point.append(np.exp(-((x-dx)**2+(y-dy)**2)/(2*sigma*sigma)))
                    self._point_info.append((sigma, dx, dy))
        edge, self._edge_info = [], []
        for width in self.EDGE_WIDTHS:
            for theta in self.EDGE_ORIENTATIONS:
                for offset in self.EDGE_OFFSETS:
                    edge.append(np.tanh((x*math.cos(theta)+y*math.sin(theta)-offset)/width))
                    self._edge_info.append((width, theta, offset))
        self._point, self._point_norm = self._templates(point)
        self._edge, self._edge_norm = self._templates(edge)

    def _templates(self, patches):
        t = np.stack(patches).reshape(len(patches), -1)
        r = t - (t @ self._q) @ self._q.T
        norm = np.linalg.norm(r, axis=1)
        if not np.all(norm > 0):
            raise ValueError("Degenerate diagnostic template")
        return r / norm[:, None], norm

    def measure(self, patch, polarity):
        if polarity not in ("bright", "dark"):
            raise ValueError("Polarity must be bright or dark")
        try:
            patch = np.asarray(patch, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("Numeric patch required") from exc
        if patch.shape != (self.SIZE, self.SIZE) or not np.isfinite(patch).all():
            raise ValueError("Finite native 25x25 patch required")
        flat = patch.reshape(-1)
        # Remove the DC component first to make offset invariance numerically
        # stable on low-contrast 8-bit patches; constant belongs to Q already.
        centered = flat - flat.mean()
        residual = centered - self._q @ (self._q.T @ centered)
        energy = float(residual @ residual)
        # Round-off guard, not a scene/noise/detection threshold.
        informative = energy > 1e-24 * max(1.0, float(centered @ centered))
        sign = 1 if polarity == "bright" else -1
        point_dots = np.maximum(sign * (self._point @ residual), 0)
        edge_dots = self._edge @ residual
        p = int(np.argmax(point_dots * point_dots))
        e = int(np.argmax(edge_dots * edge_dots))
        pg = min(1.0, float(point_dots[p]**2 / energy)) if informative else 0.0
        eg = min(1.0, float(edge_dots[e]**2 / energy)) if informative else 0.0
        sigma, dx, dy = self._point_info[p]
        width, theta, offset = self._edge_info[e]
        edge_unit = self._edge[e]
        edge_residual = residual - edge_dots[e] * edge_unit
        edge_energy = float(edge_residual @ edge_residual)
        # A point may coexist with an edge. This additional nested diagnostic
        # tests the remaining point contribution, without an acceptance rule.
        # Use unit point vectors here, then restore original template scaling
        # when reporting the fitted coefficient in native DN.
        conditional = self._point - (self._point @ edge_unit)[:, None] * edge_unit
        conditional_norm = np.linalg.norm(conditional, axis=1)
        if not np.isfinite(conditional_norm).all() or not np.all(conditional_norm > 0):
            raise ValueError("Degenerate conditional point template")
        conditional_unit = conditional / conditional_norm[:, None]
        conditional_dots = np.maximum(sign * (conditional_unit @ edge_residual), 0)
        cp = int(np.argmax(conditional_dots * conditional_dots))
        conditional_informative = edge_energy > 1e-24 * max(1.0, float(centered @ centered))
        cg = min(1.0, float(conditional_dots[cp]**2 / edge_energy)) if conditional_informative else 0.0
        csigma, cdx, cdy = self._point_info[cp]
        return dict(
            informative=bool(informative),
            background_residual_energy=energy,
            point_gain_fraction=pg,
            edge_gain_fraction=eg,
            point_minus_edge_fraction=pg-eg,
            point_amplitude_dn=float(point_dots[p]/self._point_norm[p]) if informative else 0.0,
            point_sigma_px=sigma,
            point_offset_xy=[dx, dy],
            edge_width_px=width,
            edge_orientation_rad=theta,
            edge_offset_px=offset,
            edge_amplitude_dn=float(edge_dots[e]/self._edge_norm[e]) if informative else 0.0,
            residual_rms_dn=math.sqrt(energy/flat.size),
            point_absolute_gain=pg*energy,
            edge_absolute_gain=eg*energy,
            edge_residual_energy=edge_energy,
            conditional_informative=bool(conditional_informative),
            point_gain_after_edge_fraction=cg,
            point_after_edge_absolute_gain=cg*edge_energy,
            point_after_edge_amplitude_dn=float(conditional_dots[cp]/(conditional_norm[cp]*self._point_norm[cp])) if conditional_informative else 0.0,
            point_after_edge_sigma_px=csigma,
            point_after_edge_offset_xy=[cdx, cdy],
        )
