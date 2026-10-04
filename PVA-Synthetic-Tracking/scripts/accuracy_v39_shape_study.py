"""Deterministic source-only shape diagnostic and synthetic counterexamples.

This is not a detector, association correction, or output gate. The fixed
25x25 source patch is centered by the *existing detector*, never by truth.
Each family fits quadratic background + signed edge + polarity-constrained
compact template jointly. Families remain separate: a paired template is a
shape, not proof of one object, two objects, motion, or airborne class.

The larger banks search more locations and shapes on the same pixels used to
score the fit. Eight linear coefficients do NOT capture that search's full
complexity. No p-value, probability, AIC/BIC, noise threshold, or calibrated
model comparison is implied. Conditional gain is nonnegative by construction;
curved structure, noise, and exposure changes can produce such gains too.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from accuracy_v38_evidence import FractionalPointEdge, coordinate, pixels


Y, X = np.mgrid[-12:13, -12:13].astype(np.float64)
CENTER_GRID = (-4., -2., 0., 2., 4.)
ANGLES = tuple(k * math.pi / 4 for k in range(4))
FAMILIES = ("centered_isotropic", "localized_isotropic", "localized_elongated",
            "localized_equal_pair")


def gaussian(xy, sx, sy=None, theta=0.):
    """Unit-peak Gaussian, sampled without normalization or invented pixels."""
    sy = sx if sy is None else sy
    dx, dy = X-xy[0], Y-xy[1]
    u = dx*math.cos(theta)+dy*math.sin(theta)
    v = -dx*math.sin(theta)+dy*math.cos(theta)
    return np.exp(-.5*((u/sx)**2+(v/sy)**2))


def edge(width, theta, offset):
    return np.tanh((X*math.cos(theta)+Y*math.sin(theta)-offset)/width)


def paired(xy, sigma, separation, theta):
    delta = .5*separation*np.array([math.cos(theta), math.sin(theta)])
    # Both lobes have the same coefficient. Overlap can make peak > 1; the
    # reported DN coefficient multiplies each unit-peak component, not the sum.
    return gaussian(np.asarray(xy)-delta, sigma)+gaussian(np.asarray(xy)+delta, sigma)


class SourceShapeDiagnostic:
    """No truth, time, clip, lag, track history, or prior pixels are inputs.

    current_xy is the existing detector's fractional xy within its nearest
    source pixel. It is fixed per instance; no coordinate is changed in place.
    """

    def __init__(self, current_xy=(0., 0.)):
        self.current_xy = coordinate(current_xy, fractional=True).copy()
        self._baseline = FractionalPointEdge(self.current_xy)
        u, v = X/12, Y/12
        self._basis = np.column_stack([z.ravel() for z in
                                      (np.ones_like(X), u, v, u*u, u*v, v*v)])
        self._q = np.linalg.qr(self._basis, mode="reduced")[0]
        self._edge_info = [(w, a, o) for w in (1., 2., 4.)
                           for a in (k*math.pi/8 for k in range(8))
                           for o in (-2., 0., 2.)]
        self._edge, self._edge_norm = self._normalize(
            [edge(*params) for params in self._edge_info])
        self._families = {}
        for name in FAMILIES:
            centers = (-1., 0., 1.) if name == "centered_isotropic" else CENTER_GRID
            templates, info = [], []
            for dx in centers:
                for dy in centers:
                    xy = self.current_xy+(dx, dy)
                    common = dict(center_offset_xy=[dx, dy], center_xy=xy.tolist())
                    if name.endswith("isotropic"):
                        shapes = [(gaussian(xy, s), dict(sigma_px=s)) for s in (1., 2., 3.)]
                    elif name == "localized_elongated":
                        shapes = [(gaussian(xy, a, b, theta),
                                   dict(sigma_u_px=a, sigma_v_px=b, orientation_rad=theta))
                                  for a, b in ((1., 3.), (2., 4.)) for theta in ANGLES]
                    else:
                        shapes = [(paired(xy, s, d, theta),
                                   dict(sigma_px=s, separation_px=d, orientation_rad=theta))
                                  for s in (1., 1.5) for d in (4., 6.) for theta in ANGLES]
                    for template, shape in shapes:
                        templates.append(template)
                        info.append(dict(**common, **shape))
            unit, norm = self._normalize(templates)
            rho = unit @ self._edge.T
            denominator = 1-rho*rho
            if not np.isfinite(denominator).all() or np.any(denominator <= 1e-12):
                raise ValueError("Numerically degenerate edge/compact bank")
            self._families[name] = dict(unit=unit, norm=norm, rho=rho,
                denominator=denominator, info=info, centers=centers)

    def _normalize(self, templates):
        raw = np.asarray(templates, dtype=np.float64).reshape(len(templates), -1)
        residual = raw-(raw @ self._q) @ self._q.T
        norm = np.linalg.norm(residual, axis=1)
        if not np.isfinite(norm).all() or np.any(norm <= 1e-12):
            raise ValueError("Degenerate background-projected template")
        return residual/norm[:, None], norm

    def _candidate(self, bank, i, j, residual, edge_dots, positive_dot, polarity,
                   energy, best_edge_energy, edge_informative):
        sign = 1 if polarity == "bright" else -1
        b = float(positive_dot[i, j]/bank["denominator"][i, j])
        a = float(edge_dots[j]-bank["rho"][i, j]*sign*b)
        remainder = residual-a*self._edge[j]-sign*b*bank["unit"][i]
        sse = float(remainder @ remainder)
        # Roundoff clamp only: no data-dependent evidence threshold.
        gain = max(0., best_edge_energy-sse)
        width, angle, offset = self._edge_info[j]
        parameters = dict(bank["info"][i])
        boundary = any(abs(c) == max(abs(v) for v in bank["centers"])
                       for c in parameters["center_offset_xy"])
        return dict(compact=parameters,
            compact_amplitude_dn=float(b/bank["norm"][i]),
            compact_signed_amplitude_dn=float(sign*b/bank["norm"][i]),
            amplitude_convention="coefficient per unit-peak component; equal pair shares one coefficient",
            edge=dict(width_px=width, orientation_rad=angle, offset_px=offset,
                      signed_amplitude_dn=float(a/self._edge_norm[j])),
            residual_energy=sse,
            background_gain_fraction=max(0., min(1., (energy-sse)/energy)) if energy else 0.,
            gain_over_best_edge=gain,
            gain_over_best_edge_fraction=max(0., min(1., gain/best_edge_energy))
                                        if edge_informative else None,
            compact_coefficient_on_zero_boundary=bool(b == 0.),
            search_boundary_winner=boundary,
            compact_template_index=int(i), edge_template_index=int(j))

    def measure(self, patch, polarity):
        source = pixels(patch, (25, 25))
        if polarity not in ("bright", "dark"):
            raise ValueError("Polarity must be bright or dark")
        contract = dict(schema="seaqr.accuracy-v39-source-shape-diagnostic.v1",
            diagnostic_only=True, classifier_promoted=False, physical_class="unknown",
            presence_classification=None, physical_object_count=None,
            current_xy=self.current_xy.tolist(), polarity=polarity,
            lag_selection_applied=False, family_selection_applied=False,
            family_banks_are_non_nested=True, point_and_edge_fitted_jointly=True,
            source_localization_replaces_measurement=False,
            warning="Same-pixel searched fit gains are uncalibrated and are not object probabilities")
        if not np.isfinite(source).all():
            return dict(**contract, available=False, reasons=["unsupported_source_patch"],
                        baseline=None, families={})
        baseline = self._baseline.measure(source, polarity)
        centered = source.ravel()-source.mean()
        residual = centered-self._q @ (self._q.T @ centered)
        energy = float(residual @ residual)
        informative = energy > 1e-24*max(1., float(centered @ centered))
        if not informative:
            return dict(**contract, available=False, reasons=["uninformative_source_patch"],
                        baseline=baseline, families={})
        edge_dots = self._edge @ residual
        best_edge_index = int(np.argmax(edge_dots*edge_dots))
        edge_remainder = residual-edge_dots[best_edge_index]*self._edge[best_edge_index]
        best_edge_energy = float(edge_remainder @ edge_remainder)
        edge_informative = best_edge_energy > 1e-24*max(1., float(centered @ centered))
        sign = 1 if polarity == "bright" else -1
        families = {}
        for name, bank in self._families.items():
            point_dots = bank["unit"] @ residual
            positive_dot = np.maximum(sign*(point_dots[:, None]-bank["rho"]*edge_dots), 0.)
            total_gain = edge_dots[None, :]**2+positive_dot**2/bank["denominator"]
            i, j = np.unravel_index(int(np.argmax(total_gain)), total_gain.shape)
            best = self._candidate(bank, i, j, residual, edge_dots, positive_dot,
                                   polarity, energy, best_edge_energy, edge_informative)
            # Keep the winner and two best *other fitted locations*. These
            # are not three physical objects and no evidence threshold is used.
            alternatives = []
            for dx in bank["centers"]:
                for dy in bank["centers"]:
                    indices = [k for k, p in enumerate(bank["info"])
                               if p["center_offset_xy"] == [dx, dy]]
                    ii, jj = np.unravel_index(int(np.argmax(total_gain[indices])),
                                             total_gain[indices].shape)
                    alternatives.append(self._candidate(bank, indices[ii], jj, residual,
                        edge_dots, positive_dot, polarity, energy, best_edge_energy, edge_informative))
            alternatives.sort(key=lambda a: a["residual_energy"])
            other_locations = [a for a in alternatives
                               if a["compact"]["center_xy"] != best["compact"]["center_xy"]]
            # Analytic-gain versus direct-SSE roundoff ties must not silently
            # exclude the declared winner from this presentation-only list.
            families[name] = dict(best=best, location_alternatives=[best]+other_locations[:2],
                location_alternatives_include_best=True,
                location_alternatives_order="winner first, remaining locations by direct residual energy",
                location_alternatives_are_object_count=False,
                compact_template_count=len(bank["info"]), edge_template_count=len(self._edge_info),
                joint_template_pair_count=int(total_gain.size),
                nominal_linear_coefficient_count=8,
                searched_location_count=len(bank["centers"])**2,
                center_grid_px=list(bank["centers"]),
                score_uses_fit_pixels=True, complexity_calibrated=False,
                search_complexity_excluded_from_linear_coefficient_count=True)
        return dict(**contract, available=True, reasons=[], baseline=baseline,
                    background_residual_energy=energy, best_edge_residual_energy=best_edge_energy,
                    edge_residual_informative=bool(edge_informative),
                    families=families)


def synthetic_cases():
    """Declared mechanics/counterexamples, not simulated airborne truth.

    Generation coordinates below are never inference inputs. The detector
    coordinate remains (0,0), including out-of-bank offsets 6 and 8 pixels.
    """
    background = 110+.1*X-.15*Y+.004*X*Y
    line = 22*edge(2., math.pi/4, 0.)
    cases = []
    for offset in (0., 2., 4., 6., 8.):
        cases.append(dict(name=f"point_offset_{int(offset)}_on_edge", patch=background+line+
            12*gaussian((offset, 0), 1.), polarity="bright",
            construction=dict(kind="single_compact_component", centers_xy=[[offset, 0.]],
                              amplitude_dn=12., source_center_is_not_inference_input=True)))
    for name, template, construction in (
        ("equal_pair_on_edge", paired((2., -2.), 1., 6., 0.),
         dict(kind="two_equal_compact_components", centers_xy=[[-1., -2.], [5., -2.]])),
        ("elongated_on_edge", gaussian((2., -2.), 1., 3., math.pi/4),
         dict(kind="elongated_component", centers_xy=[[2., -2.]])),
        ("nearby_unequal_separate_components", gaussian((-2., 0.), 1.)+.6*gaussian((2., 0.), 1.),
         dict(kind="two_distinct_synthetic_components", centers_xy=[[-2., 0.], [2., 0.]])),
        ("off_grid_pair", paired((3.3, -1.4), 1.3, 5., .3),
         dict(kind="off_bank_pair", centers_xy=None)),
        ("curved_cloud_boundary", 2.5*np.tanh((Y-.09*X*X+1.)/2.),
         dict(kind="continuous_curved_structure_without_injected_point", centers_xy=[])),
        ("curved_cloud_knot", 2.5*np.tanh((Y-.09*X*X+1.)/2.)+.7*gaussian((2., 0.), 4., 2.),
         dict(kind="continuous_curved_structure_and_broad_knot", centers_xy=[])),
    ):
        cases.append(dict(name=name, patch=background+line+12*template, polarity="bright",
                          construction=construction))
    # Same stationary structure, two exposures: strong source shape alone
    # cannot establish motion; quadratic subtraction does not remove a
    # multiplicative change in nonquadratic scene texture.
    stationary = background+line+9*gaussian((2., 0.), 1.)
    for factor in (1., 1.15):
        cases.append(dict(name=f"stationary_structure_exposure_{factor:g}",
            patch=factor*stationary, polarity="bright",
            construction=dict(kind="stationary_structure_exposure", exposure_factor=factor,
                              stationary=True, same_scene_group="exposure_pair")))
    rng = np.random.default_rng(39001)
    for n in range(3):
        cases.append(dict(name=f"noise_only_{n}", patch=background+rng.normal(0., 1., (25, 25)),
            polarity="bright", construction=dict(kind="deterministic_noise_without_injected_point")))
    for n, (xy, amplitude) in enumerate((((2., 0.), 12.), ((2.2, 0.), 12.),
                                        ((2.4, 0.), 0.), ((2.4, .2), 12.), ((2.4, .4), 12.))):
        cases.append(dict(name=f"slow_turn_intermittent_{n}", patch=background+line+
            amplitude*gaussian(xy, 1.), polarity="bright",
            construction=dict(kind="ordered_slow_turn_intermittent_sequence", sequence_index=n,
                              centers_xy=[list(xy)], amplitude_dn=amplitude,
                              note="Frames analyzed independently; no velocity/visibility assumptions")))
    cases.append(dict(name="dark_off_center_point_on_edge", patch=255-(background+line+
        12*gaussian((4., 0.), 1.)), polarity="dark",
        construction=dict(kind="dark_single_component", centers_xy=[[4., 0.]], amplitude_dn=12.)))
    return cases


def run_study():
    model = SourceShapeDiagnostic()
    cases = []
    for case in synthetic_cases():
        patch = case["patch"]
        cases.append(dict(name=case["name"], synthetic_construction=case["construction"],
            source_patch_sha256=hashlib.sha256(patch.astype("<f8").tobytes()).hexdigest(),
            diagnostic=model.measure(patch, case["polarity"])))
    return dict(schema="seaqr.accuracy-v39-synthetic-shape-study.v1", deterministic_seed=39001,
        diagnostic_only=True, real_media_read=False, data_thresholds_chosen=False,
        frozen_pipeline_changed=False, inference_uses_truth_coordinates=False,
        physical_classification_claimed=False, detection_accuracy_claimed=False,
        cases=cases,
        limitations=["Bank search and shape capacity are not statistically calibrated",
                     "Centers outside the finite bank and off-bank shapes remain mismatched",
                     "Equal-pair templates do not establish physical object number or identity",
                     "Curved structure and noise can yield positive conditional compact gain",
                     "Static source shape cannot establish motion or airborne class",
                     "No independent test pixels or held-out nuisance calibration",
                     "No lag, shift, family, or temporal output policy selected"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional fresh JSON path; existing files are refused")
    args = parser.parse_args()
    report = run_study()
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+"\n"
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
        print(json.dumps(dict(output=str(args.output), cases=len(report["cases"]))))
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
