# v36 source-context diagnostic, pre-extraction plan

Do not change the frozen shadow_01 experiment, existing references, detector,
tracking, learning protection, or v34/v35 defaults. No RAW16, sealed clips, remote
work or additional media-wide search. The failed simple gates motivate a new
diagnostic, not retuning those gates.

## Hypothesis and fixed features

The existing half-height footprint is computed on the filtered response; cloud
boundaries can produce small footprints too. Test whether the **original 8-bit
pixels** around an actual qualified measurement favor a compact point or an
extended edge. This is a single-current-frame diagnostic, not an airborne class
classifier, a likelihood ratio, or a claim of improved end-to-end detection.

Use an unresampled native 25x25 grayscale patch centered at the nearest source
pixel to the logged measurement, with no contrast stretch or location masking.
Fit a quadratic background (constant, x, y, x², xy, y²) and compare one additional
contrast term from two fixed template banks:

- Gaussian points: sigma 1, 2, 3px; center offsets {-1,0,1}px in x and y; contrast
  nonnegative for the candidate polarity.
- Blurred straight edges: tanh profile with widths 1, 2, 4px, eight orientations
  k*pi/8, offsets {-2,0,2}px; either sign of contrast.

Orthogonalize every template against the same background space. Record fractional
residual-energy reduction, point-minus-edge margin, best parameters, contrast,
and residual RMS. Different bank sizes and correlated pixels preclude calibrated
statistical confidence. No learned thresholds, parameter sweeps, scene-specific
branches, annotations, time or coordinates enter the feature function. Reject
malformed input; flag constant/quadratic patches as uninformative, not negatives.

A point can coexist with an edge, so also record a nested diagnostic: project
the point bank orthogonal to the best fitted edge as well as the quadratic
background, then measure the nonnegative point improvement on the remaining
residual. Record its absolute/fractional improvement and fitted parameters.
No acceptance threshold is defined for this conditional diagnostic. This
addition was specified before extraction in response to independent design
review; exclusive point-versus-edge competition alone has an obvious mixed-scene
failure mode. Tests compare these projections with joint seven/eight-column LS.

## Bounded selection and evidence

Only decode allowed source files 0029 and 0126. Verify their full SHA256 against
audited v34 metadata before and after extraction. Walk sequentially from frame0
to the last needed frame with the local decoder; do not seek and assume a frame.
Check dimensions and FPS. Convert BGR to grayscale using OpenCV. Native pixels
are not the warped/temporal detector response, and local versus Jetson decoder
builds may differ. Record OpenCV version, source indices, integer patch origins,
original measurement coordinates, patch digests and patch arrays for audit.

Select all same-polarity qualified measured alternatives within the existing
dense and stricter pilot match gates, plus all required original anchor matches.
Use their actual measured positions, never annotation-centered recentering.
Deduplicate clip/frame/segment/track keys but preserve all sample memberships.
Evaluate every one of the existing 70 qualified measured nuisance-control
responses within all seven fixed source ROIs/time windows. Missing/edge-truncated
patches must be explicit errors, not selectively omitted.

The dense denominator stays285 (baseline284 matches); the overlapping strict
pilot remains28 and required original anchors24, reported separately. A model
cannot recover the baseline miss by evaluating only available measurements.
The seven nuisance windows are purposively selected assistant-reviewed controls,
not authoritative airborne negatives or representative whole-camera exposure.

## Diagnostic decision

Report per-encounter and per-control feature distributions, not an optimized ROC
or selected operating threshold. Evaluate only the natural fixed ablation
`informative and point_minus_edge_fraction > 0` against every baseline reference
sample, retaining any alternative that passes the same existing match gate.
Report ambiguity/identity effects separately; predictions cannot rescue a miss.
No measurement gate is promoted in this diagnostic regardless of its outcome.
Preserve failed results. Native point morphology alone may reject blurred,
intermittent, faint, extended, or edge-adjacent genuine targets. A gap on these
few development encounters would still need new source review, complete output
replay, temporal verification, and an independently evaluated operating point.

Tests cover model algebra, symmetry, input validation and invariance before
source extraction. Freeze implementation, tests, plan, inputs and all selected
observation keys before computing real-source features. Rehash after analysis.
