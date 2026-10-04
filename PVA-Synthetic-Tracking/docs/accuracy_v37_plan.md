# V37: registered temporal source evidence — diagnostic only

## Decision and scope

V36 stays frozen. Its point-versus-straight-edge comparison reduced reviewed
control workload but fails a synthetic point superposed on a stronger edge.
Its 16 surviving control responses also show why positive nested point gain
cannot by itself distinguish points from curved cloud structure. V37 tests
whether independently background-registered adjacent frames supply useful
additional evidence. **This experiment is not an eligibility gate, a default
change, or a claim of airborne detection accuracy.** It cannot rescue rejected
states by merely being ANDed after V36.

Only the local 8-bit development sources 0029 and 0126 are decoded for the
first diagnostic. The permitted later review sources are 0055 and 0082.
RAW16, remote processes, and sealed holdouts remain untouched. No detector,
tracker, learning protection, production setting, or frozen report is changed.

## Freeze before pixels

Use the exact 358 observation keys already frozen in V36 `context_01`:
150 in 0029 and 208 in 0126. Copy that selection verbatim and validate its
parent full-context independent audit, source hashes, frame journals and
reference bindings. The 285 dense samples, 28 overlapping strict pilot
samples, 24 overlapping anchors, and 70 selected provisional control
measurements keep their separate denominators, including unobserved and
unavailable cases. Existing reference coverage is 284/285, 28/28, 24/24;
there is no new retention claim from a diagnostic that changes no outputs.

Before extraction, snapshot this plan, implementation and tests; record input
hashes and explicit parameters. Use a fresh exclusive output directory and
rehash all bindings after extraction. Save native/interpolated arrays and
both-frame provenance. Review code and run synthetic tests before explicitly
authorizing the extraction switch. No parameter is selected from the results.

## Two-frame geometry and independent background registration

Require exactly the previous video frame (100 ms at the recorded 10 Hz),
same reference segment, no current reset, and an actual prior measurement
of the same identity. Never substitute a coast or the last measured frame.
Missing evidence yields an explicit unknown observation, not a negative.

If H maps source to reference, map current source pixels into previous source
pixels by W0 = inverse(H_previous) @ H_current. This aligns background, not
the point. Around the nearest integer current actual measurement, take a
native 49-square current context and sample a 53-square previous context
through W0 using bilinear interpolation; borders are NaN. The target's prior
actual position is mapped into this current grid by inverse(W0).

Search local residual sample shifts in [-2,2] with 0.5-pixel steps. Each
hypothesis fits gain and offset on the **same** Chebyshev annulus radii
14 through 22. Exclude the previous point's radius-nine footprint dilated
by the entire +/-2 shift box, saturated pixels, and invalid interpolation
support across any hypothesis. The central 25-square evaluation region
is never used to estimate background shift or photometry.

Registration availability assumptions, fixed before data, are: at least
128 pixels; both annulus variances at least 0.25 DN squared; gain in [0.5,2];
non-boundary winning shift; MSE below one-quarter of current variance; and
a best alternate at least one pixel away with an MSE gap of at least 1/12
DN squared. These are conservative research availability checks, **not**
calibrated statistical confidence. Ties use MSE, shift norm squared, dx,dy.
Aperture ambiguity, weak texture, bad borders, or bad photometry abstains.

Registration returns a geometrically sampled prior patch. Apply its gain
and offset exactly once before subtraction. A prior point at p appears at
p-shift in the registered current-grid patch; do not move it to the current
target location. Preserve both raw and photometrically adjusted arrays.

## Point-pair model

On the central 25-square pair, D = current - corrected registered previous.
Project out constant, x, y, x squared, xy, and y squared. For each polarity,
fit these three model banks by least squares with the specified constraints:

- Null: negative-polarity previous point only, nonnegative amplitude.
- Point pair: the null term plus a positive-polarity current point with an
  independently nonnegative amplitude. Prior/current brightness may differ.
- Edge alternative: the null term plus an unrestricted signed straight edge.

Prior Gaussian sigmas are 1,2,3 pixels at its actual mapped coordinate.
Current sigmas are 1,2,3 with x/y offsets -1,0,1 around the actual fractional
current coordinate. Edges use tanh widths 1,2,4, orientations k*pi/8 for
k=0..7, and offsets -2,0,2. Background-projected templates are normalized;
solve the constrained two-coefficient systems including boundary solutions.
Record fitted original-template amplitudes, SSE, model parameters and pair
conditioning. Exclude numerically ill-conditioned point/edge pairs above
condition 10,000, rather than returning arbitrarily large cancelling terms.

The model explicitly abstains when the prior center is outside the 25-square
patch, the actual relative point displacement is less than one native pixel,
or the residual is numerical zero. The one-pixel bound is a conservative
quantized-centroid observability assumption, not a minimum object speed.
Subpixel, co-moving, stationary, blinking and missed-association objects are
not negatives. Available fits are still uncalibrated research features.

Compare the two **non-nested** point/edge banks against the common best null
only descriptively; record gains normalized by quadratic residual energy.
Bank-size and constraint differences prevent a probabilistic interpretation.
An added point's nested gain is nonnegative by construction: do not use
gain > 0 as a classifier, evidence of airborne class, or a promotion rule.

## Validation and reporting

Test geometry/shift signs, mask independence, photometric correction,
unequal target amplitudes, bright/dark symmetry, point on a strong persistent
edge, different motion directions, faint/intermittent targets, slow-motion
abstention, degenerate texture, invalid inputs, and direct least-squares
agreement. No constant-direction/constant-speed trajectory is assumed by
the pair fit; successful identity association is nevertheless a prerequisite.

Report all availability/failure reasons and feature distributions separately
for each known encounter and the provisional controls. Unknown fallback is
not a verified detection. No labels are inferred from this experiment. The
known points' physical class is unknown, starts are left-censored, and the
control windows are selected provisional nuisances, not authoritative
negatives. No precision, recall on unseen scenes, false-alarm rate, onset
latency, or generalization claim is justified.

If useful evidence is unavailable on most known points, report that failure
without loosening thresholds on these same samples. An independent audit
must distinguish saved-array numerical recomputation from independent video
decoding. Further multi-frame evidence or a broader background model is a
new, separately frozen experiment, not an in-place revision of these results.
