# V38 global-camera multi-lag diagnostic: pre-extraction plan

## Scope and acceptance question

Keep V34 output behavior and frozen V36/V37 experiments unchanged. Build and
test a source-evidence diagnostic, not another hard rejection filter. The first
question is whether evidence can be evaluated without systematically excluding
intermittent targets or textureless sky. Improved numerical availability is not
improved detection accuracy. No thresholds, production changes, speed claims,
Jetson operations, RAW16 source access or sealed-holdout access are authorized
by this experiment.

Use the exact V37 selection of 358 current observations, with separate dense
285, strict pilot 28, original anchor 24 and provisional-control 70 denominators.
These sets overlap. V37's availability of 3/285 is a diagnostic comparison, not
a detector recall baseline. Include separately versioned, assistant-reviewed
class-unknown compact-light reference frames from the existing 0029 frames
12–24 source crop only after independently reviewing their source positions.
Never copy tracker centers into truth, label unknowns as negatives, silently
change old references, or claim this detector-selected development case is an
independent untouched test. Preserve review-selection exposure and uncertainty.

## Causal source pairs

Fixed source lags are **1,2,4,8 frames**, nominally 0.1,0.2,0.4,0.8 seconds in
these 10 Hz containers. Decode sequentially; retain at most nine owned grayscale
source frames. Enforce exact frame/timestamp chronology. For each selected
current actual measurement, retain every lag record, even if missing/invalid.
No same-ID past measurement is required to obtain source pixels. If present,
an actual prior measurement is optional nuisance-point information, never an
interpolated or predicted location.

Use W = inverse(H_prior) @ H_current, where H maps source to the shared
reference. Source sampling is deterministic float64 bilinear interpolation,
not V37's quantized OpenCV warp. At the nearest integer current actual center,
save native current25 and a globally warped prior27 grid. NaN denotes invalid
border/projective support; exact zero-weight corners do not invalidate pixels.
Reset/segment crossings and invalid transforms yield unknown geometry. Keep
all intervening camera-quality metadata (accepted/reused/error/inlier/support),
but do not reinterpret matrix existence as calibrated camera confidence.

## Fixed evidence and uncertainty probes

Use all nine integer translations (dx,dy) in {-1,0,1} squared on the prior27
grid, each yielding a prior25 patch sampled at current-grid q+shift. This fixed
one-pixel envelope is a **sensitivity stress test**, not a calibrated bound on
camera error. No alignment maximizes the current target's score. No local
registration, local-variance threshold, or target-informed shift choice is used.

For each supported probe, D=current25-prior25. Keep the frozen V36 quadratic
background/point/edge diagnostic, centering the Gaussian bank at the actual
fractional current coordinate. Point sigmas remain 1,2,3, offsets -1,0,1 per
axis. Straight-edge widths remain 1,2,4 with eight orientations and offsets
-2,0,2. The fitted point amplitude magnitude is nonnegative; also report its
polarity-signed form. Retain edge and conditional point-after-edge diagnostics
without treating positive nested gain as a classifier.

Record the current-source spatial diagnostic separately. At zero displacement,
a persistent point can cancel from D even while remaining visible in the
current source; other shifts can manufacture a dipole. Thus zero difference is
explicitly uninformative, never object absence. Lags and probes are correlated,
not independent votes. Do not choose the best lag or best shift.

When an actual prior point exists, optionally run the unchanged V37 joint
prior/current model at mapped prior_xy-shift. Retain all of its unavailability
reasons, including slow displacement or prior center outside support. This
secondary model is not a prerequisite for reporting current source contrast.
When no actual prior point exists, explicitly flag possible negative-ghost
contamination; do not assume absence or invent a trajectory.

No photometric gain is fitted: prior/current DN are used at unit gain. Smooth
additive background changes are handled by quadratic projection; multiplicative
exposure changes on textured structure are **not** removed in general. Report
clipping counts, interpolation provenance and this limitation. This diagnostic
cannot distinguish an isolated compact cloud response from an airborne object
on appearance or temporal persistence alone.

## Availability and aggregation

Report separately: geometry availability, finite source support, informative
nominal contrast, informative all-nine-probe envelope, and optional prior/current
joint-model availability. Require **all nine probes both supported and
informative** before emitting an envelope. Otherwise envelope=null, not a
zero-filled minimum. Supported but uninformative nominal evidence stays explicit.

For complete envelopes, report min/median/max/span of point/edge gains, their
difference, point amplitude (magnitude and signed) and residual RMS. Keep every
probe and every lag; do not collapse unknowns into an acceptance decision.
Report nominal and complete-envelope availability per lag and all four lags,
separately for encounters and provisional controls. Any extra compact-light
regression remains separate from the original reference denominators.

## Freeze, checks and decision

Before source extraction: validate original audits/selection/source hashes;
finalize and review the separate reference; run synthetic/unit tests; snapshot
all code, tests, plan and input hashes in a fresh exclusive output directory.
Rehash at completion. Crosscheck every original current25 with frozen V36 native
patch bytes. Save raw sampled arrays and full chronology so numerical results
can be replayed without new media access.

Test fractional/projective mapping and shift signs; borders/resets/gaps;
missing association; brightness/polarity/direction changes; stationary/slow
and intermittent points; paired points; point on edge; deformed cloud-like
background; quantization/interpolation; additive/multiplicative exposure; and
all-nine-probe unknown propagation. Unit tests establish mechanics, not camera
sensitivity. Independently verify saved arrays, selection, geometry, model
calculations and summary. State whether source decoding itself was repeated.

Accept the diagnostic as a basis for a later policy only if its evidence is
available and stable enough across the known development cases. If nuisance
and point features overlap, report that rather than selecting thresholds from
these examples. No cleaner-overlay, population accuracy or airborne-class claim
is justified by this feature-only run. A downstream verifier cannot recover
the original missed detector birth/sample. Any actual eligibility policy is a
new separately frozen experiment with strict no-new-known-loss checks.
