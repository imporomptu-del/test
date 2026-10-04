# V52 — robust current-guard fitting, evaluated on disjoint pixels

## Scope

The user approved the V51 next step. Implement a shadow experiment, not a
detector filter: fit a gain-plus-plane correction on one part of the current
outer guard and evaluate it on the complementary part. Production, original
source decisions, references, thresholds and prior artifacts remain unchanged.
Efficiency and RAW16 remain paused. No media, image-cache NPZ, sealed holdout,
journal, SSH or Jetson access. Synthetic arrays are generated locally.

This is explicitly **current-guard estimation**, not prior-only forecasting.
Disjoint guard pixels are not independent: neighboring samples are correlated,
and tracker-selected patches are neither pure background nor airborne truth.

## Fixed model and numerical handling

The model is C = g*S + b0 + bx*(x-64)/56 + by*(y-64)/56, g>=0, with S the frozen
Median8 prior prediction. Compare constrained ordinary least squares (OLS) with
Huber loss, delta=2 DN. This breakpoint is an engineering experiment setting,
not an estimated physical noise bound, confidence threshold or detection rule.
Retain uncorrected Median8 and Median3 as controls. No model selection.

Condition the S column using only finite training data: subtract its median
and divide by max(1 DN, its median absolute deviation). Fit four coefficients.
Require at least four finite training rows, full column rank at relative
tolerance1e-12, and design condition number<=1e8. Flat/planar backgrounds can
make gain non-identifiable; such a fold is unavailable, not repaired with a
ridge, forced gain, lower-rank substitute or another arm's answer.

Each OLS/weighted least-squares step solves the nonnegative-gain constraint:
use the unconstrained solution if feasible, otherwise solve the plane with
gain exactly zero. Huber uses IRLS with at most100 updates and strictly positive
weights. Convergence requires a KKT projected-gradient residual<=1e-7 DN after
normalizing each averaged gradient component by max(1,RMS of its conditioned
design column). Coefficient-step size alone is not convergence. Check weighted
rank/conditioning, nonfinite arithmetic, bound sign and numerical failure;
report the reason and retain unknown predictions. Do not tune the cap or
tolerances after inspecting real outcomes.

Prediction receives only the frozen fit, held-out coordinates and held-out
prior S, never its current response. Return point-preserving masks/reasons and
fingerprints. Fitting uses finite training samples only; no current imputation.

## Frozen geometry splits

Support is the inherited prior-selected grid: integer coordinates8..120,
step8, Chebyshev radius40..56 about(64,64). No new response-based point selection.
Reject duplicates, off-grid points and source-core coordinates.

Primary split: left/right, x<64 versus x>=64. Sensitivity split: checkerboard,
(x//8+y//8)%2. In each split, fit on fold1 and predict fold0, then fit on fold0
and predict fold1. Fit keys name the held-out fold. Preserve both directions;
do not choose the favorable split, direction or loss after scoring.

All scaling, training weights, numerical acceptance and fitted coefficients
must be independent of that fold's held-out responses. Responses legitimately
serve as training values in the reverse direction. Thus independence tests
must compare the affected held-out fold, not assert that the whole two-fold
result is unchanged. Source-core data are not accepted by the guard-only API;
no source-to-guard transfer or full-camera noninterference claim follows.

## Generated validation before real fitting

Freeze120 snapshots:20 families crossed with three backgrounds and two noise
profiles. Grid support is144 points. Backgrounds are constant96;
planar96+.05*(x-64)+.03*(y-64); and that plane plus
12*sin((x-64)/12)+9*cos((y-64)/15)+6*sin((x+y-128)/17).
Noise profiles are zero(seed71) and uniform[-1,1](seed991), identical deterministic
9x144 arrays shared across cases. Add noise after analytic signal construction,
then apply missing masks. These are not independent trials or sensor models.

Families: stable; uniform offsets+8/-8; gains1.25/.75; zero gain(current96);
negative-gain challenge(current192-base); plane4*(x-64)/56-3*(y-64)/56;
gain1.15+offset8+that plane; recent step(last two priors+8,current+8);
ended two-prior pulse(last two priors+8,current unchanged); ended long pulse
(all eight priors+8,current unchanged); localized upper-coordinate-quarter
change(x>=64,y>=64)+16; sparse bright/dark changes(index%17==0)+40/-40;
stripe(abs(x-64)<=8)+32; missing first prior at first eight points;
missing current left; missing current right; all current missing.

Median8/Median3 use ordinary median, not nanmedian. Missing histories remain
unknown. For localized/sparse/stripe changes, retain separately the injected
contamination and clean current-background truth before contamination/noise.
For broad transformations truth is the analytic transformed current background.
Truth and family labels never reach fitting. Report held-out observed-response
error and, separately, synthetic-only clean-background error; fitting away an
injected possible source is not automatically a background improvement.

Constant/planar rank failures are expected demonstrations of model scope, not
cases to prune. Test source-core rejection, held-out-response noninterference,
gain bound, robust outliers, missingness, rank/condition failure, convergence,
mutation, fingerprints and repeatability using generated fixtures.

## Existing real compact evidence only

Pin V50 completion receipt SHA256
9213ed07fa7e8efd81f7c0032290dbe2f8c37bd06f6376b86d0b53a1af7dc2b1.
Read only literal files under its prediction_01 directory: freeze.json,
state_results.jsonl, calibration/evaluation_forecasts.jsonl,
calibration/evaluation_measurements.jsonl, reference_context.json and the receipt.
Validate hashes before JSON decoding; never follow packet/media paths in maps.

Join all493 scored archive records by original state key. Reconstruct current
guard values as Median8 prediction plus its signed residual, cross-checking
Median3 plus its residual. All14 unavailable response packets remain wholly
unavailable: their withheld1,647 values cannot be reconstructed, even though
their metadata records only one missing point per packet. Preserve all62,867
prior-selected point opportunities,61,220 reconstructable currents, all1,211
states,702 history-unknown states,16 unscored embargo states and55 fixed
partition/clip time bins. Preserve original reference context separately; it
must not enter fitting, point selection or arm/split choice.

Old calibration/evaluation names identify exposed development blocks, not fresh
validation data or calibration of the new loss. No parameters are learned
across packets. Existing geometry conditions these estimates and does not prove
a fully causal end-to-end camera pipeline.

## Execution, metrics and decision

Freeze plan, source/tests, constants, generated specifications and literal input
bindings in a fresh output child before actual synthetic/real fits. Save every
cross-fitted prediction, mask, fit diagnostic and fingerprint before evaluation
error scoring. Training necessarily accesses current training pixels first.
Recheck all bindings after scoring; never overwrite an existing valid run.

For each loss/split report all folds and unknown reasons, prediction availability
separately from response availability, conditional point errors, packet MAE,
point/packet tails and complete archived-frame macro errors. Each corrected arm
gets Median8/Median3 controls on **the exact same scored point set**. Also report
both uncorrected controls on all available responses. An incomplete archive or
failed fold makes that arm's complete-frame metric unavailable; it does not
make the frame disappear or count it as successfully suppressed noise.

Retain every clip/partition, fixed nine-frame bin and generated family/profile.
Report comparisons on per-packet shared-complete and complete-frame bases as
well as point-weighted averages. Inspect worst losses as well as averages; do
not allow a pooled mean or a fall in scoreable coverage to establish success.
For each real evaluation clip report whether primary Huber lowers shared
complete-frame MAE, whether its maximum held-out error increases, and how much
coverage it loses. Mixed results or material unavailability mean no uniformly
better replacement; favorable numbers only motivate another guarded experiment.

Independently check numerical optimality/KKT, predictions, folds, metrics,
fingerprints, state/reference accounting and file bindings, plus full regression.
No physical uncertainty interval, airborne accuracy, recovered source or reduced
false-alarm claim. No integration follows automatically: future promotion must
preserve355references/424alternatives/300original assigned states/three misses,
explicitly account for seven old positives and five later losses, and separately
measure reviewed negative-region workload without treating unknowns as negatives.
