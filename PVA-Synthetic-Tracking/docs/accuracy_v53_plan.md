# V53 — cross-fitted median-offset-only background correction

## Scope and fixed decision

The user approved V52's bounded next experiment. This is a local shadow test,
not a detector change: predict current guard values using Median8 plus a single
training-only median residual. Gain is exactly one and both spatial slopes
are zero. No fitting of gain/slopes, clipping, adaptive history, model selector,
per-clip tuning, or fallback. Efficiency and RAW16 remain paused.

Use the same already-exposed V50 compact JSON evidence and exact V52 generated
120-case benchmark. No video, image-cache NPZ, sealed holdout, journal, SSH or
Jetson access. Preserve all original source/reference decisions and old outputs.
Reusing exposed development blocks is not fresh validation or airborne truth.

## Estimator and missingness, fixed before actual fitting

For finite training pairs S=Median8 and C=current, calculate residuals C-S.
Require at least four finite pairs, retaining V52's conservative minimum rather
than tuning support on these outcomes. Otherwise the entire fit is unavailable.
Finite operands that overflow during subtraction make the fit unavailable;
never silently discard the arithmetic failure or impute missing observations.

Sort finite residuals. The median interval is the central order statistic for
odd counts, or the two central order statistics for even counts. Use its safe
midpoint as the one frozen offset. Retain the interval, finite-row mask,
training hash, mean absolute training error and L1 subgradient interval as
diagnostics. The subgradient interval must contain zero. Nonfinite offset or
diagnostic arithmetic makes the fit unavailable, not a baseline substitution.
Use overflow-safe midpoint handling; add direct extreme-value tests.

Predict only S_test + offset, accepting fitted model, held-out coordinates and
held-out prior S; never accept held-out C. Missing priors and nonfinite addition
remain pointwise unknown. Missing held-out current does not erase a prediction;
it affects scoring only. Preserve read-only arrays, hashes and original indices.

The sole fitted arm is median_offset. This is a current-guard estimate, not a
prior-only forecast. Constant/planar data need no rank test for an offset: their
availability may legitimately differ from V52. Do not compare pooled V52/V53
averages across different supports or attribute that coverage change to accuracy.

## Geometry and unchanged comparisons

Keep x,y on 8..120 step8, Chebyshev radius40..56 about(64,64), rejecting core,
duplicate and off-grid coordinates. Primary left/right split: x<64 vs x>=64.
Sensitivity checkerboard: (x//8+y//8)%2. For each held-out fold, fit only the
opposite fold; retain both directions. Scaling, finite-row masks and acceptance
must not depend on the held-out response. Responses can legitimately train the
reverse direction. Do not claim disjoint pixels are independent samples.

Compare against unchanged Median8 and Median3 on each corrected arm's exact
scored pixel set. Also report each baseline on all available responses. Retain
all prediction/current/scoring masks, conditional point metrics, packet MAEs,
tails and complete archived-frame macro errors. Shared complete-frame MAEs use
equal-weight packet means per frame then equal-weight frames. Maxima use all
matched scorable pixels including partial surviving folds. Incomplete frames
remain in denominators, never successful negatives.

## Inputs and challenge coverage

Reuse frozen accuracy_v52_benchmark.py without editing its formulas/specs:
20 families x3 backgrounds x2 deterministic noise profiles,144 points and eight
history frames. Include stable/offset/gain/plane changes, step/pulse endings,
localized/sparse/stripe contamination and missing prior/current cases. Preserve
strict temporal medians, shared noise realizations and separate synthetic clean
background truth. No labels, truth, contamination masks or Median3 reach fitting.
These are analytic tests, not a physical sensor simulation or independent trials.

Reuse frozen accuracy_v52_real_scope.py's literal JSON-only reader, pinned to
V50 completion receipt SHA256
9213ed07fa7e8efd81f7c0032290dbe2f8c37bd06f6376b86d0b53a1af7dc2b1.
Read only that receipt, freeze, state_results, calibration/evaluation forecasts
and measurements, and reference_context under V50 prediction_01. Never follow
paths in old manifests. Preserve all493 archive records/62867 point opportunities,
61220 reconstructed currents and14 wholly unavailable response packets/1647
unknown values. Retain1211 states including702 history-unknown and16 embargo,
all partitions/clips and55 fixed nine-frame bins per split. Keep355 references,
424 alternatives,300 original assigned states and3 original misses unchanged.

## Execution and verification

Freeze this plan, new model/runner/auditor and their tests, inherited benchmark/
reader/independent-audit helpers and their tests (13 source files), constants,
120 generated specifications and eight literal V50 input hashes in a fresh
output child before actual fitting. Inherited files remain byte-identical to
V52. Save all predictions before any evaluation-error scoring; training has
necessarily already used current training values. Recheck bindings afterward.
Never overwrite an existing run, tune after seeing errors, or silently rerun.

Tests must cover exact median interval/L1 optimality, robust minority outliers,
both splits, core exclusion, held-out-response noninterference, frozen masks,
missingness, minimum support, constant/planar support, extremes/overflow,
fingerprints/mutation, exact membership and frame-denominator accounting.
Use an independent implementation to audit every offset/order-statistic
certificate, prediction/mask, failure, score, aggregate and original context.
The audit may reuse V52's independent literal-input reconstruction and scalar
metric helpers, never its model/runner, and must not mutate its globals.
Run full unit regression before final delivery.

## Interpretation and promotion gate

For each real evaluation clip report whether primary offset lowers shared
complete-frame MAE versus Median8 and Median3, its maximum held-out error,
packet wins/losses/ties and any coverage loss. Retain calibration and sensitivity
results without choosing a favorable block/split. Inspect synthetic clean-truth
contamination outcomes as well as observed residuals. A pooled improvement or
reduced scorable coverage cannot establish success.

This tests whether removing gain/slope estimation avoids unnecessary transfer
error; it does not promise improvement. A mixed result is not a universally
better replacement. Even a consistent guard improvement permits only another
separate source-safety experiment, not integration or a source veto. Before any
future integration, explicitly account for seven old positives/five later
losses, unchanged reference assignments and reviewed negative-region workload.
No physical uncertainty, false-alarm, airborne accuracy or recovered-source claim.
