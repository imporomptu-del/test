# Frozen AOT spatial-holdout motion-model comparison

2026-09-28 UTC. User-approved diagnostic after the residual-pattern analysis.
Freeze this plan, runner and generated tests before evaluating any real cases.
No production promotion, threshold relaxation or target-detector execution.

## Inputs and scope

Use only the existing local `residual_patterns_01/result.json`, SHA-256
`a18cb9614b02847e5a9e34a16e5f3fab3ec1963541cb2684444e28134c0d85fc`,
under `outputs/seaqr_aot_pilot_20260927/`. Preserve its 64-case order: eight actual
adjacent pairs (previous indices 0,42,85,127,170,212,255,298), then 56 fixed
natural-texture known shifts. No images, annotations, RAW16 or sealed holdouts.

For each model use identical original accepted correspondences. Never select
points using the original global-fit inliers, errors, feature scores or new fit
residuals. Retain all-selected/accepted/lost and fixed synthetic support counts.
Acceptance was already censored by status/bounds/displacement/forward-backward
checks; this study cannot recover lost matches or prove physical motion truth.

## Fixed fitting and spatial separation

Compare exactly three models: translation, similarity (rotation plus isotropic
scale), and affine. Fit displacement q-p, using fixed centered coordinates
z=(p-[1224,1024])/2448 for numerical stability:

- Translation: d=(tx,ty).
- Similarity: d=(tx+a*zx-b*zy, ty+b*zx+a*zy).
- Affine: d=(tx+a*zx+b*zy, ty+c*zx+d*zy).

Save equivalent native previous-to-current matrices, determinants, rank and
condition information. These are diagnostic estimates, not validated camera
poses. Do not replace the saved baseline candidate or change its quality status.

Four contiguous 2×2 native-image quadrants are test blocks, ordered top-left,
top-right, bottom-left, bottom-right. Bounds are half-open, with splits x=1224,
y=1024 and full extent [0,2448)×[0,2048). Each accepted point belongs to exactly
one test fold. For each fold, training excludes the test rectangle expanded
by 64 native pixels on each side and clipped to the image extent. Guard points
are excluded from training, not deleted from their own test fold.

Training requires at least 100 points, at least 12 occupied cells of the fixed
6×8 native grid and full design-matrix column rank. Record failures rather than
falling back to a simpler model. Rank/finite checks are numerical validity, not
motion-quality acceptance. Report condition numbers; no extra condition cutoff.

Within training only, base weight is 1/(training point count in that 6×8 cell),
giving occupied cells equal total base weight. Fit initial weighted least
squares, then exactly 20 joint-Euclidean-residual Huber reweight-and-solve steps,
delta=1 native pixel; Huber factor=min(1,1/residual_norm), with zero mapped to1.
All three models use the same estimator, base weights and folds. No early
stopping, RANSAC, fit-dependent point deletion, robust-scale estimation, repeated
seed search or test-data-dependent initialization. Retain the final solve weights
separately from any weights recomputed from the final residuals.

The recomputed translation comparator uses this same fitting protocol, not the
old RANSAC candidate. Any change due to estimator/weighting therefore remains
distinguishable from adding motion-model degrees of freedom.

## Fixed evaluation and output

Save train/test/guard indices per fold; fits and training weights; held-out
predicted positions and errors for every test point; all support/failure counts.
Test error is ||predicted q - saved observed q||, not real-world truth.

Report fold and concatenated out-of-fold count, p50/p90/p95/p99/max error. Show
the original accepted evaluation denominator and unscored count. A missing fit
or empty fold is unavailable, not zero error. Mark case evaluation incomplete
if any accepted point is unscored; do not present survivor summaries as a full
case or rank incomplete cases against complete ones.

Report each of the 48 cells' out-of-fold support; where at least five test
points are scored, report median and p90 error. Keep lower-support cells
unavailable. Report worst supported-cell median/p90, not just pooled averages.

For the 56 synthetic cases separately report held-out transform prediction
error ||predicted q-(p+known_shift)|| on accepted test points. This evaluates
the estimated mapping, not raw LK point accuracy or successful recovery of lost
tracks. Keep original pre-flow support and lost-track denominators alongside it.
Actual cases have no known transform: never invent a truth error.

All eight actual pairs remain separate. Summarize per-case model differences
descriptively; there is no automatic model selection, new production pass/fail
threshold, population inference, or claim of independent-scene generalization.
These folds separate spatial regions, not recordings; the same one development
sequence and eight reused textures underlie all results.

Freeze a manifest with this plan/script/tests and input hashes. Write exclusively
under `outputs/seaqr_aot_pilot_20260927/motion_models_01/`; keep original artifacts
untouched and check input/code/manifest hashes after execution. Verify generated
translation/similarity/affine recovery, train/test separation and perturbation
invariance, support failures, weights, boundaries and full denominators. Perform
an independent bounded numerical check and save a human-readable local report.

## Larger controls and decision boundary

A separate predeclared Jetson control experiment may extend integer translations
to cover the observed 26–36px regime while retaining near-zero checks, the frozen
feature/LK method, original gates and all losses. Its exact cases and source/code
identities must be frozen before execution, separately from this local fit study.
No inference that affine is correct follows from pure-translation controls.

Do not automatically escalate to projective/local warps, alter the stationary
camera path, or rerun the target detector in this step. If no model adequately
predicts held-out regions, report that limitation and recommend the next bounded
test. Better agreement with saved correspondences alone is not better detection.
