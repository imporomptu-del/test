# Frozen support-aware regional-motion diagnostic

2026-09-28. Approved follow-up to the independent image-patch check. Written
before any new regional fits or outcome inspection. Development experiment only:
no production change, image warp, detector execution, remote access, new media,
annotation use, RAW16 or sealed-holdout access. Efficiency remains paused.

## Inputs and question

Use the eight actual pairs with previous input indices 0,42,85,127,170,212,255,298
from `residual_patterns_01/result.json`, SHA-256
`a18cb9614b02847e5a9e34a16e5f3fab3ec1963541cb2684444e28134c0d85fc`.
Use frozen image evidence from `image_patches_01/result.json`, SHA-256
`6fd7beb2e83ae8915e2ec9f5d96ae9f7fa956df66a3cee4e5a2c6146f0a82f3c`, and its
selection SHA-256 `874313f96e3eb2d537326eda46e5b52deb1458eb892474637cc4c3d9c926b6a6`.
Do not decode any images. Reuse numerical helper `compare_aot_motion_models.py`
unchanged, SHA-256 `b62b4b47dd4a767be8e3dd7d3157be01418870d3e993b54e116cf06500dfc4a9`.

Question: does a local, support-gated motion model better predict unseen image
regions, without silently extrapolating or absorbing independent small-target
motion? Known previous outcomes motivated this design; this is not a fresh
independent validation set or production-confidence calibration.

## Fixed geometry and three arms

Native image 2448×2048. Use all 48 cells of the existing 6×8 grid, row-major,
as held-out folds. Bounds are half-open. For each fold exclude its entire cell
expanded by 64px in all directions (clipped to image bounds) from all training.
Every accepted previous point is tested exactly once in its own cell. Points in
another fold's guard retain their own eventual test observation.

1. **Global translation reference:** all original accepted LK observations outside
   the cell+guard; at least100 points and12 occupied native grid cells, then the
   unchanged robust translation estimator. This is a new cell-held-out fit of
   the frozen estimator, not the old production RANSAC candidate/quadrant result.
2. **Local translation:** the same guard exclusion, additionally previous-point
   Euclidean distance ≤512px from the fixed held-out cell center.
3. **Local affine displacement:** exactly the same pre-fit local training set,
   but a six-parameter affine displacement field.

All training uses original accepted LK points, never original global inliers,
feature-score filters, NCC outcomes or hand-selected patches. These training
correspondences are not independently validated physical truth. Preserve the
selected/accepted/lost counts from the earlier capture; lost features are not
recovered by this study.

Use existing coordinate normalization z=(p-[1224,1024])/2448 and displacement
q-p. Within each arm's training set, occupied cells have equal total base weight:
each point's base weight is inverse count in its native cell. Initial weighted
least squares followed by exactly20 joint-Euclidean Huber iterations, delta1px.
No early stopping, RANSAC sweep, fit-dependent point removal or model fallback.
Save parameters/native matrices, rank, final weighted-design condition number,
actual final-solve weights and final-residual weights separately.

## Local training-consistency and support gates

Both local arms require ≥24 original training points in ≥4 occupied cells.
Fit every one of those points; never refit a coherent subset. Require full rank,
finite parameters and final weighted-design condition number ≤1000.

Define coherent training observations after that fixed fit as residual norm
≤2 native pixels. Require ≥12 coherent points in ≥4 cells, representing at
least60% of the ORIGINAL cell-balanced base-weight mass. This is descriptive
training consistency, not calibrated prediction uncertainty or correctness.

Construct a two-dimensional convex hull of coherent TRAINING previous positions,
using a deterministic monotone chain. Collinear/degenerate hulls are unavailable.
A query is eligible only inside or on that hull; cross-product boundary
tolerance1e-8 native-pixel-squared is numerical only. Do not add test positions
to the hull, inflate it, or fall back outside it. Pointwise support may differ
within a held-out cell. These gates use training q but never held-out/guard q,
NCC endpoint, target labels or test error.

The global reference remains an ungated numerical reference after its original
count/cell/rank/finite checks: do NOT apply the local2px/60% gate to remove
difficult global predictions from comparisons. Clearly distinguish numerical
global availability from local support eligibility. No quality status is
promoted to the production motion gate.

## Evaluation and retained denominators

Save each fold's train/test/guard indices, local radius support, all gate reasons,
coherent indices/hulls, and predictions or explicit unavailability for every
original accepted test point. Retain all384 cell-pair records, including empty
test cells. Never fill unavailable predictions with zero displacement/error.

Report separately per pair and model:

- All accepted LK test observations: available/unavailable counts and conditional
  displacement errors to saved LK (not physical ground truth).
- All758 frozen patch-sample observations: support and abstentions, with all586
  previously inconclusive observations retained as unresolved image evidence.
- The independent reference cohort defined ONLY by both old NCC scales qualifying
  and agreeing with each other within1.5px, NOT by their agreement with LK. It
  currently contains172 points. Score against33px and65px offsets separately;
  never average them or choose the more favorable reference.

For each fixed comparison (global/local-translation, global/local-affine,
local-translation/local-affine), report errors only on identical queries eligible
for both, plus the full cohort denominator and missing count. Also report the
all-three common set. Include median, p90, maximum, paired difference and counts;
empty sets remain unavailable. Coverage and errors must be read together.
Keep all eight pairs separate; any aggregate remains a selected, correlated
development workload, not an independent-scene or target accuracy statistic.
No automatic winner selection, threshold tuning or production promotion.

## Generated vector-level target-preservation checks

These are synthetic CORRESPONDENCES, not image-level targets or detector tests.
Use the fixed native lattice p=(24+48i,24+48j) strictly inside the image. Backgrounds:
(a) translation d=(3,-2); (b) affine
d=(3+0.008(x-1224)+0.004(y-1024), -2-0.003(x-1224)+0.006(y-1024)).
Holdout anchor cell row3,column3 (center1071,1194+2/3).

For each background, use three compact query-cluster placements: one point at
the anchor center;3×3 unit-spaced points centered there;3×3 unit-spaced points
centered on its right boundary x1224 at the same y. Apply independent target
offsets (0,0), (1,0), (4,-2), giving18 fixed cases. Route each query by its own
cell and identical guard, fitting no query/cluster points inside that guard.
Compare nonzero cases to the matched zero-offset case with identical p. Every
query's fit, support and prediction must be unchanged when its excluded target
q changes. On supported queries, compensated residual must equal the injected
vector PLUS background prediction error; a nonzero target residual is desirable.
Correctly specified clean local models should recover the analytic background
to1e-8px. A missing prediction is unavailable, never a target-preservation pass.

Separately stress contaminated training: for both backgrounds, insert1-point
and3×3 unit-spaced nuisance clusters at(1377,1194+2/3), in the neighboring cell
and within the512px training radius. Compare clean cluster motion to additional
(20,-10)px motion with a query at the anchor center carrying(4,-2)px target
motion:8 fixed cases. Report eligibility changes, background-prediction drift
and target-residual change, without choosing a new threshold from results.
These contaminants deliberately lie OUTSIDE the query's guard. Do not conflate
their robustness with guaranteed exclusion of a target in its own query guard.

Generated unit tests additionally cover empty/sparse support, fewer than4 cells,
one-sided hulls, collinearity, condition/gate boundaries, deterministic hulls,
half-open cell/radius/guard bounds, no held-out/guard q or NCC leakage, repeated
roles/independent joins, unchanged inputs and explicit abstention accounting.

These controls do not show preservation of image contrast, sampling/warp
artifacts, target identity, airborne classification or detection recall. Image
and temporal target-preservation tests are still required before integration.

## Freeze, execution, and audit

New runner `scripts/compare_aot_regional_motion.py`, generated tests
`tests/test_aot_regional_motion.py`, this plan, helper and input hashes must be
bound before any new real fits. Use only fresh
`outputs/seaqr_aot_pilot_20260927/regional_motion_01/` artifacts; fail rather than
overwrite. Check input/code/manifest identities before and after. Retain raw
results even if local support or error performance is poor. Independently
reconstruct bounded folds and comparison counts, run regressions, and deliver
a readable report with compact copied evidence in repository results.

The independent refit audit is fixed before outcomes: previous indices0 and298,
cells0,27,47, all three arms (up to18 fits, with any support failures retained).
Independently reconstruct all384 fold partitions and all patch-cohort joins and
common-support counts. Audit tolerances: native predictions/parameters and
residual summaries within1e-8px where numerical equivalence applies; exact
indices, gate outcomes and denominator counts. No outcome-selected audit cases.

No predeclared result promises improvement. If useful predictions cover too
little of the image, or common-support errors/contamination behavior worsen,
say so and leave production unchanged. A successful diagnostic still requires
separate image/temporal tests and an isolated detector evaluation before shipping.
