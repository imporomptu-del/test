# AOT residual patterns: a spatial-motion gap, not just weak-feature noise

2026-09-28 UTC (September 27 local). Local analysis completed for all eight
actual adjacent pairs and all 56 natural-texture synthetic controls using the
frozen `half_gain16_complete` capture. No imagery or labels were opened, no Jetson
job was run, no alternative model was fitted, and production remains unchanged.

## Decision

**The next useful experiment is a bounded comparison of camera-motion models,
not a stronger Harris cutoff or relaxed acceptance gates.** Nearby accepted
features agree closely, while their motion varies with position across the
image. This supports investigating whether the single image-wide translation
is too restrictive for this moving-camera sequence. It does not prove that all
matches are correct, identify the physical cause, or prove that affine correction
will solve the problem.

There is a second gap: the last two actual pairs have median accepted
displacements of **26.24 and 35.96 native pixels**, whereas the previous synthetic
controls tested only seven discrete shifts up to **4.47 pixels**. Those controls
did not validate the larger-motion regime.

This is still a diagnostic, **not restored airplane detection**. All eight saved
actual translation fits remain rejected. This moving-airborne-camera sequence
also does not establish performance for the intended mostly stationary camera.

## What the local patterns show

For each accepted feature, subtract the saved candidate translation from its
displacement. Then compare that residual with the componentwise median residual
of its five nearest accepted neighbors within 200 native pixels. Use every
anchor with all five neighbors, with equal anchor weight. The reference jointly
shuffles entire residual vectors over the same fixed graph 199 times.

Across all eight actual pairs, observed local disagreement is **0.071–0.264 of
the median shuffled reference**. For the 48 nonzero synthetic shifts, the range
is **0.615–0.937**. The eight exact static cases have zero observed and reference
disagreement, so their ratios are undefined, not zero evidence of structure.

The actual graphs include **98.34–99.77%** of accepted points. Their median
neighbor distances are 32.00–34.23 px (synthetic: 30.46–34.41 px). The actual
6×8 maps have at least five accepted points in 45–47 of 48 cells. This is not a
pattern inferred from a handful of isolated features, although support and
accepted cohorts differ between cases.

![Actual and synthetic local-agreement comparison](/Users/romanmaksymiuk/Documents/SEAQR/outputs/seaqr_aot_pilot_20260927/residual_patterns_01/figures/coherence_comparison.png)

In the maps, residual direction and magnitude change with image position,
especially toward lower image rows in several pairs. These are coordinate-field
patterns; no image-depth measurements or fresh scene interpretation were made.
Spatially coherent wrong matches, perspective/parallax, moving scene objects,
and location-dependent precision remain possible contributors.

### All actual pairs, without selection

All distances below are native pixels. Residual p50/p90 use **all accepted
matches**, not just the original RANSAC inliers. They are disagreement with a
rejected candidate translation, not ground-truth tracking error.

| Input pair | Accepted / selected | Displacement p50 / p90 | Above 4.47 px | Residual p50 / p90 | Local / shuffled | Anchors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 000→001 | 901 / 989 | 2.70 / 4.83 | 12.5% | 1.47 / 4.83 | 0.216 | 891 |
| 042→043 | 928 / 981 | 4.53 / 7.74 | 51.5% | 1.77 / 6.08 | 0.098 | 922 |
| 085→086 | 866 / 989 | 2.24 / 7.30 | 21.8% | 1.26 / 6.27 | 0.264 | 857 |
| 127→128 | 903 / 983 | 4.83 / 6.97 | 58.8% | 1.68 / 7.61 | 0.140 | 897 |
| 170→171 | 855 / 953 | 3.87 / 9.71 | 30.4% | 0.96 / 6.53 | 0.201 | 843 |
| 212→213 | 784 / 1000 | 6.91 / 11.84 | 71.8% | 7.72 / 15.07 | 0.109 | 771 |
| 255→256 | 875 / 987 | 26.24 / 29.67 | 100.0% | 3.47 / 8.95 | 0.071 | 867 |
| 298→299 | 858 / 950 | 35.96 / 42.87 | 100.0% | 2.82 / 10.08 | 0.085 | 856 |

The saved fits all fail the original minimum inlier ratio and median/p90
inlier-residual gates; 212→213 also fails spatial coverage. The lower
inlier-only errors are threshold-defined, not an independent accuracy check.
The 120 px displacement, status, bounds and forward/backward acceptance filters
already censored this population. These values are not the complete true
camera-motion range or object velocities.

[All eight full-extent maps](/Users/romanmaksymiuk/Documents/SEAQR/outputs/seaqr_aot_pilot_20260927/residual_patterns_01/figures/all_eight_residual_maps.png)
use the same arrow scale: 2.36 displayed pixels per native residual pixel in
each full-size individual panel. Source coordinates remain 2448×2048, y-down.
Gray dots are accepted scene-feature locations; blue arrows are cell median
residuals, **not detections or airplane tracks**. Cells with fewer than five
points are explicitly unavailable. The combined overview scales every panel
uniformly for viewing.

## Stronger features are not a sufficient fix

Score bins use exact raw U32 Harris scores and shared pre-flow quartile cuts for
each source image. Ties stay together, so bins are not equally populated. All
selected features, including losses, remain in the primary denominators.

| Low→high score bin | Actual selected | Actual accepted | Actual acceptance | Synthetic interior accepted / selected | Synthetic interior within 0.5 px / selected |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q1 | 1582 | 1343 | 84.9% | 5546 / 5856 | 93.25% |
| Q2 | 2277 | 2075 | 91.1% | 10746 / 11413 | 94.09% |
| Q3 | 1998 | 1824 | 91.3% | 10091 / 10794 | 93.45% |
| Q4 | 1975 | 1728 | 87.5% | 9928 / 10789 | 91.95% |

Synthetic columns include only the 48 nonzero shifts and use the previously
frozen 128 px source/expected-destination interior cohort. They are repeated
feature observations from eight source images, not independent samples.

Among accepted synthetic interior tracks, higher-score bins have smaller errors:
per-case median true error ranges from **0.158–0.235 px in Q1** versus
**0.051–0.105 px in Q4**. But Q4 has lower fixed-cohort survival, and its successful
tracks within 0.5 px per original selected interior point are not more numerous
proportionally. Precision conditioned on survival and retention are different.

Actual candidate residuals often decrease with score, but not consistently:
for 255→256, Q4 median residual is **4.64 px**, worse than Q1's **3.37 px**;
for 212→213, even Q4 remains at **6.37 px**. The bins also cover different
locations: Q1 occupies 13–21 cells per actual source, versus Q4's 22–28.
Score and scene location are confounded. Raising the score cutoff could discard
useful coverage without fixing the motion model. No cutoff was changed.

## Recommended next step — proposal, not executed

1. Freeze a small offline comparison of translation, similarity (rotation and
   scale), and affine motion on the same accepted point sets. Use predefined
   spatial blocks held out from fitting; never use the old whole-frame fitted
   inlier mask to choose training or evaluation points. Predeclare support,
   weights, seeds and reporting; retain all eight pairs and all failures.
2. Judge errors on the held-out blocks, coverage and worst regions, not only
   training inliers or pooled averages. This tests prediction of saved accepted
   correspondences, not physical ground truth. If no single model adequately
   predicts the field, report that rather than automatically escalating to
   homography/local warps or relaxing quality gates.
3. Before treating a candidate correction as validated, expand known-transform
   controls to bracket the observed 26–36 px median regime, including suitable
   near-zero controls. Freeze exact cases first; preserve loss denominators and
   border support. The existing ≤4.47 px controls cannot cover this gap.
4. Only after model and tracking checks pass, make an isolated implementation
   candidate and repeat the frozen detector evaluation, including its known
   airplane and false-alert checks. Preserve the stationary-camera path and
   confidence rejection. Better registration alone is not better detection.

No claim is made that similarity/affine will pass, that this is a PVA hardware
fault, or that the current camera-motion quality thresholds should be loosened.

## Verification and artifacts

- Plan, runner and tests were frozen before new case metrics; all 64 cases
  completed in fixed order and all source/audit/code/manifest hashes matched
  before and after. Shared source/selection/exact-score identities were checked
  across all eight source groups (one actual plus seven synthetic cases each).
- **247 tests passed**: 28 new generated analysis checks plus 219 prior AOT/motion
  regression checks. Null translations, tied scores, all-selected denominators,
  accepted-list inlier indexing, fixed graph boundaries and no-overwrite tested.
- Independent reviewer checked predeclared ordinals 0,7,8,9,63 without importing
  analysis helpers: source-array identities, quantiles, inlier/cohort accounting,
  240 grid cells, 20 score bins and all 995 permutation values passed. This is a
  five-case spot-check, not a second complete 64-case recomputation.
- All eight rendered panels and the comparison chart were inspected. No source
  media was opened. No SSH, new detector run, labels, RAW16 or holdouts accessed.
- Earlier factor failed-flow nonfinite byte payloads remain unrecoverable from
  readable JSON. This analysis uses finite accepted pairs and retains selected
  loss denominators; it does not infer invalid raw-flow values. The prior
  natural-shift exact-byte audit remains separate evidence.

Full point/graph/reference details:
[result.json](/Users/romanmaksymiuk/Documents/SEAQR/outputs/seaqr_aot_pilot_20260927/residual_patterns_01/result.json).
Compact evidence:
[summary.json](/Users/romanmaksymiuk/Documents/SEAQR/skymove/results/tiny_target/aot_pilot_20260927/residual_patterns_01/summary.json),
[independent spot-check](/Users/romanmaksymiuk/Documents/SEAQR/skymove/results/tiny_target/aot_pilot_20260927/residual_patterns_01/audit.json).

Result SHA-256:
`a18cb9614b02847e5a9e34a16e5f3fab3ec1963541cb2684444e28134c0d85fc`.
Plan SHA-256:
`54f2eafc698362fb79c56e0b6aacf6a012fffb2ec1e6b051d82150693414ef38`.
Source code and tests are preserved under the output's `provenance/` directory.
