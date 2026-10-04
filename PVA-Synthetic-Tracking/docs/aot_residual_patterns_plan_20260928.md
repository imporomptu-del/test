# AOT residual-pattern diagnosis — frozen local analysis plan

2026-09-28 UTC. The user approved analysis of existing motion captures to help
distinguish unreliable matches from an inadequate single-translation model.
Freeze this plan and analysis code/tests before computing the new case metrics.
This is a descriptive development diagnostic, not a new detector experiment.

## Data and scope

Read only the two completed local numerical captures and their audit receipts:

- `outputs/seaqr_aot_pilot_20260927/feature_factors_01/result.json`, SHA-256
  `aaeb0cb9898199548afe9c8303d27717c1694c596dec2b775df6355d24584f56`.
- `outputs/seaqr_aot_pilot_20260927/natural_shifts_01/result.json`, SHA-256
  `56b9f1a363b449706c38aee4453cae1e4eb9c0316576203fbf8f2a03721bf54f`.

Use only `half_gain16_complete`. The primary actual data are the eight adjacent
pairs with previous input indices 0,42,85,127,170,212,255,298. Include all eight,
including rejected global fits. The comparison consists of the 56 natural-texture
synthetic pairs: seven fixed shifts for each of those eight images, in their
original order. Exclude other arms and generated block controls from this new
analysis. Total: **64 case summaries**, actual eight first, then synthetic 56.

No media/image/label access, Jetson work, new PVA calls, detector run, alternative
motion-model fit, threshold change, production configuration change, RAW16,
private camera clip, sealed holdout or old-artifact overwrite. The preceding
source-pixel and hardware checks remain documented receipts, not new executions.

## Reconstruct scope and denominators

Verify both input hashes and audit identities. Recover native selected previous
points and exact U32 Harris scores via the captured selected indices. Match the
stored accepted previous/current pairs back to that selected list, uniquely and
in order. Accepted points must be finite. Check exact previous-image/selection/
score identities across the actual and synthetic rows from each source image.

Use stored accepted point pairs for spatial analysis. Old factor failed-flow
JSON contains null nonfinite values and cannot reconstruct their original bytes;
do not invent those values or analyze them as valid motion. Loss counts retain
all selected points. The accepted displacement distribution is censored by the
original status/bounds/120px-displacement/forward-backward filters, not a measure
of the complete true camera-motion range.

For every case retain the original candidate translation, fit status/reasons,
metrics and inlier indices. Null translation means residual unavailable: do not
substitute another fit. Original inlier indices address the accepted-pair list,
not the selected-feature list. Inlier/outlier differences are partly defined by
the original RANSAC threshold and are not an independent accuracy validation.

## Fixed descriptive measurements

1. Displacement `d = q - p` and candidate-translation residual `r = d - t`:
   count, median x/y, norm p10/p50/p90/p95/p99/maximum, for all accepted points.
   Also retain original inlier/outlier residual summaries. Show the fraction of
   accepted displacements exceeding sqrt(20) pixels, the largest prior synthetic
   shift magnitude; the seven discrete shifts did not validate that entire range.
   Residuals to a rejected candidate are not ground-truth errors.
2. Fixed native 6-row × 8-column grid covering 2448×2048. Report every cell's
   accepted-point count. With at least five points, report componentwise median
   residual vector and median distance from that vector. Empty/under-supported
   cells are unavailable, never zero motion. Retain the full image extent.
3. Neighborhood graph on accepted previous native points: five nearest distinct
   neighbors within 200px inclusive, excluding self, deterministic distance ties
   broken by accepted-list index. A point is an eligible anchor only if all five
   neighbors are available. Each anchor has equal weight. Report eligible and
   excluded counts, neighbor-distance statistics and directed-edge count.
   Statistic: median across anchors of the norm of each residual minus the
   componentwise median of its five neighbors' residuals.
4. Keep the graph fixed and jointly permute whole `(rx,ry)` vectors across all
   accepted positions 199 times. Use seed `20260928 + case_ordinal` (zero-based
   order defined above). Report the reference distribution's p05/median/p95,
   observed/reference-median ratio and raw reference values. With insufficient
   anchors or zero reference median, report the unavailable quantity as null.
   These are descriptive random-label references, **not p-values, confidence
   intervals or proof of a physical motion model**.
5. Exact selected-Harris-score quartiles are defined before flow from all
   selected features, once per source image: linear-interpolated 25/50/75%
   quantiles, bin assignment via right-side search. Equal scores remain together;
   ties can produce unequal or empty bins. Verify shared cuts across all seven
   synthetic rows and the actual pair. For each bin retain selected/accepted/
   rejected counts, acceptance fraction and selected/accepted grid coverage.
   Candidate residual summaries among accepted points remain non-truth metrics.
   For synthetic rows only, additionally report known-shift error among accepted
   points in the prior fixed 128px source/expected-destination interior cohort,
   with the full pre-flow cohort denominator. Actual pairs have no known target
   destination or synthetic truth: do not invent such an overlap/error metric.

Neighborhood residual differences cancel the translation: this measures local
displacement coherence, not independent evidence that the saved translation is
incorrect. Spatially coherent wrong matches, parallax, moving objects and
position-dependent precision can all produce patterns. Feature score also
correlates with texture/location; score-error associations do not authorize
discarding a low-score region or changing the feature threshold.
Accepted cohorts and their neighborhood graphs can differ between actual and
synthetic cases. Compare anchor support and neighbor distances alongside ratios;
cross-case coherence differences are descriptive, not matched-location causal
effects, even when pre-flow selected features are shared.

## Outputs and interpretation

Write exclusive new results under `outputs/seaqr_aot_pilot_20260927/residual_patterns_01/`.
Pin this plan, script, tests, fixed design and source/audit identities in a
manifest before analysis. Preserve inputs and verify their hashes afterward.
Store all 64 summaries and enough per-point/grid/graph detail to reproduce the
statistics. Create fixed eight-pair residual-vector maps and comparison plots
from those saved summaries, labeling support, coordinate units and arrow scales.
Plots are scene-feature diagnostics, not target-detector overlays. Any automatic
visual scale is a display choice only, disclosed and shared across actual panels.

Use small generated tests for geometry, fixed graph/ties/radius, constant and
spatial-ramp residuals, joint deterministic permutations, undefined cases,
score ties, acceptance/inlier indexing and denominator preservation. Independently
spot-check the computed statistics and inspect the rendered figures.

Report per-pair results and relevant counterexamples, not only pooled averages.
The eight pairs come from one development sequence; synthetic rows reuse their
textures and are not independent recordings. Recommend the next bounded test
only to the extent justified by these patterns. No new motion model, detector
accuracy claim, real-time claim or automatic production promotion in this step.
