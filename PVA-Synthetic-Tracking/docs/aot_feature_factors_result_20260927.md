# AOT feature factors: supply restored, real global-motion fits still rejected

2026-09-27 local date. The [predeclared five-arm experiment](aot_feature_factors_plan_20260927.md)
completed **105/105 pair calls** on the Jetson with execution/integrity checks
passing. No production change or detector rerun was performed.

## Outcome

Harris-only gain restored background-feature supply. Native resolution alone
did not. However, **every real adjacent AOT pair still failed the original global
translation quality checks in every arm**. This is a useful diagnosis, not an
end-to-end repair or an accuracy improvement claim.

The experiment separates two stages: enough points can now be extracted and
tracked, but those point displacements do not support an acceptable single
global translation. Inaccurate correspondences, motion-model mismatch, or both
remain possible. This experiment does not distinguish those causes.

## All arms, including failed outcomes

The same eight real pairs, eight repeated-stationary pairs and four positive
generated controls were used in every arm. These are repeated measurements of
the same inputs, not independent validation datasets.

| Arm | Raw corners on real previous images | Real global fits accepted | Stationary truth checks passed | Generated positive truth checks passed |
|---|---:|---:|---:|---:|
| Half resolution, gain 1, legacy capacity | 0–2 | 0/8 | 0/8 | 2/4 |
| Half resolution, gain 1, complete capacity | 0–2 | 0/8 | 0/8 | 2/4 |
| Half resolution, gain 16, complete capacity | 7,051–7,729 | 0/8 | 8/8 | 4/4 |
| Native resolution, gain 1, complete capacity | 0–2 | 0/8 | 0/8 | 2/4 |
| Native resolution, gain 16, complete capacity | 56,964–64,884 | 0/8 | 8/8 | 4/4 |

All five flat-field negative controls produced zero corners. That is a basic
negative control, not a real-world false-alarm test.

The strict known-motion checks required the unchanged global fit to pass,
translation-vector error ≤0.1 native pixel, at least 30 accepted interior points,
interior median error ≤0.1 pixel and maximum error ≤0.5 pixel. The original
coverage policy, including its existing sparse-translation-consensus exception,
was preserved. Error measurements on interior points are conditional on the
inherited mask, which also checks observed destination coordinates; raw evidence
retains accepted/interior counts and attrition.

## What changed in the experiment

Gain 16 was applied only to the S16 image used by Harris: the expected values
are `16 × U8`, within 0–4,080. The source video remains 8-bit. Native source
pixels, U8 tracking images and LK/pyramid intensities were not amplified.
No RAW16 camera data was used.

At a fixed Harris threshold, this changes effective corner selectivity. It adds
neither information nor signal-to-noise ratio. Therefore, thousands of newly
reported corners are not by themselves evidence of better object detection.
The known-motion controls are necessary checks, not deployment validation.

The complete-capacity arms used 19,866 slots at half resolution and 78,899 at
native resolution, without changing the 1,000-point selected-feature limit.
For the high-contrast control, capacity alone exposed 10,475 raw corners instead
of 8,192: 2,283 previously omitted, with the legacy output an exact prefix of the
complete coordinates/scores. Accepted points rose from 870 to 1,000 and coverage
from 43/48 to 48/48 cells. This confirms the control-storage caveat independently
of the AOT shortage. No U32 endpoint saturation or float32 score-rounding change
was recorded in any arm; the maximum recorded raw score was 3,185,630.
Changing capacity alone did not fix the AOT shortage. Native resolution alone
also remained insufficient. The factorial does not justify paying for a native-
resolution frontend as the remedy for this particular failure.

## The remaining rejection is substantive

Half-resolution gain 16 yielded **784–928 flow-accepted points per real pair**;
native-resolution gain 16 yielded **145–881**. Flow acceptance includes the
original status, displacement and forward/backward checks; it is not equivalent
to accurate background registration.

Every real pair in both gain-16 arms failed all three of these global checks:

- Minimum inlier ratio (configured minimum 0.60).
- Median inlier reprojection error (configured maximum 0.35 native pixel).
- 90th-percentile inlier reprojection error (configured maximum 0.75 pixel).

There were additional coverage rejections in one half-resolution and two native-
resolution pairs, and insufficient inliers in one native-resolution pair.
These quality checks were not loosened. All real gain-16 fits reported a rejected
coverage-acceptance path; none was accepted via a sparse-coverage exception.

| Gain-16 arm | Translation inliers | Inlier ratio | Inlier median residual | Inlier p90 residual |
|---|---:|---:|---:|---:|
| Half resolution | 36–434 | 4.59–50.76% | 0.406–0.649 px | 0.807–0.928 px |
| Native resolution | 26–364 | 4.41–43.70% | 0.414–0.637 px | 0.824–0.926 px |

These residuals are measured among RANSAC inliers, not among all matches.
For comparison, half-resolution gain-16 translated high/low generated controls
had fit-vector errors 0.00223/0.02239 pixel and interior maximum errors
0.00991/0.13513 pixel. The native-resolution equivalents were 0.00039/0.00551
and 0.00489/0.06782 pixel. Thus the known generated translations pass their
predeclared error limits while the real-pair global fits fail theirs.

A consistent forward/backward correspondence can still be wrong. Alternatively,
a moving airborne camera observing terrain at different depths may not admit
one image-wide translation. Both are hypotheses here, not established causes.
No alternate camera-motion model was fitted in this experiment.

## Recommended next move—not executed

Keep half-resolution gain 16 as a **diagnostic candidate**, not a production
default. First apply known small translations/jitter to these same real-image
textures and measure recovered motion against the imposed truth. This fills the
gap between identical-image checks and translated generated block textures.
Preserve the quality gates and predeclare the transforms/error criteria.

If tracking known shifts of the real textures fails, investigate point quality
and flow before changing the global-motion model. If it passes, inspect the
real-pair displacement patterns and test model adequacy separately. Do not
assume that a more flexible model or relaxed gate solves the problem.

Then validate the selected approach on representative real mostly-stable-camera
positive and negative footage, including slight jitter and feature dropouts,
before any production promotion or detector-accuracy claim. Synthetic warps move
the existing image content, including its noise, and cannot replace that test.

## Evidence, limits and unchanged state

Full local numerical evidence (including raw points/scores) is 272,675,645 bytes:
[result.json](../../outputs/seaqr_aot_pilot_20260927/feature_factors_01/result.json).
For a small entry point use the
[summary](../../outputs/seaqr_aot_pilot_20260927/feature_factors_01/summary.json)
and [artifact index](../../outputs/seaqr_aot_pilot_20260927/feature_factors_01/README.md).
Compact copies, without duplicating the large raw result, are under
[repository evidence](../results/tiny_target/aot_pilot_20260927/feature_factors_01/README.md).

The run used one worker and fresh per-pair estimators with unchanged process-
global VPI cache behavior. Scale also changes the native footprint of the fixed
LK/NMS windows. There is no independent camera-motion truth for real adjacent
pairs and no real stable-camera validation in this experiment. No speed claim,
airplane recall measurement, full-clip detector execution or production promotion.

223 local tests passed, including 36 new generated/mocked factor tests. Hardware
evidence comes from this completed Jetson run, not inferred from those tests.
Input/dependency/artifact/clock checks passed, resources were closed, and the
dedicated tmux process exited. The original baseline journal and previous
diagnostic result retain their hashes. No private media or sealed holdouts were
accessed; no packages, hardware clocks or services were changed.

The [independent audit](../../outputs/seaqr_aot_pilot_20260927/feature_factors_01/audit.json)
verified all 105 cases and conversions, 42 gain-input comparisons, 40 real/static
feature comparisons and both reversible method edits. It reconstructed 749
finite raw-array hashes, all 54 returned-correspondence acceptance masks, all
34 interior-error summaries and all 105 scientific decisions. All 41,069
accepted points were finite and matched the independently reconstructed masks.
No explicit complete-grid capacity was reached.

Serialization limitation: 32 raw failed-flow arrays contain 3,464 nonfinite
scalar entries represented as JSON null. The inherited serializer retains the
original array byte hash, but the original NaN/Inf bit patterns cannot be
reconstructed from those nulls. Those 32 raw-flow hashes are not claimed as
independently reconstructed. Invalid flow was correctly excluded; raw Harris
arrays and accepted correspondences are unaffected by this limitation.

Result SHA-256: `aaeb0cb9898199548afe9c8303d27717c1694c596dec2b775df6355d24584f56`.
Manifest SHA-256: `753ff1ba36e7921e76ba061f1f39fb31d73d171362f74411fe27483fa9399d5c`.
Remote workspace: `/tmp/seaqr_aot_factors_20260927_5FZmd7` on `serg@100.73.41.79`.
