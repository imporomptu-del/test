# AOT checkpoint: richer global models are not a sufficient fix

2026-09-28 UTC. Both approved experiments are complete: a local 64-case spatial
holdout comparison and a frozen 93-call Jetson larger-translation diagnostic.
**Do not promote similarity or affine correction yet.** The evidence does not
support a simple global-model replacement as the repair for the missed airplane.
Production, detector thresholds and camera-motion quality gates are unchanged.

## What this changes in our understanding

1. Under the fixed spatial-holdout protocol, rotation/scale-aware similarity
   improves median error in **5/8** actual pairs and p90 in **7/8**, but leaves
   large regional errors. Affine improves median error in only **1/8** and p90
   in **3/8**. Extra model freedom does not reliably predict unseen regions.
2. PVA/LK and the inherited global translation estimator can recover the tested
   larger pure translations: **80/80 nonzero natural controls** have accepted
   global fits, with maximum translation-vector error **0.0231 native pixel**
   (rounded upward). Large displacement alone is not an established explanation
   for the actual-pair failures.
3. Individual matches are still not uniformly trustworthy. Across those 80
   cases, **4,266 / 63,899 fixed-cohort feature observations were lost (6.68%)**.
   Four accepted matches at ±48px exceed the existing 0.5px truth tolerance;
   two are wrong by about **25 and 46 pixels** despite passing forward/backward
   filtering. A good global translation can coexist with seriously wrong points.

This narrows the next question to **whether the spatially varying actual motion
is genuine or partly a coherent matching error**. We should cross-check matches
against original image patches before choosing a regional motion strategy.
This step did not rerun the detector or restore its airborne-object detection.

## Experiment 1 — models tested on regions excluded from fitting

All 64 saved `half_gain16_complete` cases were retained: eight actual adjacent
pairs and 56 previously captured known-shift controls. Each accepted point is
tested once in one of four contiguous image quadrants. Training excludes that
quadrant plus a 64px guard; base weights balance the occupied 6×8 training cells.
All three models use initial weighted least squares and exactly 20 joint-residual
Huber updates (delta1px). No original global inlier mask selects fitting/testing
points; no test residual chooses weights or changes thresholds.

All **768 folds (64×3×4)** were numerically valid and all 192 model/case
evaluations scored every original accepted point. Numerical validity is not a
motion-quality pass. Previously lost features remain lost and are not recovered
by these fits. The translation comparator uses the same new diagnostic fitting
protocol as similarity/affine, not the old production RANSAC candidate.

Each table cell is **median / p90 held-out error, native pixels**, against saved
accepted correspondences. Actual correspondences are not physical ground truth.

| Input pair | Translation | Similarity | Affine |
| --- | ---: | ---: | ---: |
| 000→001 | 1.84 / 5.01 | 1.57 / 4.34 | 2.36 / 5.12 |
| 042→043 | 2.38 / 6.27 | 2.11 / 4.56 | 2.85 / 5.76 |
| 085→086 | 1.52 / 6.53 | 3.10 / 6.23 | 4.17 / 7.67 |
| 127→128 | 2.28 / 7.96 | 1.81 / 5.16 | 2.52 / 6.31 |
| 170→171 | 1.35 / 6.78 | 3.20 / 6.12 | 4.68 / 8.18 |
| 212→213 | 8.60 / 13.10 | 4.50 / 9.42 | 6.61 / 12.22 |
| 255→256 | 4.56 / 9.50 | 3.27 / 9.68 | 5.77 / 11.38 |
| 298→299 | 3.81 / 11.00 | 4.56 / 10.70 | 7.78 / 14.12 |

![All eight spatially held-out comparisons](/Users/romanmaksymiuk/Documents/SEAQR/outputs/seaqr_aot_pilot_20260927/motion_models_01/actual_holdout_errors.png)

Worst-region results also prevent a reassuring pooled interpretation: for
298→299, the worst supported cell's median error is **14.46px translation,
18.29px similarity, 21.33px affine**; corresponding worst cell p90 values are
34.65, 36.90 and 39.09px. On 212→213, similarity helps materially but still has
a worst-cell median of 14.61px. All 48 cells' support and errors are retained;
cells with fewer than five scored points are unavailable, not zero error.

The existing small synthetic controls provide a useful countercheck. Across
the 48 nonzero cases, median **mapping-prediction error to known truth** ranges
are 0.0025–0.0263px for translation, 0.0129–0.0331px for similarity and
0.0101–0.0417px for affine; their maximum pointwise mapping-prediction errors are
0.0290, 0.0968 and 0.1250px. These are fitted mapping errors on accepted test
locations, not raw LK point error or recovery of rejected tracks.

The negative real-case result applies to this preregistered estimator and
quadrant-extrapolation protocol; it does not prove every possible affine fit
would fail. The 64px guard does not create statistically independent errors,
and spatial holds are not independent recordings. No unplanned estimator sweep,
projective fit, local warp or per-case model selection was performed.

## Experiment 2 — larger exact translations on the Jetson

Eight approved source images ×11 fixed shifts plus five unchanged generated
bridge controls = **93/93 completed calls**. Same Harris-only gain16 arm,
feature scale0.5, capacity19866, U8 LK inputs, status policy and quality gates.
Only experiment identity, case list/count and matching group cadence changed
from the frozen preceding natural-shift runner; tests verified the bounded source
diff and unchanged computational function bodies before launch.

All eleven groups below had **8/8 accepted global fits**. Static cases had8/8
strict truth passes; every nonzero group had0/8. The failing criteria were the
unchanged point-median ≤0.1px and/or point-maximum ≤0.5px checks, not global
translation recovery. All five generated bridge controls passed their gates,
including the expected rejection/no-corners behavior of the flat control.

Point errors below are the inherited conditional accepted-interior metric.
Survival uses the separate fixed **pre-flow** 128px source/expected-destination
cohort; different denominators must not be conflated.

| Shift (x,y) px | Largest global error | Point-median range | Largest point error | Fixed-cohort survival range |
| --- | ---: | ---: | ---: | ---: |
| (0,0) | 0 | 0 | 0 | 86.31–98.79% |
| (1,0) | 0.0114 | 0.0945–0.1258 | 2.3268 | 85.61–98.52% |
| (-1,0) | 0.0097 | 0.0954–0.1235 | 0.8128 | 85.85–98.38% |
| (16,0) | 0.0159 | 0.1022–0.1170 | 0.3164 | 86.01–98.51% |
| (-16,0) | 0.0110 | 0.1047–0.1193 | 0.3826 | 86.00–98.63% |
| (32,0) | 0.0177 | 0.1048–0.1179 | 0.3313 | 85.96–98.63% |
| (-32,0) | 0.0146 | 0.1058–0.1173 | 0.3423 | 86.00–98.62% |
| (48,0) | 0.0230 | 0.1068–0.1174 | **45.8238** | 84.73–96.83% |
| (-48,0) | 0.0166 | 0.1076–0.1192 | **24.9268** | 85.07–96.55% |
| (32,-16) | 0.0180 | 0.1042–0.1164 | 0.3966 | 85.71–98.47% |
| (-32,16) | 0.0159 | 0.1061–0.1219 | 0.3106 | 86.27–98.75% |

For all 80 nonzero cases, 59,633/63,899 fixed-cohort observations survived and
59,566 were within0.5px: **99.89% of survivors, but 93.22% of the original
cohort**. Neither percentage is target recall. Counts repeatedly reuse eight
textures; they are not independent recording-level observations.

### Accepted large-shift outliers must remain visible

Post-hoc enumeration of every accepted fixed-cohort point above the existing
0.5px truth limit in the 16 ±48px cases found exactly these four:

| Source / imposed shift | Selected index | True position error | Forward/backward error |
| --- | ---: | ---: | ---: |
| 085 / (+48,0) | 402 | **45.8238px** | 1.5326px |
| 127 / (+48,0) | 691 | 1.7906px | 1.8735px |
| 127 / (-48,0) | 386 | **24.9268px** | 2.8524px |
| 298 / (-48,0) | 526 | 0.7239px | 0.6852px |

They passed the original ≤3px forward/backward check. The original robust global
fit excluded the first three from its inliers (including both very large errors),
but retained the 0.7239px error. These are accepted **scene-motion feature
matches**, not target detections or final object tracks. This demonstrates that
bidirectional consistency alone is not a correctness guarantee. It does not
justify choosing a tighter threshold from these four points, nor identify a
PVA hardware fault. Exact point records are saved separately for follow-up.

These controls preserve noise with texture, use integer copies and test a
limited set of directions. They do not recreate real exposure changes,
occlusion, spatially varying camera motion or independent sensor noise, and
do not establish nighttime stationary-camera detection performance.

## Next bounded move

Keep production unchanged and **verify correspondence reliability directly**:
freeze a spatially distributed image-patch review on the same eight actual
pairs, including both agreeing regions and residual-heavy regions. Cross-check
the saved LK matches with an independent image-based check and show ambiguous
or textureless patches explicitly. Use the known-shift failures as diagnostic
counterexamples, not as a threshold-tuning target.

If reliable patches confirm genuinely different regional motion, then design
and test a regional correction with safeguards against absorbing target motion.
If matches are wrong, address feature/match validation first. We have not yet
established which explanation dominates. Preserve the stationary-camera path;
do not simply loosen global confidence gates or ship affine because it is more
flexible. An isolated detector rerun comes after a justified correction, not
before. No accuracy or real-time improvement is claimed at this checkpoint.

## Verification and local artifacts

**286 tests passed**: 247 prior tests plus 28 new model tests and 11 larger-control tests.
Plans/code/tests were frozen before outcomes. Model input/code/manifest hashes
match before/after; Jetson input/helper/config/dependency/clock checks passed.
Its process-local OpenCV thread count was restored 12→2→12. The dedicated tmux
session and worker have exited; all remote evidence is retained. No production
files or clock settings were changed; no RAW16 or sealed holdouts were accessed. Approved AOT media
were decoded for the larger controls; annotations were not used for their cases
or evaluation. No target-detector execution occurred.

The full model result is
[motion_models_01/result.json](/Users/romanmaksymiuk/Documents/SEAQR/outputs/seaqr_aot_pilot_20260927/motion_models_01/result.json), SHA-256
`58b1b1c07d4cc77c0eaedd38b07969810e5d7275ef9888a606018896d446bc52`.
The larger-control result is
[large_shifts_01/result.json](/Users/romanmaksymiuk/Documents/SEAQR/outputs/seaqr_aot_pilot_20260927/large_shifts_01/result.json), SHA-256
`3342c3440dd1d190653a9aa46b5fd0210157ffd3faa811ed76506a78b09b6703`.
Its compressed transfer and decompressed bytes were verified; raw remote bytes
remain unchanged. Compact summaries, manifests, audit receipts and code
provenance are stored alongside them and copied into the repository's
`results/tiny_target/aot_pilot_20260927/` folders.

Independent numerical-check details are in the separate audit receipts; these
are bounded spot-checks, not a second complete execution of all experiments.
The model check covered 60 saved folds and 12 independently refitted
native-coordinate models; the largest prediction difference was 1.63×10⁻¹⁰px.
The larger-control check verified all 93 rows' inventory/accounting, 81,149
accepted positive-case pairs, and exact-byte reconstruction for five predeclared
plus four separately labeled post-hoc cases (36 flow/status arrays). All four
flagged ±48px observations were independently confirmed. No experiment helper
was imported for these independent numerical calculations.
