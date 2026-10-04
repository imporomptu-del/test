# Phase 3: RANSAC Global Camera Motion

## What this stage adds

Phase 2 returns individual background tracks. Phase 3 asks whether enough of
those tracks agree on one physically plausible camera motion.

The implementation supports two deliberately constrained models:

- `translation`: two parameters, used for the current controlled recording;
- `similarity`: translation, rotation, and one uniform scale, available when
  residual evidence demonstrates that translation is insufficient.

Full affine and homography are not enabled. Adding flexibility before the data
requires it would let local track errors masquerade as camera shear or
perspective.

For every pair, deterministic RANSAC proposes a model, selects tracks within
the configured reprojection threshold, refits on all inliers, and recomputes
the final residuals. The model is rejected if any configured gate fails:

- correspondence or inlier count;
- inlier ratio;
- inlier grid coverage;
- non-collinearity for similarity transforms;
- median, p90, or maximum reprojection error;
- translation, rotation, or scale limits; or
- the Phase 2 correspondence-quality gate.

Rejected matrices remain in the JSON for diagnosis but are never composed into
the stabilization chain.

## Transform direction and reset behavior

Two transform directions are named explicitly:

```text
RANSAC pair matrix:       previous frame -> current frame
stabilization chain:      current frame  -> segment reference frame
```

The chain therefore composes the inverse of each accepted pair matrix. Phase 4
will use the resulting direct current-to-reference matrix to warp each original
RAW16 frame once.

The current configuration uses a fixed reference within each quality segment.
When a pair fails, `failure_policy: reset_reference` starts a new segment at the
current frame and resets downstream temporal windows. A bounded
`reuse_previous` policy is implemented and tested, but it is not enabled for
the current evidence because silently freezing motion would be riskier than
losing one integration window.

Timestamp or sequence discontinuities force a reset even when RANSAC itself
finds a model.

## Why translation is the current model

On the first three RAW16 pairs, translation accepted the first two with all
116 correspondences as inliers:

| Pair | Inliers | Median / p90 residual | Inlier cells | Result |
|---|---:|---:|---:|---|
| 0 -> 1 | 116/116 | 0.0188 / 0.0320 px | 12/48 | accepted |
| 1 -> 2 | 116/116 | 0.0186 / 0.0324 px | 12/48 | accepted |
| 2 -> 3 | 83/94 | 0.4075 / 0.8176 px | 9/48 | rejected |

A diagnostic similarity fit on the exact same tracks reduced the accepted-pair
median residuals by less than 0.0006 px. Estimated rotation stayed within
0.00028 degrees and scale within 0.000005 of unity. On the rejected pair,
similarity still had only 9/48-cell coverage and a 0.395 px median residual.
The extra degrees of freedom therefore do not repair the evidence, so
translation remains the justified least-flexible model.

The rejected 2 -> 3 pair failed the Phase 2 gate, inlier-coverage gate, median
residual gate, and p90 residual gate. The chain did not reuse its diagnostic
matrix; frame 3 became reference frame 3 in a new segment.

## Reproducibility and tests

The tests cover exact translation and similarity parameters, deterministic
replay, 0% through 60% outliers, clustered and collinear features, parameter
limits, 100-pair composition, matrix inverse round trips, bounded reuse,
failure reset, and recovery in the new segment. The Jetson suite also exercises
the complete synthetic PVA -> RANSAC -> composed-transform path.

Run the recorded pipeline with:

```bash
python3 -m tiny_target.motion_cli \
  --config configs/tiny_target_test.yaml \
  --max-pairs 3 \
  --omit-points \
  --output /tmp/tiny_target_global_motion.json
```

The report records both transform directions, inlier metrics, parameter
stability, failure/reset counts, implementation hashes, and PVA and CPU timing.
The compact Jetson report is saved at
`results/tiny_target/phase3/raw16_global_motion_3pairs.json`.
Phase 4 consumes only accepted composed matrices; rejected diagnostic matrices
are never allowed to warp image pixels.
