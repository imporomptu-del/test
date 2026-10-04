# Exact CPU batching for the RAW16 motion frontend

This opt-in performance change follows the v5 correctness fixes. It preserves
the Harris allocation, tracked-point budget, intensity mapping, motion acceptance
limits, source masks, detector settings and VPI cache-isolation policy.

`configs/evaluation/raw16_motion_v6.json` differs from v5 only by
`motion.feature_cpu_policy: batched_exact_v1`. The default remains `reference`.

## Operations changed

- Saturation neighborhoods are gathered in batches of at most 4,096 points.
  Clamped edge coordinates repeat pixels, which preserves the original OR
  predicate over a cropped neighborhood. No padded zero can create false
  support. Radii above eight retain the scalar implementation, keeping temporary
  storage bounded. The configured radius is still two.
- Spatial selection still performs the same stable descending score sort. It
  groups ranked entries stably by cell, admits the same per-cell prefix, restores
  global rank, and applies the original overall budget. Ties, quotas, ordering
  and every selected point must match.
- Diagnostic grid coverage uses the same cell arithmetic with a unique-cell
  reduction instead of a Python loop.

The scalar expressions and array expressions must not silently use different
precision. Cell arithmetic explicitly preserves the multiply and divide result
dtypes of the existing scalar expression on the running NumPy version. This
matters because [NumPy 2 changed scalar promotion](https://numpy.org/doc/stable/numpy_2_0_migration_guide.html#changes-to-numpy-data-type-promotion).
Tests include immediately adjacent float32 values around cell boundaries on both
the local NumPy 2 environment and Jetson's NumPy 1 environment. Unusual legacy
grid arguments stay on their reference path.

## Exactness contract

The CPU microbenchmark loads the old implementations from a SHA-256-locked v5
runtime archive. It compares the frozen implementation, current reference, and
batched implementation. Eligibility masks, rejection reasons, ordered selected
indices and coverage are compared exactly. Inputs include uint8, RAW12, native
RAW16, masks, borders, saturated samples, nonfinite coordinates, tied scores,
multiple radii and chunk boundaries. Original frames must remain unmodified.

Before any media run, the harness checks the CPU evidence and all 48 unchanged
generated motion/rejection/reseeding controls against the current module hashes.
The 64-frame RAW0040 sequence must match all archived v5 point identities, then
pass the existing eight reverse-order replay checks. A failed gate stops the
sequential batch.

Full-native comparisons use only the previously authorized 64-frame RAW0029 and
RAW0040 development prefixes, with the unchanged injected RAW0040 controls. The
reference and candidate use the same source/configuration/library hashes and
instrumentation. The order is reference/candidate for RAW0040 and
candidate/reference for RAW0029. No media directory, sealed split or holdout is
opened. Results remain bounded development evidence, not an entire-clip or
real-airborne accuracy claim.

The report comparator checks all non-timing screening values, candidate scores,
retained tracks and injection measurements. Only known timing fields, the
verified execution-only motion-config identity, and the isolated workspace
prefix for the verified CUDA library/control layout are normalized. Comparator
tests ensure tiny score/position changes and changed detection gates still fail.

## Timing and remaining work

New substage timings separate feature readback, saturation eligibility, spatial
selection, upload and diagnostic metadata work. The historical `total` timer
ended before the diagnostic grid calculation; it is retained for compatibility.
`total_including_metrics` and an external estimator-call timer now expose that
extra work. End-to-end claims must use the matched wall-time measurements, not
the CPU-only speedup or sums of overlapping timing fields.

The VPI cache-isolation workaround is intentionally unchanged in this experiment
so its correctness effect is not mixed with the CPU speed change. Removing it
requires a separate reusable-state investigation with the same reseeding and
order-invariance gates. The upper injected-control miss and real-airborne
accuracy evaluation remain open; faster identical execution cannot resolve them.

Measured results are stored in `results/tiny_target/raw16_motion_v6_20260915/`.
