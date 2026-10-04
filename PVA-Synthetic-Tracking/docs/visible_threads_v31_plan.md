# v31 — one-thread numerical-library experiment

Selected after the complete six v30 diagnostic traces: a 12-thread NumPy
OpenBLAS pool is loaded; several early-created non-consumer workers are heavily
scheduled, and process CPU substantially exceeds consumer CPU even outside
tracking. This supports a controlled thread-count intervention, not yet proof
that those workers explain the previous integration regression. Clocks varied;
no thermal or scheduling cause is assumed.

The only new setting is `OPENBLAS_NUM_THREADS=1` in a fresh experiment child.
No global environment, power, clocks, affinity, services, installations or
runtime source are changed. Original v29 code, v26/v28 algorithms, full image,
thresholds, quotas, cadence and arithmetic remain frozen. Verify loaded BLAS
worker count, existing dependency hashes and full output journals each run.
RAW16 and sealed holdouts are excluded.

1. Under one thread, rerun all 36 v28 generated state/output scenarios and require
   their ordered hashes equal the archived default-thread run. Stop if different.
2. Combined one-thread private-state/output/actual-learning smokes: 128 frames
   each of development 0126 and 0082. These are not speed measurements.
3. Three clean six-arm rounds per prefix, one video worker at a time. Four main
   arms are v20 default, v26 one-thread, v28 one-thread, combined one-thread.
   Two additional causal controls are v26 default and combined default.
   Orders: v20/v26_default/v26/v28/combined_default/combined; reverse; then
   v28/combined/v20/combined_default/v26/v26_default. Retain every run.
4. Apply the original v29 gate to the four main arms: combined >=1.20x v20 pooled
   FPS on both scenes, every v20 pair positive, no pooled-plus-two-of-three p95
   regression on any of the three recorded latency boundaries, and combined no
   slower than either single-component arm. Also require combined pooled FPS
   no slower than both default-thread GPU/combined controls. Report thread-only
   paired differences separately; the arm names must not hide the thread setting.
5. Separate post-timing default/one-thread combined traces on both prefixes to
   observe whether CPU worker activity changes. Profiling never enters clean
   speed calculations. All four traces must retain exact outputs.
6. Only on all correctness and performance gates passing, run complete existing
   development 0029/0126/0055/0082 with the combined one-thread candidate and
   compare every non-timing journal field and motion against archived baselines.
   Do not retune if a gate fails. Do not promote defaults automatically.

Fresh outputs and source/unit/dependency freeze before the experiment. Preserve
failed attempts and all outliers. Independently audit compact exported journals,
actual loaded thread metadata, commands, source/library hashes and recomputed
gates locally. File replay FPS and equivalence are not live-camera latency,
new airborne recall/precision or generalization evidence.
