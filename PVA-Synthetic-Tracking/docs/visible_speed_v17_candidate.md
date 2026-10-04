# v17 candidate frozen after baseline profiles

Profiles 0126 and 0082 both pass archived 128-frame journal/motion parity.
Observed learning-mask call means with cProfile enabled: 22.16 and 30.09 ms.
Detection totals: 175.09 and 146.65 ms/frame. These are instrumentation-bearing
observations, not clean performance or isolated GPU timings.

Change only the sparse branch of shape_learning_mask: preserve every validation,
float64 coordinate conversion, NumPy ties-to-even rounding, clipping of source
points, disk construction, sparse/dense selection criterion, and dense fallback.
Replace temporary offset arrays/repeated NumPy scatter with one compiled integer
union operation. Normalize bool support bytes, preserve unsupported pixels,
exclude the same clipped disk footprint, never mutate input masks or regions.
No GPU binary changes. No thresholds, data sampling, candidate/track policy,
history or coverage changes. Lock the original function's module SHA and exact
branch text before constructing an isolated in-memory adapter.

Generated gates: byte-for-byte masks, input nonmutation, and error behavior on
empty/duplicate/overlapping/border/outside/fractional footprints, unusual masks,
noncontiguous masks, margins 0.5 through 16, dense fallback and native resolution.
Benchmark reference/candidate in reversed-order generated repeats including all
Python preparation and allocation. Fail closed if native helper returns error.

Once generated gates pass: two reversed-order 128-frame pairs on each 0126/0082
prefix, then full candidate 0029/0126/0055/0082. Existing v13 full journals are
regression references, not fresh paired full-clip timing controls. Stop if any
non-timing output differs. Report prefix speed separately from archived full-run
comparisons. All work remains isolated; RAW16 remains paused.
