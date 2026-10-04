# Opt-in exact finite float32 noise execution

`scripts/noise_v18.py` replaces only the host tile-statistics calculation inside
the existing visible 8-bit GPU-resident detector. It is an execution adapter,
not a new detector, synthetic tracker, threshold policy or default mode. The
v17 learning-mask optimization remains enabled in both comparison arms.

## Numerical contract

The same downloaded float32 samples, tile membership, stride and current
support mask are used. Each tile's median is computed by selection; an even
tile adds its two central float32 values and divides by float32 two. Absolute
deviations are float32; their median is converted to float64 before multiplication
by 1.4826 and application of the existing floor. Device statistics are float32.
The aggregate sigma median still uses NumPy float64. There is no padding,
subsampling, quantile approximation, fast-math or change in pixel coverage.

Native execution is restricted to bounded canonical layouts, float32 samples,
boolean support, and ordinary finite values. Nonfinite values, negative zero,
very small or very large magnitudes and unsupported generic layouts/dtypes use
the original NumPy function. These fallbacks are explicit and counted. Native
ABI errors instead raise; errors are not hidden behind fallback.

The value guard uses float bit patterns, since Jetson floating-point modes can
flush tiny-value comparisons or double-to-float test-data conversions to zero.
Generated subnormal fixtures are constructed from integer bits. Numeric
fallback is a semantics-preservation measure, not validation of corrupt inputs.

## State and memory

Only flat support indices and tile boundaries are cached. The cache key includes
image shape, sample count, stride, and every slice bound, offset and count.
It is rebuilt when any geometry changes. Sample values and support are reread
on every call, including warm-up and motion-reset frames. Private outputs and
scratch buffers do not alias or mutate caller inputs. The cache is bounded by
the sample geometry; its int64 indices occupy about 7.6 MB at native sampling.
No GPU library, full-frame transfer or detector-state ownership changes.

## Evidence and activation

The library is compiled in a fresh isolated directory with C++17, `-O3`,
`-fno-fast-math`, and `-ffp-contract=off`. Its exact source and binary hashes are
bound to a generated test receipt. Media execution requires that receipt and
the original frozen reference hash. `run_visible_v18.py` scopes media to four
previously used AVIs, records numerical output digests and fallback counts,
and delegates the unchanged v17/v13 regression and lifecycle checks.

354 generated cases cover native geometry, changing masks and geometry,
noncontiguous/noncanonical masks, repeated ranks, odd/even/empty samples,
signed zero, nonfinite/extreme/subnormal inputs and generic fallbacks.
Independent `verify_visible_v18.py` checks archived non-timing journals and
motion identities, full aggregate tracks, decoder lifecycle, execution
provenance, prefix noise digests and timing receipts. See the final experiment
README for observed video results and limitations; this document itself does
not assert a throughput gain or new airborne accuracy result.

RAW16 and sealed holdouts are out of scope. No production configuration is
changed by importing or building the helper.
