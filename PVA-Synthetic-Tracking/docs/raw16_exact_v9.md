# RAW16 v9: preserve reference arithmetic while changing execution

V9 replaces neither the detector definition nor its thresholds. It is an
explicit, single-worker research execution path. The source/configuration
archives from v6, v7 and v8 remain frozen and are verified before media access.
Default execution is unchanged.

## Point response: retain the FFT reference

The installed Jetson OpenCV `filter2D` dispatch chooses a DFT path for this
float32 9×9 kernel: without SSE3, its kernel-size cutoff is 50 coefficients.
`crossCorr` then uses **float64** transforms for float32 image input. At native
geometry it uses 256×256 transforms with 248×248 output tiles. The v8 direct
float32 CUDA sum did not reproduce this arithmetic; its failed gates and
artifacts are retained, and that kernel is not used by v9.

`PointFilterFftExact` retains OpenCV's forward DFT, conjugated spectrum product,
scaled inverse DFT, float32 output conversion and final normalization. It
caches the kernel spectrum and per-worker scratch, and schedules independent
tiles on a bounded pool. The media experiment fixes the pool at four workers.
Each worker fills its entire scratch tile with zeros before loading input;
tail dimensions and `nonzeroRows` follow the reference. Output tiles do not
overlap. All futures finish before return, including failure cleanup.

This is **CPU filtering**, not a bit-exact GPU point filter. It avoids promoting
the rejected direct-convolution approximation while recovering some of its
speed benefit. The SSE3 spatial-filter dispatcher is explicitly unsupported.
Outputs are owned; inputs are not modified. Errors are sticky, execution is
thread-confined, and cleanup joins the worker pool.

## Stabilization: match the actual CPU resampler

`WarpTranslationCuda` handles finite float32 images and binary uint8 masks,
constant-zero borders, and exact translation matrices only. It rejects
rotation, scale, perspective, nonfinite values, unsupported shapes and other
OpenCV builds. The constructor checks the validated OpenCV 4.10 build-info
fingerprint, interpolation-table fingerprint and Jetson flush-to-zero mode.

The custom CUDA kernel reproduces:

- CPU inverse-transform calculation, horizontal warp block origins, 1/32-pixel
  coordinate quantization and nearest-even rounding;
- the reference's 32×32 phase table, recovered using isolated basis probes
  through OpenCV's CPU cubic remap;
- the installed compiler's interior arithmetic: round the second product,
  fuse the first, then fuse terms 2–15 in order;
- the distinct border branch: skip out-of-image samples and accumulate valid
  samples from zero in reference order;
- nearest-neighbor mask sampling with the reference's short-coordinate limits.

The image and mask share a kernel launch but retain their distinct sampling
rules. Transfers are synchronous and included in host-call measurements. CUDA
errors stop execution; there is no silent fallback. Existing CPU mask erosion,
exact-identity bypass, source metadata and later processing remain unchanged.
This does not add rotation support or improve the existing translation motion
model's accuracy.

## Integration and truthful metadata

`exact_v9_common.py` installs scoped adapters in the bounded runner. The point
adapter intercepts only the known background point-filter call, validates its
kernel/arguments, and returns the **unnormalized** FFT result. The original
caller's normalization statement is unchanged. The warp adapter changes the
actual stabilized-product backend label to `cuda_translation_reference_v9`;
the requested reference configuration is retained separately.

The full intermediate audit normalizes only that explicit execution label and
two separately verified synthetic-library workspace paths. It does not alter
image hashes, masks, scores, candidates, ordering, support counts, transforms,
or track state. Raw audit-file hashes need not match because execution metadata
differs. Native audit mode also compares each new warp and unnormalized filter
response directly with the CPU reference before downstream use.

The v8 exact CPU motion scorer/conversion cache is active in both timing arms,
so the v9 comparison isolates cached FFT execution and exact CUDA stabilization.
Neither arm uses the rejected v8 direct GPU point filter.

## Gated tests and limits

The generated checker covers 40 filter fixtures with three worker-count
variants, 350 image/mask warp fixtures including native resolution, all 1,024
fractional phase combinations, 18 weak moving-response sequences, and 33
threshold-adjacent impulses. Repeated frozen PVA controls remain mandatory.
Tests include tiny/odd/tile-boundary shapes, holes, zero/constant/signed noise,
full-range sensor values, impulses, half-pixel and 1/64-pixel rounding
boundaries, and large out-of-image translations. Acceptance requires bytes to
match, not a numerical tolerance.

The native runner accepts only the first 64 frames of RAW0029 and RAW0040,
with unchanged injected controls allowed only on RAW0040. It validates archived
runtime/configuration identities, generated results, source shape/timestamps,
and accelerator hashes. The batch stops on a failed check. Timings begin only
after all native checks pass, with two reversed-order rounds per clip and
fresh processes. Outputs are exclusive and reports refuse partial schedules.

These are reused, unlabeled development prefixes, not general airborne recall
or false-alarm validation. The original upper synthetic control's limited
support and clutter/saturation miss must remain visible in the report. No
target relocation, brightness increase, mask relaxation or hit-count reduction
is part of this experiment. More CPU threads are part of the measured speed
change; no power/clock/service settings are altered.

The measured results and reproducible evidence are under
`results/tiny_target/raw16_exact_v9_20260916/`.
