# RAW-aware motion and explicit review outputs

**This candidate failed the real-development non-regression check and must not
replace the current default.** It recovers search availability on RAW0029 but
causes six reference resets and zero search windows on RAW0040. Passing all 36
generated checks was not sufficient. The experiment is retained for diagnosis.

This is an opt-in development path, not a new production default. Detector
thresholds, synthetic velocity grid, masks, budgets and motion-fit gates remain
unchanged. The historical eight-bit/PVA path remains the default.

## Motion frontend

`configs/evaluation/raw16_motion_v3.json` changes only two motion options from
`phase20_motion_v8.json`:

- `feature_intensity_mapping: raw_robust_u16_v1`
- `optical_flow_backend: CUDA`

The RAW-only frontend makes a separate U16 working image. On an eight-pixel
sampling grid, each frame's valid, unsaturated 5th and 95th percentiles map to
8192 and 49152. Values are rounded/clipped into U16. A degenerate/flat sample
produces a flat working image. The policy is identical for all clips, has no
clip IDs or target coordinates, and has no retry-until-accepted behavior.
It is adaptive to overall frame brightness, not a clip-specific tuning table.

CUDA rescales the working image. PVA builds pyramids and detects Harris corners
at the tested dimensions. Harris receives S16 via an offset of -32768, avoiding
unsigned-to-signed clipping. CUDA computes forward/backward PyrLK flow in U16.
The existing robust translation fit, spatial support, residual, displacement
and forward/backward checks still decide whether motion can be used. Rejection
continues to reset the reference; it does not silently substitute zero motion.

Original RAW pixels are not overwritten or replaced by the motion working
image. Stabilization still warps the original image; synthetic detection still
processes the configured full-native image, with its original radiometric masks.
Eight-bit inputs keep their historical intensity mapping; choosing CUDA flow
is a separate, explicit backend option.

## Why both precision and backend changed

Generated known-shift controls exposed two distinct issues. The old RAW-to-U8
mapping lost useful weak texture. A fixed asinh-to-U8 mapping did not solve the
suite. Native U16 recovered features, but PVA flow rejected bright and
sensor-pattern cases; using CUDA flow with identical U16 inputs recovered those
shifts. Unnormalized flow still failed the affine lighting-change case.
The final frontend combines higher-precision working images with a fixed global
normalization policy and CUDA flow.

NVIDIA documents U16 support for
[PyrLK](https://docs.nvidia.com/vpi/3.2/group__VPI__OpticalFlowPyrLK.html), S16 input for
[PVA Harris](https://docs.nvidia.com/vpi/3.2/group__VPI__HarrisCorners.html), and explicit
scale/offset behavior for
[image conversion](https://docs.nvidia.com/vpi/3.2/algo_imageconv.html).
The backend accuracy differences above are measured results on this Jetson,
not a claim that all PVA implementations or all U16 scenes fail.

Keeping the numeric Harris threshold at 0.5 does not preserve its effective
selectivity after changing image intensity scale: the corner score depends on
that scale. Extra weak/noisy correspondences are a hypothesis to investigate
in the real-scene regression, not yet an established cause. The final geometric
acceptance gates did remain unchanged.

The `raw_asinh_v1` and `raw_linear_u16_v1` modes are retained solely to reproduce
failed diagnostics. None of the three experimental intensity modes is approved
as a general replacement. Robust-U16 is guarded against use with the unvalidated
PVA-flow combination; its CUDA version also needs the RAW0040 regression resolved.

## Verification boundary

The 12 generated cases include static/subpixel/larger shifts, HDR texture,
lighting change, sensor-fixed noise, a separately moving foreground, and
flat/noise/rotation rejection controls. Three seeds exercise the same frozen
cases. The development runner verifies every gate and tested motion-module
hash before permitting a 64-frame prefix from only RAW0029 or RAW0040.
It never opens a split or enumerates camera media.

Generated scenes are development tests, not an unbiased estimate of real recall.
Global normalization can still amplify weak sensor patterns; coherent features
do not by themselves prove physical background identity. Rolling shutter,
parallax, scene changes and extreme lighting require additional validation.
No new claim of airborne identity, dark-target recall, acceleration robustness,
full-clip reliability, false-alarm rate or real-time speed follows from this work.

## Retained tracks versus review preview

Dense reports now include an `output_contract` identifying `track_pool` as the
complete **bounded retained pool**, and `shortlist` as a human-review preview.
The pool itself has a finite capacity; it is not every hypothesis ever considered.

The report-only exporter preserves the original track arrays and order, writes
separate `retained_tracks.json` and `review_preview.json`, and includes an index
listing every retained track. It rejects inconsistent counts/IDs and never
overwrites an existing output directory. An unavailable search with zero tracks
is labeled unavailable, not an empty scene. Neither output labels unknown
hypotheses as real airborne objects.

Run with `python -m tiny_target.dense_review --report REPORT.json
--output-directory NEW_DIRECTORY`. No video is opened by the exporter.

Measured results and source snapshots are kept in
`results/tiny_target/raw16_motion_v3_20260915/`.
