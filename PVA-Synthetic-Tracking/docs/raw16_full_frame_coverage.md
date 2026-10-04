# RAW16 full-image coverage and availability

This is an opt-in correctness increment, not a replacement for the frozen Phase 19 experiment and not a claim of real-airborne-object accuracy.

## Scope

`configs/evaluation/raw16_full_frame_v2.json` requests `coverage_mode: full_frame`. Both video-source classes resolve width and height from the source probe before bounds validation; the screener receives that resolved configuration. There is no spatial downsampling of detector pixels. The historical default remains `configured_crop`, including its 4784×1920 crop. On a 4784×3190 source the opt-in path requests all 3190 rows, including the previously excluded lower 1270 rows.

The new option is keyword-only in the Python configuration constructor, preserving existing positional arguments.

Full-image **requested area** is not the same as valid search coverage. Stabilization borders, invalid/dark/saturated sensor samples, background warmup, local filter support, temporal support and candidate margins still exclude samples. Per-frame and per-window 3×3 region counts expose these exclusions without claiming that every pixel is observable.

The only screen-configuration differences from Phase 19 are coverage mode, crop height, and the previously bit-exact-tested `masked_ufunc` background execution. Thresholds, temporal windows, candidate budgets, polarities and velocity limits are unchanged. Original RAW16 samples remain unchanged by motion-image conversion; stabilization continues to use the existing float32 interpolation path.

## Motion reliability is still a separate gate

The bounded validation uses the existing `phase20_motion_v8.json` sparse-translation consensus policy at half-size PVA feature resolution. It still requires at least 30 accepted correspondences and 30 inliers. Sparse spatial acceptance still requires independent support from at least four cells plus residual, span and leave-one-cell-out consistency checks. No thresholds have been loosened for chunk0029.

Two diagnostic-only alternatives were evaluated on the same five adjacent pairs within the first 64 frames of each authorized RAW16 development clip:

- Shared pairwise 0.5–99.5 percentile contrast mapping for the PVA feature image.
- A larger feature image derived from the existing all-PVA pyramid size cap: 3071×2048 instead of 2392×1595.

Neither resolved chunk0029's insufficient background support. Neither is enabled in the new configuration. The larger image was deliberately bounded to supported hardware geometry; [NVIDIA's VPI 3.2 Harris documentation](https://docs.nvidia.com/vpi/3.2/group__VPI__HarrisCorners.html) also imposes a PVA image-size limit.

An accepted fit is now reported separately from a transform actually applied to the reference chain. Timestamp discontinuities may force a reset even when a fit itself is accepted. PVA exceptions, rejected fits, reference resets, background-warmup frames, zero-valid-support frames and windows with valid ranking support are recorded separately. The old `frames_screened_after_background_warmup` field is preserved for compatibility; use `screening.availability` to determine whether useful support existed.

## Evidence and reproduction

`scripts/validate_raw16_full_frame.py` allows only development clips0029/0040, reads exactly the first 64 frames, verifies frozen configurations/library hashes, requires native uint16 and recorded sidecar timestamps, and writes results exclusively. It neither enumerates video directories nor opens the sealed split. Run one worker at a time on the Jetson.

Three predeclared trials:

1. Full-image, unchanged chunk0040 frames.
2. Identical chunk0040 frames with three bright synthetic controls at fixed upper/middle/lower positions, active on frames8–63.
3. Full-image chunk0029, retaining any unavailability as a failure of detection availability rather than calling it a successful negative search.

The control flux is 6000 DN integrated over the PSF, not a sensitivity limit. Velocities are fixed at (2,1), (−2,1), and (2,−1) px/s. These controls are injected **after stabilization**, so they cannot establish motion robustness to target interference. Native source pixel hashes and motion decisions are compared between the paired chunk0040 runs. The source files are not edited.

`scripts/summarize_raw16_full_frame.py` reads artifacts only. It independently recalculates saved checks, verifies consistent code/configuration provenance and paired source evidence, and requires all three controls in both the track pool and bounded shortlist for its spatial-control gate. Successful process completion is reported separately from detection availability. Measurements include diagnostic hashing/logging and are not clean throughput benchmarks.

Unit tests cover arbitrary source geometry, both source types, exact native crop preservation, motion-fit-versus-reset accounting, zero-support saturation/masks, resets and warmup, nine generated spatial target positions, and rejection of zero-window or lower-region-gap evidence.

## Remaining limits

- RAW16 chunk0029 does not yet have a validated reliable motion path.
- Bright-only synthetic integration and the narrow ±3 px/s per-axis grid remain. Dark targets, faster motion and acceleration are not validated by these controls.
- No RAW-specific independent airborne truth set is established. AVI labels cannot be transferred to same-numbered RAW files without verified alignment.
- Unlabeled candidates and tracks remain review workload; neither recall nor false-alarm rate can be measured from these runs.
- Full-frame processing covers approximately 66% more pixels than the previous crop and has a corresponding cost. Speed work must retain these correctness/availability checks.

Next correctness work should develop and independently validate RAW-aware motion evidence for low-texture scenes, including known camera shifts, foreground contamination, lighting changes and failure cases. Do not force acceptance by relaxing support checks or silently assuming zero camera motion. Once validated, accelerate filtering and stabilization against an exact/equivalence baseline.
