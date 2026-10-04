# AOT feature factors — predeclared bounded experiment

2026-09-27. The user authorized the next feature-only comparison following the
[completed diagnosis](aot_feature_diagnosis_result_20260927.md). This plan is
frozen before hardware results. It does not authorize production promotion,
target-detector tuning or a full detector rerun.

## Question and fixed scope

Does the frozen frontend's feature shortage depend on half-resolution sampling,
Harris input amplitude, or both? Does the legacy raw-corner storage ceiling
truncate the control observations? More raw corners alone is not success.

Use only the existing approved native 2448×2048 uint8 AOT video at
`/tmp/seaqr_aot_pilot_20260927_aeZ0yA/input/pilot_gray8_ffv1_10fps.avi`, SHA-256
`869c37637b68de5eb2c65a6140caebcea58f01833b653a1f2991fec3b16e4d6f`.
Verify all 300 decoded frame hashes, retaining the same 16 source images as the
previous diagnostic. No new media, RAW16, private camera clips, sealed holdouts,
target-label selection, installations, hardware clock writes or service changes.
Do not modify any previous result or production source/configuration.

## Five arms, 21 cases each: 105 pair calls total

| Fixed order | Arm | Feature image scale | Harris S16 gain | Raw Harris output capacity |
|---:|---|---:|---:|---:|
| 1 | Legacy reference | 0.5 | 1 | Legacy default 8,192 |
| 2 | Capacity-only bridge | 0.5 | 1 | Complete-grid 19,866 |
| 3 | Amplitude at half resolution | 0.5 | 16 | Complete-grid 19,866 |
| 4 | Native resolution | 1.0 | 1 | Complete-grid 78,899 |
| 5 | Native resolution plus amplitude | 1.0 | 16 | Complete-grid 78,899 |

Arms 2–5 form the 2×2 factorial. Arm 1 isolates the legacy storage ceiling.
The existing complete-grid allocation formula is
`(ceil(width/8)+1) * (ceil(height/8)+1)`. Capacity is storage, not a tracked-point
budget: the selected-feature limit stays 1,000. The legacy 8,192 ceiling being
reached is expected censored evidence, not an integrity-stop condition. Explicit
complete-grid capacity exhaustion must fail closed instead of treating censored
output as complete. Do not enlarge
capacity or otherwise rerun an arm after inspecting its results.

Gain 16 is a fixed four-bit, power-of-two expansion solely in the post-resize
U8-to-S16 conversion used by Harris. Expected codes are exactly `16 * U8`,
range 0–4,080, safely inside S16. Gain 1 retains the original conversion call.
Source images, U8 proxy images, pyramids, LK image intensities and native
eligibility masks are not amplified. This changes effective Harris selectivity
at the fixed response threshold; it adds neither radiometric information nor
signal-to-noise ratio. No adaptive normalization or gain sweep is allowed.
Check actual converted pixels and exact raw U32 scores, including saturation and
float32-score rounding diagnostics. Safe S16 codes do not prove absence of
internal Harris arithmetic limitations.

Each arm processes the exact previous experiment's case inventory:

- Eight adjacent pairs: current indices 1, 43, 86, 128, 171, 213, 256, 299;
  previous index is current minus one.
- Eight stationary counterfactuals: repeat each selected previous image as the
  current image, retaining distinct frame indices/timestamps.
- Five native generated controls, seed 20260927 and the unchanged generator:
  high-contrast static / translated; low-contrast static / translated; flat
  static. Translation is `(4, -2)` native pixels, without wrapping, fill 128.

For each case, run arms 1–5 in that fixed order. Fresh unchanged estimator
lifecycle per pair; no global VPI cache reset. One worker, 105 calls, no adaptive
selection or retries. Preserve unexpected failures and partial evidence.

## Backend, geometry and unchanged gates

PVA Harris, PVA pyramids and PVA optical flow remain selected; CUDA performs the
half-resolution resize and S16 conversion. Native scale needs no resize, which
must be recorded as `none`, not a CPU fallback. Both geometries fit the frozen
PVA pyramid limit 3264×2048 and documented Harris maximum 3264×2448.

Four pyramid levels and ratio 0.5 remain unchanged. Level sizes are
1224×1024 → 612×512 → 306×256 → 153×128 at half resolution, and
2448×2048 → 1224×1024 → 612×512 → 306×256 at native resolution.
All coordinates/errors used for quality decisions are native-source pixels.

Do not change Harris strength/sensitivity/NMS/windows; LK window, iterations,
status handling and forward/backward policy; masks/quotas; or global-motion
configuration. Permit only the declared feature scale and capacity config
differences and the gain-only conversion statement in a diagnostic in-memory
method. Assert those differences, recover the original source by reversing
the declared edit and observation hooks, and hash all versions.

Retain minimum 30 accepted points, the configured 20% point-coverage threshold,
and all original global fit inlier/coverage/residual limits. The existing
`translation_consensus` sparse-support exception also remains unchanged; record
which coverage path actually accepts a fit rather than implying that every
accepted fit must meet full-grid coverage. Do not introduce a zero-motion fallback.
Changing image scale also changes the native footprint of the fixed LK and NMS
windows. Therefore, raw Harris observations isolate pre-flow feature supply;
downstream flow differences are effects of the complete scale arm, not solely
proof of improved corner texture.

## Measurements and predeclared interpretation gates

Retain each case, including zero features and rejected fits: native/proxy/S16
hashes and statistics; exact conversion check; raw corners and U32 scores;
capacity status; eligibility/selection; flow survival and errors; and the
unchanged global fit. Compare previous-image raw features between real/static
arms, and confirm gain arms have unchanged U8 proxy inputs at a given scale.

For known-motion generated controls and repeated-stationary AOT images, report
a strict diagnostic truth check in addition to the original fit result:

1. Original fit is accepted, with the original support/coverage gates.
2. Estimated translation-vector error is at most 0.1 native pixel.
3. At least 30 accepted interior points remain under the existing 128-pixel
   post-hoc border margin.
4. Interior median individual displacement error is at most 0.1 pixel and
   maximum error at most 0.5 pixel.

Flat input must return zero raw corners; nonzero corners are a retained negative-
control failure, not an execution-integrity failure. Do not count expected flat
unavailability as a failed positive control or a successful motion estimate.
Report all accepted versus interior point counts: the inherited post-hoc mask
also excludes points whose observed current coordinates leave the interior, so
its error figures are conditional on those surviving points. Repeated block
textures can give consistent but wrong flow, hence truth error is required.
For real AOT adjacent pairs no independent camera-motion truth exists; an
accepted fit means internal consistency only, not verified correct camera motion.
Truth checks on repeated images do not establish real stationary-camera accuracy.

Scientific outcomes (including failed positive controls) remain in the report;
they do not trigger tuning or make an otherwise integrity-valid run disappear.
An integrity/runtime error stops execution and preserves partial results. The
top-level execution pass means scope/completeness/integrity, not all control
truth gates or deployment quality. Report effects per case/arm, not just the
best arm or an average that hides failures. No speed/target-recall/false-alarm
claims and no automatic selection or production promotion.

## Evidence and stop condition

Use a new isolated `/tmp/seaqr_aot_factors_20260927_XXXXXX` Jetson directory and
dedicated tmux process. Freeze script, tests, this plan, exact arm inventory and
helper/reference hashes in a manifest before execution. Import the old diagnostic
helper only after its hash is verified; never modify it. Check old baseline
journal identity, input/dependency/runtime identities, clocks and artifacts
before/after. Restore process-local OpenCV thread policy and close resources.

Copy results and compact evidence to local SEAQR. Stop after the 105-call
comparison, independent audit and report. If one arm appears useful, propose
subsequent representative stable-camera positive/negative and jitter/dropout
validation rather than treating this development encounter as generalization.

Primary API references: [Harris](https://docs.nvidia.com/vpi/3.2/group__VPI__HarrisCorners.html),
[conversion](https://docs.nvidia.com/vpi/3.2/python/build/vpi.Image.convert.html),
[PVA pyramid constraints](https://docs.nvidia.com/vpi/3.2/group__VPI__GaussianPyramid.html).
