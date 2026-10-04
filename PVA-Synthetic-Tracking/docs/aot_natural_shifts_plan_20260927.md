# AOT natural-texture shifts — frozen diagnostic plan

2026-09-27. User approval follows the feature-factor result: amplified Harris
input restored corners and passed repeated-image/generated-block controls, but
all eight real adjacent pairs still failed the original global-motion gates.
This bounded test asks whether that same frontend accurately tracks known small
translations of the actual image textures. It does not promote an algorithm.

## Fixed input and 61 calls

Use only the approved 2448×2048 uint8 AOT pilot video already on the Jetson at
`/tmp/seaqr_aot_pilot_20260927_aeZ0yA/input/pilot_gray8_ffv1_10fps.avi`.
SHA-256: `869c37637b68de5eb2c65a6140caebcea58f01833b653a1f2991fec3b16e4d6f`.
Reuse the frozen decoder/hash validation for its 300 frames and the original
selected images; do not select images or shifts using annotations or results.

Use previous input indices, in order: **0, 42, 85, 127, 170, 212, 255, 298**.
For each, apply these seven ordered native-pixel displacements to a copy:
**(0,0), (1,0), (-1,0), (0,1), (0,-1), (4,-2), (-4,2)**.
Positive x is right; positive y is down. Use the frozen exact integer
`translate_no_wrap` copy and fill newly exposed borders with 128. No host
resampling, wrapping or interpolation. The unchanged half-scale CUDA resize
still occurs downstream: a one-native-pixel shift tests half-proxy-pixel phase.

These 56 pairs are followed by the unchanged five generated bridge controls:
high-contrast static / translated, low-contrast static / translated, flat static;
seed 20260927, blocks 32 pixels, high mapping `16 + 14 * pattern`, low mapping
`120 + pattern`, translation (4,-2), fill 128. Total: **61 calls, one worker**.
Each call uses a fresh estimator and nominal 100ms previous/current metadata.
Synthetic current pixels are transformed previous pixels, **not the next AOT
frame**. Preserve original source identity separately. No global VPI cache reset.

## Sole arm and unchanged quality gates

Use exactly the previous `half_gain16_complete` arm: feature scale 0.5,
Harris-only S16 gain 16, complete-grid raw corner capacity 19,866. Selected
feature budget remains 1,000. Native inputs, LK/pyramid intensities and source
information remain 8-bit. Gain changes Harris selectivity, not information/SNR.

Reuse the two hash-pinned diagnostic helpers without editing them. Reuse the
frozen baseline harness, runtime/dependency checks, original generated motion
method and reversible gain16 observation transformation. PVA Harris, PVA pyramid
and PVA flow remain selected; CUDA resize/conversion remain unchanged.
Do not change masks, quotas, Harris/LK settings, status interpretation,
forward/backward checks, displacement limits, motion fit or coverage policy.
The existing translation-consensus sparse-coverage exception remains in force;
report its acceptance path rather than implying an unconditional 20% coverage.

The inherited strict known-motion check is retained unchanged:

1. Original global fit accepted.
2. Estimated vector within 0.1 native pixel of truth.
3. At least 30 accepted interior points under the existing 128px margin.
4. Interior median error ≤0.1px and maximum error ≤0.5px.

Flat input must have zero raw corners. Scientific failures are retained outcomes,
not permission to tune, retry or discard a case. These are observations, not
top-level execution-integrity pass conditions.

## Lost-point and identity audits

The inherited truth metric is conditional on accepted points and observed
current-position bounds. Keep it for exact bridge comparability and disclose
that conditioning. Separately define a fixed cohort **before flow rejection**:
all selected previous points within the valid source overlap after a 128px
margin, using previous p and expected p + displacement, never observed q.
Equivalently x lies in `[max(0,-dx)+128, min(width,width-dx)-128)` and similarly y.

For this fixed denominator report finite/nonfinite forward flow, status values,
forward/backward and final acceptance losses, direct truth errors and accepted
truth successes. Missing/invalid flow is a failed/missing observation, not zero
error. Rejection-stage categories must be disjoint and sum to the cohort.
This descriptive audit adds no quality threshold or new acceptance rule.
The frozen legacy policy passes the forward status object into backward flow,
then reads both status arrays after the bidirectional work. Therefore, reported
status categories describe those final readbacks and the original acceptance
filter; they are not independently timed causal forward/backward failure counts.

Preserve bounded raw flow/status bytes with dtype/shape/hash metadata alongside
the readable JSON view so NaN/Inf and signed-zero representations can be audited.
Validate byte counts and SHA-256. Preserve all original corner/score/selection
observations. Across all seven shifts per image assert identical previous
native/proxy/S16 images, raw Harris points/U32 scores and selected indices.
If those vary, stop as state/nondeterminism confounding, not a motion effect.
Report transformed current-image/proxy identities separately.

## Scope, provenance and delivery

Use a fresh `/tmp/seaqr_aot_natural_shifts_20260927_XXXXXX` workspace and dedicated
tmux session. Freeze script/tests/this plan and fixed design in the manifest
before execution. Hash both helpers, original input/harness/journal, frozen
dependencies and the prior feature-factor result. Check identities before/after.
Do not parse the old result to select cases. No overwrite of existing evidence.

No new media, private camera clips, sealed holdouts, RAW16, target detector,
alternative motion model, threshold relaxation, production source/config edit,
clock change, installation, reboot or service change. Restore process-local
OpenCV thread policy and close each estimator, including failure paths. Preserve
partial evidence on an integrity/runtime error; no automatic retry.

After closing raw JSON, produce an exclusive lossless gzip copy and verify its
decompressed hash equals raw JSON. Preserve raw remote evidence. Transfer and
verify local full evidence; place compact summary/audit and report in SEAQR.
Stop after this fixed run and independent audit.

## Interpretation limits

Failure on known translated natural textures points to feature/flow limitations
even when camera motion is a perfect translation. Passing narrows the remaining
real-pair problem but does not prove that the global model is its sole cause.
Synthetic copies move existing sensor noise and artifacts along with the scene;
they omit independent noise, exposure changes, occlusion, blur changes and
parallax. Thus this is neither a real stable-camera validation nor evidence of
airborne detection accuracy, target recall, false-alarm rate or real-time speed.
Keep per-image/per-shift results, including failures, rather than only averages.
