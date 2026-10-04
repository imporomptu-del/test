# Frozen larger-motion natural-texture controls

2026-09-28 UTC. User-approved extension of the prior 61-call natural-shift test.
Freeze design, code, tests, launcher and source identities before execution.
This is a single-worker PVA feature/flow diagnostic, not detector tuning.

## Exact cases

Reuse previous input images 0,42,85,127,170,212,255,298 from the already approved
300-frame AOT pilot. For each, run these eleven integer shifts in this order:

`(0,0), (1,0), (-1,0), (16,0), (-16,0), (32,0), (-32,0), (48,0), (-48,0), (32,-16), (-32,16)`.

Then retain the five unchanged generated bridge controls (high-contrast static,
high-contrast translated, low-contrast static, low-contrast translated, flat
static). Total **93 calls**: 88 natural-texture pairs and five bridge controls.
There are 80 nonzero natural shifts. Cases are not selected from new outcomes.

The 16/32/48px axis shifts and ~35.78px diagonals bracket the previously observed
26–36px median accepted real-motion regime. They do not exhaust every direction,
subpixel phase, rotation/scale, occlusion, exposure change or actual large-motion
tail. Images are exact uint8 integer copies without resampling/wrapping; exposed
borders receive code128. Copies move original noise with texture, not independent
sensor noise. These controls cannot prove a richer real motion model is correct.

## Frozen processing

Inherit the old natural-shift runner SHA
`79b6d5c288352bfd0c44d7a4fa3b2181eab7277719197c2f0e95243df2b0660e`.
Change only experiment schema/name/date, fresh workspace/artifact identities,
declared shifts/case count, matching group size/cadence, and descriptive text.
Preserve its source-image verification, flow capture, exact raw-byte encoding,
pre-flow support, attrition stages, runtime/clock checks, pair state lifecycle,
cleanup and compression logic. Verify unchanged computational function source
and a bounded source diff before executing.

Sole arm: `half_gain16_complete`; feature-image scale0.5, Harris-only gain16,
complete Harris capacity19866, original selected budget and native U8 LK inputs.
Reuse frozen diagnostic/factor helpers and original method/configuration hashes.
No native-library replacements, alternative backend, tracker settings, global
motion thresholds, clock/power controls, or production files are changed.

Source workspace `/tmp/seaqr_aot_pilot_20260927_aeZ0yA` and reference factor result
`/tmp/seaqr_aot_factors_20260927_5FZmd7/result.json` remain read-only. Existing
source checks decode/verify the approved 300-frame packaged pilot and retain
the same selected previous images; no new source media is transferred. Metadata
may be read for frozen identities/timestamps, never labels for choosing cases or
scoring targets. SEAQR sealed holdouts, private camera media and RAW16 untouched.

## Evaluation and safeguards

Keep the inherited global-motion and stricter per-point scientific truth gates
unchanged; report them separately. A successful script means integrity/completed
calls, not accurate point tracking or successful target detection.

Preserve exact forward/backward/status byte arrays, finite accepted pairs, and
the 128px pre-flow source/expected-destination support cohort. Keep all selected
and fixed-support denominators, missing/rejected points, disjoint status/filter
attrition, and accepted truth errors. Never count failed-status zero vectors as
correct tracking or summarize only survivors without reporting losses.

Report all eleven shifts across all eight source images: completed calls, global
acceptances, strict truth passes, global-vector error, accepted point error
median/max, fixed-support survival and counts within0.1/0.5px per full cohort.
Do not pool away source-image failures. Prior results are reference evidence,
not an extra opportunity to tune thresholds or choose a favorable subset.

Run as serg, one worker, in a new
`/tmp/seaqr_aot_large_shifts_20260928_XXXXXX` directory and dedicated tmux socket.
Never overwrite existing results/logs or interfere with another task. Verify
runtime/dependency/input/config/clock identities before and after. OpenCV's
process-local thread limit2 is restored on exit. No sudo or reboot.

Copy only new bounded experiment code/configuration to the Jetson; transfer
compressed result plus compact receipts back locally, verify decompressed hash,
preserve raw remote results, and inspect outputs. Use no unbounded parameter
sweep or automatic retry after a data/algorithm/runtime failure. A failed run
is evidence; diagnose it before any further authorization or amended design.

Generated tests cover exact eleven-shift order/count, integer copies/signs/no
wrap, border support, unchanged computational functions, group invariance at11,
source/manifest scope and no-overwrite. Independently spot-check numerical
outputs/denominators. Output folder: `outputs/seaqr_aot_pilot_20260927/large_shifts_01/`.
This test does not fit/promote similarity/affine or rerun the detector.
