# V56 — frozen production candidate diagnostics

The user approved adding diagnostic logging and replaying the unchanged pipeline
around the existing chunk0126/frame216 candidate-stage absence. This is a
previously exposed, bounded causal diagnosis, not blind evaluation, a detector
repair, threshold search, production promotion or measured accuracy gain.
Efficiency work remains paused.

## Fixed input and unchanged inference

Use only the already-exposed 8-bit MJPEG AVI
`/home/serg/project/camera_reader_sky/srcsky/chunks/chunk_0126.avi`.
Its frozen SHA256 is
`c5302b873656793da47f1da3c03f05df595f17c3f9bc407ce0bfd99b7e718344`.
The archived launch declares 4784 by 3190 pixels, 674 frames and container
playback 10 fps; physical acquisition cadence remains unverified.

Run two independently initialized, complete replays, clean then probe, using the
unchanged V29 combined arm and serial_reference execution policy. Both process
all 674 frames, indices 0 through 673, from the original segment start, in the
original order and with the same frozen decode, motion, stabilization,
background, scoring, quota, shape, association and qualification settings.
Do not crop decoding, shorten history, skip to frame213, load a future state,
change processing defaults, tune settings or substitute a newly compiled
production scorer. V54 temporal-median experiments are not this production
background model. Original production library and source identities must remain
pinned; a separate read-only bridge may capture existing native state.

Never decode, inspect or otherwise access RAW16 or sealed holdout clips. Only
the explicit source AVI and explicitly scoped configuration, code, native
libraries and compact provenance are authorized inputs. Paths mentioned inside
old receipts are not an expanded input allowlist. This plan/manifest preparation
does not itself require media or remote access.

## Fixed references and controls

The machine-readable contract is
`configs/evaluation/accuracy_v56_diagnostic_probes.json`. It pins compact source
hashes, the source AVI identity, existing six reference positions and all seven
original provisional control scopes.

Capture the six existing dense-panel/window126 references at frames213–218,
sample indices311–316 and evidence indices151–156. These are three preceding
successes, the frame216 miss and two following successes. There is no existing
frame219 reference. Original matching uses the unchanged **7 px source-coordinate
gate**. Its 5 px annotation position uncertainty is a separate field, not a new
gate or a reason to increase the matching radius.

| Frame | Source XY | Original candidate | Original strict assignment |
| --- | --- | --- | --- |
| 213 | 3137.77, 2712.46 | candidate:52 | 0/bright:1001 |
| 214 | 3142.48, 2709.06 | candidate:36 | 0/bright:1001 |
| 215 | 3147.16, 2705.95 | candidate:53 | 0/bright:1001 |
| 216 | 3152.44, 2701.83 | null | null |
| 217 | 3162.59, 2695.84 | candidate:51 | 0/bright:1001 |
| 218 | 3163.59, 2694.22 | candidate:42 | 0/bright:1001 |

All references have bright polarity; their airborne physical class remains
unverified. Preserve original sample IDs, assignments, alternatives, uncertainty
and absent values. Do not assign bright:1001 at frame216 merely because adjacent
references have that identity, select a more favorable alternative, or recenter
the diagnostic probe on newly emitted candidates or tracks.

Evaluate all seven original provisional controls from the **complete clean and
probe frame journals** using their unchanged inclusive frame windows, full
192 by 192 source crops and original scope predicates. Preserve scope identity
as (clip, frames, crop, label); the original compact summaries do not provide a
separate control ID. Preserve their original ordering. The recorded baseline
totals are 70 measured and 64 predicted states. The two zero-workload scopes
remain included. V36's separate shadow counts are provenance only, not a combined
production cleanup result. These scopes are workload controls, not authoritative
airborne negatives; no false-positive rate may be derived from them.

**No tile or pixel-map capture is performed for the controls.** Diagnostic native
maps are restricted to the six reference frames.

## Read-only pre-learning capture

At each fixed reference frame, after successful native prepare and before native
finish or any background/noise learning update:

1. Use the actual causal source-to-reference transform already owned by the
   pipeline. Persist that transform, the unchanged fractional source and
   transformed reference coordinates, the rasterization rule and selected
   pixel/tile bounds. Do not estimate a transform from labels or future tracking.
2. Form the fixed radius7 inclusive pixel box around the transformed probe,
   using ceil(center minus radius) through floor(center plus radius), clipped
   to the image with clipping explicitly recorded. Expand to every full
   production 256-pixel tile intersecting that box and add the frozen **2-pixel
   raw-absolute peak-comparison halo**. Retain the 7-pixel circular source gate
   for scoring; the bounding box/tile capture is not a broadened match gate.
   Refuse bounds beyond the frozen 3 by 3 tile and 600,000-pixel capture limits
   instead of silently expanding scope.
3. Capture the current and previous support, readiness/eligibility, current
   image and spatial-filter components, spatial response S, prior background B,
   temporal residual R, tile center and precise/float noise statistics, pixel
   variance, signed threshold margins and actual raw-absolute peak competitors
   for both polarities. Keep unsupported, nonfinite and out-of-bounds states
   explicit rather than reporting a zero score.
4. Preserve each captured full tile's prequota candidates/ranks and verify their
   predicates, counts and top-k ordering against the original native outputs.
   Record quota survival, original shape-member/centroid lineage and later
   candidate, measurement, association and qualification outcomes. Use the
   actual production shape results; do not replace them with inference from a
   truncated diagnostic crop.
5. Keep all captured arrays detached/read-only. The bridge must not change
   original native buffers, thresholds, candidate ordering, learning masks,
   tracker state or selection. Diagnostic values and reference positions must
   never feed detector or tracker inference.

## Freeze, tests and equivalence

Before any real replay, freeze the hashes of this plan, manifest, all new
diagnostic/build/run/analysis/audit sources and tests, original V29 runner and
dependencies, production source/configuration/library identities and explicit
compact provenance. Use a fresh exclusive output directory. Verify the AVI
hash before decoding; do not trust a matching basename alone. Save native
bridge compiler command, environment, source and library hashes separately from
the unchanged production libraries.

Tests must cover fixed scope and hashes, coordinate transforms and rounding,
image/tile boundaries, missing support, native eligibility, both threshold
predicates, raw-absolute peak ties/competitors, prequota order and quota
survival, shape/track identity preservation, failure cleanup and read-only
behavior. Run the relevant test suite and broader regressions before interpreting
real diagnostics. Missing state or mismatched native predicates must fail closed.

Compare the complete 674-frame clean and probe journals for exact inference
equivalence: candidate values/order, measurements, tracked states, association,
qualification, output identities and recorded deterministic decisions. Preserve
original comparison semantics and explicitly identify any excluded timing or
diagnostic-only fields; never use a broad exclusion that can hide inference
changes. Verify relevant state fingerprints when available. Diagnostics introduce
synchronization/IO, so their wall time is not an efficiency comparison.

A clean-versus-probe mismatch blocks a causal claim. A disagreement with the
historical reference must also be reported before treating a reproduced outcome
as the same archived failure. Rehash bound sources and artifacts after execution.
Retain unknowns and missing assignments rather than turning them into failures,
negative evidence or fabricated measurements.

## Decision boundary

Report the earliest directly observed rejection stage at frame216, alongside all
five fixed neighboring successes and the seven unchanged workload controls.
Separate support/readiness, threshold, raw-peak competition, tile/frame quota,
shape measurement, association and qualification. An absent post-quota candidate
alone does not identify a threshold, background, registration or PVA failure.
Do not infer detector behavior from V54's DC-sensitive synthetic projections.

Only after native diagnostic checks and unchanged-output equivalence pass may
the evidence motivate one separately frozen causal repair/comparison. V56 does
not make that repair, change production defaults, establish airborne recall or
precision, or authorize broader media/holdout exploration. If the replay cannot
be performed or exact native evidence is unavailable, state that limitation;
do not present synthetic instrumentation tests as the real clip's cause.
