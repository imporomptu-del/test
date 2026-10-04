# AOT pilot scoring policy v1 — fixed before detector outputs

This is an independent, descriptive **point-observation** evaluation of one
development convenience sample. It is not official AOT AFDR/EDR, a classification
test, an independent validation split, a nighttime benchmark, or a deployment
accuracy claim. No confidence threshold, detector setting, label, or sample is
selected using detector results. No production module is imported or changed.

## Fixed inputs and timing

The pilot is AOT `part1/00bb96a5a68f4fa5bc5c5dc66ce314d2`, source frames 3–302,
mapped in order to detector input frames 0–299. All 300 frames are required.
Manifest SHA-256 is
`425b8220417e9853a8fbf272a1119d4e1989678984c84f295c9aeb2507b72d8d`.
The pixel-verified gray8 FFV1 video SHA-256 is
`869c37637b68de5eb2c65a6140caebcea58f01833b653a1f2991fec3b16e4d6f`.

Only the native 2448×2048 coordinate system is scored. The detector journal uses
nominal 10 Hz timestamps, exactly `input_frame * 100000000` ns. Original 19-digit
acquisition timestamps remain strings in the manifest and in object histories;
they are never rounded through floating point. Packaging did not preserve exact
acquisition timing in the video. Missing, duplicate, reordered, malformed,
nonfinite, or additional journal frames cause failure, not invented absence.

Two denominator views are fixed:

- `all_300`: input frames 0–299, including initial warmup.
- `fixed_eligible_8_299`: input frames 8–299, excluding only the eight initial
  warmup frames specified before the run. This is not a runtime-ready filter.

Every frame in each view remains in its denominator, including later warmup,
rejected/reused motion estimates, PVA failures, resets, and no-search-support
frames. Availability and motion diagnostics, including affected labeled object
annotations, are reported alongside scores. Such failures cannot be silently
removed to improve the result. Their presence also limits causal interpretation
of a miss: this report cannot alone blame the detection stage.

## Labels and matching

Publisher boxes are `[left, top, width, height]`. They are approximate support,
especially for tiny targets; box area is not exact object pixel support. The
published labels supply evaluation context, not a classifier output.

The primary gate is the closed box: `left <= x <= left+width` and
`top <= y <= top+height`. Both edges are inclusive; there is no rounding,
resizing, center-radius conversion, interpolation, or implicit tolerance.
The one predeclared sensitivity gate expands every edge by exactly **3 pixels**.
Both gates are always reported; the better one is not selected after the run.
The sensitivity is not a repaired or tuned primary result.

Within each frame and stage, maximum-cardinality **one-to-one** matching is
computed across all labels. Each observation and label can participate in at
most one match. Labels are ordered lexically by publisher ID; observation edges
are ordered by distance to box center and then full identity; deterministic
augmenting paths find maximum cardinality. This is not a globally minimum-distance
or physical-identity oracle. Multiple possible observations, or observations
gated by multiple labels, are explicitly marked ambiguous. GT polarity is not
inferred, so bright and dark observations compete for the same airborne label.

Matching is performed against all labels **before** computing these strata:

- `all_labeled`: every publisher airborne annotation, regardless of range/class.
- `tiny_box_area_le_100`: original box width × height ≤100 px².
- `known_range_le_700m`: finite, nonnegative publisher range ≤700 m.

These strata overlap. Missing/unknown range is not zero; far and unknown-range
objects remain labeled objects, never negative background. Strata report the
number of labeled annotations, matched annotations, missed annotations, and
their descriptive match fraction (null for an empty denominator).

## Observation stages and unmatched context

Stages are scored independently:

1. `all_measured`: current track states with `measured == true`, using only
   `measurement_source_xy`.
2. `qualified_measured`: the subset also having `qualified_moving == true` in
   that same row. This mirrors the measured observation-output eligibility;
   qualification is not an airborne-class decision.
3. `raw_candidates_diagnostic`: current proposals' `source_xy`, before track
   qualification. This is a separate diagnostic, not pooled with measured tracks.

The tracker `source_xy` is a filtered/predicted state, even when a track is
measured; it is never substituted for `measurement_source_xy`. Coasts
(`measured == false`) are counted separately, including qualified coasts, but
never matched as current detections. A coast carrying a non-null current
measurement is malformed and rejected.

Unmatched observations are separated into those inside any label gate (e.g.
extra observations around one target) and those outside every label gate.
Neither is automatically a physical false aircraft. They are **unmatched scene
observations/candidates in publisher-label context**. No whole-scene precision,
specificity, true-negative pixel count, or classification accuracy is reported.

The 105 publisher-empty frames provide only 10.5 seconds of nominal negative
exposure in this pilot. Their observation count, frames with observations,
observations per empty frame, and distinct segment/polarity/track identities
are reported. A track-state count is not an object count. No hourly false-alarm
rate, independent-trial confidence interval, or general false-alarm claim is
extrapolated from this short correlated interval.

## Histories, first hit, and fragmentation

GT identity is scoped by the report's part and sequence plus publisher object
ID. Tracker identities retain **segment + polarity + track ID**; reused IDs in
new segments and opposite-polarity IDs are never merged. Each object history
records exact source frame/time, matched identity, current source coordinate,
all gated identities, and ambiguity. Candidate indices are only local per-frame
identifiers and never treated as temporal tracks.

The first hit is reported only if the earliest assigned match is unambiguous.
Its delay is relative to the first labeled frame **within that reporting view**,
with nominal seconds and exact source-nanosecond difference. This is not physical
appearance/onset latency; a target may already be present at the sequence start.
No hit is null/censored, not an infinite or fabricated delay. An ambiguous
earliest match yields no first-hit claim.

Identity fragments are counted only for measured-track stages when the object's
GT is contiguous within the view and no gate ambiguity occurs. A fragment starts
at a matched identity after a miss or a change of segment/polarity/track ID.
Additional fragments are `max(fragment_count - 1, 0)`. They describe association
continuity under this geometric rule, not proven physical identity switches.
Ambiguous or gapped GT yields null fragmentation; raw candidates never receive
a fragmentation score. Misses are not bridged with coast positions.

## Freeze and execution checks

`scripts/score_aot_pilot.py freeze` reads only the manifest and scorer/test/policy
source files, then writes a new receipt binding their SHA-256 values and this
policy, plus the input-only runtime harness's source SHA-256. It never opens
detector journals or media. The root agent must review
the policy and generated test result before invoking `score --review-approved`.
The flag records workflow authorization; it is not a security credential.

Scoring checks the pinned manifest/video/configuration/baseline identities,
launch and successful report consistency, nominal cadence, native size, complete
decode with zero drops, and the frozen PVA/CUDA backend choices. The separate
combined-stack execution receipt must also be checked; a bare-config baseline
run is not interchangeable with the approved frozen implementation stack.
`execution_receipt.json` and `preflight.json` default to siblings of the run
directory (or explicit `--execution-receipt` and `--preflight` paths). They must
bind the same input, scoring freeze, runtime harness, frame journal, launch, and
successful report. Checks include the five native-library and twelve adapter
identities; VPI 3.2.4; GPU-only 300-frame lifecycle; native tracking fallback
counters; all 299 adjacent PVA attempts; cleanup; reference execution order;
NumPy/OpenCV/BLAS/thread policy; and unchanged read-only clock-policy snapshots.
Expected PVA motion unavailability is allowed but retained as coverage context,
not discarded. This scorer independently verifies the receipt fields; the pinned
harness additionally validates detailed stage ownership/timing invariants.

Receipt hashes are saved with the result. Inputs are rehashed after scoring to
reject concurrent changes. Any script/test/policy/harness change after
freeze invalidates that freeze and requires explicit review before evaluation.

The scorer has no network operations, media decoding, detector imports, model
training, threshold search, production edits, or default output overwrite.
Generated fixtures exercise coordinate semantics, one-to-one assignment,
ambiguity, stage separation, coverage, provenance, and fail-closed handling.

Example, after root review (paths must identify the completed approved run):

```sh
.venv/bin/python scripts/score_aot_pilot.py freeze --manifest /ABS/PILOT/frozen_image_manifest.json --output /ABS/NEW/scoring_freeze.json
.venv/bin/python scripts/score_aot_pilot.py score --manifest /ABS/PILOT/frozen_image_manifest.json --freeze /ABS/NEW/scoring_freeze.json --run-dir /ABS/APPROVED_RUN --output /ABS/NEW/scoring_result.json --review-approved
```
