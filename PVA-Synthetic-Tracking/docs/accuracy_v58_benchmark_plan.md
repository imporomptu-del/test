# V58 — prepare a class-supported 8-bit benchmark

Status: **preparation complete; footage/provenance and annotations still needed**.
This is not a completed dataset, an accuracy result, or permission to inspect
new media. The detector is unchanged. Efficiency, RAW16 and the old scan remain
paused; the existing sealed holdouts remain untouched.

## What to provide first

Start with **one native 8-bit recording of a known airborne target** plus
independent context that connects that target to the recording and time. A
synchronized wider-view video or a time-linked operator log with unambiguous
source correspondence can support class; a moving dot or detector box cannot.
Existing footage with this context is welcome—new recording is not assumed
necessary or available. We have not requested or initiated a flight.

For the first intake, provide:

1. The exact video path or file, not a directory to scan.
2. What the target is, approximately when it appears, and the independent evidence
   establishing that it was airborne. State uncertainties rather than guessing.
3. The recording session/date and camera context, including whether this footage
   or its detector overlay has already been reviewed.
4. Original capture timestamps/settings if available. Missing metadata can be
   recorded as missing; it must not be replaced with invented camera timing.

The machine-readable [intake manifest](../configs/evaluation/accuracy_v58_benchmark_intake.json)
keeps all real recordings, encounters and negative scopes **empty** until supplied
and reviewed. Its named coverage slots are proposals, not acquired examples or
class labels. `ready_to_score` is deliberately false.

## Current evidence: useful, but not airborne truth

The allowed development clips are 0029, 0126, 0055 and 0082. Their current
annotations establish visible image features, not independently verified
airborne status. They also provide no authoritative airborne-negative exposure.
That does **not** mean the videos contain no airborne objects.

Keep all historical reference panels unchanged: dense285, pilot28, anchors24,
compact-light8 and grid11. These356 samples overlap; they are not356 independent
encounters. The old clips are development material and cannot become an
untouched test just by receiving new labels. The later V40 review adds one
visible-feature sample in0055; older inventory statements saying0055 has no
positive reference are no longer current, but its physical class is still unknown.

Relevant evidence:

- [Airborne-only protocol](phase20_airborne_accuracy_protocol.md)
- [V39 provenance inventory](accuracy_v39_validation_inventory.md)
- [V40 coverage results](../results/tiny_target/accuracy_v40_20260925/README.md)
- [V57 rejected cleanup rule](../results/tiny_target/accuracy_v57_20260926/README.md)

## Bounded coverage pilot

The proposed development pilot contains six airborne encounters, covering clear
sky/cloud edges, straight/turning motion, slow apparent motion and changing
visibility. These are desired conditions, not instructions to fabricate or force
a maneuver. If conditions are unavailable, record that gap. Do not substitute
only easy, high-score examples. Record upper/middle/lower image coverage rather
than assuming airborne targets are confined to the upper rows.

Separately include documented fixed-light brightness changes, ground-vehicle
lights and structured-cloud/camera-motion examples. **A labeled ground vehicle
does not establish that no airborne target exists elsewhere in that frame.**
These examples become authoritative negatives only for an exact, adequately
observable, exhaustively reviewed space/time scope. Darkness, obscuration,
uncertain moving lights and a detector's silence never establish absence.

This is a coverage pilot, not statistically sufficient population validation.
Repeated passes from the same session or target are correlated. Reserve a
separate later recording session for validation before tuning; adjacent chunks
and parts of one encounter must not cross development/validation splits. Do not
inspect that reserved material without a separately authorized evaluation.

## Recording and timing checklist

- Retain original native resolution, bit depth, pixel format and codec; do not
  replace originals with cropped, resized, enhanced or overlay-rendered videos.
  Store source hashes and preserve complete encounter context.
- Aim to retain at least five seconds of pre-roll and post-roll when available,
  plus the entire event. This is a collection target, not a detector parameter
  or a guarantee that warm-up is sufficient. Censored starts/ends are explicit.
- Record camera/optics/focus, exposure/shutter/gain and automatic-control changes,
  camera movement and scene conditions. Unknown settings stay unknown.
- Preserve capture timestamps and drop/duplicate accounting if available. A
  nominal10fps AVI time base is not proof of a10Hz sensor acquisition cadence.
  Without reliable capture timing, frame-based coverage may still be reviewed,
  but do not claim physical delay, real-time performance or accurate time exposure.
- Document how the independent class evidence aligns with source frames, including
  clock offset and uncertainty. An unsynchronized note alone may be insufficient.

## Annotation, before overlays

1. Declare the source hashes, recording sessions, exact review scope and reviewer
   exposure. Select encounters using recording context/source review, not detector
   proposals. Disclose any prior overlay exposure; never relabel it as blind.
2. Review every frame in each declared encounter region at native pixel sampling.
   Overview context may help establish class, but cannot establish absence of a
   tiny target. Nearest-neighbor enlargement may repeat pixels; do not invent detail.
3. Record source-coordinate positions and uncertainty only when confidently visible.
   Keep visible, uncertain, occluded, out-of-view and source-frame-unavailable states
   distinct. No interpolated coordinates or predictions become observed truth.
4. Keep event class (`supported_airborne`, `supported_non_airborne`, `unknown_class`)
   separate from frame visibility. Record physical identity only when supported;
   several nearby dots or tracker IDs do not establish object count or ID switches.
5. A second source-only review should adjudicate class, hard visibility/identity
   cases and negative scopes. Preserve disagreements and unresolved intervals.
6. Version and hash the approved labels before examining baseline/candidate scores.
   Any later correction requires a new revision and disclosed reason.

## Baseline-first scoring contract

Before the first run, bind exact source/label hashes, actual runtime code and
configuration, assignment rules, gates, processing coverage and alarm-event
definition. The manifest's runtime/label hashes are null because no new footage
or annotations exist yet. Do not pretend the old journal replay validates a new
source decode or runtime. Use the unchanged working baseline first.

Report separate stages: candidate present, actual associated measurement,
qualified measurement, coast-only continuity and displayed output. Missing
processing frames must remain visible in denominators; predictions must never
repair actual-measurement recall. Report new losses and recoveries individually,
not only net totals. Preserve old panels with their original gates and assignments.

For new annotations, freeze uncertainty/gate and one-to-one matching rules before
scores; report alternatives/ambiguity rather than assuming nearest equals physical
identity. Do not tune match radii after finding misses. Encounter detection and
tracking continuity require complete visibility context, not a favorable sparse
anchor. Onset delay requires independently established, uncensored onset and
trustworthy timing; otherwise mark it unavailable.

False alarms require a frozen emitted-alarm/event definition and verified negative
scope. Do not count every rejected candidate or every track-frame as an alarm.
Use the union of eligible reviewed space/time to avoid counting overlapping
windows twice; do not extrapolate crop exposure to a full-frame rate. Precision,
ID switches and duplicate-object rates remain unavailable without their requisite
physical-object labels.

Operational acceptance limits are deliberately unset. Agree on encounter
detection, false-alarm events, delay and identity requirements before selecting
an accuracy/speed trade-off. The pilot alone cannot justify deployment or a claim
that unseen airborne objects will always be detected.

## Work sequence and stop conditions

1. Obtain an exact source and independent class context from the user/operator.
2. Confirm its explicit access scope, record immutable provenance and session split.
3. Build and adjudicate source-only labels; leave uncertain intervals unknown.
4. Freeze the baseline and scoring contract, then run stage-specific evaluation.
5. Choose one demonstrated failure mechanism to address, retaining the historical
   regression panels and all new misses. Validate a frozen candidate on the
   separately authorized reserved session later.

Stop before new scoring if class evidence, labels or scope are missing. Do not
reopen the old holdouts, resume RAW16, tune V57 or claim an empty benchmark passed.
