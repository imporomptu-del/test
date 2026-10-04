# V39: accuracy-only continuity and model study

Efficiency, GPU/PVA performance work, RAW16 and the old dense scan are paused.
Do not access sealed holdouts. This development experiment may read the four
previously allowed clips' saved journals and V38 saved current-source patches;
it changes no production configuration, detector, assignment or learning.

## Hypotheses and exposure

The exposed compact-light example has in-radius measurements on eight visible
frames but baseline qualification on six. V36's unpromoted spatial filter
keeps four. Existing records suggest two distinct losses: motion qualification
and point-shape/centering mismatch. Their causes and remedies are not assumed
equivalent. The example's physical class is unknown; it is not airborne truth.

This plan is informed by that development failure. No claim of blind selection
or an independently optimized parameter choice is made. Freeze code, constants,
tests and reference inputs before full four-clip replay or real-patch scoring.
Do not sweep settings or amend the candidate after seeing its replay results.

## A. Bounded measured continuity candidate

Preserve all baseline-qualified outputs exactly, including clearly distinguished
predictions. Add an explicit **quality-degraded measured** state only when:

- The same segment/ID previously had an actual baseline-qualified measurement.
- Every intervening state is present and actually measured; a coast, reset,
  absent ID or other qualification failure breaks this lineage.
- The current quality check is ready but failed, while confirmation and the
  unchanged 12 px excursion requirement are still satisfied.
- No more than two source frames **and** 200 ms elapsed since the last actual
  baseline-qualified measurement. Both budgets apply; degraded states never
  refresh either budget.

The two-frame/200-ms budget is a bounded development hypothesis for brief
quality interruptions at nominal 10 Hz, not a calibrated optimum or guarantee.
It was chosen before this replay, with the two-frame example already known.
Test variable frame intervals, reset/coast/reused IDs, long failure runs,
alternating quality, malformed input and transaction atomicity synthetically.

A degraded track is **not confirmed**, not an airborne object and not a new
detector hit. Keep the confirmed-only and confirmed-plus-degraded metrics
separate. Do not count additional renderable states as improved airborne recall.
Because this candidate only adds states, measure the added workload on all four
clips and the provisional control windows; no-new-loss alone is insufficient
for promotion. No arbitrary image-score gate is attached to this experiment.

## B. Shape/localization study

Compare the frozen V38 point/edge family with a declared source-driven joint
edge-plus-compact model. Keep isotropic, elongated and paired families separate;
report nonlinear bank sizes, location ambiguity and boundary solutions. A
better maximum fit from a more flexible family does not establish a target.
Start with deterministic synthetic point-on-edge, off-center, paired, nearby
distinct-source, curved-cloud, exposure and noise cases. If mechanics pass,
apply the frozen study uniformly to all 364 saved V38 current25 patches, not
only the rejected compact-light samples. Never give reference coordinates to
the model or select the best temporal lag after inspecting outcomes.

## C. Stage-specific validation

Use all 2,741 saved development frames, preserving original source/detection/
association records and exact old reference definitions. Report candidate,
actual measurement, baseline-qualified, confirmed-plus-degraded and existing
V36 shadow-filter coverage separately for dense 285, overlapping pilot 28,
anchors 24 and new compact-light 8 samples. Unknown visibility is not absence;
predictions never count as measured hits. Report alternative matches rather
than claiming physical identity from a proximity gate.

Keep the five ambiguous compact-light frames and unreviewed full-frame space
out of true/false-positive denominators. Replay may quantify workload and known
sample losses, not population accuracy. Independently check changed states and
summary arithmetic before considering an opt-in integration or new videos.

## D. Broader validation requirement

Inventory existing class/visibility provenance without inspecting holdouts.
Predeclare any later source-review windows before scoring them. Airborne
accuracy requires independently supported airborne positives and exhaustively
reviewed non-target scope. If those labels are unavailable, state that missing
requirement and prepare a controlled-recording specification; do not invent
labels or describe another development-clip replay as generalization.

Deliver the frozen candidate, tests, stage report, model-study results and a
clear promotion/rejection decision. A failed candidate remains a useful result
but must not silently become production behavior. This turn does not promise
completion of operational airborne validation without its required data.
