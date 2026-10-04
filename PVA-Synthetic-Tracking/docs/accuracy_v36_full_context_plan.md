# v36 full-development-clip context shadow, pre-extraction plan

This bounded follow-up tests the fixed point-versus-edge diagnostic over the
complete baseline-qualified output, not only selected reference/control samples.
It does not rerun or change detection, tracking, association, learning feedback,
v34/v35 defaults, or the frozen diagnostic. No RAW16, holdout media, remote work,
threshold sweep, classifier promotion, or processing-FPS claim is permitted.

## Scope and prerequisite evidence

Require the completed independent audit of `context_01`, bound by schema,
experiment path, freeze hash, auditor source hash and every audited evidence
file. Recheck the bounded diagnostic's frozen implementation, selection,
observations, patch archive, summaries, tests and parent inputs. Independently
verify the v34 manifest and first-repeat journals for exactly four existing
8-bit development clips: 0029 (687 frames), 0126 (674), 0055 (689), and 0082
(691), totalling 2741 frames. The source files are the already-local original
AVIs, checked by full SHA256 against the audited run before and after extraction.

Freeze the gate, runner, unchanged feature function, scoring dependencies,
tests, this plan, sources and all evidence before evaluating any source pixels.
Run unit tests before extraction, save their log, and snapshot all code. An
existing output directory is an error; incomplete outputs remain for diagnosis.

## Causal output-only gate

Sequentially decode every source frame from frame0 through the complete frozen
journal, checking 10fps metadata, native4784x3190 uint8 dimensions, exact journal
frame/timestamp continuity, expected counts and decoder EOF. Never seek, skip
an inconvenient frame, resize, stretch contrast or insert a source-location
mask. Convert each decoded BGR frame to grayscale with recorded OpenCV build.

Call `CausalPointContextGate.update(row, gray)` with no labels or reference
coordinates. Baseline-qualified measurements use actual logged measurement
coordinates, nearest native pixel rounding and the same25x25 point/edge feature
function. The threshold remains the declared positive point-minus-edge margin
of zero. Missing/truncated/uninformative evidence conservatively retains the
baseline output and receives an explicit reason; it is not silently treated as
a negative. Predicted states inherit only that identity's latest measured gate
decision, never a future/current invented observation. Missing history also
retains baseline with an explicit unavailable reason. Segment and deletion
boundaries must not leak decisions between identities.

The gate can remove baseline-qualified outputs but cannot create measurements,
new IDs or newly qualified states. Assert that the complete input journal row
is unchanged by every update, and validate identity coverage and measurement
provenance. No gate decision feeds back into the frozen detector or tracker.

## Evidence and evaluation

Save one compact JSONL record for every frame containing **all** baseline-
qualified states, their measured/predicted status and coordinates, acceptance,
reason, measurement-frame provenance and diagnostic features. Do not copy the
huge original journals. Save per-frame baseline/candidate measured and predicted
workload, aggregate ID counts, evidence availability and reason counts separately
for measurements and predictions. A coast retains measurement-frame provenance,
not another copy of the old patch features; report inherited decisions separately
from missing history. Uninformative/border/history cases remain
visible in these records. Decoder identity and EOF checks are recorded.

Crosscheck all358 previously audited bounded observations during the new full
sequential decode: identity, actual measurement coordinates and native patch
bytes must match their saved hashes exactly. Feature inventories and informative
status must agree. Informative patches must preserve the bounded ablation's
decision; uninformative patches follow the new explicitly conservative unknown
policy rather than the diagnostic's zero-coded failure. All358 currently frozen
patches were informative. Floating features use the independent audit's
fixed numerical validation tolerances (relative2e-10, absolute2e-8), never an
acceptance-margin tolerance. This check is after each gate decision and cannot
change that decision or recenter a patch.

After decisions are produced, use the existing frozen scorer on sparse rows
for the285 dense references and28 stricter pilot samples; evaluate the24
required original anchors separately. Preserve all alternative matches and
one-to-one assignment behavior. The baseline must reproduce the **entire**
audited shadow baseline, including284/285 dense hits,28/28 pilot hits,24/24
anchors, assignments, and workload. The seven fixed provisional control
ROIs/times must reproduce70 baseline-qualified measured states. No reference,
control window, tolerance, identity or denominator may be edited.

Report any new visible-frame misses, changed assigned identities, lost required
anchors or loss of common anchor identity explicitly. Such a result is a failed
retention gate, not a reason to discard outputs or rerun with adjusted settings.
Also report before/after measured/predicted states and distinct IDs over all four
clips. Full-frame unlabeled workload reduction is not a false-positive-rate or
airborne-accuracy measurement. Adjacent frames and overlapping references are
correlated;055/082 do not acquire negative labels from having no known targets.

Rehash every bound input, source video, live implementation, copied snapshot,
unit log and output artifact before writing a completed summary. No automatic
promotion occurs even if all retention checks pass and provisional controls
decrease. This downstream shadow cannot reduce the original active-track cap
pressure or recover births previously denied by that cap. Retaining known
reference samples does not establish sensitivity to hidden faint objects.
Independent auditing, source review of newly removed responses,
representative truth and closed-loop validation remain separate next steps.
