# V40: source-first coverage and evidence-led nuisance diagnosis

2026-09-25. User authorized the previously proposed four-step accuracy work.
Efficiency, RAW16, sealed holdouts, remote execution and production changes
remain out of scope. Local development media allowlist: 0029, 0126, 0055, 0082.

## Chronology and review boundary

Use the immutable V39 coverage manifest (SHA-256
`cff39ab26a03178e4a350111dae78aef58ea24483e4856582c3663d46f808ebc`).
Its 108 windows were selected before V39 real scoring; V39 aggregate outcomes
are now known. Neither this review nor a later comparison is held-out or blind
to the overall development history. Initial source reviewers must not inspect
window-specific detector outputs, overlays or scores before recording their
source judgments. Existing references remain unchanged.

Freeze the extractor, its tests, this plan, the manifest and verified actual
source hashes before extraction. Decode exact sequential frame indices, retain
only the selected native crops, and show every selected frame unmarked at
native sampling. No contrast enhancement, denoising, interpolation of source
detail or generative image tools. Labels occupy margins outside crop pixels.
Keep all dark, obstructed and uninformative windows; no attractive replacements.

Each reviewer records, for every assigned window: inspected source frames,
scene content, confidently visible movement if any, uncertainty, physical-class
support, and whether any exhaustive negative claim is defensible. "No mover
discerned" is not verified absence; stationary-looking lights over two seconds
do not by themselves establish physical class. Position annotations require
individually visible samples and explicit uncertainty; no interpolation through
unknowns. Review difficulty or lack of class evidence remains explicit.

Divide the four clips among reviewers for complete coverage. A second reviewer
must inspect any proposed moving-feature reference, any proposed supported
non-target or negative region, and any ambiguous source case important to a
policy decision. Preserve disagreements. A review of this grid covers only
2.90% of spatial pixels at 60 selected frames per source, not full-frame accuracy.

## After source records are frozen

Associate existing baseline measured and predicted states with these exact
ROI/time scopes, keeping them separate. Report workload and reviewed scene
evidence without turning unlabeled states into false positives. Where nuisance
class cannot be supported, report that the intended diagnostic is inconclusive.

The predeclared V39 difference-appendix rule may be instantiated separately
(at most two added and two removed measured-output identities per clip, ordered
by first changed frame and identity; ±6 frames, native 256×192, clamped bounds).
This is output-selected diagnosis, never independent coverage or truth. It
requires its own extraction freeze and unmarked source review before overlays.

Choose at most one mechanistically justified general correction only after
source review establishes the evidence needed for that hypothesis. Document
its inference inputs and constants in a separate pre-score candidate freeze;
do not choose thresholds by replay score sweeps or use clip/time/ROI exceptions.
An unsupported hypothesis is not a reason to manufacture a production change.

## Acceptance and reporting

Preserve original dense285, pilot28, anchors24 and compact-light8 denominators,
unknown visibility, old misses and alternative associations. Predictive coasts
and degraded measurements are not strict measured detections. Require zero
newly lost known visible samples before considering a candidate, but recognize
that this alone does not establish generalization or justify added workload.

Run synthetic failure/regression tests before freezing a candidate. Use saved
journals when sufficient; a change affecting associations, background learning
or other feedback needs a separately verified closed-loop rerun rather than a
claim of equivalence from output-only replay. Independently audit results and
review any new loss before any opt-in integration. Production remains unchanged.

Deliver source-review evidence, exact workload/label limitations and either one
frozen evaluated candidate or the specific missing evidence that prevents a
defensible candidate. Never report operational airborne recall/false alarms
without supported airborne examples and exact reviewed non-target exposure.
