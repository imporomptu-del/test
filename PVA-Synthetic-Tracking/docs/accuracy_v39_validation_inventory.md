# V39 validation inventory and pre-score review proposal

Date: 2026-09-25. Scope: annotation and provenance metadata for development
8-bit clips **0029, 0126, 0055 and 0082 only**. This inventory did not open
source videos or other image arrays, discover media directories, access RAW16
or sealed holdouts, or use the Jetson. It creates no labels and changes no
existing reference. Source claims below are inherited from existing receipts,
not independently verified source decodes in this task.

The governing document is [the airborne-only accuracy protocol](phase20_airborne_accuracy_protocol.md).
The implementation direction is [the V39 next-steps plan](accuracy_v39_next_steps.md).

## What the current evidence can establish

The references support **development regression of visible moving image
features**, including intermittent visibility, a turn, fading, illumination
change and an extended/multi-lobed compact feature. They do not establish
operational airborne accuracy. The reviewed metadata contains **zero verified
airborne-positive encounters and zero authoritative airborne-negative exposure**.
These are limitations of the labels, not evidence that the clips contain no
airborne objects.

All four clips are development material. More frames from them can improve
coverage and expose general-rule failures, but cannot turn this experiment
into an untouched generalization test. The 0055/0082 clips have no established
positive references; passing their empty positive-reference sets says nothing
about their recall.

## Existing reference inventory

Paths in this document are relative to the `skymove` repository. Old records
remain immutable; the relevant hash-bound acceptance records take precedence
over older prose saying a later review was still pending.

| Reference | Supported scope | Limitations and denominator treatment |
|---|---|---|
| Dense encounter reference | 029 A: frames 40–109, 14 visible and 56 unknown. 029 B: 220–389, 132 visible and 38 unknown. 126: 80–229, 139 visible and 11 unknown. | **285 visible samples**, 105 unknown samples. All physical classes unknown. Keep unknowns and the existing miss at 126 frame 216; do not report only matched samples. |
| Strict pilot | 029 frames 70–81: 4 visible and 8 unknown; 029 frames 270–281: 12 visible; 126 frames 140–151: 12 visible. | **28 visible samples**, overlapping the dense encounters. Distinct scoring packet/tolerance, not 28 additional encounters. |
| Original confident anchors | Sparse 029 A/B and 126 manual positions. | Preserve **24 required confident anchors**. Raw files also contain lower-confidence 029 frame 360 and 126 frame 220; do not silently promote them or count every stored anchor as required. |
| V38 compact-light reference | 029 fixed ROI `[2619,2998,256,192]`, frames 12–24. Visible: 14–19,23,24. Ambiguous: 12,13,20–22. | **8 visible samples**, kept separately; **5 ambiguous** remain unpositioned. Provisional class-unknown whole-feature centers, uncertainty 3–5 px. Multi-lobed appearance does not establish physical object count or identity. |
| Seven provisional control windows | 126 cloud/building-response crops, 13 frames each; 91 crop-frames total. | Detector-selected development controls, `human_confirmed: false`, not eligible for authoritative false-alarm rates. The 70 baseline measured responses are response states, not 70 objects or verified false positives. |
| Six pilot coverage windows | 0055/0082: upper frames 100–111, middle 300–311, lower 500–511. | Upper/middle are dark or obstructed; lower are unresolved light fields. These remain unresolved, not negatives. |
| Extended light-field review | 0055 and 0082, frames 450–649, ROI `[3460,2562,256,192]` in each. | 400 reviewed crop-frames; fixed light layout/brightness variation was described, without confidently identified independent motion in those crops. No airborne-absence adjudication. Overlaps the lower pilot windows; do not double-count exposure. |

### Exact provenance files

Dense reference:

- `results/tiny_target/phase20/encounter_accuracy_v2_20260914/annotations.json`
- `results/tiny_target/phase20/encounter_accuracy_v2_20260914/scoring_packet.json`
- `results/tiny_target/phase20/encounter_accuracy_v2_20260914/review_record.md`
- `results/tiny_target/phase20/encounter_accuracy_v2_20260914/review_acceptance.json`
- `results/tiny_target/phase20/encounter_accuracy_v2_20260914/README.md`

The dense annotation SHA-256 is
`56b98b77b8acdfabc564978b80203ca0e9a698bffbca9c8d44e46ff886e67fed`.
Its 790 reviewed crop-frames include the 390 encounter frames and 400
light-field frames. Source-only component centroids assisted positioning;
the 285 retained samples were visually checked. Reviewer exposure and
purposive encounter selection are disclosed. Dense position uncertainty is
5 px with the existing 2 px additional scoring tolerance. The encounter
bounding unions are explicitly **not exhaustively reviewed negative regions**.

Strict pilot:

- `results/tiny_target/phase20/accuracy_baseline_v1_20260914/annotations.json`
- `results/tiny_target/phase20/accuracy_baseline_v1_20260914/source_review/packet.json`
- `results/tiny_target/phase20/accuracy_baseline_v1_20260914/scoring_freeze.json`
- `results/tiny_target/phase20/accuracy_baseline_v1_20260914/verified_summary.json`
- `results/tiny_target/phase20/accuracy_baseline_v1_20260914/verification.json`
- `results/tiny_target/phase20/accuracy_baseline_v1_20260914/README.md`

The pilot annotation SHA-256 is
`46ee498d31efb7a24662e10fdadcf1fac8f5f174a3d12139768b31d96ce18c37`.
Its nine 12-frame windows total 108 crop-frames. Its stricter 5 px scoring
radius must remain separate from the dense reference's 7 px radius.

Original anchors and provisional controls:

- `results/tiny_target/phase19/chunk0029_visual_review_20260913/visual_annotations.json`
- `results/tiny_target/phase19/chunk0126_avi_20260913/visual_review/visual_annotations.json`
- `results/tiny_target/phase20/clutter_review_20260913/background_controls.json`

The anchors were guided by user-provided encounter times. Unlisted frames
are unknown, not interpolated truth. The seven control windows were chosen
from diagnostic proposals after viewing failure appearance; their nominal
9.1 ROI-seconds are **not** verified negative exposure. Two control windows
already have zero baseline measured responses and cannot demonstrate a
rejection improvement.

V38 compact-light addition:

- `results/tiny_target/accuracy_v38_20260925/compact_light_reference_v1.json`
- `results/tiny_target/accuracy_v38_20260925/compact_light_reference_v1_root_review.json`
- `docs/accuracy_v38_reference_review.md`
- `docs/accuracy_v38_reference_review_root.md`

The annotation SHA-256 is
`44c954cb3c8f338f014766c5e2df801e66ade33f257e6ff1258f554ff08e6c5d`.
The second assistant review approved this unchanged version before V38
scoring, for provisional class-unknown image-feature regression only. This
was not a blind review: the sample originated in V36 removal diagnostics,
reviewers had prior exposure, and its review header disclosed a historical
track ID. Positions came from unmarked source crops, not tracker centers;
the original video's decode provenance was inherited. Both the original
source uncertainty and the pre-score `uncertainty + 2 px` match rule remain.

Whole-clip workload context:

- `results/tiny_target/accuracy_v36_20260924/shadow_01/freeze.json`
- `results/tiny_target/accuracy_v36_20260924/full_context_01/summary.json`
- `results/tiny_target/accuracy_v36_20260924/full_context_independent_audit_01.json`
- `results/tiny_target/accuracy_v36_20260924/README.md`

The four sources contain 2,741 frames: 687/674/689/691 respectively. Existing
metadata reports 4784×3190 and nominal 10 Hz AVI playback, not independently
verified physical acquisition cadence. The nominal 274.1 source-seconds
are unlabeled whole-clip workload, not negative exposure. Whole-clip output
counts can measure workload changes, not precision or false-alarm rate.

## Preserve the causal comparison and every denominator

1. Freeze the unchanged baseline and one candidate policy, code/configuration,
   journals, source/reference hashes, ordering and match rules before their
   comparison. Truth positions, clip IDs and sample timestamps must never
   become inference inputs. A separate review selection may use truth to
   locate review evidence, but not to alter policy behavior.
2. Report all **285 / 28 / 24 / 8** samples as four separate panels, broken
   down by encounter. They overlap and must not be summed into a nominal
   independent sample size. Missing journal frames, unavailable processing,
   unmatched candidates and no eligible output remain explicit failures or
   unavailable states in their original denominators—not omitted rows.
3. For each visible sample expose candidate presence, actual associated
   measurements, motion qualification, image-verifier availability/decision
   and emitted output. Preserve alternative measured IDs and use the frozen
   assignment rules. A prediction is never an actual measured detection.
4. Preserve all existing ambiguous/unreviewed frames and old misses. Keep
   class-unknown, uncertain visibility, known non-target and supported
   airborne as distinct concepts. Do not infer a negative from rejection,
   darkness, failed matching or temporal interpolation.
5. Report paired retained/newly lost/newly recovered visible samples, not
   just net totals. A gain elsewhere cannot compensate for newly losing a
   known development feature. Review every new loss before promotion.
   Keep provisional-control measured-response counts and unreviewed full-
   frame workload separate; neither is verified false-positive count.
6. Record measured and coast-only output separately, plus availability and
   reasons. Multiple nearby IDs do not establish duplicates, identity
   switches or physical object count without independent identity truth.
   Left-censored reviewed snippets cannot establish first-detection delay.
7. State the replay boundary. A decision-only replay of frozen journals can
   test output-policy changes on those journals. If the proposed policy
   changes background learning protection, associations or other upstream
   state, validate that closed-loop effect in a separately frozen run before
   claiming end-to-end equivalence. Do not silently infer it from replay.

The present compact-light failure illustrates why these stages matter:
baseline qualified measurements cover 6/8 samples and the V36 filter covers
4/8, despite near-reference actual measurements in all eight. Frames 16–17
involve quadratic motion-fit qualification and insufficient history in a
different **coexisting** identity, not established source absence or an
established identity handoff. Frame 18 is a visible feature despite edge-
preferred fits. None should disappear from the table when testing continuity.

## Proposed next bounded validation packet: freeze before scores

This is a **proposal**, not an extracted packet or new ground truth. Before
extracting or scoring it, write a separate machine-readable manifest binding
the following exact coverage, rendering procedure and source hash claims;
verify actual source hashes at that authorized extraction. Freeze the source
review and labels before reading candidate scores for these new windows.
If V39 scores have already been examined before adopting this proposal,
disclose that chronology; do not retrospectively call it a pre-score freeze.

### A. Keep the mandatory historical regression panel

Always run the four panels above and their uncertainty/unknown records. Do
not replace them with more favorable new windows. Keep the compact-light
reference separately versioned and class-unknown.

### B. Add an output-independent spatial/temporal coverage grid

Use native **256×192** crops on the same four allowed sources. Nine fixed
spatial cells use every combination of source origins:

- `x = 0, 2264, 4528`
- `y = 0, 1499, 2998`

These are left/center/right and top/center/bottom **edge-aligned** windows;
the last column and row include the image boundary. This grid is determined
from image dimensions, not detections, old target centers or attractive
background appearance. Retain dark/obstructed/uninformative cells as such;
do not replace them after viewing source or outputs.

Each cell receives three 20-consecutive-frame windows. For a source with
`N` frames, define zero-based inclusive start as
`floor((N - 20) * q)` for `q = 0.1, 0.5, 0.9`; end is start + 19.

| Clip | N | Early | Middle | Late |
|---|---:|---|---|---|
| 0029 | 687 | 66–85 | 333–352 | 600–619 |
| 0126 | 674 | 65–84 | 327–346 | 588–607 |
| 0055 | 689 | 66–85 | 334–353 | 602–621 |
| 0082 | 691 | 67–86 | 335–354 | 603–622 |

This is **108 windows / 2,160 native crop-frames**, or 216 nominal ROI-seconds
of review scope at 10 Hz. It is not 216 seconds of distinct full-frame
footage: nine ROIs share the same timestamps. The cells cover about **2.90%**
of image area and 60 frames per source. This small systematic grid is a
coverage pilot, not an unbiased estimator of the full camera's false-alarm
rate or all rare-event recall. It may contain no class-supported encounters;
that outcome must not trigger post-hoc substitution of easier cells.

Review every selected frame unmarked at native sampling, optionally with
nearest-neighbor enlargement. Record visible source positions/uncertainty,
visibility, physical-class support, obstructions and unresolved cases.
Retain unassessable cells in the coverage inventory. An exhaustive crop
review alone is insufficient for airborne-negative labeling if airborne
status of a visible mover remains unresolved. Wider-context follow-up, when
needed, requires an explicitly recorded additional scope and its own source-
review record; it cannot silently enlarge the negative denominator.

Bind the manifest, frame/crop coordinates, native-array/render hashes,
source provenance, annotation version, reviewer prior exposure and scoring
code/configuration. A second source reviewer should adjudicate difficult
cases before scores, with agreement/disagreement retained. The packet is
still development material, even if these exact crops are newly reviewed.

### C. Review output differences separately, without calling them unbiased

Before scores, declare a bounded difference-inspection rule: for each of
four clips, take at most two identities from each of the newly added
measured-output and newly removed measured-output strata, ordered by first
changed frame then `(segment, track_id)`. Around each identity's first changed
actual measurement, inspect ±6 frames in a native 256×192 source crop;
clip bounds are recorded, not padded with synthetic frames. Deduplicate
identical windows. Maximum: 16 windows / 208 crop-frames. Keep empty strata
empty. Coast-only changes are reported separately in the replay ledger.

This selection uses outputs and is therefore a **biased failure-diagnosis
appendix**, not a recall or false-alarm denominator. Source review is unmarked
first; an added output is not automatically a target and a removed output
is not automatically noise. Newly lost existing visible references require
review regardless of this appendix's sampling cap. Do not manufacture source
positions for ambiguous frames or force multiple lobes into one physical ID.

## What is still needed for operational airborne validation

Current metadata cannot provide airborne-class truth, comprehensive negatives,
physical identity, encounter onset/exit or representative generalization.
If existing footage cannot support these after contextual review, the minimum
new controlled-recording package should contain:

- A **known airborne cooperative target**, with synchronized independent
  recording/operator evidence of airborne status; image motion alone is not
  class confirmation. Include independently documented ground-light/traffic
  nuisance examples and adequately observable empty declared regions.
- Original native 8-bit sources from the intended camera path, immutable
  hashes, monotonic capture timestamps/drop accounting, image dimensions,
  exposure/gain/shutter, optics/focus, codec and camera-motion/context records.
  Playback frame rate alone is not proof of physical timing.
- Complete entries/exits with sufficient pre-roll for the pipeline's frozen
  warm-up/history and post-roll; straight and turning motion, changing speed,
  slow apparent motion, fading/intermittency/occlusion, point-on-edge and
  image-border conditions. Document what was actually covered rather than
  claiming all backgrounds or target ranges from a short recording.
- Independent per-frame visibility and position uncertainty, class evidence,
  physical identity where supportable, onset/exit uncertainty, and exact
  negative ROI/time masks. Unknown or obscured periods remain unknown.
  Source annotation precedes detector-overlay review and comparison scores.
- Separate recording sessions for development and later untouched validation,
  with the algorithm frozen before the latter is scored. Adjacent-frame
  random splits are not independent validation. Two sessions are only a
  structural separation minimum, not evidence of statistical adequacy;
  representative conditions and enough independent encounters/exposure are
  needed for useful uncertainty estimates.
- A predeclared alarm-event/track counting rule and agreed operational miss,
  false-alarm and latency requirements. Frame-level confidence intervals
  must not pretend that adjacent samples or overlapping reference panels
  are independent encounters.

This is a data requirement, not authorization to collect new recordings or
open existing sealed holdouts. Synthetic injections and adversarial unit
tests remain useful for mechanics; they do not replace real class-supported
airborne and nuisance evidence. A cleaner development overlay is a valuable
checkpoint only when its retained-feature losses and unresolved workload are
also disclosed.
