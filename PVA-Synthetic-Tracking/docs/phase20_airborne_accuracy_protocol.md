# Accuracy before acceleration: airborne-only evaluation

User-defined target: **airborne objects only**. Ground vehicles and their lights are non-target nuisance responses. A coherently moving image point is not automatically an airborne true positive. This protocol is evaluation-only and changes no detector/tracker/motion configuration.

## Keep two distinct questions separate

1. **Image-feature regression:** does the pipeline retain the observed moving point and its track? The existing three encounters in 029/126 support this question, including a turning encounter and intermittent visibility. They have unknown physical classes.
2. **Operational airborne accuracy:** does it find airborne objects while rejecting ground traffic, lights, background structure and artifacts? This requires independently supported airborne/non-target labels. Image-feature regression success cannot substitute for these labels.

The initial [bounded pilot](../results/tiny_target/phase20/accuracy_baseline_v1_20260914/README.md) adds frame-level development references but does not close the operational ground-truth gap.

## Reference-building rules

- Select temporal/spatial coverage from source footage or independent recording information, not from detector proposals. Prior reviewer exposure must be disclosed; revisited development clips are not blind/untouched tests.
- Review every frame at native pixel sampling for each declared bounded region. Overview images may provide context, never tiny-target absence labels. Nearest-neighbor inspection may repeat source pixels; no invented detail or generative enhancement.
- Distinguish **supported airborne**, **supported non-target**, **unknown class**, and **unknown visibility**. An object against dark pixels alone does not establish that it is airborne. Use wider temporal/scene context and, where available, recording/operator information. Physical aircraft subtype is not required, but airborne status needs support independent of detector output.
- Mark approximate source-coordinate centers and uncertainty on visible frames individually. Do not interpolate through invisible/ambiguous frames and treat them as truth. Known occlusion and uncertainty remain separate from verified absence.
- Negative intervals require exhaustive review of the exact ROI/time scope under the airborne-only definition, including adjudication of moving lights. Unreviewed pixels/frames and uncertain movers are not negatives. Record exposure in ROI-seconds; never extrapolate a tiny crop's rate to a full camera frame.
- Preserve the original known-object references. New labels are versioned before scoring, with hashes binding source, frame indices, crops, annotations and pipeline artifacts. An annotation correction after seeing scores must be recorded as a new revision with a reason, not silently substituted.
- Do not access sealed holdout media or resume the old Phase 19 scan for this work. Current local review allowlist is 0029, 0126, 0055 and 0082 only. No automatic directory-wide media discovery.

## Measurements and their prerequisites

| Measurement | Required reference |
|---|---|
| Visible-frame candidate/qualified-track misses | Independently positioned, confidently visible samples; unavailable detector frames remain in the denominator |
| Airborne encounter detection | Supported airborne class plus sufficiently complete visible encounter coverage; not sparse anchors alone |
| Continuous measured identity | Every frame's visibility/identity reviewed, unique assignment; predictions are not measurements |
| Detection delay | Independently established onset; left-censored windows cannot estimate onset latency |
| Duplicate/ID-switch counts | Adequate nearby physical-object/identity truth; multiple nearby responses alone are insufficient |
| False alarms | Exhaustively reviewed airborne-negative ROI/time scope and explicit alarm-event counting policy |
| Generalization | Later, separately authorized untouched evaluation after development choices are frozen |

The pilot scorer reports conditional sample hits and conservative continuity evidence. It uses one-to-one, maximum-cardinality gated assignment with distance-ordered edges. Ambiguity is reported rather than asserted away. It does not claim minimum-distance physical identity inference. Extra nearby bright/dark responses are diagnostic workload, not automatically false positives or verified duplicate objects.

## Order of work

1. Establish supported airborne encounters, representative nuisance intervals and uncertain cases. Include complete encounter starts/ends and variation in motion, brightness, background and image location. Current pilot snippets are not sufficient for this gate.
2. Score the unchanged baseline. Diagnose misses, false alarms and identity errors separately; agree on operational accuracy/latency requirements before choosing trade-offs.
3. Fix demonstrated accuracy problems with general rules, preserving the original 24-anchor checks and the new visible-frame regression cases. Do not tune to target coordinates or time intervals.
4. Freeze an accuracy-checked reference, then accelerate one stage at a time. Preserve native/full-frame coverage and compare candidate/track behavior, not just execution completion or fps. A bounded performance-feasibility profile may precede this gate, but is not permission for a broad GPU rewrite.
5. Validate the frozen optimized version on separately authorized untouched data. RAW16/faint-target synthetic tracking remains a separate integration and evaluation obligation.

If existing footage cannot support class labels, a controlled recording of a known airborne object with capture-time/operator records is the next data requirement. Synthetic injections can test mechanics but cannot replace real airborne-object accuracy evidence.
