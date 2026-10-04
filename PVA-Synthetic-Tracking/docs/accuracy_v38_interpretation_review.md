# V38 saved-evidence interpretation

## Conclusion

V38 fixes the **availability problem** of V37 on the selected development
observations. It does not establish a safe object-rejection rule. The
point-versus-edge distinction still depends on alignment, temporal separation,
the chosen feature center, and the assumed shape. A visibly supported compact
luminous feature is edge-preferred at all 36 lag/shift probes in one frame.
Therefore neither “edge always wins” nor “point must win at every probe” is a
valid absence test here.

This review inspected the frozen plan, evidence module, and saved JSON
selection/observations/summary only. No source video, saved image, patch array,
RAW16, holdout, or remote system was accessed during this interpretation.
Calculations below were made directly from logged per-probe features rather
than copying the extraction summary's feature distributions. This is an
interpretation check, not a numerical reimplementation or source-decode audit.

## Availability is now broad, but denominators still matter

There are 364 distinct selected measured states: the original 358 plus six
states matching the new compact-light reference. All 364 have finite source
support, informative nominal contrast, and a complete nine-probe envelope at
each of lags 1, 2, 4 and 8. This is 1,456 lag records and 13,104 probe records,
not independent statistical samples.

| Reference scope | Samples | Baseline matched | All-four-lag envelope available |
| --- | --- | --- | --- |
| Original dense references | 285 | 284 | 284 |
| Original strict pilot | 28 | 28 | 28 |
| Original anchors | 24 | 24 | 24 |
| New compact-light visible frames | 8 | 6 | 6 |

The original missing dense sample and the two new unmatched visible frames
remain in their denominators. The new compact-light case is separate:
baseline matches frames 14, 15, 18, 19, 23 and 24; frames 16 and 17 are
unmatched. V36 retains four of these eight reference frames, dropping the
matched states in frames 18 and 23. V38 applies no gate and does not recover
the unmatched frames merely by having available features elsewhere.

V37 had diagnostic availability on only 3/285 dense samples, 0/28 pilot and
0/24 anchors. V38 avoids its mandatory adjacent same-ID measurement and local
annulus-registration prerequisites. That explains improved access to evidence;
it is not a detector-recall improvement.

## Descriptive point-minus-edge sign stability

“Positive” below means only that the fitted point bank explains more
quadratic-residual energy than the edge bank; “negative” means the opposite.
These signs are algebraic descriptors of frozen competing models, not selected
thresholds, labels, probabilities, or an eligibility policy.

These rows count distinct selected identities at a frame, not reference
samples. The old dense references contain 288 selected states because a few
samples have multiple matching alternatives; their reference denominator
remains 285. Other reference families overlap and must not be added to them.

| Scope | Selected states | Positive at all 36 probes | Negative at all 36 | Mixed signs across probes/lags |
| --- | --- | --- | --- | --- |
| 029 A, intermittent known feature | 14 | 14 | 0 | 0 |
| 029 B, turning known feature | 136 | 127 | 0 | 9 |
| 126, known feature | 138 | 119 | 0 | 19 |
| New compact-light matched states | 6 | 3 | 1 | 2 |
| Selected provisional controls | 70 | 0 | 20 | 50 |

The apparent separation between the original known features and these selected
controls is incomplete: requiring every probe to favor the point would discard
28/288 original matched states and 3/6 new matched states. Conversely, rejecting
anything negative everywhere would also discard the visible compact feature in
frame 18. Neither observation authorizes a new gate.

At nominal shift only, all-four-lag positivity occurs on 14/14 of 029 A,
136/136 of 029 B, 134/138 of 126, 3/6 compact states, and 1/70 controls.
One nominal shift hides some sensitivity; requiring all nine shifts loses
legitimate known states. Selecting whichever lag or shift looks best would
hide other failures. Neither change is made.

### Per-lag margins and one-pixel sensitivity

Each four-number entry follows lags 1, 2, 4, 8. Medians are over the selected
states within that row. Envelope span is maximum minus minimum across the
nine fixed shifts, not calibrated camera-error uncertainty.

| Scope | Median nominal margin by lag | Median nine-probe margin span by lag |
| --- | --- | --- |
| 029 A | 0.730, 0.723, 0.731, 0.731 | 0.014, 0.012, 0.010, 0.007 |
| 029 B | 0.326, 0.354, 0.541, 0.701 | 0.087, 0.050, 0.021, 0.007 |
| 126 | 0.302, 0.342, 0.497, 0.654 | 0.097, 0.067, 0.020, 0.014 |
| Compact-light matched states | 0.097, 0.132, 0.129, 0.129 | 0.030, 0.010, 0.014, 0.011 |
| Provisional controls | −0.209, −0.264, −0.248, −0.148 | 0.270, 0.171, 0.175, 0.116 |

Controls are often more shift-sensitive on these selected examples, but the
distributions overlap. For example, the largest 029 B span at lag 8 is 0.605,
whereas the largest control span at that lag is 0.459. A typical smaller span
for known features does not justify a span threshold.

Nor are longer lags uniformly better. Nominal-positive control states increase
from 8/70 at lag 1 to 21/70 at lag 8. All-nine-positive control envelopes occur
on 0, 1, 9 and 10 states at lags 1, 2, 4 and 8 respectively. Background
structure can produce apparently point-like temporal contrast.

The two building control windows contain zero selected baseline-qualified
measured states. They supply no demonstrated rejection success. The 70 selected
control states all belong to the remaining provisional cloud/edge windows;
they are not authoritative or representative negatives.

## Concrete counterexamples that should govern the next experiment

### Visible compact feature, frame 18

For 0029/18/0/bright:138, every lag/shift margin is negative:
the combined range is −0.131746 to −0.015181. The current-source spatial
margin is also negative, −0.136547. This remains a visibly supported image
feature in the independently positioned provisional reference.

The reference center of the whole luminous patch is [2699.4,3148.8], while
the logged actual measurement is approximately [2706.049,3147.004], a 6.887 px
separation. The frozen Gaussian bank searches only +/-1 px around that logged
measurement; its best fit here is sigma 1 with offset [-1,+1].

This supports investigating **feature-center and shape-model mismatch**. It
does not prove the tracker is wrong by exactly 6.887 px: the reference has
5 px positional uncertainty and represents an extended/multi-lobed luminous
patch, whereas a detector may legitimately select a particular lobe. The
centering convention and the single-Gaussian family must be disentangled
before changing thresholds.

### Frame 23 and lag dependence

For 0029/23/0/bright:138, the reference and logged centers differ by 4.811 px.
Lag 1 has an entirely positive envelope [0.007722,0.039875], but lags 2, 4
and 8 have entirely negative envelopes. Frame 15 shows the reverse pattern:
lag 1 is negative throughout, while lags 2, 4 and 8 are positive throughout.
Choosing the favorable lag would conceal this limitation rather than resolve it.

### Original 126 reference states can also be edge-preferred

At nominal shift the following known-reference states have negative temporal
margins despite positive current-source spatial margins:

| Frame / identity | Lag | Temporal margin | Current-source margin |
| --- | --- | --- | --- |
| 147 / bright:1001 | 4 | −0.247302 | 0.710208 |
| 151 / bright:1001 | 8 | −0.431664 | 0.765933 |
| 215 / bright:1001 | 2 | −0.070730 | 0.317292 |
| 218 / bright:1001 | 1 | −0.106232 | 0.294868 |

The first two pairs share prior frame 143. The optional joint model abstains
for the first three rows because the mapped prior center lies outside its
25-square support. It is available for frame 218 and gives a small positive
margin, 0.002320. A negative subtraction-model margin is therefore neither
absence of current contrast nor a reliable nuisance label. The exact image
cause of each case is not established by this JSON-only review.

## Optional joint fits remain optional

Nominal joint-fit availability, by lag:

| Scope | Lag 1 | Lag 2 | Lag 4 | Lag 8 |
| --- | --- | --- | --- | --- |
| 029 A, 14 states | 0 | 0 | 0 | 0 |
| 029 B, 136 states | 130 | 104 | 54 | 13 |
| 126, 138 states | 130 | 114 | 68 | 32 |
| Compact-light, 6 states | 3 | 0 | 0 | 0 |
| Controls, 70 states | 37 | 38 | 38 | 17 |

Only one 029 B state and six 126 states have joint fits available at every
one of the 36 probes. No 029 A or compact-light state does. Actual prior
association, displacement, and support restrictions make this a useful
conditional diagnostic—not a universal prerequisite. An all-joint-available
gate would recreate the earlier availability failure.

## Recommended next bounded move

Do not promote a cleaner overlay or tune a scalar score from these cases.
Keep the diagnostic and existing references frozen. Address two separate
failure paths in a new, declared development experiment:

1. **First resolve measured-but-unqualified feature continuity.** The sibling
   trace investigation reports actual measurements within the fixed reference
   gates on compact frames 16–17, but one identity exceeds the fixed
   quadratic-motion RMSE qualification limit while the other coexisting
   identity has insufficient history. IDs 80 and 138 coexist with measurements
   throughout frames 14–24; no handoff is inferred. This review has not independently reread those
   journals. Use that trace to isolate localization/association changes from
   qualification. Test changing brightness, multiple lobes, acceleration,
   turning and intermittent measurements synthetically. Do not merely relax
   (raise) the global RMSE limit or shorten the history requirement.
2. **Then diagnose the evidence model at a consistent feature center.**
   Separate source-visible patch support, the detector's selected lobe, and
   the model's single-Gaussian center convention. A new bounded synthetic
   study should vary center offsets, saturated or paired compact lights,
   partial prior-point tails, straight/curved backgrounds, and exposure.
   Record where model mismatch or subtractive ghosts create edge preference.
   Any new source-only localization/extent rule must be applied uniformly,
   without reference coordinates as runtime input, and evaluated against
   nuisance examples as well as known points.

Both branches must preserve the old 285/28/24 denominators and the separate
eight-frame compact-light denominator. Keep all-five ambiguous compact frames
out of positive/negative truth; do not re-label them to make results improve.
Only after the failure mechanism is addressed should a separately frozen
eligibility policy face a strict no-new-known-loss check and a broader,
independently reviewed dataset. Supported airborne/non-target data is still
needed for the user's actual airborne-only accuracy objective.

## Saved numerical provenance

Files below are under results/tiny_target/accuracy_v38_20260925/global_multilag_01:

| Artifact | SHA-256 |
| --- | --- |
| freeze.json | 24c2f08430121edcbabb395a443c04c4a474de9af9471bd1438676faa8f4c73a |
| selection.json | b2ea836c7fafdedffdb92bc97e9457e368809a77bd9c190a87536540e1472123 |
| observations.json | 5b00ee3f38b31646050b568d19a48ac1c26a73f269b69f692e7ddf0232aac469 |
| summary.json | 444f95ab3843472661f18a34658b83585c9dea726327cb4af71179486d9e50be |

Reviewed evidence-module SHA-256:
6f36728c6910173cb02b11e55136a377394bce60dee5504f67c68911e183e2cc.
Reviewed plan SHA-256:
0eb5e450495e4e1c0d90c63bf205bba1ff72e21c1b33e7e5c59d2f409185b097.

No frozen result, implementation, annotation, or default was changed by this
interpretation. All classes remain unknown; no threshold or gate is promoted.
