# V50 — prior-value background prediction, empirical uncertainty, later-time checks

## Scope and claim

The user approved building a noise-aware background predictor and checking its
predictions on later frames, separately from source evidence. Production,
detector settings, V48 source scores, reference assignments and prior artifacts
remain unchanged. No RAW16, sealed holdout, new video decode, journal payload,
SSH or Jetson access. Efficiency remains paused.

Use the original V48 ledger of 1,211 states / 509 cached 8-bit-derived patches
from 0029 and 0126. These are previously exposed, tracker-selected development
data, NOT untouched tests, background-only truth or independently verified
airborne labels. Retain all 702 history-unknown states and all 355 reference
samples / 424 alternatives / 300 unique original assignments / three misses.

This is prior-**value**-only forecasting conditional on frozen geometry: the
original cache warp/crop uses the current whole-frame registration transform.
It is not a fully causal camera-system experiment. No inner pseudo-time errors
are used for calibration: their geometry and interpolation differ from native
current observations and can leak future information relative to pseudo-time.

## Metadata-only chronological split

Within each clip/segment, find the lower median of ALL unique response frames,
without using pixels, source scores or references. Calibration is at or before
that cutoff; the next eight response frames are embargoed; evaluation starts at
cutoff+9. All tracks share the partition. Every evaluation prior history must
start strictly after the calibration cutoff. Do not open embargoed packets.

This rule gives cutoffs 277 (0029) and 161 (0126), all segment 0:

| Partition | States | Cached patches | History unknown |
| --- | ---: | ---: | ---: |
| Calibration | 422 | 190 | 232 |
| Embargo | 16 | 16 | 0 |
| Later evaluation | 773 | 303 | 470 |

Thus exactly 493 original literal cache packet paths are eligible to be read;
the 16 embargo archives remain unopened. Retain their metadata and old results.
Evaluate reference samples by the same cutoff even when no state exists (notably
0126/frame216). No favorable identity reassignment is allowed.

## Fixed predictor comparison

Keep V47's prior-only outer-guard geometry: grid step 8, Chebyshev radius 40..56,
exclude distance <=12 from every known prior foreground center, require all
eight prior values finite, and use the union of complete horizontal/vertical
[1,-2,1] stencil triplets. The new predictor does NOT estimate a current-frame
gain, inspect the current source core, or fit a current response. Missing prior
centers remain an uncertainty about guard cleanliness, not proof of absence.

Freeze exactly these three arms before opening packet bytes:

1. `median8_unit_scale`: point forecast is the eight-prior median; scale is 1 DN.
   This is the constant-width calibrated comparator, not the V48 gain model.
2. `median8_temporal_scale`: same forecast; scale is max(1 DN, median absolute
   deviation of the eight values from their median), separately at each point.
3. `median3_temporal_scale`: forecast is the median of the latest three priors;
   scale is IDENTICAL to arm 2. This isolates the faster point forecast from the
   scale change. No model will be selected/tuned after seeing evaluation results.

The 1-DN scale floor is a fixed normalization convention, not a proven noise
bound. MAD is a variability descriptor, not Gaussian sigma or calibrated camera
noise. A data-derived multiplier below supplies empirical intervals. Unannounced
transients can remain unpredictable. Empty prior-selected support is unavailable.

Current measurement gathers only forecast-selected guard points. ANY nonfinite
current value on that support makes the packet unscorable; do not delete points,
recenter, choose an alternative arm or substitute a favorable result. Return
prediction errors/uncertainty diagnostics, never an object/no-object decision.

## Calibration and uncertainty

For each arm, a finite packet score is the maximum over ALL selected guard points
of abs(current - prediction) / prior-only scale. Combine ALL archived packets
at the same clip/segment/response frame using their maximum. If any such packet
is unscorable, the entire frame unit is unavailable. History-unknown states remain
separately counted; they are not silently called covered. Do not pool frames from
different clips or count pixels/track packets as independent acquisitions.

Predeclare two policies and report both, with no automatic fallback:

- Primary `disjoint_anchor_frames`: greedily take the earliest archived
  calibration response frame, then the next at least nine frames later, based
  ONLY on metadata. Keep all packets at each selected frame. Missing scores do
  not cause replacement anchors. There are initially 12 anchors for 0029 and 10
  for 0126; their nine-frame source spans do not overlap, but this does not prove
  statistical independence.
- Sensitivity `all_calibration_frames`: use all 167 archived calibration frame
  identities, aggregating packets as above. These overlap in time and are
  dependent; this policy cannot support iid confidence or coverage guarantees.

For m finite frame scores use rank ceil((m+1)*9/10), inclusive order statistic,
to set q. If that rank exceeds m, return unavailable calibration. Preserve the
number of missing units. With 12/10 finite primary units these ranks equal the
maximum: a deliberately explicit small-sample limitation, not fine estimation
of a 90% distribution quantile. Do not cap q or fit it to the five V49 failures.

Intervals are prediction +/- q*scale, untruncated even at 0/255. Report empirical
coverage together with widths; broad intervals are not an accuracy win. The
90% target is a calibration convention, NOT a promised probability, exchangeable
conformal theorem, simultaneous whole-image guarantee or physical error bound.

## Execution order and integrity

Pin V48 receipt/ledger/states/references, split, config, code and tests in a fresh
V50 output before packet access. Validate literal identity-derived cache paths
and expected hashes; never recursively authorize media from old receipt maps.

1. Load/decode only prior arrays from calibration packets; save all forecasts
   and their hashes before decoding/scoring calibration current arrays.
2. Score calibration responses on those immutable forecasts, then save/freeze
   both policies' per-clip/segment multipliers and sample accounting.
3. Only after calibration is frozen, decode evaluation priors, save all evaluation
   forecasts, and then score the evaluation current arrays. Current arrays must
   never be passed to the predictor or scale learner.
4. Recheck all opened packets, source/config and forecast/calibration files after
   execution. A failure leaves its run intact; no valid artifact is overwritten.

## Evaluation and decision

Keep every state, unknown and embargo entry in the report. On the 303 later
archived packets, report per-arm MAE/RMSE, max errors, and paired differences,
plus point coverage, whole-packet coverage and whole-frame maximum coverage.
For each policy report interval median/p90/max half-width in DN, missing forecast,
missing current support and missing calibration separately. Also report fixed
nine-frame-bin summaries and metadata-selected disjoint evaluation anchors as
descriptive sensitivity summaries, NOT independent statistical trials.

Reference-associated context is not recall: source scores and missed targets are
not recomputed or repaired by this experiment. No false-positive rate, improved
airborne accuracy, or new-video generalization follows from forecast coverage.
Do not couple the predictor to the source solver or promote any arm into a hard
gate in this run. If intervals are too broad or calibration unavailable, report
that limitation and stop before promotion.

Generated tests must cover prior/current/core leakage, immutable arrays and
forecasts, stable scenes, steps/transients, moving contamination, missing support,
calibration rank/ties/small sample handling, grouping and embargo separation.
Independent review must verify chronological scope, saved forecast arithmetic,
calibration order/ranks, evaluation coverage/widths, and preserved original
references/misses without accessing any other data. Full regression follows.
