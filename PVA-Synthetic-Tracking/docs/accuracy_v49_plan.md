# V49 — diagnose five guard-model contradictions without changing the detector

## Approved scope and freeze

The user approved investigating the five V48 lost-positive cases and nearby
passing controls. This is a diagnostic, not a new detector, threshold search or
production filter. Preserve every V48 file and all its results. No SSH, Jetson,
RAW16, holdout, journal payload or new video decoding. Only explicitly selected,
hash-bound V43 cached 8-bit-derived patches already used by V48 may be opened.
Efficiency remains paused. References and assignments remain unchanged.

The five fixed case keys are:

- 0029 / frame345 / segment0 / bright:2673
- 0029 / frame352 / segment0 / bright:2641
- 0029 / frame604 / segment0 / bright:4204
- 0126 / frame144 / segment0 / bright:1001
- 0126 / frame182 / segment0 / bright:1001

Select controls from V48 JSON before opening any packet. For each direction
(strictly earlier/later), prefer the closest guard-available state on the same
clip, segment and exact track ID, within 2 seconds of the case timestamp. If
absent, use the closest guard-available state on the same clip/segment/polarity
within 2 seconds. Ties use the full lexicographic state key. Label such fallback
controls unmatched; a nearby timestamp does not imply the same background or
object. Never select using source sign, reference status or pixel appearance.
Leave a direction unmatched when no control is in scope. In addition retain the
nearest archived predecessor/successor on the same exact track/segment within
2 seconds, irrespective of guard or source outcome. Deduplicate packet identities
but preserve every selection role. No more than 25 packets can be selected.
If a selected neighbor never reached the guard in V48, retain that fact and its
earlier-stage reasons, but do not open its packet or fabricate a guard result.

Write the exact manifest and source/input hashes before packet reads. Validate
each literal identity-derived packet path and digest against the pinned V48
receipt. Do not recursively traverse predecessor receipt paths. Run only into a
fresh V49 output; recheck inputs and code afterward.

## Fixed measurements

1. Reconstruct the unchanged prior background and guard using existing code.
   Require its background hash and the complete guard record to equal V48.
   Preserve all used points/stencils; do not trim current-dependent outliers.
2. Independently regenerate every exact rational contrast from the selected
   arrays. Record native patch/global coordinates, raw current/background and
   all eight prior values at binding/conflicting stencils.
3. Report the number and magnitude of exact contrast violations at diagnostic
   gain 1, and exact minimum additional response-side DN slack necessary to
   make the stencil relaxation feasible over all nonnegative gains. This is a
   dimensional severity measure, NOT a proposed error bound, a calibrated noise
   estimate, a sufficient joint pixel model, or an alternate detector score.
   Large dimensionless gain gaps can arise from near-zero denominators.
4. On the identical used guard points, report temporal range, absolute current
   minus median-background residuals and current values outside the prior range.
   Retrospectively treat each prior frame as a response, with the median of the
   other seven at the same used points as its background. Recompute all fixed
   stencils and the same exact slack. This leave-one-out consistency check uses
   other prior frames on either side; it is NOT causal validation or independent
   samples and does not estimate future-noise coverage.
5. Fit a diagnostic nonnegative gain plus affine plane to ALL used guard points
   by ordinary least squares, record rank/condition/residuals. If every used
   point has finite central-difference background neighbors, also include the
   two background-gradient columns as a first-order translation sensitivity
   diagnostic. If not, report unavailable without dropping any row. This is
   descriptive in-sample fitting with additional parameters, not proof of
   registration error and never used to change source evidence or calibration.
6. Report saved camera-transform magnitudes and render the five case histories,
   background and current patch with a shared 0–255 display range, plus fixed
   +/-8 DN residual views and stencil coordinates. Keep missing pixels explicit;
   all display enlargement is nearest-neighbor, never invented detail. No new
   source/class labels are assigned from these panels.

## Verification and interpretation

Unit-test exact slack with an independent brute-force oracle on small generated
systems; test deterministic selection, reconstruction and fixed-support fitting
using generated arrays. An independent audit checks controls, current and
leave-one-out contrasts/slack witnesses, hash bindings, counts and no source-score
changes. Keep V48's full 1,211-state denominator in context: this selected
diagnostic cannot establish rates, airborne identity, recall or generalization.

Distinguish code facts and measured inconsistencies from physical hypotheses.
The existing +/-0.5 aligned-DN bounds are measurement-sensitivity assumptions,
not camera-calibrated variability. Do not widen them, refit membership, select a
favorable alternative or combine positive results. If the local data cannot
distinguish sensor/compression variability, evolving background, contaminants or
registration error, say so and recommend the next bounded experiment.
