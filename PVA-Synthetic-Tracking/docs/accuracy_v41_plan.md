# V41: background-fixed versus transported source support

2026-09-25. User authorized the targeted background-flicker / independent-motion
investigation recommended at the V40 checkpoint. This is an offline diagnostic,
not a production filter, a threshold search, or another performance experiment.
The airborne-only protocol and four-clip local allowlist remain in force.

## Scope and chronology

Use only the already extracted V40 native source packet and frozen V34 journals
for 0029, 0126, 0055 and 0082. No additional AVI decode, RAW16, holdouts, Jetson or
production changes. Re-review the three highlighted windows in source-only form
before computing the new metrics. Prior V40 reviews, workload and regression
scores are exposed development evidence; this is not blind validation.

The diagnostic denominator is every actual measured state in every frozen V40
window: **1,367 states, including 381 strict-qualified states**. No picking only
interesting IDs, successful fits or reference-nearest states. All 108 windows
remain in the inventory, including the 93 low-information ones. Preserve every
unavailable comparison and reason. The 11 V40 visible samples remain a separate
descriptive panel, with the original proximity assignments and ambiguities; do
not modify annotations or treat unknown states as negatives.

Freeze this plan, implementation/tests, source-mechanism reviews, the V40
manifests, native-array hashes, source-review/reference records, journal hashes
and tool versions before real scoring. Rehash bound files after scoring.

## One mechanistic comparison

At each actual measurement, take a native **25x25** current patch, centered by
round-half-up on its actual source coordinate. Convert the saved uint8 BGR crop
to uint8 grayscale with OpenCV. The original source crops remain unchanged.

For each fixed lag **1, 2, 4, 8 frames**, require an actual measurement of the
same segment/ID at that exact earlier frame, available within the same retained
20-frame window. Do not substitute a prediction, interpolate an unknown source
position, use a different ID, or fetch pixels outside the saved crop. A reset,
missing measurement, insufficient retained history or unsupported pixels is an
explicit limitation, not negative evidence.
The saved nominal timestamps must differ by exactly lag times 100,000,000 ns;
otherwise the pair is unavailable. This verifies journal timing conventions,
not physical acquisition cadence.

Compare two equal-complexity explanations:

1. **Background-fixed:** map the current camera coordinates into the prior frame
   using the saved global transforms: inverse(H_prior) * H_current.
2. **Transported:** shift that same prior sampling grid by the displacement from
   the mapped current actual measurement to the prior actual measurement of the
   same tracker ID. This tests the tracker-implied displacement; it does not
   independently prove identity, and the detector positions are already exposed.

Use exact float64 bilinear sampling with missing support as NaN, not invented
zero pixels. A saturated source corner with positive bilinear weight also makes
that sample unavailable; zero-weight neighbors do not. Evaluate all nine common
prior-grid offsets in {-1,0,1}x{-1,0,1}
as a registration-sensitivity check. Do not select a favorable shift or lag.
This +/-1 pixel bank is not a calibrated registration-confidence interval.

Each model fits a nonnegative coefficient on its prior-image template plus a
planar background. Fit on one parity checkerboard and score on the other, then
swap; use identical finite, nonsaturated pixels for both models and the
plane-only comparator. Report held-out MSE, each gain, template texture and
rank/support failures. Require >=64 common pixels, >=32 per fold, full rank,
and >1/12 DN^2 prior texture after planar projection in each training fold.
These are declared numerical/quantization assumptions, not learned confidence.
Checkerboard pixels are spatially correlated and template geometry was proposed
by an exposed tracker; this is not independent statistical cross-validation.

Signed MSE difference = background-fixed minus transported. Report all lag/shift
results, the zero-offset result for each lag, and whether the sign is unchanged
over all nine offsets. Differences within 1e-9 DN^2 are numerical ties. Neither
sign, even if consistent, is a target probability or an acceptance/rejection rule.
Model mismatch, broad moving texture and alternating nearby lights are explicit
counterexamples. A poor static fit does not establish an independently moving
object; a good static fit does not authorize rejecting a slow/stopped/hovering
target. Current production output is preserved exactly.

## Tests and reporting

Before the real run test geometry direction, source origins, rounding, camera
translation, residual displacement, borders, degenerate transforms and
nonmutation. Test the model on static flicker, translated points, points on an
edge, evolving broad texture, wrong registration, prior disappearance, flat
support and alternating identical stationary lights. Report failures of the
mechanistic interpretation, rather than forcing these into a classifier.

Report availability and continuous evidence for the exact three highlighted
windows, all qualified states, all measured states, and the frozen 11-sample
panel. Independently audit selected real fits and counts. Do not report a noise
reduction percentage, airborne recall, physical-object count or whole-video
false-alarm rate from this diagnostic. Historical 285/28/24/8 regressions remain
unchanged; a later candidate would need all panels and appropriate closed-loop
validation before promotion.

Deliver a falsifiable mechanism result and explicit decision about whether this
evidence is sufficient for a rejection rule. If it is not, name the concrete
ambiguity rather than silently loosening thresholds or launching another sweep.
