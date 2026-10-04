# V39 bounded saved-source shape review

This is a development-only model diagnostic, not a detection improvement or
airborne classification result. Production behavior is unchanged. No threshold,
winning-family rule, localization replacement, or output filter is promoted.

## Audit result

The frozen study in
`results/tiny_target/accuracy_v39_20260925/shapes_01` passed the bounded independent
audit in `scripts/audit_accuracy_v39_shapes.py`. The receipt is
`results/tiny_target/accuracy_v39_20260925/shapes_independent_audit_01.json`.

- All 188 bound files were checked and rehashed after analysis, including
  frozen inputs, live/snapshot implementation, tests, synthetic study and outputs.
- All 364 original current25 arrays matched their byte hashes, source identities,
  original measurement coordinates and detector fractional coordinates.
- Every unchanged V38 baseline feature record matched all four parent lag copies.
  The four new families were present uniformly for all 364 selected measurements.
- Reference groups and all summary arithmetic were independently reconstructed.
  Dense samples remain 284/285 matched, pilot 28/28 and anchors 24/24. The new
  compact-light reference remains 6/8 matched. Dense frame 216 and compact-light
  frames 16–17 remain explicitly unmatched; this patch study did not recover them.
- Eight independently constructed full eight-column least-squares fits covered
  the first bright and first dark record in each family. The maximum residual
  energy disagreement was 1.46e-11; compact amplitude disagreement was 1.95e-13 DN.
  Polarity-constrained compact coefficients and signed edge coefficients agreed.
- The 19 frozen synthetic tests passed. No original source video was opened or
  decoded. Only current25 NPZ members were read for pixel-valued checks.

Audit boundaries: it did **not** independently repeat video-to-patch sampling or
exhaustively search every new model bank on all 364 patches. The numerical checks
validate eight selected fits, while synthetic tests cover joint-bank mechanics.
This is a bounded check, not an independent classifier validation.

## What the source fits show

The fractions below are reductions of the best-edge residual energy after a
joint edge-plus-compact fit. They are searched, in-sample fit statistics—not
probabilities or target scores. Groups overlap and are not independent trials.

| Selected scope | Centered isotropic | Localized isotropic | Elongated | Equal pair |
| --- | ---: | ---: | ---: | ---: |
| All 364 states: median conditional gain | 0.759 | 0.754 | 0.585 | 0.494 |
| Compact-light 6 matched states: median gain | 0.183 | 0.488 | 0.479 | 0.594 |
| Provisional-control 70 states: median gain | 0.331 | 0.403 | 0.397 | 0.271 |
| Compact-light center-boundary winners | 5/6 | 3/6 | 4/6 | 3/6 |
| Provisional-control center-boundary winners | 67/70 | 35/70 | 36/70 | 36/70 |

These controls are provisional selected nuisance-review states, not exhaustive
airborne-negative labels. Their gains are useful counterexamples to treating
positive compact gain as an acceptance rule; they do not establish a false-alarm
rate. The full 364-state collection is likewise selected development evidence.

For the difficult compact-light frame 18, conditional gains are respectively
0.0895, 0.1819, 0.4106 and 0.4285. The localized isotropic center is at offset
(-4, 0); the elongated and paired centers are (-4, 2). All touch the fixed center
search boundary. Frame 23 also selects boundary centers in every localized family.
This is consistent with centering/shape mismatch contributing to the poor old
fit, but does not prove the fitted location is the intended moving component.
No new source review or physical identity adjudication was performed here.

## Why this is not yet a repair rule

The banks contain 27, 75, 200 and 400 compact templates, each jointly paired with
72 edge templates: 1,944 / 5,400 / 14,400 / 28,800 searched combinations. Each fit
has eight linear coefficients, but that count does not account for location and
shape search on the same pixels. The banks are not strictly nested: the localized
center grid uses {-4,-2,0,2,4}, whereas the centered grid uses {-1,0,1}. A richer
family is neither guaranteed to dominate nor statistically calibrated against
the others. No complexity-corrected threshold has been derived.

The synthetic study deliberately exposes limits:

- Exact in-bank points on edges at offsets 0, 2 and 4 recover the injected 12 DN
  coefficient, while the old exclusive point-versus-edge score prefers an edge.
  These are mechanics tests, not generalization evidence.
- Offsets 6 and 8 and an off-bank paired shape remain mismatched. Boundary flags
  disclose the finite search rather than claiming the target was relocalized.
- A paired family fits exact paired lobes and also closely spaced distinct
  components. It cannot establish whether there is one object, two objects or
  a structured background feature. Original measurements must remain auditable.
- Curved cloud structure and deterministic noise produce positive compact gains.
  Stationary structure under multiplicative exposure retains strong source shape;
  a shape fit cannot establish movement or airborne class.
- Slow motion, a turn and intermittent visibility do not create any velocity or
  temporal evidence here: each frame is analyzed independently. A vanished point
  leaves an uninformative edge residual, not a confirmed absence label.

Conclusion: retain these separate families as diagnostic evidence, not as a new
acceptance gate. Any later localization or track-level change needs a separately
frozen candidate, bounded source review of alternative locations and regressions,
and a test that does not conflate reduced overlay workload with airborne accuracy.
