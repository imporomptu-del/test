# Regional motion: better supported alignment, not yet a detector fix

2026-09-28. The frozen numerical experiment completed all eight real frame pairs,
384 guarded spatial folds and 26 generated control cases. Local affine motion
is a promising candidate for the next image-level safety test. **Production is
unchanged; the target detector was not run.**

## What improved—and what did not

Against both independent patch-size references, local affine had lower median
and p90 prediction error than the guarded global-translation reference in all
eight pairs, comparing the **same supported query locations** within each pair.
Its per-pair median errors were 0.46–0.89 native pixels, versus 1.28–5.18 pixels
for that reference. These are errors to integer-grid patch offsets, not physical
ground truth or target-detection accuracy.

The maximum supported disagreement with either patch reference was 2.584px;
the subpixel medians must not be read as uniformly subpixel accuracy.

Coverage limits that result. Local affine made predictions at **5,612/6,970
accepted LK points (80.5%)**, not 80.5% of image pixels. In the hardest pair,
212→213, coverage was only **392/784 (50.0%)**, even though it covered 21/24
qualified patch references there. The favorable patch subset is not
representative of the difficult or unsupported regions.

Nor is affine uniformly more precise than local translation: on their common
patch-reference locations, local translation had lower median error at both
patch sizes in the first five pairs. Affine was better in the last two evaluable
pairs, with only two and eight common references; pair 212 had none. The benefit
is broader useful support for spatially varying motion, not an everywhere-better
estimator or justification for an outcome-selected hybrid.

## Fixed comparison and retained denominators

Each of the 48 native grid cells was withheld in turn, with a 64px guard excluded
from training. Local translation and local affine used the identical nearby
training subset within 512px of the held-out cell center. All fits used the
predeclared cell-balanced Huber estimator. Local predictions required coherent
training support and a query inside its coherent-training convex hull. No
held-out endpoint or patch reference affected fitting or eligibility.

The global arm is a newly guarded **numerical translation reference**, not the
original rejected production motion fit. Its numerical availability is not a
claim that it passes production quality gates. No unavailable local prediction
was filled by a fallback or counted as correct.

Original selected / accepted / lost counts remain **7,832 / 6,970 / 862**. This
experiment cannot recover the lost features or assess their displacement.

| Retained query cohort | Total | Global available | Local translation available | Local affine available |
| --- | ---: | ---: | ---: | ---: |
| All original accepted LK observations | 6,970 | 6,970 | 3,004 | 5,612 |
| Frozen patch sample | 758 | 758 | 284 | 569 |
| Both patch scales qualified and mutually consistent | 172 | 172 | 52 | 156 |

The remaining **586 patch samples stay unresolved**; a regional prediction does
not validate their correspondences. Membership in the 172-reference cohort
depends on the two patch scales, never their agreement with LK. Both patch
references are scored separately, without averaging or selecting the better one.

### Per-pair coverage and independent-reference errors

Error entries are **global → local affine**, native pixels, on exactly the same
queries. The independent-reference coverage column is also the common-query
denominator for those comparisons. Input indices are zero-based.

| Pair | Affine / accepted LK | Affine / patch references | 33px median | 65px median | 33px p90 | 65px p90 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 000→001 | 778/901 | 23/23 | 1.409 → 0.634 | 1.409 → 0.626 | 5.711 → 1.624 | 5.711 → 1.163 |
| 042→043 | 831/928 | 14/16 | 1.642 → 0.742 | 1.278 → 0.887 | 5.749 → 1.061 | 5.749 → 1.243 |
| 085→086 | 735/866 | 18/18 | 1.565 → 0.701 | 1.635 → 0.701 | 5.573 → 1.115 | 5.573 → 1.115 |
| 127→128 | 755/903 | 18/20 | 3.101 → 0.657 | 3.101 → 0.606 | 6.851 → 1.031 | 6.851 → 1.009 |
| 170→171 | 714/855 | 18/19 | 1.312 → 0.683 | 1.312 → 0.683 | 6.430 → 1.203 | 6.430 → 1.122 |
| 212→213 | 392/784 | 21/24 | 4.287 → 0.514 | 4.287 → 0.514 | 8.143 → 0.850 | 8.062 → 0.806 |
| 255→256 | 743/875 | 24/28 | 5.176 → 0.461 | 5.176 → 0.639 | 10.679 → 1.017 | 10.679 → 1.308 |
| 298→299 | 664/858 | 20/24 | 2.496 → 0.858 | 2.448 → 0.858 | 5.019 → 1.289 | 5.019 → 1.620 |

The JSON summary retains per-pair maxima, all three pairwise comparisons,
all-three-common comparisons, median paired differences, and all cohort
denominators. The paired median difference is not generally the difference of
the two medians. Root summary differences are explicitly left-minus-right
reductions; raw runner differences use the labeled second-minus-first convention.

Errors to saved LK also improve in median/p90 against global translation in all
eight pairs on affine-supported queries. However, maximum residuals remain as
large as **20.90px** and are not uniformly improved. Saved LK is not truth: such
residuals could reflect incorrect correspondences, independently moving content,
or model mismatch. They were not removed or relabeled from these outcomes.

### Where support is missing

Affine passed training-support gates in 352/384 cell-pair records, but that is
not query coverage. Of 1,358 unavailable accepted queries, 486 were in
training-ineligible cells and 872 lay outside the coherent-training hull.
Only 327/384 cells actually emitted at least one accepted-query prediction.

Local translation passed training gates in 190/384 cells and emitted predictions
in 165; it lost much more coverage when motion varied within a region. Gate
failure categories overlap and must not be summed as disjoint failures.

Of the 172 independent references, **160 are in the lower half and only 12 in
the upper half**. Affine covers 144/160 and 12/12 respectively. This cannot
certify motion compensation across the sky. Its 2,904/3,731 upper-half accepted
LK predictions have much weaker independent image evidence than those counts
alone imply. Coherent training support is not calibrated correctness.

## Target-preservation and contamination controls

These are **generated coordinate correspondences**, not rendered image targets.
They test vector subtraction and leakage, not visibility, warping, contrast,
tracking identity or recall.

- All 18 target cases completed: two analytic backgrounds, three target
  placements including a cell-boundary crossing, and three motion offsets.
  Changing an excluded target's motion left complete fit/support/weight/prediction
  receipts identical in all 12 matched comparisons.
- Affine was available at all 114 generated target-query evaluations. Its 76
  nonzero-versus-zero supported vector checks preserved the injected change to
  within 4.45×10⁻¹⁶px; clean analytic background recovery was within 2.55×10⁻¹⁴px.
  These are repeated generated queries, not independent targets or observations.
- Local translation was unavailable at all 57 affine-background query
  evaluations. The corresponding 38 unavailable preservation comparisons were
  **not** counted as passes. Global translation preserved the *change* in target
  motion but retained up to 1.83px background-model error: invariance alone does
  not mean correct compensation.
- Eight additional cases deliberately put a one-point or nine-point moving
  nuisance cluster into neighboring training support. In the four clean versus
  contaminated comparisons, affine stayed available, but prediction drift was
  nonzero: approximately **0.002875px** and **0.021960px** for the two cluster
  sizes under both backgrounds. The worst directional retention of the injected
  (4,−2) target vector was about **99.51%**. These bounded, well-supported
  synthetic cases do not prove robustness to dense or coherent moving clutter.

There was no outcome-derived acceptance cutoff for the contamination effect and
no retuning after outcomes. Generated assertions passing means the declared
invariants held; it does not promote a model into production.

## Decision and next experiment

Proceed to a **bounded image/temporal target-preservation harness**, with local
affine as a candidate and global/local translation retained as controls. Freeze
that protocol before image-level outcomes. Specifically:

1. Test the actual resampling/residual path on native 8-bit images, using paired
   background-only and known independently moving point-spread targets, including
   subpixel motion and cell-boundary crossings. Measure contrast/energy and
   localization preservation, not just motion-vector invariance.
2. Carry support masks into the result explicitly. Unsupported areas and transitions
   must stay unavailable, not become zero residual or an unvalidated global fallback.
   Check image borders, seams and loss/reacquisition of support across frames.
3. Exercise static-camera controls and temporal target trajectories. Preserve the
   stationary-camera production path; do not silently introduce repeated warps,
   reset loops or a new threshold sweep.
4. Only after the safety check, run an isolated frozen detector comparison with
   warmup, resets, coverage, nuisance output and labeled-target continuity all
   accounted for. Then validate separate recordings; this one development
   encounter cannot establish generalization or nighttime performance.

No new model-selection heuristic, relaxed quality gate or production regional
warp was added in this step. Efficiency and RAW16 remain paused.

## Reproducibility and scope

The plan, runner, generated controls, tests, numerical helper and three prior
input artifacts were bound before any actual regional fits. Before/after
identities match. Full fit receipts remain in the output directory; compact
evidence is also copied into repository results. **369 tests passed**: 328 prior
regressions plus 36 regional and five summary tests.

An independent implementation rebuilt all 384 fold partitions, all 1,152 arm
training inventories, all patch joins and all common-support comparison
counts/errors. The predeclared first/last-pair audit checked 18 arm/cell cases:
14 native-coordinate robust refits and four exactly reproduced pre-fit
abstentions. Its largest eligible prediction discrepancy was 5.68×10⁻¹⁴px
(largest discrepancy across all checked numerical quantities: 1.51×10⁻¹²),
within the fixed 1e-8 tolerance. It used neither the experiment's fitting helper
nor its hull implementation. See the saved independent audit receipt.

- [Frozen plan](/Users/romanmaksymiuk/Documents/SEAQR/skymove/docs/aot_regional_motion_plan_20260928.md)
- [Full result and artifact index](/Users/romanmaksymiuk/Documents/SEAQR/outputs/seaqr_aot_pilot_20260927/regional_motion_01/README.md)
- [Compact numerical summary](/Users/romanmaksymiuk/Documents/SEAQR/outputs/seaqr_aot_pilot_20260927/regional_motion_01/summary.json)

This step read numerical prior captures only. No source images were decoded,
annotations used, external downloads made, Jetson jobs started, detector run,
RAW16 processed or SEAQR sealed holdouts accessed. The eight correlated pairs
remain development evidence from one moving-camera daytime sequence.
