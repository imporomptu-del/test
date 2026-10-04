# AOT image-patch check: local motion is corroborated, but coverage is limited

2026-09-28 UTC. The frozen patch experiment completed **762/762 observations**:
758 real-pair samples and four separate known-shift counterexamples. Production
is unchanged. This is a scene-motion diagnostic, **not a detector repair, target
recall measurement, or validation of the missed airplane**.

## Outcome

- **172/758 real samples** passed both independent patch-size checks and agreed
  with saved LK motion within the fixed 1.5px tolerance.
- **586/758 remain ambiguous**. There were zero *qualified* disagreements, not
  proof of zero tracking errors. Neither size qualified for 487 observations;
  only 33px qualified for 85, and only 65px for 14.
- Independently supported motion still varies materially within individual
  frames. It is not explained away by changing the correspondence algorithm.
- Support is highly uneven: **160/172 corroborated points are in the lower
  half**, with only 14–20 of 48 cells supported per pair. Do not extrapolate
  that evidence across the image or treat unsupported sky as correctly tracked.
- On all four known-shift counterexamples, both patch sizes recovered the exact
  imposed integer motion. This includes LK errors of approximately 46px and 25px.

This weakens the hypothesis that the entire observed regional motion pattern is
just incorrect LK matching. It does **not** establish the physical cause of the
variation, clear the unresolved matches, or prove a regional correction will
preserve small targets. Spatially varying camera effects, mixed scene motion and
patch-support effects have not been separated here.

## How the check was kept separate

Before any new source-image decoding, the plan/code/tests and numerical inputs
were hash-bound. From each fixed 6×8 cell in eight actual pairs, select the
minimum and maximum saved translation-residual observations. Four singleton
cells were deduplicated; three empty cells stayed empty. Thus 384 cell records
produced 758 distinct real points, not an artificially filled sample. The
minimum/maximum role totals are each 381 and overlap at four points; do not sum
those role totals as independent observations. Qualified role counts are 94 and 79,
with one qualified overlapping point, hence 172 unique supported points.

The independent matcher searches native U8 patches over every integer offset in
±64px, using float64 zero-mean normalized cross correlation. Search is centered
at the previous point, **not** at the LK endpoint or known shift. Both 33×33 and
65×65 must meet frozen score, uniqueness, texture, border and cross-scale rules.
No image warp, threshold sweep, alternative point replacement or new motion fit
was performed. Known synthetic shifts are consulted only after matching.

The 758 points deliberately include residual extremes. Their qualification
fraction is diagnostic coverage of this selected sample—not a population LK
accuracy rate, target-detection rate, or statistical confidence interval.

## Actual-pair results

Errors below are patch-offset residuals to the **original saved global
translation candidate**, on the mutually supported subset only. They are not
errors to physical ground truth. Each error entry is median / p90, native pixels.
The candidate is not refitted to this subset.

| Input pair | Selected | Supported and agreeing | Ambiguous | 33px residual | 65px residual |
| --- | ---: | ---: | ---: | ---: | ---: |
| 000→001 | 96 | 23 | 73 | 1.56 / 5.94 | 1.56 / 6.00 |
| 042→043 | 96 | 16 | 80 | 2.05 / 7.70 | 2.05 / 7.70 |
| 085→086 | 94 | 18 | 76 | 1.64 / 5.71 | 1.64 / 5.71 |
| 127→128 | 96 | 20 | 76 | 4.59 / 7.15 | 4.59 / 7.15 |
| 170→171 | 95 | 19 | 76 | 1.42 / 7.05 | 1.42 / 7.05 |
| 212→213 | 92 | 24 | 68 | 9.24 / 14.55 | 9.24 / 14.53 |
| 255→256 | 96 | 28 | 68 | 6.40 / 12.92 | 6.41 / 12.90 |
| 298→299 | 93 | 24 | 69 | 2.60 / 10.29 | 2.60 / 10.29 |

For a concrete illustration already included in the frozen visual sample,
212→213 has corroborated motion **(1,1)px** at p=(1160.5,1152.5), but
**(15,3)px** at p=(2120.5,1904.5). Both independent sizes agree at both locations
and agree with LK. A single translation cannot describe both. This is an
illustration selected for explanation after scoring; it does not replace any
sample or establish what physically caused the difference.

[View that fixed review page](</Users/romanmaksymiuk/Documents/SEAQR/outputs/seaqr_aot_pilot_20260927/image_patches_01/review/pair_212_213_02.png>).

### Why most patches remain unresolved

At 33px, 388 observations fail the peak-uniqueness gap and 255 fail the correlation
score. At 65px, those counts are 446 and 195; 34 previous templates do not fit
inside the image. Current-search clipping affects 105 and 86 computable searches,
respectively. These failure categories overlap and must not be summed.

All 17 fixed contact sheets were visually inspected, covering all 64 planned real
visual records plus four counterexamples. Diffuse sky/cloud texture often has
several plausible offsets; dark/border patches also remain visible. Some
ambiguous patches look similar at LK and NCC locations, but that does not
override the frozen rules. No visual review produced new target annotations.

The post-hoc spatial-support audit finds 12 supported points in the upper half
and 160 in the lower half. Five pairs have no supported upper-half points at
all. This limitation must constrain any subsequent model; the support map is
not a license to infer motion where there is no evidence.

## Known-shift counterexamples

These are translated copies of previous images, **not real next frames**.
All eight patch-scale searches recovered exactly (+48,0) or (−48,0), as imposed.

| Source / shift | Saved LK truth error | Patch truth error, both sizes | Fixed diagnostic verdict |
| --- | ---: | ---: | --- |
| 085 / +48px | 45.8238px | 0px | Disagrees with LK |
| 127 / +48px | 1.7906px | 0px | Disagrees with LK |
| 127 / −48px | 24.9268px | 0px | Disagrees with LK |
| 298 / −48px | 0.7239px | 0px | Agrees within 1.5px |

The last row is deliberately retained: the current check cannot certify 0.5px
or 0.1px accuracy. The previous global fit excluded the first three LK points
and retained the fourth. These are scene features, not false target detections.
Four preselected failures do not measure the new method's general success rate.

[View all four counterexamples](</Users/romanmaksymiuk/Documents/SEAQR/outputs/seaqr_aot_pilot_20260927/image_patches_01/review/known_shift_counterexamples.png>).

## Recommended next move

Design a **bounded, support-aware regional-motion experiment**, not a production
replacement yet. Fix its geometry, estimator, minimum support, spatial exclusion
and comparison criteria before outcomes. Compare it with the frozen global
baseline on guarded, withheld spatial regions. Report coverage/abstentions along
with errors on identical supported test locations; never count abstention as a
correct prediction or drop the difficult observations from accounting.

Require coherent support from multiple separated scene features; a single moving
dot must not determine its local background correction. Test sparse/ambiguous
regions and independently moving small targets explicitly before any detector
integration. Do not propagate a well-supported ground correction blindly into
unsupported sky, and do not simply loosen the global confidence gates. Preserve
the stationary-camera path. The next model test still cannot establish target
recall; an isolated detector rerun and target-preservation evaluation come later.

## Artifacts and reproducibility

- Frozen plan: [aot_image_patches_plan_20260928.md](/Users/romanmaksymiuk/Documents/SEAQR/skymove/docs/aot_image_patches_plan_20260928.md).
- Full result: [result.json](/Users/romanmaksymiuk/Documents/SEAQR/outputs/seaqr_aot_pilot_20260927/image_patches_01/result.json).
- Compact numerical summary: [summary.json](/Users/romanmaksymiuk/Documents/SEAQR/outputs/seaqr_aot_pilot_20260927/image_patches_01/review/summary.json).
- Review index and provenance: [README.md](/Users/romanmaksymiuk/Documents/SEAQR/outputs/seaqr_aot_pilot_20260927/image_patches_01/README.md).

Result SHA-256: `6fd7beb2e83ae8915e2ec9f5d96ae9f7fa956df66a3cee4e5a2c6146f0a82f3c`.
**328 tests passed** (286 prior +36 matcher/selection/provenance +6 summary/crop).
Selection, manifest, source-file, decoded-pixel, synthetic-current and frozen
artifact identities passed before/after checks. Review crops preserve original
U8 values and use integer nearest-neighbor enlargement; LK fractional endpoints
are rounded only for display. See the audit receipts for independent numerical
verification. A separate implementation recomputed all valid integer candidates
directly for 20 fixed audit records: 39 full search surfaces and one correctly
unavailable template. Winner offsets and qualification gates matched exactly;
the largest saved score/gap discrepancy was 1.09×10⁻¹³. All 762 row decisions
and per-pair descriptor aggregates were independently reconstructed.
A separate numerical audit rebuilt all 384 cells, all 758 actual selections,
the four counterexamples and all 68 visual IDs from the prior numerical inputs;
every selection and denominator matched.

No production edits, Jetson execution, new downloads, target
detector run, RAW16 processing or sealed-holdout access occurred in this step.
Annotations were not used for sampling or scoring.
