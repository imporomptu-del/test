# Image-domain compensation: useful alignment, unresolved tiny-target safety

2026-09-28. The frozen experiment completed eight native 8-bit frame pairs,
3,520 injected probes across three compensation arms (10,560 evaluations), and
747 generated temporal frames. **Do not promote this regional warp into the
detector yet.** Better background alignment coexists with narrow-target
attenuation, weak-motion residual distortion, and substantial unavailable area.
Production source/configuration is unchanged; no target detector was run.

## What improved

Local affine compensation reduced clean adjacent-frame residual RMS versus the
guarded global-translation reference in all eight pairs, by 5.1–40.6%, on
**identical supported pixels within each pair**. This is not a nuisance-alert
reduction: image differences mix registration errors, exposure, noise, and real
moving content. The comparison also excludes unsupported pixels.

| Pair | Common global/affine pixels | Affine native-pixel coverage | Global → affine residual RMS, DN |
| --- | ---: | ---: | ---: |
| 000→001 | 3,973,534 | 79.3% | 4.151 → 3.749 |
| 042→043 | 4,046,603 | 80.7% | 4.054 → 3.735 |
| 085→086 | 3,806,540 | 75.9% | 3.922 → 3.723 |
| 127→128 | 4,076,564 | 81.3% | 3.905 → 3.588 |
| 170→171 | 3,905,329 | 77.9% | 3.907 → 3.671 |
| 212→213 | 2,306,113 | 46.0% | 5.856 → 3.479 |
| 255→256 | 3,766,980 | 75.1% | 4.849 → 3.538 |
| 298→299 | 3,622,000 | 72.2% | 4.144 → 3.626 |

Affine is not uniformly the best model. On the smaller local-translation/affine
common pixel sets, local translation has lower RMS in the first six pairs and
affine in the last two. No outcome-selected hybrid or fallback was added.

## Coverage is a separate requirement

Each arm has 40,108,032 possible native pixel evaluations across the eight pairs.
The global, local-translation, and local-affine valid totals are respectively
39,681,544 (98.94%), 14,736,183 (36.74%), and **29,503,663 (73.56%)**.
These are post-mask/post-erosion image pixels, not the previous study's accepted
LK points, where affine coverage was 80.5%.

| Main-probe accounting | Global translation | Local translation | Local affine |
| --- | ---: | ---: | ---: |
| Declared probes | 3,072 | 3,072 | 3,072 |
| Full target-footprint support | 3,072 | 1,168 | 2,440 |
| Entire 65×65 ROI supported | 3,072 | 1,088 | 2,224 |
| Entire ROI unavailable | 0 | 1,832 | 536 |
| ROI partial / otherwise partial | 0 | 1,984 | 848 |

Rows are **not disjoint**. A target can have complete footprint support while
the surrounding ROI has invalid margins. `full_support_eligible` is geometric
eligibility, not a target-preservation pass. Unavailable pixels remain NaN plus
false masks; they were neither zero-filled nor treated as correct predictions.
All 128 affine border probes are unavailable and none passes full-support
eligibility. Native PSF truncation is retained for all 128 border probes.

## Tiny-target preservation is mixed

Both arms below use the **same injections** at the same 1,220 fully supported
sites/conditions per width. Ratios compare the measured current-frame target
increment after U8 injection and cubic resampling with that arm's continuous
PSF oracle. They therefore include quantization and resampling effects, not an
isolated causal estimate of interpolation loss. Sigma is the Gaussian width in
native pixels, not a target box diameter; bright and dark targets are included.

| Width | Global peak ratio, median / minimum | Affine peak ratio, median / minimum |
| --- | ---: | ---: |
| σ=0.6 px | 0.890 / 0.715 | **0.904 / 0.679** |
| σ=1.2 px | 1.005 / 0.899 | 1.007 / 0.911 |

For the narrow affine targets, this is about **9.6% median and 32.1% worst-case
peak attenuation** relative to the continuous oracle. The control also loses
peak intensity, so this is not a uniquely regional-model defect. Ratios above
one occur too; cubic ringing and discretization must not be described as new
target information. Affine narrow-target L1 ratio has median 1.103 despite the
median peak reduction.

Localization looks much better than contrast: affine current-target centroid
error is median 0.035px and maximum 0.109px across the 2,440 eligible main probes,
relative to its sampled continuous oracle. Small centroid error does not mean
that a thresholded detector will still see the object.

Eight distinct main injections clip U8 source values; four remain in affine's
full-support cohort. They are retained and identified separately. The adverse
residual cases below do not clip. Geometric eligibility does not imply absence
of clipping, and clipping cases were not silently discarded.

### Weak residuals expose an additional risk

Eight supported affine main probes have signed residual-template gain below
0.5. This is a **post hoc descriptive tail count**, not a newly chosen acceptance
threshold. All are σ=0.6, with the nominal offset (0.5,0), at four pair/cell
combinations, each with both +16 and −16 DN injections:

| Previous frame / cell | Approximate gain for each polarity |
| --- | ---: |
| 42 / 28 | −0.0698 |
| 170 / 22 | 0.4245 |
| 298 / 21 | 0.0193 |
| 298 / 28 | 0.2088 |

For pair42/cell28 the continuous residual L2 is 3.057 DN and the measured
residual L2 is 5.622 DN, with a signed template response of approximately
−0.213 DN. This is **not division by zero**: the residual shape/projection
changes materially. It also does not mean the current-frame bright target
became dark, nor does it establish a detector miss. Its current-target peak
ratio is about 0.696 and its localization remains close to the oracle.

The injected displacement was defined relative to the shared guarded-global
motion, not local physical motion truth. Under affine compensation, some
injections have very small remaining target displacement. Numerical inspection
suggests that U8/subpixel shape differences then become large relative to the
weak ideal difference signal. This is a mechanism to investigate in the actual
temporal path, not a demonstrated explanation of a real airplane miss.

The isolated-response/clean-RMS median rises from about 4.26 to 8.08 on common
global/affine main probes. **This is not a twofold SNR or recall improvement**:
templates and effective compensated target motion differ between arms, and
these ratios are not calibrated detection statistics. Subtracting the clean
residual from its injected counterpart isolates target transfer algebraically;
it does not prove that the target is visible amid the original clutter.

## Seams and temporal controls: what was actually checked

All 320 affine fixed-boundary probes have full target-footprint support. Their
current-target centroids advance 0.326–0.572px for nominal 0.5px spatial steps.
There are **zero support transitions** in these particular sweeps. These are
spatial injections under frozen pair fields, not real consecutive trajectories,
and do not test loss/recovery at the actual hull boundary.

Elsewhere, the dense field has adjacent displacement differences across cell
seams as large as 4.165px (pair85, horizontal boundary y=683). This statistic
compares neighboring pixels and includes the ordinary finite-pixel field
gradient; it is not a same-coordinate mathematical discontinuity. Four fixed
boundary anchors cannot certify all seams.

The separate generated test contains 72 nine-frame moving-target trajectories,
three support-dropout trajectories, and eight static repeated-image controls:
83 trajectories, 747 frames, and 664 immediate adjacent pairs. Original frames
are independently rendered; prior warped images are never recursively sampled.

- Generated supported target-center error reaches 0.148px and adjacent-step
  error relative to the sampled oracle reaches 0.167px.
- Narrow-target residual gain falls to approximately 0.654 in the main generated
  trajectories and 0.632 in supported dropout comparisons. Good centroids do
  not eliminate the contrast/residual concern.
- Six unavailable frames and nine unavailable adjacencies remain missing. No
  synthetic observation bridges the two-frame gap.
- Static repeated frames cancel exactly. That is expected for adjacent-frame
  differencing, not evidence that a stationary target was erased from the image.

These generated controls do not establish real-video track identity, learned
background safety, estimator robustness to targets, or detector recall.

## Visual review and independent verification

All 48 predeclared review panels were inspected, including unsupported samples.
They are preserved unchanged and indexed in eight contact sheets, with no
outcome-based visual selection. Native 65×65 crops are enlarged 3× using nearest
neighbor; signed residuals use the same ±16 DN scale throughout.

Sky/edge examples visibly retain unavailable regional areas. Some supported
examples show reduced residual texture, but textured-edge residuals persist,
and the injected tiny residual is not reliably obvious within the clutter.
The isolated delta panels are cleaner by construction. No real object was
classified or labeled from these visualizations.

The independent pixel audit reconstructed the predeclared first/last-pair
sample: 18 arm/cell probes, ten supported and eight unavailable. Independent
16-tap scalar cubic interpolation checked 84,500 clean/injected pixel samples,
with maximum discrepancy 3.12×10⁻⁵ DN, below the fixed 1e-4 DN tolerance.
Standalone replay reproduced the saved metrics (largest absolute difference
2.22×10⁻¹⁶). This audit decoded only original frames 0,1,298,299 and did not
import experiment helpers. Its scope is the declared sample, not all probes.

The separate accounting audit passed: all 24 full maps/five-mask inventories,
1,152 cell/arm support records, 10,560 probe accounting records, common-support
summaries, and all 747 independently regenerated source-raster hash pairs match.
It does not independently replay all real residual pixels or all generated
warps. Its machine-readable receipt accompanies this report. **416 tests passed**
in the delivery check: 391 AOT tests and 25
visible-baseline/warp/geometry unit tests. This includes the 47 new image-harness
tests. Numerical conformance is not scientific or production acceptance.

## Decision and next move

**Keep production unchanged.** Regional compensation is useful enough to
continue, but the next gate should target the newly exposed risks, not another
broad estimator or threshold sweep:

1. Retain all eight adverse cases as regression evidence. Freeze a bounded
   weak-relative-motion, seam, and support-transition test before new outcomes.
   Include signed targets, subpixel phases, zero motion, smooth changes in
   direction/speed, and original-frame reacquisition after missing support.
2. Exercise the actual temporal residual/background/state rules in an isolated
   harness, keeping availability, warmup, reset and learning eligibility explicit.
   Missing pixels must not become learned background or fake zero-residual
   observations. The current CPU adjacent-frame diagnostic does not test that.
   The production detector uses a spatially filtered image minus learned
   background, resets newly valid state, and erodes support by another six
   pixels. Its exact backend and coordinate system must remain explicit; a
   CPU-remapped input into GPU state is not a device-resident regional warp test.
3. Evaluate leaving the current detection image on its native grid while mapping
   history/background into it, as a possible way to avoid resampling the new tiny
   target. This is a design hypothesis, not a proven fix; history transport and
   interpolation still need tests. Preserve the stationary-camera baseline.
   Never compare a native-current image with stale state in a different
   reference coordinate system.
4. Only after that bounded safety gate, run the isolated frozen detector A/B on
   the actual backend and then a separate recording. Report coverage and actual
   measurements/alerts/coasts separately; do not substitute synthetic-transfer
   results for target recall or generalization.

No new fits, production thresholds, fallback policy, GPU implementation, Jetson
job, detector run, RAW16 access, new download or sealed-holdout access occurred.
This remains eight correlated development pairs from one daytime moving-camera
recording; it does not establish performance on nighttime SEAQR clips.

## Artifacts and exact scope

- [Frozen plan](/Users/romanmaksymiuk/Documents/SEAQR/skymove/docs/aot_image_preservation_plan_20260928.md)
- [Full results, reviews and reproduction](/Users/romanmaksymiuk/Documents/SEAQR/outputs/seaqr_aot_pilot_20260927/image_preservation_01/README.md)
- [Compact repository evidence](/Users/romanmaksymiuk/Documents/SEAQR/skymove/results/tiny_target/aot_pilot_20260927/image_preservation_01/README.md)

This used CPU OpenCV float32 cubic remapping with 1/32-pixel map quantization,
complete native 16-tap source footprints, saved training/hull gates, and 2px
support erosion. It is not byte-exact production CUDA/warpPerspective or the
production learned-background residual. The guarded global reference is itself
piecewise across held-out cells, not one production global warp. Continuous
oracle positions come from the saved float64 matrices independently of remapping.
All frozen before/after code/input identities match; full receipt and copy hashes
are retained in the artifact directory.
