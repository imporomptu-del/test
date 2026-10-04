# AOT natural-texture shifts: accurate global translation, imperfect point tracks

2026-09-27 local date. The [frozen 61-call experiment](aot_natural_shifts_plan_20260927.md)
completed on the Jetson. Execution/integrity checks passed, all evidence was
copied locally and raw/compressed hashes matched. The independent numerical audit
passed. No production changes or target-detector run were performed.

## Main finding

**All 48 nonzero shifts of real AOT image textures passed the original global
translation checks. Every estimated translation was within 0.015 native pixel
of the imposed motion.** The eight identical-image cases also passed, with zero
motion error. Thus this candidate frontend can recover simple small global
translations of these textures accurately; this is not a blanket tracking failure.

However, **none of the 48 shifted natural-texture cases passed the stricter
individual-point truth check**. That is a distinct outcome, not a contradiction:
many imperfect point estimates can support a very accurate robust global estimate.
The inherited strict check requires median interior point error ≤0.1px and
maximum ≤0.5px, in addition to accepted global motion and vector error ≤0.1px.
45/48 failed the median criterion; 34/48 failed the maximum criterion, with
overlap between those failures. All passed global acceptance, vector accuracy
and the minimum 30 interior-point criterion. No thresholds were relaxed.

## Every fixed shift group

Each row uses the same eight source textures. All errors are native-source pixels;
point-error ranges below are per-case summaries conditional on surviving the
original flow and observed-current-position interior checks.

| Native shift (x,y) | Global fits accepted | Strict truth passes | Largest vector error | Interior median-error range | Interior maximum-error range |
|---|---:|---:|---:|---:|---:|
| (0,0) | 8/8 | 8/8 | 0 | 0 | 0 |
| (1,0) | 8/8 | 0/8 | 0.01138 | 0.09446–0.12577 | 0.57544–2.32684 |
| (-1,0) | 8/8 | 0/8 | 0.00971 | 0.09543–0.12354 | 0.54056–0.81278 |
| (0,1) | 8/8 | 0/8 | 0.01467 | 0.11873–0.14043 | 0.51463–0.76527 |
| (0,-1) | 8/8 | 0/8 | 0.01468 | 0.11760–0.14112 | 0.52523–2.88038 |
| (4,-2) | 8/8 | 0/8 | 0.01455 | 0.10346–0.11658 | 0.24897–0.99056 |
| (-4,2) | 8/8 | 0/8 | 0.01051 | 0.10554–0.11874 | 0.25790–2.67193 |

The five unchanged generated bridge controls all passed their respective checks:
four positive controls passed strict truth, and the flat control returned zero
corners. The translated high/low controls reproduced vector errors of
0.002234/0.022385px and maximum interior errors of 0.009908/0.135132px.
Generated block textures remain easier than natural textures on these point-error
criteria; passing those controls alone would have hidden the discrepancy.

## Count lost tracks, not only successful survivors

Across the 48 nonzero-shift cases, the fixed pre-flow interior cohort contained
**38,852 point observations**. These are repeated measurements on eight images,
not independent scene samples or object detections.

- 36,311 survived the original acceptance filter; 2,541 were rejected (6.54%).
- 36,207 were accepted and within 0.5px of truth: 99.71% of accepted cohort
  points, but **93.19% of the full pre-flow cohort**.
- 104 accepted cohort points exceeded 0.5px. These tails remain visible even
  though aggregate translation is accurate.
- 14,784 accepted cohort points were within the stricter 0.1px limit.
- Per-case support survival ranged from 85.23% to 98.78% for nonzero shifts.

The eight static cases also lost 401 of 6,484 cohort observations despite zero
error among survivors. The diagnostic retains those losses; identical-image
success is not evidence that every selected feature is trackable.

The denominator is fixed from selected previous points and their expected
destinations before observing flow. It does not shrink when observed tracks are
invalid, leave the image, or fail status/forward-backward checks. The inherited
strict metric remains separately reported for bridge comparability.

Status categories describe final post-bidirectional readbacks under the unchanged
legacy shared-status policy. They reproduce the original acceptance predicates;
they do not establish whether a loss first arose during forward or backward flow.
Nonfinite and rejected points never count as valid zero-error tracks. Bounded raw
flow/status bytes preserve NaN/Inf payloads and signed zero alongside readable
JSON, resolving the preceding experiment's nonfinite-byte serialization caveat.
The audit also found 2,375 backward-coordinate subnormal components shown as
zero in the readable view. The raw bytes preserved them exactly; none of the
1,521 affected points was accepted. This readable-view loss does not change
the reconstructed motion/acceptance results and does not identify its cause.

## Interpretation and next move

This is narrower than fixing the real video. In the preceding factor experiment,
all eight actual adjacent pairs failed the global translation gates; here all
synthetic translations pass them with small vector error. The frontend is not
simply incapable of recovering translation from these image textures. Real
inter-frame changes or a global-model mismatch remain plausible. Only seven
discrete shifts were tested, with maximum magnitude 4.47px; this does not validate
every displacement within that range or larger motion. Individual
point precision still falls short of the stronger declared diagnostic criterion.

The next useful diagnostic is **point-level error and spatial-residual analysis**:
inspect weak/ambiguous tracks in these known-shift cases, then inspect actual-pair
displacement magnitudes and whether residuals are spatially coherent (consistent with an inadequate
global translation model) or scattered (consistent with poor correspondences).
Use the existing captures first. Do not choose a more flexible model or loosen
quality gates just because the actual pairs failed. Any follow-up change needs
a bounded comparison and representative mostly-stable-camera validation before
production promotion or another detection-accuracy claim.

The sole arm remains experimental `half_gain16_complete`: half-resolution
features, Harris-only gain16 and complete-grid capacity, with the original
1,000-point budget, flow and global-quality settings. Source/pyramid/LK images
remain 8-bit. The test adds no information/SNR and does not resume RAW16 work.

Synthetic copies move existing sensor noise and artifacts with the scene. They
omit independent noise, exposure/blur changes, occlusion and parallax. These 61
calls are not real stable-camera footage, target-detection validation, airborne
recall/false-alarm measurement, or a real-time speed benchmark.

## Reproducibility and evidence

219 selected local tests passed, including 27 new generated/mocked tests.
The hardware run finished in its dedicated tmux process, which then exited.
Input/dependency/artifact/clock checks passed and process-local OpenCV thread
settings were restored. No prior results were overwritten, no private camera
clips or sealed holdouts were accessed, and no packages/services/clocks changed.

The [independent audit](../../outputs/seaqr_aot_pilot_20260927/natural_shifts_01/audit.json)
reconstructed 483 numerical-array hashes and 240 raw flow-array hashes, including
6,852 nonfinite scalar components. It independently rebuilt all 60 positive
acceptance masks/fixed cohorts and all 61 scientific decisions, verified the
eight previous-image invariance groups, and checked the eight source PNGs plus
all 56 natural transformations and five regenerated controls. A separate root
calculation also matched all 54,145 accepted point pairs and the truth decisions.

The audit verifies the runtime's exact-conversion receipts and representation
consistency; it does not replay CUDA resize/S16 conversion, whose complete pixel
arrays were not stored. It retains the original global fit rather than refitting
or introducing another model. The 300-frame video-decode check remains the
frozen runtime receipt; the independent source-pixel check used only eight PNGs.

- [Compact summary](../../outputs/seaqr_aot_pilot_20260927/natural_shifts_01/summary.json)
- [Full evidence index](../../outputs/seaqr_aot_pilot_20260927/natural_shifts_01/README.md)
- [Repository evidence](../results/tiny_target/aot_pilot_20260927/natural_shifts_01/README.md)

Full local raw JSON is 140,160,045 bytes; verified gzip is 10,723,410 bytes.
Raw SHA-256: `56b9f1a363b449706c38aee4453cae1e4eb9c0316576203fbf8f2a03721bf54f`.
Manifest SHA-256: `d29a37cb62438fbde77df95ffaf2d0f8686626b236a24b79bee0b56cc99928e9`.
Audit SHA-256: `a1b6632b4a635311d89dedc0bbe8a6cc85a17b9ed870a5a2d123ec714c71e51d`.
Remote workspace: `/tmp/seaqr_aot_natural_shifts_20260927_3AUVmi` on `serg@100.73.41.79`.
