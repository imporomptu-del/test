# AOT independent image-patch diagnostic — frozen plan

Date: 2026-09-28 UTC. Status at authorship: no source PNG decoded and no new
image-patch result computed. This diagnostic does not change production code,
detector settings, motion-model selection, or prior evidence.

## Question and scope

Test whether a separately computed image-patch displacement corroborates saved
LK scene-motion correspondences. The approved AOT sequence is
`00bb96a5a68f4fa5bc5c5dc66ce314d2`, native 2448×2048 grayscale U8. Use only adjacent
pairs with previous input indices 0, 42, 85, 127, 170, 212, 255, 298 and their
immediate successors. No target annotations enter sampling or scoring. No
private clips, RAW16, sealed holdouts, new downloads, or target-detector runs.

## Fixed sample, before image outcomes

Use accepted correspondences from `residual_patterns_01/result.json` (SHA-256
`a18cb9614b02847e5a9e34a16e5f3fab3ec1963541cb2684444e28134c0d85fc`). Divide the native
image into 6 rows × 8 columns. For each pair/cell select the minimum and maximum
saved candidate-translation residual norm. Break ties by smallest accepted-list
index. Deduplicate if the two selections coincide, preserving both roles. Keep
all 48 cell records per pair, including empty cells; never replace unavailable
points. Maximum actual sample: 768. This is a purposive extreme-residual sample,
not a representative estimate of LK's population error rate.

Also retain all four previously enumerated ±48px counterexamples, separately
from actual adjacent pairs: previous085/+48/selected402/accepted348;
previous127/+48/selected691/accepted599; previous127/−48/selected386/accepted342;
previous298/−48/selected526/accepted488. Bind the original large-shift result
SHA-256 `3342c3440dd1d190653a9aa46b5fd0210157ffd3faa811ed76506a78b09b6703`
and outlier-list SHA-256
`87f79aa1b39992819d607ebfbb2436ff1d601f8d762b722ec8f85f0b49792992`.
These use an exact integer translated copy of the previous source image, fill
128, no wrap/interpolation. They do NOT use the actual next image. Known shift
is used only after matching for truth comparison, never to guide a search or
qualify a peak. They are diagnostic counterexamples, not random controls.

## Independent matcher and fixed descriptive rules

At each original previous feature position p, round the template center
componentwise by floor(p+0.5). Use both 33×33 and 65×65 native-pixel templates.
Search every integer displacement in [−64,+64]² centered on that rounded
previous position, never on saved q, a fitted motion, or known truth. Pixel
values are original U8; no image warp/resampling. Return estimated feature q
as p + integer displacement, so the anchor offset cancels.

Compute zero-mean normalized cross correlation with NumPy float64 FFT and
integral window sums/squares. FFT padding is numerical computation only, not
padding of missing image pixels. Before source execution, test the entire
valid score surface against direct spatial ZNCC, including non-square searches.
Use population standard deviations. Keep raw scores, not rounded probabilities.

Invalid previous template: unavailable, no padding/replacement. Otherwise
compute any clipped valid current search and record its exact bounds, but
ANY clipped search is unqualified. Choose the first row-major maximum (lowest
dy then dx). The runner-up is the best finite score outside Chebyshev radius 3
around the winner. Missing runner-up means no uniqueness evidence.

A scale qualifies only with finite values, previous-template and winning-current
patch standard deviations both ≥1 U8 DN, best ZNCC ≥0.8, runner-up gap ≥0.05,
full unclipped search, and winner strictly inside all search boundaries.
The two scales must both qualify and their offsets agree within Euclidean
1.5px. Then compare BOTH offsets to saved LK displacement: both ≤1.5px means
agreement; both >1.5px means disagreement; a mixed result is ambiguous. No
averaging or favorable-scale selection. Any other result remains ambiguous
(with unavailability and each failed qualification explicitly preserved).
These are frozen diagnostic routing thresholds, not production gates.

## Predeclared reporting and visual inspection

Keep every chosen record in the denominator, including unavailable, clipped,
low-texture, nonunique and scale-inconsistent observations. Report counts per
pair and selection role, scale qualification failures, the two-scale verdict,
offset disagreement to saved LK, and counterexample truth error separately.
For mutually supported matches, describe each scale's residual to the ORIGINAL
saved global candidate and within-pair spatial variation, without fitting any
new regional/global model. Show per-pair median and p90 residuals and coordinate
component ranges only as descriptors of this selected supported subset; no
population rate, causal claim, or statistical significance claim.

Freeze a bounded visual sample before image outcomes: within each image
quadrant, among min-role selected records choose the smallest original residual;
among max-role records choose the largest original residual, ties by accepted
index. Deduplicate while preserving visual roles. No substitute if a quadrant
has no eligible point. This yields at most 8 visual records per pair (64 total),
plus all 4 counterexamples. Render source template, current patch at saved LK
and at both independent offsets, with native-pixel rectangles and integer
nearest-neighbor display enlargement. A saved fractional LK endpoint is rounded
half-up ONLY for its display crop and labeled as such, not a new measurement.
Render both scales and their raw diagnostic values. Do not draw a target label.
This fixed visual sample is inspectable regardless of qualification. Any later
extra illustration must be explicitly marked post-hoc, not replace this sample.

## Provenance and execution gates

Use `download_validation.json` SHA-256
`b602b89755e60122f1cac200808d19bc58d55443697ff5d16f83da8afde530d9`, images metadata
only (img_name, png_sha256, pixel_sha256, bytes, source_frame, timestamp_ns).
The runner, generated-only tests, this plan, numeric inputs and metadata are
hash-bound in a manifest before selection. `--select` reads metadata/numeric
records only, never source PNGs, and creates an exclusive selection artifact.
Root freezes its hash together with manifest hash before `--run` may read any
source PNG. Verify approved file and decoded-pixel hashes, shape and dtype.
Use exclusive new output paths; never overwrite valid reports. Audit source,
input, selection, manifest and code hashes before/after. Any implementation
repair after results requires an explicit versioned amendment and fresh output,
not a silent rerun or outcome-driven threshold change.

Output: `outputs/seaqr_aot_pilot_20260927/image_patches_01/`. Copy compact reports,
provenance, selection and summary into the corresponding repository results
directory. Retain full results and review images in the output directory.

## Interpretation limits

This integer-grid check cannot establish subpixel/0.1px precision. ZNCC can be
ambiguous under repetition, aperture effects, deformation, occlusion, lighting
changes or motion beyond its bounded search. A 33/65px patch may follow dominant
surrounding texture rather than its central feature. Two sizes share pixels and
are not independent physical truth. Thus agreement corroborates image motion,
but does not establish object identity; disagreement is method disagreement,
not by itself proof of a bad correspondence. Low qualification is inconclusive,
not absence of motion. Known synthetic translations can establish errors in
those counterexamples only. Camera/global features are not target detections.
This moving daytime AOT sequence does not establish stationary/night deployment
accuracy. Production remains unchanged and efficiency work remains paused.
