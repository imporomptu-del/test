# V38 compact-light source review, version 1

This is a separately versioned, **provisional moving-image-feature reference**,
not airborne-object ground truth. It is pending a second source-only review
before scoring. Existing labels and frozen V36/V37 artifacts are unchanged.

## Exact review scope

Only chunk0029 frames 12–24 (1.2–2.4 seconds at nominal 10 Hz) were reviewed,
within native source crop [2619,2998,256,192].

Inputs:

- /Users/romanmaksymiuk/Documents/SEAQR/outputs/seaqr_accuracy_v37_review_20260924/00_0029_0012_0024_unmarked.png
- The single member 00_0029_0012_0024 of
  /Users/romanmaksymiuk/Documents/SEAQR/outputs/seaqr_accuracy_v37_review_20260924/native_bgr_crops.npz
- The source hash claim in the existing review freeze.json.

The complete airborne-only protocol was read. Every one of the 13 native
unmarked panels was inspected. Only the named native array member was accessed
for pixel measurements. No marked sheet, detector/track journal, original AVI,
other crop, RAW16, holdout, or remote system was accessed for this annotation.

The reviewer already knew this clip and the earlier rejected-response
investigation. The window itself was detector-selected, and the supplied
unmarked sheet header contains a historical track identity. Therefore this is
not blind review or an independently selected evaluation set. Neither prior
tracker coordinates nor marked outputs were used to place the new centers.

## Result and coordinate method

Eight frames have independently localizable compact luminous image features;
five remain ambiguous. All physical classes and airborne statuses are unknown.
No not-visible or airborne-negative labels are asserted merely because the
feature is hard to resolve.

| Frame | Visibility | Approximate source x, y | Positional uncertainty radius |
| --- | --- | --- | --- |
| 12 | ambiguous | — | — |
| 13 | ambiguous | — | — |
| 14 | visible | 2646.9, 3153.4 | 3 px |
| 15 | visible | 2657.4, 3152.3 | 4 px |
| 16 | visible | 2675.9, 3151.1 | 4 px |
| 17 | visible | 2690.9, 3149.7 | 4 px |
| 18 | visible | 2699.4, 3148.8 | 5 px |
| 19 | visible | 2713.8, 3147.3 | 5 px |
| 20 | ambiguous | — | — |
| 21 | ambiguous | — | — |
| 22 | ambiguous | — | — |
| 23 | visible | 2766.3, 3139.9 | 4 px |
| 24 | visible | 2789.1, 3136.6 | 4 px |

Each visible frame has its own manually bounded source-intensity rectangle.
Its approximate center is the positive-excess intensity centroid, with weights
max(I minus median(I in the rectangle), 0). The BGR channels are identical in
this saved grayscale array. Positions are rounded to 0.1 native pixel and
offset by the crop origin to give zero-based source coordinates. No trajectory
fit or interpolation supplies any coordinate.

The centers describe the whole compact luminous patch, which is sometimes
elongated, saturated, or multi-lobed. They are not exact physical-object
centroids. In particular, frame 18 has multiple bright parts; selecting the
whole image patch rather than a particular lobe can change its center by
several pixels. The uncertainty radii are conservative reviewer estimates,
not calibrated measurement errors or detector-derived matching gates.

Frames 14–15 are fainter but independently localizable: their manual rectangles
have peak/median intensities of 167/6 DN and 74/5 DN respectively. Frames 16–19
and 23–24 are substantially brighter, including saturated pixels. Frames
12–13 and 20–22 do not securely establish the selected feature's position or
correspondence in this provisional review. Frame 21 does contain a small
speck; its pixel maximum alone is not enough to assert correspondence through
the dim interval. Ambiguous frames retain null positions.

Motion is visible at the image-feature level relative to the surrounding fixed
lights. The crop does **not** establish whether this is airborne, a vehicle
light, another ground-associated feature, or multiple lights. Correspondence
across ambiguous frames and physical object count remain unverified.

## Versioned artifact and use restrictions

The frame-by-frame justifications, manual pixel rectangles, uncertainty,
exposure, provenance hashes and limitations are in:

results/tiny_target/accuracy_v38_20260925/compact_light_reference_v1.json

Use only after a second reviewer independently checks the unmarked source
positions. Any disagreement or later correction should produce a new version
with a reason; do not silently move these labels after seeing model scores.
Do not use these frames as independent generalization evidence or claim
airborne recall, false-alarm rate, continuous physical identity, onset latency,
or a negative exposure denominator from them.

Source hash is inherited from the V37 review freeze rather than rehashed from
the AVI. The native array bytes, NPZ, PNG and freeze are bound in the JSON.
This review is not a second video-to-crop decoding audit.
