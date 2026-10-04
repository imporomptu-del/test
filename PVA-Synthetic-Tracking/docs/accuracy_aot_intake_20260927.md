# AOT metadata-first pilot intake — 2026-09-27

User approved the external AOT pilot. This is development/intake, not an
independent validation split or a detector accuracy claim. Existing SEAQR sealed
holdouts and RAW16 stay unopened; detector settings stay frozen.

## Pre-image selection and access limits

- Public unsigned HTTPS only, bucket
  `airborne-obj-detection-challenge-training`, part1 first (publisher order).
- Read at most the first 8 MiB of `part1/ImageSets/groundtruth.json`, accepting
  only fully decoded sequence objects; the truncated final object is discarded.
  This is a bounded convenience sample, not a dataset-wide distribution estimate.
- List image object names/sizes for at most three metadata-selected candidates.
  Listings are metadata, not image access. Do not enumerate the whole bucket.
- Select the first sequence in publisher order whose first 300 published frames
  are contiguous, have at least 30 airborne-labeled frames and 20 empty-label
  frames. Use all first 300 frames, including the causal prefix/warmup. Selection
  uses annotations only, not source appearance or detector performance. If none
  qualifies, stop and report the coverage gap rather than silently relax criteria.
- Image cap: 300 frames and 1 GiB compressed source PNG bytes, one sequence.
  Report exact source bytes before image transfer. Do not expand this cap, use
  paid access, or download an entire 120-second sequence automatically.
- This first case cannot promise both clear sky and cloud/structured backgrounds;
  metadata does not establish those classes. Inspect only after selection.

## Data and timing contract

Preserve native 2448×2048 uint8 pixels, full field of view, integer nanosecond
timestamps, publisher frame indices, part+sequence+object identity, all original
labels and source hashes. Ground-truth boxes use left/top/width/height according
to the official helper and its rendering notebook; the challenge introduction's
top/left comment conflicts with that implementation. Do not invent precise point
truth or discard unknown/range-outside benchmark objects as negatives.

Publisher empty-label frames are useful annotated negative exposure, not a claim
that SEAQR's nighttime false-alarm rate is established. Original timestamps must
be audited; nominal 10fps packaging does not imply perfectly regular acquisition.

## Execution boundary

The unchanged PVA/CUDA baseline requires the Jetson runtime, absent on this Mac.
Local intake and generated lossless FFV1 decode tests are allowed. No CPU fallback,
training, parameter tuning, Jetson upload/run, or production deployment is part of
this intake. An external-input harness and explicit timing/scoring policy must be
frozen before running that baseline. Preserve all local source-point regressions.

## Sources and license

- [AWS AOT registry](https://registry.opendata.aws/airborne-object-tracking/)
- [Official helper walkthrough](https://www.aicrowd.com/showcase/aot-dataset-walkthrough-using-helper-scripts)
- [Official YOLO conversion/rendering notebook](https://www.aicrowd.com/showcase/sample-interface-for-training-with-darknet-yolo)
- [Official challenge definitions](https://www.aicrowd.com/challenges/airborne-object-tracking-challenge)
- [CDLA-Permissive-1.0](https://cdla.dev/permissive-1-0/)

Airborne Object Tracking Dataset was accessed on 2026-09-27 from
https://registry.opendata.aws/airborne-object-tracking/. Provider: Amazon.
Any derived manifests must identify transformations and retain provenance; retain
the license with downloaded/redistributed source data.
