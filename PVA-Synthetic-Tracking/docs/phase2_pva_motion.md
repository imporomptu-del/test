# Phase 2: PVA Background Correspondences

## What this stage does

Phase 2 estimates **camera/background motion**, not target motion. For each
adjacent frame pair it:

1. preserves the original full-resolution frame and creates a separate 8-bit
   feature image;
2. resizes only that feature image on CUDA when configured;
3. builds Gaussian pyramids and detects Harris corners;
4. removes corners near borders, saturated neighborhoods, invalid source-mask
   pixels, and configured overlay rectangles;
5. applies score-ranked grid quotas so one textured region cannot consume the
   feature budget;
6. tracks the remaining corners forward and backward with PVA PyrLK;
7. rejects failed, non-finite, out-of-bounds, excessive-displacement, and
   forward/backward-inconsistent tracks; and
8. maps floating-point coordinates back to the original 4784×3190 pixel grid.

The output is a `MotionCorrespondences` object. Phase 3 fits and quality-gates
a RANSAC camera transform from these points. Phase 2 intentionally does not
pretend that raw optical-flow displacement is already a camera transform.

## Why several processors appear in one PVA path

The intended operations are explicit in every report:

- CPU: deterministic RAW16-to-U8 high-byte conversion and grid selection;
- CUDA: half-resolution rescale and U8-to-S16 conversion required by Harris;
- PVA: Harris, Gaussian pyramids when the image fits, and forward/backward
  PyrLK;
- CPU fallback: always `false` in this implementation.

VPI 3.2.4 does not implement the needed rescale on PVA. At half scale the
4784×3190 recording becomes 2392×1595, which fits the observed 3264×2048 PVA
pyramid limit. The original RAW16 detection image is never resized or replaced.

The Harris-produced VPI keypoint array is compacted in place after CPU grid
selection. On this Jetson, passing a separately host-backed VPI point array into
PVA PyrLK caused an internal PVA error; retaining the accelerator-compatible
Harris allocation avoids that failure.

## Running it on the Jetson

```bash
python3 -m tiny_target.motion_cli \
  --config configs/tiny_target_test.yaml \
  --max-pairs 3 \
  --output /tmp/tiny_target_motion.json
```

Use `--omit-points` for a compact metrics-only report. Without it, every
accepted full-resolution correspondence and its Harris score and
forward/backward error are recorded.

## Calibration and first RAW16 result

The example `harris_strength: 0.5` is calibrated to the first pair of
`chunks_raw16_test/chunk_0001.mkv`, after eligibility filtering. Strength 1.0
produced 65 accepted points in 8/48 cells. Strengths 0.5, 0.2, 0.1, and 0.05 all
reached the same 490 detected points; 0.5 is therefore the strongest tested
threshold that improved coverage, producing 116 accepted points in 12/48
cells.

Across the first three recorded pairs, the calibrated run produced 116, 116,
and 94 accepted tracks. The first two met the provisional 30-feature and 25%
grid-coverage gate. The third retained 81% of selected tracks but covered only
10/48 cells, so it correctly failed closed. Its larger median displacement
(0.46 px versus about 0.02 px) also demonstrates why Phase 3 needs robust model
fitting rather than averaging all tracks.

Median per-pair latency for the 4784×3190 input was 53.1 ms inside the motion
estimator: 7.6 ms for PVA pyramids, 1.4 ms for CUDA S16 conversion, 6.3 ms for
PVA Harris, 5.0 ms for forward PVA PyrLK, and 2.1 ms for backward PVA PyrLK.
Submit and synchronization portions are reported separately. Sequential FFmpeg
decoding made the four-frame test take 8.38 seconds end to end. These are
development timings, not a throughput claim; streaming and buffer reuse remain
later optimization work.

The sidecar timestamps are host times sampled after camera pull/copy, not sensor
exposure timestamps. They preserve retained-frame spacing but must remain
qualified in any later velocity result.

The compact reproducibility report is saved at
`results/tiny_target/phase2/raw16_motion_3pairs.json`. It includes resolved
configuration, input and timestamp-sidecar identities, hashes of the exact
implementation files, pair metrics, processor selection, and stage timings.
Phase 3 results are recorded separately so a correspondence failure cannot be
confused with a transform-model failure.
