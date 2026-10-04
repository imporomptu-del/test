# Phase 4: Full-Resolution Stabilization and Valid Masks

## Production data path

Each accepted Phase 3 transform is composed into a direct
`current frame -> segment reference` matrix. Phase 4 applies that matrix to the
original 4784x3190 frame, never to the half-resolution motion image and never
to an already-warped frame.

The source is converted to `float32` before interpolation. This exactly
represents the source `uint16` values while preserving negative cubic lobes and
future signed background residuals instead of clipping them back into an
unsigned format.

An image receives at most one interpolation operation. Exact identity
reference frames are converted to `float32` without resampling. Rejected motion
pairs become new identity references, so a diagnostic rejected matrix cannot
contaminate image pixels or a later integration window.

## Valid support

The source validity mask is warped independently with nearest-neighbor
sampling and a constant-invalid border. Mask warping deliberately uses the CPU
reference for both image backends: CPU and CUDA nearest-neighbor implementations
disagreed on about 0.084% of boundary pixels in a stress transform. A single
backend-independent mask makes validity deterministic and conservative.

The mask is then eroded by the configured support radius. The current two-pixel
radius is provisional and must be increased if the Phase 6 matched-filter
kernel requires wider support. `ValidMaskWindow` retains a bounded set of masks
within one reference segment and emits both:

- a per-pixel support count; and
- a common-valid mask where every frame in the window contributes.

The window clears immediately on a stabilization-reference reset.

## Interpolation and backend choice

A worst-phase half-pixel shift was tested at 4784x3190 with an impulse and a
normalized Gaussian PSF with provisional sigma 0.8 px.

| Backend / interpolation | Gaussian peak retained | Flux retained | L2 energy retained | Time |
|---|---:|---:|---:|---:|
| CPU linear | 53.1% | 100.0000% | 69.9% | 46.5 ms |
| CPU cubic | 67.0% | 100.0000% | 100.1% | 69.8 ms |
| CPU Lanczos4 | 67.5% | 100.0000% | 98.0% | 106.4 ms |
| CUDA linear, including transfer | 53.1% | 100.0000% | 69.9% | 73.2 ms |
| CUDA cubic, including transfer | 62.2% | 100.0000% | 88.7% | 84.1 ms |

CPU cubic is selected for the current host-resident Python pipeline. It retains
substantially more provisional PSF peak and energy than linear, is much faster
than Lanczos4, and outperforms CUDA while upload/download are required. Cubic
produces small signed ringing (minimum -33.2 for a 10,000-flux Gaussian), which
is retained in `float32` and must be included in later noise and matched-filter
calibration.

The exact one-pixel impulse is a harsher limiting case: CPU cubic retains 35.3%
of the half-pixel peak versus 25% for linear, but produces larger negative
lobes. This interpolation choice remains provisional until a measured camera
PSF is available. The CUDA implementation remains supported and tested for a
future GPU-resident path that avoids transfers.

On random high-frequency 12-bit-like data, CPU and CUDA linear interpolation
had 0.425% relative RMSE, p99 absolute error below 64 levels, and maximum error
below 128 levels. The test encodes those measured bounds rather than assuming
the implementations are bit-identical.

## Alignment and RAW16 result

The controlled 1024x768 textured-background test reduced median absolute
difference from 28.35 to 3.66 levels, an 87.1% reduction; p90 fell from 69.15 to
8.92 levels.

The first two accepted RAW16 transforms are only about 0.004 px. Their sampled
median absolute difference therefore remains 16 levels before and after the
warp, and p90 remains 144. This recording segment is useful for verifying that
stabilization does not invent an improvement, but it is not a strong camera-
shake benchmark. Pair 2 -> 3 remains rejected and frame 3 starts a new segment.

All four output masks retain 99.7911% of the image after the two-pixel erosion.
The first segment's three-frame common-valid fraction is the same because the
accepted transforms are subpixel-near-identity. The final segment correctly
contains frame 3 alone.

The two non-identity RAW16 frames took 117.9 and 112.6 ms end to end through
float conversion, cubic image warp, mask warp, erosion, and contract creation.
The estimated warp-path bandwidth was approximately 1.83 GiB/s. Sequential
FFmpeg decoding still dominates the short end-to-end run.

## Running the reports

```bash
python3 -m tiny_target.stabilization_benchmark \
  --width 4784 --height 3190 \
  --shift-x 0.5 --shift-y 0.5 \
  --output /tmp/interpolation_comparison.json

python3 -m tiny_target.motion_cli \
  --config configs/tiny_target_test.yaml \
  --max-pairs 3 --omit-points \
  --output /tmp/raw16_stabilization.json
```

The checked reports are:

- `results/tiny_target/phase4/interpolation_comparison.json`
- `results/tiny_target/phase4/raw16_stabilization_3pairs.json`

Optional crop and track-overlay rendering remains a diagnostics convenience;
it is not part of the image contract. Phase 5 consumes the float image,
per-frame valid mask, and window support directly.
