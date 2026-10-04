# Phase 5: Background Subtraction and Noise Normalization

## Output contract

`RobustPreprocessor` consumes one full-resolution `StabilizedFrame` and emits a
`ResidualFrame` containing four same-size arrays:

- signed `float32` residual, `current - background`;
- strictly positive `float32` noise sigma;
- signed `float32` whitened residual, `residual / max(sigma, floor)`; and
- a boolean detection-valid mask.

The current frame never contributes to its own background or noise estimate.
Warm-up is explicit: residuals remain available for diagnostics, but the
detection-valid mask is empty until enough prior samples exist. A stabilization
segment reset clears all background history. Arrays are immutable after the
stage returns.

Update validity and detection validity are deliberately different. Valid,
unsaturated pixels may update the model during warm-up, while no pixel may be
used for detection. Invalid geometry, clipped pixels, configured dead pixels,
bad-pixel-map entries, under-supported history, unstable noise, and global
illumination events are excluded from detection.

## Saturation must be masked before the warp

The RAW recording is stored in `uint16`, but observed values are 12-bit codes
left-shifted by four. Its maximum and saturation code are therefore 65520, not
65535. The first 16 frames contain roughly 1.65--1.98 million pixels at that
code out of 15.26 million pixels.

The radiometric validity mask is attached to the source `Frame` before
stabilization. It is then warped with nearest-neighbor interpolation and eroded
along with geometric validity. Testing saturation only after cubic image
interpolation would incorrectly recover diluted clipped pixels near a mask
boundary. The original intensity image is not modified.

After saturation and stabilization-border exclusion, approximately 88.1% of
the current RAW pixels are detection-usable. A boolean `.npy` bad-pixel map,
dead-level threshold, and unstable-sigma ceiling are supported but remain
unset until camera-specific calibration is available.

## Reference and streaming models

The deterministic reference is a temporal median background with per-pixel
temporal MAD noise. It keeps a bounded history, ignores invalid samples, and is
useful as a correctness ruler. It is too expensive for the live Python path.

The selected streaming model keeps per-pixel running location, variance, and
support. Innovations are clipped at four sigma. Once warm, pixels exceeding
three sigma are excluded from model updates so a moving PSF is not immediately
absorbed. The selected update rate is 0.2. A scene-wide normalized median of
1.8 sigma, or a scene-wide normalized robust scale above 1.5, suppresses
detection for that frame while preserving the signed residual. During a global
event, per-pixel candidate exclusion is bypassed so clipped whole-scene updates
can recover instead of freezing the old model.

## Synthetic characterization on Jetson

The deterministic test uses 48 96x128 frames, 4x different noise levels across
the image, slow illumination drift, saturated/bad pixels, and a moving
sigma-0.8-pixel PSF with fractional motion.

| Model | Median target flux retained | Bright/dark normalized-scale ratio | Median latency |
|---|---:|---:|---:|
| Temporal median + MAD | 97.49% | 1.049 | 66.39 ms |
| Robust running + EWMA | 96.80% | 1.030 | 3.89 ms |

For the selected running model, the minimum retained flux is 93.86%, median
noise-only frame maximum is 6.95 normalized units, and the mean fraction of
noise samples beyond absolute 5 is 0.094%. The running model's final drift
residual is 0.70 levels versus 1.29 for the temporal median configuration.

## RAW16 characterization on Jetson

The 16-frame run accepted 14 of 15 global transforms. Frame 3 remains rejected
for low coverage and high reprojection residual; that rejection correctly
starts a new background segment. The measured coverage gate was adjusted from
25% to 20% because later good pairs consistently have 100--102 inliers, 100%
inlier ratio, 0.005--0.011 px motion, and support in 10--11 of 48 grid cells.

The sidecar cadence changes from about 100 ms to about 320 ms. These timestamps
were captured after host pull/copy rather than at sensor exposure, so the
recording-specific gap threshold is 4x the container interval. Actual timestamp
values remain unchanged and must still be used by synthetic tracking.

Seven frames complete warm-up and are detection-ready. Their normalized robust
scale converges from 1.33 to 1.10, with median 1.17. Two later frames contain a
scene-wide intensity change and are explicitly suppressed by the global-change
guard. Detection-valid area is approximately 88.06--88.07% on ready frames.

The real residual distribution is not Gaussian: about 3.4--3.8% of ready-frame
samples exceed absolute 5 normalized units, and the maximum positive residual
peak is 537.8. This likely combines real scene variation, interpolation error,
and uncalibrated sensor defects. Phase 6/7 thresholds must therefore be
calibrated empirically after matched filtering and temporal integration; a
theoretical Gaussian threshold would be misleading.

Median full-resolution preprocessing latency is 2.90 seconds in the current
NumPy implementation. This is a correctness prototype, not a real-time result.
The bounded-state model is suitable for a future fused C++/CUDA implementation
after Phase 6 defines the exact support and normalization requirements.

## Running and reports

```bash
python3 -m tiny_target.preprocessing_benchmark \
  --output /tmp/synthetic_preprocessing.json

python3 -m tiny_target.motion_cli \
  --config configs/tiny_target_test.yaml \
  --max-pairs 15 --omit-points \
  --output /tmp/raw16_preprocessing.json
```

Checked reports:

- `results/tiny_target/phase5/synthetic_preprocessing.json`
- `results/tiny_target/phase5/raw16_preprocessing_15pairs.json`

Both reports contain SHA-256 identities for the exact implementation files
used on the Jetson. Phase 5 does not choose a detection threshold; it produces
the signed, support-aware residual consumed by Phase 6 PSF matched filtering.
