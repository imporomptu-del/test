# Phase 6: PSF Matched Filtering

## Implementation

Phase 6 consumes the signed whitened `ResidualFrame` from Phase 5. It emits a
`MatchedFilterFrame` containing the best signed response, selected subpixel
phase, valid-support count, and detection-valid mask. Suppressed warm-up or
global-change frames bypass all convolution and remain explicitly invalid.

The NumPy implementation is the deterministic reference. The selected OpenCV
CPU implementation uses the same zero-border correlation and matches reference
validity, phase selection, and response. Phase kernels are streamed: only the
current best response and phase index are retained, rather than materializing
a `[phase, height, width]` volume.

All PSF templates have unit flux. Each correlation is divided by the template
L2 norm. With independent, unit-variance whitened pixels, an individual phase
response is therefore in nominal SNR units. The response remains signed;
bright-target policy controls only which phase wins and how candidate score is
interpreted.

## Provisional phase-bank selection

The current camera PSF has not been measured. The benchmark uses a
pixel-integrated Gaussian with sigma 0.8 px and radius 3. Arbitrary true phases
from -0.45 to +0.45 px were tested.

| Phases per axis | Kernels | Worst exact-template response retained | Median localization error | Maximum localization error |
|---:|---:|---:|---:|---:|
| 1 | 1 | 87.02% | 0.424 px | 0.636 px |
| 2 | 4 | 95.85% | 0.224 px | 0.354 px |
| 4 | 16 | 98.95% | 0.106 px | 0.177 px |

The 2x2 bank is selected provisionally. Its median SNR gain over testing the
brightest single pixel is 1.68x. The 4x4 bank gains only about 1.3 percentage
points in median exact-template response while increasing the correlations
from four to sixteen.

On 512x640 Jetson inputs, OpenCV median times were 29.01 ms for one phase,
49.16 ms for four phases, and 111.07 ms for sixteen phases. The selected path
processed 6.67 megapixels/second in that benchmark. OpenCV and NumPy agreed to
7.15e-7 maximum absolute error, identical masks, and identical phase indices;
OpenCV was 1.98x faster in the direct comparison.

## Noise-only behavior

For the selected 2x2 bank on independent Gaussian noise, phase maximization
shifts the response median to 0.300 while retaining robust scale 0.977. Across
31,008 valid samples there were no scores above 5. The median frame maximum was
3.35 and p90 was 3.87. These values characterize the multiple-phase selection
bias; thresholds must not treat the selected response as an unbiased standard
normal variable.

## RAW16 result

Seven of sixteen frames are warm and pass Phase 5's illumination guard. Full
7x7 support reduces detection-valid area from about 88.06% at Phase 5 to
87.34% at the matched-filter output.

The median full-resolution ready-frame filter latency is 1.94 seconds. Median
matched-response robust scale is 1.43, median score fraction above 5 is 5.60%,
and the largest score is 724.15. These large real tails are not interpreted as
targets: the recording contains structured scene residuals, uncalibrated
defects, interpolation effects, and no ground-truth target labels. Phase 7 must
measure how much motion-consistent temporal integration rejects these spatial
false peaks before any detection threshold is selected.

The provisional Gaussian warning is recorded in the report. See
`docs/psf_calibration.md` for the measured-kernel contract.

## Running and reports

```bash
python3 -m tiny_target.matched_filter_benchmark \
  --output /tmp/psf_phase_bank_benchmark.json

python3 -m tiny_target.motion_cli \
  --config configs/tiny_target_test.yaml \
  --max-pairs 15 --omit-points \
  --output /tmp/raw16_matched_filter.json
```

Checked reports:

- `results/tiny_target/phase6/psf_phase_bank_benchmark.json`
- `results/tiny_target/phase6/raw16_matched_filter_15pairs.json`

Phase 6 does not emit candidates. Its signed, support-aware response is the
input to the timestamp-aware reference shift-and-stack detector in Phase 7.
