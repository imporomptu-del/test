# PSF Calibration Contract

## Current status

No isolated point-source calibration from the target camera is available. The
current Phase 6 kernel is therefore explicitly provisional: a pixel-integrated
isotropic Gaussian with sigma 0.8 px and a 7x7 support. Results using it can
validate software and relative sensitivity, but cannot establish final camera
detection probability or flux calibration.

## Required measured calibration

Capture unsaturated isolated point sources at the operational focus, aperture,
wavelength, exposure, gain, and sensor temperature. Include sources across the
field and enough subpixel positions to distinguish optical PSF shape from pixel
integration. Preserve native RAW codes, black level, exposure metadata, and
the source's approximate photon-flux ordering.

For each region and condition:

1. subtract a local background without including the source;
2. reject saturated, defective, or overlapping source crops;
3. estimate a subpixel centroid without resampling the original crop multiple
   times;
4. align and robustly combine crops on an oversampled grid;
5. integrate the model back onto sensor pixels for each required phase;
6. verify unit signed flux, encircled energy, centroid, and field dependence;
7. retain the capture conditions and calibration code identity with the bank.

The runtime accepts a 2-D `.npy` array for one kernel or a 3-D
`[phase, height, width]` bank. Dimensions must be odd and each plane must have
finite positive signed flux. Kernels are normalized to unit flux on load and
the exact file SHA-256 is recorded in every report. A measured bank should be
accompanied by separate metadata describing phase order and calibration
conditions; until that metadata format is implemented, non-square phase banks
are indexed but assigned zero phase offsets.

## Runtime normalization

Let `p` be a unit-flux PSF template and `z` the whitened residual. The response
at a pixel is correlation in nominal white-noise SNR units:

```text
r = sum(p * z) / sqrt(sum(p^2))
```

This retains linearity with target flux. The validity footprint is correlated
separately, and the selected configuration requires all 49 kernel samples to
be valid. The current bright-target policy chooses the largest signed response
across phases. Dark and absolute-polarity modes exist, but bright-only is the
explicit initial operating assumption.

The maximum over a phase bank has a positive noise bias, so its score is not a
standard normal variable even when each individual phase response is. Final
thresholds must use noise-only sequences passed through the identical phase
selection and later temporal integration.
