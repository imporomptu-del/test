# Phase 7: Reference Synthetic Tracking

## What this stage does

Phase 7 takes the signed, PSF-matched response maps from Phase 6 and asks a
different question for every trial velocity: "If a target moved this way, do
the weak responses line up over time?"

For output position `(x, y)`, velocity `(vx, vy)`, timestamp `t_i`, and the
temporal midpoint `t_ref`, the reference samples:

```text
x_i = x + vx * (t_i - t_ref)
y_i = y + vy * (t_i - t_ref)
```

Velocity is in pixels per second. Actual nanosecond timestamps determine every
offset; frame numbers are never substituted for time. Sampling is bilinear.
A sample is valid only when every nonzero bilinear tap is valid in the incoming
matched-filter mask.

For the current uniformly weighted, nominally whitened input, the score is:

```text
score = sum(valid signed responses) / sqrt(valid sample count)
```

The result retains a `float32` best-score map, a `uint16` velocity-grid index,
a `uint16` temporal-support map, and a validity mask. Dark-target mode negates
the signed response before integration. The Phase 6 `both` policy is rejected
because a per-frame polarity maximum cannot be added coherently through time.

## Window and failure behavior

Window length, overlap, velocity bounds, velocity spacing, and minimum temporal
support are required configuration; omitted values fail as "not calibrated."
Windows never cross stabilization segments, rejected/suppressed frames, or
timestamp discontinuities. A suppressed frame clears the window rather than
silently shortening it.

The velocity order is deterministic: `vy` is the outer loop and `vx` the inner
loop. Equal scores retain the first velocity. The implementation processes one
row tile and one velocity at a time, retaining only the best full-frame maps;
it never creates a `[velocity, height, width]` score volume.

The maximum endpoint quantization error per searched axis is:

```text
0.5 * velocity_step_px_s * window_duration_s
```

An axis containing only one trial reports zero *grid quantization* error. That
does not claim that motion outside the configured search bounds is covered.

## Deterministic and noisy synthetic results

The checked benchmark embeds complete small golden score, velocity-index,
support, and validity maps for later elementwise CUDA comparison. Its noiseless
case recovers reference position `(7, 6)` and velocity `(1, -1) px/s` exactly.

Score scaling is correct from one to eight frames: a per-frame score of 3 grows
as `3 * sqrt(N)`, with maximum relative error `6.72e-8`.

Across 20 irregular-timestamp trials with a 3.5-SNR injected target:

- reference position was exact in all trials;
- median integrated peak score was 7.55;
- exact velocity was selected in 45% of trials;
- 85% were within 1 px/s, one configured grid step;
- median velocity error was 1 px/s.

The location result demonstrates useful temporal accumulation. The weaker
velocity result is also expected: a short 1.2-second observation of a broad,
noisy PSF does not strongly distinguish neighboring trajectory slopes. Phase 9
track confirmation must not treat a single-window velocity index as a precise
measurement without an uncertainty model.

For a velocity exactly halfway between grid points, the observed half-window
endpoint mismatch was 0.354 px. In 20 independent Gaussian-noise windows and
35,200 retained best-over-25-velocity scores, none exceeded 5; the p90 window
maximum was 3.97 and the overall maximum was 4.22. Maximizing over velocities
changes the null distribution, so thresholds require empirical calibration.

## Jetson performance and RAW16 result

On the AGX Orin, a 256x320, four-frame, 25-velocity reference window took
162.37 ms. Tiled streaming retained 737,280 bytes instead of an 8,192,000-byte
full score volume, an 11.11x reduction for that case. This CPU implementation
is a correctness oracle, not the production performance path.

The 15-pair RAW16 run produced four overlapping four-frame windows after
warm-up and before a rejected transform cleared the stream. The deliberately
restricted zero-velocity diagnostic grid took a median 576.91 ms per
full-resolution window and retained at least 87.28% valid output pixels.

Its median score fraction above 5 was 12.99%, and the maximum was 1333.09.
These are structured real-scene residuals, not target detections. They show
that an independent-Gaussian threshold would be invalid for this recording.
The run therefore emits no candidate decision and records warnings for the
provisional PSF, acquisition-timestamp semantics, and uncalibrated physical
velocity bounds.

## Running and reports

```bash
python3 -m tiny_target.synthetic_tracking_benchmark \
  --output /tmp/reference_synthetic_benchmark.json

python3 -m tiny_target.motion_cli \
  --config configs/tiny_target_test.yaml \
  --max-pairs 15 --omit-points \
  --output /tmp/raw16_synthetic_reference.json
```

Checked Jetson reports:

- `results/tiny_target/phase7/reference_synthetic_benchmark.json`
- `results/tiny_target/phase7/raw16_synthetic_reference_15pairs.json`

All 71 tests pass on the Jetson with warnings treated as errors. Phase 8 can
now implement CUDA shift-and-stack against the serialized golden maps and the
reference output contract.
