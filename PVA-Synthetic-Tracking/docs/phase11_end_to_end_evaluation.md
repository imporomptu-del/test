# Phase 11: End-to-End Evaluation

## Purpose

Phase 11 adds a reproducible evaluation layer around the Phase 1–10 pipeline.
It separates what controlled synthetic data can prove from what the current
unlabeled Jetson recording can prove, and it refuses to call unmatched objects
false alarms unless the source is verified empty.

This phase does not declare the detector production-ready. The checked
inventory explicitly records that authoritative real-target labels and a
versioned stress-scene collection are still missing.

## Dataset and injection contracts

`configs/evaluation/phase11_dataset_inventory.json` is the versioned inventory.
It must contain all five required categories: verified noise-only, labeled real
targets, RAW with injected targets, stress scenes, and controlled geometry.
Unavailable categories remain in the manifest with `availability: missing`;
they are not silently omitted.

`configs/evaluation/raw16_injection_v1.json` describes the initial Jetson RAW16
experiment. Each synthetic target records total flux, subpixel position,
velocity, acceleration, active frame interval, PSF parameters, and seed. The
injector uses a pixel-integrated Gaussian and records requested in-bounds flux,
clipping, quantization, and achieved source-image flux for every frame.

Injection happens immediately after source decoding and before motion
estimation, stabilization, preprocessing, matched filtering, or integration.
The signal therefore experiences the same resampling, background model, masks,
and sensor-value quantization as the original pixels.

## Controlled accuracy benchmark

The controlled benchmark runs the complete CPU reference detection path on
`uint16` frames:

```text
source + early injection
  -> identity controlled stabilization
  -> robust preprocessing
  -> PSF matched filtering
  -> timestamp-aware shift and stack
  -> thresholded candidates
  -> Kalman association and independent-hit confirmation
```

It sweeps SNR thresholds 4, 5, 6, 7, 8, 10, and 12 and integration lengths 2,
4, and 8. Reports include detection probability by flux, subpixel phase,
speed, direction, field quadrant, and integration length; verified-empty false
alarms; precision/recall; localization and velocity error; track-confirmation
probability and latency; and false confirmed tracks.

For the checked four-frame case, threshold 8 gives 0.55 detection probability,
0.6875 precision, 0.25 false alarms per integration window over the controlled
Gaussian background, 0.50 confirmed-track probability, 0.4 s confirmation
latency for confirmed cases, and zero false confirmed tracks. Threshold 10
reduces the controlled false-alarm count to zero but lowers detection
probability to 0.40. These values characterize only the small deterministic
Gaussian fixture; they are not camera operating points.

## Execution modes and checked local results

The evaluator supports four explicit modes:

- `correctness`: deterministic replay with assertions enabled;
- `throughput`: synchronous recorded input with no application queue;
- `real_time`: input pacing at the requested rate with deadline accounting;
- `soak`: repeated deterministic runs with bounded retained detail and RSS
  sampling after each repetition.

The checked local reports show 114 tests passing with 11 expected
hardware-dependent skips. Controlled throughput is about 156 completed frames/s.
At 10 Hz pacing it completes 576/576 frames with zero queue depth, zero drops,
zero deadline misses, and 55.3 ms p99 end-to-end processing latency. The five
repetition smoke-soak produces identical accuracy hashes, bounded zero-depth
queues, and peak RSS within the 1 MiB stability tolerance after the first
repetition.

This five-repetition run is a memory-regression smoke test, not a long thermal
soak. `ru_maxrss` is a high-water mark rather than current allocation, and the
controlled runner is not a live camera driver.

Run the modes with:

```bash
python3 -m tiny_target.evaluation_benchmark \
  --mode correctness \
  --output /tmp/phase11_correctness.json

python3 -m tiny_target.evaluation_benchmark \
  --mode throughput \
  --output /tmp/phase11_throughput.json

python3 -m tiny_target.evaluation_benchmark \
  --mode real_time \
  --input-rate-hz 10 \
  --output /tmp/phase11_real_time.json

python3 -m tiny_target.evaluation_benchmark \
  --mode soak \
  --soak-repetitions 5 \
  --output /tmp/phase11_soak.json
```

Checked reports are under `results/tiny_target/phase11/`.

The isolated Jetson snapshot rebuilt CUDA for SM 8.7 and passed all 118 tests
with warnings treated as errors. Its controlled throughput is about 56.7
frames/s. The 10 Hz paced run completes all 576 frames with a zero-depth queue
and no drops, but records 27 deadline misses and 173.7 ms p99 processing
latency. The five-repeat Jetson smoke-soak produces identical curve hashes and
RSS plateaus at 42,552 KiB after the second repetition. These results validate
the mode accounting; the deadline misses mean the Jetson controlled run does
not establish strict per-frame 10 Hz completion.

## Jetson and RAW16 evaluation

`tiny_target.hardware_evaluation` combines the existing checked Phase 10 RAW16
motion report with Phase 8 CUDA telemetry. It reports stage distributions,
throughput, PVA buffers, CUDA allocation and copies, GPU utilization and power,
temperature, and known telemetry gaps. The current RAW16 result is 0.207
frames/s against a 10 Hz target, so `real_time` is explicitly false. It also
does not claim a measured live-input drop rate, PVA utilization, or long-run
thermal stability.

The motion CLI's `--injection-spec` option creates a version 10 report with the
early-injection ledger and a multi-threshold candidate sweep. The companion
analyzer maps each injected trajectory through the measured stabilization
homographies before matching candidates:

```bash
python3 -m tiny_target.motion_cli \
  --config configs/tiny_target_cuda_test.yaml \
  --max-pairs 15 \
  --omit-points \
  --injection-spec configs/evaluation/raw16_injection_v1.json \
  --evaluation-thresholds 8,16,32,64,128,256 \
  --output /tmp/raw16_injected_threshold_sweep.json

python3 -m tiny_target.injected_raw_evaluation \
  --motion-report /tmp/raw16_injected_threshold_sweep.json \
  --output /tmp/raw16_injected_accuracy.json
```

The analyzer reports injected-target detection and localization, plus unmatched
candidate burden. It deliberately does not relabel unmatched candidates as
false alarms because the underlying recording is not verified target-free.

### Checked injected RAW16 result

The checked run processes 16 full 4,784 x 3,190 RAW16 frames in 92.5 seconds.
It retains 88.62% of the requested in-bounds injected flux after source-value
quantization and saturation clipping. All four candidate windows hit both the
4,096 pre-NMS limit and the 256-output limit.

No injected target reaches the bounded candidate output at thresholds 8, 16,
32, 64, 128, or 256. Diagnostic probes on the untruncated dense score surface
show why:

- 1,000 DN: median local SNR 26.6, median valid-surface rank 224,092;
- 2,000 DN: the selected field location has no valid integration support;
- 4,000 DN: median local SNR 66.5, median valid-surface rank 95,708;
- 8,000 DN: median local SNR 198.6, median valid-surface rank 17,688.

The signal therefore survives injection, stabilization, preprocessing, matched
filtering, and CUDA integration, but the real background produces thousands of
stronger responses. At threshold 8 the bounded output contains 1,024 unmatched
candidates across four windows; even threshold 256 retains 894. Because the
recording is unlabeled, these are candidate burden rather than certified false
alarms.

The immediate algorithmic problem is background/noise normalization and
structured-artifact rejection, not simply increasing the candidate caps.
Increasing the caps would raise host work and tracker load while leaving the
underlying ordering problem intact. Dense truth probes are explicitly marked
diagnostic-only and do not alter production candidate selection.

The final checked artifacts are
`raw16_injected_threshold_sweep_probed_jetson.json` and
`raw16_injected_accuracy_probed_jetson.json` under the Phase 11 results
directory.

### Human-viewable diagnostic

`tiny_target.evaluation_visualizer` converts the numeric report into a slower
annotated MP4. It replays the exact injected source frames and maps the top
bounded candidates from stabilized reference coordinates back into each source
frame. Colored circles show injection truth; red crosses show the top 30
candidates; the side panel shows dense truth-location SNR/rank, candidate-cap
state, stabilization translation, and locally normalized target crops.

```bash
python3 -m tiny_target.evaluation_visualizer \
  --video results/tiny_target/phase11/media/chunk_0001.mkv \
  --timestamps results/tiny_target/phase11/media/chunk_0001_timestamps.csv \
  --motion-report results/tiny_target/phase11/raw16_injected_threshold_sweep_probed_jetson.json \
  --accuracy-report results/tiny_target/phase11/raw16_injected_accuracy_probed_jetson.json \
  --injection-spec configs/evaluation/raw16_injection_v1.json \
  --threshold-snr 8 \
  --max-frames 16 \
  --output-fps 2 \
  --output results/tiny_target/phase11/media/phase11_raw16_annotated.mp4
```

The visualization is diagnostic evidence, not a new detector product or a
claim that the injected targets were detected.

## Exit status

The command, manifest, threshold-sweep, deterministic correctness, bounded
queue, latency-distribution, and telemetry-reporting machinery is implemented.
The checked evidence does not yet satisfy a production real-time or long-soak
claim. Closing that gap requires a live or equivalently instrumented Jetson
run, a substantially longer thermal soak, a verified-empty real background
set, labeled real targets, and the missing stress-scene collection.
