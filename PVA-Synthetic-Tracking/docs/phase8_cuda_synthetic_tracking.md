# Phase 8: CUDA Synthetic Tracking

## Implementation

Phase 8 implements the Phase 7 shift-and-stack contract as a native CUDA 12.6
library. It has no CuPy, Numba, PyCUDA, or PyTorch runtime dependency. A small
Python `ctypes` wrapper loads the compiled library and fails explicitly if the
library, CUDA device, ABI, allocation, launch, or synchronization is invalid;
it never silently substitutes the CPU reference.

The AGX Orin build target is compute capability 8.7. Build the ignored native
artifact from the checked CUDA source with:

```bash
python3 -m tiny_target.cuda_build --architecture 87
```

This creates `build/cuda/libtiny_target_cuda.so`. The CUDA test configuration
resolves that path relative to `configs/tiny_target_cuda_test.yaml`.

## Kernel decomposition

The complete matched-response and validity stacks are copied to the device
once per integration window and remain resident across velocity batches. A
small CUDA kernel precomputes `(vx * dt, vy * dt)` for every frame and velocity
using the actual timestamps.

One accumulation thread owns one output pixel. It walks trial velocities in
the same deterministic order as the reference, bilinearly samples each frame,
accumulates in FP32, applies the required temporal-support gate, and retains
only the best score, velocity index, and support. Equal scores keep the first
velocity. A final kernel converts untouched negative-infinity entries to
invalid zero-score outputs.

The implementation retains:

- one FP32 temporal response stack;
- one byte-per-pixel temporal validity stack;
- a tiny frame-by-velocity displacement table;
- one FP32 best-score map;
- one `uint16` velocity-index map;
- one `uint16` support map;
- one byte-per-pixel output validity map.

It never materializes `[velocity, height, width]` scores. Every kernel launch is
checked, and an explicit event synchronization boundary surfaces asynchronous
errors before any device-to-host copy. CUDA events separately measure upload,
displacement generation, accumulation/finalization, download, and total GPU
time.

## Numerical agreement

The configured CUDA-versus-reference acceptance tolerance is `1e-5` absolute
score units. The final Jetson run observed `4.77e-7` maximum error.

- Integer golden output is bit-exact.
- Fractional, irregular-timestamp golden output recovers position `(12, 10)`
  and velocity `(1, -1) px/s`, with `4.77e-7` maximum score error.
- Across 24 randomized combinations of odd/even dimensions, two to eight
  frames, irregular timestamps, masks, support fractions, velocity counts,
  batch sizes, and thread-block sizes, velocity indices, support, and validity
  have zero mismatches.
- Repeated calls and batch sizes produce identical maps; stale output buffers
  are not retained.
- Bright and dark polarity behavior agrees with the reference.

## Batch-size selection

At 4784x3190 with four frames and 25 velocities, an interleaved three-repeat
sweep produced these median accumulation times:

| Velocity batch | Kernel launches including setup/finalize | Median kernel time |
|---:|---:|---:|
| 1 | 28 | 215.12 ms |
| 2 | 16 | 164.46 ms |
| 4 | 10 | 151.64 ms |
| 8 | 7 | 159.00 ms |
| 16 | 5 | 145.36 ms |
| 32 | 4 | 145.20 ms |

The default is 32. It covers the current 25-hypothesis characterization in one
accumulation launch and was narrowly fastest by median. Larger calibrated grids
are still divided into bounded batches.

## Jetson performance

For 256x320, four frames, and 25 velocities:

- median kernel time: 2.21 ms;
- median host-to-host tracker time: 7.00 ms;
- effective attempted-sample rate: 3.72 billion/s;
- CPU-reference speedup: 59.47x for the kernel and 18.73x host-to-host;
- retained outputs: 737,280 bytes versus an 8,192,000-byte score volume.

The maximum intended Phase 8 characterization uses the real sensor dimensions,
4784x3190, with four frames and 25 velocities. Three repeated runs completed
without allocation or launch failure and produced identical output hashes:

- median kernel time: 153.08 ms;
- median host-to-host tracker time: 445.30 ms;
- median upload/kernel/download GPU span: 231.44 ms;
- effective attempted-sample rate: 9.97 billion/s;
- native allocation total: 442,568,872 bytes;
- retained outputs: 137,348,640 bytes versus a 1,526,096,000-byte score volume.

CUDA runtime occupancy analysis reports six 256-thread blocks per SM, 38
registers per thread, no local-memory spill, and 100% theoretical occupancy.
During the full benchmark, `tegrastats` observed 99% peak GPU utilization,
9.972 W peak GPU/SoC rail power, and 48.906 C maximum junction temperature.

The main remaining performance cost is outside the kernel: at full resolution,
about 213.86 ms is spent stacking pageable host arrays, allocating/freeing CUDA
buffers, and copying immutable output products. Phase 9 should avoid optimizing
that blindly: on-device candidate extraction can eliminate the 137 MB map
download and much of the Python output copy.

## RAW16 result

The CUDA configuration replayed the same 15 RAW16 pairs and produced the same
four windows, maximum scores, tail fractions, and valid fractions as the Phase
7 CPU report. The current zero-velocity diagnostic had:

- median accumulation-kernel time: 18.11 ms;
- median total GPU span: 78.40 ms;
- median host-to-host tracker time: 403.63 ms;
- CPU-reference host-to-host time: 576.91 ms.

The score maximum remains 1333.09 and the median fraction above 5 remains
12.99%. CUDA changes execution speed, not the uncalibrated real-noise statistics;
candidate thresholds still belong to Phase 9.

## Profiling boundary

The checked report contains CUDA-event timings, transfer bytes, allocation
sizes, launch counts, runtime occupancy attributes, register/local/shared-memory
usage, throughput, `tegrastats` utilization, temperature, RAM, and rail power.

Nsight Systems and Nsight Compute CLIs are not installed on the Jetson. An
attempt to use the available PyTorch CUPTI profiler returned
`CUPTI_ERROR_INSUFFICIENT_PRIVILEGES`; `/proc/driver/nvidia/params` reports
`RmProfilingAdminOnly: 1`. Hardware-counter bandwidth and instruction analysis
therefore requires a separately authorized system/profiling change. No package
or device permission was changed during this phase.

## Running and reports

```bash
python3 -m tiny_target.cuda_build --architecture 87

python3 -m tiny_target.cuda_tracking_benchmark \
  --output /tmp/cuda_tracking_benchmark.json

python3 -m tiny_target.motion_cli \
  --config configs/tiny_target_cuda_test.yaml \
  --max-pairs 15 --omit-points \
  --output /tmp/raw16_cuda_15pairs.json
```

Checked Jetson reports:

- `results/tiny_target/phase8/cuda_tracking_benchmark.json`
- `results/tiny_target/phase8/raw16_cuda_15pairs.json`

All 79 tests pass on the Jetson with warnings treated as errors. The live
Jetson checkout was not modified; validation and native builds stayed under
`/tmp/seaqr_phase8_20260904`.
