# Jetson Tiny-Target Pipeline Environment Audit

Audit date: 2026-09-03  
Implementation repository: `/Users/romanmaksymiuk/Documents/SEAQR/skymove`  
Jetson checkout: `/home/serg/project/methods_test/skymove`

## Audit outcome

The existing repository proves that PVA Harris and PVA Pyramidal Lucas-Kanade
(PyrLK) can run on the target AGX Orin. It is a benchmark repository, however,
not yet a reusable stabilization and synthetic-tracking application.

The first implementation work should establish a deterministic frame contract
before adding RANSAC, warping, background subtraction, or CUDA synthetic
tracking. A frame must keep its source timestamp, sequence number, exposure,
gain, image type, and dropped-frame history together. The current camera path
retains only an 8-bit image.

No system packages were changed during this audit. The camera was not opened.

## Repository state

- Git repository: `skymove/` (the parent SEAQR directory is not a Git repo)
- Branch: `scrum-75-pva-pyrlk`
- Commit: `a797835d72c22c76b3e79eddf3269ebd60564fa5`
- Upstream: `origin/scrum-75-pva-pyrlk`
- Tracked working tree: clean before and after the audit
- Build system: none
- Python package metadata: none
- Automated test suite: none
- Existing syntax baseline:
  - all tracked Python files passed `python3 -m py_compile`
  - all tracked shell scripts passed `bash -n`

Ignored `.DS_Store` and `__pycache__/` files already exist locally. They were
not modified deliberately and are not implementation inputs.

## Target hardware and software

| Item | Verified value | Evidence/notes |
|---|---|---|
| Host | `agxorin1` | Live SSH audit |
| Board | NVIDIA Jetson AGX Orin Developer Kit | `/proc/device-tree/model` |
| Memory | 61 GiB visible | `free -h` |
| Storage | 3.6 TiB filesystem, 2.2 TiB available | `df -h` |
| Power mode | MAXN, mode 0 | `nvpmodel -q` |
| Jetson Linux/L4T | R36.4.7 | `/etc/nv_tegra_release`, package `nvidia-l4t-core 36.4.7` |
| Kernel | `5.15.148-tegra` | `uname -a` |
| Architecture | AArch64 | `uname -a` |
| Python | 3.10.12 | Live host |
| NumPy | 1.26.1 | Live Python import |
| OpenCV | 4.10.0 | Live Python import |
| VPI | 3.2.4 | Live Python import and `libnvvpi3` package |
| VPI backends exposed | PVA, OFA, CUDA, VIC | Live Python introspection |
| CUDA toolkit | 12.6, nvcc 12.6.68 | `/usr/local/cuda/bin/nvcc`, installed packages |
| C++ compiler | GCC/G++ 11.4.0 | Live host |
| CMake | 3.22.1 | Live host |

`nvcc` exists at `/usr/local/cuda/bin/nvcc`, but `/usr/local/cuda/bin` is not
on the default non-interactive shell `PATH`. Build scripts should resolve or
document the CUDA toolkit path explicitly; the audit did not change shell or
system configuration.

The CUDA `deviceQuery` sample is not installed, so compute capability was not
measured directly during this audit. AGX Orin is expected to use Ampere SM 8.7,
but build configuration must verify this before treating it as an observed
environment fact.

## Development host

The Mac checkout currently has:

- Python 3.12.8
- Apple Clang 17
- CMake 3.22.2
- no `nvcc` on `PATH`
- no importable OpenCV, VPI, or PyYAML in the active Python environment

Therefore:

- accelerator-independent types, geometry, RANSAC, scoring, and tests should
  be runnable locally with explicitly declared dependencies;
- VPI/PVA and CUDA integration tests must run on the Jetson;
- importing a core module must not require the ToupTek SDK or VPI.

## Existing accelerator baseline

### Historical live-camera benchmark

The committed report under
`results/ofa/pyrlk_cam_run_20260719_043023_documented/` records VPI 3.2.4 on
the same host and real SkyEye62AM frames.

| Resolution | PVA pipeline mean | Reported equivalent FPS | Feature count |
|---|---:|---:|---:|
| 1920x1080 | 15.831 ms | 63.17 | mean 10.5 |
| 3184x2124 | 27.046 ms | 36.97 | 19 |

At 1920x1080 the Gaussian pyramid, Harris detector, and PyrLK flow use PVA
(with CUDA U8-to-S16 conversion). At 3184x2124 the 2124-pixel image height
exceeds the observed PVA pyramid limit of 2048, so pyramid construction uses
CUDA while Harris and PyrLK remain on PVA.

The live result proves backend execution, not stabilization quality. Ten to
nineteen features are too few to accept without measuring spatial coverage,
correspondence error, and RANSAC conditioning.

### Current synthetic accelerator smoke baseline

On 2026-09-03, the existing benchmark was run unchanged against generated
frames. It used three measured pairs plus one warm-up pair per resolution and
wrote only temporary files under `/tmp` on the Jetson.

| Resolution | PVA pipeline mean | Pyramid backend | Harris/PyrLK backend |
|---|---:|---|---|
| 1920x1080 | 22.853 ms | PVA | PVA |
| 3184x2124 | 38.340 ms | CUDA | PVA |

The benchmark explicitly reported `cpu_fallback: false`. The higher numbers
than the historical live report are not a regression conclusion: this was a
short smoke run on a different date and workload, not a controlled performance
study.

Temporary artifacts:

- `/tmp/seaqr_phase0_20260903_1920.json`
- `/tmp/seaqr_phase0_20260903_3184.json`

## Camera and frame-data audit

Current camera code is in `pipeline.py` and `camera_bench.py`.

Observed current behavior:

- `CaptureThread` allocates `numpy.uint8` frames.
- `PullImageV3` is called for 8-bit output.
- `PullImageV3` receives `None` for its frame-info output.
- The global queue stores the image only.
- When full, the queue discards its oldest image and increments an aggregate
  counter; the consumer cannot identify which frame sequence was skipped.
- Camera open enables automatic exposure.
- A bounded gain is computed and printed, but the current function does not
  call `put_ExpoAGain` to apply it.
- Downstream CSV timestamps use processing-time wall clock rather than camera
  exposure time.

The installed ToupTek Python SDK exposes `ToupcamFrameInfoV3`, containing:

- `seq`
- `timestamp` in microseconds
- `shutterseq`
- `expotime`
- `expogain`
- `blacklevel`

The SDK also exposes raw 8/10/11/12/14/16-bit flags, RAW mode, 16-bit output
mode, fixed exposure setters, and fixed gain setters. This establishes API
availability, not that every mode is supported by the connected SkyEye62AM.
The camera model flags and usable high-bit-depth capture modes must be queried
in a controlled camera session before selecting the detection format.

## Available recorded data

The user-provided recording directory is:

`/home/serg/project/camera_reader_sky/srcsky/chunks/`

It contains 288 AVI files totaling approximately 40 GiB. Representative files
have these properties:

- 4784x3190
- Motion JPEG (`mjpeg`)
- decoded pixel format `yuvj420p`, converted to 8-bit grayscale for detection
- declared frame rate 10 FPS
- approximately 661-669 frames per file
- declared playback duration approximately 66-67 seconds

File creation times are spaced about five minutes apart and the recorder source
uses a five-minute chunk duration. The files therefore contain roughly five
minutes of wall-clock acquisition but encode the retained frames at a fixed
10 FPS. They do not contain timestamp sidecars. These AVIs are suitable for
stabilization development and qualitative testing, but the container timeline
must not be used for authoritative target velocity or integration-duration
claims.

Two sibling datasets were also discovered:

| Directory | Content | Size | Timing |
|---|---|---:|---|
| `chunks_raw16/` | one native RAW16 file, JSON, CSV, preview | 18 GiB | host timestamp sidecar |
| `chunks_raw16_test/` | 100 FFV1 MKVs, JSON, CSV | 689 GiB | host timestamp sidecars |

The FFV1 files are 4784x3190 `gray16le` and are the preferred current source
for weak-signal experiments because they preserve the recorded 16-bit samples.
The first chunk contains 968 retained frames over 298.736 seconds, or 3.226 FPS.
Its median timestamp interval is 319.612 ms even though the capture target was
10 FPS. The metadata reports 2,770 frames received and 1,794 dropped for that
chunk.

Inspection of the recorder shows that the timestamp is produced by
`time.time_ns()` immediately after `PullImageV3` and the image copy. It is an
explicit per-frame host timestamp, not a camera exposure timestamp. It is still
substantially more truthful than reconstructing time from the video container,
and its gaps must be used by synthetic tracking.

## Existing code map

| Existing file | Reuse | Limitation for the new pipeline |
|---|---|---|
| `camera_bench.py` | SkyEye open/close and center crop behavior | Returns image arrays without source metadata |
| `pipeline.py` | ToupTek integration, queue policy, telemetry ideas | Imports camera SDK at module import; global queues; 8-bit image-only contract |
| `bench_pyrlk_pva.py` | Verified VPI calls, backend selection, timing boundaries | Benchmark-oriented; returns counts rather than correspondence objects |
| `run_pyrlk_pva_matrix.sh` | Exclusive camera ownership checks and reproducible invocation | Benchmark launcher, not runtime orchestration |
| `run_pyrlk_pva_summary.py` | Report validation and deterministic summaries | Coupled to SCRUM-75 comparison schema |
| `backends.py` | Existing MOG2 and experimental backend context | Interfaces describe blob detection, not stabilization or track-before-detect |
| `flow_engines.py` | OFA/Farneback comparison code | Dense-flow contract does not match sparse PVA correspondences |

The existing benchmark files should remain intact as historical regression
evidence. Production code should call reusable modules rather than importing
private functions from the benchmark.

## Proposed repository mapping

Because the repository and verified VPI integration are Python-first, begin
with Python orchestration and reference implementations. Introduce C++/CUDA
only for the measured high-throughput synthetic-tracking kernel and any
unavoidable zero-copy interop.

```text
skymove/
  configs/
    tiny_target_test.yaml
    tiny_target_default.yaml
  docs/
    environment.md                 # this file
    architecture.md                # Phase 1+
    coordinate_conventions.md      # before motion transforms
  tiny_target/
    __init__.py
    types.py                       # Frame and stage result contracts
    config.py
    frame_source.py                # protocol + deterministic recording source
    telemetry.py
    motion/
      pva_pyrlk.py                 # Jetson-only adapter
      ransac.py                    # accelerator-independent
    stabilization/
      warp.py
      masks.py
    detection/
      background.py
      matched_filter.py
      synthetic_reference.py
      candidates.py
    tracking/
      kalman.py
  cuda/
    synthetic_tracker.cu           # added only after reference validation
  tests/
    unit/
    integration/
```

This mapping replaces the plan's proposed all-C++ structure without changing
the architecture. It matches the current codebase and preserves the option to
move performance-critical stages into a small compiled extension.

## Required coordinate and timing conventions

These are decisions to encode before RANSAC or warping:

- image coordinates are `(x, y) = (column, row)` in pixels;
- points represent pixel centers;
- velocities use stabilized full-resolution pixels per second;
- timestamps are monotonic nanoseconds in the core contract;
- camera microsecond timestamps are converted exactly and retain their source;
- a pairwise transform name must state its direction, for example
  `current_to_previous`;
- transforms estimated on a scaled motion image are lifted to full resolution
  by matrix conjugation, not by adjusting translation alone;
- keyframe/reference changes are explicit events and invalidate or remap
  downstream windows.

These conventions remain provisional until encoded in documentation and unit
tests.

## Risks and open questions

1. **Radiometric precision:** the current path is 8-bit. Supported sensor raw
   modes and their real noise/throughput tradeoffs are not yet measured.
2. **Timestamp correctness:** SDK metadata is available but unused.
3. **Exposure stability:** automatic exposure changes the statistical meaning
   of residual intensity unless recorded and compensated; fixed settings are
   preferable for initial experiments.
4. **Feature geometry:** live footage produced few Harris features. Their
   spatial distribution was not recorded.
5. **Scene rigidity:** water, clouds, foliage, and parallax may not obey one
   global transform. Feature masks and quality rejection will be necessary.
6. **GPU contention:** full-resolution pyramid construction currently uses
   CUDA, competing with the future synthetic tracker. A smaller PVA-compatible
   motion image should be evaluated while keeping detection full resolution.
7. **Synthetic-tracking cost:** a full-resolution exhaustive velocity grid can
   require tens of billions of samples per window. Search bounds and latency
   must be derived before kernel design.
8. **Local reproducibility:** the Mac Python environment has no declared project
   dependencies yet.

## Phase 1 patch plan

The next patch should be deliberately small and accelerator-independent where
possible.

1. Add a `tiny_target` Python package and immutable `Frame` data contract with
   image, monotonic timestamp, source timestamp, sequence, exposure, gain,
   black level, valid mask, and drop/discontinuity metadata.
2. Add validation for image shape/type, monotonic sequence/timestamps, duplicate
   timestamps, gaps, and explicit bit depth.
3. Add a deterministic recorded-frame source for small `.npy` fixtures plus a
   manifest. Replaying the same manifest must produce identical frame hashes
   and metadata.
4. Add a versioned configuration loader with resolved-configuration output.
   Declare dependencies rather than relying on the Jetson global Python state.
5. Add per-stage timing slots and run-manifest identity fields, initially with
   no image processing.
6. Add unit tests runnable on the Mac without importing VPI, CUDA, OpenCV, or
   the ToupTek SDK.
7. In a separate Jetson-only follow-up, adapt SkyEye capture to populate `Frame`
   from `ToupcamFrameInfoV3`; do not change bit depth or exposure policy until
   camera support is measured explicitly.

### Phase 1 acceptance checks

- The same recorded fixture replays identically twice.
- Sequence gaps and non-monotonic/duplicate timestamps are reported.
- `uint8` and high-bit-depth integer frames retain their dtype without clipping.
- Core unit tests run on both Mac and Jetson without accelerator imports.
- Resolved configuration and run manifest record input identity and random seed.
- Existing PVA benchmark syntax and tracked results remain unchanged.

## Phase 2 preparation

After Phase 1, extract a production PVA motion adapter that returns both source
and destination floating-point points, PVA status, Harris scores, feature-cell
coverage, and timing. The first geometry test should use known translated and
rotated images and feed correspondences into a CPU reference RANSAC estimator.

Do not begin CUDA synthetic tracking until the frame/timestamp contract,
coordinate conventions, reference scoring equation, and a concrete velocity
grid budget are all testable.

## Phase 1 validation results

Phase 1 was validated on 2026-09-03 without opening the camera. The new files
were copied to the isolated Jetson directory `/tmp/seaqr_phase1_20260903`; the
live Git checkout was not used as a deployment target.

- 11 unit/integration tests passed on the Mac.
- The same 11 tests passed under Python 3.10 on the Jetson.
- The FFmpeg integration test verified lossless `gray16le` decoding with an
  explicit timestamp sidecar and timestamp-gap annotation.
- Eight real FFV1 frames decoded as 4784x3190 `<u2` with bit depth 16.
- Two timestamp gaps greater than 150 ms were reported within those first
  eight RAW16 frames.
- Repeating the RAW16 inspection produced the identical frame-set SHA-256:
  `9a7b05dd4d8b24107188ee0e0a25f9559f581c303a5e159665ff1c31d6d1d2f5`.
- Eight real Motion-JPEG frames decoded as 4784x3190 `uint8` and produced the
  expected warnings for lossy pixels and container-derived timestamps.

The RAW16 eight-frame inspection took 9.485 seconds and the AVI inspection
took 5.619 seconds. These are audit-mode measurements that include sequential
FFmpeg startup/decoding and SHA-256 hashing of every full-resolution frame.
They are not estimates of the future steady-state capture or detector rate.

Jetson validation artifacts:

- `/tmp/seaqr_phase1_20260903/raw16_inspection.json`
- `/tmp/seaqr_phase1_20260903/raw16_inspection_repeat.json`
- `/tmp/seaqr_phase1_20260903/avi_inspection.json`
