# PVA Synthetic Tracking

SEAQR's experimental pipeline for detecting and tracking faint point targets in
camera recordings. NVIDIA Jetson PVA tracks background features to estimate
camera motion; CUDA synthetic tracking integrates weak signals along trial
target trajectories. The detector then selects candidates and associates them
over time.

This folder contains a source snapshot prepared on **2026-10-04**, including the
core pipeline, subsequent experimental work, tests, configurations, and design
notes. It is a development snapshot, not a production release or a claim of
general real-time performance. Historical experiment names and results in the
documentation describe particular configurations and datasets.

## Pipeline

```text
RAW16 frames + timestamps
  -> PVA Harris corners and forward/backward PyrLK background tracking
  -> robust global camera-motion estimation (RANSAC)
  -> full-resolution stabilization and validity masks
  -> background subtraction and noise normalization
  -> point-spread-function (PSF) matched filtering
  -> timestamp-aware CUDA shift-and-stack over trial velocities
  -> local clutter normalization, peaks, quotas, and duplicate suppression
  -> Kalman association and confirmation using independent frame evidence
  -> JSON reports and optional evaluation/visualization
```

Only the image used for motion estimation is reduced in resolution. Detection
uses full-resolution pixels. Actual timestamps determine trajectory offsets.
Synthetic tracking means testing possible motion trajectories; synthetic target
injection is a separate tool used to evaluate detection accuracy.

## What is here

| Path | Purpose |
| --- | --- |
| `tiny_target/motion_cli.py` | Main recorded-input PVA + synthetic-tracking pipeline |
| `tiny_target/frame_source.py`, `types.py`, `config.py` | Video/NPY ingestion, timestamps, frame contracts, configuration |
| `tiny_target/motion/` | PVA/VPI features and optical flow, geometry, global-motion fitting |
| `tiny_target/stabilization/` | Image warping and valid-support handling |
| `tiny_target/preprocessing/` | Background and noise models |
| `tiny_target/detection/` | PSF filtering, CPU reference and CUDA integration, candidate selection |
| `tiny_target/detection/cuda/` | CUDA kernel source; compiled libraries are generated locally |
| `tiny_target/tracking/` | Temporal association, Kalman state, and additional motion models |
| `tiny_target/evaluation/`, `*_benchmark.py` | Injection, controlled accuracy tests, and performance reports |
| `tiny_target/dense_screen.py`, `visible_*.py` | Later discovery and visible-point experimental paths |
| `configs/pva_synthetic_tracking.example.yaml` | Portable starting configuration for the core pipeline |
| `configs/evaluation/` | Historical experiment settings, manifests, and injection specifications |
| `tests/` | Unit, integration, native, and experimental regression tests |
| `scripts/` | Experiment runners, diagnostics, audits, native builds, and review renderers |
| `docs/` | Architecture, phase documentation, calibration notes, and experiment plans |
| `legacy/` | Earlier standalone camera/optical-flow/PVA benchmark code |
| `pyproject.toml` | Python dependencies, optional extras, and console commands |

Recordings, generated results, local environments, caches, and compiled
libraries are excluded. Historical documents may reference local absolute paths
or reports under `results/` that are not included. Some experiment scripts also
require frozen external artifacts and exact source hashes; they are retained
for reproducibility and are not general-purpose launchers. Start with the
commands below.

The older code in `legacy/` may additionally require the vendor camera SDK,
GStreamer, and NVIDIA multimedia bindings. Those dependencies are separate
from the recorded-input `tiny_target` package.

## Requirements

For CPU development and the controlled smoke test:

- Python **3.10+** (development validation uses Python 3.12).
- NumPy; OpenCV for image-processing paths; Pillow for optional visualization.
- `ffmpeg` and `ffprobe` on `PATH` for video input and decoding tests.
- A C++ compiler for optional native experimental tests/builds.

For the complete accelerated pipeline:

- NVIDIA Jetson with PVA support; the hardware development target is **AGX Orin**.
- NVIDIA VPI Python bindings with PVA enabled, installed through the Jetson
  software stack. `pip install` of this project does not install VPI.
- CUDA toolkit and `nvcc`; the recorded development environment used VPI 3.2.4,
  CUDA 12.6, Jetson Linux R36.4.7, Python 3.10, and OpenCV 4.10.
- An OpenCV build with CUDA support only if selecting an OpenCV CUDA backend.

See [the historical environment audit](docs/environment.md) for details. PVA
cannot run on a Mac or an ordinary desktop CPU. CPU reference benchmarks remain
useful on those machines.

## Install for CPU development

```bash
git clone https://github.com/imporomptu-del/test.git
cd test/PVA-Synthetic-Tracking
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[vision,visualization,yaml]'
```

Install FFmpeg separately if needed: `brew install ffmpeg` on macOS, or
`sudo apt-get install ffmpeg` on Ubuntu. Run all following commands from this
folder with the environment activated.

### First run: no camera, recording, or GPU required

```bash
mkdir -p results
python -m tiny_target.evaluation_benchmark \
  --mode correctness \
  --output results/cpu_correctness.json

python -m tiny_target.clutter_benchmark \
  --output results/cpu_clutter.json
```

These generate controlled inputs and evaluate the CPU detection/tracking and
clutter-selection paths. They do not exercise PVA or validate real-camera
accuracy. Reports include thresholds, detection/false-alarm metrics where
ground truth is available, and timing. Many tools deliberately refuse to
overwrite output files; use a new output name when rerunning.

## Install and build on Jetson

Use the Python interpreter matching the installed VPI bindings. A virtual
environment with system packages can access Jetson's VPI and OpenCV packages:

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -e '.[visualization,yaml]'
python -c 'import vpi, cv2, numpy; print("VPI, OpenCV, NumPy imports OK")'

python -m tiny_target.cuda_build --architecture 87
```

Keep the Jetson-provided OpenCV installation when it supplies the required
accelerated backends. The desktop `vision` extra installs a headless OpenCV
wheel and is not a replacement for a CUDA-enabled Jetson OpenCV build.

The build command uses `/usr/local/cuda/bin/nvcc` and creates
`build/cuda/libtiny_target_cuda.so`. Override the compiler location with
`--nvcc /path/to/nvcc` if necessary. Architecture `87` is for AGX Orin. Later
experimental CUDA/native libraries have separate builders in `scripts/`; they
are not required for the core example below.

## Run the PVA + CUDA pipeline on your recording

Supply a lossless grayscale recording, ideally RAW16/FFV1, and a timestamp CSV.
The CSV must contain zero-based, consecutive `frame_index` values and integer
`unix_time_ns` timestamps:

```csv
frame_index,unix_time_ns
0,1700000000000000000
1,1700000000100000000
2,1700000000200000000
```

There must be timestamp rows for the frames being decoded. Use actual
acquisition times and set `input.timestamp_semantics` to describe them correctly.
The supplied example records the original development convention of host time
after frame pull/copy.

```bash
cp configs/pva_synthetic_tracking.example.yaml configs/camera.local.yaml
mkdir -p results

python -m tiny_target.motion_cli \
  --config configs/camera.local.yaml \
  --input-video /absolute/path/to/recording.mkv \
  --timestamp-csv /absolute/path/to/recording_timestamps.csv \
  --max-frames 64 \
  --velocity-grid=-3,3,-3,3,1 \
  --omit-points \
  --output results/pva_tracking.json
```

`--input-video` and `--timestamp-csv` must be supplied together. Alternatively,
edit `input.path` and `input.timestamp_csv` in the local config. Relative paths
are resolved relative to the configuration file; the portable example points
to `data/recording.mkv` and `data/recording_timestamps.csv`.

The grid argument is `vx_min,vx_max,vy_min,vy_max,step`, in pixels per second in
stabilized coordinates. This example searches 49 velocities. It is illustrative,
not a calibrated physical speed range. The config's zero-only grid is retained
as a cheap diagnostic default if the override is omitted.

The JSON report contains effective configuration, input/source identities,
motion quality, stage timings, candidate batches, and track batches. `--omit-points`
omits individual background correspondences, reducing report size.

Review these settings for your sensor and scene before interpreting detections:

- Bit depth, saturation level, valid-pixel masks, and timestamp semantics.
- PSF shape and width, background warm-up, and noise model.
- Integration length, velocity range/spacing, and support requirements.
- Raw-score and local-CFAR thresholds, spatial quotas, and track uncertainty.

The example is based on the Phase 12 local-CFAR configuration. Its saturation
code of `65520` belongs to the development recording's left-shifted 12-bit
encoding; do not assume every 16-bit camera uses that value. Motion failures,
timestamp discontinuities, or global illumination events can reset temporal
state, so a short recording may produce no confirmed tracks.

Use `python -m tiny_target.motion_cli --help` for optional injection, threshold
sweeps, and tracking overrides. The installed `tiny-target-motion` command is
equivalent. `python -m tiny_target --config configs/camera.local.yaml --output
results/input_inspection.json` inspects input only and reads paths from the config.

## Tests and verification

The self-contained core integration suite is the recommended first check:

```bash
python -m unittest discover -s tests/integration -v
```

PVA/CUDA tests skip when their hardware or compiled library is unavailable.
FFmpeg/OpenCV-dependent tests require those dependencies. The complete research
suite can be invoked with:

```bash
PYTHONPATH=.:tests/unit:tests/integration:scripts python -m unittest discover -s tests -v
```

The explicit test paths support historical tests that import sibling helpers.
Some complete-suite tests require archived reports, frozen external workspaces,
or native libraries omitted from this source distribution. An absent artifact
can cause a failure rather than a skip; this command is not a promise of a
green checkout without those artifacts. Consult test failures and the relevant
experiment documentation before running a hardware or dataset-specific test.

Snapshot validation on macOS/Python 3.12:

- Both CPU example commands above completed; deterministic accuracy replay and
  the clutter benchmark acceptance checks passed.
- Core integration suite: 47 tests, 42 passed and 5 hardware-dependent skips.
- Full research suite with the explicit `PYTHONPATH`: 4,294 tests, 4,255 passed,
  10 skipped, and 29 errors from omitted historical artifacts/metadata. This is
  not a fully passing research-suite result.
- All Python source compiled; the 1,105 original files in the five project
  directories matched the local development snapshot byte for byte.
- PVA/CUDA execution was not rerun on Jetson for this upload.

## Reading guide and scope

Start with [architecture](docs/architecture.md), then
[PVA motion](docs/phase2_pva_motion.md),
[global camera motion](docs/phase3_global_motion.md),
[stabilization](docs/phase4_stabilization.md),
[preprocessing](docs/phase5_preprocessing.md),
[PSF filtering](docs/phase6_matched_filter.md),
[synthetic tracking](docs/phase7_synthetic_reference.md),
[CUDA implementation](docs/phase8_cuda_synthetic_tracking.md),
[temporal tracking](docs/phase10_temporal_tracking.md), and
[local clutter normalization](docs/phase12_clutter_normalization.md).

The [evaluation documentation](docs/phase11_end_to_end_evaluation.md) explains
how controlled tests, injected RAW16 targets, and unlabeled camera footage
support different conclusions. Later `visible_*` and Phase 20 work follows a
separate visible-point development path; see
[its scope](docs/phase20_visible_baseline.md) before selecting its scripts.

Camera calibration, clutter specificity, labeled real-target validation, and
throughput depend on the chosen path and workload. Neither a synthetic smoke
test nor a historical timing proves deployment readiness. The core runtime
uses Python, NumPy/OpenCV, NVIDIA VPI, and native CUDA; it does not require a
trained neural-network model.
