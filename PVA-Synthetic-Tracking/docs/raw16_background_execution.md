# Bounded RAW16 execution work

## Scope

The recent AVI pipeline speed numbers do **not** measure the RAW16 pipeline.
The Phase 19 RAW16 entry point still uses FFmpeg/FFV1 decoding, native uint16
frames with timestamp sidecars, PVA motion, CPU cubic stabilization, CPU
background/filtering and CUDA shift-and-stack. FFmpeg already runs as a separate
process; AVI's one-frame OpenCV decode overlap is not a drop-in optimization.

The test runner in `scripts/profile_raw16_efficiency.py` accepts only development
clips 0029 and 0040 at their exact RAW16 paths, and only 32–128 consecutive frames
from the beginning. It never loads the discovery/holdout split or enumerates the
media directory. The current experiment uses 64 frames per clip. The old 80-clip
batch is not restarted, and no sealed holdout recording is used.

## Execution-only change

`DenseScreenConfig.background_execution` defaults to `indexed_reference`.
The opt-in `masked_ufunc` implementation removes repeated boolean-index
gather/scatter arrays in whitening and causal background updates. Two reusable
float32 scratch images retain the reference's separate multiplication, square,
addition and assignment steps. There is no arithmetic fusion, intensity
quantization, frame dropping, spatial subsampling or threshold change.

The extra reusable state is two float32 images (about 70 MiB for the frozen
4784 × 1920 crop), replacing several transient gathered arrays. Scratch contents
are read only where initialized during the same update. Background state still
resets on a stabilization-segment change; uint16 support counts still saturate.
The source frame remains immutable. PVA, warp, point-filter coefficients, CUDA
binary, candidate selection and tracking policy are unchanged.

## Evidence separation

- `profile` wraps stages and records cProfile call costs. Nested stage costs
  include their children; exclusive costs avoid double counting. These are
  diagnostic measurements, not benchmark throughput.
- `audit` hashes every source/stabilized/cropped image, motion correspondence
  array and transform, background state, point-filter response and mask, CUDA
  window surface, and candidate-ranking surface. It also records all candidates,
  active/qualified synthetic tracks and the final non-timing report. Dtype and
  shape are part of each array identity; one DN of RAW16 intensity matters.
- `timed` invokes the uninstrumented production entry point. The existing
  production timer includes decoder startup/teardown, but excludes construction
  of the source/screener, final report assembly and serialization. This is a
  short-file-prefix measurement, not a sustained live-camera rate.

`scripts/compare_raw16_efficiency.py` checks matching source/configuration/code
provenance, complete frame and window counts, PVA errors and exact non-timing
output. Audit mode additionally requires all intermediate records to match.
It excludes only explicitly listed timing/environment fields, never timestamps,
window duration, scores, support counts or detection decisions.

`scripts/repeat_raw16_efficiency.py` requires successful array audits first and
uses three alternating AB/BA/AB pairs on the eligible clip, one video worker, 600-second child
timeouts and a process lock. It records a separate owned `tegrastats` process,
stops only that process, and verifies each timed result against the audited
configuration/code/output. It never overwrites an existing result directory.

The first 64 frames of development clip 0040 exercise the full frozen pipeline.
Clip 0029's first 64 frames instead produce 63 rejected motion fits (insufficient
correspondences), repeated background resets, and **zero synthetic windows**.
The end-to-end comparison correctly rejects that sample. Its before/after audit
hashes match, but that demonstrates matching suppression behavior, not successful
detection. It is excluded from speed claims; the gate is not weakened to admit it.

Media-free tests cover native uint16 and interpolated float32 inputs, adjacent
16-bit levels, normal and protected-outlier updates, dark/saturated samples,
changing and empty masks, warmup, segment resets and saturated support counts.
They require byte-identical state and responses, not approximate numerical error.
Evidence-comparison tests reject changed configurations, package hashes,
candidate counts, PVA errors and runs without synthetic windows.

## Accuracy limits that speed work does not resolve

The frozen RAW16 configuration searches only the **upper 1920 of 3190 rows**.
Its coherent synthetic branch searches bright targets and a narrow ±3 px/s
Cartesian motion grid. The per-frame bright/dark event branch is disabled.
Those are older discovery-policy limits, not the newer AVI detector's coverage.
Faster execution does not repair a target excluded by geometry, polarity or
motion range, and successful execution does not establish real-object recall.

RAW16 and AVI files with the same numeric suffix are not automatically aligned.
The inspected RAW16 files have a nominal 10 fps container rate, while their
sidecar intervals are nonuniform. The sidecar timestamps are retained for motion;
they should not be equated with precise sensor-exposure timestamps. Establish
actual footage/time alignment before reusing AVI annotations.

The appropriate next accuracy step is a separately versioned, full-coverage RAW16
baseline with verified RAW-specific examples and controls. After freezing that
policy, the large remaining CPU filtering/warp stages are candidates for GPU
work, each with a separate numerical and detection-equivalence gate. Do not
promote this legacy discovery optimization as a finished RAW16 detector.
