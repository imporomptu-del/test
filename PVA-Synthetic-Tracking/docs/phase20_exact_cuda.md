# Exact cubic CUDA execution for the visible-point baseline

This is an opt-in execution backend for the native **8-bit visible AVI** branch.
It does not enable the independent faint RAW16 synthetic-tracking branch, change
the detector thresholds, or establish that a proposal is an airborne object.
The existing CPU execution remains the default.

## What changes

`VisibleConfig.stabilization_execution` accepts:

| Value | Stabilization and detector input |
|---|---|
| `reference` | Existing full-resolution CPU cubic warp; existing detector path |
| `cuda_cubic_host` | Conformant CUDA cubic warp, followed by a host image round trip |
| `cuda_cubic_resident` | Conformant CUDA cubic warp and Gaussian filter; the resident detector borrows their device buffers |

Both CUDA modes require `motion_backend="pva"`,
`state_update_backend="cuda_resident"`, `spatial_filter_backend="cuda_median5"`
and an explicit `cuda_median_library` pointing to the combined library.
The frozen motion model must be **camera translation**, with CPU cubic
interpolation and zero constant borders as its reference. This restriction
concerns camera motion, not the direction or speed of a tracked object.

The original full image is searched. There is no frame dropping, image resizing,
new crop, interpolation downgrade, threshold relaxation or candidate-cap change.
Nearest-neighbor validity masks and both support erosions retain their reference
semantics. Masks, sampled tile statistics, shape grouping, and tracking still
use the CPU. This is not an entirely GPU-resident application.

## Numerical contract

The cubic coefficient table is obtained by applying the installed CPU remap
operator to sixteen unit basis images at its existing 32-by-32 phase grid.
The translation coordinate maps reproduce the CPU's block-relative arithmetic
before that grid is selected. No additional motion quantization is introduced.

Explicit round-to-nearest multiplication/FMA operations match the installed
OpenCV 4.10 ARM64 cubic accumulation schedule, including its separate border
path. The Gaussian5, sigma0.8 implementation preserves reflect101 borders and
the different SIMD, two-lane and scalar accumulation orders of that build.
The library is compiled without fast math or implicit FMA contraction.

These are deliberately build-specific numerical implementations, not a claim
of equivalence to every OpenCV build or arbitrary floating-point environment.
Startup conformance compares 32 synthetic warp/mask cases and, for resident
integration, 33 Gaussian widths. Failure refuses the requested backend; there
is no silent CPU fallback. The launch journal records the compiled library hash,
OpenCV build hash, coefficient-table hash and conformance result.

## Ownership and scheduling

`CudaWarpFrame` is a distinct, single-use device-frame ticket, not an ndarray or
the normal `Frame` contract. A generation counter rejects overwritten or closed
workspaces. The original validity mask must accompany it. The detector checks
that both handles belong to the same compiled library and consumes the ticket.
There is no implicit image download or Python-visible raw device pointer.

CUDA kernel arguments contain non-owning, trivially-copyable views. Owning host
handles release allocations explicitly; a kernel argument copy cannot destroy
them. The combined prepare operation borrows the warp buffers only during its
synchronous execution. Background/variance state remains owned by the detector.

This implementation is serial. The full-resolution float image and Gaussian
output no longer travel through the CPU between these stages, but validity masks
still make transfers. Performance timings include required transfers and
synchronization. Gaussian work moves into the motion/warp timing in resident
mode, so compare the sum of motion/warp and detection when comparing stages.

## Build and verification

On the tested Jetson AGX Orin, from an isolated frozen runtime directory:

```sh
/usr/local/cuda/bin/nvcc -O3 --fmad=false -arch=sm_87 -Xcompiler -fPIC \
  -shared scripts/phase20_cuda_integrated.cu -o libseaqr_integrated.so
```

The build includes the warp, Gaussian, median and resident-detector interfaces.
`scripts/prepare_phase20_integrated.py` creates the source/configuration archive
and an explicit four-source development allowlist. Its single-worker batch
first compares both 230-frame modes against the accepted reference, then runs
complete development clips 0126, 0029, 0055, and 0082. It stops on a parity failure,
unexpected source hash, existing output or child-process failure. It does not
enumerate the media directory, read labels, tune settings or retry automatically.

`scripts/analyze_phase20_integrated.py` scores completed results separately,
using frozen reference labels. Predictions never count as measured hits.
`scripts/diagnose_phase20_light_drift.py` reports coordinate behavior in the
preselected light-field review regions; it does not classify responses or
implement a new rejection rule.

The experiment archive is the exact executed version. Preserve it and its output
journals when making subsequent source or diagnostic-reader cleanups. Always use
new output directories for another execution.

See `results/tiny_target/phase20/efficiency_v4_20260914/` for conformance results,
the frozen runtime, benchmark outputs, and final accuracy/coverage assessment.
