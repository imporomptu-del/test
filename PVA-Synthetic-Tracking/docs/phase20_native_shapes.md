# Exact native shape bookkeeping

This opt-in execution path accelerates the bounded observed-shape stage of the
visible-AVI PVA/GPU experiment. It does not change detection/tracking policy or
accelerate the separate faint RAW16 synthetic-tracking branch. The reference
Python implementation stays available and remains the numerical oracle.

## Scope and invariants

The helper operates on at most 512 already-accepted peaks and their downloaded
17×17 float32 patches. It cannot create threshold seeds, inspect labels, consult
track predictions, or change spatial coverage. Inputs are immutable. It performs
8-connected half-height footprint extraction, conservative pairwise grouping,
asymmetric-neighbor rejection and sparse-coordinate gathering in C++.

The order and semantics match `consolidate_half_height` and `SparseSpatial`:

- Footprints touching the patch boundary or unsupported pixels are rejected.
- Same-polarity peaks merge only when every pair is mutually compatible; a
  transitive chain is insufficient. Spatial ordering is stable, and output groups
  retain original-input order.
- Duplicate windows use the last supplied patch. Global-coordinate reads use
  the first supplied overlapping patch, even for deliberately inconsistent
  patches. Although real gathered patches agree, the public reference's
  precedence is preserved rather than assumed away.
- Region pixels are sorted in raster order. Float64 centroid sums, products,
  divisions, nearest-even rounding, strongest-peak selection and public shape
  dictionaries remain in Python/NumPy with the original arithmetic order.
- A rounded centroid in a hole/valley is not a new measurement. Every original
  member is preserved exactly as in the reference.
- Candidate scores, measured support, learning protection, tracking, thresholds,
  caps, resolution, cadence and PVA/CUDA operations are unchanged.

## ABI and bounded execution

`scripts/phase20_native_shapes.cpp` exports ABI version1 and a single batch
function. The Python adapter validates typed, sized inputs before passing
contiguous arrays through ctypes. The C++ entry point rechecks counts, dimensions,
coordinate bounds, polarity and pointers. Exceptions cannot cross the C ABI.
Each call owns its scratch; there are no persistent pointers to caller arrays.

Output capacities are fixed by the input count: at most 512 groups/members and
512×289 region pixels/values. Each accepted connected component is within its
17×17 patch; unions cannot exceed the sum of the component sizes. BFS marks
pixels before enqueueing; its queue has 289 slots. Border detection precedes
neighbor access. No approximate arithmetic or fast-math is enabled.

The full mask is read on the CPU as before, but the old Python sparse-index
construction, footprint sets, grouping loops and per-feature sparse searches
are bypassed. Centroid reductions intentionally remain in NumPy: changing their
reduction order would require a separate conformance proof.

## Explicit activation and provenance

Both `native_shape_library` and `native_shape_library_sha256` default to null.
Activation requires an explicit hashed library, the CUDA-resident detector and
the existing `mutual_half_height_r8` measurement mode. Missing files, hash/ABI
mismatches and native failures fail closed. There is no automatic compilation,
library discovery or fallback during processing.

Build explicitly with the existing system compiler:

```sh
python3 scripts/build_phase20_native_shapes.py --output /absolute/new/path/libseaqr_shapes.so
```

The builder refuses existing outputs and records the compiler version, flags,
machine, source hash, builder hash and binary hash. No packages or system settings
are changed. The experiment freezer accepts this target-machine build record via
`--native-shape-build`, pins it into the frozen config/manifest and includes the
native source, adapter, builder and numerical verifier in the runtime archive.

Launch records identify the native CPU helper separately from the unchanged
CUDA library. The exact-run comparator permits only this explicit execution
choice/library hash and the CUDA library path to differ; **all non-timing journal
fields still have to match**. It does not exempt shape fields or loosen numeric
tolerances. The finalizer verifies the downloaded binary and build record
against the freeze. Missing/changed provenance is a failure.

## Validation and interpretation

The unit suite covers random shapes, invalid support, valleys, hollow shapes,
duplicate/overlapping patches, polarity/order/ties, subnormal thresholds,
half-pixel rounding, noncontiguous inputs, native coordinates, the candidate cap
and fail-closed validation. A separate 600-case verifier loads the previous
shape/sparse implementations from their hash-checked frozen archive. Alternating
paired synthetic timings include sparse setup and are not pipeline FPS.

The separate `tests/native/phase20_shapes_sanitizer.cpp` harness exercises 800
bounded cases under AddressSanitizer and UndefinedBehaviorSanitizer, including
empty/full-cap inputs, duplicate seeds, small images and border support. It
checks input immutability, output canaries, complete group membership and sorted
pixel bounds. This is a memory-safety check, not a substitute for numerical
conformance or a proof for arbitrary C callers bypassing the typed adapter.

Full-video acceptance independently checks complete journals against both the
original accuracy reference and the immediate speed predecessor. Source hashes,
cadence and frame counts must match. Controls remain unlabeled review material;
output identity does not establish airborne precision, recall or generalization.

The experiment also retains synthetic reference-CPU detector/tracker/feedback
checks; these run with native shapes **disabled** and must not be represented as
GPU/native closed-loop evidence. The complete PVA/GPU video comparisons supply
that execution-path evidence.

## Remaining device handoffs

This change does not alter the CUDA binary or remove CPU/GPU transfers. The
compiled helper consumes the same bounded downloaded patches. Large masks,
sample statistics and synchronous stage boundaries remain in the integrated
GPU path. They require separate kernel/transfer measurements and a separately
validated change. No future transfer-related speedup is assumed here.

The diagnostic-only `phase20_cuda_transfer_probe.cu` includes hash-checked,
unchanged integrated CUDA sources, intercepting host `cudaMemcpy` and explicit
`cudaDeviceSynchronize` calls. It records source call sites, bytes and host API
durations, with CUDA events around copy calls. Host copy-call time includes
waiting for previously queued kernels; copy-event intervals can include API and
queue gaps. Neither should be relabeled pure DMA bandwidth, GPU utilization or
kernel occupancy. Synchronization calls have no copy-event measurement.

`build_phase20_transfer_probe.py` requires the original kernel freeze's SHA256
and records compiler/source/binary hashes. `probe_phase20_cuda_transfers.py`
replays only the authorized chunk126 prefix, enabling instrumentation for
frames 72–95. It checks **all non-timing journal fields for all 96 frames** against
the completed benchmark. The diagnostic library and output stay separate from
the accepted speed run. Always run this after the one-worker benchmark; its
instrumented FPS is not a throughput result.

Experiment evidence is under
`results/tiny_target/phase20/native_shapes_v8_20260914/`. Its final README and
verified summary, once present, are authoritative for measured throughput and
remaining accuracy limitations.
