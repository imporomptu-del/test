# Exact GPU median-selection experiment

This opt-in execution change applies only to the visible 8-bit branch. It uses
the v17 learning-mask helper and the original NumPy tile-noise function; v18 is
not enabled. It does not alter thresholds, coverage, frame rate, tracking or
the separately disabled faint-target synthetic branch.

## What is computed

The existing median5 kernel loads a replicated-border 5×5 neighborhood and
returns its 13th ordered value. Its source contains a complete odd-even sort.
The candidate generates an ascending odd-even merge network on 32 symbolic
wires, with seven positive-infinity padding wires. Constant comparisons are
folded before code generation, then operations that cannot affect output wire
12 are removed. No padding values are loaded or stored by the runtime kernel.

The result is a 202-node min/max DAG over the original 25 floats. This is a
source-level count, not a GPU instruction count: the CUDA compiler can also
eliminate dead work from the old routine. There is no quantization, new float
arithmetic, change of rank, resampling or approximate median.

## Proof boundary and unusual values

A generated C++ verifier evaluates all 2^25 zero/one inputs, 64 cases at once
using bitwise AND/OR. An independent popcount oracle checks whether at least
13 inputs are one. Thresholding a totally ordered finite input commutes with
min/max; therefore correctness for every binary pattern establishes the middle
rank for ordinary finite values, including duplicates. A separate randomized
float test compares against NumPy.

Binary rank proof alone does not establish CUDA bit-level semantics for NaNs,
signed zero or subnormals. A per-neighborhood integer-bit guard sends any such
window (including infinities) through the original sorting routine on the GPU.
Positive zero is ordinary. No CPU fallback, synchronization or device/host
transfer is added. The unusual-value path can be slower because of the guard
and divergent execution; synthetic stress timings retain that limitation.

Compiled candidate and separately built diagnostic versions are checked
byte-for-byte against the frozen GPU library on 259 generated images. Ordinary
finite cases also use OpenCV as an independent pixel oracle. Tests include tiny
shapes, borders/partial blocks, adjacent floats, repeated ranks, signed zero,
raw subnormal/NaN bit patterns, infinities, extreme finite values and native
4784×3190 geometry. Pixel tests, not the rank proof alone, establish the tested
compiled conformance.

## Frozen source and library transition

The builder requires the exact four-source set that produced the current
threshold-first peak-selection library. It changes only the median sorting
block, retains the guarded original block, and builds into new directories.
Warp, Gaussian, resident detector, peak selection, kernel launch geometry and
all arithmetic flags remain unchanged. Production and diagnostic libraries are
separate. Diagnostic event timing excludes copies and pipeline scheduling;
generated production timings include copies. Neither is pipeline FPS.

The candidate config differs from the frozen config only in the CUDA library
path. Launch metadata records the real candidate config/library hashes. An
explicit directional transition binds the old and new binaries to the build
receipt. The video comparator accepts only that transition and preserves exact
non-timing journal, motion, decoder-lifecycle and aggregate-track comparisons.
No launch metadata is edited after execution and no numerical tolerance is added.

Three alternating prefix pairs per workload and four full regressions are the
whole-pipeline gates. Final timing/acceptance conclusions belong in the completed
experiment README. Frozen dependencies and production defaults remain unchanged;
no real-time or new airborne-detection-accuracy claim follows from equivalence.

## Bounded diagnostic follow-up

After the complete timed batch, a separate 128-frame run for each timing clip
may inspect warped median inputs at frame indices 31, 63, 95 and 127. It uses
the explicit verification-download API, counts special values and affected 5×5
windows, and compares frozen/candidate diagnostic-kernel outputs and event times.
No image is saved, no detector state is written, and the complete prefix journal
must still match. Its added download/upload/timing work makes its pipeline FPS
invalid for performance comparison; these runs are excluded from the timed
schedule and reported only as diagnostic evidence. Eight sampled inputs cannot
establish guard frequency throughout all footage or prove a clock/power cause.
