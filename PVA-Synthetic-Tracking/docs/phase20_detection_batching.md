# Exact detection execution batching

This revision changes execution, not the detection or tracking policy. It builds
on the short-coast appearance experiment, whose remaining duplicate-track and
missing-measurement issues are still open. It does not increase shape-merging
radii, relax thresholds, change candidate caps, skip frames or crop the detector.

## Noise statistics

The resident detector downloads the same float32 samples at the same positions.
Tiles are grouped by the number of valid samples. Every row in a group is a
complete original tile sample set: no padding values, invalid pixels or samples
from neighboring tiles enter its median or median absolute deviation (MAD).

One batched median call replaces many separate calls. Each batch is a new private
scratch array, so partitioning and absolute deviations can reuse that memory
without mutating the download or support mask. The median center and deviation
use the same input precision. The MAD is then promoted to float64 before the
1.4826 scale factor and noise floor, exactly as in the previous Python loop.
Device statistics are rounded to float32; aggregate sigma telemetry retains the
original float64 calculation. Empty tiles retain center zero and the noise floor.

The finite-floor/NaN behavior is also preserved. Neither version's handling of
nonfinite synthetic inputs is a claim that such inputs are valid camera data.
The runtime's input and motion-validity checks remain in place.

The tradeoff is bounded extra scratch memory proportional to the existing
downloaded sample count. At the tested native layout there are about one million
samples; a float32 copy is about four megabytes. Samples are never expanded to
a native full-image float64 noise workspace.

## Observed shape measurements

Sparse patch coordinate IDs are generated in a broadcasted batch instead of
rebuilding 17 × 17 grids in a Python loop. Patch/raster order, boundary rules,
first-pixel precedence, and last-duplicate-window behavior remain unchanged,
including deliberately inconsistent overlapping patches in tests.

The shape centroid reuses its identical weight-sum denominator for x and y.
Coordinate products and reductions remain float64 and retain the original order;
no approximate reduction or fast-math kernel is introduced. Bounding-box extrema
are reused; the first and last rows of the already sorted support provide the
y bounds. Connectivity, half-height support, asymmetric-neighbor rejection,
hollow-shape rejection, grouping order and output support coordinates are unchanged.

## Evidence required before a speed claim

The local unit tests cover masked/empty/edge tiles, odd/even sample counts,
float32/float64 arithmetic, nonfinite reference semantics, non-mutation, and
overlapping/duplicate sparse access. A hash-checked frozen implementation is the
shape oracle for randomized streaks, holes, invalid support and duplicate seeds.
Closed-loop CPU detector/tracker/feedback checks remain separate from these
component tests.

Jetson component tests repeat old/new timings in alternating order after warmup.
They are synthetic component measurements, not end-to-end throughput. Final
promotion additionally requires all four authorized full clips to match every
non-timing journal field under the same PVA/GPU configuration. Timings are
compared against the immediate preceding speed revision, while accuracy is still
checked directly against the fixed original accuracy trial.

The experiment is `results/tiny_target/phase20/detection_efficiency_v7_20260914/`.
Its frozen runtime, component tests, full-run journals and verification reports
record the executed implementation. No sealed holdout is used for this work.
