# Threshold-first CUDA peak selection

This is an opt-in execution experiment on the visible 8-bit AVI branch, not a
change to detection/tracking policy and not an acceleration claim for faint
RAW16 synthetic tracking. Default configurations are unchanged.

## Single device-code change

For each eligible pixel, the existing selector requires both:

1. The unchanged temporal and spatial threshold predicates pass.
2. No pixel in the unchanged 5×5 neighborhood has a strictly greater absolute
   temporal residual.

The frozen reference checks (2) before (1). The candidate checks (1) before (2).
Both predicates only read immutable per-selection inputs. Neither changes state,
counts or ranking. Only pixels satisfying both reach the original counter,
score calculation and top-k insertion. Threshold equality remains accepted;
plateaus, polarity, border checks, score/y/x ordering, caps and all output fields
remain unchanged. Detection resolution and cadence remain native and unchanged.

This optimization is independent of clip identity, coordinates, appearance and
labels. Its speed benefit depends on how many pixels fail thresholds early. A
dense above-threshold scene may see little benefit; it must still retain every
candidate the reference retains.

## Build boundaries

`scripts/build_phase20_peak_gate.py` requires the exact frozen four-file CUDA
source set. It replaces one exact, uniquely occurring source block in a fresh
generated directory. Original CUDA sources are not edited. The compiler keeps
`--fmad=false` and the same architecture/optimization flags; no fast math is added.
The candidate retains the existing integrated library ABI and launch geometry.
Missing, duplicate or changed reference source blocks fail closed.

Test builds expose bounded state seeding and device-event timers. These extra
entry points are **absent** from the non-diagnostic candidate library. They are
not runtime fallbacks. The native CPU shape helper is reused unchanged.

## Validation and promotion boundaries

- Media-free tests compare every byte in peaks, including invalid slots and
  floating-point fields, and every count against the frozen kernel. Small cases
  also use an independent scalar selector. Cases cover tiny and partial tiles,
  cold state, masks, threshold equality/adjacent floats, ties, sparse and dense
  scenes, and candidate caps.
- Separate event-only diagnostic libraries bracket launches without inserting
  per-kernel synchronization. They preserve kernel bodies and are not used for
  pipeline FPS claims. Profiling is bounded to frames 72–95 of a 96-frame prefix.
- A frozen four-clip full-video batch compares **all non-timing journal fields**
  to the fixed accuracy reference. A downloaded independent comparison also
  checks the immediate speed predecessor. Each candidate binary transition is
  explicit, directional and bound to SHA256 build provenance. An unspecified
  or unexpected GPU binary change still fails. No fields gain a tolerance.
- Repeated before/after runs use two predetermined 128-frame prefixes, three
  alternating pairs each. They measure end-to-end execution, not just kernels.
  They are not repeated full-video or locked-clock distributions. No frame
  skipping, thresholds, clocks, power settings, services or drivers are changed.

Exact equivalence does not establish ground-truth accuracy or generalization.
Known missed measurements, split tracks and the failed strict accuracy gate
remain separate and must not be hidden by a successful speed test. Sealed
holdouts remain untouched. No live deployment or production promotion occurs.

Evidence is collected under
`results/tiny_target/phase20/kernel_efficiency_v9_20260914/`; use its final README
for observed timings and completion status, not estimates from this contract.
