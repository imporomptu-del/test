# Exact native tracking geometry candidate

The v20 candidate changes one guarded block of the frozen Kalman manager's
update function. It fuses residual subtraction, two-coordinate squared sums,
square roots, inclusive position/velocity gates, rejection counters and the
any-survivor check into a strict-arithmetic C++ loop. Float64 storage and
operation order are retained. In position-only mode the original empty velocity
slice has zero norm; the candidate produces the same zero. It does not replace
Euclidean gates with squared comparisons, reorder measurements or tracks, skip
covariance validation, or change subsequent likelihoods and assignment.

The adapter requires the frozen module SHA256 and exactly one matching source
block. It records the transformed method SHA256 and runs only in the isolated
experiment. The original module and GPU libraries remain unchanged. The launch
snapshot alone does not describe the adapter: retain the v20 receipt, generated
gate, build receipt and freeze file with each trial.

The fast path accepts aligned C-contiguous float64 candidate arrays with four
columns and four-element means, dimensions 2/4, and bounded candidate counts.
Nonfinite thresholds/means/residuals, residual magnitudes above 1e150 or nonzero
magnitudes below 1e-140 use the unchanged NumPy reference. Unsupported layouts
and dtypes also use that reference. The magnitude bounds prevent square overflow
and underflow on the native path; unusual values preserve NumPy behavior rather
than being clipped or silently discarded. FMA contraction and fast math are
disabled. Native output arrays are independently owned and cannot be overwritten
by the next call. Inputs are never changed.

This is an empirical bit-exactness contract for the tested host/toolchain, not a
proof for every CPU/NumPy combination. Generated tests cover both dimensions,
empty/tiny/large batches, zeros, ties, adjacent gate boundaries, large/small
finite residuals, nonfinite/exceptional-value fallbacks, unsupported layouts and
complete generated tracker sequences. All residual/norm/mask bytes and gate
counts must agree. Replays compare complete non-timing track output, including
covariances, quality evidence, lifecycle, resets, capacity competition and
assignment. Further full-video journal equivalence is required before acceptance.

Fresh baseline profiles measure tracking Python calls and host accelerator call
boundaries. The latter combine GPU execution, copies and waits; they do not
isolate synchronization overhead. No synchronization was removed in v20 because
these measurements do not justify changing ownership/error/completion boundaries.

The source audit found these concrete producer/consumer boundaries:

| Native call | Returned data/state | Immediate dependency |
| --- | --- | --- |
| `seaqr_resident_prepare_warp` | sampled residual values | CPU tile-noise statistics |
| `seaqr_resident_select` | peak records and counts | CPU peak decoding and shape seeds |
| `seaqr_resident_patches` | 17×17 patches | CPU/native shape consolidation |
| `seaqr_resident_finish` | updated background/variance; completion/error status | correct next-frame state and current-call failure reporting |

The first three use blocking device-to-host copies; deleting the waits would let
consumers read unfinished data. The final call has an explicit device completion
check. Deferring it would require a separate design for ownership, error timing,
close/drain behavior and overlap; its complete host-call duration is not all
removable overhead. Neither `nsys` nor `nvprof` was found on the tested Jetson
command path, and nothing was installed. Detailed GPU timeline attribution
therefore remains unmeasured rather than being inferred from these host times.

Only the ordinary visible 8-bit path is in scope; RAW16 remains paused. The
existing v17 mask helper stays enabled in both timed arms; v18 and v19 remain
disabled. Candidate performance must be judged by alternating whole-pipeline
timings, not its generated helper microbenchmark. No default deployment or
general airborne-accuracy claim follows from parity on development clips.
