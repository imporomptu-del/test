# V48 — repair the synthetic observation contract; unchanged V47 mathematics

## Scope and predecessor

The user approved correcting V47's synthetic renderer, rerunning independent
audits, and then (only if those pass) the frozen 0029/0126 diagnostic comparison.
Preserve every V47 file and its failed audit. No production, threshold, gain,
error-bound, template, support, fixed-alternative or classifier changes.
No SSH/Jetson action, RAW16, sealed holdouts, new video decoding or journal reads.
Efficiency remains paused.

V47 completed 54 cases / five calculations but its independent audit rejected
`correlated_guard_error_extremes`: multiplication rounding pushed 73/141 used
current pixels outside the declared +/-0.5-DN error box. The exact guard solver
correctly returned unknown. Its original observation arrays, declared truth,
results and failure remain intact and hash-pinned, never silently relabeled.

## Predeclared renderer repair

Retain all 54 cases and their order: six V46 controls, 28 V46 stress cases and
20 V47 guard cases. All arrays and metadata of 53 cases must remain identical.
Change only the current image of `correlated_guard_error_extremes`, on the fixed
full annulus 40 <= Chebyshev radius from (64,64) <= 56. This is not a subset
chosen using guard feasibility, source scores or detector results.

Interpret the existing binary64 background B and gain g=1.2 as exact rationals.
For each annulus pixel, form the exact latent value g*B and the intended signed
boundary observation g*B - .5*s, where s is the original checkerboard sign.
Choose the nearest representable observation to that intended endpoint that is
inside the unchanged exact interval [g*B-.5, g*B+.5]. If nearest rounding is
outside, move inward; verify containment exactly. Fail if no finite observation
can represent the contract. No epsilon, enlarged uncertainty, noise averaging,
gain change, altered detector or result-dependent selection is allowed.

The current compact-source contribution is zero in stored binary64 throughout
this annulus; verify that fact. Keep all history images, source/core pixels,
polarity, trajectory, forecast and geometry unchanged. Add explicit repair and
actual bounded-error metadata only to the repaired case. Realized errors are
boundary-near with correlated signs; do not claim every magnitude is exactly .5.

Before any calculation, verify full-annulus current observations against exact
latent values and the fixed .5 bound, and prior observations outside all fixed
prior-center exclusion footprints against their declared witness. Save a
hash-bound `rendering_contract.json` before `score_start.json`. A separate
auditor must independently reconstruct the corrected pixels and exact witness;
it must not reuse the repair function as its numerical oracle. Keep a regression
showing the original stored invalid witness is still rejected.

## Freeze, execution and audits

Pin the V47 synthetic receipt, original failed synthetic audit and original math
audit. Verify all inherited source/synthetic dependencies. Freeze all V48 source,
tests, plan, ordered case metadata, all 54 archives and in-memory fingerprints
before scoring; recheck after. Truth and rendering witnesses are not adapter
arguments. Run the same four original arms and unchanged V47 guard-gain method
on every case. Require bit-for-bit V46 four-arm reproduction on the original
34 cases, plus exact V47 five-arm results on all 53 unchanged cases. Preserve all
five outputs for the repaired case regardless of the resulting sign/availability.
No desired positive count is a readiness threshold.

Repeat the independent V47 mathematical audit using the unchanged 24-system,
768-perturbation matrix; only its output location and bound dependency inventory
change. Require full unit regression and a completed independent V48 synthetic
audit with no issues. Freeze audit code/dependencies and bind its report to the
exact synthetic completion receipt. Unit success does not override audit failure.

## Conditional real comparison

Only after audited readiness, run unchanged V47 mathematics on the exact V46
ledger: 1,211 states, 509 explicitly allowlisted cached 8-bit-derived packets from
0029/0126, and 702 history unknowns. No other packet bytes are authorized by old
receipt maps. Reuse all four saved V46 arm outputs exactly, rather than rerunning
or tuning them. Preserve all 355 overlapping references, all alternatives,
original strict assignments, original denominators and the frame216 miss.

Use a fresh V48 output, never overwrite a scientific run. Hash allowlisted packets
and all code before scores and recheck afterward. Independently audit the
completed JSON accounting, reasons, reference assignments and comparisons.
Report the full denominator; any original background-only subgroup is diagnostic
only. No optimistic reassignment or removal of inconvenient unknowns.

## Interpretation

The method remains conditional bounded-gain partial regression, not the original
unrestricted-background least-squares quantity and not a physical detector.
Same-frame outer guard calibration is not prior-only. Its necessary-constraint
relaxation cannot certify guard cleanliness, full affine feasibility or gain
transfer to the source core. Sparse sampling and bilinear-brightness blind spots
remain. Actual sensor/model error bounds are not calibrated by synthetic success.
The inherited current-whole-frame camera-motion caveat remains. No airborne
accuracy, false-alarm, generalization or production noise-removal claim follows
from positive numerical intervals. Unsupported evidence remains unknown.
