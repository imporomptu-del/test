# Next accuracy work after V37 — proposal, not a frozen experiment

V37 is complete as a bounded diagnostic, not an output improvement. It exposes
two structural prerequisites that exclude useful cases: local texture is
needed to resolve a small registration patch, and an actual same-ID
measurement is required at exactly f-1. Neither prerequisite is appropriate
for all real small moving points. Do not loosen V37's checks in place.

## Implementation order

1. Preserve and separately annotate the reviewed 0029 frames 12–24 moving
   bright feature as a **class-unknown image-feature** regression. Review
   unmarked source before defining positions/visibility/uncertainty; do not
   copy tracker centers into truth. Ambiguous frames remain ambiguous. This
   must be a new reference version, not an alteration of the frozen 285/28/24
   samples or an airborne-class label.
2. Implement a bounded causal source buffer and fixed lag bank `{1,2,4,8}`.
   Extract past source pixels even when no prior track measurement exists.
   Preserve actual lag, timestamp, transform segment, reset and validity
   information. Never substitute a prediction or silently change the lag.
3. Start from the existing logged global camera transforms. Carry their
   accepted-chain, error, inlier and spatial-support diagnostics. Test a
   predeclared residual-translation envelope and report evidence sensitivity,
   not a central-point-score-maximizing alignment. Treat that envelope as a
   stress test, not calibrated camera covariance. Optional local refinement
   must stay separate from the global-only path.
4. Keep per-lag current compact-point, background/edge and optional measured
   previous-point evidence separate. Missing prior association is not proof
   the prior point was absent; do not invent its location from a velocity
   extrapolation. Mark potential previous-point ghosts and cancellations
   explicitly. Do not use "best lag wins," positive nested gain, or unknown
   fallback as a successful verification.
5. Freeze code, tests and selection before new extraction. First use the same
   bounded 358 observations and all original denominators, plus a separately
   reported new regression only after step 1 is complete. Report availability
   and evidence stability before any proposed eligibility change.
6. Only after the features are demonstrably usable, define and freeze a
   limited output policy and test for **no new losses of known visible image
   features**, including edge-adjacent and intermittent cases. Independently
   review workload reductions. Keep actual airborne accuracy separate until
   physical-class positives/negatives are adjudicated.

## Required synthetic challenges

- Bright/dark and weak points; blinking and missing associations.
- Slow/co-moving/stationary points, changing direction and acceleration.
- Paired or slightly extended compact features, not only isolated Gaussians.
- Genuine points on strong edges and independently deforming cloud texture.
- Quantization, interpolation, exposure changes and camera-error sensitivity.
- Source borders, resets, missing lag frames and invalid transforms.
- Explicit causal processing and unknown propagation; no future frames.
- Independent numerical replay and reference/array provenance checks.

## Limits and non-goals

This is not a speed phase, a RAW16 experiment, a Jetson deployment, a holdout
scan or an instruction to change production defaults. A downstream verifier
cannot recover a detector candidate that was never born. No hard lower-image
mask or target-size restriction should be introduced from these examples.
Neither temporal persistence nor compact appearance establishes airborne
class. Reduced overlay clutter is useful only when reported separately from
verified detection accuracy.

No V38 implementation, source extraction, output change or accuracy gain is
claimed by this proposal. Its exact features, uncertainty envelope and
decision criteria still need pre-extraction review and freezing.
