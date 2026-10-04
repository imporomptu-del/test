# V44 — synthetic-only localized contrast preflight

2026-09-25. Authorized follow-up to V43. Do not edit frozen V43 files, use real
media/caches/journals, access holdouts, run remotely, change production, or resume
efficiency/RAW16 work. This is a mathematical and synthetic integration experiment,
not a detector accuracy evaluation or new threshold search.

## Contract

Measure a signed localized source coefficient relative to an exact caller-supplied
affine brightness subspace. Eliminate that subspace before fitting the uncertain
background/fixed/source columns. Estimate a coefficient interval under declared
elementwise deterministic response/design perturbations; do not require an absolute
background prediction accurate to 1 DN. Bounds are neither camera calibration nor
statistical confidence, and no independence/sample-count reduction is assumed.

For Q = I - P P+, scale each column by its nominal projected norm ||Q X_j||,
then let a = Q[Z,m] in those units and b = Qy. Scales stay fixed under perturbation. For a
full-rank nominal design, theta = a+ b, r = b-a theta, and h is the last row of a+.
The source coefficient is theta_last divided by the source column scale. Raw
scaled elementwise design bounds U also bound the projected error in spectral
norm: eta = ||U||2. Require q = smin(a)-eta > 0 and numerical rank support.

The exact coefficient-change identity is

    delta_theta_last = h(e-E theta) + (h'-h)(r+Qe-QE theta).

Here hQ=h, and h' is the perturbed dual row. Thus a deterministic bound is

    |h| dot (response_bound + U|theta|)
      + D * (||r|| + ||response_bound|| + ||U|theta|||),

divided by the source scale, with D bounding ||h'-h||. Use the smaller of the
V43 global pseudoinverse-change bound eta/q² + eta/(q*smin) and the independently
derived dual-row bound below. If h as a column equals a z, z=(aᵀa)^-1 e_last,
orthogonal decomposition into col(a') and its complement gives

    D_row² = (||Uᵀ|h|||/q)² + ||U|z|||².

Compute z with stable factorization, not explicit normal-matrix inversion.
Unsupported rank or unrepresentable bounds yield unknown, no operative interval.
An interval containing zero is available but inconclusive; a positive interval
certifies only a conditional fitted coefficient, not a moving or airborne object.

Exact affine columns may be orthonormalized without changing their span. Reject
rank-deficient affine bases. Remove nuisance columns only when their uncertainty
is exactly zero AND they are literally zero or exactly +/- an identical supplied
affine basis column. Other near-affine columns remain unsupported, even with zero
uncertainty: a tiny nonzero departure can carry the source with unbounded nuisance
coefficients. Numerical tolerance is not proof of redundancy. Never drop an
uncertain column to improve availability. Roundoff is separate from the admitted
image perturbations and is not a calibrated noise model.

## Two separate levels of synthetic evidence

1. Predeclared oracle-vector cases: ordinary source, background only, uniform/plane
   brightness changes, fixed flicker away from/overlapping the source, source plus
   fixed flicker, uncertainty containing zero, and metadata-only missing-history/
   unknown-background/slow-hover/curved-path scenarios. Inputs and truth are synthetic;
   supplying a known template does not validate its discovery or association.
   Identical image inputs with moving-source versus sequential-fixed-emitter
   interpretations must produce identical numerical outputs and unknown motion.
2. Causal-image probe: the unchanged protected V43 component builder and component
   error propagation run on synthetic 8-frame histories plus a current patch.
   Current intensity values cannot choose template position, contents or support;
   the finite-pixel validity mask may intersect the common support, which is then
   fixed for the uncertainty calculation. Compare
   current source present/absent/shifted and affine changes at fixed prior geometry.
   Record protected-background ambiguity and all fixed-geometry/used-stamp limits.
   The prior measured centers remain ideal supplied history, not validated tracking.

Keep numerical availability, interval sign, causal provenance, fixed-alternative
coverage and physical-class status separate. No scalar acceptance/suppression
policy; motion and airborne identity remain unknown.

## Validation and decision

Freeze new plan/code/tests/case manifest before the persisted synthetic matrix.
Independently check dual bounds and interval containment via direct perturbed
least-squares refits, including response-only extremal signs and correlated design
perturbations. Sampling is a bug-finding check, not proof; retain the derivation.
Check offset/plane invariance, nuisance rescaling/permutation, small-energy and
near-dependent columns, nonfinite inputs, nonmutation, missing uncertainty, and
observationally identical temporal alternatives. Preserve all case outcomes and
never adjust fixture amplitudes/bounds or omit failures to force positivity.

Run the full unit suite and record inputs/hashes/results in a fresh V44 output.
No real replay follows automatically: first report whether ordinary controls
produce usable intervals and whether adversarial/causal limitations remain. A
successful preflight is not a measured accuracy gain.
