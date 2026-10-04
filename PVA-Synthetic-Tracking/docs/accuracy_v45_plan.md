# V45: conditional source-presence evidence with unchanged synthetic inputs

## Scope and decision

V44 did not close the ordinary prior-learned source interval. V45 investigates
two mathematical sources of conservatism, not new detector settings. Production,
real videos, journals/caches, Jetson, RAW16, sealed holdouts, and efficiency work
remain out of scope. No algorithm promotion or accuracy claim follows from this
synthetic experiment.

The inputs and controls are exactly V44's 16 oracle-vector cases and six causal
prior-learning probes. Existing V44 files are immutable. No current-response
template learning, recentering, support selection by score, noise reduction,
missing-stamp omission, or outcome-driven case selection is allowed.

## Two independent changes

### Positive-template normalization intervals

Keep the V43 nominal templates, stamp membership, finite support, Hann aperture,
geometry, and prior associations. Aligned sample uncertainty stays +/-0.5 DN;
residual and high-pass stamp uncertainty stays +/-1 DN before the same aperture.
For a nonnegative weighted stamp box L <= v <= U, the exact marginal extrema of
v_i / ||v|| are

- minimum: L_i / sqrt(L_i^2 + sum(j != i, U_j^2));
- maximum: U_i / sqrt(U_i^2 + sum(j != i, L_j^2)).

Require the complete lower-vector norm to exceed the unchanged normalization
acceptance threshold. An uncertain used stamp makes the bound unavailable;
never remove it. Propagate the intervals through the coordinatewise median,
then normalize once again using the same extrema. Dependencies among extrema
are retained conservatively by the surrounding box, not treated as independent
physical possibilities. Supply the older solver API with a symmetric envelope
about the unchanged nominal template. Original support and zero-Hann pixels
must be preserved. Finite-precision checks are not a formal interval-arithmetic
proof.

### Numerator/sign certificate

Project out the exact affine basis, and fit nuisance-only columns. With Q the
affine residual projector, N = QZ, R the nuisance residual projector, and
w = RQm, r = RQy, evaluate s = (Qm)' R (Qy). Its sign determines the source
coefficient sign whenever the residual source is nonzero; do not divide by an
uncertain source energy or claim an amplitude interval.

For source error u, response error e and nuisance-projector change bounded by g,
a conservative numerator error is

    |w|' e + |r|' u + ||u|| ||e||
    + g (||Qm|| + ||u||) (||Qy|| + ||e||).

Nuisance rank must be robust to its admitted design perturbation. For fixed
nominal column scaling, eta < sigma_min(N) supplies a projector-gap bound
g <= eta / sigma_min(N). Empty nuisance gives g = 0. Also evaluate the valid
blockwise projector bound: the two diagonal blocks have norm at most g^2 and
the cross blocks at most g. Use the smaller of these two algebraic bounds,
reporting both; this is not an empirical threshold.

Keep exact-redundancy and near-affine safeguards from V44: only literal zero or
literal +/- supplied affine columns with zero uncertainty can be removed.
An arbitrarily small nonzero nuisance component cannot silently be discarded.
Unknown support, uncertified nuisance rank, or nominal source redundancy remain
unknown. An available zero-containing numerator interval does not certify
perturbed source rank. Positive/negative intervals certify only a conditional
coefficient sign, not movement, object identity, or airborne class. Tiny signs
within an explicit numerical roundoff resolution may remain unresolved; this is
separate from sensor uncertainty, not an arbitrary DN threshold. Report the
analytic interval/error separately from an operation-scale numerical resolution
margin. The public interval expands the analytic interval by that margin, so
sign and interval fields agree. This conservative decline guard is not a formal
IEEE rounding enclosure and can only withhold a sign, never create one.

## Frozen paired experiment

Run all four arms for every causal probe, with no selected winner:

1. V44 amplitude interval + V43 bounds (exact baseline).
2. V45 numerator interval + V43 bounds.
3. V44 amplitude interval + V45 bounds.
4. V45 numerator interval + V45 bounds.

Oracle-vector cases have predeclared supplied template-error arrays, not learned
stamps: their old/new-bound arms are deliberately identical. Report this as
duplicate accounting, not extra independent evidence. Numerator and amplitude
interval widths measure different quantities and must not be directly compared.

Freeze this plan, dependencies, tests, baseline receipt/results, and all 22
synthetic archives before scoring. Verify arrays against the stored V44 inputs
and baseline results against V44. Rehash before and after scoring. Use exclusive
output creation and preserve failures. All arms must retain equal nominal
design/support hashes. Observationally identical moving/fixed-emitter twins must
stay numerically identical. Missing history and physical/fixed-light ambiguity
must not become favorable decisions.

## Verification and reporting

- Unit tests: support/NaN/aperture preservation, normalization acceptance,
  median monotonicity, exact/near overlap, missing errors, rank failures,
  simultaneous errors, affine/scaling invariance, negative signs and unchanged
  inputs. Test adapter injection without mutating frozen module globals.
- Independent deterministic numerical audit: positive-box endpoints (including
  high-precision checks), sampled normalized stamps, numerator perturbations,
  adversarial degeneracies, and invariance. Record assumptions and scope;
  random containment tests are not proof or camera-noise calibration.
- Full unit suite before the frozen run and final verification afterwards.
- Paired accounting: all 16+6 cases, matching baseline, matching nominal designs
  and support, unknown motion/class, unchanged production, receipt integrity.

Success here means narrower valid conditional uncertainty under the same
assumptions, ideally resolving the ordinary learned source without resolving
the absent/off-forecast or ambiguous-light controls as airborne objects. Even
that would only justify a further isolated validation step, not deployment.
