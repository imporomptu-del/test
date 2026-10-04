# V47: conditional guard-calibrated background gain

## Scope and decision

V46 found 239 nuisance-separation failures among 509 eligible real packets,
including 104 rank failures with the background as the only nuisance column.
The user approved testing a justified, uncertainty-bounded brightness model.
Keep production, thresholds, V42–V46 code/results, efficiency work and all
physical labels unchanged. No SSH, RAW16, sealed holdouts, new video decoding,
or journal payload access. This is a separate diagnostic experiment, not a
noise-suppression gate or an accuracy claim.

Past gains cannot bound an arbitrary unseen current gain without a new temporal
assumption. Instead, calibrate a nonnegative current gain using an outer guard
of the current image and a prior-learned background. The guard has no target-core
pixels. This is same-frame causal calibration, **not prior-only calibration**.
Its validity is conditional on background-valid guard pixels and a common
gain/affine brightness model transferring to the core. Neither assumption is
certified by a successful fit or by the reference labels.

## Frozen guard construction and necessary constraints

Use raw native-DN samples from the existing 129×129 coordinate system. Candidate
grid coordinates are 8,16,…,120 on each axis, restricted to Chebyshev distance
40…56 from (64,64). Exclude points at Chebyshev distance <=12 from any supplied
actual prior center. Require finite background/bound and all eight prior image
values at a candidate point. This support is selected without current values.
Retain every horizontal/vertical spacing-8 triplet with three eligible points;
its weights are (1,-2,1). Only the union of retained triplet points is used.
Record candidate, eligible, used and stencil counts/hashes separately.

The central source/fixed-template supports and fixed high-pass dependencies are
inside this annulus's inner edge; arbitrary prior moving-stamp footprints are
also excluded. This is computational isolation, not physical absence of long
PSF tails, untracked objects, or lighting changes. The caller must preserve
inherited saturation/invalid-pixel NaNs. Finite 0/255 values are not reclassified
as saturation in floating-DN synthetic tests. If a used current guard sample is
nonfinite, decline; do not select a favorable current-dependent subset.

For a guard pixel, the declared model is

    y_i = g B_i + p_i beta + errors, g >= 0,
    |response error_i| <= e_yi, |background error_i| <= e_Bi.

Both nominal image-error assumptions remain +/-0.5 DN. For each integer
second-difference stencil, define Y=sum(w*y), C=sum(w*B), E_y=sum(|w|*e_y),
E_B=sum(|w|*e_B). Since the stencil exactly annihilates an affine plane,
every admissible gain obeys

    (C+E_B) g >= Y-E_y,
    (C-E_B) g <= Y+E_y,
    g >= 0.

Intersect these scalar halfspaces using exact rational representations of the
supplied binary floats; handle positive, negative and zero coefficients without
division assumptions. Convert finite endpoints to floats outwards. Empty,
unbounded, unsupported or unrepresentable intersections are unknown. There is
no arbitrary gain cap, regularizer, trimmed residual subset or reduced noise.

This intersection is an **outer relaxation** of joint pixel/affine feasibility,
not a full fit certificate. In addition to between-grid contamination, horizontal
and vertical second differences miss a bilinear xy term. A nonempty interval
does not prove an affine model, guard cleanliness, or transfer to the core.
Explicit counterexamples must remain in the test/report, not be tuned away.

## Bounded-background source evidence

Keep the prior-learned source template m, background B, all fixed-light columns
F, their V45 bounds, common core support and exact affine basis P unchanged.
The background is not silently removed: its coefficient is restricted by the
separately justified guard interval [g_lo,g_hi]. This changes the estimand from
unrestricted-background least squares to bounded-gain partial regression.

For each endpoint g_k, invoke the immutable V45 numerator solver with

    response = y - g_k B,
    response error <= e_y + g_k e_B,
    nuisance = F, source = m, exact affine basis = P.

Use exact-rational centering of the supplied floats, round once, and outwardly
enclose that centering error and the endpoint response-error bound. The existing
V45 numerical-resolution guard and its limitations remain intact; this does
not make the entire SVD/projection pipeline formally IEEE-certified.

For any one admitted realization of y,B,m,F, its partial-regression numerator
is affine in g. The hull of the two endpoint intervals therefore encloses every
interior gain, including correlated errors and a gain chosen adaptively inside
the interval. Both endpoints must be available. Never choose the favorable
endpoint, drop a troublesome fixed alternative or discard inconvenient support.
Keep both endpoint records, zero-containing hulls, and explicit unknowns.
This is conditional feature evidence, not motion or airborne classification.

## Synthetic experiment, declared before persisted scores

Retain all 34 V46 synthetic inputs: six legacy controls and 28 stress cases.
All four old mathematical arms must exactly reproduce their saved V46 results.
Evaluate the new method separately, without ranking or tuning the old arms.

Add exactly 20 deterministic guard-focused cases declared by the new generator
manifest: ordinary gain1; shared gains0.8 and1.2; gain1.2 plus plane; current
source absent; dim and dark source; flat and near-affine guard; identical priors
with independent current gains1 and2; persistent guard light with shared gain;
independently blinking guard light; new guard object; broad PSF; core-only gain
change; guard-only gain change; its byte-identical alternative-explanation twin;
a used-guard NaN; and correlated bounded guard errors. The regional-gain
counterexamples have no current moving source. Current truth is used only to
render/audit the scene and never enters calibration, placement or evidence.

The original 129-pixel inputs, polarity and forecast assumptions stay explicit.
Synthetic image values are floating native DN, not camera quantization or a
calibrated noise model. Geometry, association and photometric model violations
must be identified in truth metadata but must not create an oracle switch in
the algorithm. Identical observations must produce identical output despite
different latent explanations.

Freeze code, tests, plan, ordered cases, all 54 archives, metadata, and in-memory
fingerprints before persisted scoring; check again afterwards. Preserve every
case and all five calculations, including all unavailability reasons. No desired
positive-sign count is a readiness threshold.

## Verification and conditional real shadow

Require full unit regression; independent exact-halfspace/outward-rounding
checks; independent endpoint-hull and simultaneous perturbation checks;
core-blindness/membership tests; exact V46 baseline reproduction; complete
case accounting; and explicit guard-validity/transfer and identity limitations.
An independent audit must pass before new real-packet access. Failure stops
real evaluation rather than weakening the model or overwriting valid reports.

If safe, compare diagnostically on the same frozen 1,211-state V46 real ledger,
509 explicit 0029/0126 packets and 702 unknowns. Reuse the four saved V46 results
exactly; evaluate only the new method. Retain all 355 overlapping references,
original strict assignments/alternatives and the frame216 miss. Do not select a
better alternative or restrict the denominator to the 100 background-only
reference failures; those form a reported subgroup only. Preserve the inherited
current-whole-frame camera-motion caveat. No production filters, new physical
labels, airborne accuracy or false-alarm claims follow from this comparison.

Freeze the exact packet allowlist and code before scores, verify hashes after,
write a completion receipt, audit counts/assignments independently, and report
whether the new model actually helps and where it remains unsupported. If
synthetic safety or implementation checks do not pass, deliver that evidence
and do not access real packets in this experiment.
