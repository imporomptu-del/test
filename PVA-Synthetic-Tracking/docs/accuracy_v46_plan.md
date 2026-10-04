# V46: geometry/shape stress, then a bounded 8-bit cache-only shadow comparison

## Scope

Keep V42–V45 code, configurations, input-error assumptions and saved results
unchanged. No production detector/tracker changes, threshold search, output
suppression, new physical labels, efficiency work, Jetson access, RAW16 or sealed
holdout access. The user approved challenging imperfect tracking and fixed-light
confusers, followed by an isolated 8-bit comparison. Perform the synthetic
preflight first; proceed to real cached patches only after its independently
audited diagnostic-readiness checks pass.

## Synthetic experiment, declared before scores

Repeat all six frozen V45 causal probes under all four immutable arms, requiring
exact saved-result reproduction. These legacy controls include an explicitly
supplied forecast, not a newly validated forecasting algorithm.

Add exactly 28 deterministic prior-image stress cases under the same four arms.
The generator manifest declares their order, formulas and roles before scoring.
There are no oracle source-template columns and no signs selected as targets.
The new cohort uses true historical world positions (32+4*i,64), i=0..7, at times
-8..-1. Unless a case explicitly changes it, the independent current source is
at (64,64), Gaussian sigma1 and peak30 DN on background
50+20*(world_x>=64)+8*sin(world_y/12).

The test-only forecast is fixed linear least squares over the last four
available supplied historical measurements, using their original timestamps
and extrapolating to time0. It is not the production or cached V42 quadratic
forecaster. It tests the localized evidence under honest past-derived placement,
not forecasting superiority. The crop center is floor(forecast+.5), and all
rendered world pixels and supplied prior positions are transformed into this
crop. The adapter gets only current/history patches, reported local historical
positions, fractional forecast offset and polarity. Current truth is used only
to render/evaluate the scene; it cannot recenter the crop, alter the forecast,
select a current template, choose support or select another track identity.

Predeclared groups:

- Ordinary prior-derived placement and current source absent.
- All-prior measurement bias of .5 or2 pixels; fixed-pattern jitter at .5 or2
  pixels; independent current departures of .5,2 or6 pixels; combined errors.
- Current width1.5; rotated elliptical shape(2,.75); gradual width change;
  current brightness7.5; temporal brightness ramp; intermittent prior source.
- Three/four missing historical measurements; two incorrect last measurements;
  reversed reported positions while image chronology remains unchanged.
- Persistent/blinking fixed emitters at or near the forecast, with the moving
  source absent at the current time; moving source plus fixed emitters.
- Identical pixel/measurement observations interpreted as one moving source or
  sequentially switched fixed emitters. Their numerical outputs must match.

The generator keeps true source positions, visibility, fixed-emitter presence,
reported measurements and physical interpretation separate. Cases retaining
measurements while the rendered source is invisible explicitly test incorrect
associations; they are not asserted to be correct tracker observations.

The original +/-0.5 DN aligned-image uncertainty and +/-1 DN residual/high-pass
stamp assumptions remain unchanged. Geometry, trajectory and shape changes are
model-mismatch stress **outside** that fixed-geometry uncertainty guarantee.
Native-DN floating synthetic images are not a claim of sensor or AVI quantization
simulation. No score is a probability or airborne-class decision.

Freeze code, tests, plan, baseline receipt, manifest, all34 input archives and
in-memory value/geometry fingerprints before scoring. Retain every case and
every arm, including unavailable cases. Check identical prior design/support
for declared current-only variants and exact equality for observational twins.
Report results by group without selecting an arm or adjusting settings afterwards.

## Diagnostic readiness (not an accuracy threshold)

Require full regression preflight and an independent audit establishing:

1. Exact V45 baseline reproduction and unchanged nominal/error contracts.
2. Correct prior-only synthetic forecast/crop construction, no current-truth
   leakage, finite/valid outputs and no unresolved mathematical implementation bug.
3. Complete case/input accounting and unchanged fixed-light ambiguity; physical
   motion/class remain unknown and inconclusive evidence never becomes negative.
4. Model-mismatch cases and limits are explicitly identified rather than claimed
   covered by the +/-0.5 DN uncertainty contract.

Do not invent a required fraction of stress positives after seeing results.
Stationary lights may legitimately produce positive localized coefficients.
That is not automatically a numerical failure; labeling them airborne would be.
If these safety/provenance checks pass, a diagnostic-only cached comparison is
informative even when stress sensitivity is limited. Persist a separate hashed
readiness record binding the stress completion and independent audit before any
real-packet payload access. Failed readiness stops real evaluation in this turn.

## Bounded real comparison, conditional on readiness

Use only the existing V43 stability_01 8-bit-derived cache entries for previously
reviewed clips0029 and0126: exactly1,211 saved measured states, including509
eligible129x129 packets and702 geometry-unknown states. No new AVI decoding,
journal payload access or remote work is needed. Hash only explicitly selected
packet paths plus allowlisted compact provenance/source files; never recursively
follow the V43 completion receipt into videos, journals or other clips.

Pinned compact provenance:

- V43 completion: b3ae0fa68296f12e2493bc87419da861e06f5c45f7d4349f1e8396bdabad5042
- V43 cache manifest: 619997859738ab81eecc1aa4031292a39e83dacc2cb2eb066ac89cbf3f0e9f80
- V42 reference inventory: 444df960282569ef4c60a80a15bc6a50756fe0eadba421969061f6fb5aab86df

Preserve the exact two-clip state ledger, original production qualification,
all355 overlapping reference samples, their original assignments and every
alternative identity. Keep the original0126 frame216 missing assignment visible.
References describe visible image features, not proven airborne objects; there
is no authoritative negative-exposure denominator. These are already exposed
development clips, not held-out generalization evidence.

Run all four frozen mathematical arms per eligible packet. Geometry-unknown
records remain in the denominator with no fabricated interval. No current
detection/reference position or label is passed into the adapter. Cached V42
placement used previous actual same-ID measurements, but its current whole-frame
global-motion transform is not independent of the current image. Preserve that
distinction and inherited provenance; do not claim to reverify journals here.

The inherited V45 adapter has synthetic-origin metadata. A new wrapper may
correct only these origin/provenance declarations on a copied result, recording
the exact overrides and immutable mathematical source. All numerical outputs,
design/support hashes and ambiguity must remain unchanged. The real comparison
must not claim its inputs are synthetic, nor claim an old V45 real baseline
exists. This is the first real-packet comparison of these V45 mathematical arms.

Freeze selection, packet hashes and code before any scores; use a fresh output
directory, retain all unknowns, check input hashes after scoring and write a
completion receipt. Report numerical availability and sign separately from
original assignments/qualification. Positive feature evidence is not detector
recall, false-alert reduction or a new airborne label. No cleaned video or
production promotion is implied. Persist the results and a concise explanation
of which limitation should be addressed next.
