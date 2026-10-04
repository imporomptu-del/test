# V43: supported numerical fits and foreground-independent fixed components

2026-09-25. Follow-up to V42, authorized by the user's requests to continue.
Offline accuracy experiment only. No production, GPU/PVA/speed, RAW16, remote
execution, holdout access, new labels, or media directory discovery.

## Locked scope and factorial comparison

Keep the exact V42 1,698 state keys, 675 strict states, four explicit local AVIs,
1,367 grid states/381 strict states/108 windows, and all 356 overlapping reference
samples with all 368 measured alternatives. Original assignments and misses remain.
Retain V42's forecast and native geometry unchanged; current detections cannot
place templates. Preserve all 1,093 geometry-unavailable records.

Freeze source/journal/reference/code/test hashes before native extraction. Retrieve
only the same 630 requested source frames, sequentially grab through each declared
EOF, and retain at most nine native grayscale frames. Cache every geometry-eligible
current129/history8x129/centers/offset input (605 expected), not only a few favorable
states. Check geometry against V42 and its seven existing input archives. Hash the
complete cache before calculating any new arm score. Then run four fixed arms:

1. `baseline`: unchanged V42; require reproduction of all saved outcomes.
2. `stable_only`: V42 components plus supported-fit sensitivity checks.
3. `protected_only`: dependency-protected fixed components, original V42 fitting.
4. `combined`: both changes.

No best-arm or best-ID selection. Compare availability, support hashes and warning
status. Scores from different supports are not a like-for-like MSE improvement.
The two V42 failures (0029 frame298 bright:980; 0126 frame140 bright:1001) and the
0029 frame346/347 bright:2641 counterexamples are predeclared *exposed diagnostics*,
not independent validation. Saving all eligible inputs permits their full audit.

## Fixed component protection

Keep the causal background and moving-template calculations unchanged. For fixed
anchor learning, use only prior frames with an actual same-ID center. A missing
center is not evidence that the source is absent. Exclude the prior foreground
Chebyshev-radius8 footprint. A retained anchor observation/stamp must have its
whole 17x17 highpass stamp and all interpolation dependencies finite and outside
that exclusion. The 9x9 highpass makes the integer-anchor raw dependency footprint
25x25. Apply the original local-max comparison before eligibility filtering so
removing a brighter unsafe neighbor cannot manufacture a new maximum.

Repeat support (three distinct frames, radius2) and capacity (four) stay unchanged.
Record eligible opportunities at every frozen seed, raw versus protected anchors,
and rejected prior evidence. Fewer than three eligible opportunities or removed
unresolved raw alternatives overlapping the core remain explicit ambiguity.
An empty retained dictionary is not certified background absence. No new physical
classification follows from the term "fixed".

## Numerical sensitivity contract

Use a declared **±0.5 DN input perturbation experiment**, not an estimate of actual
camera noise or a calibrated confidence interval. Freeze geometry, masks, anchor
identities, used-stamp membership and support. Median with fixed support preserves
that bound; moving residual and 9x9 highpass are conservatively bounded by ±1 DN.
Positive clipping/Hann weighting are Lipschitz. For a normalized vector v and
componentwise perturbation bound e, r=||v|| and d=||e|| must satisfy r>d; propagate
the conservative componentwise bound e/(r-d)+abs(v)*d/[r*(r-d)]. Propagate the
maximum per-entry stamp bound through median and final normalization, then through
positive bilinear weights. Do not average correlated uncertainty away. Unbounded
normalization or changed usable support is unknown.

These bounds are conditional: they do NOT encompass unknown sensor/compression
noise, registration error, temporal model mismatch, changing peak selections,
changing used-stamp membership or physical classification. Record those limits.

Pre-freeze integration testing already shows an important cost: the clear synthetic
moving point over a fixed edge passes V42 but the new annulus certificate declines
it (conservative bound approximately 20 DN against a 1 DN budget). Preserve this
counterexample and report real abstention coverage. Do not interpret a more cautious
verifier as a better detector or loosen its budget in response to replay outcomes.

Scale fit columns and corresponding design bounds consistently; record energies,
singular values, condition, training-to-test prediction leverage, and perturbation
effects. Robust rank requires the least singular value to exceed the design
perturbation norm. With supplied response/design bounds, compare the conservative
prediction sensitivity to the fixed engineering budget 2*sigma_dn (the difference
of two observations at that supplied bound). Reject unsupported numerical fits as
unknown: no operative coefficients/predictions/score. Nonnegative-last constraints
must account for both free and zero-last branches, including boundary uncertainty.

Apply this to the shared annulus-to-core background fit and both central models
on both checkerboard folds. Annulus prediction uncertainty propagates into the
central residual response bound; no current core intensity can choose numerical
budgets, component templates or regularization. Bounds do not establish motion.

## Verification and reporting

Synthetic tests: tiny nonzero training tails, nearly cancelling columns, scaling
and permutations, constraints/boundaries, missing and saturated support, protected
foreground poisoning, moving-source contamination, fixed flicker away from the
foreground, missing/hovering/crossing histories, and input nonmutation. Keep all
old tests. Independent preflight before native extraction; new code changes after
real outcomes require another frozen version, not an overwritten run.

Audit complete state/reference/hash accounting, reproduce all V42 baseline scores,
inspect both prior numerical failures on saved inputs, and independently check
sensitivity calculations for bounded predeclared cases. Report abstention cost and
new failure cases honestly. Zero production changes is not a measured accuracy gain.
No acceptance/rejection policy, airborne recall, or false-alarm reduction is claimed.
