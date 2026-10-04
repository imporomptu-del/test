# V42: causal background plus localized-source verifier experiment

2026-09-25. User authorized implementing the direction from V41. This is an
offline, opt-in evidence stage. It must not remove production detections, change
associations/background learning, or claim airborne accuracy without class truth.
No efficiency/GPU/PVA work, RAW16, sealed holdouts, remote execution or media
directory discovery. Use only local development AVIs 0029/0126/0055/0082.

## Fixed evaluation scope and prior exposure

Retain all 1,367 actual measured states / 381 strict states in the 108 frozen V40
windows. Add the union of every original gated actual-measurement alternative
in the historical dense285, pilot28, anchors24, compact8 and V40grid11 panels.
Deduplicate repeated state keys (clip, frame, segment, ID), not reference samples.
Keep each panel's denominator, unknowns, original assignments and original misses.
No source label, class or matching tolerance is revised. These exposed development
references are not independent encounters or held-out generalization evidence.

This experiment needs eight preceding native source frames and wider context
than V40 retained. Freeze exact target keys, their requested history frames, the
four original source hashes, journal/reference hashes, code/tests, constants and
versions before decoding/scoring. Decode each explicit AVI sequentially from zero
through its declared EOF; retrieve only the union of requested nine-frame spans.
Retain no more than nine full grayscale frames at a time. No scene/absence labels
are inferred from the extra inference context. This is an explicitly wider source
context than V40/V41, not a like-for-like availability comparison.

## Causal forecast and geometry

For each current actual state, use only its exact same segment/ID's **actual**
measurements in the previous eight rows. Require at least five. Transform them
to the saved reference coordinates; fit one quadratic position model and predict
the current timestamp. Map that forecast into current-camera coordinates using
the saved current global transform. Never use the current detection position,
filtered state, velocity, truth coordinates or future frames to locate the model.
Current global geometry was itself estimated from the current image; that
exposure is disclosed and is not a source of independently calibrated uncertainty.

Require contiguous frames and exact nominal 100ms timestamp spacing, with no
reset/segment crossing. Actual acquisition cadence is not independently verified.
At the round-half-up forecast center, sample current and eight previous images
on a 129x129 current-camera coordinate grid. Prior frames use inverse(Hprior)*
Hcurrent, exact float64 bilinear sampling, NaN for out-of-frame pixels and any
positive-weight 0/255 source corner. Do not translate the whole background with
the object. Preserve subpixel forecast offset and prior actual point locations
on the aligned grid. Insufficient history/support is unknown, not rejection.

## Separate image components

1. **Protected causal background:** median of the eight aligned prior frames,
   masking Chebyshev radius8 around that frame's actual source point. Require
   at least three unmasked finite observations per pixel; never fill missing
   background under a slow/stopped source. No current image enters this model.
2. **Localized moving template:** at least three prior 17x17 residual stamps,
   each background-subtracted at its own actual prior position, polarity-normalized
   and positive-aperture normalized. Use a fixed Hann aperture and median across
   available stamps. Require at least64 finite positive-aperture samples/stamp
   and at least three observations per combined-template pixel. No current image
   or reference coordinate may learn the template. Its placement is the forecast.
3. **Independent fixed-light alternatives:** detect prior-only local-contrast
   maxima (9x9 full-finite box residual, 3x3 max, >=1DN, central41x41 search).
   Group within Chebyshev2px across at least three distinct prior frames, rank
   deterministically by repeat support/strength/position, and retain at most four
   empirical 17x17 components. Independently varying amplitudes can be signed
   relative to the baseline. More anchors than capacity means ambiguous, not a
   confident reduction to four physical objects. Persistent moving/stopped sources
   can overlap this dictionary; it is a competing explanation, not nuisance truth.

Fit current photometric gain>=0 plus a plane only in the 65x65 background annulus
(Chebyshev16..30), outside the central source area, with >=64 finite samples and
rank checks. Hold that background prediction fixed for the central25x25 test.
The current image may fit these outside-source photometric terms, but cannot move
the forecast or learn the source/fixed-light templates.

Compare the fixed-component residual model with the same model augmented by the
localized forecast template (nonnegative moving amplitude). Checkerboard train/
test swap on identical finite support; >=64 total/32 per fold. Report both held-
out MSEs, gains, complexities and moving-template energy outside the fixed span.
Near-collinearity (relative energy<=1e-8) is explicitly ambiguous. The augmented
model is also explicitly ambiguous if a persistent anchor lies within Chebyshev
2px of the forecast: empirical template shape differences do not disambiguate
physically overlapping explanations. This flag does not change continuous scores.
The augmented
model has an extra parameter; improvement is not a target probability or an
automatic acceptance/rejection criterion. Spatially correlated checkerboards and
exposed tracker histories are not independent statistical validation.

## Safety and tests

Test current-detection poisoning and future exclusion, camera motion versus object
motion, quadratic turns, coast/missing history, resets, origins, native borders
and saturation. Test moving points over stationary edges, independently blinking
fixed lights, overlap/collinearity, slow/hovering background holes, intermittent
visibility, broad deformations, noise, input nonmutation, and prior-only component
learning. Do not force synthetic ambiguities into a binary physical class.

Freeze after synthetic tests and a preflight review, before real outcomes. Any
post-score algorithm change requires a new version/run; do not silently tune this
one. Audit full state/reference accounting and independently check selected native
inputs and numerical fits. Retain a predeclared bounded audit-input subset: first
two state keys per clip, first two gated keys per reference panel, plus every
selected state at the already-known 0029 frame346/347 counterexamples. Selection
precedes new scores and is a diagnostic subset, not independent validation.

## Deliverable and decision boundary

Report geometry/core availability, explicit ambiguity, continuous improvement and
prediction error (current actual position used only after inference for diagnostic
comparison). Keep all reference alternatives and separately report old candidate/
measured/strict coverage. Missing candidates remain misses even though there is
no state to verify; unavailable verification does not erase those denominators.

This experiment emits evidence only: zero production additions/removals by design.
Do not present unchanged output as a measured accuracy gain or an optimized
false-alarm/recall trade-off. A later suppression policy would require its own
freeze, all panels, reviewed source/class evidence and closed-loop validation if
upstream feedback changes. The immediate objective is to test whether separating
background/localized source resolves V41's demonstrated failure without forcing
flicker/neighbor ambiguity into a false physical-motion decision.
