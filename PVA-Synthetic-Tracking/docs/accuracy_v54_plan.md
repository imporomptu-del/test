# V54 — frozen background methods, synthetic source-preservation stress test

## Scope

The user approved V53's next bounded test. Keep Median8, Median3 and V53's
Median8-plus-training-median-offset estimator unchanged. Generate paired source-on
and source-off arrays locally. No real JSON measurement analysis, video, camera
cache NPZ, RAW16, sealed holdout, SSH or Jetson access. Old compact receipts/reports
may be hashed for preservation only. No detector threshold, classifier, selector,
gate, history tuning or production change. Efficiency remains paused.

This is a controlled analytic stress test, not a physical sensor model, a detector
run, airborne ground truth, or a real recall/false-positive estimate. Pixel grids,
ideal source shape and perfectly registered histories are deliberately simplified.

## Frozen generated matrix: 420 cases

Use times -8 through0 (eight prior frames and current). Core grid is x,y52..76
step1, y-major then x-major (625 points), center(64,64). Guard grid is x,y8..120
step8, Chebyshev radius40..56 about(64,64), y-major then x-major (144 points).
They are disjoint. Core values never enter guard fitting.

Backgrounds: constant96, or textured
96+.05*dx+.03*dy+12*sin(dx/12)+9*cos(dy/15)+6*sin((dx+dy)/17), dx=x-64,dy=y-64.
Evaluate the same formula separately at core and guard coordinates.

Eight conditions:
- stable: all times unchanged;
- recent_plus8/recent_minus8: last two priors and current shifted +/-8 globally;
- ended_short_plus8: last two priors+8 globally, current unchanged;
- ended_long_plus8: all eight priors+8 globally, current unchanged;
- local_guard_plus16/local_guard_minus16: only current guard samples with
  x>=64,y>=64 receive +/-16 contamination; clean core/background unchanged;
- all_guard_plus8: all current guard samples+8 contamination, core unchanged.

Keep clean current background separate from guard contamination. No case label,
oracle truth, source mask or source parameters reach fitting/prediction.

Thirteen source configurations: absent (amplitude0), and each of six motions
with signed amplitude +4 or -4 DN. The ideal separable compact profile is
max(1-abs(x-cx)/2,0)*max(1-abs(y-cy)/2,0), multiplied by signed amplitude. It is
not a calibrated optical PSF. At current time all present sources center(64,64)
with peak amplitude +/-4. Motions, evaluated at each t in -8..0:
- appearing: no source in any prior; center(64,64) only at current;
- stationary: center(64,64) at all times;
- slow_linear: center(64+.25*t,64);
- linear: center(64+t,64);
- turning: t<=-4 center(64+t+4,60), otherwise center(64,64+t);
- move_stop: center(64+min(t+3,0),64).

Sources are confined to the core by construction and never contaminate the guard.
Stationary means an explicit temporal self-subtraction challenge, not an airborne
label. Turning/stopping are analytic trajectories, not tracking performance tests.

Two noise profiles: uniform[-level,+level], (level0,seed71) or (level.5,seed991).
For each case initialize default_rng(seed), draw guard noise shape(9,144) first,
then core noise shape(9,625). Reuse the same draws across cases of a profile and
across each source-on/off pair; these are paired controls, not independent trials.
Apply background changes, add noise, add source only to core-on, then missingness.
No clipping or quantization. Ordinary strict median over all8 or last3 priors;
never nanmedian. Base factorial:8conditions x13sources x2backgrounds x2noise=416.

Four additional availability cases use stable/textured/appearing/+4/noise0:
guard_current_left_missing (x<64 current guard NaN), guard_current_all_missing,
core_first_prior_center_missing, and core_current_center_missing. Core missing
masks apply identically to on/off arrays. Truth remains known but does not replace
missing observations. These cases are a separate availability stratum, not pooled
into performance claims. Total420cases. No adaptive case expansion after scoring.

## Frozen methods and guard-to-core application

Predict core background independently from core-on and core-off histories using
Median8 and Median3. For V53, call the unchanged frozen guard-only crossfit with
guard Median8/current and retain both left_right and checkerboard splits. Each
fit key names its held-out guard fold (0 or1), trained exclusively on the other
fold. Apply each available offset separately to the core Median8 of each pair.
Keep four offset branches: offset_left_right_fold0/1 and
offset_checkerboard_fold0/1. Never choose or blend a favorable fold/split.

This core extrapolation is a NEW experimental adapter, not a claim that V53's
guard-only predict API accepts core pixels. That API stays unchanged. Fitting
never accepts core observations. Missing/failed guard fits, missing prior values,
and nonfinite addition produce explicit unknown predictions without fallback.
The same guard fits apply to source-on/off; freeze them once. The adapter accepts
only numeric observed histories/current guard and fixed geometry, no truth/labels.

Six methods total: median8,median3 and four offset branches. Freeze predictions
for all cases before any truth-based evaluation. Save enough arrays, masks and
fit certificates for independent reconstruction; no media-generation requirement.

## Evaluation: absolute residual and paired target increment are different

For each method, on/off residual is its observed current core minus its predicted
background. Compare each background prediction to known clean current background
on its own finite prediction/current support. Report availability and completion,
conditional errors and spurious source-off residual magnitudes; unknown is not0.

For present targets use fixed current source template P (unit peak, positive):
project residual onto P using sum(P*residual)/sum(P^2). Require EVERY P>0 point
finite for that amplitude estimate; never renormalize a partly missing source.
Report signed raw on-amplitude, off-amplitude, paired difference on-minus-off,
and polarity-normalized ratios divided by4. Paired difference isolates the
incremental retained source under this controlled pair; raw amplitude additionally
contains background/noise error. A source can retain its increment while an
incorrect background offset makes its raw residual vanish or reverse sign.
The positive-template projection is DC-sensitive, not local contrast or the
actual detector: sum(P)=4 and sum(P^2)=2.25, so a uniform background error b
shifts its amplitude by -16*b/9. A projected sign reversal is not proof of a
detector miss or even a center-pixel sign reversal. Identical spatially constant
offsets cancel algebraically in the source-on/off difference, so available V53
branches should share Median8's paired retention on matched template support.

Also project the oracle residual (core_on-clean_background) for context. Projection
of this oracle uses its own finite current-on/truth footprint, independently of
method prediction availability, and records oracle_available separately; it never
fills a missing method result or missing current observation. Projection
of noiseless known source has signed amplitude+/-4. Measure source amplitude error,
attenuation and sign category: polarity-normalized amplitude <-1e-10 is reversed,
within +/-1e-10 is numerically zero, otherwise same sign. This is a floating-point
reporting tolerance, NOT a detection threshold. Absence has no target retention
ratio/sign category: use null, never a perfect score. No-source core residual is
an artifact diagnostic, not a false detection/false-positive rate.

For each corrected branch retain Median8/Median3 controls on the exact same
finite scored support (intersection across on/off predicted/observed arrays for
that branch and both controls); record indices. Also retain each method's own
available metrics. Projection remains all-or-unknown on the complete template
support. Do not let changing coverage make a correction appear safer.

Aggregate factorial and availability strata separately, retaining every condition,
source motion/sign, background, noise and branch. Include distributions/min/max
and sign/unknown counts, not only pooled means. Do not combine four correlated
offset branches as independent repetitions or select a favorable method after
scoring. Explain noiseless representative failures alongside complete tables.

## Freeze and audit

Hard-pin unchanged V53 model SHA256
aa490ecf8305dd0c5f3facff83a5fa8fef43c67460eb55572f516b298a079d04
and its tests SHA256
d5c53052fca8f47bf1f380582702e62217a353587b906fe759d564b460e70f05.
Freeze this plan, new benchmark/adapter/runner/auditor and four test files plus
those two inherited files (11 files),420specifications,constants before actualrun.
Fresh output child, exclusive writes; save input/prediction hashes and chronology;
recheck all bindings. No tuning or source changes after actual results.

Generated unit tests cover source confinement/motions/pairing, exact counts,
missingness, core exclusion from fits, unchanged V53 behavior, separate folds,
held-out/core-response noninterference, projection/matching denominators,
zero-amplitude and sign conventions, source-history absorption and arithmetic.
Independent auditor imports no producer benchmark/adapter/runner/model. It
reconstructs generated arrays, fits/certificates, predictions, masks, metrics,
aggregate groups, fingerprints and bindings. It may reuse pinned independent V53
audit helpers only if their dependencies are explicitly added to the source freeze
before actual execution; prefer a standalone small implementation for this test.
Run full unit regression and independently review report arithmetic and caveats.

## Decision boundary

Report what survives, what self-subtracts and what is obscured by background error.
No synthetic result alone approves source vetoes, production deployment, real
airborne accuracy or false-alarm claims. Any later real-data source-preservation
check must retain original assignments, seven old positives/five later losses,
and reviewed negative-region workload, with holdouts still sealed.
