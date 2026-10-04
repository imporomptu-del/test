# V51 — observed persistence, return and unavoidable forecast ambiguity

## Scope and decision

The user approved the next accuracy experiment after V50. Implement and test a
prior-only fast/slow history diagnostic before connecting it to detection.
Keep all production code, V50 results, source evidence, reference assignments,
detector thresholds and presentation videos unchanged. Efficiency stays paused.
No new video, NPZ cache, RAW16, sealed holdout, journal payload or Jetson access.

This version is a **descriptor and ambiguity benchmark**, not an automatic
fast/slow selector, a new forecast, a step/flash classifier, or an uncertainty
calibration. Eight observed priors cannot determine whether an ongoing departure
will persist in the next frame. Identical-prefix paired cases must expose this
limitation rather than be assigned fabricated correct regime labels.

## Frozen diagnostic

Input is only an oldest-to-newest real array H with shape (8,N), representing
fixed sampled background locations. No current response, scenario name, signal
truth, source core, reference or future event metadata is accepted.

For every point with finite history and finite arithmetic:

- S = median(all eight priors); F = median(latest three); B = median(first five).
- s = max(1 DN, median(abs(H-S))), matching V50's normalization convention.
- Save S, F, B, s, fast/slow displacement F-S and early/recent displacement F-B.
- Save all three signed recent departures d_i = h_i-B, i=6,7,8, and their
  envelope A = max(abs(d_i)). Save raw DN and scale-normalized descriptors.
- Observed return fraction R = 1-abs(h_8-B)/A, defined only if A>0. R=1 means the
  latest prior is back at the estimated early center; it is not a probability or
  prediction of the current response. Preserve signs to expose reversals.
- When F!=B, save recent-center margins m_i = abs(h_i-B)-abs(h_i-F), and the
  length 0..3 of the latest consecutive suffix with m_i>0. A tie stops a suffix.
  When F=B, margins are mathematically zero but the suffix is undefined, not
  evidence of stability.

No amplitude threshold is fitted. Tiny positive margins must remain visible;
counts alone do not establish a meaningful change. The early center can itself
be contaminated and an event older than the eight-frame memory may disappear
from these descriptors. No descriptor certifies guard purity or transfers to
the source core.

All input point indices remain. Nonfinite history/arithmetic makes a point
unavailable, never a zero residual or a negative source. Outputs are independent,
read-only arrays with explicit availability masks and a content fingerprint.

## Frozen generated benchmark

Use 64 frames at 144 fixed ring locations: grid step8 over x,y=8..120 with
Chebyshev radius40..56 about (64,64). The base is
96 + 0.05*(x-64) + 0.03*(y-64). These are analytic sampled signals, not rendered
camera footage, registered images, a detector benchmark or a physical noise model.
Responses are frames8..63; every predictor receives exactly the previous eight.

Freeze 98 cases before the scored run: two stable noise profiles, plus the
following twelve families crossed with signed amplitudes -32,-8,8,32 and the
same two noise profiles. Profiles are deterministic zero noise (seed71) and
uniform[-1,1] noise (seed991). Reuse the identical noise array for each profile
across all families/amplitudes: this deliberately supports matched comparisons,
not independent statistical trials. No quantization or clipping.

1. Sustained step at frame20.
2. Ramp starting at20, reaching amplitude after16 frame intervals, then plateau.
3–7. Pulses starting20 with durations1,2,4,8,12.
8. Step on the fixed quadrant x>=64,y>=64.
9. Duration2 pulse on that same quadrant.
10. Moving stripe during20..43, center x=8+((t-20)%15)*8, width abs(x-center)<=8.
11. Spatial shift during20..43: amplitude times
    sin((x-64+shift)/8)-sin((x-64)/8), with repeating shifts0,1,-1,2,-2,1,0,-1.
12. Step with missing values at frames18..20, point indices0..7.

Event phases are scheduled metadata, never inferred physical labels: baseline
before20; onset at20; event while scheduled active; post_event afterward. Stable
cases remain baseline. Zero ramp/shift values within an active schedule remain
in their scheduled phase. Signal truth and phase metadata never reach the
diagnostic or the fixed V50 forecast comparators.

Bind 40 identical-prefix pairs: sustained step versus each pulse duration at
response20+duration, crossed with all eight amplitude/noise combinations. Require
identical histories, descriptors and forecasts despite differing currents. If
two possible responses differ by D, any single forecast must incur at least
D/2 error in one world; an interval covering both needs width at least D. This
is a generated identifiability check, not a calibrated physical error bound.

## Execution and reporting

Freeze plan, source/test hashes, case specifications, event conventions and twin
bindings in a fresh V51 output child. Preserve failures rather than overwriting.
Generate deterministic inputs and save every diagnostic/forecast before scoring
any current response. The only point forecasts are the unchanged S and F
comparators from V50; no diagnostic-driven model selection is introduced.

Persist the diagnostic arrays and their fingerprints. Freeze a manifest of all
input and forecast file hashes. During scoring reload frozen forecasts, verify
hashes, and retain all 98 cases, 5,488 response windows and 790,272 point-response
opportunities, including missing values. Summarize by case, family, scheduled
phase and noise level, not just a favorable pooled mean. Report conditional
point MAE, whole-window availability, maxima, recent-center support/margin and
return descriptors; none is object accuracy or a predicted regime label.

Independently reconstruct scalar history statistics and errors, verify all
identical-prefix pairs, and run generated edge/leakage tests plus full regression.
No real-data threshold selection, conditional calibration or detector promotion
is authorized by a good synthetic result.

## Separate retrospective real residual structure

Read only a literal allowlist of the existing V50 compact artifacts, bound to
receipt SHA9213ed07fa7e8efd81f7c0032290dbe2f8c37bd06f6376b86d0b53a1af7dc2b1.
Do not follow receipt maps to images, NPZs, media or other results. Preserve all
1,211 states, both calibration/evaluation partitions, embargo metadata and all
unknowns. These clips and later frames are already exposed development data.

For every available frozen arm/packet, report signed median residual, coherence
abs(sum(r))/sum(abs(r)), sign fractions, and the fraction of absolute residual
mass carried by the largest ceil(10% of used points). All-zero mass produces an
undefined fraction, not zero concentration. Use fixed quadrants at x64/y64 and
fixed nine-response-frame bins. At the frame level, report equal-frame mean
packet MAE and the largest ceil(10% of archived frames)' share, while any missing
packet makes its full archived-frame metric unavailable. Preserve no-archive
frames and all-unknown/embargo bins. Never call these current-response diagnostic
statistics available prior predictors or source/object decisions.

The result should tell us which information is present in the observed history,
which errors are concentrated in space/time, and which future outcomes remain
indistinguishable. Any adaptive predictor or uncertainty model comes in a
separately frozen follow-up, not an outcome-driven change within this run.
