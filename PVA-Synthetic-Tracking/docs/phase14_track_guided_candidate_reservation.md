# Phase 14: bounded track-guided candidate reservation

## Purpose

Phase 13 showed that the faint 1,000-DN target in held-out chunk 0040 already
passes the raw-SNR and local-CFAR thresholds, but normally ranks below the four
peaks allowed from its spatial quota cell. Globally raising the cell quota from
4 to 8 confirms that target, but also drives 47 of 52 windows into the global
256-candidate cap and increases unmatched confirmed/coasted tracks from 542 to
869.

This phase tests a narrower intervention: retain the default quota of four and
allow a strictly bounded number of otherwise quota-suppressed peaks near causal
predictions from tracks that already exist.

## Causal data flow

For synthetic window `t`, the Kalman manager computes read-only predictions
from state last updated at window `t-1` or earlier. Asking for these predictions
does not advance a track timestamp, covariance, age, miss count, or lifecycle.
Hints are unavailable across coordinate-segment changes, non-increasing
timestamps, or timestamp gaps above the configured tracking limit.

The candidate extractor still performs the normal sequence first:

1. raw-SNR and local-CFAR thresholding;
2. support and validity gating;
3. deterministic local-maximum selection;
4. balanced pre-NMS retention;
5. joint position/velocity NMS and the default per-cell output quota.

Reservation then considers only peaks rejected at step 5 because their cell is
full. It cannot lower a threshold, synthesize a new peak, bypass validity, or
rescue a joint-NMS duplicate. A peak must fall within both configured position
and velocity radii of an uncovered prediction. Only zero-miss tentative tracks
observed continuously for at least one full integration-window length are
eligible. The mechanism is specifically a confirmation-rescue path at the
first non-overlapping evidence boundary, so new one-off tracks and already
confirmed/coasted clutter cannot consume its capacity. Eligible tracks are
ordered by fewest misses, highest observation fraction, most associated
updates, age, and finally track ID. Each track proposes its highest-ranked
gated suppressed peak, at most one output is needed to cover a prediction, and
the total reservation count is capped per window.

If the global output is already full, reservations replace the lowest-ranked
baseline outputs rather than increasing the total. Output remains at or below
`max_candidates_per_window`.

## Configuration and auditability

Reservation is disabled by default. These three values must be either all zero
or all positive, and spatial quotas must be enabled:

- `track_guided_reservation_position_radius_px`
- `track_guided_reservation_velocity_radius_px_s`
- `max_track_guided_reservations_per_window`

The motion CLI exposes paired experiment overrides:

```text
--track-reservation-position-radius-px
--track-reservation-velocity-radius-px-s
--max-track-reservations-per-window
```

Reports record the effective values, number of causal hints, baseline-covered
hints, available and examined quota-suppressed peaks, retained reservations,
guiding track IDs, and any baseline candidates displaced at the global cap.
Reserved candidates identify their guiding track in `selection` metadata.

## Safety boundary

This mechanism cannot solve cold start: the target must first appear in the
ordinary candidate output and create a track, then survive one integration
window of continuous ordinary observations. It can only prevent that mature
tentative track from starving at the next independent-confirmation boundary.
Persistent tentative clutter tracks receive the same causal treatment, so a
small hard reservation cap is essential; reservation is not a production default,
until held-out replay measures target recall, candidate burden, unmatched
tracks, cap pressure, and runtime.

## Verification

Deterministic tests cover configuration validity, read-only prediction,
segment/gap fail-closed behavior, quota-suppressed recovery, joint output
bounding, baseline displacement, and disabled-path equivalence. The complete
local suite passes 141 tests with 11 environment-dependent skips; the Jetson
suite passes 142 tests, including its CUDA-specific case.

The hardware replay uses the already-selected held-out chunk 0040 and the same
synthetic injection, 7 x 7 velocity grid, and CFAR-8 operating point as Phase
13. Results are recorded below after the isolated Jetson run.

## Chunk 0040 hardware results

The first attempted reservation replay accidentally omitted the explicit 7 x 7
velocity override and inherited the configuration's single zero-velocity trial.
It is retained for provenance but rejected before comparison.

Two valid intermediate policies were also rejected:

- Global-score cap 16 consumed 792 reservations, hit its reservation cap in
  49/52 CFAR-8 windows, did not add a target match, and raised unmatched
  confirmed/coasted tracks from 542 to 590.
- Tentative-only cap 16 produced the desired fourth target confirmation, but
  consumed 779 reservations and raised unmatched confirmed/coasted tracks to
  750. Letting immature tentative clutter use the feedback path was too broad.

The final predeclared experiment requires a zero-miss tentative track with a
continuous observation on every age window for at least the four-frame
integration length. Position/velocity gates are 12 px and 5 px/s, matching the
tracker's hard association limits, and the reservation cap is 8. The baseline
and experiment both have 63/63 accepted transforms, 58 detection-ready frames,
52 synthetic windows, and 49 velocity trials.

| CFAR-8 metric | Quota-4 baseline | Continuous-tentative reservation |
|:---|---:|---:|
| Target opportunity matches | 139/196 | 140/196 |
| 1,000-DN opportunity matches | 3/49 | 4/49 |
| Confirmed injected target IDs | 3/4 | 4/4 |
| Unmatched candidate burden | 8,260 | 8,327 |
| Unmatched confirmed/coasted tracks | 542 | 589 |
| Frozen-policy unmatched, latest observation | 17 | 17 |
| Windows at global 256-candidate cap | 0/52 | 0/52 |

The one added weak-target match is candidate 69 in frames 10--13. Its selection
metadata records `reservation_track_id: 165`; the same track is then confirmed
with two independent evidence hits. This is a causal recovery, not a post-hoc
truth match.

Across the CFAR-8 evaluation, 10,826 active hints are exposed, but only 506 meet
the continuous-tentative eligibility rule. The extractor retains 68
reservations across 33 windows, with a median of 1 and maximum of 7; the
configured cap of 8 is never reached. No baseline candidate is displaced at
the global cap.

Primary candidate extraction median time rises from 360.57 ms to 376.33 ms
(+4.4%), while end-to-end time changes from 486.24 s to 486.56 s (+0.06%) in
these single runs. These are characterization values, not a latency guarantee.

The operating point remains disabled by default. It achieves the target-side
goal with far less burden than quota 8 or the earlier reservation variants,
but a +47 unmatched-track increase still requires transfer validation. The
frozen motion/persistence diagnostic does not qualify the newly confirmed
1,000-DN track, so that diagnostic must not be promoted to a live rejection
gate on this evidence.

## Frozen transfer validation: chunk 0060

Chunk 0060 was selected from the previously declared fallback order after only
its motion eligibility had been inspected. The chunk 0040 operating point was
then frozen unchanged: CFAR 8, 7 x 7 velocity grid, 12 px and 5 px/s gates,
continuous zero-miss tentative eligibility, and reservation cap 8. A paired
no-reservation replay uses the same input, timestamps, injection, frame count,
and velocity override.

Both runs have 63/63 accepted transforms, no rejected transforms, 56
detection-ready frames, 47 synthetic windows, and 49 velocity trials.

| Chunk 0060 CFAR-8 metric | Quota-4 baseline | Frozen reservation |
|:---|---:|---:|
| Target opportunity matches | 131/176 | 131/176 |
| 1,000-DN opportunity recall | 100.0% | 100.0% |
| 2,000-DN opportunity recall | 15.9% | 15.9% |
| 4,000-DN opportunity recall | 81.8% | 81.8% |
| 8,000-DN opportunity recall | 100.0% | 100.0% |
| Confirmed injected target IDs | 4/4 | 4/4 |
| Unmatched candidate burden | 7,196 | 7,247 |
| Unmatched confirmed/coasted tracks | 452 | 489 |
| Frozen-policy unmatched, latest observation | 29 | 29 |
| Windows at global 256-candidate cap | 0/47 | 0/47 |

None of the 51 retained reservations matches injected truth. They occur across
27 windows, with a maximum of 7 and no reservation-cap hit. Target recall and
confirmation are exactly unchanged, while unmatched candidate burden rises by
51 and unmatched confirmed/coasted tracks rise by 37. Median candidate
extraction time rises from 367.24 ms to 378.54 ms (+3.1%); single-run
end-to-end time is 1.1% lower and is treated as runtime noise rather than a
speedup.

Across chunks 0040 and 0060, reservation changes target matches from 270/372 to
271/372 and confirmed injected targets from 7/8 to 8/8. It also changes
unmatched candidate burden from 15,456 to 15,574 (+118, 0.8%) and unmatched
confirmed/coasted tracks from 994 to 1,078 (+84, 8.5%). The diagnostic
motion/persistence policy leaves the combined latest unmatched count unchanged
at 46, but it also rejects the recovered chunk 0040 weak target.

The transfer verdict is therefore conditional rather than positive: the
mechanism solves a demonstrated quota-starvation case, but it provides no
target benefit and adds track burden on a scene where ordinary quota selection
is sufficient. It remains an opt-in diagnostic. Promotion requires an
admission discriminator that preserves the recovered weak target while
rejecting mature tentative clutter, followed by another frozen transfer test.

## Evidence files

- `results/tiny_target/phase14/chunk_0040_64frames_cfar8_continuous_tentative_reservation12_5_cap8_vgrid7x7_jetson.json`
- `results/tiny_target/phase14/chunk_0040_64frames_cfar8_continuous_tentative_reservation12_5_cap8_vgrid7x7_accuracy.json`
- `results/tiny_target/phase14/chunk_0040_64frames_cfar8_continuous_tentative_reservation12_5_cap8_vgrid7x7_motion_policy.json`
- `results/tiny_target/phase14/chunk_0040_64frames_cfar8_tentative_reservation12_5_cap16_vgrid7x7_jetson.json`
- `results/tiny_target/phase14/chunk_0040_64frames_cfar8_reservation12_5_cap16_vgrid7x7_jetson.json`
- `results/tiny_target/phase14/chunk_0060_64frames_cfar8_continuous_tentative_reservation12_5_cap8_vgrid7x7_jetson.json`
- `results/tiny_target/phase14/chunk_0060_64frames_cfar8_continuous_tentative_reservation12_5_cap8_vgrid7x7_accuracy.json`
- `results/tiny_target/phase14/chunk_0060_64frames_cfar8_continuous_tentative_reservation12_5_cap8_vgrid7x7_motion_policy.json`
- `results/tiny_target/phase14/chunk_0060_64frames_cfar8_quota4_baseline_vgrid7x7_jetson.json`
- `results/tiny_target/phase14/chunk_0060_64frames_cfar8_quota4_baseline_vgrid7x7_accuracy.json`
- `results/tiny_target/phase14/chunk_0060_64frames_cfar8_quota4_baseline_vgrid7x7_motion_policy.json`
