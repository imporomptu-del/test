# Phase 10: Kalman Tracking and Temporal Confirmation

## Purpose

Phase 9 emits independent candidate records for each synthetic-integration
window. Phase 10 connects compatible candidates through time and requires
genuinely new frame evidence before calling a track confirmed.

The state is `[x, y, vx, vy]` in stabilized reference pixels and pixels per
second. The constant-velocity transition matrix uses the actual difference
between candidate-window reference timestamps. A white-acceleration process
model grows position/velocity covariance according to that elapsed time.

## Measurement and association

A candidate measures all four state components. Position and velocity
measurement sigmas are explicit configuration values and produce a fixed
diagonal covariance. Each configuration also declares whether those values are
provisional, derived from synthetic characterization, or from empirical camera
calibration. The checked camera configuration is still `provisional`; reports
warn accordingly.

An association must pass all three gates:

1. configured Euclidean position residual;
2. configured Euclidean velocity residual;
3. squared Mahalanobis distance using predicted plus measurement covariance.

Passing pairs are assigned greedily in deterministic order by Mahalanobis
distance, track ID, then candidate index. This is inspectable and adequate for
the current bounded candidate count. It is not a globally optimal Hungarian or
joint probabilistic assignment; Phase 11 crossing/background evaluation should
decide whether that extra complexity is justified.

Candidate residuals and Mahalanobis distances are evaluated as vectorized
arrays for each predicted track. Assignment remains scalar and deterministic,
so this optimization does not alter gating or lifecycle semantics.

The covariance update uses the Joseph form and is symmetrized after prediction
and correction. Unit tests require it to remain finite, symmetric, and positive
definite.

## Independent confirmation

The initial evidence policy is `non_overlapping_frames`. A new track starts
with one independent hit. Associated sliding windows may improve its Kalman
state, but they do not increment confirmation evidence until every frame index
in the new window is later than the last credited window.

For four-frame windows with stride one, windows `[0,1,2,3]`, `[1,2,3,4]`,
`[2,3,4,5]`, and `[3,4,5,6]` contribute only one independent hit. Window
`[4,5,6,7]` provides the second. This prevents a single target/noise event from
being counted four times merely because the integration windows overlap.

Detector SNR remains under `detector_evidence` and is explicitly marked as not
track confidence. Confirmation progress is the ratio of independent hits to
the configured requirement; it is not represented as a calibrated
probability.

## Lifecycle and resets

- `tentative`: not enough independent hits;
- `confirmed`: confirmation requirement met and currently measured;
- `coasted`: a confirmed track is temporarily unmeasured but inside the miss
  budget;
- `deleted`: the miss budget was exceeded or the coordinate state reset.

Tracks reset across stabilization segment changes, non-increasing reference
timestamps, and excessive timestamp gaps. IDs are never reused inside one
manager instance. Candidate births are score-ordered by Phase 9 and bounded by
`max_active_tracks`; rejected births are counted.

## Deterministic characterization

The Phase 10 benchmark covers persistent motion in overlapping windows,
crossing targets, and isolated false candidates. At the synthetic operating
point:

- the persistent target confirms after 0.4 seconds, when the second disjoint
  four-frame window becomes available;
- position RMSE is 0.285 px and velocity RMSE is 0.033 px/s;
- crossing identities remain intact with zero association failures;
- fragmentation is zero;
- 12 isolated false candidates produce zero confirmed tracks.

The false-confirmed-track rate is zero for this small deterministic synthetic
fixture, not a real-scene false alarm claim. Phase 11 must measure it over
substantial background-only footage.

Run with:

```bash
python3 -m tiny_target.tracking_benchmark \
  --output /tmp/tracking_benchmark.json

python3 -m unittest discover -s tests
```

The checked local report is
`results/tiny_target/phase10/tracking_benchmark.json`; the matching Jetson
report is `results/tiny_target/phase10/tracking_benchmark_jetson_v2.json`.
End-to-end reports use schema `seaqr.tiny-target.motion.v9` and include both
Phase 9 candidate batches and Phase 10 track batches.

## Jetson RAW16 characterization

The isolated Jetson workspace rebuilt CUDA for SM 8.7 and passed all 102 tests
with warnings treated as errors. The live checkout remained untouched. The
same four RAW16 candidate windows from Phase 9 produced:

- 1,024 bounded input candidates;
- 698 candidate-to-track associations;
- 326 track births and 17 miss-budget deletions;
- 309 peak active tracks, below the configured 512-track cap;
- zero rejected births at the active-track cap;
- zero confirmed tracks.

Zero confirmation is the correct result for this short diagnostic. Its four
windows are `[7..10]`, `[8..11]`, `[9..12]`, and `[10..13]`; all overlap the
first credited frame set. A second independent hit would require a window that
starts after frame 10. The result therefore validates the overlap policy, not
the absence of real targets.

Vectorized association reduced median tracking time from 637.34 ms in the first
hardware characterization to 89.10 ms without changing any association or
lifecycle count. Candidate extraction remains about 373 ms/window and is now
the larger Phase 9/10 host-side cost.

The checked end-to-end report is
`results/tiny_target/phase10/raw16_cuda_tracking_15pairs_v2.json`. Its warnings
retain the important constraints: candidate output caps trigger in every
window, timestamps are host callback rather than exposure time, the velocity
grid is zero-only, the PSF and measurement covariance are provisional, and no
camera-calibrated detection/tracking claim is justified.
