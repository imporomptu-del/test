# Phase 13: Track-Quality Discrimination

## Purpose and safety boundary

The 48-frame Phase 12 moving-target sweep confirms all four injected targets at
local-CFAR threshold 8, but also produces 60 confirmed or coasted tracks that
do not match injected truth. Raising the detector threshold removes individual
candidates much faster than it removes these persistent responses. Phase 13
therefore measures evidence accumulated along a track rather than treating
confirmation alone as sufficient quality.

This phase does **not** reject tracks yet. The available RAW scene is unlabeled,
the four positive examples are injected into the same clip used to discover
the policy, and their Gaussian PSF exactly matches the provisional detector
template. An unmatched track may be structured clutter or a real object.
Applying a same-run cutoff as a production gate would overstate both positive
and negative ground truth.

## Bounded online telemetry

Each track now maintains fixed-memory running population moments. No sample
history or unbounded queue is retained. Serialized `quality_evidence` includes:

- observation count and observation fraction over track age;
- raw detector-score and local-CFAR selection-score moments;
- peak contrast and peak-to-neighbor-ratio moments;
- temporal support and measured-speed moments;
- measurement-velocity step moments; and
- Kalman Mahalanobis, position-residual, and velocity-residual moments.

Every quality block declares
`used_for_association_or_confirmation: false`. Association gates, Kalman state,
independent-hit confirmation, lifecycle transitions, and candidate output are
unchanged.

The offline analyzer labels a track `injected_truth_matched` only when its
confirmed/coasted state passes the existing 3 px position and 2 px/s velocity
truth gates. Everything else is `unmatched_to_injected_truth`, not a false
track. It reports feature distributions, one-feature discovery envelopes,
two-feature discovery envelopes, and causal qualification over every serialized
track snapshot.

Run it with:

```bash
python3 -m tiny_target.track_quality_evaluation \
  --motion-report results/tiny_target/phase12/raw16_cfar_moving_48frames_track_quality_cfar8_jetson.json \
  --diagnostic-min-observations 5 \
  --diagnostic-min-observation-fraction 1 \
  --diagnostic-max-score-std 8 \
  --diagnostic-min-peak-ratio 1.05 \
  --output /tmp/track_quality.json
```

## Checked Jetson evidence

The instrumented threshold-8 replay preserves the Phase 12 outcome: 61 of 64
target opportunities match, all four injected targets confirm, and 60 of 64
ever-confirmed/coasted tracks remain unmatched to injected truth. All four
targets receive complete observable injection coverage.

The strongest individual separators at the final track observation are:

| Feature | Injected-track range | Unmatched median | Unmatched retained by injected envelope |
|:---|:---|---:|---:|
| Mean peak/neighbor ratio | 1.065–1.105 | 1.025 | 14/60 |
| Detector-score standard deviation | 2.15–7.03 | 13.55 | 15/60 |
| Associated updates | 8 for every target | 7 | 26/60 |
| Observation fraction | 1 for every target | 1 | 40/60 |

Kalman smoothness is not the separator on this scene. Unmatched tracks have a
median velocity-residual RMS of only 0.056 px/s, compared with 0.861 px/s for
the injected tracks. A naive low-innovation gate would preferentially retain
stable structured clutter and risk rejecting real moving targets.

The exact same-run two-feature envelope—detector-score standard deviation no
greater than 7.034 and mean peak/neighbor ratio at least 1.065—retains all four
injected tracks and 6 of 60 unmatched tracks. Those extrema are overfit and are
not configuration values.

A rounded diagnostic policy is evaluated causally over every track snapshot:

```text
confirmed or coasted
observations >= 5
observation fraction == 1.0
detector-score standard deviation <= 8.0
mean peak/neighbor ratio >= 1.05
```

All four injected tracks satisfy it at their existing first confirmation,
1.225 s after track birth. Eight unmatched tracks satisfy it at least once and
six remain qualified at their last observation. This reduces the same-run
ever-qualified unmatched burden from 60 to 8 (86.7%) without added target
latency. The analyzer explicitly records that the policy was not used by the
live tracker.

## What this establishes

Phase 13 establishes that fixed-memory track telemetry is feasible and that
peak isolation plus score stability are plausible discriminators on the
checked scene. It also rejects the intuitive but incorrect hypothesis that
lower Kalman innovation identifies the injected moving targets here.

It does not establish a deployable gate. Before track rejection is enabled,
the rounded policy must be frozen and evaluated on held-out clips containing:

- verified-empty RAW scenes for a valid false-qualified-track rate;
- labeled real targets with measured, mismatched PSFs;
- injected targets across independent positions and backgrounds;
- sub-grid velocity, acceleration, intermittent visibility, and missed-window
  cases; and
- enough duration to measure qualification stability, revocation, latency,
  fragmentation, and hourly burden.

The primary evidence files are
`results/tiny_target/phase12/raw16_cfar_moving_48frames_track_quality_cfar8_jetson.json`
and
`results/tiny_target/phase12/raw16_cfar_moving_48frames_track_quality_cfar8_analysis_jetson_v2.json`.

## Held-out result

The rounded score-stability/peak-isolation policy failed independent-scene
validation and remains disabled. It retained only one of four injected targets
on cadence-corrected chunk 0050 and three of four on the 64-frame chunk 0025
coverage experiment. Both scenes otherwise matched every valid injected-target
opportunity and confirmed all four targets. See
`docs/phase13_heldout_validation.md` for the timestamp-cadence correction,
spatial-observability results, and the next persistence/motion hypothesis.
That hypothesis was subsequently frozen and tested on independently screened
chunk 0040. It qualifies every truth-matched track for the three injected
targets that confirm and reduces 542 unmatched confirmed/coasted tracks to 17
qualified at their latest observation. The 1,000-DN target does not confirm
because CFAR 8 selects it in only 3 of 49 opportunities, placing the next
bottleneck before track-quality gating.

A causal persistence/motion evaluator is now available separately from the
rejected rule. With five observations, at least 96% observation coverage, and
mean measured speed of at least 0.25 px/s, it qualifies all 12 injected tracks
in the three discovery/calibration scenes. On independently selected chunk
0040 it qualifies every truth-matched track for the three targets that confirm,
while 17 of 542 unmatched tracks remain qualified at their latest observation.
It remains explicitly non-live.
