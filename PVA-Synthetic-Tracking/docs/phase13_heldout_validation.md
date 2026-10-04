# Phase 13 Held-Out Validation

## Outcome

Independent RAW16 scenes reject the Phase 13 score-stability/peak-isolation
policy as a live gate. The policy remains diagnostic and is not used for
association, confirmation, or track output.

The validation also found two upstream eligibility issues before a tracking
comparison was possible:

1. video sidecars were incorrectly checked against the container's nominal
   frame rate instead of the acquisition cadence in the sidecar; and
2. PVA feature coverage varies materially by scene, including one genuinely
   low-observability clip.

## Auditable source and cadence handling

The motion CLI accepts paired `--input-video` and `--timestamp-csv` overrides
and records their absolute paths in the effective configuration. Timestamp-gap
detection now derives its expected interval from the median positive sidecar
delta when explicit timestamps are available. It falls back to the container
frame rate only when that estimate is unavailable. Reports record the interval,
its source, and the configured gap factor under
`source.timestamp_gap_detection`.

This matters on the checked recordings: the container advertises 10 fps while
the sidecars have a median interval near 319.6 ms. The old 400 ms gap boundary
therefore mislabeled ordinary 420 ms acquisition intervals and repeatedly reset
the coordinate/background state. With sidecar-derived cadence and the existing
4x factor, chunk 0050 has zero timestamp-gap resets.

An explicit `--minimum-grid-coverage` experiment override changes both the PVA
correspondence gate and global-motion inlier gate and records both effective
values. It exists for characterization; it does not alter the configuration
default.

## Fixed held-out scenes

Chunks 0025, 0050, and 0075 were selected before inspecting their tracking
outcomes. All use lossless 4784 x 3190 gray16 FFV1 recordings and matching
timestamp sidecars from
`/home/serg/project/camera_reader_sky/srcsky/chunks_raw16_test/`.

At the default 20% spatial-coverage gate:

| Clip | Typical occupied PVA cells | Accepted pairs (48-frame run) | Result |
|:---|---:|---:|:---|
| 0025 | 9/48 | 8/47 | Below the discrete 10-cell requirement; no tracking window |
| 0050 | 10/48 or more initially | 31/47 | 25 windows after cadence fix |
| 0075 | 5/48 | 0/47 | Fails closed as low spatial observability |

PVA correspondence accuracy was not the failure in these cases. Typical
forward/backward errors were about 0.01--0.03 px, and translation-fit residuals
were similarly small. The limiting signal was spatial distribution.

Chunk 0075 remains rejected. Relaxing the default from 10 occupied cells to 5
would remove most of the spatial-observability protection and is not justified
by this experiment.

Chunk 0025 was also run with a narrowly scoped 18% coverage threshold, which
accepts 9 occupied cells. The 48-frame run matched all 28 available target
opportunities but ended before two non-overlapping confirmation windows were
available. Extending the identical experiment to 64 frames resolved that
boundary condition.

## Detection and tracking results

All results below use local-CFAR 8 and the fixed 7 x 7 velocity grid from
-3 to +3 px/s on each axis.

| Scene | Target matches | Confirmed injected targets | Unmatched confirmed/coasted tracks | Old diagnostic policy: targets / unmatched ever qualified |
|:---|---:|---:|---:|---:|
| 0001 discovery replay | 61/64 | 4/4 | 60 | 4 / 8 |
| 0050 cadence-corrected held-out | 88/88 | 4/4 | 204 | 1 / 26 |
| 0025 64-frame, 18% coverage experiment | 92/92 | 4/4 | 165 | 3 / 15 |

The held-out target confirmations are accurate. On chunk 0050, first-confirmed
position errors are at most 0.51 px and velocity errors at most 1.19 px/s. On
chunk 0025, first-confirmed position errors are at most 0.52 px and velocity
errors at most 1.25 px/s. These are inside the existing 3 px and 2 px/s truth
gates.

The frozen diagnostic rule from the discovery clip—at least five observations,
complete observation fraction, detector-score standard deviation at most 8,
and mean peak/neighbor ratio at least 1.05—retains only 8 of the 12 injected
tracks across the three successful scenes. It is overfit and must not become a
live filter.

## Next discriminator hypothesis

Across the three successful runs, injected targets are more consistently
distinguished by persistence plus nonzero selected motion than by absolute
score stability:

```text
observation fraction of age >= 0.96
mean measured speed >= 0.25 px/s
```

Applied retrospectively to each track's last observation, that combined
envelope retains all 12 injected tracks and 26 of 429 unmatched tracks. This is
a 93.9% reduction in unmatched burden, but it is derived from all three scenes
and is therefore another discovery hypothesis, not held-out evidence. It also
does not establish that unmatched tracks are false objects.

## Frozen-policy unseen test

The persistence/motion values were frozen before selecting a new scene:

```text
confirmed or coasted
observations >= 5
observation fraction of age >= 0.96
mean measured speed >= 0.25 px/s
```

Chunk 0090 was preselected first and retained as an upstream eligibility
failure: its pairs have only 22--27 accepted correspondences and occupy 5--7
of 48 grid cells, so all 63 transforms fail closed at the default gate.

A fallback order of 0010, 0040, 0060, and 0080 was declared before a four-frame
PVA-only screen. Chunks 0010 and 0080 failed the independent motion prerequisite;
0040 was the first passing scene and was selected without inspecting detection
or policy output. Chunk 0060 also passed but was not selected.

The 64-frame chunk 0040 run has 63/63 accepted transforms, no resets, 58
detection-ready frames, and 52 synthetic windows. Its threshold-8 result is:

| Metric | Result |
|:---|---:|
| Injected target opportunities matched | 139/196 (70.9%) |
| 1,000-DN opportunity recall | 3/49 (6.1%) |
| 2,000-DN opportunity recall | 47/49 (95.9%) |
| 4,000-DN opportunity recall | 41/49 (83.7%) |
| 8,000-DN opportunity recall | 48/49 (98.0%) |
| Injected target IDs confirmed | 3/4 |
| Unmatched confirmed/coasted tracks | 542 |
| Frozen policy: injected track IDs ever qualified | 6/6 across the 3 confirmed target IDs |
| Frozen policy: unmatched tracks ever qualified | 95/542 |
| Frozen policy: unmatched tracks qualified at latest observation | 17/542 |

The unconfirmed 1,000-DN target fails at candidate selection, before a quality
policy can act. The policy therefore transfers conditionally for every target
that confirms, while reducing the latest unmatched burden by 96.9%. It does not
recover detector false negatives, and the 17 unmatched tracks are not verified
false objects.

## CFAR and spatial-quota diagnosis

A frozen CFAR 6/7/8 sweep on the same chunk 0040 score surfaces leaves target
recall unchanged at every threshold: 139/196 overall and 3/49 for the 1,000-DN
target. Lowering CFAR only increases unmatched candidates and tracks. The weak
target's diagnostic local-CFAR score is already 13.5--28.3, so it is not failing
the tested thresholds.

Its 8 x 8 quota cell is full at 4/4 outputs in all 49 target windows. The fourth
retained peak exceeds the target's local score by a median 8.0 sigma. A recorded
quota-8 experiment confirms the starvation diagnosis:

| CFAR 8 result | Quota 4 default | Quota 8 experiment |
|:---|---:|---:|
| Total target matches | 139/196 | 147/196 |
| 1,000-DN target matches | 3/49 | 9/49 |
| Confirmed injected target IDs | 3/4 | 4/4 |
| Unmatched candidate burden | 8,260 | 13,003 |
| Unmatched confirmed/coasted tracks | 542 | 869 |
| Frozen-policy unmatched, latest observation | 17 | 36 |
| Windows truncated at global 256-candidate cap | 0/52 | 47/52 |

Quota 8 recovers confirmation but is not a viable global default: it saturates
the bounded output on nearly every window and materially increases downstream
burden. It remains an experiment-only runtime override.

Phase 14 implements bounded track-guided reservation. The global per-cell quota
stays at 4, while a small fixed number of otherwise discarded peaks near causal,
non-mutating predictions from already-born tracks may be retained. The default
path remains disabled, and the global candidate bound is unchanged. See
`docs/phase14_track_guided_candidate_reservation.md`.

The quality policy still needs labeled real targets with a measured, mismatched
PSF. A verified-empty dataset is required before reporting a false-track rate.

## Evidence files

- `results/tiny_target/phase13/heldout_chunk_0050_cfar8_cadencefix_jetson.json`
- `results/tiny_target/phase13/heldout_chunk_0050_cfar8_cadencefix_accuracy.json`
- `results/tiny_target/phase13/heldout_chunk_0050_cfar8_cadencefix_track_quality.json`
- `results/tiny_target/phase13/heldout_chunk_0025_64frames_cfar8_coverage018_jetson.json`
- `results/tiny_target/phase13/heldout_chunk_0025_64frames_cfar8_coverage018_accuracy.json`
- `results/tiny_target/phase13/heldout_chunk_0025_64frames_cfar8_coverage018_track_quality.json`
- `results/tiny_target/phase13/heldout_chunk_0075_cfar8_jetson.json`
- `results/tiny_target/phase13/heldout_chunk_0090_64frames_cfar8_jetson.json`
- `results/tiny_target/phase13/heldout_chunk_0040_64frames_cfar8_jetson.json`
- `results/tiny_target/phase13/heldout_chunk_0040_64frames_cfar8_accuracy.json`
- `results/tiny_target/phase13/heldout_chunk_0040_64frames_cfar8_motion_policy.json`
- `results/tiny_target/phase13/chunk_0040_64frames_cfar6_7_8_sweep_jetson.json`
- `results/tiny_target/phase13/chunk_0040_64frames_cfar6_7_8_sweep_accuracy.json`
- `results/tiny_target/phase13/chunk_0040_64frames_cfar6_7_8_sweep_motion_policy.json`
- `results/tiny_target/phase13/chunk_0040_64frames_cfar8_quota8_jetson.json`
- `results/tiny_target/phase13/chunk_0040_64frames_cfar8_quota8_accuracy.json`
- `results/tiny_target/phase13/chunk_0040_64frames_cfar8_quota8_motion_policy.json`
