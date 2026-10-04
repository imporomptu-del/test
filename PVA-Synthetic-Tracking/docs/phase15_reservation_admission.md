# Phase 15: motion-gated reservation admission

## Motivation

Phase 14's continuous-tentative reservation confirms the quota-starved
1,000-DN target in chunk 0040, but adds 47 unmatched confirmed/coasted tracks.
On transfer chunk 0060 it adds 37 unmatched confirmed/coasted tracks and no
target detections. A reservation deliberately occurs at an independent
confirmation boundary, so admitting persistent stationary structure tends to
convert it directly into a confirmed track.

Phase 15 narrows who may enter that feedback path. It does not change detector
thresholds, spatial quotas, confirmation rules, or the global candidate cap.

## Causal admission analysis

The saved chunk 0040 and 0060 tracker batches were replayed without changing
state or consulting future windows. For each of 119 retained reservations, the
analysis captured the guiding track's prior-only evidence and the candidate's
residual from its prediction. Only one reservation is truth matched: chunk
0040 track 165.

Tightening residual gates alone is ineffective. Of the 119 observed
reservations, 101—including track 165—are within 1 px and 1 px/s. Most
unmatched reservations have exactly zero residual because persistent scene
structure is stationary and highly repeatable.

Before its rescued window, track 165 has four observations in four age windows
and a mean measured speed of 0.25 px/s. This is the same motion boundary frozen
during Phase 13, now evaluated before reservation rather than after a reserved
zero-velocity measurement dilutes the mean. Requiring prior mean measured speed
of at least 0.25 px/s retains 13 of the 119 historical reservation events.
Combining it with a one-grid-bin velocity residual retains seven, including
track 165. This retrospective count is a screening result, not validation,
because changing admission can change later tracker state and proposals.

## Frozen operating point

The next hardware test is declared before replay:

- local-CFAR threshold: 8;
- velocity grid: -3 to +3 px/s in each axis, 1 px/s step;
- ordinary spatial quota: 4 per 8 x 8 cell;
- reservation position radius: 3 px;
- reservation velocity radius: 1 px/s;
- minimum prior mean measured speed: 0.25 px/s;
- continuous, zero-miss tentative history for at least one four-frame window;
- reservation cap: 8; global candidate cap remains 256.

The 3-px position radius is the existing truth-matching tolerance, the 1-px/s
velocity radius is one discrete velocity-grid step, and 0.25 px/s was frozen in
Phase 13 before this admission analysis. The rule is first replayed on chunk
0040 to verify it still rescues track 165, then transferred unchanged to chunk
0060 against the existing paired baselines.

## Implementation and telemetry

`TrackPredictionHint` now includes the causal running mean of measured speed.
The candidate configuration and CLI expose
`track_guided_reservation_minimum_mean_speed_px_s` and
`--track-reservation-minimum-mean-speed-px-s`. The value is recorded in the
effective configuration and per-window reservation telemetry. A deterministic
test verifies that a mature stationary tentative track is ineligible while a
track at the configured motion boundary remains eligible.

The feature remains disabled by default.

## Isolated Jetson results

The implementation passes 142 local tests with 11 environment-dependent skips
and 143 Jetson tests including CUDA coverage. Both hardware replays use the
frozen operating point above and are paired with the existing no-reservation
baselines.

### Chunk 0040 verification

Both runs have 63/63 accepted transforms, 58 detection-ready frames, 52
synthetic windows, and 49 velocity trials.

| CFAR-8 metric | Quota-4 baseline | Phase 15 admission |
|:---|---:|---:|
| Target opportunity matches | 139/196 | 140/196 |
| 1,000-DN opportunity matches | 3/49 | 4/49 |
| Confirmed injected target IDs | 3/4 | 4/4 |
| Unmatched candidate burden | 8,260 | 8,262 |
| Unmatched confirmed/coasted tracks | 542 | 544 |
| Frozen-policy unmatched, latest observation | 17 | 17 |

Only three reservations are retained across two windows, down from 68 under
Phase 14's broad gate. Candidate 69 in frames 10--13 is still truth matched to
the 1,000-DN target and records `reservation_track_id: 165`; that evidence
confirms the fourth target. The other two reservations are unmatched. Median
candidate extraction changes from 360.57 ms to 367.55 ms (+1.9%).

### Chunk 0060 frozen transfer

Both runs have 63/63 accepted transforms, 56 detection-ready frames, 47
synthetic windows, and 49 velocity trials.

| CFAR-8 metric | Quota-4 baseline | Phase 15 admission |
|:---|---:|---:|
| Target opportunity matches | 131/176 | 131/176 |
| Confirmed injected target IDs | 4/4 | 4/4 |
| Unmatched candidate burden | 7,196 | 7,200 |
| Unmatched confirmed/coasted tracks | 452 | 454 |
| Frozen-policy unmatched, latest observation | 29 | 29 |

Four reservations occur across two windows, none truth matched. Target metrics
remain identical, while unmatched confirmed/coasted tracks increase by two.
Median candidate extraction changes from 367.24 ms to 371.29 ms (+1.1%).

### Combined verdict

Across the two scenes, Phase 15 changes target opportunity matches from 270/372
to 271/372 and confirmed injected targets from 7/8 to 8/8. Unmatched candidate
burden changes from 15,456 to 15,462 (+6, 0.04%), and unmatched
confirmed/coasted tracks change from 994 to 998 (+4, 0.4%). The frozen policy's
latest unmatched count remains 46. No replay reaches either the reservation or
global candidate cap, and no baseline candidate is displaced.

This is a successful synthetic transfer at the tested operating point: the
motion admission retains the demonstrated weak-target rescue while removing
112 of Phase 14's 118 incremental unmatched candidates and 80 of its 84
incremental unmatched tracks. It remains disabled by default because two
injected scenes do not establish real-target sensitivity or a false-track rate.
The recovered chunk 0040 weak target still fails the downstream frozen
motion/persistence diagnostic after confirmation, so that diagnostic remains
non-gating.

## Evidence files

- `results/tiny_target/phase15/chunk_0040_64frames_cfar8_motion025_reservation3_1_cap8_vgrid7x7_jetson.json`
- `results/tiny_target/phase15/chunk_0040_64frames_cfar8_motion025_reservation3_1_cap8_vgrid7x7_accuracy.json`
- `results/tiny_target/phase15/chunk_0040_64frames_cfar8_motion025_reservation3_1_cap8_vgrid7x7_motion_policy.json`
- `results/tiny_target/phase15/chunk_0060_64frames_cfar8_motion025_reservation3_1_cap8_vgrid7x7_jetson.json`
- `results/tiny_target/phase15/chunk_0060_64frames_cfar8_motion025_reservation3_1_cap8_vgrid7x7_accuracy.json`
- `results/tiny_target/phase15/chunk_0060_64frames_cfar8_motion025_reservation3_1_cap8_vgrid7x7_motion_policy.json`
