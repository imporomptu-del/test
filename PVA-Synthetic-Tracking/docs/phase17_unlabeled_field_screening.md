# Phase 17: untouched Jetson field screening

## Outcome

The frozen Phase 15 PVA plus synthetic-tracking pipeline was run on the first
64 frames of three previously unused RAW16 Jetson clips: chunks 0020, 0070,
and 0095. No synthetic targets were injected and no setting was tuned between
clips.

Chunks 0020 and 0095 reached detection and tracking. Chunk 0070 did not: every
frame-to-frame transform failed the existing correspondence or spatial-coverage
gate, so stabilization reset on every pair and never produced a four-frame
detection window. This is a useful fail-closed result, not evidence that the
clip contains no objects.

The two usable clips generated substantial screening workload. Across them,
the CFAR-8 evaluation stream produced 14,937 candidates and 879 unique tracks
that appeared in a confirmed or coasted state. The frozen, diagnostic-only
motion/persistence policy was satisfied at least once by 115 tracks and by 11
tracks at their latest observation. These are not 115 real objects, and the
remaining tracks are not established false alarms: all three clips are
unlabeled.

Visual review of the rendered tracks and source-pixel crops did not establish
an obvious point target. Most displayed examples are coincident with roof or
horizon structure, cloud boundaries, or broad gradients. That appearance is
consistent with structured-clutter contamination, but it is not a truth label.

## Untouched-clip selection

Before selection, the checked repository documents, configurations, and result
names were searched for prior clip references. Chunks 0020, 0070, and 0095 did
not occur in that history; previously used or discussed chunks were excluded.
The selected files are separated across the numbered capture collection rather
than being adjacent samples.

Each source is an FFV1, 4784 x 3190, `gray16le` recording in:

```text
/home/serg/project/camera_reader_sky/srcsky/chunks_raw16_test
```

Each run required its matching `_timestamps.csv` sidecar. The full recordings
contain 863, 897, and 935 frames respectively; Phase 17 intentionally used the
same 64-frame prefix from each so the comparison had equal observation length.
The runs were performed only in the isolated Jetson workspace
`/tmp/seaqr_phase12_20260905`. The live checkout was not modified.

## Frozen run

The same command shape was used for every clip:

```bash
python3 -m tiny_target.motion_cli \
  --config configs/tiny_target_phase12_cfar_test.yaml \
  --input-video /home/serg/project/camera_reader_sky/srcsky/chunks_raw16_test/chunk_NNNN.mkv \
  --timestamp-csv /home/serg/project/camera_reader_sky/srcsky/chunks_raw16_test/chunk_NNNN_timestamps.csv \
  --max-frames 64 \
  --velocity-grid=-3,3,-3,3,1 \
  --track-reservation-position-radius-px 3 \
  --track-reservation-velocity-radius-px-s 1 \
  --track-reservation-minimum-mean-speed-px-s 0.25 \
  --max-track-reservations-per-window 8 \
  --evaluation-cfar-thresholds 8 \
  --omit-points \
  --output results/tiny_target/phase17/chunk_NNNN_64frames_phase15_unlabeled_jetson.json
```

The reviewed stream is specifically
`evaluation.candidate_threshold_sweep["8"]`. That stream has its own CFAR-8
extractor and causal tracker while using the same Phase 15 reservation policy.
It must not be confused with the configuration's primary CFAR-6 stream.

Other material settings were:

- PVA Harris/PyrLK at 0.5 image scale, with at least 30 accepted features and
  0.20 grid coverage;
- translation-only RANSAC camera motion and reset-on-failure stabilization;
- provisional pixel-integrated Gaussian PSF, sigma 0.8 px;
- four-frame, stride-one CUDA shift-and-stack over 49 integer velocity trials,
  with both axes spanning -3 to +3 px/s;
- tile-robust CFAR ranking, four candidates per spatial quota cell, and a
  256-candidate global cap;
- track reservation bounded to eight candidates per window, a 3-px position
  gate, a 1-px/s velocity gate, and 0.25-px/s minimum prior mean speed.

The review overlay also applies the previously frozen diagnostic rule: at least
five observations, an observation fraction of at least 0.96, and mean measured
speed of at least 0.25 px/s. It does not remove or promote candidates in the
live tracker.

## Results

| Clip | Accepted transforms | Ready frames | CFAR-8 windows | Candidates | Unique confirmed/coasted tracks | Ever diagnostic-qualified | Qualified at latest observation | Reservations | End-to-end |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0020 | 50 / 63 | 47 | 44 | 6,434 | 365 | 52 | 5 | 3 | 413.2 s |
| 0070 | 0 / 63 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 61.2 s |
| 0095 | 56 / 63 | 53 | 50 | 8,503 | 514 | 63 | 6 | 4 | 452.3 s |

No CFAR-8 window reached the 256-candidate output cap. The five latest
diagnostic tracks in chunk 0020 are 53, 129, 163, 750, and 918. The six in
chunk 0095 are 7, 507, 676, 862, 1553, and 1602.

The 365 and 514 track counts are screening workload: they count unique track
identities observed in a confirmed or coasted output snapshot. They are not
object counts. Likewise, summing them across clips is safe only as workload,
because track identifiers are local to each run.

## Stabilization finding

Chunk 0070 produced a median of only 29 accepted PVA tracks per pair, with a
range of 23 to 41. Thirty-five pairs failed the minimum of 30 correspondences;
the other 28 failed the 0.20 spatial-coverage gate. Consequently all 63
transforms were rejected. The appropriate next investigation is why this scene
has sparse or concentrated usable features, followed by a predeclared
stabilization fallback experiment. Lowering the safety gates after seeing this
clip would turn the held-out screen into tuning data and is not part of this
result.

Chunks 0020 and 0095 also finish with consecutive low-grid-coverage failures
starting at frames 51 and 57 respectively. Those resets explain why the last
overlay frames can say that no current window or track exists even though
earlier frames contain many detections. They do not mean the end of either
scene is empty.

## Review artifacts

The preferred 1920 x 1080 review videos contain 64 frames at 3 fps. Red crosses
show up to the top 24 CFAR candidates, purple crosses show retained reservation
candidates, and cyan circles show tracks satisfying the diagnostic policy.
The right panel enlarges up to four 49 x 49 source-pixel crops. Every overlay is
marked `UNLABELED SCREENING`.

- `results/tiny_target/phase17/chunk_0020_phase15_review_crops.mp4`
- `results/tiny_target/phase17/chunk_0070_phase15_review_crops.mp4`
- `results/tiny_target/phase17/chunk_0095_phase15_review_crops.mp4`
- `results/tiny_target/phase17/phase17_screening_summary.json`
- per-clip compact summaries and track-review CSVs in the same directory;
- full `motion.v10` reports in the same directory for exact replay evidence.

`tiny_target.field_review` creates the compact summaries, review CSVs, and
overlays without inventing ground truth. Its output schema explicitly leaves
real-object count, false-alarm count, precision, and recall null.

## Interpretation and next boundary

This screen establishes two things:

1. The Phase 15 algorithm runs end to end on two genuinely untouched camera
   clips with the frozen operating point.
2. It is not yet operationally validated. One of three scenes cannot be
   stabilized, and the usable scenes produce a large number of tracks whose
   reviewed examples look dominated by structured scene content.

The immediate next step is a focused human review of the 11 tracks that remain
diagnostic-qualified at their latest observation, recording labels and review
notes in copies of the generated CSVs. That can quickly determine whether any
candidate merits extraction at native resolution. It still cannot produce a
defensible sensitivity or false-alarm metric. Those claims remain blocked on
the Phase 16 evidence contract: measured camera PSF, authoritative real-target
labels, and a separately verified-empty cohort.
