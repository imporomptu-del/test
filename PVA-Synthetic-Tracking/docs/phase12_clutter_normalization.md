# Phase 12: Structured-Clutter Normalization

## Why this phase exists

The Phase 11 RAW16 injection run showed that target signal survives the image
pipeline. The strongest injected target reached a median dense score of 198.6
SNR, but its median global score rank was 17,688. Candidate extraction retained
only the top 4,096 pre-NMS peaks and the top 256 final candidates, so the target
never reached the output.

That is a ranking failure, not a signal-preservation failure. Raising the caps
would transfer thousands of structured background responses into host-side NMS
and temporal tracking without fixing the ordering.

## Selection model

Phase 12 keeps two scores with deliberately different meanings:

- `normalized_score_snr` is the original shift-and-stack score. It remains the
  physical detector evidence, the minimum acceptance floor, and the value
  exposed to temporal tracking.
- `selection.ranking_score` is used only to compare candidates. In the new mode
  it measures how unusual the raw score is relative to its local image tile.

For each configured tile, the selector computes:

```text
center = median(valid raw SNR)
scale  = max(1.4826 * median(abs(raw SNR - center)), configured floor)
ranking score = (raw SNR - center) / scale
```

Tiles with too few valid samples fail closed. A candidate must pass both the
raw-SNR floor and the local-CFAR threshold. Spatial local maxima are still
defined on the raw score surface, so discontinuities between tile statistics
cannot manufacture peaks at tile boundaries.

This is robust local standardization commonly described as CFAR-like ranking.
It does not claim an analytically calibrated probability of false alarm: real
scene samples are correlated and non-Gaussian, and the current RAW16 recording
does not have verified-empty labels.

## Spatial quotas

Local normalization can still leave one difficult region with many highly
ranked peaks. Phase 12 therefore applies deterministic per-cell limits before
and after joint position/velocity NMS.

The checked Jetson trial configuration divides the full sensor into an 8 x 8
quota grid. Each cell may contribute at most 64 peaks to pre-NMS evaluation and
at most four candidates to the 256-candidate output. The order within and
between cells remains deterministic: descending ranking score, then ascending
flat pixel index.

The quota is a bounded-compute and coverage policy. It does not make an
unmatched candidate a false alarm, and it may suppress a real second target in
a crowded cell. Those tradeoffs must be measured with labeled multi-target and
stress-scene data before production use.

## Configuration and diagnostics

The Phase 11 raw-ranking behavior remains the default. The new behavior is
enabled explicitly in `configs/tiny_target_phase12_cfar_test.yaml` with:

```json
{
  "ranking_mode": "tile_robust_cfar",
  "score_threshold_snr": 8.0,
  "cfar_threshold_sigma": 6.0,
  "cfar_tile_height_px": 256,
  "cfar_tile_width_px": 256,
  "cfar_minimum_samples": 16384,
  "cfar_scale_floor_snr": 1.0,
  "quota_grid_rows": 8,
  "quota_grid_cols": 8,
  "pre_nms_candidates_per_cell": 64,
  "max_candidates_per_cell": 4
}
```

Candidate reports now include the ranking units, score, local tile center and
scale, and quota cell. Batch metrics record tile validity, robust-statistic
summaries, threshold counts, quota truncation, occupied cells, and maximum
output occupancy per cell.

Injected truth probes report both the raw surface rank and the selection
surface rank. The motion CLI sweeps the appropriate parameter explicitly:

```bash
python3 -m tiny_target.motion_cli \
  --config configs/tiny_target_phase12_cfar_test.yaml \
  --max-pairs 15 \
  --omit-points \
  --injection-spec configs/evaluation/raw16_injection_v1.json \
  --evaluation-cfar-thresholds 3,4,5,6,7,8 \
  --output /tmp/raw16_phase12_cfar_sweep.json
```

Raw-SNR and CFAR threshold sweeps are mutually exclusive. CFAR sweeps fail
closed unless the candidate configuration selects `tile_robust_cfar`.

## Controlled characterization

Run the deterministic selector benchmark with:

```bash
python3 -m tiny_target.clutter_benchmark \
  --output /tmp/phase12_clutter_benchmark.json
```

In its heterogeneous-clutter fixture, a 10-SNR target has raw global rank
18,632 and is absent from the bounded raw output. Tile-robust ranking gives it
9.994 local sigma, rank 1, and retains it. In the separate spatial-monopoly
fixture, the unbalanced eight-candidate output occupies one of four cells;
the quota-controlled output occupies all four.

These fixtures establish deterministic behavior and regression coverage. They
are not a camera operating-point calibration.

## Checked Jetson RAW16 result

The implementation was transferred to an isolated Jetson workspace, rebuilt
for SM 8.7, and exercised without modifying the live checkout. All 125 tests
whose fixtures were present passed with warnings treated as errors, including
the PVA, OpenCV, CUDA, CFAR, and quota tests. One unrelated hardware-summary
test could not start because the minimal approved transfer deliberately omitted
its historical Phase 8/10 result files.

The same 16 frames and injection specification used in Phase 11 were then run
through the Phase 12 configuration. At the checked primary CFAR threshold 6:

- 12 of 16 in-frame injected-target opportunities reached candidate output;
- the other four are the 2,000-DN target in a region with invalid integration
  support, giving 12/12 detection conditional on valid support;
- output contains 723 candidates across four windows instead of the Phase 11
  cap of 1,024, and no Phase 12 window reaches the 256-output cap;
- each window occupies 46–48 of the 64 quota cells and contains 180–182
  candidates;
- median localization error is 0.334 px and median velocity error is
  0.0276 px/s for the stationary injections in stabilized coordinates.

Raw target SNR is unchanged, as intended. Selection rank improves materially:

| Flux (DN) | Phase 11 median raw rank | Phase 12 median local rank | Result |
|---:|---:|---:|:---|
| 1,000 | 224,092 | 8,698 | 4/4 detected |
| 2,000 | no valid support | no valid support | not evaluable |
| 4,000 | 95,708 | 7,150 | 4/4 detected |
| 8,000 | 17,688 | 82 | 4/4 detected |

The 1,000- and 4,000-DN local ranks remain larger than the global pre-NMS cap;
their recovery demonstrates that spatially balanced selection, not CFAR alone,
is material to this result.

An extended sweep shows 12/12 supported detections from CFAR thresholds 3
through 24. Unmatched candidate burden decreases from 764 at threshold 3 to
245 at threshold 24. At threshold 28, the 1,000- and 4,000-DN targets are lost,
leaving 4/12 supported detections. The checked configuration remains at the
more conservative threshold 6 because selecting 24 from one injected scene
would overfit the operating point; the sweep is evidence for later calibration,
not a production threshold decision.

No track confirms. Only four completed windows are available and their frame
sets overlap, while the tracker correctly requires two non-overlapping pieces
of evidence. This is a sequence-length/confirmation-evidence limitation, not a
candidate miss.

The checked run takes 126.6 seconds for 16 frames (0.126 frames/s). Candidate
extraction itself has a 369.7 ms median, comparable to the Phase 11 374.6 ms;
the run remains far from real time and makes no live-throughput claim.

Reports are under `results/tiny_target/phase12/`. The primary evidence files
are `raw16_cfar_threshold_sweep_jetson.json` and
`raw16_cfar_accuracy_jetson_v2.json`; the upper sweep uses the corresponding
`raw16_cfar_upper_*` names.

### Extended confirmation run

A follow-up run overrides the input limit to 32 frames and keeps the injected
targets active after frame 7. The override is recorded in both the original and
effective configuration identities. Eighteen frames become detection-ready
and produce 12 candidate windows across frame sets 7–13 and 16–26; the gap is
caused by the configured global-change suppression/reset.

The 8,000-DN injection confirms as track 383 on window `[20, 21, 22, 23]` with
two independent hits. Its confirmation latency is 1.225 s, position error is
0.302 px, and velocity error is 0.0116 px/s. This is the first checked
truth-matched confirmed target in the RAW16 pipeline.

The weaker stationary injections do not confirm. The 1,000- and 4,000-DN
targets are detected in the first four windows, but after the reset they are
present during background-model warm-up and are largely absorbed into the
stationary background. The 2,000-DN location remains outside valid integration
support. Across all windows the run therefore detects 20/36 valid-support
opportunities and confirms one of the three injected targets that ever had
valid support.

The tracker reports 64 confirmed tracks, but only track 383 matches injected
truth under the configured position/velocity gates. The remaining 63 tracks
are unmatched persistent responses in an unlabeled scene. They cannot be
called real objects or false tracks without labels, but they demonstrate that
temporal confirmation alone does not yet reject the structured background.
The machine-readable association is in
`raw16_cfar_tracking_32frames_accuracy_jetson_v2.json`.

The corresponding 32-frame visualization is
`media/phase12_raw16_tracking32_annotated.mp4`. Its panel shows candidate
detections separately from truth-matched confirmed tracks; frame 23 displays
track 383 after confirmation.

This extended run takes 249.7 seconds for 32 frames (0.128 frames/s). It still
uses the intentionally limited zero-velocity diagnostic grid. A moving-target
confirmation claim requires calibrated nonzero velocity hypotheses and an
injection whose trajectory is not learned as stationary background.

### Moving-target velocity-grid run

The next checked run processes 48 frames and overrides the synthetic-tracking
grid to the 49 integer velocity hypotheses from -3 through +3 px/s on both
axes. Four targets become active at frame 7 with source-plane velocities
`(+1,-1)`, `(-2,+1)`, `(+2,+2)`, and `(-3,0)` px/s and fluxes from 1,000 to
8,000 DN. The video source rebases its sidecar timestamps to zero, so the
injection specification uses `reference_timestamp_ns: 0`.

An initial attempt incorrectly used the absolute Unix timestamp from the CSV.
That made every trajectory fall billions of pixels outside the image and
injected zero flux. Its `raw16_cfar_moving_48frames_jetson.json` artifact is
retained only as invalid-run provenance; none of its target-accuracy results
are evidence. Injection evaluation now fails if any configured target is not
active in the processed frame range or never receives positive in-frame
support, and valid reports record a per-target coverage summary.

The corrected report records 41 active processed frames and the complete
requested in-frame flux for every target. Quantization and clipping retain
614,808 of 615,000 requested DN (99.969%). Across the 16 complete detection
windows at local-CFAR threshold 6:

- 61 of 64 truth opportunities match a candidate, for 95.3% detection
  probability conditional on valid support;
- detection by flux is 16/16 at 1,000 DN, 15/16 at 2,000 DN, 14/16 at 4,000
  DN, and 16/16 at 8,000 DN;
- all four injected targets form truth-matched confirmed tracks after two
  non-overlapping hits, each first confirming after 1.225 s;
- median localization error is 0.411 px (0.505 px RMSE), and median velocity
  error is 0.996 px/s (1.001 px/s RMSE) under the 3 px position and 2 px/s
  velocity matching gates;
- candidate output contains 2,129 items, with a maximum of 185 in one window
  and no final-output cap truncation; and
- 71 tracks confirm or coast, of which four match injected truth and 67 remain
  unmatched in the unlabeled scene.

The result demonstrates moving-target sensitivity and end-to-end temporal
confirmation on this sequence. It does not establish specificity: unmatched
candidates and tracks are a large burden, but cannot be labeled false alarms
without verified-empty or exhaustively labeled source data. The 48-frame run
takes 263.8 seconds (0.182 frames/s). Median CUDA kernel and GPU-total time per
synthetic window are 208.1 ms and 266.1 ms respectively; the complete pipeline
is still far from real time.

The machine-readable artifacts are
`raw16_cfar_moving_48frames_corrected_jetson.json` and
`raw16_cfar_moving_48frames_corrected_accuracy_jetson.json`. The corresponding
48-frame visualization is
`media/phase12_raw16_moving48_corrected_annotated.mp4`.

### Moving-target threshold frontier

An eight-point follow-up evaluates the same corrected 48-frame input and 49
velocity hypotheses without changing the injected truth. The RAW source is
unlabeled, so the quantities below are unmatched burden rather than certified
false alarms.

| Local-CFAR threshold | Matched opportunities | Detection probability | Confirmed injected targets | Unmatched candidates | Unmatched confirmed/coasted tracks |
|---:|---:|---:|---:|---:|---:|
| 6 | 61/64 | 95.3% | 4/4 | 2,068 | 67 |
| 8 | 61/64 | 95.3% | 4/4 | 1,571 | 60 |
| 10 | 60/64 | 93.8% | 4/4 | 1,340 | 61 |
| 12 | 58/64 | 90.6% | 4/4 | 1,228 | 60 |
| 16 | 55/64 | 85.9% | 4/4 | 1,118 | 60 |
| 20 | 51/64 | 79.7% | 4/4 | 1,038 | 55 |
| 24 | 43/64 | 67.2% | 3/4 | 964 | 52 |
| 28 | 28/64 | 43.8% | 2/4 | 896 | 49 |

Threshold 8 strictly improves the checked threshold-6 point on this clip: the
same 61 target matches, four confirmations, first-confirmation evidence, and
error metrics are retained while unmatched candidate burden falls 24.0% and
unmatched confirmed/coasted tracks fall 10.4%. Threshold 20 retains target-level
confirmation but loses 10 window-level target opportunities. At threshold 24,
the 4,000-DN target no longer confirms; at 28, only the 2,000- and 8,000-DN
targets confirm. Local clutter, not flux alone, therefore controls survival.

Threshold 8 is the conservative candidate for held-out validation, not a new
production setting. The checked configuration remains at 6 until the result is
repeated on independent scenes. More importantly, the shallow reduction from
67 to 60 unmatched tracks shows that threshold tuning alone does not solve
persistent structured clutter. The next implementation work should add track-
level discrimination and evaluate it on verified-empty and labeled clips.

The sweep artifacts are
`raw16_cfar_moving_48frames_threshold_sweep_jetson.json` and
`raw16_cfar_moving_48frames_threshold_sweep_accuracy_jetson.json`. The
threshold-8 visualization is
`media/phase12_raw16_moving48_cfar8_annotated.mp4`.

## Human-viewable comparison

The Phase 12 renderer uses the same injected source frames as the Phase 11
video. Red crosses show the top 30 globally ranked candidates, colored circles
show truth, and a white cross with green ring identifies a matched candidate.
The panel keeps raw SNR/rank and local-CFAR score/rank separate.

```bash
python3 -m tiny_target.evaluation_visualizer \
  --video results/tiny_target/phase11/media/chunk_0001.mkv \
  --timestamps results/tiny_target/phase11/media/chunk_0001_timestamps.csv \
  --motion-report results/tiny_target/phase12/raw16_cfar_threshold_sweep_jetson.json \
  --accuracy-report results/tiny_target/phase12/raw16_cfar_accuracy_jetson_v2.json \
  --injection-spec configs/evaluation/raw16_injection_v1.json \
  --threshold-snr 8 \
  --max-frames 16 \
  --output-fps 2 \
  --output results/tiny_target/phase12/media/phase12_raw16_cfar8_annotated.mp4
```

## Remaining validation

The result closes the specific Phase 11 bounded-ranking failure and establishes
moving-target confirmation on this sequence. Production calibration still
requires verified-empty RAW recordings, labeled real targets, broader and
sub-grid velocity trials, multi-target crowding, stress scenes, longer
sequences, and live camera throughput/thermal measurements. Unmatched
candidates in the current recording remain burden rather than certified false
alarms.

Phase 13 now instruments this boundary without changing live decisions. See
`docs/phase13_track_quality_discrimination.md` for the first truth-labeled
track-feature comparison and causal diagnostic policy.
