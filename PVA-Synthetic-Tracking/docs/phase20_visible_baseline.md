# Phase 20: full-frame visible-point baseline

This is a development branch alongside the frozen Phase 19 discovery screen. It is not yet the complete PVA + synthetic-tracking system or a production detector.

Newest: [profile-guided execution cleanup and rejected CUDA warp](/Users/romanmaksymiuk/Documents/SEAQR/skymove/results/tiny_target/phase20/efficiency_v3_20260914/README.md). Exact indexed shape searches and cached NumPy contraction planning improve the same actual-PVA prefix from 1.041 to 1.110 fps (+6.6%). All 230 output frames match and all 255 tests pass. The ordinary CUDA cubic warp is faster but fails exact pixel equality and remains unused. Accuracy-policy defaults and the inherited miss/clutter limitations are unchanged; exact cubic GPU stabilization is the next larger opportunity.

Newest: [resident GPU core and exact-output verification](/Users/romanmaksymiuk/Documents/SEAQR/skymove/results/tiny_target/phase20/efficiency_v2_20260914/README.md). Optional GPU-resident residual/noise state and peak extraction improve the same 230-frame actual-PVA prefix from 0.726 to 1.041 fps (+43.4%), with zero candidate/track/coverage differences. Detector mean time falls 759→344 ms; all 252 tests pass. The inherited frame216 miss and clutter-response trade-off remain unresolved. Defaults remain unchanged; this is neither real-time nor general airborne validation.

Newest: [first GPU/memory speedup and accuracy trade-off](/Users/romanmaksymiuk/Documents/SEAQR/skymove/results/tiny_target/phase20/efficiency_v1_20260914/README.md). Optional raw-response consistency with measured-shape learning protection recovers four126 frames while preserving the stricter PVA pilot, but increases0082 light-field response persistence and remains off by default. Optional float32 CUDA median and exact in-place state updates improve actual230-frame PVA throughput0.611→0.726fps with zero candidate/track/coverage differences.247 tests pass. These are separate accuracy and execution experiments; no production promotion or real-time/airborne validation is claimed.

Newest: [tracking maturity and final 1–4 validation](/Users/romanmaksymiuk/Documents/SEAQR/skymove/results/tiny_target/phase20/maturity_validation_v5_20260914/README.md). The opt-in soft history prior fixes029's ID handoff with132/132 reviewed frames on one ID. All frozen dense/pilot/anchor regressions and236 tests pass. Fresh230-frame PVA126 and full CPU0055/0082 checks are complete. Both variance-protection variants fail the stricter PVA144 localization check and remain off; five faint126 misses persist. Fresh PVA decisions match replay, with tiny continuous roundoff differences rather than bitwise equality. No production promotion or airborne/general accuracy claim is made.

Newest V4b: [shape-aware measurements, full CPU tests and approved PVA prefix](/Users/romanmaksymiuk/Documents/SEAQR/skymove/results/tiny_target/phase20/shape_features_v4b_20260914/README.md). The opt-in image-footprint policy improves measured coverage and removes measured duplicates in the reviewed intervals, but one brief CPU029 ID handoff remains. The corrected230-frame PVA126 test improves132→134/139 with no new misses, all134 matches using one ID, and229 actual PVA pairs without errors/resets/fallback. Throughput0.621fps.221 tests pass; exact original peak evidence is preserved on all1,591 CPU/PVA frames. Defaults remain off; full-clip PVA, nuisance and airborne validation remain incomplete.

Newest V3: [isolated variance-learning and joint-assignment experiments](/Users/romanmaksymiuk/Documents/SEAQR/skymove/results/tiny_target/phase20/variance_only_v3_20260914/README.md). Both are opt-in and remain off: variance-only improves the PVA126 interval132→136/139 but adds two later CPU misses; global assignment worsens same-ID continuity.209 tests pass. Next work must address competing peak hypotheses, with no threshold relaxation or video-specific exceptions.

Newest: [extended encounter review and guarded experiments](/Users/romanmaksymiuk/Documents/SEAQR/skymove/results/tiny_target/phase20/encounter_accuracy_v2_20260914/README.md). Reviewed790 source frames and froze285 visible-point references. The denser checks expose noise self-contamination and competing tracks. Two accuracy experiments failed promotion checks and remain off; an exact-output residual-noise memory optimization saves61MB and measured about3% detector-only time on the Jetson. All201 unit tests pass. Accuracy fixes, airborne classification and real-time operation remain unfinished.

Current priority: [airborne-only accuracy before acceleration](phase20_airborne_accuracy_protocol.md). The first [source-reviewed accuracy pilot](/Users/romanmaksymiuk/Documents/SEAQR/skymove/results/tiny_target/phase20/accuracy_baseline_v1_20260914/README.md) retains 28/28 visible CPU samples and 12/12 of the same samples on PVA, plus the original 24-anchor checks. These are moving-feature regressions; airborne class and reliable negative exposure remain unverified. No pipeline settings changed; all 189 unit tests pass.

The completed V4 development results, known limitations, and review videos are in [the Phase 20 result report](../results/tiny_target/phase20/README.md).

The [V6 follow-up](../results/tiny_target/phase20/v6_followup.md) records verified CPU clutter reduction and a failed PVA-prefix tracking test despite successful PVA execution. V6 is not production-ready.

The [V7 tracking-fix report](../results/tiny_target/phase20/v7_tracking_fix.md) records fresh Jetson recovery of all six confident anchors and preservation of all 24 CPU anchors. Faint-tail identity ambiguity, complete hardware/generalization validation, and real-time performance remain unresolved.

Latest: the [complete V8c PVA transfer runs](/Users/romanmaksymiuk/Documents/SEAQR/skymove/results/tiny_target/phase20/v8c_full_transfer_20260913/README.md) processed all of clips0055/0082 with unchanged settings: 681/689 and 683/691 detection-ready frames, only initial warm-up, zero resets/errors or CPU fallback. All 1,378 motion pairs used PVA. The 23-proposal bounded review still shows light/background and overlapping-track workload; precision and recall remain unmeasured. Throughput remains about 0.73 fps.

The preceding [V8 motion-coverage fix](/Users/romanmaksymiuk/Documents/SEAQR/skymove/results/tiny_target/phase20/v8c_motion_fix_20260913/README.md) restored full PVA clip0027 to 679/687 detection-ready frames, with only eight initial warm-up frames and zero resets/errors. The 126 PVA prefix retains all six confident anchors, both complete CPU regressions retain all 24, and eight additional fixed motion pairs pass. The rule is opt-in; detector/tracker settings are unchanged. These are development checks, not general accuracy or production readiness.

Previously, the [frozen V7 transfer evaluation](/Users/romanmaksymiuk/Documents/SEAQR/skymove/results/tiny_target/phase20/v7_frozen_evaluation_20260913/README.md) completed CPU clips 0027/0055/0082 and full Jetson PVA clip0027 without tuning. PVA executed without runtime errors but rejected all 686 motion fits and never left warm-up. Four fixed-settings diagnostic pairs failed spatial-coverage gates despite low fitting error. This was a motion-policy generalization gap, not evidence of an empty scene; the CPU proposals remain unlabeled review workload.

## Contracts

- Native 8-bit AVI input only in this implementation. Uncalibrated high-bit-depth formats fail explicitly. RAW16 retains its separate pipeline; do not reinterpret it as 8-bit.
- Every source frame is decoded, with full source width and height. No object-location/time hints or reference annotations enter the detector. A motion-only proxy may be downsampled; detection pixels are not downsampled.
- Full-frame filters share support across tile quota boundaries. Warp-invalid pixels, filter-edge margins, startup/reset warm-up, and capacity drops are reported explicitly. There is no hard-coded horizon or bottom-image exclusion.
- CPU phase-correlation translation is a development/reference backend. PVA mode reuses the existing PVA correspondences, global-motion fit, and full-resolution warp and preserves the source-to-reference matrix. CPU results are not PVA validation or Jetson speed measurements.
- Per-frame detections have position but no velocity measurement. The existing Kalman manager has a backward-compatible `position_only` measurement mode. Velocity is estimated from position history, not artificially measured as zero. Bright/dark association is separate.
- Track predictions are never scored as detections. Independent observations are required for temporal confirmation; moving-track qualification additionally requires spatial excursion. These provisional gates do not establish physical-object identity or calibrated confidence.
- This branch currently does **not** run synthetic tracking. Independent faint-target search still needs to be integrated after validating this baseline.

## Versioned experiments

`phase20_visible_v1.json` uses a box-filter spatial background. The original run and implementation snapshots are preserved.

`phase20_visible_v2.json` changes only the spatial-background estimator to a 5x5 median. A synthetic bright 3x3 patch exposed negative-ring candidates from mean-background subtraction. The median change fixes that explicit regression without raising global detection thresholds or using target coordinates. It does not eliminate all clutter or establish generalization.

`phase20_visible_v2_pva.json` selects PVA with the same visible-detection settings. The supplied `--motion-config` is read for motion/global-motion/stabilization sections only; video input comes solely from `--source`.

`phase20_visible_v3.json` adds causal per-pixel noise learned from consecutive-frame differences. This exposed a limitation: a slowly changing background can have small frame differences but a large persistent residual against the background model. The small noise estimate then allows too many clutter candidates to consume track capacity.

`phase20_visible_v4.json` instead learns the squared residual against the actual background model. Scoring precedes the current frame's noise update, and the update is clipped to reduce immediate absorption of a newly arriving point. A deterministic background-drift test distinguishes this from V3. This remains a provisional visible-point model: slow/fading targets can still be absorbed, and the independent faint-target branch is still required. V4 uses the same thresholds, quotas, tracking settings, and full-frame coverage for both reviewed recordings.

The experiments are development iterations, not comparisons on an untouched test set. The source coordinates are used only by the post-run scorer/review renderer. No detection configuration contains an event interval or object coordinate.

After explicit approval, `phase20_visible_v4_pva.json` passed a bounded 32-frame Jetson smoke test: 31/31 motion fits accepted, no PVA errors or resets. It does not cover the target interval or establish full-clip performance. A longer V6 prefix is recorded separately. All remote work is isolated under `/tmp/seaqr_phase20_20260913.V31gWd`; the frozen Phase 19 workspace/configuration is unchanged.

## Follow-up: causal motion quality and rejected peak suppression

`phase20_visible_v5.json` combines a quality check with 4-pixel same-polarity peak suppression before association. This experiment is **rejected**: although the target remains detected, the turning encounter fragments across IDs, leaving only 9/12 confident anchors on its dominant track.

`phase20_visible_v6.json` retains the quality check and disables peak suppression. Its moving-track qualification requires at least five raw measured reference positions, using at most the most recent eight, to fit a local quadratic trajectory with RMSE no greater than three pixels. This allows acceleration/turning rather than requiring one constant velocity across the whole encounter. Predictions never add evidence. A subsequent inconsistent measurement can revoke qualification; the track state remains available in the complete journal.

The gate reduces qualified review proposals, not raw detector candidates or active-track capacity usage. Its localization tolerance and quadratic approximation are provisional: erratic, poorly sampled, or fading genuine motion still needs testing. Duplicate tracks and coherent cloud responses remain unresolved. The experimental NMS code retains suppression provenance but is off in V6; suppressing close peaks must not silently merge real neighboring objects.

For tracking-only changes, `scripts/replay_phase20_tracking.py` reuses an immutable completed full-clip candidate journal. It rejects changes to detector/camera-motion settings and records parent hashes. Replay speed is **not** end-to-end speed. `scripts/compare_phase20_replay.py` checks a fresh full-decode run against the replay, including every candidate, track, coordinate mapping, and coverage record, ignoring wall-clock timings only.

The bounded visual review and provisional background controls are in [review notes](../results/tiny_target/phase20/clutter_review_20260913/review_notes.md). Neither the source crops nor their annotations enter the detector or tracker. No dataset-wide false-positive rate is inferred from this selected sample.

## Local run and separate scoring

### V7 tracking capacity and likelihood association

`phase20_visible_v7.json` and `phase20_visible_v7_pva.json` preserve V6 detection, motion, qualification, and capacity settings. They enable two tracking-only changes:

- **Spatially fair births:** use a 256-pixel reference-coordinate grid; admit candidates from the least occupied cells first. Rank strength only within each cell and rotate tie-breaking across frames. At the unchanged 256-track-per-polarity cap, a new candidate may replace an unobserved, never-confirmed tentative track in a more occupied cell. Confirmed tracks, current measurements, and same-frame births are protected. This cannot guarantee admission when protected tracks fill all slots.
- **Gaussian likelihood association:** keep all old gates, but rank eligible pairs by squared Mahalanobis distance plus log determinant of innovation covariance. This accounts for uncertainty volume instead of rewarding diffuse predictions. Assignment remains greedy and one-to-one, not joint multi-hypothesis inference. Closely competing alternatives are logged; these are not calibrated probabilities or proof of identity.

Both changes are opt-in. The shared Kalman manager retains its old policies by default. Candidate rejections and tentative replacements are journaled. Peak suppression remains disabled, and no target coordinates, event times, or hard-coded image-region exclusions enter either policy.

Tracking replay now permits an explicitly requested **completed prefix** with `--allow-prefix`; the parent must have processed exactly its declared frame limit. Prefix results retain `full_clip: false` and cannot pass the full-clip scorer. `scripts/summarize_phase20_pva_prefix.py` distinguishes inherited motion evidence from actual hardware execution. The renderer can use its post-run diagnostics with `--diagnostics`; this does not promote a prefix into a full-clip result.

### V8 opt-in sparse translation support

The visible detector/tracker configurations remain V7. The new motion-only configuration [phase20_motion_v8.json](/Users/romanmaksymiuk/Documents/SEAQR/skymove/configs/evaluation/phase20_motion_v8.json) opts into `global_motion.coverage_policy: translation_consensus`. All original optical-flow, inlier, error, displacement and stabilization controls remain unchanged. The default `full_grid` policy remains available; similarity transforms cannot opt into this translation-specific check.

When full-grid support is insufficient, the alternate gate requires at least four agreeing cells with at least four consistent feature pairs each, cell-center span of at least a quarter of the image width or height measured only among those agreeing cells, and sufficient point support. Each cell is predicted using the median translation from the other cells, with equal weight per cell. A cell contributes only if its held-out raw median, robust inlier fraction/count and inlier 90th-percentile error pass the existing limits. The existing minimum-inlier fraction is required at both the region and supporting-point levels, allowing local foreground motion to lose its vote without vetoing a strong background consensus. The original point-weighted fit must also agree with the cell-balanced estimate. The check uses all accepted flow pairs, not only the global RANSAC inliers. A densely sampled moving cluster cannot pass solely by dominating the point count.

Only an explicitly identified correspondence spatial-coverage failure may be overridden. Unknown quality failures, insufficient features and all other global-fit failures still reject. Rejected motion still resets the segment and requires fresh background warm-up; no identity/zero-motion fallback or previous-transform reuse was added.

This remains a provisional model-specific criterion. Agreement among visible features does not prove that those features are stationary background or that motion is correct in unobserved image regions. Coherent foreground-dominated scenes and motion beyond the translation model remain limitations requiring further validation. There are no clip IDs, target coordinates or target time intervals in the gate.

Normal journals now include fit rejection reasons, support diagnostics, actual motion backends and per-frame detection availability. Reports separate execution completion from `available_unlabeled` / `unavailable`. A completed **full-clip** CLI run with zero detection-ready frames writes its report and exits with code 2. A short warm-up-only prefix remains explicitly unavailable but does not claim full-clip failure. The availability counter `no_spatial_support_frames` counts zero searchable-pixel frames, including warm-up; it is not an independent measure of warp-invalid area.

Run the new motion policy explicitly:

```sh
.venv/bin/python -m tiny_target.visible_baseline \
  --source /absolute/path/source.avi \
  --config configs/evaluation/phase20_visible_v7_pva.json \
  --motion-config configs/evaluation/phase20_motion_v8.json \
  --output results/tiny_target/phase20/new_unique_v8_run
```

The initial V8 regional check and its failed fourth diagnostic pair are preserved in [the first-attempt record](/Users/romanmaksymiuk/Documents/SEAQR/skymove/results/tiny_target/phase20/v8_motion_fix_20260913/README.md). V8b applied robust point-outlier handling but its full-clip run exposed regional vetoes. The current spatial-majority revision is separately frozen under `v8c_motion_fix_20260913`, with [revision notes](/Users/romanmaksymiuk/Documents/SEAQR/skymove/results/tiny_target/phase20/v8c_motion_fix_20260913/revision_notes.md). V7 and both prior attempts remain preserved.

### CPU baseline and separate reference scoring

Run from the `skymove` repository with `.venv/bin/python`:

```sh
.venv/bin/python -m tiny_target.visible_baseline \
  --source /absolute/path/chunk_0029.avi \
  --config configs/evaluation/phase20_visible_v7.json \
  --output results/tiny_target/phase20/new_unique_run

.venv/bin/python -m tiny_target.visible_regression \
  --run results/tiny_target/phase20/new_unique_run \
  --annotations results/tiny_target/phase19/chunk0029_visual_review_20260913/visual_annotations.json \
  --output results/tiny_target/phase20/new_unique_run/regression.json
```

Output directories must be new. `launch.json` captures source/config/code hashes and resolved controls. `frames.jsonl` contains candidate coordinates, measured vs predicted track states, camera mapping, coverage/capacity diagnostics, and per-stage timings. `report.json` is written only after successful completion. Partial runs and source-hash mismatches cannot pass full-clip scoring.

For subsequent repeatable, single-worker development checks on both reviewed clips:

```sh
.venv/bin/python scripts/run_phase20_visible_regression.py \
  --source-root /Users/romanmaksymiuk/Documents/SEAQR/outputs/jetson_review_clips_20260913 \
  --config configs/evaluation/phase20_visible_v7.json \
  --output-root results/tiny_target/phase20/new_unique_two_clip_run
```

This runner validates source hashes and starts the detector without reference annotations, then scores in a separate step. Existing complete runs are reused only when code/config/source hashes match. Incomplete or conflicting outputs are preserved and rejected. The initial exploratory runs were concurrent local jobs, so their throughput numbers are not a controlled performance comparison or Jetson benchmark.

The two-clip runner now adds a stricter acceptance check: every confident anchor must match one consistent measured, qualified ID per encounter. It exits unsuccessfully if any of the 24 anchors regress, even when the historical scorer's 80% minimum would pass. Earlier reports keep their original policy and are not rewritten.

The scorer requires the same qualified, measurement-supported automatic track at at least 80% of confident manual anchors, with at least two matching anchors. Matching tolerance is annotation uncertainty plus two pixels. Lower-confidence endpoints are diagnostic only. This is a **sparse-anchor development gate**, not exhaustive recall, continuous identity verification, or a promise of 100% detection.

Unmatched automatic tracks are review workload. False tracks/minute remains `null` until sufficiently reviewed negative intervals/regions are supplied. The three reviewed encounters in clips 029/126 are not untouched evaluation data. No sealed holdout media is part of this work.

## Verification

```sh
.venv/bin/python -m unittest discover -s tests/unit
```

New tests cover full-height detection, quota seams, both polarities, static hot pixels, invalid support, explicit capacity loss, a filter-ring counterexample, flickering pixels and persistent background-model error, intermittent fast motion, a turn, motion inversion/reset, coordinate-segment isolation, source-identity checks, incomplete runs, and prediction-only scoring rejection. Existing position+velocity tracker behavior remains the default.

Before deployment: complete same-config full-clip checks, quantify/reduce clutter using reviewed negatives, verify PVA behavior and reset-related loss on the Jetson, integrate independent faint-target discovery, and benchmark memory/latency on the target hardware. Frozen Phase 19 reports/configuration are not overwritten.
