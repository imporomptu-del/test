# Tiny-Target Pipeline: What We Are Building

## The problem

An approximately one-pixel target has no useful shape or texture. A normal
object detector cannot reliably distinguish it from sensor noise, a hot pixel,
compression artifacts, or a small background highlight in one frame.

The useful evidence is motion consistency: a real target should contribute a
small amount of signal at positions that follow one plausible trajectory over
several frames.

## The two motion problems

The pipeline solves two different motion problems. They must not be confused.

```text
camera/background motion             possible target motion
          |                                  |
PVA Harris + PyrLK                    CUDA trajectory search
          |                                  |
RANSAC camera transform               shift and add weak signal
          |                                  |
stabilized full-resolution frames --> candidate position and velocity
```

PVA does not detect the target. It tracks strong background features so we can
remove camera shake. Synthetic tracking then evaluates faint target motion in
the stabilized coordinate system.

## Why stabilization comes first

Without stabilization, every background edge moves. Subtracting successive
frames then creates millions of structured residuals that can be brighter than
the target.

For each frame pair we will:

1. create a smaller motion-estimation image;
2. find distributed Harris corners with PVA;
3. track those corners with PVA PyrLK;
4. reject failed or inconsistent tracks;
5. fit the least-flexible valid camera transform using RANSAC;
6. compose that transform into the chosen reference coordinate system;
7. warp the original full-resolution frame exactly once.

Only the motion-estimation copy may be reduced in resolution. Target detection
continues to use the original pixel grid and radiometric precision.

## Why timestamps are part of the pixels

Synthetic tracking tests a velocity `(vx, vy)` by sampling frame `i` at:

```text
x_i = x_ref + vx * (t_i - t_ref)
y_i = y_ref + vy * (t_i - t_ref)
```

If a frame was dropped, the next displacement is larger. Treating frames as
equally spaced would sample the wrong location and smear the integrated target.

This is why the Phase 1 `Frame` contract carries timing, sequence, exposure,
gain, bit depth, masks, and discontinuities together with the image. A stage is
not allowed to receive an anonymous NumPy array and guess those facts later.

## What Phase 1 implements

Phase 1 is intentionally a no-op image pipeline. It answers: "Can we read the
same evidence twice and interpret it the same way?"

It provides:

- an immutable grayscale `Frame` contract;
- preservation of `uint8` and `uint16` data;
- exact NPY fixtures for unit tests;
- FFmpeg-based 8/16-bit video decoding;
- explicit timestamp sidecar support;
- honest fallback to container-rate timestamps when no sidecar exists;
- timestamp and sequence discontinuity detection;
- versioned configuration and deterministic configuration hashes;
- per-stage wall-time measurement;
- a machine-readable input inspection report.

The FFmpeg path is deliberate. A convenience OpenCV video read returned the
provided AVI as 8-bit, three-channel BGR. For the FFV1 recordings, explicitly
requesting `gray16le` avoids losing the weak low-order signal before processing.

### Running the Phase 1 inspector

From the Jetson checkout, inspect eight RAW16/FFV1 frames with explicit
sidecar timestamps:

```bash
python3 -m tiny_target \
  --config configs/tiny_target_test.yaml \
  --output /tmp/tiny_target_raw16_inspection.json
```

Inspect the lossy AVI source separately:

```bash
python3 -m tiny_target \
  --config configs/tiny_target_avi_test.yaml \
  --output /tmp/tiny_target_avi_inspection.json
```

The output records resolved configuration, input identity, FFprobe metadata,
decoded dtype and shape, hashes of every inspected frame, discontinuity counts,
and input timing. It also warns when timestamps come from container FPS or
pixels come from Motion JPEG.

The `.yaml` examples use JSON syntax, which is valid YAML. This lets the core
inspector run without PyYAML; conventional YAML syntax is supported when the
optional PyYAML dependency is installed.

## Current data policy

Use the available recordings according to what they can prove:

- `chunks/*.avi`: camera-stabilization development, scene selection, and
  qualitative debugging. Timing and weak-signal sensitivity claims are invalid.
- `chunks_raw16_test/*.mkv` with CSV: current end-to-end reference input. The
  timestamps are host callback times rather than exposure timestamps, but they
  preserve real retained-frame spacing.
- future camera recordings using `ToupcamFrameInfoV3`: desired source for
  hardware sequence, camera timestamp, exposure, gain, and black-level data.

## Current implementation boundary

Phase 2 now turns the proven PVA benchmark into a reusable motion component.
It returns floating-point source/destination coordinates, Harris scores,
forward/backward errors, spatial coverage, rejection accounting, memory bytes,
and submit/synchronization timing. See `docs/phase2_pva_motion.md` for the
hardware path and first RAW16 result. Phase 3 consumes these correspondences
through an independently tested RANSAC transform estimator.

Phase 3 now fits deterministic translation or similarity transforms, rejects
weak or implausible solutions, and composes accepted pair transforms into a
fixed reference for each quality segment. See `docs/phase3_global_motion.md`.
Phase 4 is the first stage that resamples full-resolution image pixels. It
warps each original frame directly into its segment reference, retains
`float32` radiometry, and carries deterministic valid support. See
`docs/phase4_stabilization.md`.

Phase 5 attaches saturation/bad-pixel validity before the geometric warp, then
produces signed background residuals, per-pixel sigma, whitened residuals, and
an explicit detection-valid mask. It includes a temporal median/MAD reference
and a bounded robust running model with candidate-update and global-change
guards. See `docs/phase5_preprocessing.md`.

Phase 6 correlates the whitened residual with a unit-flux, L2-normalized PSF
bank while requiring complete validity over the kernel support. A NumPy golden
reference and OpenCV CPU path agree numerically. The current 2x2 subpixel bank
uses a provisional pixel-integrated Gaussian until measured calibration is
available. See `docs/phase6_matched_filter.md` and `docs/psf_calibration.md`.

Phase 7 now searches timestamp-aware velocity hypotheses with bilinear sampling
and explicit temporal support. It streams velocities and row tiles while
retaining only the best score, velocity index, support, and validity at each
reference pixel. Deterministic serialized maps are the numerical oracle for the
Phase 8 CUDA implementation. See `docs/phase7_synthetic_reference.md`.

Phase 8 now executes the same contract in a dependency-free native CUDA
library. The temporal frame stack stays resident while velocity batches are
searched, timestamp-derived displacements are precomputed on device, and only
the best score/velocity/support maps are retained. CPU and CUDA products agree
within `4.77e-7` on the checked characterization, and the full sensor workload
runs repeatably without constructing a velocity score volume. See
`docs/phase8_cuda_synthetic_tracking.md`.

Phase 9 converts the retained best-over-velocity surface into bounded,
machine-readable candidate batches. It gates normalized SNR and temporal
support, rejects configured border/invalid margins, finds deterministic spatial
maxima, and suppresses duplicates jointly in position and selected velocity.
Every candidate carries score, support, peak-shape, validity-distance, and
quality-policy evidence. The reports explicitly state that discarded velocity
planes are unavailable. See `docs/phase9_candidate_extraction.md`.

Phase 10 adds constant-velocity Kalman prediction/correction and deterministic
multi-target association. Confirmation credits only non-overlapping frame sets,
so overlapping synthetic windows may update a state without artificially
raising confirmation evidence. Tentative, confirmed, coasted, and deleted
states are explicit, as are segment/gap resets and active-track caps. Detector
SNR remains separate from confirmation progress. See
`docs/phase10_temporal_tracking.md`.

Phase 11 wraps the pipeline in versioned dataset and early-injection contracts,
then measures threshold curves, localization/velocity error, temporal
confirmation, false-alarm burden, stage latency, throughput, queues, memory,
and available Jetson telemetry. Correctness, throughput, camera-paced, and soak
modes are explicit. Reports keep controlled Gaussian false alarms separate from
unlabeled RAW16 candidate burden, and the current evidence explicitly declines
real-time and long-soak claims. Early RAW16 injection confirms that strong
target signal survives the image pipeline, but structured background responses
currently outrank it before the bounded candidate output. See
`docs/phase11_end_to_end_evaluation.md`.

Phase 12 addresses the structured-clutter ranking failure exposed by that
evaluation. It preserves raw shift-and-stack SNR as detector evidence while
ranking peaks by tile-local median/MAD standardization, then enforces explicit
spatial quotas so one difficult region cannot monopolize the bounded candidate
output. Raw and selection ranks remain separately observable, and the legacy
raw-ranking path stays the default. On the checked 16-frame Jetson injection,
the new selector recovers all 12 target opportunities with valid integration
support without reaching the final candidate cap. In the 32-frame extension,
the persistent 8,000-DN injection becomes the first truth-matched confirmed
RAW16 track; weaker stationary injections are absorbed after background-model
reset. A corrected 48-frame run over a 7 x 7 velocity grid detects 61/64 moving
target opportunities and confirms all four injected targets, while 67
unmatched confirmed/coasted tracks show that clutter specificity is still the
next limitation. A moving-target threshold sweep identifies local-CFAR 8 as a
held-out validation candidate with unchanged 61/64 target recall and 24% less
unmatched candidate burden than threshold 6, but only 10% fewer unmatched
tracks; track-level discrimination is therefore the next implementation
boundary. Injection coverage is now validated so a timestamp-reference mistake
cannot silently produce a zero-flux evaluation. See
`docs/phase12_clutter_normalization.md`.

Synthetic PSF injection and the CPU shift-and-stack reference now provide the
ground-truth ruler for later acceleration. Every optimized image stage is
evaluated by how much injected target flux and detection SNR it preserves—not
merely by whether its output looks smooth.

Phase 13 adds fixed-memory, non-gating quality evidence to every temporal track
and an offline truth-labeled discriminator analysis. On the checked 48-frame
threshold-8 replay, peak isolation and detector-score stability reduce the
same-run causal unmatched qualification burden from 60 tracks to 8 while all
four injected targets qualify at their existing confirmation time. Kalman
innovation is lower for persistent scene clutter than for injected movers and
is not a useful rejection direction here. The discovered policy remains
diagnostic until it passes independent empty-scene and labeled-real-target
validation. See `docs/phase13_track_quality_discrimination.md`.

Held-out Phase 13 validation rejects that score-stability/peak-isolation rule as
a deployable gate: it retains only 8 of 12 injected tracks across three
successful scenes. The validation also corrects timestamp-gap detection to use
the explicit sidecar's median acquisition cadence rather than the container's
nominal frame rate. One held-out scene remains safely rejected for inadequate
PVA spatial coverage; a second forms accurate tracks under a recorded one-cell
coverage-relaxation experiment. Persistence plus nonzero measured motion is the
next discovery hypothesis, but remains non-gating pending another frozen
held-out test. See `docs/phase13_heldout_validation.md`.

The frozen persistence/motion diagnostic was then evaluated on independently
screened chunk 0040 at the default motion gate. All 63 transforms are accepted,
but CFAR 8 confirms only the 2,000-, 4,000-, and 8,000-DN targets; the 1,000-DN
target is selected in only 3/49 opportunities. Every truth-matched track from
the three confirmed targets satisfies the new policy, while 17 of 542 unmatched
tracks remain qualified at their latest observation. The next calibration
boundary is therefore detector sensitivity in structured clutter, not enabling
track rejection.

A CFAR 6/7/8 sweep confirms that the weak chunk 0040 target is not threshold
limited. Its quota cell is full in every opportunity. Raising per-cell capacity
from 4 to 8 confirms the fourth target but causes 47/52 windows to hit the
global 256-candidate cap and increases unmatched tracks from 542 to 869. That
override remains diagnostic. The next bounded design is track-guided candidate
reservation near causal predictions while retaining the default global quota.

Phase 14 implements that bounded reservation as an opt-in experiment. Before a
new synthetic window is extracted, the Kalman manager can expose a non-mutating
prediction derived only from earlier windows. After ordinary thresholding,
local-maxima selection, joint NMS, and quota selection, the extractor may retain
a small fixed number of otherwise quota-suppressed peaks that pass both a
position and velocity gate around an uncovered tentative-track prediction.
Only zero-miss tentative tracks observed continuously for one full integration
window are eligible; confirmed/coasted and one-off tracks are deliberately
ineligible. Reservations cannot
lower detector thresholds or bypass validity and NMS, and they replace
lowest-ranked baseline outputs if the global candidate cap is full. The
disabled path preserves existing candidates and metrics. See
`docs/phase14_track_guided_candidate_reservation.md`.

On the valid 64-frame chunk 0040 replay, the final continuous-tentative cap-8
policy adds one direct 1,000-DN match at the first independent evidence window
and confirms the previously missed fourth injected target. It retains 68
reservations without reaching either the reservation or global candidate cap.
Unmatched candidate burden rises by 67 and unmatched confirmed/coasted tracks
rise by 47, so the mechanism remains disabled pending transfer validation. The
frozen motion/persistence diagnostic also fails to retain the newly confirmed
weak target and therefore remains non-gating.

Frozen transfer validation on chunk 0060 shows the limit of this mechanism.
The paired baseline and reservation runs both confirm all four targets with
identical 131/176 opportunity matches; none of 51 reservations is truth
matched. Reservation adds 37 unmatched confirmed/coasted tracks without a
target benefit. Across chunks 0040 and 0060 it recovers one additional target
confirmation at the cost of 84 additional unmatched confirmed/coasted tracks.
The feature is therefore a scene-conditional diagnostic, not a default. The
next boundary is an admission discriminator for mature tentative tracks that
retains the rescued weak target before another frozen transfer test.

Phase 15 evaluates prior-only measured motion as that admission discriminator.
Replay of 119 Phase 14 reservation events shows that residual tightening alone
cannot separate the weak mover from persistent stationary structure: 101
events lie within 1 px and 1 px/s. The rescued track has a pre-reservation mean
measured speed of 0.25 px/s, the threshold frozen during Phase 13. The next
predeclared operating point therefore requires that speed, a 3-px position
gate, and a one-grid-bin 1-px/s velocity gate. See
`docs/phase15_reservation_admission.md`.

The frozen Phase 15 rule preserves the chunk 0040 weak-target rescue with only
three reservations and transfers to chunk 0060 with four unmatched
reservations and unchanged target metrics. Across the paired scenes it raises
confirmed injected targets from 7/8 to 8/8 while adding six unmatched
candidates and four unmatched confirmed/coasted tracks; Phase 14's broader
rule added 118 and 84 respectively. This is a successful injected-scene
transfer, but the mechanism remains disabled until labeled real targets and a
verified-empty sequence establish operational sensitivity and false-track
burden.

Phase 16 audits that external-evidence boundary and makes it executable. The
Jetson contains substantial AVI and RAW16 capture collections, but the checked
trees contain no authoritative point-target annotations, no verified-empty
camera intervals, and no measured PSF array with calibration metadata. A new
readiness gate binds the recording, timestamp sidecar, evidence document, and
measured kernel by SHA-256 and requires a shared operating-condition identity.
It fails closed today instead of turning unlabeled detections into false alarms
or treating Gaussian injections as camera sensitivity. Once the missing data
is supplied, the same gate preflights the frozen matched/mismatched-PSF target
test and verified-empty false-track run. See
`docs/phase16_real_data_readiness.md`.

Phase 17 screens three previously unused RAW16 Jetson clips without injection
or per-clip tuning. The full PVA, stabilization, CUDA synthetic-tracking, CFAR,
reservation, and Kalman path runs on chunks 0020 and 0095; chunk 0070 fails
closed because no frame pair satisfies the frozen transform-quality gates.
The usable clips create 14,937 CFAR-8 candidates and 879 unique
confirmed/coasted track identities, of which 11 satisfy the diagnostic
motion/persistence policy at their latest observation. Those counts are
unlabeled review workload, not detections of real objects or measured false
alarms. Source-pixel review crops are visually dominated by structured scene
content, keeping measured-PSF and truth-labeled validation as the operational
boundary. See `docs/phase17_unlabeled_field_screening.md`.

Phase 18 performs a bias-controlled visual discovery screen over the complete
100-clip RAW16 collection. A deterministic split assigns 80 clips to discovery
and seals 20 previously unreferenced clips as a holdout. The discovery clips
are decoded uniformly, sampled every 15 source frames, resized to 600 pixels
wide, and reviewed in three disjoint batches. All 80 clips and 329 contact
sheets were reviewed without finding a plausible compact airborne or
independently moving object, so no native-resolution candidate follow-up was
triggered. The result is sampling-limited: it does not certify empty clips or
measure one-pixel sensitivity. The holdout remains unopened, and controlled
truth-labeled capture remains the operational evidence boundary. See
`docs/phase18_raw16_object_discovery.md`.

Phase 19 closes Phase 18's five-second sampling gap with an every-frame,
native-resolution computational screen over only the 80 discovery clips. A
two-speed causal robust background model and point-source spatial filter feed
16-frame CUDA shift-and-stack windows over 48 nonzero velocities; fine spatial
quotas preserve local candidates before cross-window trajectory persistence
reduces the output. Independent injected validation recovers 1000- and
8000-DN trajectories as top-eight tracks on chunk 0040, while different
2000- and 4000-DN placements are missed, establishing that sensitivity remains
scene/clutter dependent rather than a flux threshold. The full resumable scan
uses one Jetson worker and leaves all 20 holdout clips sealed. Its outputs are
unlabeled review workload, not object or false-alarm counts. See
`docs/phase19_dense_raw16_discovery.md`.
