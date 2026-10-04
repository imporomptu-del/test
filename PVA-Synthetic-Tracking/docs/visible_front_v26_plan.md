# v26: connected resident detector front end, serial baseline

Research only. Preserve v20 and all prior experiments. No RAW16 or sealed
holdout media, installs, reboot, clock/power/service changes, frame skipping,
downsampling, changed coverage, thresholds or target-specific tuning. One
Jetson media process at a time, fresh isolated /tmp workspace and outputs.

## Component and boundaries

Start from serial v20 (v17 exact learning helper and v20 geometry helper).
Do not enable v23/v24 staged execution or v25 native motion scoring. Retain
v24 reference-only admission/timestamp bookkeeping for both comparison arms.

Build a new explicit CUDA library that includes the four frozen original
integrated CUDA sources unchanged and adds a versioned detector-front ABI.
Keep warp/Gaussian/median5/peak ordering/state arithmetic unchanged. Replace
the connected CPU support/noise/learning path with:

1. Device support erosion from the warp's retained original mask, composing
   its stabilization erosion with the detector's radius6 square erosion.
   Use separable binary minimum operations, including constant-zero borders.
2. Original median/residual preparation, device sample selection by original
   tile/stride/support, exact median and MAD, original float64 sigma scaling
   and float32 device-stat rounding; then original peak selection.
3. Device eligibility/count reduction and device learning protection from the
   same validated, rounded prior observed shape pixels and disk offsets.
   Run original background/variance finish with that mask.

Retain the original CPU shape consolidator and tracker. Download one eligibility
mask for the unchanged shape ABI, plus compact peaks/counts/tile sigma values
and patches. Keep the original CPU median of the compact float64 tile sigmas
for identical telemetry. The existing warp still returns its CPU validity mask;
this is not a claim that every mask or pipeline operation is GPU-resident.

No full temporal-sample download, CPU tile-noise calculation, statistics upload,
CPU support erosion, full CPU learning-mask construction or learning-mask upload
in the candidate. Small host shape validation/rounding remains reference-exact.
Use one owning thread, generation-checked device frames, explicit synchronous
completion/error handling and no cross-library borrowed pointers. Unsupported
configuration or numerical domain fails closed; no silent approximate fallback.

## Correctness before video

Use original source hashes, original reference GPU binary, and complete original
detector outputs/state as the oracle. Generated tests must cover partial tiles,
empty support, support holes/borders, nonbinary source masks, odd/even sample
counts, ties/zeros, threshold neighbors, resets/warmup, changing support,
out-of-frame/duplicate prior footprint pixels, observed-shape protection,
invalid input, stale/consumed frames, ownership/close and repeated allocation.
Exact comparisons include proposals/order, coverage/telemetry, background,
variance, support/eligibility/learning masks, tile statistics and sigma values.

The bounded noise path permits <=4096 stride-selected locations per tile;
tile<=256 and stride>=1. GPU noise input must be finite; never sort nonfinite
values as if they were ordinary residuals. Preserve float32 median averaging,
float32 absolute deviations, float64 MAD scaling/flooring and final telemetry.
Compile with no implicit FMA, no fast math or flush-to-zero approximation.

Run generated device tests and small/native-resolution diagnostic timings
before freezing candidate media source/config/build/gate hashes. Generated
timings are feasibility evidence, not pipeline speed. No real video is read
for those tests. Stop and diagnose correctness failures before proceeding.

## Frozen video experiment after generated gate

Two candidate 128-frame smokes (0126 then0082), followed by three alternating
reference/candidate pairs per prefix. Reference first repeats0/2, candidate
first repeat1. Use only previously authorized development AVIs0126/0082.
Keep v20 geometry in both arms, original v17 learning in reference, and the
explicit new learning path in candidate; record actual helper/ABI call counts.
Record the directional GPU-library transition and runtime adapter provenance;
do not claim an unchanged GPU binary in inherited receipts.

Require every checked non-timing journal/motion output to match and >=20%
pooled FPS gain on each workload, every paired gain positive, and no consistent
p95 regression in consumer cadence, grayscale-ready age or pre-admission
request age. Consistent means pooled worse and >=2/3 paired ratios worse.
Only if all gates pass, run full candidate regressions on existing development
AVIs0029/0126/0055/0082. Otherwise preserve the candidate as research only.

Copy frozen source/build/gates/logs/journals to local SEAQR, independently audit
hashes, schedule, exactness and speed/latency. Preserve failed/partial artifacts.
Report rejection, GPU pressure, or incomplete work honestly. No general
airborne accuracy, sensor-to-alert latency, production or10-FPS claim.
