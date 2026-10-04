# AOT feature diagnosis — fixed experiment, not a detector change

The user confirmed SEAQR is intended mostly for a stable camera and authorized
diagnosis of the failed external pilot. Treat moving-airborne-camera AOT as a
diagnostic stress case, not a representative primary deployment benchmark.
The known failure is too few Harris features before a camera-motion model can
be fit; camera movement alone has not been established as its cause.

## Scope and predeclared comparisons

Use only the previously approved 300-frame native gray8 AOT input already at
`/tmp/seaqr_aot_pilot_20260927_aeZ0yA/input/pilot_gray8_ffv1_10fps.avi`.
Its SHA-256 is `869c37637b68de5eb2c65a6140caebcea58f01833b653a1f2991fec3b16e4d6f`.
Do not open RAW16, private camera clips, sealed holdouts, or new external media.
Do not modify the completed baseline, production code, thresholds, settings,
clock controls, packages, or services. No target detector/tracker is run here.

Eight adjacent pairs are fixed from uniform coverage of input indices 0–299:
current indices **1, 43, 86, 128, 171, 213, 256, 299**, each paired with its
immediate predecessor. Selection uses neither the airplane's position nor
successful detector outputs. Stream and pixel-verify all 300 input frames for
exact indexing, retaining only the 16 required source images.

For each pair also construct a **stationary counterfactual**: repeat the previous
image as the current image, retaining distinct monotonic frame indices/times.
These eight synthetic pairs are not real stationary-camera recordings and cannot
establish detection accuracy. They isolate temporal scene movement while keeping
the previous-frame Harris input identical.

Five generated controls use native 2448×2048 uint8 images, fixed seed 20260927,
and a shared spatial texture:

- High-contrast texture, identical pair.
- High-contrast texture, known integer translation (4, -2) source pixels.
- Low-contrast version of that texture, identical pair.
- Low-contrast version, the same known translation.
- Flat field, identical pair; lack of observable motion features is expected.

Total: **21 pair calls**, one worker, no scale/gain/threshold sweep. The final
script precisely specifies control construction and is bound before execution.
Use a fresh unchanged estimator for each pair; this is a pair-local diagnostic,
not a reproduction of the completed full clip's temporal cache history.

## Measurement contract

Use the exact frozen v12 reuse estimator source and runtime identities from
the successful execution receipt. Make only a diagnostic in-memory copy with
read-only hooks at asserted unique source anchors and existing synchronization
boundaries. Record original/generated/instrumented code hashes. Do not alter
selection, conversion, Harris, flow, fit settings or output points.

Record native/prepared/resized-U8/S16 shape, dtype, pixel hash, intensity range,
distribution and gradient summaries; compare S16 codes against the resized U8
codes to test default conversion. Retain raw Harris coordinates/scores/counts
before eligibility and spatial selection, including zero outputs. Retain
eligibility reasons, selected features, forward/backward flow survival and
motion-fit diagnostics. Never silently drop failed or featureless cases.

Retain immutable input hashes, unchanged clocks/runtime/configuration checks,
owner-thread resource cleanup and before/after source hashes. Explicitly label
additional host readbacks and fresh pair initialization: no speed claims.
The 30-correspondence and spatial-coverage gates remain unchanged. A generated
textured control failing must be reported, not retuned until it passes.

## Interpretation and stopping rule

Compare real and stationary counterfactual raw features to isolate motion from
single-frame feature supply. Check conversion against pixel values, not assumed
API behavior. Compare high/low contrast controls to separate sensitivity from
general runtime failure. Distinguish raw-Harris absence, selection rejection,
flow loss, and model-fit rejection. Insufficient correspondences cannot be
attributed to an unsuitable translation model because that model was not fit.

If the bounded diagnostic establishes a defect, explain it and propose a generic
fix with controls/regression coverage. If it only narrows the cause, state the
remaining uncertainty and next bounded check. Do not implement a production fix
or weaken motion gates under this diagnosis-only authorization. Do not report
object recall, false-alarm performance, synthetic tracking, or validated stable-
camera performance from these feature probes.

Create new local/Jetson artifacts, retain any failed attempts, and save a compact
result with source links locally. The completed baseline remains unchanged.
