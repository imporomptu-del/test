# Phase 16: real-data readiness and provenance gate

## Outcome

The existing recordings cannot yet support a camera sensitivity or false-track
claim. A scoped repository and Jetson audit found video, capture metadata, and
timestamp sidecars, but no authoritative real point-target annotations, no
human-verified empty sequence, and no measured camera PSF array with calibration
metadata.

This phase therefore does not relabel unmatched detections as false alarms and
does not present Gaussian injections as real targets. It adds an executable,
fail-closed contract so the missing artifacts can be collected once and then
used reproducibly by the pipeline.

## Asset audit

The audit covered the repository, the isolated Jetson workspace
`/tmp/seaqr_phase12_20260905`, the live checkout without modifying it, and the
camera capture tree under `/home/serg/project/camera_reader_sky/srcsky`.

Observed assets include:

- 288 AVI chunks in `srcsky/chunks`;
- 100 RAW16 FFV1 test chunks with per-chunk JSON metadata and timestamp CSVs in
  `srcsky/chunks_raw16_test`;
- capture/session logs and compression-benchmark metadata;
- the provisional 0.8-px Gaussian PSF description already documented by the
  project.

No file matching a measured PSF/kernel array, target/ground-truth annotation,
or empty-scene verification was found in the scoped data and project trees.
The capture JSON describes acquisition and compression, not scene truth. The
word `target` in fields such as `target_fps` is not an object label.

The Phase 11 inventory was also checked. Its only `verified_empty` entry is
deterministically generated Gaussian noise and explicitly says that it is not
valid for camera false-alarm claims. Its RAW16 entry is marked unlabeled, and
its authoritative real-target entry is marked missing.

## Executable contract

`tiny_target.real_data_validation` validates three linked artifacts before any
operational claim is allowed:

1. A measured `.npy` camera PSF, plus metadata that identifies its exact hash,
   camera, operating condition, capture method, reviewer, and review time.
2. A real-target recording, exact timestamp sidecar, and authoritative
   point-source trajectories with frame index, timestamp, centroid, visibility,
   and localization uncertainty.
3. A camera recording whose declared intervals were fully reviewed and asserted
   to contain no point targets, again bound to the exact recording hash and
   timestamp sidecar.

The PSF and both cohorts must share an `operating_condition_id`. This prevents a
kernel measured at a different focus, band, exposure/gain regime, or field
condition from silently being treated as matched. Every available recording and
sidecar must match a declared SHA-256. `--skip-source-hash` is inventory-only and
deliberately leaves the cohort claim-ineligible.

The checked-in manifest records the current state as missing:

```text
configs/evaluation/phase16_real_data_manifest.json
```

Example PSF, real-target, and verified-empty evidence documents are in
`configs/evaluation/templates`. They are templates, not evidence, and contain
placeholders that cannot pass the gate unchanged.

## Running the gate

Audit and write a machine-readable report:

```bash
tiny-target-real-data-validation \
  --manifest configs/evaluation/phase16_real_data_manifest.json \
  --output results/tiny_target/phase16/real_data_readiness.json
```

For CI or a preflight before an expensive Jetson replay, require both claim
cohorts:

```bash
tiny-target-real-data-validation \
  --manifest configs/evaluation/phase16_real_data_manifest.json \
  --require-ready
```

The second command exits with status 2 today. This is the intended result: it
prevents downstream automation from running an invalid experiment. Structural
manifest errors raise an error; absent or inaccessible evidence is reported as
a claim blocker.

## Evaluation frozen before data arrival

Once the gate passes, use the same labeled real-target cohort for two detector
runs:

- measured PSF in the matched filter, which is the intended operational run;
- the current provisional Gaussian kernel, which is the declared PSF-mismatch
  stress run.

Keep the Phase 15 detector, synthetic-tracking grid, tracker, and reservation
admission values otherwise frozen. Compare target opportunity recall,
confirmation recall, localization/velocity error, and time-to-confirm. Run the
frozen operational configuration on the verified-empty intervals to report
candidate and confirmed-track rates per time and field area. Do not tune on the
empty cohort; if tuning is needed, acquire a separate development cohort.

## Current verdict

Phase 16 makes the missing-data boundary executable, but it cannot manufacture
the evidence. The Phase 15 reservation remains disabled by default. Enabling it
requires a passing manifest followed by the predeclared matched/mismatched PSF
real-target comparison and verified-empty false-track measurement.
