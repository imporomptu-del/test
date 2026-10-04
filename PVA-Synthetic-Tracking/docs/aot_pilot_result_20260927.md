# AOT external pilot checkpoint — 2026-09-27

The authorized unchanged v34 PVA/GPU run completed, but the pipeline failed to
become detection-ready on this one labeled daytime encounter. Do not promote it
as a successful airborne detection or speed-validation result.

Local complete artifacts:
`/Users/romanmaksymiuk/Documents/SEAQR/outputs/seaqr_aot_pilot_20260927/jetson_baseline_01/`.
Open its README for the review video, exact evidence and next diagnostic proposal.
Compact evidence is mirrored under `results/tiny_target/aot_pilot_20260927/jetson_baseline_01/`.

300/300 frames were consumed with exact native pixel hashes and no drops. Frozen
implementation/runtime gates passed; actual detection readiness was 0/300.
250 adjacent motion attempts returned zero Harris features, one had no eligible
features, and 48 had only 0–4 correspondences against minimum 30. All 299 failed
pairs reset temporal state, leaving every frame in warmup. There were no candidates
or tracks. This establishes feature starvation, not a PVA device crash or a proven
translation-model failure. That model could not be fit with so few points.

Fixed descriptive frame-annotation matches: 0/195 all labels, 0/29 tiny boxes,
0/93 known-range ≤700 m. Excluding only the initial eight frames: 0/187, 0/27,
0/93. Primary and +3 px gates agree. Unavailable frames remain counted. Only one
encounter and 10.5 seconds of unavailable empty-label context exist; no independent
generalization, official AOT metrics, or false-alarm conclusion is supported.

Next recommendation: a small, predeclared feature-preparation diagnostic with
source/proxy/S16 statistics, Harris output evidence and generated textured/motion
controls. Retain current safety/coverage gates while diagnosing. No subsequent
tuning, new detector run, RAW16 work, holdout access or deployment was performed.
