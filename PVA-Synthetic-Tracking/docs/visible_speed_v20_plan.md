# v20: exact CPU tracking work, visible 8-bit only

Retain the v17 learning-mask helper and original GPU/noise implementations.
Do not enable v18/v19, touch RAW16 recordings or open sealed holdouts. Only
existing development AVIs 0029, 0126, 0055 and 0082 are in scope. Use a fresh
isolated Jetson workspace and one sequential worker; keep frozen dependencies,
power/clocks/services and production defaults unchanged.

First profile tracking and host accelerator call boundaries on 128-frame 0126
and 0082 prefixes. Profiling time is not benchmark FPS and nested times are
not additive. Verify complete archived journal/motion parity for both profiles.
Select a bounded execution-only change from these measurements. Do not change
gates, numerical expressions, association order, evidence, tracking policy,
coverage, resolution, thresholds or quotas. Keep unsupported cases on the
unchanged reference path. Record the exact source transformation and compiled
helper identities in addition to the frozen original launch snapshots.

Before video candidate runs, require adversarial generated primitive tests and
complete generated tracker replay equivalence. Then freeze the candidate and
run three alternating 128-frame reference/candidate pairs per workload,
followed by complete regressions of all four development clips. Compare all
non-timing fields, complete motion identities, configuration/package/accelerator
provenance, aggregate tracks and decoder lifecycle. Do not overwrite valid
reports, tune to an individual target or promote on isolated helper speed alone.
Keep negative/mixed results and prefer the previous baseline unless the measured
pipeline improvement is consistent. Finally copy evidence locally and verify
independently. No new general airborne accuracy or real-time claim is implied.
