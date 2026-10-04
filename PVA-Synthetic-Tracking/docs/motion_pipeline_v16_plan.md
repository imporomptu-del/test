# v16: frozen RAW pipeline integration of v15

Scope: only the existing first 64 native RAW16 frames of 0029 and 0040,
plus the unchanged downstream-injected 0040 control. No holdout access,
configuration tuning, service changes, production defaults or accuracy claims.

Use exact-v9 as a read-only runtime, v12 reuse as reference, and the already
generated-gated v15 preparation as candidate. Verify v15 source/library/config
identities before any media access. Preserve original reports, including truthful
backend/allocation metadata. At comparison boundaries only, use v15's narrow
logical identity normalization; explicitly validate each recorded correspondence's
backend and buffer size. All pixel hashes, timestamps, points, fits, decisions,
candidates and tracks must remain exact against the archived full runs.

One sequential worker: two repeats of both clips, reference then candidate in
repeat 0 and reversed in repeat 1; then candidate injected 0040; then separately
profile reference and candidate 0040. Stop on any gate failure. Never overwrite
an output. Preserve unsuccessful attempts.

Timing: existing validation-run wall time (includes decoding, source hashes,
progress logging and report writing), not clean live capture throughput. Profile
runs are excluded from throughput aggregates; use exclusive host spans and
require zero accounting error. Keep the unchanged 2/3 control recovery visible.
Full-pipeline exactness is not airborne recall/FAR validation. Ten FPS is only a
provisional engineering budget, not a verified live camera requirement.

After all gates pass, copy evidence locally, independently recheck identities,
report measured end-to-end gains and current bottlenecks. Choose the next
full-frame preprocessing experiment from that evidence, without promoting v15
or changing detection policy in this experiment.
