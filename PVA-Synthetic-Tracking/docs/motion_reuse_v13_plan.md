# v13 — real-input regression of frozen v12 reuse

Only development AVI0029/0126/0055/0082 and the existing 64-frame RAW16
0029/0040 prefixes are permitted. Never read a sealed split or enumerate media.
Use the unchanged v12 adapter, current SHA-locked RAW motion implementation and
the existing exact-v9 RAW runtime. The visible pipeline's prior motion module
predates RAW extensions: require the current reference to match its archived
journal before accepting reuse. Both new arms use the same runtime/configuration.

Run one worker, no production changes, reboot, services, packages or power/clock
changes. Existing runtime directories are read-only dependencies; new harness,
outputs and logs live in a fresh temporary directory. Preserve failed attempts.
Every run records the harness/adapter/runtime/configuration identities and closes
the reuse cache explicitly on its owning thread. Capture complete non-timing
motion correspondence identities and occasional process RSS (not GPU allocation
or proof of bounded lifetime). No thresholds, cadence, coverage or caps change.

Visible gate: both arms on the first 96 frames of AVI0126, matching each other
and the archived prefetch baseline on every non-timing journal field. If passed,
run both arms on all four full clips, alternating arm order between clips.
Compare all motion identities and all journal fields; verify source, config,
accelerator identity and decoder cleanup. Then two reversed-order 128-frame
prefix pairs on 0126 and 0082. Stop that branch on any discrepancy/error; do not
tune or silently relax comparisons. Timing includes the same added small motion
identity instrumentation in both arms; archived FPS is not the new baseline.

RAW gate: candidate on each 64-frame prefix and unchanged injected0040 controls
through the existing exact-v9 runner, retaining all original verification gates.
Require correspondence/fit/candidate/report equality against saved exact-v9
evidence. If passed, two reversed-order reference/reuse pairs per clip. Both arms
keep exact-v9 filter/warp, cached CPU feature conversion and existing tracking;
do not mix in the separate generated resident-tracking experiments. The known
missed upper injected control must remain explicit, not counted as a pass in
accuracy. Prefix timings include hashing/journaling and are not sustained FPS.

Each branch has its own gate; failure in one must not be represented as success
in the other. No automatic default promotion. Bring back compact evidence, verify
locally and report exactness, timing boundary, any failure and remaining limits.
