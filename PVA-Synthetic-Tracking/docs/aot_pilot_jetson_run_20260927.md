# Authorized AOT Jetson baseline — 2026-09-27

User approved transferring the validated pilot and running the unchanged PVA/GPU
baseline. This extends the earlier metadata/intake authorization only for this
single 300-frame pilot; it does not open SEAQR holdouts, restart RAW16, tune the
detector, change clocks/power, install packages, or deploy production changes.

Use an isolated `/tmp/seaqr_aot_pilot_20260927_*` workspace on
`serg@100.73.41.79`. Existing frozen runtime directories remain read-only.
Transfer the pixel-exact 700,203,890-byte FFV1 input and compact validation/provenance
artifacts, not the whole dataset. Run unprivileged and serially with existing
numerical settings and a disconnect-safe launcher.

The detector implementation/configuration and v12/v24/v26/v28 combined adapters
must match the v34 baseline. A new external-input harness authorizes only this
exact source hash; it must not patch the old private-camera source allowlists.
Retain PVA adjacent-frame reuse, serial stage ownership, GPU front/median/warp,
native shape extraction, and exact optimized tracking. No CPU fallback.

Before detector execution: check frozen code/library identities; verify all 300
video pixel hashes through the Jetson's reader; freeze independent scoring code,
tests, labels and rules. Preserve full field of view and causal frame order.
Run at nominal 10 Hz input timestamps, as explicitly documented in intake. This
is not proof of exact acquisition-time support. No score-based subset selection.

The expected output is a complete 300-frame journal plus runtime/lifecycle receipt,
independent descriptive scores and source/annotation/observation review evidence.
Do not require old-clip exact-output parity on a new source. Genuine PVA failure,
motion-model rejection, reset and warmup are results to report, not reasons to
silently drop frames or automatically tune settings. Preserve failed attempts.

This is one development encounter with only 10.5 seconds of publisher-empty-label
exposure. Keep candidate detections, actual track measurements, qualified measured
observations and predicted coasts separate. No official AOT benchmark performance,
general false-alarm rate, independent generalization or nighttime accuracy claim.
