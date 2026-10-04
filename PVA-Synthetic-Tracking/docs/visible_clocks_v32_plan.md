# v32 — bounded CPU/GPU clock-policy diagnostic

Approved after v31's rejected numerical-thread cap. This is a diagnostic,
not deployment, a power-mode change or permission to retune detection.

Only development 8-bit prefixes 0126 and 0082, 128 frames, are permitted.
Execute the already frozen v31 child and v29/v26/v28 dependencies without
editing them. No RAW16, holdout media, frame skipping or threshold changes.

Compare the existing automatic policy with CPU/GPU frequency floors equal to
their *currently allowed* maxima: CPU 2,201,600 kHz and GPU 1,300,500,000 Hz.
Only four minimum-frequency files may be written. Maxima, governors, EMC,
power mode, fan controls, CPU affinity, services and drivers remain untouched.
NVIDIA documents the GPU min_freq control in its R36 Clocks guide:
https://docs.nvidia.com/jetson/archives/r36.4.4/DeveloperGuide/SD/Clocks.html
The installed /usr/bin/jetson_clocks also uses CPU/GPU minimum-frequency
controls; do not run its broader CPU/GPU/EMC configuration for this test.

## Frozen scope and order

1. Save and validate all four frequency policies and thermal telemetry. Refuse
   unexpected hardware/policies, pre-existing outputs, changed sources, or
   another experiment. Require the existing completed v31 evidence.
2. Fixed-clock, one-thread combined private-state/output smokes on both prefixes.
3. Three repeats of eight cells per scene: automatic/fixed CPU+GPU, GPU-only/
   combined pipeline, inherited 12/one BLAS thread. This is 48 timing runs plus
   two smokes (6400 frame instances; only 256 unique frames). Hardware order
   alternates by scene and repeat; arm order is reversed/rotated between rounds.
   One child at a time. Preserve every run, including failures and outliers.
4. All runs use the same external 500 ms thermal/clock monitor; none uses Nsight
   or injected timing instrumentation. Report this small monitoring overhead,
   not a claim of completely uninstrumented timing. Use the original internal
   pipeline elapsed time, not process startup or cooldown, as FPS denominator.
5. Audit the existing exact journal checks, v31 loaded thread identities, full
   source/config/library receipts, all latency boundaries and per-stage times.
   Re-audit exported journals independently after completion. Do not auto-run
   full clips, promote defaults or select a winner from partial runs.

Compare clock-policy effects within each identical software/thread cell and
thread effects at each clock policy. Report paired repeats and pooled FPS.
Confirm sampled *achieved* frequencies; fixed bounds alone are not evidence
that thermal/electrical limits did not reduce clocks. CPU/GPU are changed
together, so the experiment does not attribute their individual contributions.
EMC stays dynamic: this is not a completely fixed SoC operating point.

## Safety and credential boundary

An interactive sudo password is required. Never request or store that password
in chat. The user starts the prepared launcher once in a terminal. Only the
small standard-library supervisor/independent restoration watchdog run as root;
video and source verification children run as serg with normal group access.

Fresh readings must match the saved 729600/2201600 kHz schedutil CPU policies
and 306000000/1300500000 Hz nvhost_podgov GPU policy before any write. Save all
values to evidence before starting the independent watchdog and any clock
change. Keep native protection active. Require valid CPU and junction sensors,
record unavailable optional sensors (inactive GPU/CV can return EAGAIN).
Refuse startup at >=65 C; abort on any valid temperature >=75 C or loss of a
mandatory sensor. These are conservative experiment cutoffs, not vendor limits.

Each child has a 180 s limit; the batch has a 55 minute limit. An independent
watchdog restores on controller death, heartbeat loss (60 s), or a 60 minute
hard deadline. It is started before any hardware write and acknowledges readiness.
TERM/INT/HUP cause controlled shutdown/restoration. A disconnected SSH session
may abort, not resume or silently duplicate a trial. Neither watchdog nor
cleanup can promise recovery from kernel/device lockup or power loss.

On every exit: terminate only the owned active process group, restore only the
four saved minimums, reread all saved policies, and save the verified restoration
receipt. A restoration mismatch is a failure, never a success. No reboot,
installation, persistent service or sudoers change. The launcher does not
change settings during preparation or read-only preflight.
