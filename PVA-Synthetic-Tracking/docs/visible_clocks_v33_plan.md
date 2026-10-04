# v33 — stable clock transitions, unchanged 50-trial diagnostic

This is a fresh version and fresh run of the v32 diagnostic. Preserve v32's
code, 46 completed trials, failure and restoration evidence unchanged. The
leading explanation is asynchronous CPU-frequency policy readback, not a
detector failure. Fake delayed-update tests demonstrate that mechanism; the
installed device/driver cause has not been conclusively reproduced.

## Change and evidence

Write only the same four CPU/GPU frequency floors as v32. After each request,
require three consecutive matching policy snapshots, spaced 50 ms apart,
within three seconds. A read failure resets stability; every observation and
error is recorded. Maximum/governor changes, thermal failures, lost watchdog
or batch deadline fail immediately. Never silently accept a mismatch. During
settling and video processing, the original immediate policy check stays strict.

Before any video, run three fixed-to-automatic cycles (six transitions) under
the watchdog and thermal checks, with a 30-second overall preflight budget.
Abort on failure. Save preflight and per-trial transition receipts, hashes,
failed phase/trial and traceback. These checks do not alter pipeline timing.

Restoration always resubmits all four saved floors, including when readback
still appears original: an earlier request may be queued. Attempt every floor
even if one fails, then apply the same bounded stable verification. Restore
even if temperature/sensor checks fail. Retain any write/heartbeat/readback
failure as failure. Both controller and independent watchdog must confirm
restoration for the run to be considered successful.

## Frozen benchmark scope

Use the same unmodified v31 runner and v29/v26/v28 dependencies. Only development
8-bit prefixes 0126 and 0082, 128 frames each. No RAW16, sealed holdouts, full
clips, retuning, frame skipping, detection changes, or promotion of defaults.

The schedule is identical to v32: two fixed-clock combined/one-thread output
and private-state smokes, then three repeats of eight cells per scene:
automatic/fixed CPU+GPU, GPU-only/combined pipeline, inherited 12/one BLAS
thread. Same alternating/reversed/rotated order, 50 trials total, one child at
a time, 6400 frame instances and 256 unique frames. Every trial retains the
same five-second settle interval and external 500 ms thermal/clock monitor.
No Nsight or injected timing. FPS uses the unchanged child's internal elapsed
time, not startup, transitions or settling. Preserve failures and outliers.

Re-audit exact journals, private state, loaded thread identities and source/
configuration/library hashes independently after completion. Report per-stage
and latency boundaries, paired repeats and pooled FPS, plus achieved sampled
frequencies. CPU and GPU change together; their contributions are not isolated.
EMC stays dynamic. A short prefix benchmark does not establish sustained
full-video real-time operation or detection accuracy on new objects.

## Safety and execution

Only minimum-frequency controls for CPU policies 0/4/8 and GPU 17000000.gpu:
original CPU 729600/2201600 kHz with schedutil; original GPU
306000000/1300500000 Hz with nvhost_podgov. Fixed mode raises minimum to the
currently allowed maximum (2.2016 GHz CPU, 1.3005 GHz GPU). Do not change maxima,
governors, EMC, power mode, fan, affinity, services, drivers or native protections.

Require original policies, CPUs 0–11, supported frequencies, fresh outputs,
unchanged source/dependency freezes, no competing experiment, and the shared
v32/v33 advisory lock. Save originals before any write. Refuse startup at
>=65 C; abort at >=75 C or missing/unsafe mandatory CPU/junction sensors.
Optional inactive sensors can report EAGAIN and are recorded. These are
conservative experiment cutoffs, not vendor operating limits.

Interactive sudo authentication stays in the tmux terminal. Only this small
standard-library supervisor/watchdog run as root. Video and dependency checks
run as serg, with unchanged thread environment and normal groups. Preparation
performs unit/dependency checks without decoding videos or modifying clocks.
The source freeze includes supervisor, tests, plan and terminal wrapper.

The independent watchdog acknowledges readiness before any hardware write and
restores on controller pipe closure, heartbeat loss (60 seconds), or its hard
60-minute deadline. Child timeout is 180 seconds and controller batch limit is
55 minutes. The terminal launcher runs in a dedicated tmux session, so normal
SSH detachment does not terminate the run. INT/TERM/HUP stop only the owned child
group and restore. Do not duplicate a run or overwrite its evidence. No reboot,
installation, persistent service or sudoers changes. Kernel lockup or power
loss cannot be recovered by software safeguards alone.

## Regression checks

Test deferred policy application in both directions, stable-streak resets,
permanent mismatch/read failures, external changes, partial writes, thermal/
interrupt exits, queued-update cancellation during restoration, heartbeat
failure, exclusive failure evidence, preflight failure and watchdog pipe loss.
These tests use fake hardware and do not claim an on-device transition pass.
