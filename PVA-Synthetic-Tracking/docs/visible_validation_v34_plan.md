# v34 — full-video fixed-clock validation, then separate diagnostic traces

Approved after the completed v33 diagnostic. Do not start hardware/video work
until the independent v33 journal/provenance/transition audit passes. The audit
summary and its verifier source are frozen with this package. Preserve every
v29/v30/v31/v32/v33 source, binary, configuration, runtime and report unchanged.

## Configuration and scope

Use the exact v29 combined GPU-front/tracking implementation and unchanged
detector, PVA motion, image coverage, precision, cadence and confirmation rules.
Keep numerical-thread environment unset and verify the inherited 12-thread
OpenBLAS runtime before and after every child, matching the reported v33
combined_default comparison. Do not select the one-thread arm for its tiny
observed fixed-clock advantage. Verify runtime/library/configuration identities;
OpenCV must finish with the established two-worker policy. The new wrapper only
expands the experiment schedule; the original v29 runner already supports full
combined runs on these four development clips.

Only 8-bit development AVI clips 0029 (687 frames), 0126 (674), 0055 (689),
0082 (691). Validate exact source hashes before decoding via the frozen runner.
Do not access RAW16 or sealed holdouts, capture the camera, change persistent
defaults, skip frames, reduce resolution or tune against these clips.

## Fixed schedule: 16 trials, one worker

1. Same six guarded clock transitions as v33, without video.
2. Two 128-frame combined/default-thread private-state and output smokes on
   0126 and 0082. Any failure prevents full runs.
3. Three complete-video repeats, with orders 0029/0126/0055/0082,
   0082/0055/0126/0029, and 0055/0029/0082/0126. Twelve clean full trials.
4. Only after all clean trials pass: two separate 128-frame diagnostic traces,
   0126 then 0082, using the unchanged v30 profiler/NVTX bridge and existing
   Nsight Systems installation. Trace CUDA/NVTX/OS-runtime/context switches;
   CPU statistical sampling remains off. Never mix instrumented FPS into clean
   full-video timing. These predefined prefix traces can explain that window's
   scheduling, not every hotspot that might appear later in the full clips.

All stages use the same fixed CPU/GPU policy and five-second settle plus
external 500 ms thermal/clock monitoring. This is not completely uninstrumented
execution: the unchanged pipeline records its existing timing boundaries.
Total 8735 frame instances, including 8223 clean full-run instances; only 2741
unique development frames. Repeats are not independent accuracy examples.

## Safety and credentials

Reuse the v33 supervisor's stable-readback and independent restoration design.
Only CPU policy 0/4/8 minimum frequencies and GPU 17000000.gpu minimum may change;
fixed means minimum equals the existing allowed maximum (CPU 2.2016 GHz and GPU
1.3005 GHz). Maxima, governors, memory clocks, power mode, fan, services,
affinity, drivers and electrical/thermal protections remain unchanged. The
shared advisory lock excludes other clock experiments.

Require original CPU 729600/2201600 kHz schedutil and GPU
306000000/1300500000 Hz nvhost_podgov, CPUs 0–11, supported frequencies and no
competing experiment. Save originals and start the independent watchdog before
any write. Three matching readbacks at 50 ms spacing within three seconds;
maximum/governor changes fail immediately. Refuse start at >=65 C, abort at
>=75 C or failed mandatory CPU/junction temperature. Optional unavailable
sensors are recorded. These conservative experiment limits are not vendor
hardware ratings.

Each full/video/profile child gets a 600-second timeout (v33's 180 seconds was
for prefixes). The controller still has a 55-minute overall deadline. The
independent watchdog still has a 60-second heartbeat lease and 60-minute hard
deadline. Stop on the first failed gate or timeout; preserve partial results.
Restore and verify all four saved floors on exit, in both controller and
watchdog. Signal only the owned child process group, including the Nsight
launcher/children when profiling. No reboot, installation, sudoers change or
permanent deployment. Kernel lockup/power loss remains outside these safeguards.

Run in a fresh isolated directory and dedicated tmux session seaqr-v34. User
enters sudo password only in the terminal. Supervisor/watchdog are standard
library only and elevated; dependency checks and all video/profiling children
run unprivileged as serg with a clean frozen environment. Preparation executes
mock-hardware tests and dependency verification, but no video decode or clock
writes. No automatic duplication or overwrite of old evidence.

## Acceptance and reporting

Every child must pass the unchanged complete frame-by-frame candidate/score/
track/coverage and motion-identity comparisons against the existing full
reference, plus exact full aggregate equality. Smokes also compare private
tracker state. Require decoder/admission/GPU-front/motion cleanup, no drops,
complete frame counts and no unexpected fallback. Re-audit exported journals
independently; a child success flag alone is insufficient for final acceptance.

Report clean full FPS by clip and repeat, pooled total frames/total time,
per-frame median/p95/p99/max durations and fraction exceeding 100 ms, stage
costs, sliding-window throughput, memory high-water marks, and time-resolved
temperatures/achieved CPU/GPU frequencies. Do not compare historical full
timings as a fresh controlled speedup or use trace FPS for that claim.

Analyze actual internal admission occupancy (bounded at two), admission wait,
ready-to-completion and request-to-completion. The file reader applies
backpressure: bounded internal queues do NOT prove that a live camera would
keep up or avoid dropping frames. Any 10 FPS arrival/backlog calculation must
be explicitly labeled a replay-derived model, not measured live latency. The
AVI's nominal 10 FPS is not proof of physical acquisition cadence. Capture-to-
alert delay includes sensor/transport/buffering/confirmation work not measured
by these file-replay boundaries. We do not change timestamp-based tracking.

Use the full-run stage results and the separately labeled traces to identify
the next substantial bottleneck. Additional tracing of different intervals,
algorithm changes, default promotion or live-camera integration require a
separate bounded next step; none is silently added here.

Exact reference equality preserves known misses, coasted observations, ID
ambiguities and nuisance/light responses; it is not improved airborne recall
or precision. No new accuracy generalization or sustained production real-time
claim follows from this four-clip development test.
