# v24: stage-aware full-resolution GPU release and shared admission

Execution-only, opt-in research prototype. Preserve v20's complete 8-bit
algorithm, configuration, CUDA library, native image coverage, frame order,
timestamps and every non-timing output. RAW16 and sealed holdout media remain
out of scope. No production-default, driver, power, clocks, service or reboot
changes. Use one experiment process at a time in a fresh Jetson /tmp directory.

## Frozen policies and ownership

- `reference`: v20 serial motion/detection/tracking with the original one-frame
  decode prefetch. Add the same admission/timestamp bookkeeping as the candidate.
- `bounded`: v23 worker/warp-buffer ownership, but one shared admission limit of
  two frames including decode in flight, decoded/prepared queues and the main
  lease. This isolates the admission change before adding GPU-stage waiting.
- `staged`: the same shared capacity and v23 ownership, with full-resolution
  warp/Gaussian for frame n released only after detector n-1 returns successfully.
  Frame0 bootstraps without a prior detector. Original synchronization stays.
  CPU/PVA/global motion preparation can run ahead; its existing small VPI proxy
  CUDA operations remain part of that preparation, not subject to the full-warp
  gate. No GPU stream, kernel, arithmetic, threshold or detector changes.

One thread constructs, uses and closes all motion/VPI state and both warp
workspaces. Detection, learning, association, tracks and journals remain ordered
on the main consumer. Slots remain leased through completed journals. Admission
is released at that same completion boundary. Stop wakes both admission and GPU
release waits before joining the original source and preparation workers. Errors
and timeouts fail closed; never free a live worker's native state from another
thread. Preserve failed attempts and all valid reports.

## Gates and schedule

1. Generated ownership/controller/adapter tests on Mac and Jetson, with no media:
   exact fake outputs, real decoder thread, EOF, reset metadata, partial resource
   construction, cancellation at both gates, failing detector, stale handoff,
   startup, strict ordering, bounded capacity and clean owner destruction.
2. Two bounded 128-frame smoke runs (0126,0082), then two staged smoke runs on
   those same prefixes. All archived non-timing journal and motion outputs must
   match. Stop on any correctness/lifecycle failure; do not relax comparisons.
3. Three alternating reference/staged pairs per prefix, reference-first for
   repeats0/2 and staged-first for repeat1. All settings remain frozen. Throughput
   includes normal reporting boundary and conservative staged owner drain/close.
4. Require >=20% pooled FPS gain on EACH workload, all three paired FPS ratios>1,
   and no consistent p95 regression in any of consumer cadence, grayscale-ready
   to completed journal, or pre-admission source-request to completed journal.
   Consistent means worse pooled p95 AND worse in at least two of three pairs.
   No tolerances or post-hoc workload selection. A failed performance gate is
   sufficient to reject adoption and skip the full regression.
5. Only after a passing gate, full exact staged regression on development clips
   0029,0126,0055,0082. No other clips, no holdouts, no new accuracy claim.
6. After all clean prefix checks, capture one staged Nsight trace per prefix to
   check actual stage placement, correlated CUDA execution and scheduled thread
   overlap. Traced FPS is not the performance gate. No installation/permissions
   changes; use the already installed Nsight and frozen v22 NVTX bridge.
7. Archive compact source/results/traces, verify complete output parity locally,
   independently recompute the decision, and report adoption or rejection with
   queue-aware latency, RSS, actual stage durations and trace limitations.

## Timing and interpretation

`request_ns` is taken when the source decoder requests a frame, BEFORE waiting
for admission; `admitted_ns` precedes native decode; `ready_ns` follows grayscale
conversion. Record every processed frame through `consumer_complete_ns` after
its journal. Measuring before the gate prevents reduced decoded buffering from
merely hiding admission waiting. This is still file replay: it does not measure
sensor acquisition, codec-internal backlog, or a true 10-FPS live-source queue.

CPU preparation, waiting at the warp gate, and the full-resolution warp host call
have separate timestamps and NVTX markers. The old motion timing is only proxy
handoff in prepared modes. Host spans overlap; do not sum them as wall latency.
CUDA API waits are not device execution time, and neither CUDA-active intervals
nor thread scheduling establishes whole-device utilization or a GIL diagnosis.

The v23 control/adapter source is reused unchanged and hash-bound. New policy
code is separate; no frozen runtime or previous evidence is overwritten.
