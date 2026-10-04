# v23: bounded next-frame motion overlap

This is an isolated execution experiment, not a new algorithm or production
default. RAW16 and sealed holdout media remain out of scope. Only development
AVIs 0029, 0126, 0055 and 0082 are authorized. Prefix runs are 128 frames of
0126 and 0082 only. All pixels, thresholds, timestamps, frame order, detector
updates, learning protection, association, and original CUDA arithmetic stay
unchanged. The v20 candidate is the frozen reference.

## Ownership and stages

1. Unit tests use generated inputs, no media. Construction, VPI reuse, motion
   state, and destruction belong to one preparation worker. Detection/tracking
   and journals remain ordered on the main thread. One motion estimator feeds
   two independent CUDA warp handles; a slot cannot be reused until its entire
   frame, including the journal, has been consumed.
2. `serial` mode exercises the worker and alternating buffers with only one
   outstanding lease. Verify exact outputs on both prefixes before concurrency.
3. `overlap` permits at most two outstanding motion slots, including the main
   current frame: exactly one prepared/in-progress future frame. The original
   one-frame decoder queue remains a separate bounded stage. There is no frame
   skipping, dropping, or unbounded queue. Keep existing CUDA synchronization.
4. Check exact outputs on both prefixes. If these fail, stop and diagnose; no
   threshold relaxation or selective output comparison is permitted.
5. Run three alternating unprofiled reference/overlap pairs per workload. The
   reference has the same lightweight clock bookkeeping. Timing includes EOF
   and drain in reported throughput. Prepared modes conservatively include
   owner cleanup in this boundary. Queue-aware latency is gray decode completion
   to post-journal completion, NOT camera-to-alert latency. Also report consumer
   cadence, preparation duration, queue waits, extra RSS, and host interval
   overlap. Nested or overlapped stage times must not be summed as elapsed time.
6. Keep only if pooled throughput improves by at least 20% on BOTH workloads,
   all paired speedups exceed 1, and there is no consistent p95 regression in
   either consumer cadence or queue-aware latency. A p95 regression is consistent
   when the pooled p95 is worse and at least two of three pairs are worse. No
   tolerance is used to hide a consistent regression. A failed performance gate
   rejects this version early; full regression is not needed to reject it.
7. Only a candidate passing the prefix correctness and performance gates earns
   full exact comparison on all four development clips (all proposals/tracks,
   motion identities, masks/status and non-timing fields, plus full aggregates).
   These clips are regression evidence, not independent accuracy validation.
8. Capture a bounded Nsight diagnostic of the candidate on both prefixes after
   exact prefix checks, even if the clean performance gate rejects it. Check
   worker/main ranges and CUDA activity. Traced timing is not the speed gate.

Run a single experiment process at a time in a fresh Jetson /tmp directory.
Preserve every failed receipt and log; do not overwrite frozen inputs, prior
reports, or valid trials. Do not install software, reboot, change clocks/power,
change GPU streams/synchronization, or inspect any other media. Hash experiment
sources/tests/plan, all dependency identities, and output receipts. Record worker
join/release/close success and zero discarded frames for successful runs. A
shutdown timeout is a permanent failed-close result; never free live worker
resources from another thread.

Known limitations: two slots add a full warp workspace (roughly 275 MB before
allocator effects). Python's GIL, OpenCV/VPI thread behavior, shared memory and
CUDA device-wide synchronizations may serialize work or increase latency.
Thread relocation may affect numerical behavior; exact parity is mandatory.
Useful CPU/PVA/GPU overlap does not by itself demonstrate 10 FPS or real-time
camera latency. A rejected experiment leaves v20 unchanged.
