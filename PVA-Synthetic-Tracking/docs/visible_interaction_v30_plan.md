# v30 — matched GPU-front/tracking interaction diagnosis

Diagnostic phase, frozen before execution. Reuse the complete v29 source and
dependency freeze unchanged. Only 128-frame prefixes of development AVIs 0082
and 0126 are authorized. RAW16 and sealed holdouts are excluded. No production,
threshold, coverage, scheduling-policy, driver, service, power or clock changes.

One video worker at a time. For each mode (traced, then clean), execute light
0082 in order v26/combined/combined/v26, then heavy 0126 v26/combined. These are
12 diagnostic/control runs, not a replacement v29 adoption experiment. Retain
every run. Stop on any output/provenance/lifecycle failure. Fresh outputs only.

Traced runs use existing Nsight CUDA/NVTX/OS-runtime and process-tree scheduling
capture, no CPU instruction sampling or system-wide capture. Stage annotations
measure host elapsed, consumer CPU and whole-process CPU time. Read-only 500 ms
samples capture this process's thread counters, CPU/GPU frequencies and thermal
zones; a separately owned tegrastats child is stopped after each traced run.
Loaded numerical-library worker settings are queried, never changed. No packages
are installed. Profiling includes overhead and is not used for clean speedups.

Harness correction before the complete batch: the first attempt in
`/tmp/seaqr_visible_interaction_v30_qsP2aT` passed video parity and produced a
CUDA trace, but Python's buffered read failed on an unavailable sysfs clock.
That attempt remains preserved, excluded from matched timing/telemetry analysis.
The fresh batch uses bounded raw reads, records individual unavailable sensors,
and performs a read-only telemetry preflight. No video algorithm was changed.

Clean controls invoke the original v29 harness directly, without markers,
telemetry, profiler or private-state serialization. All v29 exact output gates
remain active in both modes. Analyze common-clock interval unions rather than
adding overlapping waits/copies/kernels. Compare both whole and steady windows,
all repetitions, and full frame journals. No assertion of causal contention,
clock throttling or useful GPU utilization from correlation alone.

After examining the diagnostic evidence, freeze a separate explicit candidate
and its tests before measuring it. Any candidate must pass state/output parity,
fresh clean four-arm v29 throughput/latency gates, and only then complete full
development regressions on 0029/0126/0055/0082. A failed gate stops promotion;
do not quietly weaken thresholds, select only favorable runs or choose a backend
by clip ID. No default changes are automatic.
