# v35 — isolate native camera-motion scoring

Pre-run protocol, 2026-09-24. No results or speedup claim yet.

## Question and single change

Does the frozen v25 native translation-hypothesis scorer materially improve the
current v29 combined **serial** pipeline? Its earlier video timing combined it
with v24 staged scheduling, so that timing does not answer this question.

Reuse the source-bound v25 binary/adapter unchanged. Keep the v29 GPU front,
tracking batch adapter, detector/configuration, source resolution, all frames,
serial scheduling, prefetch/admission capacity, thresholds and original 12-thread
BLAS/OpenCV-2 policies unchanged. No RAW16, holdouts, new clips, installs, rebuilds,
profile instrumentation, tracking changes or automatic promotion.

In both arms a tiny symmetric wrapper counts motion-fit entry calls. Only the
native arm replaces translation scoring; sampling, refinement, transform gates,
reset/composition and tracking remain the reference implementation. Require
fit calls = frames minus one, zero fallback/pass-through, native scoring used in
the native arm and never in reference. Reference quality gates may reject a fit
before its hypothesis loop, so native calls may be fewer than fit calls.

## Prerequisites and correctness

- Completed independent v34 audit and unchanged frozen dependencies.
- Fresh unprivileged rerun against the existing native binary: 47 full-fit cases,
  11 primitive cases, independent buffers/reentrancy/GIL-release checks, and all
  254 previously captured development fits from 0126 and 0082. No video decoding.
- Two 128-frame native smokes (0126, 0082), including exact private tracking state
  and actual learning-input digests, complete non-timing journals, motion identity,
  configuration/source checks, native counters, frame ordering and cleanup.
- Every subsequent video trial runs the unchanged full v29 correctness gates;
  full regressions additionally compare every frame and full aggregates.
- No existing evidence is overwritten. Any execution/correctness error stops the
  experiment and restores hardware settings; it is not an invitation to retune.

The v29 harness hardcodes a disabled-native flag in its receipt. The new wrapper
therefore intercepts only its one final serialization and writes **.v35base.json**
under a new schema with a truthful native flag and provenance. All computational
checks and remaining receipt fields are preserved. No .v29.json is created or
presented as an unchanged v29 execution. Independent post-run auditing must
explicitly understand this new envelope; old v29 evidence remains immutable.

## Frozen schedule and advancement gate

1. Two native correctness smokes, 128 frames each; exclude these from timing.
2. Three paired 128-frame trials per workload (0126 heavy, 0082 light): 12 runs.
   Workload order alternates by repeat; reference/native order alternates by
   repeat and workload. The schedule is serialized in freeze.json before video.
3. Advance only when pooled throughput (384 frames / sum of three run times)
   improves at least **5% on both** workloads and **10% on at least one**, and
   every corresponding repeat pair improves. No consistent p95 regression in
   either consumer cadence or decode-request-to-completion: reject when both
   pooled p95 worsens and at least two of three paired p95 values worsen.
   Percentiles use linear interpolation across all samples, including startup.
   This is a practical screening gate, not a confidence interval or proof.
4. If and only if that gate passes, run one native full regression on each of
   0029 (687 frames), 0126 (674), 0055 (689), 0082 (691). These measure absolute
   full-clip performance and correctness, not a contemporaneous full-clip A/B.
   No other source videos are accessed. A failed speed gate skips all four,
   preserves its report and still performs verified restoration.

Maximum 18 runs / 4,533 frame instances. Passing prefixes is not a production
decision. Require independent full-journal, motion, source/config, numerical,
timing, serial-lifecycle and thermal/restoration auditing before interpreting the
result. Replay parity preserves existing behavior, including existing misses;
it is not new airborne accuracy, recall, precision or generalization evidence.

## Hardware and launch safety

The v34 hardware/thermal/PID-ownership/lease watchdog guard functions are copied
unchanged and AST-tested against frozen v34. Only four CPU/GPU minimum-frequency
floors change temporarily to their existing maximum; maxima/governors/thermal
protection remain untouched. Six guarded no-video transitions must first pass.
Start below 65 C, stop at 75 C; require three stable readbacks within three
seconds, 500 ms external monitoring and five-second settling in every trial.
Controller and independent watchdog each restore and verify original settings.
Keep the shared lock, normal-user video children, 600-second child deadline,
55-minute batch limit and 60-minute watchdog deadline.

Prepare a fresh /tmp/seaqr_visible_native_serial_v35_* workspace as serg. Launch
only via interactive sudo inside the dedicated tmux session; never request or
capture the password in chat. Preparation is **not** a started benchmark.
Do not alter frozen v25/v29/v34 files, resurrect the RAW16 monitor, or restart
unrelated processes. Preserve the local v34 presentation checkpoint.
