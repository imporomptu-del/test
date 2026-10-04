# Frozen RAW16 v6 stage profiling

This experiment profiles the runtime left after exact CPU batching in v6. It
does not alter a production module, detection setting, motion policy or default.
See the [completed results](../results/tiny_target/raw16_profile_v6_20260915/README.md)
for measured values and the selected next optimization.

## Safety and exactness contract

- Only the existing RAW0029 and RAW0040 development prefixes, exactly 64 frames
  each, native 4784 × 3190 uint16. No directory scan or sealed-split access.
- One Jetson video/GPU worker under a workspace lock; separate temporary
  workspace; no reboot, deployment, clock or power changes.
- Verify the SHA-locked v6 source archive, every archived source/config/harness
  file, the full runtime-module inventory and the existing CUDA-library hash.
- Run the unchanged v6 CPU/generated-control gates before video. The original
  generated reports resolve an unused configuration path against their old
  workspace, so the same three seeds are regenerated in the new workspace.
  Do not relax the gate or rewrite old evidence to make it pass.
- Match complete non-timing reports, every source-frame pixel hash and all 63
  motion-point identities to SHA-locked v6 reference reports. Preserve numeric
  values exactly; normalize only the already-validated paths/timing fields in
  the existing v6 comparator.
- Outputs use exclusive creation. Preserve the failed pre-media setup log and
  all earlier valid evidence.

## Timing interpretation

`scripts/profile_raw16_v6.py` wraps existing call boundaries; runtime code remains
byte-identical to the tested v6 archive. One root span covers the unchanged v6
validation entry point, including evidence gates, pipeline work, source hashing
and report writes. Preflight archive verification and post-run comparison/profile
serialization are outside that root. The inherited `checks.json` timer retains
its previous, slightly narrower boundary for comparison with v6.

Each child span subtracts its entire inclusive time from its immediate parent's
exclusive time. Therefore **exclusive categories sum to root wall time exactly**
in integer nanoseconds. Inclusive rows are useful for drilling into a category
but must not be added together. Wrapper bookkeeping outside a child span falls
into the enclosing exclusive span; these are instrumented wall measurements.

The source generator is timed around `next()` and closure, never across `yield`.
Consequently consumer processing cannot be counted as decoding. FFmpeg runs
asynchronously in a subprocess, so this category measures frame-delivery wait,
copies/validation and decoder termination—not isolated decoder compute time or
CPU utilization. The termination may make the final `next()` slow.

OpenCV operations inherit their enclosing category. Candidate ranking/extraction,
motion correspondence estimation, global motion fitting, stabilization, temporal
background/filtering, support diagnostics, synthetic integration, association,
finalization and evidence overhead are measured separately. CUDA event fields
are nested within the integration host span; never add them to it.

The main two runs exclude `cProfile`. A separate optional RAW0040 diagnostic run
collects Python/C function call statistics and repeats exactness checks. Its
timing includes that additional instrumentation and is not a throughput result.
The run-to-run timing ratio is not an isolated estimate of instrumentation cost.

`scripts/summarize_raw16_profile_v6.py` is report-only: it rechecks exactness and
timing-tree/group accounting, and rejects failed, incomplete or changed runs.
Its hypothetical 2×/4× stage improvements and zero-cost-stage ceilings use
Amdahl's law; they are not measured speedups or promises.

## Reproduction

Stage the verified v6 runtime and the new profiling harness in an isolated
workspace with the matching CUDA binary and freshly valid generated evidence.
From that workspace, hold a single worker lock and run sequentially:

```sh
python3 scripts/profile_raw16_v6.py --clip 0040 --evidence evidence_current \
  --runtime-archive verified_runtime.tgz --output results/0040_stages
python3 scripts/profile_raw16_v6.py --clip 0029 --evidence evidence_current \
  --runtime-archive verified_runtime.tgz --output results/0029_stages
```

Use another new output directory and `--cprofile` for the diagnostic repeat.
Do not rerun into a completed directory. The result README records archive
hashes, exactness outcomes, test counts and the runtime breakdown.

These remain unlabeled development inputs. Unchanged candidate tracks are not
new verified airborne detections, and profiling provides no new recall/FAR or
camera-exposure-rate evidence.
