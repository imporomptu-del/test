# v25: native translation hypothesis scoring, frozen 8-bit pipeline

Research only. Keep v20 as reference and default. No RAW16 or holdout media,
new clip selection, pixel/frame dropping, changed thresholds, new detector
policy, software installation, clock/power/service changes or reboot.
Use a fresh Jetson /tmp directory and one media process at a time.

## Attribution before candidate measurement

Run four 128-frame diagnostics on the already authorized development prefixes:
0126 reference then staged; 0082 staged then reference. Reuse unchanged v24
ownership/admission/stage gates and v20 algorithms. Record per-thread CPU and
wall intervals for PVA correspondence estimation, global fitting, transform
composition, full warp, detection and tracking. Retain every non-timing journal
and motion check. Collect compact correspondence inputs from the two reference
diagnostics only, for exact replay without decoding video again. Diagnostics
are not uninstrumented speed claims; no GIL causal attribution from wall time.

## Bounded native component

Prototype the complete translation-RANSAC hypothesis scoring loop as one
ctypes.CDLL native call, retaining original deterministic sample generation,
winner tie rules, refinement, acceptance gates, sparse coverage policy, and
transform composition. Similarity models and unsupported execution/numerical
cases retain the original implementation. Source-hash-bind the original
global_motion module and exact replaced block. Strict float64 operation order,
no fast math or contracted multiply/add; do not rewrite residual algebra or
compare squared distances in place of the existing norm threshold.

The native fast path is limited to finite nonnegative float32-representable
pixel coordinates up to 65536, without negative zero; at most 4096 points and
1024 hypotheses. Unsupported values/layouts/sizes use the original scorer.
Do not alter the fallback or widen guards after measuring video performance.
No Python callback or process-global mutable scratch inside the native loop.

Before any candidate media, require generated exact full-fit and primitive
comparisons (including ties, threshold neighbors, odd/even medians, degeneracy,
fallbacks, reset sequences, nonmutation, independent outputs, reentrancy and
GIL-release behavior), plus every captured reference correspondence replay.
Check complete masks, residuals, matrices, scores and all non-timing metadata.
Freeze candidate source/test/build/plan hashes after these gates.

## Candidate schedule and adoption

Two native-staged 128-frame smoke runs, 0126 then0082. Stop on correctness error.
Then three alternating reference/native-staged pairs per prefix; reference
first for repeats0/2 and candidate first for repeat1. Both use identical v24
bookkeeping; the candidate adds only native scoring to the staged execution.
Record CPU-fit call counts/fallbacks without a per-frame full reference replay
inside the timed runs. All frozen v20 output/provenance/lifecycle checks remain.

Require >=20% pooled FPS improvement on each workload, every paired ratio>1,
and no consistent p95 regression in consumer cadence, grayscale-ready latency,
or pre-admission request latency. Consistent means worse pooled and worse in
at least two of three pairs. No changed tolerance or post-hoc policy selection.
Only if this gate passes, run full exact candidate regression on development
AVIs0029/0126/0055/0082. Otherwise reject adoption and preserve v20.

After the clean schedule, run two native-staged diagnostics (one per prefix)
with the same low-frequency host attribution. No new Nsight installation or
trace is required: this experiment's question is the attributed CPU boundary.
Compare diagnostics as explanatory measurements, not clean speed controls.

Archive all sources/builds/receipts/logs and compact fit replays, verify them
locally, independently recompute the acceptance gate, and report partial or
rejected results honestly. A larger coherent GPU-resident front end is outside
this experiment. Development parity is not general airborne accuracy or live
10-FPS validation. Decoder-request timing is not sensor acquisition timing.
