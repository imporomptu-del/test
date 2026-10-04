# RAW16 v10 — bounded real-time feasibility decision

## Requirements and authority

Provisional engineering target: 10 input frames/s, native 3190×4784 pixels,
original uint16 input information. This is a test target, not a verified live
sensor rate. Report the existing RAW0029/0040 sidecar cadence separately from
their nominal 10 FPS container metadata. No live capture, reboot, clock/power
change, installation, service mutation, default promotion or sealed-data access.
Use a new isolated Jetson workspace and one experiment worker.

No deployment latency requirement has been supplied. Report acquisition span,
window cadence, service time, and confirmation delay separately. Preserve the
16-frame window, stride 8, all 48 nonzero velocities, 12-sample minimum support,
and original candidate thresholds. At 10 FPS, a window advance has an 800 ms
budget for ALL stages; kernel-only timings must never be called pipeline FPS.
Three complete qualifying windows span at least 32 frame positions, or 3.1 s
between first/last timestamps at 10 FPS, before warmup, alignment and processing.

## Stage A: exact persistent tracking, no camera reads

Reuse the unchanged CUDA synthetic-tracking kernels and compilation arithmetic.
Replace per-window allocation/upload with an owned persistent mirrored frame
ring, preserving frame order, relative timestamps, velocity order and tie rules.
Keep outputs on device until explicitly requested. An 8-frame advance uploads
8 new frames, not all 16. Record duplication traffic and memory honestly.
Reject nonfinite inputs, shape/segment/polarity discontinuities, nonmonotonic
timestamps, premature windows, stale downloads and use after failure/close.

Generated tests compare all score bytes, velocity indices, support and masks
against the existing frozen CUDA library. Test weak moving targets, nulls,
clutter, dark polarity, changing masks, gaps, turns, ring wrap and resets. Apply
the existing candidate extractor to both outputs; do not retune thresholds or
interpret null-scene candidates as actual objects. Sensitivity is a generated
regression check, not real-airborne accuracy certification.

Benchmark native full-frame geometry with dense and structured-hole support,
at 10 FPS timestamps. Include warmup and multiple reversed-order repetitions.
Report initialization separately, per-advance upload, resident kernel time,
full-map download and host overhead. Verify every timed output after timing.
The strict quality gate must pass before performance evidence is accepted.

## Stage B: resident RAW16 front-end feasibility probe

If Stage A passes exactness, connect a generated-input, GPU-resident front end:
uint16 upload/conversion, reference-arithmetic translation warp, mask erosion,
frozen temporal background/support, and point filtering feeding the ring. Keep
the full frame and original RAW16 precision. Use known generated transforms;
motion estimation is explicitly excluded, never treated as free in the final
decision. Debug downloads are outside timed calls.

The direct GPU point filter remains a KNOWN NON-BIT-EXACT diagnostic alternative
to the CPU FFT. It may be used ONLY on generated data to bound architecture
performance and measure weak-target regressions; this does not relax the
existing equivalence gate or promote v8. Test and report its discrepancy
separately from the exact residency change. No new real-camera run with this
alternative is allowed by this protocol. Existing v9 native reports are the
read-only full-pipeline baseline.

## Decision

Separate: component correctness, generated sensitivity, resident-core budget,
complete pipeline budget, and deployment readiness. A failed quality gate is
not overridden by speed. If resident tracking alone exceeds the entire window
budget, stop additional full-resolution optimization work for this test and
record that transfers alone cannot solve the problem. Otherwise measure the
front end and remaining budget, and identify unimplemented motion/candidate/
association work explicitly. Never infer whole-pipeline feasibility from a
partial GPU core. Do not purchase/recommend specific hardware from unmeasured
scaling claims. Preserve all failed attempts and valid reports, copy compact
evidence locally and independently verify the final report.
