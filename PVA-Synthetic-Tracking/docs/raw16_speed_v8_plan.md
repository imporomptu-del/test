# RAW16 v8: frozen speed experiments and upper-control diagnosis

Predeclared before new media runs. This is an experiment, not a deployment or
permission to weaken detector gates. The v7 detector/motion configurations,
existing control positions, and default execution paths stay unchanged.

## Work and decision gates

1. Implement a separate synchronous CUDA 9x9 point correlation using the exact
   stored float32 kernel, constant-zero borders and the same float32 normalizer.
   Do not approximate the kernel as separable. Keep all detector parameters fixed.
2. Compare against CPU OpenCV and a float64 direct oracle on generated noise,
   impulses, flat images, borders, masks, weak moving targets, threshold-adjacent
   values, and native geometry. Record bit equality independently from error.
   A diagnostic numerical envelope is 2e-5 * max(1, max(abs(input))) in normalized
   response units. This is a predeclared screening bound, **not an approved
   production-equivalence tolerance**. Report every threshold/peak/decision change,
   including near ties. Nonfinite output or an exceeded envelope stops media work.
3. On only the first 64 frames of development RAW0029/0040, compare responses in
   shadow mode and compare the complete experimental pipeline outputs. Preserve
   all reference and candidate reports, including failed gates. Strict bit-exact
   and decision-equivalence results are separate; never replace a failed exact
   gate with a tolerance and call it exact. Use two fresh-process alternating
   timing pairs per clip only after numeric/processing-integrity checks.
4. Prototype exact batched translation-RANSAC scoring and one-frame CPU intensity
   conversion reuse. Preserve sample order, tie breaks, refinement, coverage,
   rejection rules, VPI per-pair synchronization/cache clearing and status buffers.
   Validate the CPU changes on generated correspondences and full pipeline
   identities. Probe CPU affine and existing GPU cubic stabilization against CPU
   perspective warping using generated data first. Reject a stabilization
   replacement if image or mask values differ; no interpolation-policy waiver.
5. Trace unchanged upper/middle/lower controls on RAW0040: original and injected
   DN, saturation, warp validity, filter support, temporal support, candidates,
   and final matching. Do not move/brighten targets or lower thresholds. Explicitly
   separate unavailable sensor data from an observable target the detector misses.

Only a single worker may use the Jetson at a time, in a new isolated workspace.
Do not enumerate media directories, read the sealed split, access another clip,
reboot, install packages, or change services/clocks/power. No valid artifacts are
overwritten. Timings include the same hashing/report/decode boundaries as v7;
they are bounded instrumented comparisons, not sustained camera throughput.

Real-airborne recall and false alarms remain unvalidated. No new defaults are
enabled automatically, even if development checks pass.
