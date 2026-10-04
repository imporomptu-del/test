# RAW16 exact v9 — arithmetic-preserving execution

Resolve v8's quality failures without changing detector policy, control targets,
scores, tie rules, or sealed data. No default promotion is automatic.

The installed Jetson OpenCV chooses double-precision tiled FFT correlation for
the float32 9×9 point filter (256×256 transforms and 248×248 output tiles at
native geometry). Direct float32 convolution is not that reference algorithm.
Keep the rejected v8 prototype and all reports untouched. Evaluate cached,
independent CPU FFT tiles using the same OpenCV operations/precision to retain
the reference exactly. Any direct GPU-filter alternative remains unaccepted
unless it independently passes full bit identity; no threshold correction or
score quantization can substitute for this gate.

Implement a separate CUDA translation-only cubic stabilizer matching the
reference's inverse-coordinate rounding, 32-phase coefficient table,
floating-point evaluation, constant-zero borders and nearest-neighbor masks.
Use generated probes to establish installed-compiler arithmetic, then freeze
the implementation before media. Reject unsupported transforms/configurations;
do not silently approximate similarity/perspective transforms. Preserve CPU
mask erosion, identity bypass, source pixels and frame metadata.

Generated gates cover small/odd/native shapes, all interpolation phases,
integer/fractional/boundary shifts, masks, saturated/flat/noisy/impulse images,
weak targets, threshold-adjacent values and repeated/reset behavior. Compare
bytes, not only error tolerances. Record unsuccessful prototypes; do not use
their results as acceptance evidence. New execution is thread-confined and
fails closed on errors. CPU FFT worker counts are explicit and bounded.

Only after final complete generated gates pass: run full intermediate audits
and unchanged injected controls on the first 64 native RAW0029/0040 frames.
Compare with archived v8 exact-CPU/v7 evidence and retain every report. Then
perform two reversed-order timing rounds per clip for reference versus accepted
combined execution, sequentially under one isolated-workspace lock. Hashes,
configuration, source identities and output decisions must match across runs.
Timing includes the same bounded instrumentation as v8, not sustained camera
throughput. No sealed manifest or other clip is opened or enumerated.

No reboot, package installation, clock/power/service change or extra GPU worker.
Source transfer/build in a fresh temporary Jetson workspace is part of the
approved experiment. All result paths are exclusive; old artifacts are retained.
Report failures and speed honestly. The upper control's saturation/clutter miss
remains an accuracy limitation; this speed experiment must not "fix" it by
altering its trajectory, flux, saturation cutoff or confirmation rule.
