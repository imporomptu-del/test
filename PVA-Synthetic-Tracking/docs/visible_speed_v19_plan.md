# v19: exact GPU median selection, 8-bit only

Use v17 as the execution baseline, including its compiled learning-mask helper.
Do not use the v18 noise candidate. RAW16 stays paused and sealed holdouts stay
unopened. Only development AVIs 0029, 0126, 0055 and 0082 are authorized here.
Frozen runtime/configs/binaries are read-only dependencies; build into a fresh
isolated directory. No installations, clocks/power/services, live capture,
resolution, sample/candidate limits, thresholds or tracking changes.

Replace the median5 kernel's complete 25-input odd-even sort with a generated
median-selection DAG: construct an ascending 32-wire odd-even merge network,
symbolically pad the last seven wires with positive infinity, fold those
constants, and retain only ancestors of output wire 12. Emit explicit min/max
operations with no quantization or arithmetic reassociation. Keep the original
sort for any window containing nonfinite/subnormal values or negative zero.
Neighborhoods, replicated borders, launch geometry and all other CUDA sources
remain unchanged, including the existing threshold-first peak selection.

Before media: exhaustively verify the finite comparator network on all 2^25
zero/one patterns using bit-parallel execution and an independent popcount
oracle. Thresholding commutes with min/max, so this establishes the selected
rank for all totally ordered finite values (special float cases use the original
sort). Separately compare compiled GPU output bytes with the frozen GPU and,
for ordinary finite inputs, OpenCV. Include borders, tiny shapes, partial blocks,
signed zero, subnormals, NaNs/infinities, duplicate ranks and native geometry.
Use separately built diagnostic event timers, not instrumented pipeline FPS.

Freeze candidate/source/library identities after generated gates. Run three
alternating 128-frame reference/candidate pairs for each of 0126 and 0082,
followed by complete candidate regressions on all four clips, one worker.
Validate every non-timing journal field, complete motion identities, aggregate
tracks and decoder lifecycle. Permit only the explicit directional CUDA library
path/hash transition in launch/config provenance; do not weaken numerical
comparisons. Retain failed attempts and never overwrite results. Compare actual
whole-pipeline throughput, not just isolated kernel speed; no promotion unless
the measured benefit is consistent and complete equivalence passes.
