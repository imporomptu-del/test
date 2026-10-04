# v11: exact tracking-kernel and motion-conversion execution tests

Continue the v10 feasibility work, not a deployment. The user also requested
clarification of the 8-bit result: the completed visible-AVI development path
measured 3.175 aggregate FPS, not the separate RAW16 faint-target algorithm.

This pass has two bounded, generated-data-only changes. No new real media or
sealed manifest reads, threshold/search/coverage changes, production defaults,
reboots, installation, service changes or clock/power changes. One Jetson GPU
worker and a fresh isolated workspace. Preserve failed attempts and artifacts.

1. Tracking: precompute each velocity/frame interpolation stencil once rather
   than its identical offsets/weights at every output pixel. Preserve source
   arithmetic, neighbor accumulation order, frame order, support tests and strict
   greater-than velocity tie behavior. Reuse the existing 240-window generated
   equality suite and 20 geometry/gap/wrap cases. Add near-integer displacement,
   tiny-weight cutoff, velocity ties and sparse/full/empty-mask controls. Accept
   timing evidence only after all score/support/index/mask bytes match.
2. Motion preparation: apply the unchanged robust-U16 affine mapping through a
   65,536-entry per-frame lookup table. Percentiles, float32 operations, rounding,
   clipping and valid-mask policy must stay identical. Preserve the existing
   immutable-frame cache and VPI cache-reset workaround. Compare all uint16
   codes, restricted bit depths, masks, flat/empty support, extremes, noncontiguous
   views and normal inputs. Benchmark full-native generated frames separately
   from end-to-end motion. Run generated estimator point/fit equivalence before
   calling this an eligible motion execution candidate.

Four alternating timing repeats and three measured tracking advances per arm
per scene; prefill and verification outside timing. Compare the frozen v10
resident binary against v11 on the same response maps. Preserve native 4784×3190,
16 frames, stride 8, 48 velocities, support 12, and float32 responses. No direct
GPU point-filter change is included. Generated conversion times include the
unchanged percentile calculation and lookup construction/application.

Do not infer complete-pipeline FPS from component timing or add unrelated
medians to claim measured throughput. Inspect saved motion substage timings to
identify remaining costs without rereading camera files. Freeze and report
source/build identities and copy compact evidence locally. A slower or nonexact
candidate is rejected, not excused by theoretical operation counts. Real-video
regression, production integration and sustained real-time certification remain
separate gates, even if all generated checks pass.
