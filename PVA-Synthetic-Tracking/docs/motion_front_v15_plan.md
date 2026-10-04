# v15: direct preparation of the unchanged half-size motion image

Research-only, one worker in a fresh Jetson directory. No defaults, sealed data,
services, clocks, installs, reboots or live capture. Existing runtime directories
are read-only dependencies. First validate generated inputs; only then evaluate
the same 64-frame RAW0029/0040 development prefixes. Keep native detection pixels,
original 0.5 motion scale, feature quotas, Harris/flow/fitting gates and VPI
fresh-per-pair workaround. No new smaller motion proxy or detector calibration.

The installed VPI 3.2 rescale uses LINEAR/CLAMP, with integer pixel-center mapping
(https://docs.nvidia.com/vpi/3.2/algo_rescale.html). For even source dimensions and
exact half scale, test whether output equals the even/even source lattice. If
the installed implementation disagrees, fail the pixel gate; never substitute
a close interpolation or relax the byte-identity requirement.

Candidate: preserve robust RAW affine parameters computed from the SAME full
source fixed-grid samples and validity mask. Apply the SAME float32 affine,
clipping and rounding only to the even/even lattice. For U8, copy that lattice.
Wrap the half-size result in VPI directly. Preserve v12 adjacent-frame immutable
buffer/pyramid reuse, owner-thread rules and invalidation. Odd shapes, different
scales/mappings or malformed declarations fail closed in this experiment.
Detection/stabilization always retain the original source image.

Generated gates: source bit depths 8/9/10/12/14/16; all codes, masks, saturation,
flat images and native geometry; compare every prepared pixel to full mapping
followed by actual VPI CUDA rescale. Require source arrays unchanged. Then known
translations/subpixel motion, independent noise, gain changes, foreground motion,
unsupported rotation, clean/noise/clean recovery, discontinuities and reordered
sequences. Require complete correspondences and fits equal to v12, not just a
similar fitted shift. Execution metadata may differ ONLY in rescale backend and
the count of allocated motion-image bytes, both reported truthfully. These two
fields are the only added comparison exclusions; all other non-timing fields
remain in the equality gate.

If generated checks pass, time complete estimator calls on native generated
sequences in alternating order. Global fitting and fixture generation excluded
from estimator timing. Report startup/cold calls separately from cache hits.

Then run bounded real-input motion-only trials for RAW0029/0040, exactly 64
frames, two reversed-order repeats. Use the frozen source reader/configuration,
hash decoded pixels/timestamps and compare exact point/fit decisions to reference.
Do not run outside those IDs or inspect a split/holdout manifest. Source decoding
and immutable Frame construction are excluded from estimator-only timing, never
called whole-pipeline FPS. Compare reference first against saved v13 identities
after the same two documented metadata exclusions. A discrepancy stops the test.

Any failure is retained. No speed claim before correctness, no automatic default
promotion, no claimed whole-pipeline/accuracy result from a motion-only test.
Do not proceed to a changed full-frame temporal screening detector until this
front-end change has a documented correctness/performance disposition.

## Generated-only correction after attempt 01

The first U8 pixel check disproved the even/even lattice hypothesis; all 16,384
output pixels differed. The failed report and exact original sources are kept.
A generated 16x16 ramp shows the installed CUDA backend instead averaging each
2x2 source block with half-up output rounding. No camera source was read.

Replace the unproven lattice arithmetic with a two-thread, single-pass native
CPU implementation: normalize/clip/round EACH of the four original source
samples with the original float32 arithmetic, then compute their integer average
with half-up rounding. Disable floating-point contraction and fast-math. Require
the same byte identity to installed VPI, including native random/flat cases,
before any estimator or real-input test. This changes only the implementation
hypothesis; no detector, mapping parameters, motion geometry, thresholds or
equality tolerance change. It still avoids the full-size normalized buffer and
its VPI upload/rescale, but does NOT avoid reading the four source pixels.
