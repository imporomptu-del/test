# Opt-in exact GPU temporal background and support (v7)

The new `cuda_temporal_exact_v1` execution option moves temporal background
state, whitening, and spatial-support validity to CUDA. It preserves the CPU
OpenCV point-response filter and all existing detector, motion and tracking
settings. The default remains `indexed_reference`; the previous RAW16 experiment
configuration remains `masked_ufunc`. Nothing is deployed automatically.

The measured results and limitations are recorded in
`results/tiny_target/raw16_background_v7_20260915/README.md` after validation.

## Why the point filter stays on CPU

The installed Jetson OpenCV 4.10 ARM build selects its DFT path for this 9×9
kernel. A direct row-major CUDA convolution is mathematically similar but does
not reproduce its floating-point rounding. The diagnostic probe found 5,146
different pixels among 6,700, with maximum absolute difference
`3.0517578125e-05` and RMS difference `1.4112556023571148e-06`. This is not evidence
that detections necessarily change, but it fails the chosen bit-exact gate.

The direct-filter kernel is therefore diagnostic only, never called by the
pipeline. Its two device buffers are allocated only on an explicit probe call.
No tolerance, threshold, candidate cap, search coverage or target selection was
changed to accept it.

## Implementation and lifetime

- `tiny_target/detection/cuda/raw_background.cu`: resident float32 location and
  variance, uint16 support history, whitening and two-pass integer 9×9 support.
- `tiny_target/raw_background_cuda.py`: synchronous ctypes boundary, geometry
  and dtype validation, explicit reset/close, no automatic CPU fallback.
- `tiny_target/dense_screen.py`: one opt-in dispatch before the unchanged CPU
  implementation. The frame-validity test uses the original image dtype;
  interpolation results stay float32. The unchanged CPU filter consumes the
  downloaded float32 whitened image, then applies the same normalization.
- `configs/evaluation/raw16_background_v7.json`: exactly one field differs from
  `raw16_full_frame_v2.json`: the background execution enum.

The CPU implementation is checked structurally against the archived v6 source:
after removing the new dispatch, its AST must be identical. All other archived
runtime modules/configurations remain hash-identical. The additional GPU module
and current dense-screen integration must match the generated parity report.

CUDA state updates use separate round-to-nearest operations for subtraction,
division, square root, multiplication and addition. In particular, `1-rate` is
rounded from its Python double result separately; subtracting an already-rounded
float32 rate is not equivalent. The build disables FMA contraction, enables
precise division/square root, and matches this Jetson CPU's flush-to-zero mode.
The host checks that mode without modifying the CPU control register and rejects
an unsupported environment before allocating GPU state. ABI 2 rejects the
initial gradual-underflow prototype. See the
[NVIDIA floating-point intrinsics reference](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html).

The warmup decision uses history **before** incrementing; counters saturate at
65,535. Invalid pixels retain state, newly valid pixels initialize without a
normal update, protected outliers use the original slower update, and segment
resets restart background history. The image border remains invalid at four
pixels. Output arrays are owned by each step so later calls cannot overwrite
earlier synthetic-tracking frames.

The initial supported mode is RAW16, radius four, synthetic-tracking screening
without the per-frame event branch, and at most 32 million pixels. Frame shape
cannot change on an active workspace. Nonfinite input and unsupported options
are rejected. A CUDA error poisons the workspace; processing does not silently
fall back to a different implementation. An initialized, closed GPU workspace
cannot resume. Explicit `close()` is used in the validation harness's `finally`
block; the wrapper also has best-effort destructor cleanup.

## Validation and boundaries

1. Build a separate library using `scripts/build_raw16_background_cuda.py` in a
   new output directory. The builder records source, flags and binary hashes.
2. Run `scripts/check_raw16_background_cuda.py --native` with that library. It
   compares 1,128 generated frames across 28 cases, including native 4784×3190
   uint16 and interpolated float32 images, masks, dark/saturated pixels, warmup,
   resets, outliers, saturated history and alternate rates. A 512-frame static
   case drives variance through underflow naturally; another eight-frame case
   seeds adjacent normal/subnormal bit patterns. State, whitening,
   support and retained CPU point responses must match byte for byte.
3. Run the unchanged 48 generated motion controls and the frozen CPU motion
   parity gate. Verify sources/configurations/libraries before any media access.
4. Run CPU/GPU array audits on only the first 64 frames of RAW0029 and RAW0040.
   Hash decoded/stabilized images, whitened images, background state, matched
   responses/masks, motion, integration score/support/velocity arrays, ranking,
   extracted candidates, association state and finalized results. Reject
   incomplete audits even if their remaining rows happen to match.
5. Only after both audits pass, run two fresh-process timing pairs per clip in
   opposite CPU/GPU order, plus the unchanged injected RAW0040 control. Every
   run must match the archived v6 non-timing report, source-frame hashes and all
   63 motion-point identities. Preserve the existing upper-control miss.

Use a single `flock` worker in an isolated Jetson workspace. Do not open a sealed
split, enumerate video folders, test another clip, change clock/power/service
settings, overwrite reports or tune on the holdout set. This experiment accesses
only the two explicitly allowed RAW16 prefixes and their timestamp sidecars.

Audit durations are not speed measurements: downloading and hashing resident
state is intentionally expensive. The separate timing boundary includes source
hashing, progress output, report writes, cold initialization and FFmpeg cleanup.
It measures bounded instrumented wall time, not sustained capture throughput.
The timing schedule is written before the first run and every process exit is
recorded in an append-only journal. Any failure stops the batch.

Exactness is empirical within these checks and frozen configuration, not a proof
for every possible input. These are previously used, unlabeled development
prefixes. Neither unchanged tracks nor injected controls establish real airborne
recall, false-alarm rate, deployment readiness, or camera exposure FPS.
