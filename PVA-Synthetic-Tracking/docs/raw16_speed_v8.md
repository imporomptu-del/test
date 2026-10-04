# RAW16 v8: exact CPU motion and experimental GPU filtering

This work separates execution equivalence from detection accuracy. It does not
change detector settings, public defaults, capture services, or the sealed split.
The results are recorded in `results/tiny_target/raw16_speed_v8_20260915/`.

## Exact CPU motion path

`fit_global_motion` accepts the keyword-only execution option
`translation_batched_exact_v1`. The default remains `reference`; the batched
option explicitly rejects similarity models rather than silently treating them
as translations.

The original sample generator, samples, threshold, deterministic tie breaking,
three refinement iterations, coverage/quality checks and rejection behavior stay
unchanged. Hypotheses are scored in blocks of at most 64. Translation residuals
retain the operation order `(previous + translation) - current`; they are not
rewritten as a difference of displacement vectors. Only medians whose primary
integer inlier count cannot win are omitted. The original fitter constructs the
winning matrix and performs all final refinement.

The experiment's `FeaturePixelsCache` retains one immutable `Frame` identity and
its motion-intensity mapping. It never keys on frame number or timestamps alone.
An adjacent pair can reuse the prepared pixels of the preceding pair's current
frame. A changed object/mapping misses the cache, and leaving the experiment
clears it. This caches **CPU pixel conversion only**: VPI synchronization,
`clear_cache()`, fresh per-pair flow state and forward/backward checks remain.

The cache adapter uses scoped patches in the single-worker research harness;
it is not a thread-safe streaming-service integration. The reusable batched
fitter is an explicit API option, not a newly enabled default.

Reference-source AST checks against the archived v6 implementation ensure that
the default fitter has only the declared dispatch additions. Generated tests
compare full estimates, residual arrays and masks against that archived oracle.
Native checks compare the entire pipeline, including stabilized images,
background state, filtering, synthetic integration, candidates and tracking.

## Experimental point filter

`point_filter_v8.cu` computes the stored float32 9x9 **correlation** kernel without
a separable approximation, using a shared-memory tile, explicit round-to-nearest
FMA accumulation, constant-zero borders and float32 normalization. The Python
wrapper is synchronous, thread-confined, validates shape/dtype/finite values,
owns each returned output and fails closed on CUDA errors. There is no automatic
fallback, default detector dispatch, or enabled production configuration.

The builder records source, compiler command and binary hashes. The library is
separate from both v7 temporal state and synthetic integration, so the proven v7
background/support computation is unchanged. The prototype still downloads the
whitened image and uploads it to the filter; this is not a fully resident fused
pipeline. Reported host-call timings include validation and transfers.

OpenCV's CPU filtering and direct CUDA accumulation have different rounding. Generated
checks therefore report three separate facts:

1. Bit identity, which **fails** for this point-filter prototype.
2. A predeclared numerical screening bound, not a production tolerance.
3. Candidate/track behavior, with ordering/index changes exposed separately
   from physical position/velocity/support changes.

The generated suite includes a float64 direct oracle for smaller arrays, native
geometry, constants, noise, impulses, border cases, holes, 33 threshold-adjacent
amplitudes and 18 synthetic sequence cases. These sequence tests isolate filter
and downstream integration/extraction behavior; they are not new real-airborne
recall or false-alarm evidence. A threshold crossing anywhere in a generated
point-response array is reported even when retained candidates are unchanged.

## Bounded runner and evidence

`run_raw16_speed_v8.py` has four explicit modes:

| Mode | Background | Motion CPU | Point response |
| --- | --- | --- | --- |
| `reference` | Frozen exact v7 GPU | Frozen reference | CPU OpenCV |
| `cpu` | Frozen exact v7 GPU | Batched scoring + conversion cache | CPU OpenCV |
| `filter` | Frozen exact v7 GPU | Frozen reference | Experimental GPU |
| `combined` | Frozen exact v7 GPU | Batched scoring + conversion cache | Experimental GPU |

Only RAW0029/0040 and their timestamp sidecars, first 64 native frames, are
accepted. Allowlist rejection precedes media access. Frozen archives,
configurations, controls, binaries, runtime inventory and complete generated
PVA checks are verified before decoding. Every output directory is exclusive.
The unchanged injected controls are allowed only on RAW0040, after stabilization.

`--audit` records intermediate identities for the exact CPU path. `--shadow`
computes both CPU/GPU responses during the experimental run and records every
frame's error. `--trace` adds original/injected intensity and support diagnostics;
its separate true-velocity ROI calculation never steers actual detection.
`--profile` cannot be combined with these expensive diagnostics.

Cross-workspace audits normalize only the two known synthetic-library paths,
whose binary hash is separately verified. They do not normalize any numbers or
array hashes. Raw audit-file hashes still differ because those paths differ.
The initial raw comparison correctly exposed this metadata difference; the
dedicated checker handles it explicitly instead of altering saved audit records.
The timing launcher uses that checker; no detector code changed for this fix.

The final timing schedule has two fresh-process rounds per clip, ordered
reference/CPU/combined and then combined/CPU/reference. The worker is sequential
under `flock`. The diagnostic GPU arm remains non-bit-exact while being timed.
An exit code of zero or `diagnostic_run_passed` **does not mean** exactness,
all controls recovered, or production approval. Those results have distinct
fields. The report-only summary rejects partial schedules, mixed provenance,
changed timing outputs and incomplete evidence; failed quality gates remain
false rather than being relabeled.

## Why stabilization was not switched

Both proposed replacements were tested on generated images before any media:
CPU `warpAffine` and existing OpenCV CUDA cubic `warpPerspective`. The affine
path differed at fractional-coordinate rounding boundaries. The CUDA cubic
path also differed substantially on noninteger shifts.

Inspection of the installed OpenCV source explains why matching the name
"cubic" is insufficient: CPU interpolation uses coefficient A=-0.75 and a
discrete interpolation table; CUDA `CubicFilter` uses A=-0.5-style coefficients
at continuous coordinates with weight normalization. The source excerpts and
file hashes are recorded with the generated evidence. Neither replacement was
enabled on RAW media. A future exact GPU stabilizer needs an implementation
matching the reference resampling definition, not just a backend flag change.

## Upper-control interpretation

The original control trajectory and flux are unchanged. Its center has valid
filter support only on frames 8–21 of its 56 active frames. Original center DN
exceeds the frozen saturation cutoff on frames 28–63; the entire original 7x7
patch is above it on frames 39–63. All these patch pixels remain warp-valid, so
this is not a missing image region or a stabilization-border exclusion.

Saturation is not the complete early-window explanation. A true-velocity
diagnostic still finds a score about 13.39 in window 4–19, while the actual
best-motion map at the truth center selects a different velocity. In window
12–27 the true center lacks the required support; a nearby true-velocity peak
has only 12 supported samples and score about 5.27. Subsequent windows have no
supported true-velocity peak within the fixed truth neighborhood. There are
therefore too few such windows for the unchanged three-hit confirmation rule,
and early clutter competition also matters.

Invalid best-map support is stored as zero by the integrator; that zero means
"no retained valid hypothesis," not necessarily zero usable input samples.
The diagnostic does not justify weakening the saturation mask, moving the
control to an easier location, or calling unlabeled tracks confirmed aircraft.
The generic follow-up accuracy problem is short visibility plus clutter/motion
hypothesis competition, evaluated separately from these speed changes.
