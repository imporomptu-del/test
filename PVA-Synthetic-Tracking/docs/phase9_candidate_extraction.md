# Phase 9: Candidate Extraction

## What this stage does

Phase 8 produces a dense score, selected-velocity, temporal-support, and
validity map at every reference pixel. Phase 9 turns those dense maps into a
small, auditable set of detections suitable for temporal confirmation.

The configured threshold is expressed in normalized shift-and-stack SNR units.
It is applied together with an integer minimum temporal-support requirement.
Candidates too close to the image border or an upstream invalid region are
removed before local-maximum extraction. Equal-score plateaus have a stable
owner: the smallest row-major pixel index.

Remaining peaks are sorted by decreasing score and then row-major index.
Non-maximum suppression removes a weaker peak only when it is close in both
reference position and selected velocity. This preserves close physical
targets with materially different motions while collapsing the usual spatial
and neighboring-velocity replicas of one target. A deterministic spatial-bucket
index limits comparisons to nearby retained peaks; it does not change the NMS
decision or ordering.

## Important Phase 8 boundary

The CUDA and reference integrations intentionally retain only the best velocity
at each pixel. They do not retain the full `[velocity, height, width]` score
volume. Candidate extraction therefore finds spatial maxima on the retained
best-over-velocity surface and performs joint NMS using each pixel's selected
velocity. It cannot reconstruct local maxima that existed only in discarded
velocity planes.

This distinction is recorded in every candidate batch as
`per_velocity_local_maxima_available: false`. Exact per-hypothesis candidate
extraction would need to be fused into the CUDA velocity loop before the score
planes are discarded; it should not be claimed from the present output.

## Candidate evidence

Each machine-readable candidate contains:

- discrete reference position and selected discrete velocity;
- normalized SNR score and reconstructed raw temporal sum;
- supporting frame count and unit support weight;
- strongest spatial neighbor, peak contrast, and peak/neighbor ratio;
- distance to the image border and bounded Chebyshev distance to invalid data;
- explicit `null` refined position/velocity fields;
- explicit unavailable direct saturation/hot-pixel flags, plus the upstream
  validity-mask policy that excludes those pixels.

The retained map stores support count, not the identities of the contributing
frames, so this limitation is also explicit in the record. The integration
window's complete frame-index list remains on the enclosing candidate batch.

## Defensive behavior

Pathological noise cannot create unbounded serialized output. Local maxima are
first reduced to a deterministic top-scoring pre-NMS limit, then joint NMS is
applied, followed by the final output cap. Reports retain counts before and
after each step, both truncation flags, and mark post-limit counts as lower
bounds whenever the pre-NMS limit was reached.

The current checked RAW16 configuration uses an 8-SNR diagnostic threshold and
a 256-candidate cap. This is deliberately not called a camera-calibrated
operating point: Phase 8 already showed strongly non-Gaussian real tails. A
production threshold requires background-only false-alarm characterization in
Phase 11.

## Tests and benchmark

The deterministic benchmark runs the real reference shift-and-stack stage,
then candidate extraction, on one-target, two-close-target, crossing-track,
and no-target sequences. At the benchmark-only 7-SNR Gaussian-noise operating
point it recovers every injected track at exact discrete position and velocity,
emits one candidate for one target, emits two for the close and crossing cases,
and emits none for the no-target sequence.

Unit tests separately verify adjacent-velocity duplicate collapse, preservation
of close peaks with distinct velocities, border rejection, invalid-region
margin rejection, low-support rejection, stable equal-score ordering, and both
candidate caps.

Run locally with:

```bash
python3 -m tiny_target.candidate_benchmark \
  --output /tmp/candidate_benchmark.json

python3 -m unittest discover -s tests
```

The checked synthetic report is
`results/tiny_target/phase9/candidate_benchmark.json`. The Jetson rerun is
`results/tiny_target/phase9/candidate_benchmark_jetson_v2.json`. The end-to-end
CLI writes candidate batches into the same versioned JSON report as the prior
pipeline stages.

## Jetson RAW16 characterization

The isolated Jetson workspace rebuilt the CUDA library for SM 8.7 and passed
all 91 tests with warnings treated as errors. The live checkout was not
modified. The 15-pair RAW16 run formed four candidate windows. With the
provisional 8-SNR, four-frame, zero-velocity configuration:

| Window | Pixels over threshold | Spatial maxima | Retained after bounded NMS | Serialized |
|---:|---:|---:|---:|---:|
| 0 | 719,977 | 72,478 | at least 3,780 | 256 |
| 1 | 775,262 | 73,270 | at least 3,751 | 256 |
| 2 | 796,547 | 71,918 | at least 3,730 | 256 |
| 3 | 810,658 | 70,486 | at least 3,704 | 256 |

Every pre-NMS limit and final output cap triggered. “At least” is essential:
NMS only evaluated the deterministic top 4,096 spatial maxima after the first
cap. This proves that the current threshold has an unacceptable real-camera
false-alarm burden. It does not prove that the highest-scoring peaks are
physical targets.

Spatial-bucket NMS reduced median full-resolution candidate extraction from
6,120.29 ms in the initial characterization to 374.09 ms without changing any
candidate or count. This remains a host-side characterization path. Fusing the
threshold and local-maximum pass into CUDA would avoid downloading and scanning
the full dense maps, but belongs after a real threshold and velocity grid are
calibrated.

The checked RAW16 report is
`results/tiny_target/phase9/raw16_cuda_candidates_15pairs_v2.json`.
