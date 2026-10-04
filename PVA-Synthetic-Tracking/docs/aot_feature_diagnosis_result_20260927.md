# AOT feature diagnosis: too few background corners before motion fitting

2026-09-27. The bounded Jetson diagnostic completed all 21 planned pair calls.
No production algorithm or configuration was changed; no target detector was
rerun. The original 300-frame baseline and its scores remain unchanged.

## Conclusion

The immediate failure is **insufficient Harris corners under the frozen
half-resolution feature frontend**. Removing interframe movement did not cure
the shortage. Input conversion preserved pixel values exactly, and the same
PVA/flow/motion-fit path succeeded on a generated strong-texture control.

This does not establish that the entire backend is correct, that camera movement
never matters, or that any particular feature-setting change will fix detection.
It identifies where the current failure occurs and narrows the next experiment.

For the intended mostly stable camera, this matters: an unchanged image can
still supply too few corners to satisfy the current motion gate. Assuming
unmeasurable motion is zero would hide the failure, not diagnose or repair it.

## Experiment and results

The [predeclared plan](aot_feature_diagnostic_20260927.md) selected eight adjacent
pairs uniformly from the already-approved AOT clip, without using target labels:
current input indices 1, 43, 86, 128, 171, 213, 256 and 299. Each was compared with
an identical-image pair made by repeating its previous image. Five generated
controls completed the 21 calls. The 30-correspondence and spatial-coverage gates
were unchanged. Every case used a fresh estimator, not full-clip cache history.

| Input / comparison | Raw Harris corners | Selected / accepted flow points | Motion-fit outcome |
|---|---:|---:|---|
| First seven AOT pairs, real and repeated-stationary | 0 each | Not reached | Unavailable |
| Last AOT pair, real and repeated-stationary | 2 each | 2 / 2 | Insufficient correspondences |
| High-contrast generated static texture | 8,192, at capacity | 870 / 870 | Accepted; exactly zero displacement |
| High-contrast generated translated texture | 8,192, at capacity | 870 / 870 | Accepted; recovered known translation |
| Low-contrast versions, static and translated | 0 each | Not reached | Unavailable |
| Flat static control | 0 | Not reached | Expected unavailable |

All eight real/repeated-stationary comparisons had exactly identical previous
proxy images, S16 images, raw Harris coordinates and raw U32 scores. The final
AOT pair's two points occupy only one of 48 coverage cells (2.08%); the minimum
coverage remains 20%. Their presence does not support a reliable global motion
estimate. A failed fit from two neighboring points cannot determine whether a
translation model would be adequate with sufficient observations.

Both strong controls supplied points in 43/48 cells (89.58%) and 870/870 global
inliers. The translated control expected `(4, -2)` native pixels and recovered
`(4.0000547913, -1.9975907249)`: approximately 0.00241 pixel translation-vector
error. For 720 post-hoc interior points, median/max individual flow error was
0.00463/0.01983 pixel. These are synthetic control measurements, not airplane
tracking errors or speed measurements.

The strong controls reached the legacy 8,192-corner output ceiling. Returned
points were unique, but their proxy y range was only 14–863 of 1,024 rows. Thus
their raw output is capacity-limited, not an exhaustive full-image corner census.
The controls were reported as generated; they were not adjusted after seeing
this result. Spatial coverage and fits above refer only to returned points.

## Pixel conversion and scene interpretation

All 21 resized-U8 to S16 comparisons were exact: zero unequal pixels and zero
maximum numerical error. All 300 decoded AOT frames matched the validated input
pixel hashes. No bit shift, clipping, decoding corruption or fallback-backend
defect was demonstrated.

The AOT images are not globally blank or uniformly low-contrast: the eight native
images span codes 0–255, with standard deviations roughly 57–60. See the
[independent source sanity check](../../outputs/seaqr_aot_pilot_20260927/feature_diagnostic_01/source_sanity.md).
Global intensity range is different from usable local corner structure. The
generated high/low controls share the same spatial pattern and differ in contrast;
their different responses make local response sensitivity a concrete hypothesis.
Scale, intensity mapping and feature policy have not yet been separated.

NVIDIA's default conversion preserves numerical values, and its PVA-capable Harris
sample converts to S16 without specifying gain. Therefore, absence of full-range
S16 expansion is not by itself an API-use bug. Harris strength is a response
threshold, not a confidence percentage. See the official
[conversion reference](https://docs.nvidia.com/vpi/3.2/python/build/vpi.Image.convert.html),
[Harris sample](https://docs.nvidia.com/vpi/3.2/sample_harris_detector.html), and
[Harris API](https://docs.nvidia.com/vpi/3.2/group__VPI__HarrisCorners.html).

## What this explains—and what it does not

The original baseline had 299 motion resets, 300 warmup frames and zero ready
frames. The diagnostic identifies the feature shortage upstream of those resets;
it does not turn the original zero detections into a passing result. A visible
airplane can be missed because the pipeline never becomes operational, even
though the airplane itself is clear to a viewer.

Repeated images isolate interframe displacement while preserving the original
single-frame appearance. They do not undo any exposure blur, demonstrate a real
stationary-camera scene, or test jitter, vibration, rotation or parallax. Camera
movement can still affect image formation and later tracking. Those effects were
not established as the immediate cause here.

One daytime AOT encounter, eight sampled pairs and one generated texture seed do
not validate nighttime or stable-camera deployment, object recall, false-alarm
rate, general backend correctness, full-clip cache behavior or processing speed.
No gain, scale or threshold ablation was run. No particular fix is proven.

## Recommended next step—not executed

Predeclare a small feature-only comparison that separates response amplitude from
half-resolution scaling, using the same fixed images and generated controls. Use
a separately declared sufficient-capacity diagnostic arm to expose raw feature
coverage without silently replacing this completed experiment. Keep the existing
30-point, coverage and motion-residual quality gates; more corners alone is not
success. Require correct known motion and well-distributed, reliable matches.

Only then select a generic frontend change and check it on representative
stable-camera positive and negative clips, including controlled jitter and
feature dropouts. Any explicit stationary-camera mode needs its own evidence and
failure handling; do not introduce an automatic zero-motion fallback for unknown
motion. This sequence addresses accuracy before returning to efficiency work.

## Integrity and reproducibility

- Completed 21/21 calls, all resources closed; diagnostic tmux server exited.
- Native input, frozen dependencies, script, tests, plan and clocks checked
  before/after; no hardware clock writes, installs or services changed.
- Only process-local OpenCV thread count followed the frozen policy and was restored.
- Six read-only observation hooks in an in-memory method copy; removing them
  recovers the exact original generated estimator source.
- Independent audit reconstructed 76 raw-array hashes, checked source frame
  identities and exact counterfactual equality, and recomputed geometry errors.
- 187 local diagnostic/baseline/scoring/output/intake tests passed, including
  42 new diagnostic tests. Hardware evidence is in the Jetson result, not inferred
  from these local tests.
- No RAW16, private camera media, sealed holdouts, target labels for selection,
  detector settings, or original baseline reports were changed/accessed by this
  experiment. The diagnostic accessed only approved AOT data and generated controls.

The full [local evidence directory](../../outputs/seaqr_aot_pilot_20260927/feature_diagnostic_01/README.md)
contains the result, manifest, log and frozen provenance. Compact copies are under
[repository evidence](../results/tiny_target/aot_pilot_20260927/feature_diagnostic_01/README.md).

Result SHA-256: `b415844215b32b692fe9d55a80e64674f6bbdadfbd8913c525dba62c5a75c844`.
Manifest SHA-256: `e3d8e07bb125a94ef69af0817a72fa9451d95e87b8110f274de22d5d138410a3`.
Original baseline journal remains `f1b23ea70b758bea4fe5079ae897e7aa50fc338d293daa91fa6c0125b78d1211`.

Remote diagnostic: `/tmp/seaqr_aot_features_20260927_GUg5Vm` on `serg@100.73.41.79`.
Baseline input workspace: `/tmp/seaqr_aot_pilot_20260927_aeZ0yA`.
