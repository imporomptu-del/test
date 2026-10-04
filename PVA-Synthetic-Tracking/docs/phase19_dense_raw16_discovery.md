# Phase 19: dense native-resolution RAW16 discovery

## Goal and evidence boundary

Phase 19 searches every frame of the 80-clip Phase 18 discovery split for a
compact source moving independently of the stabilized scene. It is a discovery
funnel, not a real-target sensitivity or false-alarm measurement. The 20-clip
holdout remains sealed, and every output from an unlabeled recording remains
review workload until a human examines the source pixels.

The full Phase 15 pipeline is too expensive for a complete collection scan:
its robust full-frame preprocessing and multi-phase matched filter dominate
runtime. Phase 19 retains the parts that materially help discovery—PVA global
motion, native-resolution radiometry, a point-source spatial filter, CUDA
shift-and-stack, local clutter normalization, and motion persistence—while
using a cheaper causal background model.

## Frozen funnel

For each RAW16 frame, the screen:

1. estimates translation with the existing PVA Harris/PyrLK implementation;
2. applies the existing fail-closed global-motion and stabilization policy;
3. crops the upper 4784 x 1920 native pixels without spatial downscaling;
4. subtracts a causal robust EWMA background and whitens by its per-pixel noise
   estimate;
5. correlates one provisional Gaussian point-source template after local
   spatial-background removal;
6. integrates 16 frames with the existing CUDA shift-and-stack backend over 48
   nonzero velocity hypotheses from the -3 to +3 px/s Cartesian grid;
7. applies tile-robust CFAR and a 64 x 64 spatial quota grid;
8. links candidates across windows by predicted position and velocity; and
9. retains a bounded per-clip pool plus an eight-track review shortlist.

Strong temporal outliers update the background at 0.03 rather than being
frozen forever. That lets fixed stabilization residuals decay while a moving
point occupies any one sensor location only briefly. Zero velocity is removed
before CUDA integration because Phase 19 is explicitly searching for motion
relative to the scene.

The executable configuration is
`configs/evaluation/phase19_dense_screen_v1.json`. The implementation is
`tiny_target/dense_screen.py`, and the holdout-safe resumable driver is
`scripts/run_phase19_dense_discovery.py`.

## Calibration and independent validation

The operating point was developed on the first 64 frames of discovery chunk
0020 using four synthetic targets at 500, 1000, 2000, and 4000 DN. The frozen
64 x 64 quota configuration retained the 4000-DN target in all three eligible
synthetic windows. Its linked track had three hits, exact discrete velocity,
0.32 px median position error, and 0.03 px linear-fit RMSE. The weaker targets
remained visible on the dense score surfaces but did not survive unbiased local
candidate competition.

The unchanged configuration was then evaluated on previously used calibration
chunk 0040 with different locations and velocities. It retained the 1000-DN
and 8000-DN targets in five windows each. Both formed six-hit tracks, reached
the top-eight shortlist, selected the exact velocity hypothesis, and had less
than 0.5 px median position error. The 2000-DN and 4000-DN targets were missed
in their local clutter. This non-monotonic result is expected in structured
scene content and forbids a flux-only sensitivity claim.

The paired no-injection chunk 0040 run produced 8,147 intermediate candidates,
310 qualified persistence tracks, and eight bounded review tracks. Adding the
four validation targets produced 8,151 intermediate candidates and 312
qualified tracks: exactly the two recovered trajectories were added. Those
counts characterize screening workload on one scene; they do not make the
remaining 310 tracks false alarms.

Measured 64-frame throughput was 0.78 frames/s for the clean baseline and 0.71
frames/s for the injected validation. A full 80-clip pass is therefore expected
to take roughly 24--30 hours on one Jetson worker. One worker is deliberate:
PVA, stabilization, and CUDA integration share the same memory and accelerator
resources.

## Discovery run

The complete batch was launched in the isolated Jetson workspace
`/tmp/seaqr_phase12_20260905` with:

```bash
python3 scripts/run_phase19_dense_discovery.py \
  --workspace /tmp/seaqr_phase12_20260905 \
  --dataset-root /home/serg/project/camera_reader_sky/srcsky/chunks_raw16_test \
  --split configs/evaluation/phase18_raw16_discovery_split.json \
  --config configs/evaluation/phase19_dense_screen_v1.json \
  --motion-config configs/tiny_target_phase12_cfar_test.yaml \
  --output-dir results/tiny_target/phase19/discovery
```

The driver validates the frozen 80/20 split, refuses any holdout ID, writes one
report per completed clip, records failures without silently skipping them, and
resumes from already valid reports. Its atomic status file is
`results/tiny_target/phase19/discovery/phase19_dense_discovery_status.json`.

After completion, `scripts/summarize_phase19_dense_discovery.py` creates a
machine-readable collection summary and a human-labeling CSV. Its global
follow-up policy takes at most two tracks per clip and 64 total, ranked by
persistence, independent temporal evidence, fit RMSE, and local-CFAR score.
Native source crops and target-aligned accumulations must be reviewed before
calling any item an object.

