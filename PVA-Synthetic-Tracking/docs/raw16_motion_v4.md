# Full-image Harris output capacity for RAW16 motion

This is an opt-in correction to the failed v3 RAW motion experiment. It is not
a production-default change. Real airborne recall and full-clip robustness are
not established by these development tests.

The capacity-only revision passed the three 64-frame availability experiments,
but follow-up tracing found history-dependent lost-track state. The combined
correctness path is described in [the v5 follow-up](raw16_motion_v5.md).

## Measured cause

V3 retained the VPI Python Harris detector's implicit output allocation. On the
tested Jetson this returned exactly 8,192 corners on dense RAW images. The
returned locations were concentrated in the top of the image; our subsequent
6-by-8 spatial selection could not select corners that had already been omitted.

A one-frame RAW0040 diagnostic holding Harris strength at 1.0 in both arms,
with an explicitly sized array, returned 26,675 corners rather than 8,192.
The pipeline's separate frozen strength remains 0.5. All six horizontal image
regions were represented. The signed U16-to-S16 conversion was exact in that
diagnostic; changing its offset is not part of this correction. Comparing
floating-point Harris calculations was diagnostic only, not a replacement
detector or a new scoring threshold.

Forward/backward tracking error alone did not expose the problem: many points
returned nearly to their starting location yet disagreed with the dominant
motion. The unchanged fit appropriately rejected those poor sets. Increasing
the available spatial support repaired all eight selected RAW0040 diagnostic
pairs without relaxing the fit limits. These pairs were selected from known
development failures, not from a held-out evaluation cohort.

## Exact implementation change

`harris_capacity_policy: complete_grid` explicitly allocates feature and score
arrays to cover the PVA eight-pixel NMS grid. Capacity is
`(ceil(width / 8) + 1) * (ceil(height / 8) + 1)`, including boundary headroom.
For a 2392-by-1595 motion image this is 60,300 entries. If the array is exhausted,
motion becomes unavailable with an explicit error rather than silently treating
truncated features as complete coverage.

This does **not** track 60,300 points. Harris output is still filtered by the
original source masks, saturation policy and border checks, then reduced by the
same grid quotas to at most 1,000 tracked points. The eight-bit/PVA default path
keeps its existing allocation behavior for compatibility.

The public [VPI 3.2 Harris Python API](https://docs.nvidia.com/vpi/3.2/python/build/vpi.Image.harriscorners.html)
supports explicit output arrays. NVIDIA's
[Harris algorithm description](https://docs.nvidia.com/vpi/3.2/algo_harris_corners.html)
describes the grid-based suppression, and the
[C API](https://docs.nvidia.com/vpi/3.2/group__VPI__HarrisCorners.html)
requires the eight-pixel setting on PVA. The observed 8,192-entry truncation is
our measurement on this installation, not a claimed universal hardware limit.

`raw16_motion_v4.json` differs from v3 only in the capacity policy. It preserves
the original v3 RAW-only normalization and CUDA PyrLK, and the historical final
motion acceptance thresholds. The full-native detector, target velocity grid,
candidate budgets and frozen injected-target locations are unchanged.

## Verification and limitations

The generated suite keeps the original 12 cases and adds a separate dense,
native-size fixture family: known subpixel motion, known integer translation
with spatially varying independent noise, and an unrelated-scene rejection case.
Three frozen seeds give 45 cases. Every native positive must also retain motion
points in all 48 selection cells, which catches the old spatial truncation even
when its estimated translation happens to be correct.

The full-pipeline validation wrapper checks the generated results and motion
module hashes before running. It opens only the already-authorized RAW0029 and
RAW0040 sources, for 64 frames each. The separate RAW0040 injection remains
downstream of stabilization and therefore is not a test of target interference
with motion estimation.

Search availability, review workload and target accuracy are separate. A returned
track is not an identified airborne object. The earlier upper synthetic control
was already compromised by source saturation; its location and mask policy are
not changed to improve a pass rate. No sealed holdout clips are used.

Detailed measurements and source snapshots are stored in
`results/tiny_target/raw16_motion_v4_20260915/`.
