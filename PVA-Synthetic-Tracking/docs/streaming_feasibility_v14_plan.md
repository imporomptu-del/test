# v14: fail-fast selective-search feasibility experiment

This is an architectural preflight, NOT an implementation or certification of a
new live detector. Freeze this protocol before viewing its results. No defaults,
thresholds, annotations, original reports, camera files, or sealed data change.

## Question and scope

Can a bounded event-triggered temporal search, with independent blind coverage,
plausibly replace dense temporal search without concealing overload or gaps?
Use only saved v13 full AVI journals 0029/0126/0055/0082 and timing records for
the two repeated 64-frame RAW0029/0040 development prefixes. No media decoding,
manifest enumeration, or labels enter the scheduler. No claimed airborne recall
or false-alarm rate: the existing labels do not establish either.

Provisional target remains native 4784x3190 at 10 input FPS. This is not a
verified live-camera requirement. Preserve the dense temporal search's 16-frame
window, 8-frame advance, four-frame initial warmup and 48 velocities. Delayed
windows and dropped work must be reported, never presented as free acceleration.

## A. Implement and falsify the scheduling policy

At each eligible window, spend at most 32 native 256x256 core tiles, including
at least eight oldest-unvisited tiles regardless of events. Tiles partition the
entire sensor, including partial bottom/right tiles. A response-space halo of
eight pixels surrounds a tile; validate its adequacy against timestamps and
velocity bounds. This halo does NOT cover the upstream image filter, which is
assumed full-frame. Candidate-triggered tiles include a fixed 45-pixel seed
radius, persist for 16 frames, and are ordered by recency, score, then tile ID.
Expired requests do not consume memory; a segment change resets all state.
This fixed bound is a hypothesis, not a latency requirement or a tuned setting.

Replay saved full-frame visible proposals as an illustrative seed-workload
proxy. They are NOT RAW proposals or a newly accelerated detector. Measure
requested/serviced/deferred tiles, processed area including repeated halos,
blind revisit gaps, and expose the existing fast branch's costs unchanged.
Replay means no new detection/tracking accuracy result is claimed.

An independent no-seed transient stress test enumerates every core tile and
every onset phase in a complete steady-state blind sweep, using 16/32/64-frame
visibility. A window qualifies only with at least 12 active observations. Count
opportunities for >=1 and >=3 qualifying windows separately. For three-window
counts also give the optimistic case where the first searched window detects
the target and its tile is pinned for the next two advances at no extra cost;
do not mistake lack of fast-detector seeds for permanent lack of temporal feedback.
These are necessary
scheduling opportunities, NOT detections. Losing opportunities is a hard stop
on promotion of this schedule, even if known visible candidates are covered.
Keep a full-coverage control, so intrinsic window/onset limits are not silently
attributed to the selective schedule. Add moving, turning and border test cases.

Simulate single-worker queue delay from archived per-frame service times at
10 FPS; label it modeled, not measured live latency. Include the extremely
optimistic zero-detector/zero-tracker lower bound. RAW motion-estimator time alone
is another optimistic bound. Do not add overlapped decode times to pipeline wall
time. Missing stages are excluded explicitly. No stage-only FPS claim.

## B. Generated Jetson component check

Use one worker in a new temporary directory and the hash-locked existing v11
resident CUDA library. Do not install packages, change power/clocks/services,
reboot, capture video or overwrite existing experiments. Generate deterministic
response-space frames (not RAW sensor images), with native full-frame geometry.
Compare cropped tile-core outputs with full-frame outputs for scores, velocity,
support and validity; retain numerical discrepancies. Include both polarities,
interior/seam/corner/partial-edge tiles, and structured invalid support.

Measure selected-region work INCLUDING allocating/pushing all 16 frames for
new regions, kernel, output download and close. A rotated-in ROI has no cached
history: do not benchmark only eight-frame updates and claim that for new ROIs.
Compare fixed sets of 8 and 32 regions with full-frame work in reversed-order
repeats. Input generation/full-frame ring storage, filtering, motion estimation,
candidate extraction, association and camera decode are explicitly excluded.
Record memory requirement of the source history separately; this is NOT a
streaming end-to-end implementation or a speedup for the whole pipeline.

## Decision

Report implementation tests, scheduling omissions, numerical equivalence,
component timing and complete-pipeline feasibility as separate gates. Failed
coverage or budget is a no-go for this exact policy; it does not prove every
selective or full-frame architecture impossible. Do not run a new camera trial
or expand/tune this schedule after a failure. Preserve evidence and recommend
the next specific architectural change from these results.
