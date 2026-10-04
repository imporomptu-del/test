# Two-clip 8-bit detector-led discovery

User-approved task, 2026-09-28: run the existing algorithm on two more videos,
then provide source/overlay views for human review. This is discovery, not a
new airborne benchmark, classifier training, threshold tuning or promotion.

## Fixed scope, selected before outputs

Only these original Jetson files under
`/home/serg/project/camera_reader_sky/srcsky/chunks/`:

| Clip | Bytes | SHA256 |
| --- | ---: | --- |
| chunk_0170.avi | 162613430 | 12848c0f0caedd697a3da51776ab1579bd634a7ae94343f8cbd2a8830ee340bc |
| chunk_0240.avi | 175156036 | 2f86f28785e302572a86e23688143edbd7f5f1f65e8a3434b86a427e79c6a585 |

Metadata: each MJPEG/yuvj420p, native 4784×3190, 673 frames, nominal 10 FPS,
67.3 seconds. Container timing is not verified physical acquisition timing.
Index separation is not proof of independent recording sessions. File modification
times differ, but are not capture metadata. No archived non-RAW AVI launch for
these IDs was found in the reviewed local records; undocumented exposure remains
possible. The exact source hashes differ from the four existing development clips.

Both IDs are outside the Phase18 sealed split's index range 1–100. No sealed
recording is decoded, copied, inspected, or substituted. No RAW16 is accessed.
Only split metadata was read for exclusion. No clips are selected/replaced after
seeing detector results, and no broad collection scan is authorized here.
These two clips become development-exposed after this run, not held-out tests.

## Frozen algorithm and safe execution

Run the identical v34-validated v29 combined serial PVA/CUDA path using the
existing hash-checked dependency stack. Detector, motion, tracker, qualification,
learning, frame cadence, bit depth and full input resolution stay unchanged.
The new wrapper accepts only these two exact source/hash pairs; it is not the
new regional compensation experiment or a stationary-mode implementation.

Reuse the existing AOT wrapper's dependency/runtime checking functions, not its
fixed AOT input or annotation/scoring entrypoints. Check native library identities,
generated tracking-method identity, source/code/config hashes, device-call counts,
PVA attempts, lifecycle cleanup, complete frame ordering and absence of unexpected
fallbacks. No claim of output parity to an unavailable historical new-clip run.
Do not import ground-truth labels into detection. Known feature-starvation
abstentions remain explicit, not CPU fallback or target absence.

Use a new unprivileged `/tmp/seaqr_discovery_pair_20260928_XXXXXX` workspace, one
worker and fresh outputs. Existing data/results remain intact. No sudo, installs,
native recompilation, process killing outside the owned child, reboot, persistent
settings or clock/power changes. Check clock policy before/after and keep default
current clocks; do not compare discovery throughput as a controlled v34 speedup.
Start below65°C; externally check available CPU/junction temperatures every2s,
stop owned job at75°C or missing mandatory telemetry. Bound each inference/render
job to900s and the entire supervisor to3600s. Stop on unexpected failure; no
automatic algorithm/configuration changes, overwrites or retry loops.

Run complete clips from frame0, preserving warmup and causal history. Source/hash
and decoded shape/order checks are retained. Per-frame SHA256 checks are not
added to this new run; complete source hashes and the frozen native frame reader
bind media. Hardware/runtime guards are still correctness checks, not real-time
deployment validation. Rendering happens separately after inference.

## Human-review deliverables

For each clip:

1. Complete overview with unmarked source on the left and all qualified tracker
   states on the right. Preserve every frame and nominal playback rate. Overview
   is downscaled for display only; the detector processes native input. A tiny
   point may disappear in the overview, so absence cannot be inferred from it.
2. A bounded native-resolution crop reel, up to six windows, one per equal
   source-frame bin. Choose distinct qualified-measurement identities using the
   largest first-to-last actual source-coordinate displacement within the bin
   (at least two samples); tie by earliest frame and identity. If no two-sample
   identity exists, choose the earliest qualified actual observation. Empty bins
   stay empty; do not borrow from another bin. Previously chosen identities are
   not repeated. This deliberately detector-led selection is not unbiased review.
3. Each window uses a fixed native384×384 crop centered at the first qualified
   observation of the chosen identity in that bin, origin shifted inward only
   to keep the crop in bounds; 2s before and4s after, truncated at clip ends.
   No moving recentering or interpolated evidence. Show source, current qualified
   measurements, and full qualified track context in separate panels. Show all
   qualified states in the crop, not just the selected identity. If the object
   exits the crop, leave it exited.

Green circles marked M are current qualified measurements. Dashed amber P marks
are predictions, with age since last measurement; neither color establishes
airborne class. Use actual measurement coordinates for M, filtered/predicted
coordinates only for P. Keep offscreen marks offscreen, never clamp them onto an
image edge. Caption/frame/time and scale belong outside raw source panels.
No contrast changes or synthetic detail. CRF12 H264 files are lossy viewing
copies; original AVIs remain the authoritative source on Jetson.

Freeze rendering selection rules and code before detector outputs. Preserve full
selection/omission counts and source/journal hashes. Verify full decoded output
frame counts and save representative encoded QA frames. Keep empty detection or
empty crop selections explicit, not replaced with positive-looking examples.
Copy review videos, compact reports/receipts and compressed journals to local
SEAQR outputs; full original AVIs need not be downloaded for this first review.

## Interpretation and next decision

Report processing coverage, actual measurement records, predictions and selected
review identities separately. Counts of per-frame states are not object counts.
No confirmed airborne count, recall, precision, or false-alarm rate is assigned
from detector outputs. No output does not prove no airborne objects are present.

The user reviews source pixels and supplies timestamps/regions and any class
context. Confirmed visible trajectories can then join the development reference
list, preserving uncertainty about physical class. A source-only follow-up can
also search for misses; this detector-led shortlist cannot measure recall.
Two clips are a useful bounded start, not enough to establish generalization.
