# Two-clip frozen 8-bit discovery: delivered for review

Run scope: exactly Jetson `chunk_0170.avi` and `chunk_0240.avi`, both native
4784×3190, 673 frames, nominal10 FPS. Detector/motion/tracker settings unchanged
from the v34-validated v29 PVA/CUDA stack. No RAW16 or sealed holdout media accessed.

Both inference runs passed the full-frame/hash/backend/lifecycle checks, but the
outcome is **not new confirmed airborne examples**:

- 0170: only1/673 detection-ready frames,627 motion resets, no qualified identities.
  Effectively unavailable, not proof of an empty scene.
- 0240:447/673 ready frames,78 resets,704 qualified track identities,3,858 measured
  states and4,302 predicted states. None of these quantities is an object count.
  Six predeclared detector-led native crop windows are pending human review.
- Both sampled views are heavily obstructed by a roof/awning; sampled crops do
  not establish clear airborne objects. Do not silently promote candidates to truth.

Initial launch stopped before media due an idle optional thermal sensor read;
bounded supervisor fix and fresh workspace used. Both inference runs then passed
in `/tmp/seaqr_discovery_pair_20260928_ZGLHH7`. Remote export failed because
ffmpeg4.4.2 lacked `fps_mode`; that failed batch receipt is preserved. No detector
rerun or selection changes: both exact originals and journals were transferred,
then the same frozen renderer completed locally with compatible ffmpeg.

Local package: [review guide and complete evidence](/Users/romanmaksymiuk/Documents/SEAQR/outputs/seaqr_discovery_pair_20260928/README.md).
Includes two original AVIs, two67.3-second source/overlay overviews, six native
windows, one36-second stream-copy reel, per-frame journals/receipts, and a pending
review queue. No benchmark labels changed.73 local unit tests passed; all videos
fully decoded and reel frames exactly match the six original encoded windows.

Next: user reviews source pixels and provides clip/source timestamps/region for
clear motion. Annotate those trajectories, retaining unknown physical class where
unsupported. These clips are development-exposed, not independent validation.
