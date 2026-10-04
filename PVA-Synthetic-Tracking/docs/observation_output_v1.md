# Observation alerts and retained track context

`tiny_target.visible_output.ObservationOutput` is a downstream consumer of
completed tracker rows. It emits two distinct channels without altering detector,
tracker, qualification, or learning inputs:

| Channel | Inclusion | Position | Meaning |
|---|---|---|---|
| `observation_alerts` | Qualified and measured on this frame | `measurement_source_xy` | A current qualified motion observation |
| `track_context` | Every qualified record | Actual measurement if measured, otherwise `source_xy` | Track continuity, including predictions |

Neither channel verifies airborne identity. Every entry carries
`physical_class: "unknown"` and `airborne_confirmed: false`. An observation alert
is a per-frame data record, **not** a newly discovered object, deduplicated incident,
notification, or command to another system. Repeated observations keep the same ID.

## Using the API

```python
from tiny_target.visible_output import ObservationOutput

# One instance per camera/recording session, outside the frame loop.
output = ObservationOutput("camera-A/session-001")

# After the existing tracker finishes each frame, including empty frames:
channels = output.update(completed_tracker_row)
current_observations = channels["observation_alerts"]
retained_context = channels["track_context"]
# Original completed_tracker_row["tracks"] remains unchanged.
```

Consumers can use `current_observations` for an observation-backed display and
`retained_context` for continuity. Do not feed either list back into tracking or
background learning. Unqualified internal tracks still exist in the original
tracker but are absent from these qualified-output channels.

`last_measurement_age_ns` and `last_measurement_age_frames` are zero on fresh
observations. Predictions carry time since the last observation actually seen by
this consumer, including observations made before qualification. No invented
zero-age value is assigned when history is unknown: all age fields stay null.
Age uses the row's source timestamps, **not wall-clock/network latency**.

Call once per ordered frame. The first frame may be any nonnegative index;
subsequent frames must be contiguous, timestamps strictly increase, and segments
cannot decrease. A new recording/session needs a new instance. Missing IDs and
segment changes clear age provenance, preventing stale identity reuse. A rejected
row leaves policy state unchanged, so a corrected row can be retried.

Identity is `stream_id/segment/track_id`. Track IDs cannot contain `/`; stream IDs
may, so parse with `rsplit("/", 2)` if parsing is necessary. Prefer retaining the
explicit stream, segment, and track fields instead.

## Local saved-journal export

```text
.venv/bin/python scripts/replay_visible_output.py \
  --journal /absolute/path/frames.jsonl \
  --journal-sha256 <verified-sha256> \
  --expected-frames <full-frame-count> \
  --stream-id camera-A/session-001 \
  --output /absolute/path/new-output-directory
```

This reads metadata only and creates `channels.jsonl` plus a completed
`summary.json`. The full journal must begin at frame0. Hash mismatch, malformed
input, incomplete inventory, or an existing destination fails the export. A
failure after output creation may leave a partial directory; without a completed
summary it must not be consumed as a verified export. Existing files are never
overwritten.

The separate `scripts/audit_visible_output.py` independently verifies the export
against the original journal, including every coordinate, identity, age, and
count. It imports neither the policy nor the replay implementation.

## Adoption status and limitations

This version is implemented and tested locally through the API, replay CLI, and
new review-video consumer. It is **not deployed to the Jetson or wired into a
live camera UI**. Existing pipeline/renderer behavior remains unchanged until a
consumer explicitly adopts these channels.

Removing predictions from the observation channel can make that view less busy;
it does not remove false measured detections, classify airborne objects, shorten
acquisition delay, or improve measured recall. The retained context still bridges
gaps using the unchanged tracker. Use both channels where intermittent visibility
matters, making predictions visibly distinct from current measurements.
