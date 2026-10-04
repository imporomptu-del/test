# Observation-backed output layer

Approved scope: separate fresh observation alerts from predicted track context,
without changing the detector, qualification, association, or track retention.
This is a new downstream API and replay/export consumer, not an airborne classifier.

## Contract fixed before saved-journal evaluation

- Observation alerts are exactly current `qualified_moving && measured` records,
  using `measurement_source_xy`. No fallback to filtered coordinates.
- Track context retains every current qualified state, using actual measurement
  coordinates when measured and filtered predicted coordinates otherwise.
- Qualification is never cached or inferred. Per-stream history supplies only
  time/frame age since a measurement, including measurements before qualification.
- Explicitly separate channels, include stream/segment/track identity, reject bad
  rows before changing state. Unknown previous observation age stays null.
- Keep physical class unknown; do not describe output counts as real objects,
  false positives, improved precision, or lower false-alarm rate.
- Do not mutate tracker rows or feed the output policy back into learning/tracking.
  Historical code, source packets, labels, video exports, and journals are immutable.

## Bounded validation

Generated unit tests, an independent code review, then replay the already exposed
v34 full-repeat0 journals for0029,0055,0082,0126. Verify historical manifest hashes,
complete frame counts, every alert's exact identity/measurement coordinate, and
every retained qualified prediction. Check the frozen source-first references for
0029 frames0–59 and0126 frames500–579 without changing labels or denominators.
The0082 fixed ROI is output workload, not a negative benchmark.

Render small original/observation-alert/track-context comparisons using existing
decoded crops from the prior source-first packet only. Validate all encoded frames
and inspect representative output frames. No new media search, RAW16, sealed
holdouts, remote deployment/jobs, threshold tuning, or tracker/coast changes.

Deliver local module, replay CLI, generated tests, independent audit, exact results,
and new videos. The historical pipeline's detector/tracker and renderer defaults
stay unchanged. Live applications can adopt the downstream API explicitly; this
work does not claim a deployed Jetson/UI integration.
