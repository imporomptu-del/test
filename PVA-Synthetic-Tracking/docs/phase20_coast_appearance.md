# Measured appearance through a short coast

This optional visible-AVI tracking change addresses an observed failure in which
one missed frame erased all brightness-consistency evidence. A mature track could
then take a nearby weak background response while the actual moving feature
started a new, unqualified track. The fix does not invent a measurement during a
miss and does not lower a detection threshold.

Use `tracking_association_appearance: log_response_coast` only with position-only
Gaussian association. The default remains `none`; the previous `log_response`
mode is unchanged. For independently confirmed tracks, finite positive observed
responses contribute the existing squared log-amplitude difference. During a
coast, its weight is:

`1 / (1 + missed_windows / (max_missed_windows + 1))`

Evidence beyond the existing coast budget contributes no penalty. This is a
provisional soft association regularizer, not calibrated probability or airborne
classification. Unopposed fading responses are still eligible; bright responses
are not preferred simply for being bright. Position/Mahalanobis gates,
independent-hit confirmation, motion-quality criteria and candidate selection
remain unchanged.

## Development evidence and limits

The fresh closed-loop PVA trial uses only previously reviewed clips 29, 126, 55
and 82. No labels enter the detector, and no sealed holdout is opened. The exact
pre-speed-optimization runtime and policy are preserved under
`results/tiny_target/phase20/tracking_continuity_v9_20260914/`.

Completed target-clip scoring recovers all 14 confident visible samples of the
first chunk29 encounter on one track ID, versus 12/14 and two IDs before this
change. The overlapping pilot check is also 4/4 versus 3/4. These two label sets
are not independent validation data.

The second chunk29 encounter remains 132/132 but has two observed track IDs and
four ambiguous samples. Chunk126 remains 138/139 on one ID: at frame 216 the
track coasts, and the scorer correctly does not count that prediction as a
measurement. Neither remaining issue is declared fixed. This experiment does
not establish population-level airborne recall, precision or false-alarm rate.

## Separate exact host execution improvements

The speed work leaves the above policy unchanged:

- Shared Kalman prediction/correction algebra is reused only for byte-identical
  covariance inputs and exact integer time deltas within one update. Mutable
  track covariance arrays remain separate. No covariance rounding or fast math.
- The visible adapter omits a generic quality summary it never consumes; all
  running evidence is still accumulated and the generic API defaults to full
  summaries. Requesting a summary later retains the entire observation history.
- Sparse observed-footprint protection paints the exact disk dilation union;
  dense inputs retain the full-image implementation. No search pixels change.
- GPU peak records are filtered and converted in batches, preserving numeric
  values, within-cell order and round-robin candidate admission exactly.
- The second host revision caches a recognized NumPy contraction graph instead
  of rebuilding an einsum plan for each track. NumPy's tensordot/reduction and
  newer batched-matmul graphs have different floating-point behavior, so the
  implementation preserves the installed graph rather than substituting one
  for the other. Unknown plans or unsupported layouts use the reference path.
  Pairs already excluded by every geometric gate do not need likelihood work;
  covariance validation and rejection accounting still occur.

These changes have separate frozen-source numerical tests and an execution-only
full-video comparison. They are not a reason to relax the unresolved accuracy
criteria. Actual Jetson timing and final verification are recorded in the
experiment artifacts, not inferred from unit-test speed.
