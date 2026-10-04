# V57 — bounded causal persistence shadow (not a production change)

## Decision and scope frozen before scoring

The purpose is to measure whether deferring an edge-based veto until repeated
actual measurements reduces the damage of V36's single-measurement rule. This
is a development diagnostic, not a new detector or an airborne classifier.
The known generated point-on-strong-edge counterexample is a promotion blocker
even if every currently labeled development sample is retained.

Only the four already audited V34 8-bit development journals (0029, 0126, 0055,
0082; 2,741 frames) and saved V36 native-patch diagnostics are used. No videos
are decoded, no RAW16 or sealed holdouts are accessed, no remote job is started.
The cached features have audited provenance/algebra and bounded independent
patch checks, not a new independent pixel refit for every measurement.

## One candidate; no sweep

- Eligibility begins with original baseline-qualified output only.
- An informative point-minus-edge margin above zero retains the measurement.
- A nonpositive informative margin starts an edge run. Reject only on the
  second and subsequent consecutive actual qualified measurements, in adjacent
  source frames and no more than 200 ms apart.
- A point-preferred or unknown measurement, coast, missing/unqualified track,
  segment transition, or motion reset breaks the edge run. A same-segment reset
  also clears all cached evidence.
- Coasts can inherit only the most recent actual qualified measurement verdict,
  for at most seven source frames AND 700 ms. They never refresh evidence or
  become measurements. Missing/expired history retains baseline output as
  explicitly unknown, not as point support.
- Baseline and historical V36 are reported alongside the candidate. No V39
  degraded-measurement additions are combined with this candidate.

All thresholds, detector births (4 sigma), tracking, association, qualification,
coordinates, candidate inventory and background-learning feedback are unchanged.
No clip, ROI, ID, time or labeled-reference exceptions are allowed in the rule.

## Freeze, execution and independent audit

Generated core, runner, stress and independent-FSM tests must pass first.
The runner hashes exact allowlisted inputs, verifies their existing receipts,
and copies/hash-binds implementation, tests, plan and configuration before any
real candidate decisions. It creates a fresh output directory, with no overwrite
or partial-resume behavior. The independent auditor reconstructs every decision
without importing the candidate core. Inputs and code are rehashed after each
analysis. Any repair after real scoring requires a new, disclosed experiment.

## Evaluation and interpretation

Report measured and predicted output-state counts separately for every clip,
reason/evidence tiers, all seven original provisional control scopes, and their
known baseline and V36 counts. These are unlabeled review workload, not false
positives, precision, recall, or verified airborne-negative controls.

Preserve the frozen reference coordinates, gates, polarity and assigned
identities. Report the dense (285), pilot (28), anchor (24), compact-light (8)
and grid (11) panels separately: they overlap and are not independent encounters.
Verify saved actual alternatives and qualification against the original journal.
Score retention of the original qualified assignment and any surviving qualified
alternative separately; never silently replace an assignment to hide a loss.
The baseline's existing misses remain misses. In particular, the frame-216
coast in 0126 is not an actual-measurement hit.

Generated stress includes an explicitly present point on a stronger edge.
The rule is rejected for production if it suppresses that known point, regardless
of workload reduction or familiar-reference results. This tests the proposed
mechanism's failure, not the complete detector's end-to-end synthetic accuracy.

The next decision follows the evidence, without tuning this run. Physical class
of the current visible features remains unknown; airborne-only operational
accuracy still requires suitable independently labeled footage.
