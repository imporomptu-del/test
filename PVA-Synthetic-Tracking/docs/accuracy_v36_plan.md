# v36: accuracy diagnosis and causal shadow-gate experiment

2026-09-24. Prepared before evaluating the candidate policies. Preserve v34/v35,
all prior references and default detector/tracker policies. No RAW16, sealed
holdouts, remote jobs, clock changes or media-wide discovery in this experiment.

## Evidence and problem

The frozen v34 full-frame overview includes substantial clouds/edges and predicted
states. Qualification currently uses cumulative confirmation, lifetime movement,
and a fit to the latest eight measurements, irrespective of their age. Motion can
stop without losing lifetime excursion. Sparse observations and coasted states
can retain their previous passing motion fit. Shape consolidation does not reject
an unbounded/unsupported peak. These are testable mechanisms, not proof that all
unmatched outputs are false alarms.

Changing internal qualification changes causal variance-learning protection.
Therefore this first experiment is **shadow output eligibility only**, downstream
of immutable v34 journals. It is not a simulated closed-loop detector rerun and
will not claim changed measurements, association, GPU speed or airborne accuracy.

## Frozen ablations

Apply to all four allowed development clips: 0029, 0126, 0055, 0082. Stream every
frame in order, use measured source coordinates mapped to reference coordinates,
never filtered/predicted positions, and reset histories at segment changes and
track deletion. No annotations, clip IDs, absolute times or locations enter policy.

Use existing baseline constants (8 measurements, 5 minimum hits, 12px excursion),
not a parameter sweep. Compare:

1. Baseline qualification, unchanged.
2. Recent support: at least five measurements in the last eight **frames**.
3. Recent excursion: at least 12px bounding-box excursion in the latest eight
   measured positions, rather than across the track's lifetime.
4. Bounded shape: the latest measured peak has valid nonempty observed half-height
   support. Absence means unsupported evidence, not proof of a nonexistent object.
5. All three predicates combined.

Each is a subset of the baseline qualification. Predictions add neither hits nor
motion evidence. They stay separately identified; removing predictions is not
counted as a gain in measured accuracy. These rules can reject genuine intermittent,
slow, blurred or faint objects. That is precisely what the reference checks test.

## Scoring and decision

- Verify every parent journal/launch/report against the independently audited v34
  manifest before analysis. Freeze new source/test/plan and all reference hashes.
- Reuse dense source-reviewed 285 visible-point samples and the overlapping,
  stricter 28-sample pilot **separately**. Preserve the original 24 confident
  manual anchors and compare per-frame matches/identity ambiguity, not just totals.
- Any previously passing reference measurement lost rejects an arm, even if
  overall displayed counts drop. Predictions never rescue a missed measurement.
- Compare all measured/coasted responses in all seven already reviewed nuisance
  source ROIs/time windows. These are provisional assistant-reviewed controls,
  not independently adjudicated negatives or whole-frame false-alarm rates.
- Report all four clips' unlabeled output workload separately. Do not label
  unmatched tracks false positives, treat unknown visibility as absence, or
  extrapolate these seven purposively selected crops to the whole camera.
- No automatic threshold retuning after seeing results and no default promotion.
  A surviving output gate would still need source-level nuisance review and a
  separate closed-loop evaluation if used to alter tracking/learning feedback.

The intended operational class is airborne objects only. Existing references
establish moving image features, not independently verified airborne identity.
Keep physical class unknown; future controlled recordings/authoritative labels
are needed for operational precision, recall and false alarms per minute.
