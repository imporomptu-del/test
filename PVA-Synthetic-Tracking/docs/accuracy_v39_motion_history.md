# V39 compact-light motion-history diagnostic

This is a journal-only, development-exposed diagnostic. It changes no detector,
association, qualification, source reference, or production setting. The feature
has provisional image-space visibility, not established airborne class or
continuous physical identity. No video, RAW16, holdout, or remote source was read.

## Reproduction and scope

Run from the repository using a **new** output filename:

```sh
./.venv/bin/python scripts/diagnose_accuracy_v39_motion_history.py \
  --output results/tiny_target/accuracy_v39_20260925/motion_history_02.json
```

The first result is
`results/tiny_target/accuracy_v39_20260925/motion_history_01.json`. The script
refuses overwrite, verifies the pinned original `0029` journal, reads only its
frame 0–24 prefix, and rechecks all input hashes before writing. It records 15
actual measurements of `bright:80` from its frame-9 birth and 11 of `bright:138`
from its frame-14 birth. Whole-file SHA verification is provenance, not analysis
of later journal observations. No reference annotations enter the calculation.

The measurement is the accepted detector candidate's raw **reference-coordinate
peak**, recovered by exact candidate/source matching and checked against
`source_to_reference`. The filtered track position is not the raw measurement.
An independent, unscaled-seconds QR quadratic fit reproduces all 18 ready
motion-quality states through frame 24; maximum RMSE disagreement with the
saved production results is **1.49214e-13 px**. The production rule uses up to
eight measured hits, requires five, and permits at most 3 px radial RMS residual.
A coast adds no hit. The diagnostic records every residual and leave-one-out
influence; none is removed from any accepted history.

## What actually prevents qualification at frames 16–17

Both IDs have actual associated measurements on every frame 14–24. There is no
need to synthesize a detection at the two missed reference samples.

| Frame | `80` hits / fit RMSE | `80` qualification | `138` hits / fit RMSE | `138` qualification |
| --- | --- | --- | --- | --- |
| 14 | 5 / 1.777954 px | qualified | 1 / not ready | not ready |
| 15 | 6 / 2.757098 px | qualified | 2 / not ready | not ready |
| 16 | 7 / 3.051856 px | fit fails | 3 / not ready | not ready |
| 17 | 8 / 3.321055 px | fit fails | 4 / not ready | confirmed lifecycle, fit not ready |
| 18 | 9 / 3.170235 px | fit fails | 5 / 1.074377 px | qualified |

The total-hit count may exceed the eight-hit fit window. ID 80 has adequate
confirmation and excursion; its ready-but-failed motion fit is the qualification
blocker. ID 138 does not yet have five actual positions at frames 16–17. Its
frame-17 lifecycle confirmation cannot substitute for the missing fifth fit hit.

For ID 80 at frame 16, the fit uses frames `[9,10,11,12,14,15,16]` and has total
squared residual **65.196773 px²**. Horizontal residuals contribute **98.7487%**.
Frame 15 has residual `(-5.614719,+0.628689)` px and contributes **48.9600%** of
the squared residual; the current frame 16 contributes only **8.6902%**. At
frame 17, horizontal residuals contribute **99.0722%** of **88.235271 px²**.
Frames 15 and 16 contribute **22.5748%** and **34.3522%**, respectively. Thus
qualification is failing on the shape of a short history, not simply a weak
current detection.

The all-sample influence report makes this concrete: omitting frame 15 would
reduce frame-16 RMSE to 1.802834 px; omitting frame 16 would reduce frame-17 RMSE
to 2.561202 px. These are diagnostic counterfactuals, **not a defensible rule for
discarding whichever point makes a track pass**. It is not known which peak is
the physically appropriate representation.

A prior-only quadratic forecast also shows that a replacement innovation gate
would need independent justification. The innovations at frames 16 and 17 are
8.228968 and 8.377130 px, while frame 15 already has an 11.359927 px innovation
and nevertheless passes the existing fit rule. No innovation threshold was
tested or selected.

## Localization and association evidence, without identity claims

The selected raw horizontal positions are:

| Frame | ID 80 x | ID 138 x | `138 − 80` x |
| --- | --- | --- | --- |
| 14 | 2644 | 2648 | +4 |
| 15 | 2652 | 2660 | +8 |
| 16 | 2678 | 2673 | −5 |
| 17 | 2687 | 2691 | +4 |

ID 80's successive horizontal increments here are **8, 26, 9 px** at 100-ms
intervals, versus ID 138's **12, 13, 18 px**. The IDs exchange their left/right
ordering at frame 16 and exchange it back at frame 17. At frame 16 both selected
peaks have y=3151; their five-pixel separation is not a rounding artifact.
The saved competing-alternative flag is true for both IDs at frames 16–18.
This is concrete evidence of nearby response/association ambiguity coinciding
with the fit failure, but not proof that one track was assigned the wrong
physical object. Multiple peaks may be lobes of one feature or distinct sources.

At frame 16, ID 138 selects peak `(2673,3151)` with likelihood cost 6.430582;
its alternative peak `(2678,3151)` is 1.744973 higher. ID 80 selects the latter
peak with likelihood cost 23.418849, and the recorded competing ID-138 option
for that same peak is 15.243294 lower. These likelihood differences are **not**
differences of the full regularized assignment objective. The association also
uses maturity/appearance terms and greedy one-to-one constraints. This bounded
journal does not supply a replay of every rejected option and covariance, so a
negative alternative likelihood gap is not an independent proof of wrong
assignment or suboptimal global assignment. The JSON retains the exact audit
fields and both global/bright-only candidate indexes for subsequent inspection.

## One defensible bounded continuation experiment

Preserve strict qualification, and represent a short interruption of an
established, continuously measured ID as **quality-degraded measured evidence**,
not a confirmed detection. Only a preceding actual strict-qualified measurement
may establish or refresh this lineage. Require unchanged segment/ID, actual
measurements on every intervening frame, and continuing confirmation/excursion;
coasts, resets, missing IDs, other qualification failures, or an expired absolute
budget terminate it. Never borrow evidence from ID 80 to qualify ID 138, merge
the peaks, or move measurements toward a review position.

The separately declared V39 candidate uses two frames **and** 200 ms from the
last strict-qualified actual measurement. That explicit non-refreshing bound is
a development hypothesis informed by this exposed two-frame interruption,
**not** an independently calibrated optimum. The journal analysis does not
establish that this budget improves airborne accuracy. It can test whether the
output gap is bridged transparently while making added workload and expiration
auditable. An increasingly wrong assignment must not extend itself indefinitely
through degraded states, and an alternating genuinely strict-pass history can
still repeatedly re-establish the budget; report that workload honestly.

Before any promotion, test sustained bad associations, neighboring distinct
targets/crossings, paired lobes, turns/acceleration, intermittent visibility,
variable frame intervals, coast/reset/deletion/reused IDs, and transaction
atomicity. Full saved-journal replay must preserve all original known-positive
denominators, count predictions separately, and report added measured states in
all four allowed development clips and the fixed provisional-control windows.
Source review and independent class/identity evidence remain necessary before
calling changes airborne recall, false-positive reduction, or identity repair.

This policy cannot correct raw localization or assignment and does not repair
V36's separate spatial rejection of visible extended/paired support. Raising
the 3-px threshold, selecting a best alternative assignment by reference fit,
dropping the largest residual, or declaring every nearby lobe a duplicate is not
supported by this diagnostic.
