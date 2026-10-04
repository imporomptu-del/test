# V38 compact-light track audit: 0029, frames 12–24

## Outcome and scope

**The unmatched reference samples at frames 16 and 17 are not detector
absences, coasts, or position-gate misses.** Logged bright detector candidates
were assigned to actual measured tracks inside the unchanged reference gate.
None of those tracks was `qualified_moving` on those frames. The mature
`bright:80` track failed its logged motion-quality check; the other nearby
tracks did not yet have a ready motion-quality history.

The reference-matched qualified identity changes from `bright:80` at frames
14–15 to `bright:138` at frames 18–19 and 23–24. Both IDs coexist and have
actual measurements on **every frame 14–24**. This is a change in which logged
track meets output eligibility and the reference gate, not evidence of a
verified physical-identity handoff, an ID switch, disappearance, or rebirth.

This audit inspected only saved journal/decision rows for frames 12–24 and
the separately reviewed reference and frozen V38 selection. It decoded no
media, opened no RAW16 or holdout sources, made no remote calls, and changed
no frozen file, coordinate, uncertainty, gate, or tracker setting. Whole-file
SHA-256 values were read for provenance, but semantic journal inspection was
limited to the stated frame interval. The new document is a diagnostic report,
not a new truth annotation or parameter-selection exercise.

The reference describes a **provisional class-unknown luminous image feature**.
It supplies eight visible-frame locations and five ambiguous frames, not a
verified airborne target, a physical-object count, or verified correspondence
through the ambiguous frames. The review window was selected from development
filter removals. It is not held-out validation.

## Fixed matching and layer-by-layer accounting

The unchanged reference gate is Euclidean source-coordinate distance no larger
than `position_uncertainty_radius_px + 2.0`. Matching uses an actual
`measurement_source_xy`, not a filtered/predicted `source_xy`. A baseline
qualified match additionally requires `measured=true`, `qualified_moving=true`,
and bright polarity. All qualifying alternatives are kept; the named baseline
ID is the nearest qualifying alternative, not physical-identity ground truth.

Within the eight visible-frame gates, there is at least one logged bright
candidate and associated actual measured track at **8/8** samples, a qualified
measured baseline match at **6/8**, and a V36-retained qualified measured match
at **4/8**. These are layer-specific coverage counts against this exposed,
provisional image-feature reference—not population recall or airborne accuracy.

| Frame (seconds) | Gate radius, px | Logged bright candidates inside gate | Qualified measured match | Measurement distance, px | V36 outcome for that match |
| --- | ---: | ---: | --- | ---: | --- |
| 14 (1.4) | 5 | 2 | `0/bright:80` | 2.881348 | Retained |
| 15 (1.5) | 6 | 2 | `0/bright:80` | 5.398755 | Retained |
| 16 (1.6) | 6 | 2 | None | — | Not presented to the qualified-output gate |
| 17 (1.7) | 6 | 4 | None | — | Not presented to the qualified-output gate |
| 18 (1.8) | 7 | 3 | `0/bright:138` | 6.886723 | Removed |
| 19 (1.9) | 7 | 3 | `0/bright:138` | 3.487992 | Retained |
| 23 (2.3) | 6 | 2 | `0/bright:138` | 4.811325 | Removed |
| 24 (2.4) | 6 | 2 | `0/bright:138` | 2.148010 | Retained |

All candidates counted above have matching actual measured track coordinates
in the journal. At frame 24, `bright:80` is also measured and qualified but its
measurement is **6.085055 px** from the reference, outside the fixed 6 px gate.
It is therefore correctly absent from that reference's selected keys. The gate
was not enlarged to include it. The in-gate `bright:138` match remains retained.

## Frames 16 and 17: exact missing-output mechanism

Reference positions are `(2675.9,3151.1)` at frame 16 and `(2690.9,3149.7)` at
frame 17; both gates have radius 6 px. Every row below is an actual measured
bright track, associated with a logged detector candidate at the same source
coordinate. None is a coast. All have `qualified_moving=false`.

| Frame | ID | Actual measured source x,y | Distance, px | Lifecycle | Hits / independent hits | Motion-quality history / result |
| --- | --- | --- | ---: | --- | --- | --- |
| 16 | `bright:80` | 2678.045436, 3151.000529 | 2.147741 | Confirmed | 7 / 7 | Ready; 7 positions; RMSE **3.051855944** > 3 px; failed |
| 16 | `bright:138` | 2673.045436, 3151.000529 | 2.856297 | Tentative | 3 / 3 | 3 positions; not ready; RMSE null |
| 17 | `bright:80` | 2687.044251, 3150.003035 | 3.867639 | Confirmed | 8 / 8 | Ready; 8 positions; RMSE **3.321055392** > 3 px; failed |
| 17 | `bright:127` | 2691.044251, 3148.003035 | 1.703086 | Tentative | 3 / 3 | 3 positions; not ready; RMSE null |
| 17 | `bright:138` | 2691.044251, 3151.003035 | 1.310995 | Confirmed | 4 / 4 | 4 positions; not ready; RMSE null |
| 17 | `bright:140` | 2695.044251, 3148.003035 | 4.478226 | Tentative | 2 / 2 | 2 positions; not ready; RMSE null |

`bright:138` becoming lifecycle-confirmed at frame 17 does **not** imply that
it is a qualified moving output: its separate motion-quality state remains
not ready. At frame 18 it has five measured-history positions, becomes ready,
passes with RMSE **1.074376896 px**, and becomes qualified. These are observed
journal transitions; this audit did not reconstruct earlier, out-of-window
history or independently refit the logged RMSE.

The detector was ready and out of warmup on both frames. Logged
`dropped_at_tile_cap`, `dropped_at_frame_cap`, and bright
`dropped_birth_count_at_active_track_cap` are all zero on both frames. The
recorded candidate/association evidence rules out a local detector miss or
failed association for the listed measurements. The missing *qualified
display output* occurs at qualification, before V36's point-versus-edge gate.

## Coexisting IDs and association ambiguity

The two main logged IDs have the following causal states. `M` is an actual
measurement, `C` a coast, `Q` qualified, and `U` unqualified. A V36 dash means
the state is absent from the qualified-output decision journal; it does not
mean V36 examined and rejected it.

| Frame | Reference visibility | `bright:80` | `bright:138` | V36 for 80 / 138 |
| --- | --- | --- | --- | --- |
| 12 | Ambiguous | M, U; history 4, not ready | Not present | — / — |
| 13 | Ambiguous | C, U; history 4, not ready | Not present | — / — |
| 14 | Visible | M, Q; RMSE 1.777954 | M, U; history 1 | Retained / — |
| 15 | Visible | M, Q; RMSE 2.757098 | M, U; history 2 | Retained / — |
| 16 | Visible | M, U; RMSE 3.051856 | M, U; history 3 | — / — |
| 17 | Visible | M, U; RMSE 3.321055 | M, U; history 4 | — / — |
| 18 | Visible | M, U; RMSE 3.170235 | M, Q; RMSE 1.074377 | — / Removed |
| 19 | Visible | M, U; RMSE 3.128875 | M, Q; RMSE 1.743355 | — / Retained |
| 20 | Ambiguous | M, U; RMSE 3.231457 | M, Q; RMSE 1.946565 | — / Retained |
| 21 | Ambiguous | M, U; RMSE 3.314830 | M, U; RMSE 3.201562 | — / — |
| 22 | Ambiguous | M, U; RMSE 3.625718 | M, Q; RMSE 2.938102 | — / Retained |
| 23 | Visible | M, U; RMSE 3.104936 | M, Q; RMSE 2.868134 | — / Removed |
| 24 | Visible | M, Q; RMSE 2.873576 | M, Q; RMSE 2.978220 | Removed / Retained |

The actual measured x ordering changes from 80 left of 138 at frames 14–15,
to 80 right of 138 at frame 16, and back at frame 17. The two source x
separations are 8 px at frame 15, 5 px at frame 16, and 4 px at frame 17.
The bright association audit also flags a competing alternative within its
stated likelihood-factor-three comparison for both IDs on frames 16–18.
Those cost fields explicitly say their priors/appearance are not calibrated
probabilities. Candidate indexes in `tracking_metrics.bright.association_audit`
index the **bright-only candidate list**, not the entire mixed-polarity list;
the audit was checked against the corresponding source positions.

This supports investigating ambiguous assignment among nearby response peaks.
It does **not** establish that two physical objects exist, that a particular
assignment is wrong, or that peak-order changes caused all of the motion-fit
error. Multi-lobed appearance, source saturation, association choices and real
motion remain competing explanations. Continuous physical identity cannot be
decided from these logs.

## The separate V36 removals

The matched `bright:138` states at frames 18 and 23 already passed baseline
qualification. V36 explicitly removes both with `edge_preferred_or_tie`:

| Frame | Point gain | Edge gain | Point-minus-edge margin |
| --- | ---: | ---: | ---: |
| 18 | 0.066109900 | 0.201965693 | −0.135855793 |
| 23 | 0.131177272 | 0.140356056 | −0.009178784 |

These downstream removals are distinct from the upstream unqualified states
at frames 16–17. V38 only computes diagnostics on its frozen selected qualified
observations: it neither repairs qualification nor reintroduces these removed
outputs. Raising the motion RMSE threshold, shortening history, merging IDs,
enlarging the reference gate or changing V36's margin using this case would be
new tuning, not a conclusion licensed by this audit. No such change was made.

## Evidence bindings

Paths below are relative to the `skymove` repository. Coordinates and distances
in this report are rounded only for readability; original artifacts retain
full precision. Hashes cover the original files without rewriting them.

| Input | SHA-256 |
| --- | --- |
| `results/tiny_target/visible_validation_v34_20260923/audit_20260924/evidence/run/full_repeat0_0029/frames.jsonl` | `e97064888d5901c98ced2be81bf7982f6f21fdd571e6a5197e59aaee4b4fbcf8` |
| `results/tiny_target/accuracy_v36_20260924/full_context_01/0029_decisions.jsonl` | `03aa65a6f4e377f34cbd4d6e2bc1bb4ed55695e705a703df41347148b157fa30` |
| `results/tiny_target/accuracy_v38_20260925/compact_light_reference_v1.json` | `44c954cb3c8f338f014766c5e2df801e66ade33f257e6ff1258f554ff08e6c5d` |
| `results/tiny_target/accuracy_v38_20260925/global_multilag_01/selection.json` | `b2ea836c7fafdedffdb92bc97e9457e368809a77bd9c190a87536540e1472123` |
