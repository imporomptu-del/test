# After V38: repair the evidence-to-output chain

V38 removes the local-registration prerequisite from source extraction, but
does not establish a safe rejection rule. Keep V34 production behavior and
the frozen V36–V38 experiments intact. Continue accuracy work on the allowed
8-bit development clips; RAW16, Jetson speed work and sealed holdouts remain
out of scope. No new labels may be inferred from detector acceptance.

## First: make each output loss explainable

The separate compact-light reference has eight visible frames. Its source
positions are provisional and class-unknown. Saved journals show actual
in-radius detector candidates and associated measurements on all eight,
but only six frames have qualified moving measurements. V36 retains four.
Frames 16–17 are not missing source detections: existing motion-quality
requirements prevent qualification. Multiple measured IDs coexist, so the
change in matched ID is not automatically an identity switch or track birth.

Build a causal replay report of the stages:

`source reference -> candidate -> association -> motion qualification -> image verifier -> output`

Preserve the original reference radius and ambiguous frames. Report every
visible sample at every stage, including unmatched cases. Inspect the actual
measurement history causing the motion-fit rejection, rather than simply
raising its residual threshold. Distinguish an erroneous association from
uncertain feature localization and genuine nonconstant motion. Do not merge
nearby measurements solely because a provisional whole-feature reference
encompasses both; they can represent separate lobes or separate objects.

## Second: test a source-driven localization/model correction

The current point bank is centered on a detector measurement. A visible
extended or multi-lobed feature need not be well explained by that bank, even
when the measurement falls inside the reference's uncertainty radius. In the
new case, frame 18 remains edge-preferred at all 36 probes. That counterexample
must survive any proposed cleanup.

Design one bounded, source-driven localization/model diagnostic before
looking at its scores. Use no truth coordinates, clip IDs or timestamps as
inference inputs. Keep original measurements and all alternatives auditable;
do not replace the actual measurement with the review center. Test point-on-
edge, multi-lobed/extended features, close separate targets, curved clouds,
exposure change, slow motion, turns and intermittent visibility synthetically.
Account for the complexity of any richer model; raw maximum fit gain is not
by itself evidence of a target.

## Then: one separately frozen causal policy experiment

Only after the two failure mechanisms are understood, specify one candidate
track-level policy before evaluation. An isolated poor/unknown image fit must
not silently become proof of absence. At the same time, remembering old good
evidence cannot confer indefinite validity on a wrong association. Any grace
period, decay, uncertainty model or qualification hysteresis needs explicit
bounded behavior and synthetic failure tests, not scene-specific exceptions.

Use the original full-frame saved journals where the necessary information
exists; rerun only when an explicitly identified missing intermediate makes
that necessary. Preserve chronology, all reference denominators, old misses,
alternative IDs and provisional-control scope. Require no newly lost known
development samples before considering output promotion. Count nuisance
workload honestly; unlabeled responses are not verified false positives.

Finally review a predeclared bounded before/after set in native source pixels,
including regressions and ambiguous cases, before regenerating a cleaner
presentation. Neither a smaller overlay count nor these few development
encounters establishes general airborne detection accuracy. No production
promotion or holdout access is authorized by this planning document alone.
