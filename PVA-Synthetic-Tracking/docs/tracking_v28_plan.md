# v28 — broader exact tracking-stage prototype

Keep serial v20 as the reference and v27 as a separate intermediate control.
Do not combine v26, decode new media, access RAW16/holdouts, change detection
settings, or alter installed/default behavior in this replay experiment.

## Implementation scope

- Retain v27's strictly compiled, tested geometry batch.
- Group prediction setup within each update, retaining per-track matrix-vector
  products and byte-keyed covariance reuse with independent mutable outputs.
- Prepare unique innovation covariances together and use NumPy's stacked inverse
  and slogdet operations. No analytic inverse, rounding, or covariance binning.
  Unsupported/nonfinite layouts retain the original scalar path.
- Index spatial-fair replacement candidates by cell, preserving the complete
  victim ordering, proposal ordering, protected tracks, and rejection diagnostics.
- Pack float64 record arrays through NumPy's Python-value conversion rather than
  Python scalar generators, with fallback for other layouts/types.

Source and code guards make these opt-in research adapters. All scratch and
caches are local to an update. No private manager state is added.

## Gates and evidence

1. Generated component tests: byte equality, near-singular matrices, signed zeros,
   fallback layouts, exact error behavior, independent array ownership; seeded
   complete lifecycle/tie/capacity scenarios and private-state comparisons.
2. On Jetson, reproduce the saved v20 journal outputs and v27 state/learning hashes
   for only the 128-frame 0126 and 0082 development prefixes. Stop on any mismatch.
3. Freeze source/dependency hashes before clean timing. Two alternating triples
   per prefix: v20/v27/v28, then v28/v27/v20. Report all 12 replays, pooled durations,
   p95, and both repeat comparisons. No profiled passes in timing statistics.
4. Profile the final candidate separately; inspect the remaining costs. Audit
   hashes, schedule, state/learning digests and arithmetic independently from the
   timing summary; retain all attempts without overwriting valid reports.

Tracking replay is neither an accuracy evaluation nor whole-pipeline FPS. A
substantial tracking improvement can justify a separately frozen video test,
but this experiment alone does not promote a default or establish real time.
No end-to-end gain will be inferred as a measured result.
