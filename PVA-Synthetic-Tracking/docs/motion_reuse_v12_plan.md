# v12: adjacent-frame motion image/pyramid reuse

Generated-only, default-off execution experiment. No real camera or sealed
manifest reads, no new media IDs, service/system changes or live deployment.
Use a new isolated Jetson workspace with one experiment worker. Retain all
failures, reports and prior sources. No subagents are needed.

The visible-AVI pipeline genuinely matched 284/285 reviewed measurement samples
over the known three encounters; this is not population recall or airborne
classification. One unmeasured/coasted sample and an ambiguous/split encounter
remain. RAW16 has separate unresolved sensitivity limitations. This optimization
must not be represented as an accuracy fix.

Reuse only the last successful call's current-frame prepared VPI image and
pyramid when that exact immutable Frame object becomes the immediately adjacent
previous frame. Preserve the original mapping, resize, pyramid parameters,
Harris/flow algorithms, thresholds and per-pair VPI cache reset. Never cache flow
objects, statuses, Harris features or fit results. Keep wrapped host pixels alive.

Configuration changes, source changes, index/sequence gaps, discontinuities,
different Frame identities, explicit reset, exceptions and close invalidate reuse.
The adapter is bound to one thread and stream. Publish a new reusable entry only
after the original estimator completes successfully. Fail closed on misuse.
Keep the existing one-frame CPU conversion cache in BOTH comparison arms.

Before timing, compare complete non-timing point/fit outputs on four seeds of
the existing 12 motion control families, true adjacent sequences (including
clean/noise/clean recovery), repeated/reordered pairs, masks and explicit resets.
Exercise actual cache hits, not only independent pairs. Include native-size
generated sequences and guards for identity/configuration/lifetime invalidation.
Unobservable/rejected motion is not counted as successful detection.

If exactness fails, stop timing and diagnose; do not loosen comparisons. If it
passes, four alternating reference/candidate repeats on native generated 8-bit
and RAW16 sequences, with identical inputs and configuration per comparison.
Measure the full motion-estimator call, not just submit times; output verification
is outside the timing interval. Report cold and reused calls separately. Do not
infer full-pipeline FPS or guaranteed latency from generated estimator timing.

Copy compact evidence locally and verify source/gate hashes and complete schedules.
Any faster candidate still requires real-video/end-to-end regression and safe
integration before promotion; existing runtime defaults remain unchanged.

## Fixture correction after attempt 01, before timing

The resized native U8 fixture returned zero/no-surviving Harris features in
both arms, with identical outputs and unchanged inputs. It therefore could not
exercise reuse as a positive. Retain its first three pairs as explicit
unobservable negative controls. Extend the positive fixture by tiling the same
generated texture at its original feature scale instead of enlarging it. No
estimator thresholds, acceptance criteria, adapter code or data precision change.
The failed attempt and source archive are retained. Rerun all gates before timing.
