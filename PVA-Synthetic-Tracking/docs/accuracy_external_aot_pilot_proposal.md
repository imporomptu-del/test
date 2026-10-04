# Next accuracy stage: class-supported external pilot proposal

2026-09-27. Status: documentation researched; **no external dataset metadata files
or imagery downloaded, no scoring, training, deployment, or detector change**.
User confirmed that independent class context for the existing moving points is
unavailable. Do not ask that same question again or relabel the points by assumption.

## Why the current footage cannot answer the next question

Existing0029/0126/0055/0082 references support visible-feature regression, not
airborne-versus-ground classification. The V58 intake remains empty. The recent
0082 fixed-light reference has no qualified baseline matches, so relabeling it
would not establish a false-detection reduction. Its nearby transient remains
physically unresolved. V57 already demonstrated that a plausible cleanup rule
can remove useful visible-feature measurements; another arbitrary veto is not
the next step.

Class evidence does not categorically require operator logs: unambiguous source
context can suffice. We simply do not have that evidence for these tiny points.
The new observation/prediction split is complete and stays unchanged.

## Recommended candidate: Amazon Airborne Object Tracking (AOT)

The publisher describes native2448×2048,8-bit grayscale PNG sequences at10Hz,
with airborne-object annotations. Public training data is accessible without an
AWS account. These characteristics make it a candidate for our8-bit pilot, not
proof that our current pipeline will work on it.
[Official AWS registry](https://registry.opendata.aws/airborne-object-tracking/)

The challenge documentation includes tiny targets, aircraft and birds, and
publisher-labeled no-airborne frames. Imagery was captured by aircraft in daylight,
so it differs materially from our dark local footage and camera motion. Adjacent
sequences may contain portions of one encounter. Tiny-object annotations are
approximate, and published scoring has ignore regions/cases.
[Official challenge documentation](https://www.aicrowd.com/challenges/airborne-object-tracking-challenge)

## Proposed work after explicit access approval

1. Inspect a bounded amount of publisher metadata first. Identify exact source
   objects, annotation version, license/attribution, frame inventory and download
   size before transferring imagery. Propose a hard byte/frame limit; do not fetch
   the complete dataset or accept paid/requester-pays access.
2. Select a small pilot by annotations/recording metadata before seeing SEAQR
   outputs. Include a clear-sky case, a structured/cloud-background case, and
   publisher-supported negative exposure if available within the budget. Preserve
   complete causal input context and make unavailable coverage explicit.
3. Group splits by original flight/day where that provenance is available. Do
   not pretend randomly divided adjacent sequences are independent. Keep a later
   evaluation group unopened and separate from development; existing SEAQR sealed
   holdouts remain sealed.
4. Verify original dimensions/bit depth, timestamps, frame availability and
   annotation-coordinate conventions. Test conversion with generated examples;
   do not silently guess center versus top-left coordinates or convert approximate
   boxes into exact point truth. Preserve uncertain/ignore cases.
5. Freeze source/annotation hashes and stage-specific matching rules, then run
   the unchanged baseline. Keep actual observations, qualifications, predictions,
   and observation alerts separate. First assess input/motion-model compatibility;
   do not attribute stabilization/domain mismatch automatically to target rejection.
6. Diagnose one demonstrated failure before proposing a rule. Preserve all local
   visible-feature regressions. A small pilot cannot establish a reliable rare
   false-alarm rate or production generalization, and a public-data improvement
   cannot prove accuracy on the user's nighttime recordings.

This is evaluation-first. No neural-network training, public challenge submission,
remote upload, Jetson experiment, or production filter is implied. The existing
source allowlist is not expanded until the user approves this external-data step.

## Decision needed

Approve a metadata-first, size-capped AOT pilot, with exact image scope and size
reported before imagery transfer. Alternatively provide a different independently
annotated source. Until then preserve the working detector and all existing labels.
