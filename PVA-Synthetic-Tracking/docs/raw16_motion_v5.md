# RAW motion: full-image features and isolated per-pair flow state

This opt-in development path combines the v4 Harris-capacity correction with a
correctness workaround for reused VPI flow state. It does not change the
production default, the detector, final motion-quality limits, source masks or
target-specific settings.

## Two separately reproduced problems

1. **Truncated feature extraction.** The implicit 8,192-entry Harris output
   buffer filled before reaching lower image regions. A complete-grid output
   allocation exposes the whole image to the unchanged spatial quotas. The
   tracked-point limit remains 1,000. See [the capacity investigation](raw16_motion_v4.md).
2. **History-dependent lost-track state.** On identical RAW pixels, detected and
   selected corners were identical between isolated-pair and sequential runs,
   but the flow-status rejection count grew. A generated clean/noisy/clean test
   reproduced the effect: the clean pair retained 1,000 points initially and
   only 800 after the unrelated pair. The coordinates of surviving points were
   unchanged. This was not a reason to relax geometric quality thresholds.

Fresh CPU-initialized status arrays, VPI zero-initialized arrays, explicit stream
ordering, and freshly wrapped host buffers did not remove the state dependence
in the tested VPI 3.2.4 integration. Clearing unused VPI cache between pairs did:
the clean replay recovered the same 1,000 points exactly. The diagnostic records
both the zero-valued incoming flags and the returned flags. This establishes the
integration failure and a workaround, not which private SDK implementation detail
caused it or that every VPI version/backend has the same behavior.

## Opt-in contract

`configs/evaluation/raw16_motion_v5.json` adds only
`flow_status_policy: fresh_per_pair` to v4. That policy:

- synchronizes the estimator's stream and clears unused **process-local VPI
  cached objects** before building a new pair;
- explicitly initializes new forward-status slots to zero;
- gives backward flow a separate copy of forward flags;
- initializes and executes flow on the same stream;
- preserves all flags and failures produced by the actual tracking operations.

This is not a device, CUDA-context, operating-system or Jetson reset. It runs only
inside the isolated single-worker experiment process. Cache eviction affects
unused VPI objects process-wide, so this option is a conservative correctness
path, not a proposed concurrency/throughput optimization. Default behavior stays
unchanged. The option is guarded to the tested CUDA optical-flow backend.

VPI documents tracking status as an input/output parameter in the
[PyrLK API](https://docs.nvidia.com/vpi/3.2/group__VPI__OpticalFlowPyrLK.html).
The [stream documentation](https://docs.nvidia.com/vpi/3.2/python/build/vpi.Stream.html)
describes explicit/current/default streams and asynchronous execution. The
cache-related observations above come from the recorded Jetson experiments,
not an assertion in those documents.

## Evidence required before each media run

The generated gate requires the original 12 cases, three native-size dense/noisy
cases, and clean-pair recovery after an unrelated/lost-track case. Three frozen
seeds give 48 checks. Known-motion error is limited to 0.35 pixels; negative cases
must remain rejected. Dense positives must cover all 48 selection cells.
Recovery must return exactly the same point data, not merely another accepted fit.

An additional RAW0040 sequence check processes 63 adjacent pairs from 64 source
frames, then replays eight selected pairs in reverse order. Previous/current
points, Harris scores and round-trip errors must be byte-identical. Source
integrity is recorded. This is a development order-invariance check, not an
independent motion ground-truth measurement.

The full-native runner then checks the same 64-frame RAW0029 and RAW0040 prefixes,
plus the unchanged downstream-injected RAW0040 controls. It verifies generated
evidence, source/configuration hashes and motion module hashes before reading
media. No sealed split or holdout media is opened.

## Remaining boundaries

Neither source has authoritative airborne labels in this experiment. Retained
tracks are bounded, unlabeled hypotheses. The upper injected control was already
affected by source saturation; its position and the masks remain frozen, so
recovering the previous two controls is not reported as three out of three.

This repair increases available spatial support and isolates state; it is not a
speed improvement. The v4 point-level diagnostics already showed CPU feature
eligibility/grid preparation dominating motion time. The next performance work
should preserve these tests while batching those checks and investigating
explicitly managed reusable flow state that does not need per-pair cache eviction.
Larger labeled development sequences are still needed before deployment claims.

Results, failed diagnostic attempts and the exact tested runtime are recorded in
`results/tiny_target/raw16_motion_v5_20260915/` and the preceding v4 results folder.
