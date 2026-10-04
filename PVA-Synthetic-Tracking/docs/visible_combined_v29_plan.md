# v29 — frozen four-arm 8-bit integration experiment

Compare serial v20, v26 resident GPU front only, v28 tracking only, and their
combination. Reuse both verified native libraries and frozen algorithms without
rebuilding, retuning thresholds, changing coverage/resolution, skipping frames,
changing power/clocks/services, or enabling staged v24 or native-motion v25.
RAW16 and all sealed holdout media remain out of scope. No default is promoted
automatically, even if every experiment gate passes.

## Sequence and gates

1. Verify the existing v26 generated/build/unit freeze, v28 source/replay freeze,
   runtime package and library identity. Run local and Jetson harness unit tests.
2. Run combined 128-frame smoke checks for only development 0126 and 0082. Compare
   full non-timing journals/motion against the original archives, and per-frame
   private tracking state, outputs and actual prior learning footprints against
   v27's checked baseline digests. These instrumented smokes are not timings.
3. Three fresh four-arm rounds on each 128-frame development prefix. Order:
   v20/v26/v28/combined; combined/v28/v26/v20; v28/v20/combined/v26. Preserve every
   run and outlier, stop on correctness failures, never overwrite valid receipts.
   A single process/worker owns each video run; no concurrent experiments.
4. Independently compute pooled FPS as frames/summed pipeline elapsed time. Use
   the same serial v24 admission/timestamp instrumentation in all four arms.
   Report consumer cadence, grayscale-ready-to-journal-complete and
   request-before-admission-to-journal-complete p95, both pooled and per round.
   These are replay boundaries, not exposure-to-alert/live-camera latency.
5. Combined candidate proceeds to full regressions only if both workloads have
   at least 1.20x pooled v20 FPS, every paired v20 gain is positive, none of the
   three latency boundaries has both a worse pooled p95 and regression in at
   least two of three pairs, and combined pooled FPS is no worse than either
   single-component arm. Do not relax this gate after measurement.
6. On passing, run the combined candidate on complete existing development clips
   0029/0126/0055/0082 and check all non-timing journals, motion and aggregates.
   These have archived correctness references, not fresh paired full-video FPS.
   If the prefix gate fails, do not run full regressions or tune another variant.
7. Bring compact receipts and exact journal evidence back locally; independently
   audit source/build/gate hashes, arm selection, frame/ownership accounting,
   smoke state/learning parity, all journals, timing arithmetic and gate decision.

The speed gate does not validate airborne precision/recall, generalization or
real-time operation. Keep the unchanged serial reference available. Report any
failed/incomplete run honestly rather than treating a partial experiment as done.
