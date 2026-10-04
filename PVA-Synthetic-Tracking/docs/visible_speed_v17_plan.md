# v17: 8-bit-only execution efficiency

RAW16 work is paused. Read/decode only existing development AVIs 0029, 0126,
0055 and 0082. Never access sealed holdouts. Use the archived v13 visible reuse
branch as the reference: unchanged native geometry, thresholds, quotas, cadence,
coverage, track rules, GPU libraries and decoder contract. No production/default
promotion, installs, live capture, power/clock/service changes, or reboot.

1. Verify archived source/config/package/library identities and profile the
   existing complete 8-bit pipeline on the first 128 frames of 0126 and 0082.
   Use cProfile plus named host-call timings; nested spans are not additive and
   overlapped decode time is not summed into consumer wall time. Profile runs
   are not throughput measurements. Preserve complete non-timing journal parity.
2. Choose one execution-only hotspot from those observations. Freeze a separate
   candidate specification before camera trials. Test its equivalence against
   the original implementation using generated cases, including edge cases.
   Keep rejected experiments and failures; do not loosen equality gates.
3. If generated gates pass, run fresh reversed-order end-to-end prefix pairs,
   then full candidate regression on all four existing development clips. Stop
   on unexplained output changes. Preserve timings even when slower. Compare all
   non-timing journals (candidate scores/order, coverage, motion and tracks),
   motion identities, decode lifecycle, and full-run aggregate outputs.
4. Independently verify copied evidence locally and report actual end-to-end
   throughput, service-time distribution, limits and the next bottleneck. Saved
   journal replay is component evidence only, never a full-pipeline FPS claim.

The reviewed objects establish working examples, not general airborne recall or
false-alarm rate. Existing misses/ID splits must remain visible. No frame
skipping, reduced image coverage, looser threshold or label-specific rule is an
execution-only speed improvement. Camera-to-alert latency is not established by
offline throughput or per-frame service times. Ten FPS remains provisional.
