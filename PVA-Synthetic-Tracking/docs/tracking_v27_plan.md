# v27: exact heavy-scene tracking optimization

Research only; v20 remains the reference and v26 remains unpromoted. No RAW16,
sealed holdout media, threshold/coverage/resolution/cadence changes, installs,
reboots, service/clock/power changes, or target-specific tuning. One Jetson
experiment worker at a time. Preserve all prior valid artifacts.

First replay the saved 128-frame development journals for 0126 and 0082 using
the unchanged v20 tracker/geometry. Profile only tracking, not JSON parsing or
comparison, and verify every track and tracking metric against the journals.
Record complete normalized private tracker-state and learning-footprint digests
so candidate replay must preserve more than just displayed centers. These are
instrumented diagnostics, not end-to-end FPS or independent accuracy samples.

Choose one bounded implementation change from that evidence. Preserve all
measurements, arithmetic/contraction order, gates and rejected-pair counters,
tie/order rules, association policies/audits, covariance/state, births/deaths,
quality evidence and measured-shape learning protection. Use an opt-in adapter
with exact source/anchor guards; do not modify the frozen runtime. Unsupported
domains retain the reference path; native errors must not silently approximate.

Before new media: generated primitive/complete tracker tests, reset/miss/tie/
capacity cases and both saved real-data replays must match. Repeat unprofiled
tracking replays in alternating order to establish whether a meaningful change
exists. Freeze candidate code/build/config/protocol before media measurements.

If replay evidence warrants media testing, compare serial v20 with the single
tracking optimization only: same original GPU detector, same v17 learning and
v24 reference-only timestamp/admission bookkeeping in both arms. Do not fold
the unpromoted v26 front end into this experiment or misattribute combined gains.
Use two 128-frame candidate smokes and three alternating pairs on each existing
0126/0082 prefix. Full candidate regressions on 0029/0126/0055/0082 only after
exact outputs and the predeclared adoption gate pass: >=20% pooled FPS gain on
heavy 0126, no pooled throughput regression on light 0082, all heavy paired
ratios >1, and no consistent p95 regression at cadence, grayscale-ready age or
pre-admission request age on either workload. Consistent means worse pooled
p95 and at least two of three paired ratios worse. This targeted tracking trial
does not demand a 20% gain where tracking was only about12ms/frame.

If the chosen implementation fails exactness or replay benefit, stop before
media and preserve the negative result. If the media adoption gate fails, do
not promote or run full clips. Independently recheck transferred evidence and
report the actual outcome, limitations and next step. No production, general
airborne accuracy or live-real-time claim. Any v26 combination is a later,
separate experiment, not part of this trial.
