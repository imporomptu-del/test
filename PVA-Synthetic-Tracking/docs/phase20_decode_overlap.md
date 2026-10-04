# Bounded visible-video decode overlap

`VisibleConfig.frame_decode_execution` defaults to `sequential`. The opt-in
`prefetch_one` mode transfers the capture to one producer thread after metadata
queries. That thread alone calls `read`, BGR-to-gray conversion, and `release`.
The consumer alone owns motion estimation, CUDA buffers, detection, tracking,
learning feedback, timestamps and journals.

The producer waits for an empty slot **before** decoding. Queued plus in-flight
frames are bounded to one, excluding the current consumer frame, codec-internal
storage and normal algorithm state. Conversion has no `dst` buffer: each gray
image owns independent storage even if a decoder reuses its BGR buffer. Frames
are neither skipped nor reordered. A prefix limit never decodes past its bound.

Initialization failures close the capture on the caller. After ownership is
transferred, shutdown signals cancellation and joins the owner before success is
reported. A blocked native decode cannot be safely cancelled cross-thread: a
five-second join timeout fails closed rather than releasing a live capture or
reporting successful cleanup. The existing experiment subprocess timeout remains
the outer safety bound; this is not a hard-real-time source implementation.

Per-frame `timings_ms.decode` and `grayscale` are work durations on the decoder
owner. `decode_wait` is consumer wait for a prefetched frame (zero in sequential
mode). These overlapping durations **must not be summed as elapsed latency**.
`processed_fps` includes thread start, reading, processing, journaling, drain and
join; source hashing and pipeline initialization remain outside the timed loop.

Launch provenance records the execution contract. Completed reports additionally
record frame/read counts, the observed buffer bound and successful join/release.
The exact-output comparator validates that provenance and compares every
non-timing journal field. A changed decode mode never permits changed geometry,
scores, candidates, tracks, coverage, policy, cadence or feedback.

Validation uses synthetic ownership/lifecycle/failure tests and a synthetic MJPEG
closed-loop comparison, followed by unchanged development clips 0029, 0126, 0055
and 0082. The original accuracy reference and latest accepted GPU/native binaries
are kept separate from timing evidence. No sealed holdout media, thresholds or
object-specific rules are involved. Paired prefix timings use the same current
Python package with sequential versus prefetch configuration, avoiding a package
version confound. No speed or accuracy improvement is assumed before measurement.
