# Phase 18: bias-controlled RAW16 object discovery

## Goal

Search the lossless RAW16 collection for at least one visually reviewable,
compact moving object without consuming every recording as development data.
This is a discovery screen, not a sensitivity or false-alarm evaluation.

## Frozen split

The collection contains 100 clips named `chunk_0001.mkv` through
`chunk_0100.mkv`. Before opening new content, the clip identities were split
into 80 discovery clips and 20 untouched holdouts. Clips referenced during
Phases 1--17 were ineligible for the holdout. The remaining holdout identities
were sampled with Python `random.Random(75)` and then sorted.

The executable split record is:

```text
configs/evaluation/phase18_raw16_discovery_split.json
```

The holdout content must not be decoded, previewed, or used to choose scanning
or detector settings during discovery.

## Preview protocol

Every discovery clip receives the same coarse visual treatment:

- decode the complete lossless recording;
- retain source frames whose zero-based index is divisible by 15;
- convert each retained frame to a 600-pixel-wide grayscale preview;
- arrange the previews into 4 x 4 contact sheets in temporal order;
- make no detector-dependent selection at this stage.

For sheet number `s` starting at 1 and row-major tile number `k` from 0 through
15, the sampled source-frame index is:

```text
15 * ((s - 1) * 16 + k)
```

At the observed acquisition rate this is roughly one frame every five seconds.
The generator is `scripts/generate_raw16_contact_sheet.sh`. Two bounded workers
create the previews centrally on the Jetson so every reviewer receives the
same pixels without competing full-video decodes.

## Parallel review

The 80 discovery clips are divided into three disjoint batches recorded in the
split manifest. Three reviewers inspect every generated sheet and report:

- clip identity;
- sheet and tile position;
- visual reason for considering the item a compact airborne or moving object;
- whether the evidence is strong, weak, or ambiguous.

A negative coarse review means only `no visible object in sampled frames`.
Downscaling can erase a one-pixel source, and five-second sampling can skip a
short transit. It cannot certify a clip as empty.

## Native-resolution follow-up

Every shortlist item must be reopened around the mapped frame at native source
resolution. The follow-up should show consecutive raw crops, a stabilized crop,
and the target-aligned accumulation. Only persistent motion relative to the
scene can advance to a candidate label. Static enclosure highlights, cloud
structure, roof edges, insects on the window, and compression/display artifacts
remain clutter categories.

If no convincing item survives, the result is that this coarse screen found no
visible example. It does not prove that the collection contains no point
targets; a controlled capture remains the authoritative route to sensitivity
evidence.

## Outcome

The complete discovery screen finished with the frozen protocol:

- all 80 discovery clips generated successfully with no missing or unexpected
  clip identities;
- three reviewers inspected 329 contact sheets across the disjoint 27-, 27-,
  and 26-clip batches;
- no clip or tile was shortlisted as a plausible compact airborne or
  independently moving object;
- no native-resolution temporal follow-up was triggered because no item passed
  the shortlist gate;
- none of the 20 holdout clips was generated, copied, decoded, or reviewed.

Visible content was dominated by evolving cloud structure, saturated sky, and
fixed enclosure or optical features. Occasional small marks were rejected when
they remained fixed in image coordinates or moved with the surrounding cloud
field instead of tracing an independent path.

This is a sampling-limited negative result. The approximately five-second
stride can miss a short transit, and shrinking a 4784 x 3200 source frame to
600 pixels wide can erase a one-pixel source target. Therefore Phase 18 does
not establish that the discovery clips are empty and does not provide a real
target sensitivity measurement. The machine-readable result is:

```text
results/tiny_target/phase18/phase18_discovery_summary.json
```

The evidence boundary remains a controlled real-target capture with
synchronized truth. A dense native-resolution computational scan is a valid
separate discovery experiment, but its unlabeled responses must remain review
workload rather than ground truth. The sealed holdout should be opened only
after that scan and its review policy are frozen.
