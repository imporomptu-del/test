# V37 bounded visual review: root observations

This is a post-extraction, provisional review, not new authoritative labels.
The pre-view selection and source hashes are frozen in
`outputs/seaqr_accuracy_v37_review_20260924` under the SEAQR workspace root.
Sixteen 256-by-192 native windows contain 202 selected crop-frames. Unmarked
source sheets were inspected before their marked counterparts. There was no
inspection of sealed holdouts, RAW16, or additional source windows.

## Directly inspected windows

| Clip / identity | Frames inspected | Source appearance and uncertainty |
| --- | --- | --- |
| 0126 / bright:201 | 6–18 | Broad, connected bright/dark cloud structure. The removed states at 12–16 follow a changing boundary/tip; frame 17 is retained at the same general structure. No separately resolved compact moving point was confidently seen. This supports a provisional cloud-response interpretation, not proof of object absence. |
| 0126 / bright:6301 | 393–405, 528–540, 660–672 | Curved cloud lobes and a dark diagonal foreground edge. Marked positions follow cloud tips/boundaries. Retention flips within similar-looking structure, especially 667–672. No confidently independent point was seen in these short windows; these are not authoritative negative labels. |
| 0082 / bright:51 | 8–20 | A compact light moves within a structured scene of many lights and a diagonal road-like band. Its appearance is compatible with a ground-vehicle light, but these crops alone do not establish physical class. Frame 14 is removed while later visible moving-light states are retained. Do not call the moving image feature noise. |
| 0082 / bright:2738 | 639–651, 652–664, 684–690 | Repeated bright structures and distant lights dominate. The selected identity crosses positions along a bright horizontal structure and later dim lights above it. No clearly distinct independent object trajectory is established by these windows. Provisional background/association-response interpretation; class remains unknown. |
| 0029 / bright:138 | 12–24, independent cross-check of second reviewer | A plainly visible compact, sometimes paired bright feature moves rightward through the lower crop, especially 16–19 and 23–24. Removed circles at 18 and 23 coincide with that visible feature; 19/20/22/24 are retained. A ground-light interpretation is plausible, but it is not an adjudicated airborne or ground label. This is direct evidence that the spatial filter can intermittently remove genuine moving image structure. |

Exact files viewed: unmarked then marked PNGs with prefixes
`04_0126_0006_0018`, `05_0126_0393_0405`, `06_0126_0528_0540`,
`07_0126_0660_0672`, `12_0082_0008_0020`, `13_0082_0639_0651`,
`14_0082_0652_0664`, `15_0082_0684_0690`, and
`00_0029_0012_0024`. PNG hashes are recorded in the export manifest.

## Consequences

The known three-point regression set is not enough to justify promotion of
V36. Its reduction in displayed states must not be equated with false-positive
removal. This selected review includes plausible clutter, real moving lights,
and unresolved cases. It does not establish precision or generalization.

A gate aimed at airborne objects needs to separate two questions: whether
there is a real moving image feature, and whether that feature is an airborne
object rather than ground activity. A local Gaussian-versus-edge score does
neither reliably by itself. Do not introduce a hard bottom-of-image exclusion:
the known encounters are near the bottom, and image location is not class.

These diagnostic observations may inform a future development protocol, but
must not silently be added to the frozen 285/28/24 reference denominators.
Any new regression case requires a separately versioned source annotation,
explicit visibility/uncertainty, and independent review. No settings or truth
files were changed during this review.
