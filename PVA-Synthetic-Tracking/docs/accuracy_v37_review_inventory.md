# V37 bounded source-review inventory (not new truth labels)

## Scope and provenance

This inventory uses only the four permitted 8-bit development clips, 0029,
0126, 0055, and 0082. It is preparation for follow-up review, not a classifier
change or an accuracy claim. No new video decoding, source AVI reads, RAW16,
holdout media, SSH, or remote changes were performed. No frozen V36 artifacts
were changed. In particular, the newly rejected samples below have **not**
been visually reviewed or assigned physical-class labels.

Protocols read in full:

- `docs/phase20_airborne_accuracy_protocol.md`
- `docs/accuracy_v36_plan.md`
- `docs/accuracy_v36_context_plan.md`
- `docs/accuracy_v36_full_context_plan.md`
- `results/tiny_target/accuracy_v36_20260924/README.md`

Data read were `context_01/observations.json`, `context_01/summary.json`, and
the four `full_context_01/<clip>_decisions.jsonl` files beneath
`results/tiny_target/accuracy_v36_20260924/`. Existing control manifests and
exactly the four contact sheets listed below were also inspected. These
contact sheets contain native 192-by-192 source crops for 13 frames each;
they are not new extractions. Archived patch arrays were not opened.

All control descriptions remain provisional assistant-reviewed nuisance
descriptions. They do not establish that all pixels in a window are
object-free. The protocol still has no verified airborne positives or
exhaustively adjudicated negatives; output-workload reduction must not be
reported as precision or an operational false-alarm rate.

## The 16 accepted measured control responses

The seven existing control windows contained 70 baseline-qualified measured
responses; V36 retained 16. All 16 survivors are in four cloud/edge control
windows from 0126, and belong to five current identities. The building
controls contained no selected baseline-qualified measured responses, so
they do not demonstrate successful rejection. The other cloud window,
frames 449–461, changed from three measured responses to zero.

The old green rings in the contact sheets identify historical V4 tracks,
**not the current V36 identities**. Current coordinates must be compared
with the full sheet crop rather than assuming the old ring marks them.

| Existing sheet / historical ID | Frames and native crop `[x,y,w,h]` | Current V36 ID | Surviving measured frames | Baseline → retained in whole window |
| --- | --- | --- | --- | --- |
| `bright_4510.png` | 289–301; `[2674,1671,192,192]` | `0/bright:4943` | 299, 300, 301 | 10 → 3 |
| `dark_5539.png` | 324–336; `[2251,1670,192,192]` | `0/bright:5251` | 324, 325, 329, 331, 333 | 26 → 8, shared with next row |
| `dark_5539.png` | same | `0/dark:5321` | 325, 332, 336 | shared with preceding row |
| `dark_7927.png` | 478–490; `[2527,1715,192,192]` | `0/bright:7793` | 483, 487, 489 | 23 → 3 |
| `dark_8496.png` | 522–534; `[2757,2032,192,192]` | `0/bright:7692` | 533, 534 | 8 → 2 |

All sheet paths are under
`results/tiny_target/phase20/clutter_review_20260913/chunk0126/dense_controls/`.

### What the existing source sheets actually show

The four sheets show broad, curved bright/dark cloud formations, connected
edges, and lobes/notches changing position and shape through the 13-frame
sequences. I did not confidently identify a separately resolved independent
point around the surviving response positions in these sheets. This
supports the existing provisional cloud/edge interpretation, but it does
not create authoritative negative labels or rule out a point superposed
on an edge. The lower-right sheet is close to the bottom of its crop;
limited surrounding context remains a reason for caution.

For the `dark_5539` sheet, current bright responses begin near crop-local
x=170 rather than the historical ring near the crop center. The nearby
bright and dark V36 responses can describe different sides of one changing
cloud structure; two IDs are not evidence for two independent objects.

### Exact surviving-state locations and logged diagnostic values

Coordinates below are actual measured source coordinates rounded to 0.001
pixel for readability. Full precision is retained in `observations.json`.
Times in seconds are frame index / 10 for these fixed 10 Hz clips. Margin is
point gain fraction minus edge gain fraction; conditional gain is the
point-after-best-edge fraction, not an eligibility threshold.

| Frame | Time (s) | Current ID | Source x, y | Point-minus-edge margin | Conditional gain |
| --- | --- | --- | --- | --- | --- |
| 299 | 29.9 | `bright:4943` | 2745.447, 1813.081 | 0.278038 | 0.361297 |
| 300 | 30.0 | `bright:4943` | 2742.441, 1812.073 | 0.070070 | 0.261996 |
| 301 | 30.1 | `bright:4943` | 2742.445, 1811.072 | 0.224304 | 0.322607 |
| 324 | 32.4 | `bright:5251` | 2421.398, 1784.035 | 0.150505 | 0.357218 |
| 325 | 32.5 | `bright:5251` | 2420.404, 1781.032 | 0.054265 | 0.285390 |
| 329 | 32.9 | `bright:5251` | 2414.410, 1778.009 | 0.026256 | 0.282636 |
| 331 | 33.1 | `bright:5251` | 2409.397, 1776.003 | 0.003188 | 0.322123 |
| 333 | 33.3 | `bright:5251` | 2407.398, 1771.998 | 0.153960 | 0.309806 |
| 325 | 32.5 | `dark:5321` | 2426.404, 1792.032 | 0.062688 | 0.418507 |
| 332 | 33.2 | `dark:5321` | 2416.395, 1783.000 | 0.141188 | 0.598443 |
| 336 | 33.6 | `dark:5321` | 2411.412, 1774.989 | 0.135243 | 0.310598 |
| 483 | 48.3 | `bright:7793` | 2616.570, 1832.767 | 0.004246 | 0.505499 |
| 487 | 48.7 | `bright:7793` | 2612.557, 1827.752 | 0.001191 | 0.560671 |
| 489 | 48.9 | `bright:7793` | 2610.549, 1824.759 | 0.095983 | 0.522489 |
| 533 | 53.3 | `bright:7692` | 2925.611, 2139.725 | 0.079109 | 0.454585 |
| 534 | 53.4 | `bright:7692` | 2925.609, 2140.724 | 0.017692 | 0.259922 |

All 16 fits choose sigma=3 pixels, the widest member of the frozen
sigma=1,2,3 point bank. All 16 conditional fits also choose sigma=3. Their
positive margins span 0.001191–0.278038, and conditional gains span
0.259922–0.598443. These observations are consistent with broad or curved
structure fitting a point template better than the restricted straight-edge
bank. They are not evidence that a sigma cutoff or a newly selected margin
would generalize. In particular, conditional point-after-edge gain alone
does not separate these provisional nuisances.

## Deterministic newly rejected sample: selection before viewing source

For each allowed clip, group the frozen full-context logged states by
`(segment, track_id)`. Restrict to actual measurements that were baseline
qualified and newly rejected by the output gate. An eligible identity must
have at least five such measurements. Select:

1. The earliest identity by first rejected measured frame; break ties by
   `(segment, track_id)`.
2. The longest identity by temporal span between first and last rejected
   measured frames; break ties by rejected-measurement count (descending),
   first rejected frame (ascending), then `(segment, track_id)`.

Deduplicate identities within a clip. Here all eight selected identities
are distinct. "Longest" explicitly means temporal span, not the highest
number of rejected measurements and not a continuous rejection duration.
All eight happen to be bright-polarity IDs; no truth-driven polarity
rebalancing was applied. The selection used no physical-class labels,
reference-match outcome, or source viewing.

The counts of eligible identities were 204, 330, 123, and 28 for 0029,
0126, 0055, and 0082, respectively. This purposive diagnostic sample is
not a random sample for estimating population precision.

| Clip / selection | Segment / ID | Rejected measured span (frames; seconds) | Rejected / accepted measured states over full logged identity | First → last rejected source coordinate |
| --- | --- | --- | --- | --- |
| 0029 earliest | `0/bright:138` | 18–48; 1.8–4.8 | 11 / 11 | (2706.049,3147.004) → (3118.945,3053.010) |
| 0029 longest | `0/bright:2079` | 289–650; 28.9–65.0 | 20 / 270 | (2463.515,3065.113) → (2717.598,3080.081) |
| 0126 earliest | `0/bright:201` | 12–16; 1.2–1.6 | 5 / 1 | (2904.000,1740.972) → (2897.999,1711.967) |
| 0126 longest | `0/bright:6301` | 399–666; 39.9–66.6 | 39 / 13 | (3067.505,1795.880) → (2922.485,1619.538) |
| 0055 earliest | `0/bright:3` | 12–44; 1.2–4.4 | 19 / 15 | (2390.918,3153.996) → (2541.791,3153.961) |
| 0055 longest | `0/bright:1312` | 232–616; 23.2–61.6 | 293 / 120 | (1971.230,3035.916) → (1468.713,2880.787) |
| 0082 earliest | `0/bright:51` | 14–29; 1.4–2.9 | 5 / 6 | (3245.018,2994.028) → (3258.014,2966.053) |
| 0082 longest | `0/bright:2738` | 645–690; 64.5–69.0 | 15 / 0 | (2261.191,3064.131) → (2284.243,3047.143) |

All eight remain **unreviewed / physical class unknown**. These are newly
removed measured responses, not eight established false positives or eight
wholly removed physical objects. For example, 0029 `bright:2079` was still
accepted on 270 measured states; 0055 `bright:1312` was accepted on 120.
Both deserve careful source review for intermittent loss of a real point
on an edge. Log duration and displacement alone establish neither a real
object nor an airborne identity.

| Clip / ID | Rejected margin min / median / max | Rejected conditional gain min / median / max |
| --- | --- | --- |
| 0029 `bright:138` | −0.232083 / −0.061607 / −0.004180 | 0 / 0.056423 / 0.345600 |
| 0029 `bright:2079` | −0.301614 / −0.154249 / −0.007010 | 0.011578 / 0.065322 / 0.152630 |
| 0126 `bright:201` | −0.489150 / −0.298967 / −0.238058 | 0.011292 / 0.127754 / 0.219371 |
| 0126 `bright:6301` | −0.677734 / −0.172168 / −0.001632 | 0.018858 / 0.148967 / 0.531408 |
| 0055 `bright:3` | −0.177008 / −0.030121 / −0.000729 | 0.003633 / 0.023743 / 0.267816 |
| 0055 `bright:1312` | −0.154491 / −0.034360 / −0.000742 | 0.001568 / 0.080473 / 0.202958 |
| 0082 `bright:51` | −0.156735 / −0.066201 / −0.039339 | 0.000856 / 0.011543 / 0.033943 |
| 0082 `bright:2738` | −0.317864 / −0.109851 / −0.072857 | 0.003326 / 0.026070 / 0.133147 |

These are diagnostic model coefficients, not classifications. A fitted
point amplitude can exceed 255 DN after background/edge projection (one
selected 0029 fit is 265 DN); that is not an observed out-of-range 8-bit
pixel or proof of a saturated object.

## Proposed bounded native follow-up windows — not yet decoded

For each earliest identity, take the first rejected measured frame ±6.
For each longest identity, take the first, middle rejected-sample index
(`len(rejected)//2`), and last rejected frame, each ±6. Clip to valid frame
indices. Center a fixed 256-by-192 native crop on the bounding midpoint of
that ID's qualified measured positions, both accepted and rejected, within
the window. Round the midpoint with `floor(value+0.5)`, subtract the crop
half-size, and clamp the origin to the 4784-by-3190 source bounds. All
selected measured positions fit these crops.

This yields 16 small windows, totaling 202 crop-frames. It bounds review
work while exposing early, middle, and late rejection behavior; it does
not cover entire identities or establish absence outside the selected
space/time. Some bottom crops are clamped to the image boundary, which is
a review geometry constraint, never a detection mask.

| Clip | Selection / ID | Inclusive frames | Native crop `[x,y,w,h]` |
| --- | --- | --- | --- |
| 0029 | earliest `bright:138` | 12–24 | `[2619,2998,256,192]` |
| 0029 | longest `bright:2079`, first | 283–295 | `[2336,2970,256,192]` |
| 0029 | longest `bright:2079`, middle | 330–342 | `[2360,2970,256,192]` |
| 0029 | longest `bright:2079`, last | 644–656 | `[2586,2985,256,192]` |
| 0126 | earliest `bright:201` | 6–18 | `[2769,1625,256,192]` |
| 0126 | longest `bright:6301`, first | 393–405 | `[2938,1697,256,192]` |
| 0126 | longest `bright:6301`, middle | 528–540 | `[2860,1624,256,192]` |
| 0126 | longest `bright:6301`, last | 660–672 | `[2794,1524,256,192]` |
| 0055 | earliest `bright:3` | 6–18 | `[2274,2998,256,192]` |
| 0055 | longest `bright:1312`, first | 226–238 | `[1841,2939,256,192]` |
| 0055 | longest `bright:1312`, middle | 391–403 | `[1668,2904,256,192]` |
| 0055 | longest `bright:1312`, last | 610–622 | `[1340,2784,256,192]` |
| 0082 | earliest `bright:51` | 8–20 | `[3109,2899,256,192]` |
| 0082 | longest `bright:2738`, first | 639–651 | `[2139,2968,256,192]` |
| 0082 | longest `bright:2738`, middle | 652–664 | `[2159,2963,256,192]` |
| 0082 | longest `bright:2738`, last | 684–690 | `[2157,2951,256,192]` |

If extraction is separately authorized, review unmarked native source
first. A separate overlay may subsequently identify the selected ID and
all of its accepted/rejected measured states; predictions must remain
visually distinct. Keep ambiguous physical identity and ambiguous source
appearance as unknown. Existing four control sheets already provide 52
additional crop-frames; a future unmarked export would remove their old
ring ambiguity without expanding the time/space scope.

## Implications for the next hypothesis

- Broad curved cloud structure can defeat a local point-versus-straight-edge
  comparison. The maximum-width fit pattern is a hypothesis clue, not a
  license to shrink the accepted point bank using these known controls.
- A point can coexist with an edge. A negative exclusive margin is not
  proof of no point; the synthetic point-on-edge limitation in V36 remains
  relevant to the unreviewed newly rejected sample.
- Conditional point-after-edge gain alone is not a demonstrated remedy:
  the surviving control responses also have substantial conditional gain.
- Local surrounding structure and its temporal evolution are sensible
  next diagnostic dimensions. Any temporal proposal must remain causal,
  use real measurements, preserve intermittent/turning/slow reference
  points, and avoid reintroducing failed recent-hit or recent-excursion
  assumptions as a stronger eligibility requirement.
- The initial evaluation should retain all existing strict dense, pilot,
  and anchor denominators, including baseline misses. Newly reviewed
  diagnostics must not silently become independent held-out validation.

## Input SHA-256 inventory

Paths in the first five rows are relative to
`results/tiny_target/accuracy_v36_20260924/`; PNG rows are relative to the
control-sheet directory given above.

| Input | SHA-256 |
| --- | --- |
| `context_01/observations.json` | `1b989e6d09918358040c924e1aabf421f886e7ebf833d526ab25c842d0e52ec7` |
| `full_context_01/0029_decisions.jsonl` | `03aa65a6f4e377f34cbd4d6e2bc1bb4ed55695e705a703df41347148b157fa30` |
| `full_context_01/0126_decisions.jsonl` | `a32aad01b74f0150dcbb1d54c6502068a02ff097480095684ea9855c8151b832` |
| `full_context_01/0055_decisions.jsonl` | `fb2ff2e107f2163f92f2e614ac350637bfbad562cde2800c31505223ee09b337` |
| `full_context_01/0082_decisions.jsonl` | `5a9b73071e9c46d20ad8cded273fde34f051a6612557072534ec979d036ab90b` |
| `bright_4510.png` | `967faf2ff5e779cc53203de55f4e0716f81a739fe66fc701fc3ed8192c997ff5` |
| `dark_7927.png` | `343ee54b536677f467de657d2f8fa53ea7ac5eafd0da3a7a07a088bafac822f1` |
| `dark_8496.png` | `dd850b7510263219441396b1053a4306104c393aa5011b6a396e3954a8713099` |
| `dark_5539.png` | `1296ef060bf33f6146bcae538d9b9f16de46074a791c2c0cceed2409604e3634` |
