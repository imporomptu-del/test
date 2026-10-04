# v37 bounded visual review: assigned 0029 and 0055 windows

Preliminary assistant visual observations, 2026-09-24. This is **not new ground
truth**, an exhaustive negative review, an airborne-class adjudication, or an
accuracy estimate. No frozen selection, image, annotation, or experiment output
was edited. All eight assigned unmarked sheets were inspected before opening any
marked sheet. Marked sheets were then used only to locate the selected state and
read its displayed v36 retained/removed/predicted status.

The scope was exactly windows 00–03 of 0029 and 08–11 of 0055, each 13 frames at native
1:1 source sampling. Each crop is 256x192. The small field and short interval limit
physical interpretation. Apparent motion of image light is not proof of an
airborne object; absence of a separately resolved dot in a sheet is not proof of
object absence. No source AVI, NPZ pixel archive, RAW16, holdout, remote state, or
other rendered crop was opened.

## Main finding

Window 00 contains an obvious compact moving bright image feature. Its displayed
v36 state alternates between removal and retention. Therefore, newly removed
responses cannot all be treated as noise merely because the morphology gate
removed them. This feature's physical class remains unknown. Its sometimes paired
appearance and the surrounding built/lighted scene make a ground-light explanation
plausible, but these 13 frames do not establish a vehicle or airborne identity.

The other assigned crops place selected responses on or very near illuminated
linear/built structures. They do not resemble the cloud-control examples. That
is relevant source context, but it does not convert all selected responses into
authoritative nuisance labels: a real small light superposed on an edge remains
possible, particularly in the 0055 sequences.

## Window-by-window observations

| Window | Clip / frames / crop `[x,y,w,h]` | Unmarked-source observation | Marked localization and uncertainty |
| --- | --- | --- | --- |
| 00 |0029 /12–24 / `[2619,2998,256,192]` | A compact bright feature, sometimes resolving into two adjacent bright spots or a short bright streak, travels rightward across the lower part of the crop. It is especially clear at16–19 and23–24; its visibility is much less clear around20–22. Numerous relatively fixed lights and linear structures surround it. | Selected `0/bright:138`: red at18 and23; green at19,20,22,24. Thus a clearly moving image feature is intermittently rejected. I do not infer clear visibility at20/22 just because their markers are green. No verified physical class. |
| 01 |0029 /283–295 / `[2336,2970,256,192]` | A dense, nearly horizontal band of lights and vertically extended bright pixels crosses the middle. A separate much brighter moving light traverses the bottom edge early in the window; it is not the selected middle-band state. At the selected location, a separate compact moving object is not confidently resolved from the band. | `0/bright:2079`: red at289–294, predicted at295. The selected feature lies in the structured light band, not on the conspicuous bottom-edge moving light. Edge/structure response is plausible; independent point absence is not established. |
| 02 |0029 /330–342 / `[2360,2970,256,192]` | Similar dense horizontal light-band scene. Local isolated bright pixels and vertical streaks occur beside its bright section. I cannot confidently distinguish a separately moving point from changes in these small structures over the13 frames. | Same selected ID: red336,337,339,342; green338; predicted340,341. State flicker is demonstrated, but visual evidence does not settle whether the small changing signal is structural or an independent light. |
| 03 |0029 /644–656 / `[2586,2985,256,192]` | A faint localized bright feature appears near, above and left of a much brighter fixed-looking paired/rectangular light. Independent motion and persistence of that faint feature are less obvious than in00; it becomes difficult to resolve later in the window. | Same selected ID: green644–647, red648–650, predicted651–656. The marked trajectory is not independent evidence of continued visibility. Physical class and whether this is a compact light versus small structural fluctuation remain uncertain. |
| 08 |0055 /6–18 / `[2274,2998,256,192]` | Upper dense light band; darker lower scene with spaced lights and a faint approximately horizontal structure. The selected lower-region signal is very weak. Before markers, I did not confidently resolve a clean isolated moving target there. | `0/bright:3`: red12–17, green18. A faint local response appears near the lower structure, but certainty is insufficient for either a true-object or a negative label. |
| 09 |0055 /226–238 / `[1841,2939,256,192]` | A slanted luminous line below numerous fixed-looking scene lights. A small localized brightening lies on/near that line and appears to change position slightly; it is not cleanly separated from the line itself. | `0/bright:1312`: red232,233,235–238; predicted234. The signal could be a small moving light superposed on the line or a response to line structure. This crop does not resolve that ambiguity. |
| 10 |0055 /391–403 / `[1668,2904,256,192]` | Another section of a slanted luminous line, with a small local brightening/short streak near its upper-left section. Its apparent shift along the line is subtle; the source feature is not a clean isolated dot against uniform background. | Same ID: red391–403. Uniform rejection does not itself establish noise. Point-on-edge or ground-light interpretations remain plausible; physical class unknown. |
| 11 |0055 /610–622 / `[1340,2784,256,192]` | A diagonal luminous line with multiple brighter lights and local streaks. The selected area is a faint compact/short brightening immediately beside/on the line, close to a much brighter light. Modest motion along the structure is plausible but difficult to separate confidently from the edge at this sampling. | Same ID: green610,611,617–622; red612–616. This is another output-eligibility transition in a structured scene, not independently adjudicated object/no-object truth. |

Repeated track IDs in separated windows are journal identities, not independent
proof that the same physical object persists between those sampled intervals.
The intervening source frames were not reviewed here.

## Implications for the next diagnostic

- Preserve window 00 as a review caution: real compact image motion can receive
  negative point-versus-edge decisions. Do not silently add it to the positive
  reference denominator or claim an airborne miss without a separate label step.
- The 0055 line-adjacent cases justify keeping point-plus-edge evidence explicit.
  A simple edge-dominance rejection cannot establish absence of a point on that
  edge. Conversely, a small residual gain is not proof of an object.
- Local-background-relative motion is relevant, but scene structure can contain
  genuine moving ground lights. Temporal persistence alone cannot determine the
  user-requested airborne class.
- The markings demonstrate decisions, not truth. No window from this review is
  being promoted to an authoritative negative or a supported airborne positive.

## Exact files inspected

Base directory:
`/Users/romanmaksymiuk/Documents/SEAQR/outputs/seaqr_accuracy_v37_review_20260924/`

Unmarked images, inspected first in this order:

1. `00_0029_0012_0024_unmarked.png`
2. `01_0029_0283_0295_unmarked.png`
3. `02_0029_0330_0342_unmarked.png`
4. `03_0029_0644_0656_unmarked.png`
5. `08_0055_0006_0018_unmarked.png`
6. `09_0055_0226_0238_unmarked.png`
7. `10_0055_0391_0403_unmarked.png`
8. `11_0055_0610_0622_unmarked.png`

Marked images, inspected afterward in the same window order:

1. `00_0029_0012_0024_marked.png`
2. `01_0029_0283_0295_marked.png`
3. `02_0029_0330_0342_marked.png`
4. `03_0029_0644_0656_marked.png`
5. `08_0055_0006_0018_marked.png`
6. `09_0055_0226_0238_marked.png`
7. `10_0055_0391_0403_marked.png`
8. `11_0055_0610_0622_marked.png`

Metadata read: `manifest.json` top-level keys and only the eight assigned window
records; first 80 lines of `pre_view_inventory.md` for provenance context. The
directory's filenames were listed, but no other image or archived pixel array
was opened. This note records qualitative review only; it does not independently
validate the renderer or source-to-sheet pixel extraction.
