# V38 second source-only review

Before V38 source-evidence extraction or scoring, the root reviewer inspected
all 13 unmarked native panels of chunk0029 frames 12–24 and the fixed lower
crop enlarged 2× by nearest-neighbor pixel repetition. No detector markers
were used to establish the feature centers. The review script independently
recomputed the eight manually bounded source-intensity centroids and source
offsets; all match annotation version 1. The original source AVI was not
decoded again for this reference review.

The luminous feature moves rightward relative to the surrounding lights.
Frames 14–15 are faint but localizable, and frames 16–19 and 23–24 show a
stronger, sometimes paired/elongated feature. The specified centers and
uncertainty radii encompass the luminous patches visible in those panels.
Frames 12–13 and 20–22 remain ambiguous for this feature: faint specks in
some panels do not establish reliable correspondence. No missing coordinate
is filled by interpolation. Correspondence across the gap is provisional.

Approve the unchanged eight visible / five ambiguous labels only for the
separate **class-unknown moving-image-feature regression**. They are not
airborne-positive labels, authoritative object centers, continuous physical
identity truth, or evidence of an empty background. The whole luminous patch,
not a particular physical object or lobe, defines each center. The review
does not establish whether these are airborne or ground-associated lights.

This is not blind review: the window was selected from V36 removals, the
native sheet includes a historical identity in its header, and both reviewers
had prior contextual exposure. Root knew the first review's labels when
checking source positions. Independently checking those source positions
does not make this an independent, untouched evaluation set. No detector
coordinate was substituted for a source annotation, and no V38 scores had
been computed. Unmatched visible frames must remain in the denominator.

The separate JSON approval receipt binds the exact annotation, review
script, and 2× source-only sheet. The extraction freeze additionally binds
this document and the first review. Any later reference correction requires
a new version and explanation rather than silently changing scored labels.
