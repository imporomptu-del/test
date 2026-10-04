"""Conservative, observed-image feature consolidation; no tracking/label input."""
import cv2
import numpy as np


def consolidate_half_height(proposals, spatial, eligible, radius=8, *, include_support=False):
    """Merge only mutually connected same-polarity half-height footprints.

    Each accepted peak keeps its original threshold evidence. A bounded connected
    footprint supplies an observed centroid, not a predicted measurement. A valley
    below either peak's half height separates them. Clipped/unsupported footprints
    are left unchanged. Pairwise compatibility prevents transitive chain merging.
    This is a provisional image-shape model, not proof of physical identity.
    """
    if spatial.ndim != 2 or spatial.shape != eligible.shape or radius < 1:
        raise ValueError("Matching 2D image/mask and positive radius required")
    h, w = spatial.shape
    footprints = []
    seeds = []
    rejected = 0
    for p in proposals:
        x, y = int(p["x"]), int(p["y"])
        seeds.append(y * w + x)
        sign = 1 if p["polarity"] == "bright" else -1
        if x - radius < 0 or y - radius < 0 or x + radius >= w or y + radius >= h:
            footprints.append(None)
            rejected += 1
            continue
        patch = sign * spatial[y - radius : y + radius + 1, x - radius : x + radius + 1]
        mask = eligible[y - radius : y + radius + 1, x - radius : x + radius + 1]
        height = float(patch[radius, radius])
        if height <= 0 or not mask[radius, radius]:
            footprints.append(None)
            rejected += 1
            continue
        # Connectivity is image evidence only; no dilation closes a real valley.
        binary = (patch >= 0.5 * height).astype(np.uint8)
        _, labels = cv2.connectedComponents(binary, connectivity=8)
        region = labels == labels[radius, radius]
        if (
            region[0].any()
            or region[-1].any()
            or region[:, 0].any()
            or region[:, -1].any()
            or np.any(region & ~mask)
        ):
            footprints.append(None)
            rejected += 1
            continue
        yy, xx = np.nonzero(region)
        footprints.append(set(((yy + y - radius) * w + xx + x - radius).tolist()))

    def compatible(i, j):
        return (
            proposals[i]["polarity"] == proposals[j]["polarity"]
            and footprints[i] is not None
            and footprints[j] is not None
            and seeds[i] in footprints[j]
            and seeds[j] in footprints[i]
        )

    groups = []
    # An eligible group must contain a first seed inside this peak's footprint.
    # Index that necessary condition, then keep the original group-order and
    # all-member compatibility checks. This does not approximate connectivity.
    groups_by_first_seed = {}
    indices_by_seed = {}
    for i, p in enumerate(proposals):
        indices_by_seed.setdefault((p["polarity"], seeds[i]), []).append(i)
    # Spatial ordering makes grouping independent of input strength ordering.
    for i in sorted(
        range(len(proposals)),
        key=lambda i: (proposals[i]["polarity"], proposals[i]["y"], proposals[i]["x"]),
    ):
        possible = set()
        if footprints[i] is not None:
            for seed in footprints[i]:
                possible.update(groups_by_first_seed.get((proposals[i]["polarity"], seed), ()))
        group = next((groups[g] for g in sorted(possible) if all(compatible(i, j) for j in groups[g])), None)
        if group is None:
            groups_by_first_seed.setdefault((proposals[i]["polarity"], seeds[i]), []).append(len(groups))
            groups.append([i])
        else:
            group.append(i)
    output = []
    merged = 0
    for group in sorted(groups, key=min):
        strongest = min(
            group,
            key=lambda i: (
                -proposals[i]["score"],
                proposals[i]["y"],
                proposals[i]["x"],
            ),
        )
        p = dict(proposals[strongest])
        if footprints[strongest] is None:
            output.append(p)
            continue
        region_set = set().union(*(footprints[i] for i in group))
        region = sorted(region_set)
        members = set(group)
        if any(
            j not in members
            for seed in region_set
            for j in indices_by_seed.get((p["polarity"], seed), ())
        ):
            # An asymmetric footprint can include a weaker/stronger neighbor
            # without mutual connectivity. Do not move either seed onto it.
            output.extend(dict(proposals[i]) for i in sorted(group))
            rejected += len(group)
            continue
        yy, xx = np.divmod(np.asarray(region, dtype=np.int64), w)
        sign = 1 if p["polarity"] == "bright" else -1
        weight = sign * spatial[yy, xx].astype(np.float64)
        # np.average performs these same float64 multiply/sum/divide operations
        # twice, including an identical denominator reduction. Reuse only the
        # denominator; do not reorder the coordinate/weight reductions.
        total_weight = weight.sum()
        if total_weight == 0:
            raise ZeroDivisionError("Weights sum to zero, can't be normalized")
        cx = float((xx * weight).sum() / total_weight)
        cy = float((yy * weight).sum() / total_weight)
        if (
            round(cy) * w + round(cx) not in region_set
            or not eligible[round(cy), round(cx)]
        ):
            # Non-convex/hollow shapes must not relocate a measurement into a
            # valley or invalid hole. Preserve every original peak instead.
            output.extend(dict(proposals[i]) for i in sorted(group))
            rejected += len(group)
            continue
        xmin, xmax, ymin, ymax = int(xx.min()), int(xx.max()), int(yy[0]), int(yy[-1])
        p["shape"] = dict(
            method="mutual_half_height_r8",
            centroid_reference_xy=[cx, cy],
            peak_reference_xy=[p["x"], p["y"]],
            member_peak_reference_xy=[
                [proposals[i]["x"], proposals[i]["y"]] for i in sorted(group)
            ],
            support_pixels=len(region),
            bbox_reference_xywh=[
                xmin,
                ymin,
                xmax - xmin + 1,
                ymax - ymin + 1,
            ],
            score_location="original accepted peak; not the centroid",
            position_quantization="nearest native pixel",
        )
        p["x"], p["y"] = round(cx), round(cy)
        if include_support:
            p["shape"]["support_reference_xy"] = np.column_stack((xx, yy)).tolist()
        output.append(p)
        merged += len(group) - 1
    return (
        output,
        dict(
            method="mutual_half_height_r8",
            input_peaks=len(proposals),
            output_features=len(output),
            merged_peak_count=merged,
            unmodified_unbounded_or_unsupported_peaks=rejected,
            radius_px=radius,
            half_height_fraction=0.5,
            no_new_threshold_seeds=True,
        ),
    )
