"""Causal observed-footprint variance protection; never a detection mask."""
import math
import cv2
import numpy as np


def shape_learning_mask(support, regions, margin_px):
    """Protect transported previous observed pixels plus position-noise margin.

    Unsupported/out-of-frame pixels cannot become learnable. Regions are prior
    measured shape samples, not a generated box, current response or annotation.
    Invalid shapes are rejected; a missing shape gets no circular fallback.
    """
    if support.ndim != 2 or not math.isfinite(margin_px) or not 0 < margin_px <= 16:
        raise ValueError("2D support and bounded positive uncertainty margin required")
    h, w = support.shape
    points = []
    for region in regions:
        xy = np.asarray(region["support_reference_xy"], dtype=np.float64)
        if xy.ndim != 2 or xy.shape[1] != 2 or not 0 < len(xy) <= 1024 or not np.isfinite(xy).all():
            raise ValueError("Bounded finite prior observed footprint required")
        xy = np.rint(xy)
        inside = (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
        xx, yy = xy[inside].astype(np.int64).T
        if len(xx):
            points.append(np.column_stack((xx, yy)))
    radius = math.ceil(margin_px)
    yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    disk = (xx * xx + yy * yy <= margin_px * margin_px).astype(np.uint8)
    offsets = np.column_stack(np.nonzero(disk)) - radius
    point_count = sum(len(p) for p in points)
    if point_count * len(offsets) <= support.size:
        # Dilation of a union of points is exactly the union of translated
        # structuring elements. Paint only these pixels for sparse footprints;
        # do not scan a full native image once per kernel offset.
        learn = support & True
        if points:
            xy = np.concatenate(points)
            for dy, dx in offsets:
                x, y = xy[:, 0] + dx, xy[:, 1] + dy
                inside = (x >= 0) & (x < w) & (y >= 0) & (y < h)
                learn[y[inside], x[inside]] = False
        return learn
    protected = np.zeros(support.shape, np.uint8)
    for xy in points:
        protected[xy[:, 1], xy[:, 0]] = 1
    protected = cv2.dilate(protected, disk, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return support & ~protected.astype(bool)
