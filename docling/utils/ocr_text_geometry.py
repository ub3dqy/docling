# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Read complete tilted text rows without the surrounding row envelopes."""

from math import hypot

import numpy as np
from docling_core.types.doc import CoordOrigin, Size
from docling_core.types.doc.page import BoundingRectangle


def extend_text_rect(
    rect: BoundingRectangle, *, left: float = 0, right: float = 0, vertical: float = 0
) -> BoundingRectangle:
    """Extend along the printed baseline, rather than its axis envelope."""
    dx = rect.r_x2 - rect.r_x3
    top_slope = (rect.r_y2 - rect.r_y3) / dx if dx > 0 else 0.0
    dx = rect.r_x1 - rect.r_x0
    bottom_slope = (rect.r_y1 - rect.r_y0) / dx if dx > 0 else 0.0
    return BoundingRectangle(
        r_x0=rect.r_x0 - left,
        r_y0=rect.r_y0 - bottom_slope * left + vertical,
        r_x1=rect.r_x1 + right,
        r_y1=rect.r_y1 + bottom_slope * right + vertical,
        r_x2=rect.r_x2 + right,
        r_y2=rect.r_y2 + top_slope * right - vertical,
        r_x3=rect.r_x3 - left,
        r_y3=rect.r_y3 - top_slope * left - vertical,
        coord_origin=rect.coord_origin,
    )


def rectified_text_crop(
    image: np.ndarray, rect: BoundingRectangle, page_size: Size
) -> np.ndarray | None:
    """Straighten source pixels in an OCR quadrilateral, retaining edge marks."""
    import cv2

    if image.ndim != 2 or rect.coord_origin != CoordOrigin.TOPLEFT:
        return None
    if page_size.width <= 0 or page_size.height <= 0:
        return None
    sx, sy = image.shape[1] / page_size.width, image.shape[0] / page_size.height
    points = [
        (rect.r_x3 * sx, rect.r_y3 * sy),
        (rect.r_x2 * sx, rect.r_y2 * sy),
        (rect.r_x1 * sx, rect.r_y1 * sy),
        (rect.r_x0 * sx, rect.r_y0 * sy),
    ]
    if not np.isfinite(points).all():
        return None
    width = (
        hypot(*(np.subtract(points[1], points[0])))
        + hypot(*(np.subtract(points[2], points[3])))
    ) / 2
    height = (
        hypot(*(np.subtract(points[3], points[0])))
        + hypot(*(np.subtract(points[2], points[1])))
    ) / 2
    if min(width, height) < 4:
        return None
    reduction = min(1.0, 4096 / width, 512 / height)
    w, h = max(4, round(width * reduction)), max(4, round(height * reduction))
    target = np.asarray([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    transform = cv2.getPerspectiveTransform(
        np.asarray(points, dtype=np.float32), target
    )
    straight = cv2.warpPerspective(image, transform, (w, h), borderValue=255)
    return cv2.copyMakeBorder(straight, 3, 3, 4, 4, cv2.BORDER_CONSTANT, value=255)
