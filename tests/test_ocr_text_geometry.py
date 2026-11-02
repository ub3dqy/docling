# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Tilted rows keep their slope and do not pull ink from the next row."""

import numpy as np
import pytest
from docling_core.types.doc import BoundingBox, CoordOrigin, Size
from PIL import Image, ImageDraw

from docling.utils.ocr_text_geometry import rectified_text_crop
from docling.utils.orientation import (
    ocr_box_to_bounding_rectangle,
    ocr_quad_to_bounding_rectangle,
)


@pytest.mark.parametrize("angle", [0, 90, 180, 270])
def test_axis_aligned_ocr_quads_keep_legacy_orientation_and_offsets(angle):
    offset = BoundingBox(l=17, t=31, r=200, b=160)
    box = BoundingBox(l=20, t=30, r=120, b=55)
    before = ocr_box_to_bounding_rectangle(
        box, original_offset=offset, scale=2, orientation=angle, im_size=(300, 200)
    )
    after = ocr_quad_to_bounding_rectangle(
        [[20, 30], [120, 30], [120, 55], [20, 55]],
        original_offset=offset,
        scale=2,
        orientation=angle,
        im_size=(300, 200),
    )
    assert after == before


def test_ocr_keeps_the_true_sloped_corners_and_rectifies_only_the_printed_row():
    points = [[20, 30], [720, 65], [720, 85], [20, 50]]
    rect = ocr_quad_to_bounding_rectangle(
        points,
        original_offset=BoundingBox(l=0, t=0, r=800, b=200),
        scale=1,
        orientation=0,
        im_size=(800, 200),
    )
    assert rect.r_y0 != rect.r_y1
    image = Image.new("L", (800, 200), 255)
    draw = ImageDraw.Draw(image)
    draw.polygon([tuple(point) for point in points], fill=120)
    draw.line((20, 10, 720, 45), fill=0, width=4)
    draw.line((20, 70, 720, 105), fill=0, width=4)
    crop = rectified_text_crop(np.asarray(image), rect, Size(width=800, height=200))
    assert crop is not None
    assert 24 <= crop.shape[0] <= 28
    assert np.percentile(crop[6:-6, 8:-8], 1) >= 115
    assert np.median(crop[6:-6, 8:-8]) == 120


def test_degenerate_or_nonfinite_source_quad_is_not_warped():
    rect = ocr_quad_to_bounding_rectangle(
        [[2, 2], [2, 2], [2, 2], [2, 2]],
        original_offset=BoundingBox(l=0, t=0, r=20, b=20),
        scale=1,
        orientation=0,
        im_size=(20, 20),
    )
    assert (
        rectified_text_crop(
            np.ones((20, 20), dtype=np.uint8), rect, Size(width=20, height=20)
        )
        is None
    )
    assert rect.coord_origin == CoordOrigin.TOPLEFT
