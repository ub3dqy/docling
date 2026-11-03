# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Reuse accepted grid geometry while reading original pixels at another DPI."""

import numpy as np
import pytest
from docling_core.types.doc import BoundingBox, DocItemLabel
from docling_core.types.doc.page import BoundingRectangle, TextCell
from PIL import Image, ImageDraw

from docling.datamodel.base_models import Cluster
from docling.utils import ocr_photo_grid
from docling.utils.ocr_table_orientation import OrientedGrid, turn_box

cv2 = pytest.importorskip("cv2")


@pytest.mark.parametrize("grid_scale", [1.0, 2.0])
@pytest.mark.parametrize("angle", [0, 90, 270])
def test_accepted_grid_reads_the_same_physical_cell_at_another_dpi(
    monkeypatch: pytest.MonkeyPatch, grid_scale: float, angle: int
) -> None:
    image = Image.new("RGB", (720, 860), "white")
    draw = ImageDraw.Draw(image)
    xs, ys = [30, 190, 350, 510, 670], list(range(40, 811, 110))
    for x in xs:
        draw.line((x, ys[0], x, ys[-1]), fill="black", width=2)
    for y in ys:
        draw.line((xs[0], y, xs[-1], y), fill="black", width=2)
    marked_row, marked_col = 4, 2
    center = (430, 535)
    draw.ellipse(
        (center[0] - 5, center[1] - 5, center[0] + 5, center[1] + 5), fill="black"
    )
    transform = np.array([[1, 0.012, 0], [-0.008, 1, 8]], dtype=float)

    def source_box(left: float, top: float, right: float, bottom: float) -> BoundingBox:
        points = (
            np.array(
                [[left, top, 1], [right, top, 1], [right, bottom, 1], [left, bottom, 1]]
            )
            @ transform.T
        )
        return BoundingBox(
            l=float(points[:, 0].min()) / grid_scale,
            t=float(points[:, 1].min()) / grid_scale,
            r=float(points[:, 0].max()) / grid_scale,
            b=float(points[:, 1].max()) / grid_scale,
        )

    words = [
        TextCell(
            index=row * 4 + col,
            text=str(1000 + row * 4 + col),
            orig=str(1000 + row * 4 + col),
            from_ocr=True,
            confidence=0.95,
            rect=BoundingRectangle.from_bounding_box(
                source_box(x + 30, y + 40, x + 70, y + 60)
            ),
        )
        for row, y in enumerate(ys[:-1])
        for col, x in enumerate(xs[:-1])
    ]
    image = Image.fromarray(
        cv2.warpAffine(np.asarray(image), transform, image.size, borderValue=(255,) * 3)
    )
    cluster = Cluster(
        id=7,
        label=DocItemLabel.TABLE,
        bbox=source_box(xs[0], ys[0], xs[-1], ys[-1]),
        cells=words,
    )
    table = ocr_photo_grid.table_from_photo_grid(
        image, cluster, words, page_no=1, scale=grid_scale
    )
    assert table is not None
    if angle:
        source_size = (image.width / grid_scale, image.height / grid_scale)
        original = image.rotate(360 - angle, expand=True)
        original_cluster = cluster.model_copy(
            update={"bbox": turn_box(cluster.bbox, 360 - angle, source_size)}
        )
        view = OrientedGrid(
            image,
            cluster,
            words,
            angle,
            (original.width / grid_scale, original.height / grid_scale),
        )
        table = view.restore(table, original_cluster)
        image = original

    def reject_redetection(*args, **kwargs):
        raise AssertionError("Accepted grid must not be detected again at reading DPI")

    monkeypatch.setattr(ocr_photo_grid, "photo_grid", reject_redetection)
    high_resolution = image.resize((image.width * 2, image.height * 2))
    regions = ocr_photo_grid.accepted_grid_regions(
        high_resolution, table, scale=grid_scale * 2
    )
    assert len(regions) == 28
    for cell in table.table_cells:
        region = regions[id(cell)]
        height, width = region.shape
        middle = region[height // 4 : 3 * height // 4, width // 4 : 3 * width // 4]
        if (cell.start_row_offset_idx, cell.start_col_offset_idx) == (
            marked_row,
            marked_col,
        ):
            assert middle.min() < 30
        else:
            assert middle.min() > 220
