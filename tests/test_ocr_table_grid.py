# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Regression coverage for dense grids and curved two-column photographs."""

import numpy as np
import pytest
from docling_core.types.doc import BoundingBox, DocItemLabel
from docling_core.types.doc.page import BoundingRectangle, TextCell
from PIL import Image, ImageDraw

from docling.datamodel.base_models import Cluster
from docling.utils.ocr_table_grid import table_from_aligned_ocr, table_from_ocr_grid

pytest.importorskip("cv2")


def _cell(index: int, text: str, box: tuple[float, float, float, float]) -> TextCell:
    return TextCell(
        index=index,
        text=text,
        orig=text,
        from_ocr=True,
        confidence=0.95,
        rect=BoundingRectangle.from_bounding_box(
            BoundingBox(l=box[0], t=box[1], r=box[2], b=box[3])
        ),
    )


@pytest.mark.parametrize("uneven_light", [False, True])
def test_dense_grid_keeps_adjacent_amounts_in_separate_rows(uneven_light: bool) -> None:
    image = (
        Image.fromarray(
            np.tile(np.linspace(70, 220, 700).astype(np.uint8), (420, 1))
        ).convert("RGB")
        if uneven_light
        else Image.new("RGB", (700, 420), "white")
    )
    draw = ImageDraw.Draw(image)
    xs = [20, 110, 350, 510, 680]
    ys = list(range(30, 391, 20))
    for x in xs:
        draw.line((x, ys[0], x, ys[-1]), fill="black", width=2)
    for y in ys:
        draw.line((xs[0], y, xs[-1], y), fill="black", width=2)
    words = []
    for row, y in enumerate(ys[:-1]):
        for column, x in enumerate(xs[:-1]):
            text = [str(1100 + row), f"Article {row}", f"{row} 123", f"{row} 456"][
                column
            ]
            words.append(_cell(len(words), text, (x + 5, y + 5, x + 65, y + 15)))
    cluster = Cluster(
        id=0, label=DocItemLabel.TABLE, bbox=BoundingBox(l=20, t=30, r=680, b=390)
    )
    table = table_from_ocr_grid(
        image, cluster, words, page_no=1, scale=1, adaptive_thresholds=uneven_light
    )
    assert table is not None
    assert (table.num_rows, table.num_cols) == (18, 4)
    if not uneven_light:
        assert not table.ocr_grid_rectified
    rows = {}
    for cell in table.table_cells:
        rows.setdefault(cell.start_row_offset_idx, []).append(cell.text)
    assert rows[11] == ["1111", "Article 11", "11 123", "11 456"]
    assert rows[12] == ["1112", "Article 12", "12 123", "12 456"]


@pytest.mark.parametrize("adaptive", [False, True])
@pytest.mark.parametrize("skewed", [False, True])
def test_short_wide_grid_keeps_the_last_financial_row(
    adaptive: bool, skewed: bool
) -> None:
    import cv2

    image = Image.new("RGB", (1120, 360), "white")
    draw = ImageDraw.Draw(image)
    xs, ys = [20, 140, 500, 590, 750, 910, 1080], [40, 90, 135, 175, 215, 255, 295]
    for x in xs:
        draw.line((x, ys[0], x, ys[-1]), fill="black", width=2)
    for y in ys:
        draw.line((xs[0], y, xs[-1], y), fill="black", width=2)
    transform = np.array(
        [[1, -0.005 if skewed else 0, 5], [0.014 if skewed else 0, 1, 3]],
        dtype=float,
    )
    image = Image.fromarray(
        cv2.warpAffine(np.asarray(image), transform, image.size, borderValue=(255,) * 3)
    )
    words = []
    for row, y in enumerate(ys[:-1]):
        for col, x in enumerate(xs[:-1]):
            left, top = transform @ np.array([x + 8, y + 8, 1])
            words.append(
                _cell(
                    len(words),
                    f"label{row}" if col < 2 else str(1100 + row) if col == 2 else "10",
                    (float(left), float(top), float(left + 35), float(top + 12)),
                )
            )
    corners = [
        transform @ np.array([x, y, 1])
        for x in (xs[0], xs[-1])
        for y in (ys[0], ys[-1])
    ]
    cluster = Cluster(
        id=0,
        label=DocItemLabel.TABLE,
        bbox=BoundingBox(
            l=min(p[0] for p in corners) - 6,
            t=min(p[1] for p in corners) - 6,
            r=max(p[0] for p in corners) + 6,
            b=max(p[1] for p in corners) + 6,
        ),
    )
    table = table_from_ocr_grid(
        image, cluster, words, page_no=1, scale=1, adaptive_thresholds=adaptive
    )
    assert table is not None
    assert (table.num_rows, table.num_cols) == (6, 6)
    assert [
        cell.text for cell in table.table_cells if cell.start_row_offset_idx == 5
    ] == ["label5", "label5", "1105", "10", "10", "10"]


@pytest.mark.parametrize("adaptive", [False, True])
def test_photo_grid_recovers_skewed_rules_without_duplicate_rows(
    adaptive: bool,
) -> None:
    import cv2

    image = Image.fromarray(
        np.tile(np.linspace(95, 210, 700).astype(np.uint8), (420, 1))
    ).convert("RGB")
    draw = ImageDraw.Draw(image)
    xs, ys = [20, 110, 350, 510, 680], list(range(20, 401, 20))
    for x in xs:
        draw.line((x, ys[0], x, ys[-1]), fill=40, width=2)
    for y in ys:
        draw.line((xs[0], y, xs[-1], y), fill=40, width=2)
    transform = np.array([[1, -0.012, 5], [0.025, 1, -8]], dtype=float)
    skewed = Image.fromarray(
        cv2.warpAffine(
            np.asarray(image), transform, image.size, borderValue=(210, 210, 210)
        )
    )
    words = []
    for row, y in enumerate(ys[:-1]):
        for col, x in enumerate(xs[:-1]):
            x0, y0 = transform @ np.array([x + 10, y + 5, 1])
            words.append(
                _cell(
                    len(words),
                    f"r{row}c{col}",
                    (float(x0), float(y0), float(x0 + 35), float(y0 + 10)),
                )
            )
    cluster = Cluster(
        id=0, label=DocItemLabel.TABLE, bbox=BoundingBox(l=20, t=12, r=681, b=410)
    )
    table = table_from_ocr_grid(
        skewed, cluster, words, page_no=1, scale=1, adaptive_thresholds=adaptive
    )
    assert table is not None
    assert (table.num_rows, table.num_cols) == (19, 4)
    assert table.ocr_grid_rectified
    assert [
        cell.text for cell in table.table_cells if cell.start_row_offset_idx == 11
    ] == [f"r11c{col}" for col in range(4)]


@pytest.mark.parametrize(
    "with_divider,missing_row", [(True, False), (False, False), (True, True)]
)
def test_curved_columns_pair_rows_and_keep_the_extra_heading_outside(
    with_divider: bool,
    missing_row: bool,
) -> None:
    image = Image.new("RGB", (1000, 750), "white")
    if with_divider:
        ImageDraw.Draw(image).line((700, 10, 720, 730), fill="black", width=2)
    lines = [_cell(0, "Value heading", (755, 10, 900, 30))]
    for row in range(12):
        y = 45 + row * 42
        if not (missing_row and row == 5):
            lines.append(_cell(len(lines), f"Subject {row}", (130, y, 600, y + 20)))
        y = 80 + row * 34
        lines.append(_cell(len(lines), str(row + 1), (755, y, 810, y + 20)))
    result = table_from_aligned_ocr(image, lines, page_no=1, cluster_id=55, scale=1)
    if not with_divider or missing_row:
        assert result is None
        return
    assert result is not None
    assert result.table.num_rows == 12
    assert 0 not in result.cell_indices
    rows = {}
    for cell in result.table.table_cells:
        rows.setdefault(cell.start_row_offset_idx, []).append(cell.text)
    assert rows[0] == ["Subject 0", "1"]
    assert rows[11] == ["Subject 11", "12"]
