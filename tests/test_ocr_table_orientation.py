# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""A sideways grid must preserve row associations and source coordinates."""

import pytest
from docling_core.types.doc import BoundingBox, DocItemLabel
from docling_core.types.doc.document import Orientation
from docling_core.types.doc.page import BoundingRectangle, TextCell
from PIL import Image, ImageDraw

from docling.datamodel.base_models import Cluster
from docling.utils.ocr_table_grid import table_from_ocr_grid
from docling.utils.ocr_table_orientation import turn_box

pytest.importorskip("cv2")


@pytest.mark.parametrize("angle", [90, 270])
@pytest.mark.parametrize("open_labels", [False, True])
def test_sideways_grid_restores_rows_and_original_page_boxes(
    angle: int, open_labels: bool
) -> None:
    image = Image.new("RGB", (700, 420), "white")
    draw = ImageDraw.Draw(image)
    xs, ys = [20, 110, 350, 510, 680], list(range(30, 391, 20))
    for x in xs:
        if open_labels and x == xs[0]:
            continue
        draw.line((x, ys[0], x, ys[-1]), fill="black", width=2)
    for y in ys:
        draw.line((xs[0], y, xs[-1], y), fill="black", width=2)
    source_size = (float(image.width), float(image.height))
    words = []
    for row, y in enumerate(ys[:-1]):
        for column, x in enumerate(xs[:-1]):
            box = BoundingBox(l=x + 5, t=y + 5, r=x + 65, b=y + 15)
            text = [str(200 + row), f"Article {row}", str(row + 1000), str(row + 2000)][
                column
            ]
            if open_labels and column < 2:
                text = [f"Article {row}", str(200 + row)][column]
            words.append(
                TextCell(
                    index=len(words),
                    text=text,
                    orig=text,
                    confidence=0.95,
                    from_ocr=True,
                    rect=BoundingRectangle.from_bounding_box(
                        turn_box(box, angle, source_size)
                    ),
                )
            )
    original_box = BoundingBox(l=20, t=30, r=680, b=390)
    turned_box = turn_box(original_box, angle, source_size)
    cluster = Cluster(id=7, label=DocItemLabel.TABLE, bbox=turned_box)
    table = table_from_ocr_grid(
        image.rotate(angle, expand=True), cluster, words, page_no=1, scale=1
    )
    assert table is not None
    assert table.cluster.bbox == turned_box
    assert table.orientation == (
        Orientation.ROT_270 if angle == 90 else Orientation.ROT_90
    )
    selected = [cell for cell in table.table_cells if cell.start_row_offset_idx == 8]
    assert [cell.text for cell in selected] == (
        ["Article 8", "208", "1008", "2008"]
        if open_labels
        else ["208", "Article 8", "1008", "2008"]
    )
    # Provenance must still enclose the actual word on the unmodified source page.
    for cell, word in zip(selected, words[8 * 4 : 9 * 4], strict=True):
        assert cell.bbox is not None
        assert word.rect.to_bounding_box().intersection_over_self(
            cell.bbox
        ) == pytest.approx(1)


def test_quarter_turn_box_round_trip_keeps_fractional_page_coordinates() -> None:
    box = BoundingBox(l=12.75, t=5.25, r=98.125, b=45.75)
    turned = turn_box(box, 90, (210.5, 297.25))
    restored = turn_box(turned, 270, (297.25, 210.5))
    assert restored.as_tuple() == pytest.approx(box.as_tuple())
