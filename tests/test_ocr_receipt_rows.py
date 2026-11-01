# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Receipt amounts must retain their printed caption and row in text exports."""

from docling_core.types.doc import BoundingBox, DocItemLabel, Size
from docling_core.types.doc.page import (
    BoundingRectangle,
    SegmentedPdfPage,
    TextCell,
)

from docling.datamodel.base_models import Cluster, LayoutPrediction, Page
from docling.utils.ocr_receipt_rows import is_receipt_page, join_receipt_fields


def _cell(index: int, text: str, box: tuple[float, float, float, float]) -> TextCell:
    left, top, right, bottom = box
    return TextCell(
        index=index,
        text=text,
        orig=text,
        confidence=0.9,
        from_ocr=True,
        rect=BoundingRectangle.from_bounding_box(
            BoundingBox(l=left, t=top, r=right, b=bottom)
        ),
    )


def _page(cells: list[TextCell]) -> Page:
    page = Page(page_no=1, size=Size(width=600, height=800))
    page.parsed_page = SegmentedPdfPage.model_construct(textline_cells=cells)
    page.predictions.layout = LayoutPrediction(
        clusters=[
            Cluster(
                id=i, label=DocItemLabel.TEXT, bbox=c.rect.to_bounding_box(), cells=[c]
            )
            for i, c in enumerate(cells)
        ]
    )
    return page


def test_receipt_detection_uses_distinct_fiscal_cues_even_with_latin_ocr_letters() -> (
    None
):
    assert is_receipt_page(["CYM HAC 10%", "БЕЗНАЛИЧНЫМИ"])
    assert not is_receipt_page(["CYM HAC 10%", "CYM HAC 10%"])
    assert not is_receipt_page(["Ordinary table", "10721"])


def test_small_amount_is_kept_with_caption_once_even_with_intervening_noise() -> None:
    caption = _cell(0, "ИТОГО", (40, 100, 180, 130))
    noise = _cell(1, ":", (200, 108, 204, 112))
    amount = _cell(2, "=123.45", (260, 104, 340, 122))
    next_caption = _cell(3, "НАЛИЧНЫМИ", (40, 145, 180, 170))
    next_amount = _cell(4, "200.00", (260, 145, 340, 165))
    page = _page([amount, caption, noise, next_caption, next_amount])
    join_receipt_fields(page)
    assert page.parsed_page is not None
    assert [c.text for c in page.parsed_page.textline_cells] == [
        "ИТОГО =123.45",
        ":",
        "НАЛИЧНЫМИ 200.00",
    ]
    assert page.predictions.layout is not None
    assert [
        c.text for cluster in page.predictions.layout.clusters for c in cluster.cells
    ] == ["ИТОГО =123.45", ":", "НАЛИЧНЫМИ 200.00"]
    assert page.parsed_page.textline_cells[0].rect.to_bounding_box() == BoundingBox(
        l=40, t=100, r=340, b=130
    )
    join_receipt_fields(page)
    assert len(page.parsed_page.textline_cells) == 3


def test_receipt_row_join_does_not_take_neighboring_or_native_amounts() -> None:
    caption = _cell(0, "ИТОГО", (40, 100, 180, 125))
    other_row = _cell(1, "123.45", (260, 130, 340, 150))
    native = _cell(2, "200.00", (260, 102, 340, 123)).model_copy(
        update={"from_ocr": False}
    )
    page = _page([caption, other_row, native])
    join_receipt_fields(page)
    assert page.parsed_page is not None
    assert [c.text for c in page.parsed_page.textline_cells] == [
        "ИТОГО",
        "123.45",
        "200.00",
    ]
