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


def test_card_payment_is_joined_but_a_memory_card_product_is_not_a_fiscal_caption():
    payment = _cell(0, "*Плат.картой", (40, 100, 180, 125))
    amount = _cell(1, "=456.78", (260, 100, 340, 123))
    product = _cell(2, "Карта памяти", (40, 145, 180, 170))
    price = _cell(3, "321.00", (260, 145, 340, 165))
    page = _page([payment, amount, product, price])
    join_receipt_fields(page)
    assert page.parsed_page is not None
    assert [cell.text for cell in page.parsed_page.textline_cells] == [
        "*Плат.картой =456.78",
        "Карта памяти",
        "321.00",
    ]


def test_wide_amount_gutter_needs_two_other_rows_that_confirm_both_columns():
    captions = [
        _cell(0, "ИТОГО", (40, 100, 120, 125)),
        _cell(2, "НДС", (40, 140, 120, 165)),
        _cell(4, "НАЛИЧНЫМИ", (40, 180, 120, 205)),
    ]
    amounts = [
        _cell(1, "456.78", (450, 100, 530, 125)),
        _cell(3, "12.34", (450, 140, 530, 165)),
        _cell(5, "456.78", (450, 180, 530, 205)),
    ]
    isolated = _page([captions[0], amounts[0]])
    join_receipt_fields(isolated)
    assert isolated.parsed_page is not None
    assert [cell.text for cell in isolated.parsed_page.textline_cells] == [
        "ИТОГО",
        "456.78",
    ]
    page = _page([*captions, *amounts])
    join_receipt_fields(page)
    assert page.parsed_page is not None
    assert [cell.text for cell in page.parsed_page.textline_cells] == [
        "ИТОГО 456.78",
        "НДС 12.34",
        "НАЛИЧНЫМИ 456.78",
    ]


def test_wide_gutter_does_not_use_fields_in_a_different_caption_column():
    target = _cell(0, "ИТОГО", (40, 100, 120, 125))
    amount = _cell(1, "456.78", (450, 100, 530, 125))
    unrelated = [
        _cell(2, "НДС 12.34", (300, 140, 530, 165)),
        _cell(3, "НАЛИЧНЫМИ 456.78", (300, 180, 530, 205)),
    ]
    page = _page([target, amount, *unrelated])
    join_receipt_fields(page)
    assert page.parsed_page is not None
    assert page.parsed_page.textline_cells[0].text == "ИТОГО"
