# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Receipt amounts must retain their printed caption and row in text exports."""

# ruff: noqa: RUF001 -- Russian fiscal captions are the test source text.

import pytest
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


@pytest.mark.parametrize("tilt", [0.07, 0.14])
def test_tilted_field_uses_its_own_amount_despite_neighboring_axis_overlap(tilt):
    caption = _cell(0, "НДС", (40, 100, 120, 118))
    amount = _cell(1, "12.34", (300, 100, 400, 118))
    neighbor = _cell(2, "999.00", (300, 78, 400, 96))
    for cell in (caption, amount, neighbor):
        rect = cell.rect
        # Project every corner using the same physical paper tilt.
        cell.rect = rect.model_copy(
            update={
                "r_y0": rect.r_y0 + rect.r_x0 * tilt,
                "r_y1": rect.r_y1 + rect.r_x1 * tilt,
                "r_y2": rect.r_y2 + rect.r_x2 * tilt,
                "r_y3": rect.r_y3 + rect.r_x3 * tilt,
            }
        )
    page = _page([caption, neighbor, amount])
    join_receipt_fields(page)
    assert [cell.text for cell in page.cells] == (
        ["НДС 12.34", "999.00"] if tilt < 0.1 else ["НДС", "999.00", "12.34"]
    )


def test_curved_paper_uses_wide_text_instead_of_misleading_short_digit_baseline():
    cells = [
        _cell(0, "ИТОГО К ОПЛАТЕ С УЧЕТОМ СКИДКИ", (228, 991, 577, 1038)),
        _cell(1, "=1234.50", (760, 989.5, 858, 1019)),
        _cell(2, "БЕЗНАЛИЧНЫМИ", (233, 1025, 398, 1063)),
        _cell(3, "=1234.50", (762, 1015.5, 860, 1046.5)),
        _cell(4, "СУММА НДС 18%", (555, 1055, 683, 1076.5)),
        _cell(5, "=188.31", (765, 1044, 841, 1077)),
    ]
    cells[0].rect = BoundingRectangle(
        r_x0=229.5,
        r_y0=1037.5,
        r_x1=576.5,
        r_y1=1028.5,
        r_x2=575.5,
        r_y2=991,
        r_x3=228.5,
        r_y3=1000,
        coord_origin=cells[0].rect.coord_origin,
    )
    cells[2].rect = BoundingRectangle(
        r_x0=235,
        r_y0=1062.5,
        r_x1=397.5,
        r_y1=1055.5,
        r_x2=396,
        r_y2=1025,
        r_x3=233.5,
        r_y3=1032,
        coord_origin=cells[2].rect.coord_origin,
    )
    cells[5].rect = BoundingRectangle(
        r_x0=765.625,
        r_y0=1073.958333,
        r_x1=839.583333,
        r_y1=1076.041666,
        r_x2=840.625,
        r_y2=1045.833333,
        r_x3=766.666666,
        r_y3=1044.791666,
        coord_origin=cells[5].rect.coord_origin,
    )
    page = _page(cells)
    page.size = Size(width=1200, height=1600)
    join_receipt_fields(page)
    assert [cell.text for cell in page.cells] == [
        "ИТОГО К ОПЛАТЕ С УЧЕТОМ СКИДКИ =1234.50",
        "БЕЗНАЛИЧНЫМИ =1234.50",
        "СУММА НДС 18% =188.31",
    ]
    for cell in page.cells[1:]:
        assert cell.rect.r_y0 == cell.rect.r_y1
        assert cell.rect.r_y2 == cell.rect.r_y3


def test_two_caption_baselines_keep_a_bad_digit_quad_out_of_the_previous_row():
    total = _cell(0, "ИТОГ", (40, 90, 160, 120))
    tax = _cell(1, "СУММА БЕЗ НДС", (40, 135, 280, 165))
    cash = _cell(2, "БЕЗНАЛИЧНЫМИ", (40, 180, 260, 210))
    amount = _cell(3, "240.10", (480, 135, 600, 165))
    rect = amount.rect
    amount.rect = rect.model_copy(
        update={
            "r_y0": rect.r_y0 + (rect.r_x0 - 540) * 0.08,
            "r_y1": rect.r_y1 + (rect.r_x1 - 540) * 0.08,
            "r_y2": rect.r_y2 + (rect.r_x2 - 540) * 0.08,
            "r_y3": rect.r_y3 + (rect.r_x3 - 540) * 0.08,
        }
    )
    page = _page([total, tax, cash, amount])
    join_receipt_fields(page)
    assert [c.text for c in page.cells] == [
        "ИТОГ",
        "СУММА БЕЗ НДС 240.10",
        "БЕЗНАЛИЧНЫМИ",
    ]
