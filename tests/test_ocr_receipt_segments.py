# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT
# ruff: noqa: RUF001 -- Mixed-script OCR evidence is intentional.

"""Separate neighboring papers without losing numbers or native text."""

import pytest
from docling_core.types.doc import BoundingBox, DocItemLabel, Size
from docling_core.types.doc.page import BoundingRectangle, SegmentedPdfPage, TextCell

from docling.datamodel.base_models import Cluster, LayoutPrediction, Page
from docling.utils.ocr_receipt_segments import split_receipt_lines


def _cell(index, text, left, right, *, native=False):
    return TextCell(
        index=index,
        text=text,
        orig=text,
        from_ocr=not native,
        confidence=0.95,
        rect=BoundingRectangle.from_bounding_box(
            BoundingBox(l=left, t=20, r=right, b=45)
        ),
    )


def _page(lines, words):
    page = Page(page_no=1, size=Size(width=600, height=800))
    page.parsed_page = SegmentedPdfPage.model_construct(
        textline_cells=lines,
        word_cells=words,
        char_cells=[],
    )
    page.predictions.layout = LayoutPrediction(
        clusters=[
            Cluster(
                id=0,
                label=DocItemLabel.TEXT,
                bbox=BoundingBox(l=10, t=10, r=590, b=60),
                cells=list(lines),
            )
        ]
    )
    return page


def test_company_and_bank_on_neighboring_papers_keep_separate_source_regions():
    words = [
        _cell(0, "о00", 20, 60),
        _cell(1, '"Первая', 70, 140),
        _cell(2, 'компания"', 150, 245),
        _cell(3, "БАНК", 350, 420),
    ]
    native = _cell(0, "Native text", 450, 590, native=True)
    page = _page([_cell(0, 'о00 "Первая компания" БАНК', 20, 420), native], words)
    split_receipt_lines(page)
    assert page.parsed_page is not None
    assert [c.text for c in page.cells] == [
        "о00",
        '"Первая компания"',
        "БАНК",
        "Native text",
    ]
    assert page.cells[-1] is native
    assert page.predictions.layout is not None
    assert page.predictions.layout.clusters[0].cells[-1] is native
    assert page.cells[1].rect.to_bounding_box().r == 245
    split_receipt_lines(page)
    assert len(page.cells) == 4


def test_split_does_not_append_neighboring_bank_digits_to_the_amount():
    words = [_cell(0, "513.88", 100, 170), _cell(1, "77ID", 200, 245)]
    page = _page([_cell(0, "513.8877ID", 100, 245)], words)
    split_receipt_lines(page)
    assert [c.text for c in page.cells] == ["513.88", "77ID"]
    assert page.cells[0].rect.to_bounding_box().r == 170


def test_partial_word_output_cannot_remove_an_unrepresented_part_of_a_line():
    page = _page([_cell(0, "513.8877ID", 100, 245)], [_cell(0, "513.88", 100, 170)])
    split_receipt_lines(page)
    assert [c.text for c in page.cells] == ["513.8877ID"]


def test_sparse_ctc_alignment_keeps_the_whole_sentence_for_source_rereading():
    words = [
        _cell(0, "1", 120, 126),
        _cell(1, "0U", 190, 202),
        _cell(2, "09", 250, 262),
    ]
    page = _page([_cell(0, "10U09", 120, 410)], words)
    split_receipt_lines(page)
    assert [c.text for c in page.cells] == ["10U09"]
    assert page.cells[0].rect.to_bounding_box().r == 410


def test_widely_spaced_tax_caption_stays_together_before_its_amount():
    words = [
        _cell(0, "СУММА", 20, 75),
        _cell(1, "НДС", 130, 165),
        _cell(2, "20%", 180, 210),
        _cell(3, "=43.20", 310, 400),
    ]
    page = _page([_cell(0, "СУММА НДС 20% =43.20", 20, 400)], words)
    split_receipt_lines(page)
    assert [c.text for c in page.cells] == ["СУММА НДС 20%", "=43.20"]


def test_split_keeps_the_original_spacing_inside_a_payment_abbreviation():
    words = [
        _cell(0, "*Плат.", 20, 90),
        _cell(1, "картой", 95, 160),
        _cell(2, "=120.00", 300, 400),
    ]
    page = _page([_cell(0, "*Плат.картой =120.00", 20, 400)], words)
    split_receipt_lines(page)
    assert [c.text for c in page.cells] == ["*Плат.картой", "=120.00"]


def test_geometry_only_split_cannot_enable_a_weaker_price_replacement():
    from docling.models.stages.ocr.ocr_region_refiner import (
        RegionReading,
        choose_line_number,
    )

    words = [
        _cell(0, "86.52", 100, 170).model_copy(update={"confidence": 0.55}),
        _cell(1, "77ID", 200, 245),
    ]
    page = _page([_cell(0, "86.5277ID", 100, 245)], words)
    split_receipt_lines(page)
    amount = page.cells[0]
    alternatives = [
        RegionReading("88.52", 0.99, engine, "wide")
        for engine in ("tesseract-7", "tesseract-13")
    ]
    assert (
        choose_line_number(amount.text, alternatives, confidence=amount.confidence)
        == "86.52"
    )


def test_product_price_quantity_and_tax_code_keep_the_original_expression():
    words = [
        _cell(0, "86.52", 100, 170),
        _cell(1, "3=259.5B", 200, 280),
        _cell(2, "HC10Z", 300, 360),
    ]
    text = "86.52 3=259.5B HC10Z"
    page = _page([_cell(0, text, 100, 360)], words)
    split_receipt_lines(page)
    assert [c.text for c in page.cells] == [text]


@pytest.mark.parametrize("price_first", [False, True])
def test_product_description_and_price_keep_the_original_detector_region(price_first):
    words = [
        _cell(0, "Хлеб", 20, 75),
        _cell(1, "Дом", 95, 135),
        _cell(2, "86.52", 220, 290),
    ]
    if price_first:
        words[0].text, words[1].text, words[2].text = "86.52", "Хлеб", "Дом"
    text = " ".join(w.text for w in words)
    page = _page([_cell(0, text, 20, 290)], words)
    split_receipt_lines(page)
    assert [c.text for c in page.cells] == [text]


def test_split_numeric_parts_keep_the_detector_edge_and_request_strict_reading():
    words = [_cell(0, "86.52", 112, 170), _cell(1, "77ID", 200, 240)]
    page = _page([_cell(0, "86.5277ID", 100, 245)], words)
    protected = split_receipt_lines(page)
    assert page.cells[0].rect.to_bounding_box().l == 100
    assert protected == {page.cells[0].index}
    assert page.cells[1].index not in protected


def test_numeric_identifiers_are_not_reinterpreted_as_fiscal_amounts():
    words = [_cell(0, "12345", 100, 170), _cell(1, "77ID", 200, 245)]
    page = _page([_cell(0, "1234577ID", 100, 245)], words)
    assert split_receipt_lines(page) == set()
    assert [c.text for c in page.cells] == ["1234577ID"]
