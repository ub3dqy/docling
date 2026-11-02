# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Caption verification cannot invent fiscal words, amounts or tax rates."""

# ruff: noqa: RUF001 -- Deliberately damaged Cyrillic OCR labels.

import pytest
from docling_core.types.doc import BoundingBox, CoordOrigin, Size
from docling_core.types.doc.page import BoundingRectangle, SegmentedPdfPage, TextCell

from docling.datamodel.base_models import Page
from docling.models.stages.ocr.ocr_receipt_text import (
    choose_amount_marks,
    choose_caption,
    remove_duplicate_fiscal_marks,
    retain_company_quotes,
)
from docling.models.stages.ocr.ocr_region_refiner import RegionReading


def test_company_merge_keeps_a_quote_observed_by_an_independent_reader():
    original = '"Северная компания Б17"'
    selected = RegionReading(
        'ООО "Северная компания Б-17', 0.98, "cyrillic-v6-medium", "source"
    )
    other = RegionReading(
        'о00 "Северная компания б 17"', 0.89, "cyrillic-v6-tiny", "source"
    )
    retained = retain_company_quotes(original, selected, [other])
    assert retained.text == selected.text + '"'
    assert retained.confidence == other.confidence
    assert retain_company_quotes(original, selected, []).text == selected.text
    assert (
        retain_company_quotes(original.rstrip('"'), selected, [other]).text
        == selected.text
    )
    for candidate in (
        RegionReading(other.text, 0.99, selected.engine, "clean"),
        RegionReading(other.text.replace("17", "18"), 0.99, other.engine, "source"),
    ):
        assert (
            retain_company_quotes(original, selected, [candidate]).text == selected.text
        )


@pytest.mark.parametrize("suffix", ["2", "7"])
@pytest.mark.parametrize("rate", [2, 5, 18, 22])
def test_percent_glyph_repair_needs_an_independently_observed_same_rate(rate, suffix):
    original = f"СУННА НС {rate}{suffix}"
    readings = [
        RegionReading(f"СУММА НДС {rate}%", 0.98, "cyrillic-v6-medium", "source"),
        RegionReading(f"СУННА НДС {rate}%", 0.7, "tesseract", "clean"),
    ]

    def known(word):
        return False

    assert choose_caption(original, readings[:1], known) is None
    selected = choose_caption(original, readings, known)
    assert selected is not None and selected.text == f"СУММА НДС {rate}%"
    assert choose_caption(f"СУННА НС {rate + 1}7", readings, known) is None
    assert choose_caption(f"СУННА НС {rate}7.00", readings, known) is None
    assert choose_caption(f"СУННА НС {rate}7%", readings, known) is None
    assert choose_caption(f"код {rate}7", readings, known) is None


def test_known_cash_caption_cannot_be_replaced_with_cashless_caption():
    readings = [
        RegionReading("БЕЗНАЛИЧНЫМИ", 0.99, "cyrillic-v6-medium", "source"),
        RegionReading("БЕЗНАЛИЧНЫМИ", 0.99, "cyrillic-v6-tiny", "source"),
    ]
    assert choose_caption("НАЛИЧНЫМИ", readings, lambda word: True) is None


def test_amount_marker_needs_two_models_and_preserves_values_and_negative_sign():
    readings = [
        RegionReading("=127.30", 0.99, engine, "source")
        for engine in ("cyrillic-v6-medium", "cyrillic-v6-tiny")
    ]
    assert choose_amount_marks("+127.30", readings).text == "=127.30"
    for original in (
        "-127.30",
        "127.31",
        "127.300",
        "12730",
        "20% 127.30",
        "2*127.30",
        "*127.30",
    ):
        assert choose_amount_marks(original, readings) is None
    assert choose_amount_marks("+127.30", readings[:1]) is None


def _cell(index, text, box, confidence=0.99):
    return TextCell(
        index=index,
        text=text,
        orig=text,
        confidence=confidence,
        from_ocr=True,
        rect=BoundingRectangle.from_bounding_box(
            BoundingBox.from_tuple(box, origin=CoordOrigin.TOPLEFT)
        ),
    )


def test_repeated_amount_pixels_are_removed_without_deleting_real_operators():
    amount = _cell(0, "=123.45", (100, 80, 230, 110))
    duplicate = _cell(1, "+", (108, 90, 116, 100), 0.4)
    standalone = _cell(2, "+", (70, 90, 78, 100), 0.4)
    strong = _cell(3, "+", (108, 90, 116, 100), 0.99)
    native = duplicate.model_copy(update={"index": 4, "from_ocr": False})
    lines = [amount, duplicate, standalone, strong, native]
    page = Page(
        page_no=1,
        size=Size(width=320, height=200),
        parsed_page=SegmentedPdfPage.model_construct(
            textline_cells=lines, word_cells=[], char_cells=[]
        ),
    )
    remove_duplicate_fiscal_marks(page, set())
    assert len(page.cells) == 5
    remove_duplicate_fiscal_marks(page, {amount.index})
    assert [cell.index for cell in page.cells] == [0, 2, 3, 4]
    amount.rect = amount.rect.model_copy(update={"r_y1": 130, "r_y2": 100})
    neighbor = _cell(5, "+", (108, 116, 116, 124), 0.4)
    page.parsed_page.textline_cells = [amount, neighbor]
    remove_duplicate_fiscal_marks(page, {amount.index})
    assert len(page.cells) == 2
