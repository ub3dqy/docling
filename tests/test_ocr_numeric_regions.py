# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""A local integer correction must remain attached to its original OCR token."""

import pytest
from docling_core.types.doc import BoundingBox, CoordOrigin
from docling_core.types.doc.page import BoundingRectangle, TextCell
from PIL import Image

from docling.backend.image_backend import _ImagePageBackend
from docling.datamodel.base_models import Page
from docling.models.stages.ocr.ocr_numeric_regions import (
    choose_numbered_suffix,
    replace_word_text,
)
from docling.models.stages.ocr.ocr_region_refiner import RegionReading


def _cell(index, text, box, *, from_ocr=True):
    return TextCell(
        index=index,
        text=text,
        orig=text,
        confidence=0.7,
        from_ocr=from_ocr,
        rect=BoundingRectangle.from_bounding_box(
            BoundingBox.from_tuple(box, origin=CoordOrigin.TOPLEFT)
        ),
    )


def _page(lines):
    backend = _ImagePageBackend(Image.new("RGB", (300, 200)), 0, (72, 72))
    parsed = backend.get_segmented_page()
    parsed.word_cells = [_cell(0, "491", (100, 50, 125, 65))]
    parsed.textline_cells = lines
    return Page(page_no=1, size=backend.get_size(), parsed_page=parsed)


def test_replacement_updates_only_one_complete_ocr_token_in_its_source_row():
    page = _page(
        [
            _cell(0, "Record 491 dated 2022", (20, 48, 250, 68)),
            _cell(1, "Record 491", (20, 100, 250, 120)),
            _cell(2, "Record 491", (20, 48, 250, 68), from_ocr=False),
            _cell(3, "Record 1491", (20, 48, 250, 68)),
        ]
    )
    assert replace_word_text(page, 0, "167") == {0}
    assert page.parsed_page is not None
    assert [line.text for line in page.parsed_page.textline_cells] == [
        "Record 167 dated 2022",
        "Record 491",
        "Record 491",
        "Record 1491",
    ]
    assert page.parsed_page.word_cells[0].text == "167"
    assert page.parsed_page.word_cells[0].orig == "491"


def test_repeated_ambiguous_token_does_not_change_the_line_or_word():
    page = _page([_cell(0, "491 / 491", (20, 48, 250, 68))])
    assert replace_word_text(page, 0, "167") == set()
    assert page.parsed_page is not None
    assert page.parsed_page.word_cells[0].text == "491"
    assert page.parsed_page.textline_cells[0].text == "491 / 491"


@pytest.mark.parametrize(
    ("original", "observed", "expected"),
    [
        ("\u041816", "\u2116 167", "\u2116 167"),
        ("\u2116 28", "\u2116281", "\u2116281"),
        ("\u2116 28", "\u2116381", "\u2116 28"),
        ("\u2116 28", "\u21162814", "\u2116 28"),
        ("\u2116 28", "281", "\u2116 28"),
        ("28", "\u2116281", "28"),
        ("-28.00", "\u2116281", "-28.00"),
    ],
)
def test_a_clipped_number_needs_the_source_number_sign_and_complete_original_prefix(
    original, observed, expected
):
    readings = [
        RegionReading(observed, 0.99, engine, "source")
        for engine in ("cyrillic-v6-medium", "cyrillic-v6-tiny")
    ]
    assert choose_numbered_suffix(original, readings) == expected
    assert choose_numbered_suffix(original, readings[:1]) == original
