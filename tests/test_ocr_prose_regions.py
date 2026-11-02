# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""A prose repair must preserve real numbers and their actual source positions."""

import pytest
from docling_core.types.doc import BoundingBox
from docling_core.types.doc.page import BoundingRectangle, TextCell

from docling.models.stages.ocr.ocr_prose_regions import (
    ProseRow,
    numbers_have_source_positions,
    prose_numbers,
    repair_prose_words,
)
from docling.models.stages.ocr.ocr_region_refiner import RegionReading


def _known(word):
    return word.casefold() in {"союз", "из", "за"}


@pytest.mark.parametrize(
    "source,expected",
    [
        ("\u04210\u044e3 \u04183 2025", ("2025",)),
        ("3a 2025", ("2025",)),
        ("код \u04183", ("3",)),
        ("ОГРН 1167700052112", ("1167700052112",)),
        ("№ \u041003, формат A3", ("03", "3")),
        ("Сумма -12,50; НДС 20%", ("-12.50", "20")),
    ],
)
def test_word_like_glyphs_do_not_relax_numbers_identifiers_rates_or_signs(
    source, expected
):
    assert prose_numbers(source, _known) == expected


def _word(index, text, left):
    rect = BoundingRectangle.from_bounding_box(
        BoundingBox(l=left, t=10, r=left + 40, b=25)
    )
    return TextCell(
        index=index, text=text, orig=text, confidence=0.99, from_ocr=True, rect=rect
    )


def test_equal_number_counts_cannot_move_years_to_each_others_source_positions():
    first, second = _word(0, "2024", 10), _word(1, "2025", 100)
    row = ProseRow("2024 2025", 0.99, first.rect, (first, second))
    assert numbers_have_source_positions([row], [first, second], _known)
    swapped = ProseRow(
        "2025 2024", 0.99, first.rect, (_word(2, "2025", 10), _word(3, "2024", 100))
    )
    assert not numbers_have_source_positions([swapped], [first, second], _known)
    assert not numbers_have_source_positions([row], [first], _known)


def test_secondary_word_repair_needs_two_models_and_cannot_change_a_number():
    words = (_word(0, "от%ету", 10), _word(1, "2025", 100))
    row = ProseRow("от%ету 2025", 0.95, words[0].rect, words)

    def known(word):
        return word == "отчету"

    readings = [
        RegionReading("отчету 2025", 0.99, "cyrillic-v6-medium", "source"),
        RegionReading("отчету 2025", 0.99, "cyrillic-v6-tiny", "source"),
    ]
    assert repair_prose_words(row, readings[:1], known) == row
    repaired = repair_prose_words(row, readings, known)
    assert repaired.text == "отчету 2025"
    assert repaired.words[0].orig == "от%ету"
    assert repaired.words[0].rect == words[0].rect
    readings = [
        RegionReading("отчету 2026", r.confidence, r.engine, r.variant)
        for r in readings
    ]
    assert repair_prose_words(row, readings, known).words[1].text == "2025"
