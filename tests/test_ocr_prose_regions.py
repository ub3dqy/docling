# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT
# ruff: noqa: RUF001 -- Observed Cyrillic, digit and dash variants are regression inputs.

"""A prose repair must preserve real numbers and their actual source positions."""

import pytest
from docling_core.types.doc import BoundingBox
from docling_core.types.doc.page import BoundingRectangle, TextCell

from docling.models.stages.ocr.ocr_prose_regions import (
    ProseRow,
    confirm_prose_row,
    confirmed_quote_repair,
    numbers_have_source_positions,
    prose_numbers,
    prose_operators,
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


@pytest.mark.parametrize(
    "original,candidate,deficit,expected",
    [
        ("Членам объединения Союз", "Членам объединения «Союз", 1, True),
        ("Членам объединения «Союз»,2025", "Членам объединения «Союз»,2025", 1, False),
        ("Объединение «Союз", "Объединение «Союз»", -1, True),
        ("Объединение «Союз", "«Объединение Союз»,2025", -1, False),
        ("Объединение Союз,2025", "Объединение «Союз,2026", 1, False),
        ("Объединение Союз", "«Объединение «Союз", 1, False),
    ],
)
def test_only_observed_quotes_can_reduce_imbalance_without_moving_existing_quotes_or_numbers(
    original, candidate, deficit, expected
):
    word = _word(0, candidate, 10)
    row = ProseRow(candidate, 0.99, word.rect, (word,), quotes_confirmed=True)
    assert confirmed_quote_repair(original, [row], deficit) is expected
    unconfirmed = ProseRow(candidate, 0.99, word.rect, (word,))
    assert not confirmed_quote_repair(original, [unconfirmed], deficit)


def test_complete_source_rows_can_confirm_a_known_word_and_a_quote_together():
    original = "Решение «Союз за 2027 рода"
    observed = "Решение «Союз» за 2027 года"
    words = tuple(_word(i, token, i * 40) for i, token in enumerate(observed.split()))
    row = ProseRow(observed, 0.94, words[0].rect, words)
    readings = [
        RegionReading(observed, 0.99, engine, "source")
        for engine in ("cyrillic-v6-medium", "cyrillic-v6-tiny")
    ]
    assert not confirmed_quote_repair(original, [row], -1)
    confirmed = confirm_prose_row(row, readings, _known)
    assert confirmed.source_confirmed
    assert confirmed_quote_repair(original, [confirmed], -1)
    assert confirmed.words[0].rect == row.words[0].rect
    assert confirmed.words[0].orig == row.words[0].orig
    assert not confirmed_quote_repair(original.replace("2027", "2028"), [confirmed], -1)
    assert not confirmed_quote_repair(original + "»", [confirmed], -1)


def test_row_consensus_cannot_move_numbers_or_reuse_one_model():
    text = "Решение за 2027 год"
    words = tuple(_word(i, token, i * 40) for i, token in enumerate(text.split()))
    row = ProseRow(text, 0.94, words[0].rect, words)
    same_model = [
        RegionReading(text, 0.99, "iso:ru", variant) for variant in ("source", "clean")
    ]
    assert not confirm_prose_row(row, same_model, _known).source_confirmed
    for candidate in (
        "Решение за 2028 год",
        "Решение 2027 за год",
        "Решение за2027 год",
    ):
        readings = [
            RegionReading(candidate, 0.99, engine, "source")
            for engine in ("cyrillic-v6-medium", "cyrillic-v6-tiny")
        ]
        assert confirm_prose_row(row, readings, _known) == row


def test_typographic_prose_dashes_do_not_relax_amount_or_identifier_signs():
    assert prose_operators(
        "Организация области - региональное объединение"
    ) == prose_operators("Организация области – региональное объединение")
    assert prose_operators(
        "Организация области - региональное объединение"
    ) == prose_operators("Организация области — региональное объединение")
    assert prose_operators("Убыток -120.50") != prose_operators("Убыток 120.50")
    assert prose_operators("Период 2026 - 2027") != prose_operators(
        "Период 2026 – 2027"
    )
    assert prose_operators("Код РД-7") != prose_operators("Код РД7")
