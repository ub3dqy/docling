# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT
# ruff: noqa: RUF001 -- Observed Cyrillic, digit and dash variants are regression inputs.

"""A prose repair must preserve real numbers and their actual source positions."""

import pytest
from docling_core.types.doc import BoundingBox, DocItemLabel
from docling_core.types.doc.page import BoundingRectangle, TextCell
from PIL import Image

from docling.backend.image_backend import _ImagePageBackend
from docling.datamodel.base_models import (
    Cluster,
    LayoutPrediction,
    Page,
    PagePredictions,
)
from docling.models.stages.ocr.ocr_prose_regions import (
    ProseRow,
    _bad_letters,
    _needs_reading,
    confirm_prose_row,
    confirmed_quote_repair,
    numbers_have_source_positions,
    prose_numbers,
    prose_operators,
    repair_prose_words,
)
from docling.models.stages.ocr.ocr_prose_replacement import replace_verified_prose
from docling.models.stages.ocr.ocr_region_refiner import RegionReading


@pytest.mark.parametrize(
    "text,expected",
    [
        ("ПРИМЕР СМЕШАННОГО РУССКОГО TЕКСТА", True),
        ("ПРИМЕР СМЕШАННОГО РУССКОГО ТЕКСТА", False),
        ("Проверка названия компании Example", False),
    ],
)
def test_a_known_word_with_a_foreign_lookalike_still_needs_source_verification(
    text, expected
):
    word = _word(0, text, 10)
    cluster = Cluster(
        id=0, label=DocItemLabel.TEXT, bbox=word.rect.to_bounding_box(), cells=[word]
    )
    assert _needs_reading(cluster, lambda _: True) is expected
    if expected:
        assert _bad_letters(text, lambda _: True) > _bad_letters(
            text.replace("T", "Т"), lambda _: True
        )


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


@pytest.mark.parametrize("punctuation_confidence", [0.0, 0.75, 0.95])
def test_row_consensus_can_discard_only_weak_isolated_punctuation_tokens(
    punctuation_confidence,
):
    words = (
        _word(0, "Пример", 10),
        _word(1, "—", 60).model_copy(update={"confidence": punctuation_confidence}),
        _word(2, "текста", 110),
        _word(3, "2041", 160),
    )
    row = ProseRow("Пример — текста 2041", 0.75, words[0].rect, words)
    readings = [
        RegionReading("Пример текста 2041", 0.99, engine, "source")
        for engine in ("cyrillic-v6-medium", "cyrillic-v6-tiny")
    ]
    assert confirm_prose_row(row, readings[:1], _known) == row
    result = confirm_prose_row(row, readings, _known)
    if punctuation_confidence >= 0.8:
        assert result == row
    else:
        assert result.text == "Пример текста 2041"
        assert result.source_confirmed
        assert [w.rect for w in result.words] == [
            w.rect for w in (words[0], words[2], words[3])
        ]
        assert (
            confirm_prose_row(
                row,
                [
                    RegionReading(
                        r.text.replace("2041", "2042"),
                        r.confidence,
                        r.engine,
                        r.variant,
                    )
                    for r in readings
                ],
                _known,
            )
            == row
        )


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


def _prose_cell(index, text, left, top, right, bottom, *, native=False):
    return TextCell(
        index=index,
        text=text,
        orig=text,
        from_ocr=not native,
        confidence=0.99,
        rect=BoundingRectangle.from_bounding_box(
            BoundingBox(l=left, t=top, r=right, b=bottom)
        ),
    )


def _prose_page(source, *, confirmed=True):
    backend = _ImagePageBackend(Image.new("RGB", (1400, 300), "white"), 0, (72, 72))
    parsed = backend.get_segmented_page()
    fragment = _prose_cell(6, "продолжение", 225, 16, 330, 28)
    repeated = source.model_copy(
        deep=True,
        update={
            "index": 7,
            "rect": BoundingRectangle.from_bounding_box(
                BoundingBox(l=10, t=76, r=210, b=100)
            ),
        },
    )
    donor = Cluster(
        id=1,
        label=DocItemLabel.TEXT,
        bbox=source.rect.to_bounding_box(),
        cells=[source],
    )
    target = Cluster(
        id=2,
        label=DocItemLabel.TEXT,
        bbox=BoundingBox(l=5, t=5, r=350, b=110),
        cells=[fragment],
    )
    other = Cluster(
        id=3,
        label=DocItemLabel.TEXT,
        bbox=repeated.rect.to_bounding_box(),
        cells=[repeated],
    )
    parent = Cluster(
        id=4,
        label=DocItemLabel.TEXT,
        bbox=target.bbox,
        cells=[source],
        children=[donor, target],
    )
    words = tuple(
        _prose_cell(i, text, left, 16, right, 28)
        for i, text, left, right in [
            (10, "Уверенная", 10, 80),
            (11, "первая", 90, 140),
            (12, "строка", 150, 210),
            (13, "продолжение", 225, 330),
        ]
    )
    extra = _prose_cell(14, "*", 15, 7, 20, 12)
    native = _prose_cell(15, "данные", 30, 17, 70, 27, native=True)
    repeated_word = _prose_cell(16, "Уверенная", 10, 82, 80, 94)
    parsed.textline_cells = [source, fragment, repeated]
    parsed.word_cells = [*words, extra, native, repeated_word]
    page = Page(
        page_no=1,
        size=backend.get_size(),
        parsed_page=parsed,
        predictions=PagePredictions(layout=LayoutPrediction(clusters=[parent, other])),
    )
    row = ProseRow(
        "Уверенная первая строка продолжение",
        0.99,
        BoundingRectangle.from_bounding_box(BoundingBox(l=10, t=16, r=330, b=28)),
        words,
        source_confirmed=confirmed,
    )
    return page, target, row


def test_verified_paragraph_consumes_only_the_same_printed_prefix_and_all_layout_aliases():
    source = _prose_cell(5, "Уверенная первая строка", 10, 10, 210, 34)
    page, target, row = _prose_page(source)
    replace_verified_prose(page, target, [row])
    assert [c.text for c in page.cells] == [source.text, row.text]
    assert page.cells[0].index == 7  # Same words printed on the next row stay.
    assert page.cells[1] is target.cells[0]
    parent = page.predictions.layout.clusters[0]
    assert parent.cells == []
    assert [child.id for child in parent.children] == [2]
    assert {w.index for w in page.parsed_page.word_cells} == {
        14,
        15,
        16,
        17,
        18,
        19,
        20,
    }
    assert len({c.index for c in page.cells}) == len(page.cells)


@pytest.mark.parametrize(
    "kind", ["unconfirmed", "native", "different", "number", "quote", "weak"]
)
def test_prose_replacement_never_consumes_unverified_native_or_changed_source(kind):
    source = _prose_cell(
        5, "Уверенная первая строка", 10, 10, 210, 34, native=kind == "native"
    )
    if kind == "different":
        source.text = "Уверенная другая строка"
    elif kind == "number":
        source.text += " 2041"
    elif kind == "quote":
        source.text = "«" + source.text
    elif kind == "weak":
        source.confidence = 0.7
    page, target, row = _prose_page(source, confirmed=kind != "unconfirmed")
    if kind in {"number", "quote"}:
        changed = row.words[2 if kind == "number" else 0]
        text = "2041" if kind == "number" else "«" + changed.text
        words = tuple(
            word.model_copy(update={"text": text}) if word is changed else word
            for word in row.words
        )
        row = ProseRow(
            " ".join(word.text for word in words),
            row.confidence,
            row.rect,
            words,
            source_confirmed=True,
        )
        if kind == "number":
            source.text = "Уверенная первая 2041"
    replace_verified_prose(page, target, [row])
    assert any(c.index == 5 and c.text == source.text for c in page.cells)
    assert page.predictions.layout.clusters[0].children[0].cells[0] is source


def test_identical_tilted_rows_keep_distinct_pixels_even_when_envelopes_overlap():
    def tilted(index, text, left, right, baseline):
        return TextCell(
            index=index,
            text=text,
            orig=text,
            confidence=0.99,
            from_ocr=True,
            rect=BoundingRectangle(
                r_x0=left,
                r_y0=baseline + left * 0.08,
                r_x1=right,
                r_y1=baseline + right * 0.08,
                r_x2=right,
                r_y2=baseline + right * 0.08 + 12,
                r_x3=left,
                r_y3=baseline + left * 0.08 + 12,
            ),
        )

    source = tilted(5, "Уверенная первая строка", 0, 1200, 0)
    page, target, _ = _prose_page(source)
    words = tuple(
        tilted(i, text, left, right, 14)
        for i, text, left, right in [
            (20, "Уверенная", 0, 400),
            (21, "первая", 410, 800),
            (22, "строка", 810, 1200),
        ]
    )
    row = ProseRow(
        source.text,
        0.99,
        tilted(0, source.text, 0, 1200, 14).rect,
        words,
        source_confirmed=True,
    )
    target.cells[0].text = source.text
    target.cells[0].orig = source.text
    target.cells[0].rect = row.rect
    target.bbox = row.rect.to_bounding_box()
    source_words = tuple(
        tilted(i, text, left, right, 0)
        for i, text, left, right in [
            (10, "Уверенная", 0, 400),
            (11, "первая", 410, 800),
            (12, "строка", 810, 1200),
        ]
    )
    target_words = tuple(
        word.model_copy(update={"index": i + 13}) for i, word in enumerate(words)
    )
    page.parsed_page.word_cells = [*source_words, *target_words]
    replace_verified_prose(page, target, [row])
    assert any(c.index == source.index for c in page.cells)
    assert {10, 11, 12} <= {word.index for word in page.parsed_page.word_cells}
    assert not {13, 14, 15} & {word.index for word in page.parsed_page.word_cells}
