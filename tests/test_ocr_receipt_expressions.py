# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Receipt repairs need complete independent readings and actual source ink."""

from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
from docling_core.types.doc import BoundingBox, CoordOrigin
from docling_core.types.doc.page import BoundingRectangle, TextCell
from PIL import Image, ImageDraw

from docling.backend.image_backend import _ImagePageBackend
from docling.datamodel.base_models import Page
from docling.models.stages.ocr.ocr_receipt_expressions import (
    choose_damaged_amount,
    choose_printed_separator,
    choose_receipt_expression,
    choose_unitless_expression,
    compact_amount_marker,
    opaque_amount_marker,
    restore_receipt_separator,
    verify_receipt_page,
)
from docling.models.stages.ocr.ocr_receipt_regions import (
    ReceiptPriceRefinements,
    _apply_source_word,
    _source_expression,
)
from docling.models.stages.ocr.ocr_region_refiner import OcrRegionRefiner, RegionReading

if TYPE_CHECKING:
    from tesserocr import PyTessBaseAPI


def _readings(text: str) -> list[RegionReading]:
    return [
        RegionReading(text, 0.99, "cyrillic-v6-medium", "v6-source-1.0"),
        RegionReading(text, 0.99, "cyrillic-v6-tiny", "v6-clean-1.0"),
    ]


def test_quantity_and_unit_are_observed_before_recovering_a_merged_expression():
    assert (
        choose_receipt_expression("425.0017.", _readings("425.00*1шт."), confidence=0.8)
        == "425.00*1шт."
    )
    assert (
        choose_receipt_expression("325.0017.", _readings("425.00*1шт."), confidence=0.8)
        == "425.00*1шт."
    )


@pytest.mark.parametrize(
    "original,candidate",
    [
        ("25.00*2\u043a\u0433", "25.00*1\u043a\u0433"),
        ("25.00*2\u043a\u0433", "25.00*2м"),
        ("25.00*0.500\u043a\u0433", "25.00*0.5\u043a\u0433"),
        ("25.00*2\u043a\u0433", "250.00*2\u043a\u0433"),
        ("25.00*2\u043a\u0433", "35.10*2\u043a\u0433"),
        ("-25.00*2\u043a\u0433", "25.00*2\u043a\u0433"),
        ("25.0017.", "25.00"),
        ("25.0017.", "25.00*1"),
    ],
)
def test_expression_repair_retains_quantity_unit_sign_and_decimal_positions(
    original, candidate
):
    assert (
        choose_receipt_expression(original, _readings(candidate), confidence=0.8)
        == original
    )


def test_two_preprocessings_of_one_model_cannot_rewrite_an_expression():
    readings = [
        RegionReading("425.00*1шт.", 0.99, "cyrillic-v6-medium", variant)
        for variant in ("v6-source-1.0", "v6-clean-1.0")
    ]
    assert (
        choose_receipt_expression("325.0017.", readings, confidence=0.8) == "325.0017."
    )


@pytest.mark.parametrize(
    "original,candidate",
    [("37.421", "37.42*1"), ("37.41", "37.42*1"), ("37.42*2", "137.42*2")],
)
def test_complete_unitless_expression_is_observed_not_calculated(original, candidate):
    readings = _readings(candidate)
    assert (
        choose_unitless_expression(original, readings, printed_operator=False) is None
    )
    selected = choose_unitless_expression(original, readings, printed_operator=True)
    assert selected is not None and selected.text == candidate


@pytest.mark.parametrize(
    "original,candidate",
    [
        ("37.42*2", "137.42*1"),
        ("37.42*0.500", "137.42*0.5"),
        ("137.42*1", "37.42*1"),
        ("37.421", "37.42*2"),
        ("37.4217.", "37.42*1"),
        ("-37.42", "37.42*1"),
        ("(37.42)", "37.42*1"),
        ("+37.42", "37.42*1"),
        ("37.42-1", "37.42*1"),
        ("ID37.42*1", "137.42*1"),
        ("37.42*1", "37.42*1"),
    ],
)
def test_unitless_repair_protects_quantity_scale_signs_ids_and_intact_values(
    original, candidate
):
    assert (
        choose_unitless_expression(
            original, _readings(candidate), printed_operator=True
        )
        is None
    )


def test_unitless_repair_requires_independent_and_unambiguous_readings():
    repeated = [
        RegionReading("37.42*1", 0.99, "ch", view) for view in ["source", "wide"]
    ]
    assert choose_unitless_expression("37.421", repeated, printed_operator=True) is None
    conflicting = _readings("37.43*1") + _readings("37.44*1")
    assert (
        choose_unitless_expression("37.42", conflicting, printed_operator=True) is None
    )


def _marked_numeric_line(marker: str = "=") -> np.ndarray:
    image = Image.new("L", (112, 40), 255)
    draw = ImageDraw.Draw(image)
    for left in (40, 50, 60, 80, 90):
        draw.rectangle((left, 8, left + 9, 27), fill=0)
        draw.rectangle((left + 2, 10, left + 7, 15), fill=255)
        draw.rectangle((left + 2, 19, left + 7, 25), fill=255)
    draw.rectangle((73, 25, 75, 27), fill=0)
    draw.rectangle((32, 22, 39, 24), fill=0)
    if marker == "=":
        draw.rectangle((32, 17, 39, 19), fill=0)
    return np.asarray(image)


@pytest.mark.parametrize("original", ["7.", ".-7.42", ".7,42"])
def test_missing_sum_digits_need_a_source_mark_and_two_complete_readings(original):
    readings = _readings("=137.42")
    selected = choose_damaged_amount(original, readings, _marked_numeric_line())
    assert selected is not None and selected.text == "=137.42"
    assert choose_damaged_amount(original, readings, _marked_numeric_line("-")) is None
    assert choose_damaged_amount(original, readings[:1], _marked_numeric_line()) is None


@pytest.mark.parametrize(
    "original", ["7.42", "-7.42", "(7.42)", "+7.42", "7.421", "ID7.", "8.", ".-7.43"]
)
def test_damaged_sum_repair_keeps_complete_values_signs_ids_and_surviving_digits(
    original,
):
    assert (
        choose_damaged_amount(original, _readings("=137.42"), _marked_numeric_line())
        is None
    )


def test_damaged_sum_repair_rejects_conflicting_families_and_negative_readings():
    pixels = _marked_numeric_line()
    for readings in (
        _readings("137.42"),
        _readings("=137.42") + _readings("=237.42"),
        [*_readings("=137.42"), RegionReading("-137.42", 0.96, "ch", "source")],
        [RegionReading("=137.42", 0.999, "ch", v) for v in ["clean", "wide"]],
    ):
        assert choose_damaged_amount("7.", readings, pixels) is None


def _source_page(text: str, *, confidence: float = 0.8) -> tuple[Page, TextCell]:
    backend = _ImagePageBackend(Image.new("RGB", (400, 120)), 0, (72, 72))
    parsed = backend.get_segmented_page()
    word = TextCell(
        index=0,
        text=text,
        orig=text,
        from_ocr=True,
        confidence=confidence,
        rect=BoundingRectangle.from_bounding_box(
            BoundingBox.from_tuple((160, 30, 280, 70), CoordOrigin.TOPLEFT)
        ),
    )
    parsed.word_cells = [word]
    parsed.textline_cells = [word.model_copy(deep=True)]
    return Page(page_no=1, size=backend.get_size(), parsed_page=parsed), word


def test_high_primary_confidence_does_not_hide_a_missing_price_digit(monkeypatch):
    page, word = _source_page("37.42*2", confidence=0.999)
    refiner = OcrRegionRefiner.__new__(OcrRegionRefiner)
    monkeypatch.setattr(refiner, "_rapid_readings", lambda gray, **kw: [])
    monkeypatch.setattr(
        refiner, "_cyrillic_readings", lambda gray: _readings("137.42*2")
    )
    selected = _source_expression(
        refiner, page, word, np.full((120, 400), 255, np.uint8)
    )
    assert selected is not None and selected.text == "137.42*2"


@pytest.mark.parametrize(
    "line_text,accepted",
    [
        ("Описание.37.41", True),
        ("137.41", False),
        ("1.37.41", False),
        ("37.411", False),
    ],
)
def test_source_word_is_saved_after_stray_join_punctuation_but_not_inside_numbers(
    line_text, accepted
):
    page, word = _source_page("37.41")
    assert page.parsed_page is not None
    page.parsed_page.textline_cells[0].text = line_text
    reading = RegionReading("37.42*1", 0.99, "consensus", "source")
    assert (
        _apply_source_word(page, word, reading, ReceiptPriceRefinements()) is accepted
    )
    expected = "Описание.37.42*1" if accepted else line_text
    assert page.parsed_page.textline_cells[0].text == expected


def test_verified_price_after_join_punctuation_survives_later_row_reading():
    text = "Описание.37.42*2"
    verified = ReceiptPriceRefinements()
    verified.record(0, text, (text.index("37"), len(text)))
    assert verified.preserves(0, "Название.37,42*2")
    assert not verified.preserves(0, "Название.37.43*2")


def test_later_row_cannot_erase_a_source_verified_operator_or_quantity():
    original = "Товар 37.42*2 =74.84"
    start = original.index("37")
    verified = ReceiptPriceRefinements()
    verified.record(0, original, (start, start + len("37.42*2")))
    assert verified.preserves(0, "Описание 37,42\u00d72 =74,84")
    for damaged in (
        "Товар 37.42\u04402 =74.84",
        "Товар 37.422 =74.84",
        "Товар 37.42 =74.84",
        "Товар 37.42*3 =74.84",
        "Товар 37.42*2.0 =74.84",
        "Товар 37.42*2 37.42*2 =74.84",
    ):
        assert not verified.preserves(0, damaged)
    readings = [
        RegionReading(text, 0.99, "cyrillic", "row")
        for text in ("Описание 37,42\u00d72 =74,84", "Товар 37.42\u04402 =74.84")
    ]
    assert [r.text for r in verified.readings_for(0, readings)] == [readings[0].text]


def test_each_verified_expression_keeps_its_position_and_observed_unit():
    original = "Первый 18.70*0.500\u043a\u0433 Второй 18.70*2шт."
    verified = ReceiptPriceRefinements()
    verified.record(0, original, (0, len(original)))
    assert verified.preserves(
        0, "Первый 18,70x0,500\u041a\u0413 Второй 18,70*2\u0428\u0422"
    )
    for damaged in (
        "Первый 18.70*2шт. Второй 18.70*0.500\u043a\u0433",
        "Первый 18.70*0.5\u043a\u0433 Второй 18.70*2шт.",
        "Первый 18.70*0.500\u0433 Второй 18.70*2шт.",
        "Первый 18.70*0.500 Второй 18.70*2шт.",
    ):
        assert not verified.preserves(0, damaged)


def test_a_unit_outside_the_verified_source_word_stays_unconstrained():
    text = "Товар 18.70*2 шт."
    start = text.index("18")
    verified = ReceiptPriceRefinements()
    verified.record(0, text, (start, start + len("18.70*2")))
    assert verified.preserves(0, "Товар 18.70*2шт.")
    assert verified.preserves(0, "Товар 18.70*2")
    assert not verified.preserves(0, "Товар 18.70\u04402шт.")


def test_an_earlier_verified_expression_survives_individual_price_refinement():
    pytest.importorskip("rapidocr")
    from docling.models.stages.ocr.ocr_receipt_regions import refine_receipt_prices

    text = "37.42*2шт."
    page, _ = _source_page(text)
    # Every character was already verified upstream. No additional recognizer
    # is needed here; the returned protections must retain that complete proof.
    refiner = OcrRegionRefiner.__new__(OcrRegionRefiner)
    result = refine_receipt_prices(
        refiner,
        page,
        Image.new("L", (400, 120), 255),
        cast("PyTessBaseAPI", object()),
        verified_lines={0},
    )
    assert result.preserves(0, text)
    assert not result.preserves(0, "37.42\u04402шт.")
    assert not result.preserves(0, "37.42*2")


@pytest.mark.parametrize("native_word", [False, True])
def test_observed_fiscal_amount_replaces_only_its_complete_split_ocr_span(
    monkeypatch, native_word
):
    from docling.models.stages.ocr.ocr_receipt_amounts import refine_fiscal_numbers

    page, word = _source_page("7.")
    assert page.parsed_page is not None
    line = page.parsed_page.textline_cells[0]
    label = line.model_copy(deep=True, update={"index": 1, "text": "ИТОГ"})
    label.rect = BoundingRectangle.from_bounding_box(
        BoundingBox.from_tuple((10, 30, 110, 70), CoordOrigin.TOPLEFT)
    )
    page.parsed_page.textline_cells.append(label)
    word.text = "7"
    word.from_ocr = not native_word
    punctuation = word.model_copy(deep=True, update={"index": 1, "text": "."})
    page.parsed_page.word_cells.append(punctuation)
    pixels = Image.new("L", (400, 120), 255)
    pixels.paste(Image.fromarray(_marked_numeric_line()), (164, 30))
    refiner = OcrRegionRefiner.__new__(OcrRegionRefiner)
    monkeypatch.setattr(refiner, "_rapid_readings", lambda gray, **kw: [])
    monkeypatch.setattr(
        refiner, "_cyrillic_readings", lambda gray: _readings("=137.42")
    )
    changed = refine_fiscal_numbers(refiner, page, pixels)
    assert changed == (set() if native_word else {0})
    assert line.text == ("7." if native_word else "=137.42")
    assert [w.text for w in page.parsed_page.word_cells] == (
        ["7", "."] if native_word else ["=137.42"]
    )


@pytest.mark.parametrize("separator", [".", ",", ":"])
def test_source_separator_keeps_every_digit_and_its_observed_punctuation(separator):
    selected = choose_printed_separator(
        "=3728", _readings("37" + separator + "28"), printed_separator=True
    )
    assert selected is not None and selected.text == "=37" + separator + "28"
    assert (
        choose_printed_separator(
            "3728", _readings("37" + separator + "28"), printed_separator=False
        )
        is None
    )


@pytest.mark.parametrize(
    "original", ["-3728", "(3728)", "+3728", "37.28", "003728", "1234567890"]
)
def test_separator_recovery_cannot_drop_signs_leading_zeros_or_change_identifiers(
    original,
):
    assert (
        choose_printed_separator(original, _readings("37.28"), printed_separator=True)
        is None
    )


def test_an_ambiguous_or_single_model_separator_stays_unresolved():
    repeats = [RegionReading("37:28", 0.99, "ch", view) for view in ["source", "wide"]]
    assert choose_printed_separator("3728", repeats, printed_separator=True) is None
    conflict = _readings("37.28") + _readings("37:28")
    assert choose_printed_separator("3728", conflict, printed_separator=True) is None


def _marker(kind: str) -> np.ndarray:
    image = Image.new("L", (190, 52), 255)
    draw = ImageDraw.Draw(image)
    for left in (28, 45, 62, 79, 96, 128, 145):
        draw.rectangle((left, 24, left + 12, 37), fill=0)
    draw.line((0, 16, 170, 21), fill=0, width=2)
    draw.line((0, 42, 170, 47), fill=0, width=2)
    if kind == "filled":
        draw.rectangle((11, 26, 28, 34), fill=0)
    elif kind == "minus":
        draw.rectangle((11, 29, 28, 31), fill=0)
    elif kind == "colon":
        draw.rectangle((14, 26, 17, 28), fill=0)
        draw.rectangle((14, 32, 17, 34), fill=0)
    return np.asarray(image)


@pytest.mark.parametrize("kind", ["minus", "colon", "empty"])
def test_a_real_minus_colon_or_missing_mark_cannot_be_a_filled_separator(kind):
    assert not compact_amount_marker(_marker(kind))
    readings = [
        RegionReading("=425.00", 0.999, "cyrillic-v6-medium", variant)
        for variant in ("v6-source-1.0", "v6-clean-1.0")
    ] + [RegionReading("425.00", 0.99, "ch", "clean")]
    assert restore_receipt_separator("-425.00", readings, _marker(kind)) == "-425.00"


def test_a_filled_marker_needs_two_views_independent_digits_and_no_negative_vote():
    image = _marker("filled")
    assert compact_amount_marker(image)
    readings = [
        RegionReading("=425.00", 0.999, "cyrillic-v6-medium", variant)
        for variant in ("v6-source-1.0", "v6-clean-1.0")
    ]
    assert restore_receipt_separator("-425.00", readings, image) == "-425.00"
    readings.append(RegionReading("425.00", 0.99, "ch", "clean"))
    assert restore_receipt_separator("-425.00", readings, image) == "=425.00"
    assert restore_receipt_separator("-425.01", readings, image) == "-425.01"
    readings.append(RegionReading("-425.00", 0.99, "cyrillic-v6-tiny", "source"))
    assert restore_receipt_separator("-425.00", readings, image) == "-425.00"


def _detached_marker(kind: str, scale: int) -> np.ndarray:
    image = Image.new("L", (130, 40), 255)
    draw = ImageDraw.Draw(image)
    for left in (30, 47, 64, 88, 105):
        draw.rectangle((left, 10, left + 11, 29), fill=0)
    if kind == "filled":
        draw.rectangle((15, 14, 23, 24), fill=0)
    elif kind == "compact":
        draw.rectangle((15, 17, 23, 23), fill=0)
    elif kind == "minus":
        draw.rectangle((12, 19, 25, 21), fill=0)
    elif kind == "bold-minus":
        draw.rectangle((5, 16, 24, 23), fill=0)
    elif kind == "colon":
        draw.rectangle((19, 14, 21, 16), fill=0)
        draw.rectangle((19, 22, 21, 24), fill=0)
    return np.asarray(image.resize((130 * scale, 40 * scale)))


@pytest.mark.parametrize("scale", [1, 2, 3])
@pytest.mark.parametrize(
    "kind", ["filled", "compact", "minus", "bold-minus", "colon", "empty"]
)
def test_detached_source_mark_requires_independent_digits_and_source_equals(
    kind, scale
):
    image = _detached_marker(kind, scale)
    filled = kind in {"filled", "compact"}
    assert opaque_amount_marker(image) is filled
    readings = [
        RegionReading("=317.42", 0.99, "ch", "source"),
        RegionReading("317.42", 0.99, "cyrillic", "clean"),
    ]
    expected = "=317.42" if filled else "-317.42"
    assert restore_receipt_separator("-317.42", readings, image) == expected


def test_detached_mark_cannot_override_negative_or_disagreeing_source_readings():
    image = _detached_marker("filled", 2)
    source_equals = RegionReading("=317.42", 0.99, "ch", "source")
    independent = RegionReading("317.42", 0.99, "cyrillic", "clean")
    for readings in (
        [source_equals],
        [source_equals, RegionReading("317.42", 0.99, "ch", "clean")],
        [source_equals, RegionReading("317.43", 0.99, "cyrillic", "clean")],
        [RegionReading("=317.42", 0.99, "ch", "clean"), independent],
        [
            source_equals,
            independent,
            RegionReading("-317.42", 0.96, "iso:ru", "source"),
        ],
    ):
        assert restore_receipt_separator("-317.42", readings, image) == "-317.42"


def test_later_row_reading_preserves_verified_amount_without_freezing_other_fields():
    original = "Товар 55.30*2 =110.60 НДС 102"
    start = original.index("=")
    verified = ReceiptPriceRefinements()
    verified.record(7, original, (start, start + len("=110.60")))
    assert verified.preserves(7, "Описание 56,30*2 =110,60 НДС 10%")
    assert verified.preserves(7, "Описание 55.30*2 = 110.60 НДС 10%")
    assert verified.preserves(8, "Иная строка -110.60")
    for damaged in (
        "Товар 55.30*2 -110.60 НДС 102",
        "Товар 55.30*2 110.60 НДС 102",
        "Товар 55.30*2 =110.61 НДС 102",
        "Товар 110.60*2 =55.30 НДС 102",
        "Товар =110.60 НДС 102",
    ):
        assert not verified.preserves(7, damaged)


def test_verified_negative_amount_keeps_its_sign_and_each_repeated_position():
    original = "Возврат -36.80 НДС 6.13 Оплата -36.80"
    verified = ReceiptPriceRefinements()
    for start in (original.index("-"), original.rindex("-")):
        verified.record(4, original, (start, start + len("-36.80")))
    assert verified.preserves(4, "Возврат (36,80) НДС 7.14 Оплата \u221236.80")
    assert not verified.preserves(4, "Возврат 36.80 НДС 6.13 Оплата -36.80")
    assert not verified.preserves(4, "Возврат -36.80 НДС 6.13 Оплата 36.80")


def test_verified_separator_attached_to_quantity_is_not_dropped():
    original = "Товар 21.50*2=43.00"
    verified = ReceiptPriceRefinements()
    start = original.index("=")
    verified.record(2, original, (start, len(original)))
    assert verified.preserves(2, "Описание 21,50*2=43,00")
    assert not verified.preserves(2, "Товар 21.50*2 43.00")
    assert not verified.preserves(2, "Товар 21.50*2-43.00")


def test_row_letters_can_improve_with_the_verified_source_marker_and_unchanged_digits():
    original = "Товар 55.30*2 =110.60 НДС 102"
    start = original.index("=")
    verified = ReceiptPriceRefinements()
    verified.record(7, original, (start, start + len("=110.60")))
    readings = [
        RegionReading("Название 55.30*2 -110.60 НДС 10%", 0.9, "cyrillic", "source"),
        RegionReading("Название 55.30*2 110,60 НДС 10%", 0.9, "iso:ru", "source"),
        RegionReading("Название 55.30*2 -110.61 НДС 10%", 0.99, "ch", "source"),
        RegionReading("Название -110.60 НДС 10%", 0.99, "ch", "wide"),
    ]
    retained = verified.readings_for(7, readings)
    assert [r.text for r in retained] == [
        "Название 55.30*2 =110.60 НДС 10%",
        "Название 55.30*2 =110,60 НДС 10%",
    ]
    assert [r.engine for r in retained] == ["cyrillic", "iso:ru"]
    assert all(
        r.confidence == 0.9 and "+verified-marker" in r.variant for r in retained
    )
    assert readings[0].text == "Название 55.30*2 -110.60 НДС 10%"
    assert verified.readings_for(8, readings) == readings


@pytest.mark.parametrize("independent", [False, True])
def test_a_damaged_fiscal_caption_needs_independent_confirmation(
    monkeypatch, independent
):
    backend = _ImagePageBackend(Image.new("RGB", (400, 240)), 0, (72, 72))
    parsed = backend.get_segmented_page()
    texts = [
        "\u041a\u0410\u0421\u0421\u0418\u0420",
        "25.00",
        "70.00",
        "=45.00",
        "\u0421\u0423\u041d\u041d\u0410 \u041d\u0421 107",
    ]
    parsed.textline_cells = [
        TextCell(
            index=index,
            text=text,
            orig=text,
            from_ocr=True,
            confidence=0.8,
            rect=BoundingRectangle.from_bounding_box(
                BoundingBox.from_tuple(
                    (20, 20 + 30 * index, 200, 40 + 30 * index), CoordOrigin.TOPLEFT
                )
            ),
        )
        for index, text in enumerate(texts)
    ]
    page = Page(page_no=1, size=backend.get_size(), parsed_page=parsed)
    refiner = OcrRegionRefiner.__new__(OcrRegionRefiner)
    readings = [
        RegionReading(
            "\u0421\u0423\u041c\u041c\u0410 \u041d\u0414\u0421 10%",
            0.99,
            engine,
            variant,
        )
        for engine, variant in (
            ("cyrillic-v6-medium", "v6-source-1.0"),
            (
                "cyrillic-v6-tiny" if independent else "cyrillic-v6-medium",
                "v6-clean-1.0",
            ),
        )
    ]
    monkeypatch.setattr(refiner, "_cyrillic_readings", lambda gray: readings)
    assert verify_receipt_page(refiner, page, backend.get_page_image()) is independent
    # Classification does not replace damaged text with a guessed caption.
    assert page.parsed_page.textline_cells[-1].text == texts[-1]
    parsed.textline_cells = parsed.textline_cells[3:]
    assert not verify_receipt_page(refiner, page, backend.get_page_image())
