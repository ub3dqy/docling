# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Safety regressions for crop-based corrections of financial numbers."""

import numpy as np
import pytest
from PIL import Image, ImageDraw

from docling.models.stages.ocr.ocr_region_refiner import (
    RegionReading,
    SymbolReading,
    choose_amount_punctuation,
    choose_line_number,
    choose_number,
    choose_text_punctuation,
    clean_region,
    compatible_text_reading,
    normalize_mixed_script,
    punctuation_line_crop,
    refine_word_choices,
    supports_leading_equals,
)


def test_multiple_preprocessings_of_one_recognizer_are_not_independent_votes():
    readings = [
        RegionReading("32 795", 0.99, "recognizer-a", "clean"),
        RegionReading("32795", 0.99, "recognizer-a", "wide"),
        RegionReading("32795", 0.99, "recognizer-a", "threshold"),
        RegionReading("32 785", 0.97, "recognizer-b", "clean"),
    ]
    assert choose_number("32 785", readings) == "32 785"


def test_two_recognizers_can_correct_a_digit_without_a_reference_answer():
    readings = [
        RegionReading("32 795", 0.96, "recognizer-a", "clean"),
        RegionReading("32795", 0.98, "recognizer-b", "clean"),
        RegionReading("32 785", 0.99, "recognizer-c", "clean"),
    ]
    assert choose_number("32 785", readings) == "32795"


@pytest.mark.parametrize(
    ("original", "cropped"),
    [
        ("(37 921)", "37 921"),
        ("12.50", "1250"),
        ("32795", "2795"),
        ("-12.50", "-125.0"),
    ],
)
def test_crop_agreement_cannot_drop_sign_digits_or_move_decimal(original, cropped):
    readings = [
        RegionReading(cropped, 0.99, "recognizer-a", "clean"),
        RegionReading(cropped, 0.99, "recognizer-b", "clean"),
    ]
    assert choose_number(original, readings) == original


def test_integer_column_can_remove_a_spurious_mark_only_with_matching_digits():
    readings = [
        RegionReading("2 813 947", 0.98, "recognizer-a", "clean"),
        RegionReading("2813947", 0.97, "recognizer-b", "wide"),
    ]
    assert choose_number("2.813947", readings) == "2.813947"
    assert (
        choose_number("2.813947", readings, allow_decimal_removal=True) == "2 813 947"
    )
    assert (
        choose_number("2.814947", readings, allow_decimal_removal=True) == "2 813 947"
    )


def test_mixed_script_cleanup_preserves_english_words_and_numbers():
    assert normalize_mixed_script("СYММA НДC 10% OpenAI USB 12.50") == (  # noqa: RUF001
        "СУММА НДС 10% OpenAI USB 12.50"  # noqa: RUF001
    )


def test_rule_cleanup_retains_a_decimal_dot_next_to_digits():
    image = Image.new("L", (180, 60), 255)
    draw = ImageDraw.Draw(image)
    draw.line((0, 0, 0, 59), fill=0, width=2)
    draw.line((0, 59, 179, 59), fill=0, width=2)
    for left in (42, 67, 100, 125):
        draw.rectangle((left, 18, left + 12, 43), fill=0)
    draw.rectangle((88, 40, 91, 43), fill=0)
    cleaned = clean_region(np.asarray(image))
    assert cleaned.shape[0] < image.height
    # Four digit components and the decimal dot survive, while the grid is gone.
    import cv2

    count, _, stats, _ = cv2.connectedComponentsWithStats(
        (cleaned < 128).astype(np.uint8), connectivity=8
    )
    assert count - 1 == 5
    assert min(stats[1:, cv2.CC_STAT_AREA]) == 16


@pytest.mark.parametrize(
    ("reading", "expected"),
    [("15.64", "15.64"), ("1564", "16.64"), ("16.84", "16.84"), ("45.94", "16.64")],
)
def test_receipt_correction_preserves_decimal_scale_and_limits_digit_changes(
    reading, expected
):
    candidates = [
        RegionReading(reading, 0.9, "tesseract-7", "wide"),
        RegionReading(reading, 0.9, "tesseract-13", "wide"),
    ]
    assert choose_line_number("16.64", candidates) == expected


def test_text_refinement_cannot_turn_a_price_decimal_into_two_numbers():
    assert not compatible_text_reading("Цена 94.99", "Цена 94 99")
    assert compatible_text_reading("Цена 94.99", "Цена 94,99")
    assert not compatible_text_reading("Цена 94.99", "Цена 94.98")


def test_extra_spacing_does_not_block_a_complete_grade_reading():
    assert compatible_text_reading("(онио)", "5 (отлично)")
    assert compatible_text_reading("(онио)", "5(отлично)")
    assert not compatible_text_reading("4(хорошо)", "5 (хорошо)")


def test_agreeing_neighbors_allow_words_from_another_actual_reading():
    original = RegionReading("ИТОГО К ОПЛАТЕ СУЧЕТОН СКИДКИ", 0.9, "rapid", "wide")  # noqa: RUF001
    observed = RegionReading("ИТОГО К ОПЛАТЕ С“УЧЕТОМ СКИДКИ", 0.8, "tesseract", "wide")  # noqa: RUF001
    known_words = {"с", "учетом"}  # noqa: RUF001
    chosen = refine_word_choices(
        original, [observed], lambda word: word.casefold() in known_words
    )
    assert chosen.text == "ИТОГО К ОПЛАТЕ С УЧЕТОМ СКИДКИ"  # noqa: RUF001
    assert chosen.confidence == 0.8


def test_dictionary_word_alignment_cannot_replace_an_english_product_name():
    original = RegionReading("Проект OpenAI выполнен", 0.9, "rapid", "clean")
    observed = RegionReading("Проект Опенай выполнен", 0.95, "other", "clean")
    assert (
        refine_word_choices(original, [observed], lambda word: True).text
        == original.text
    )


def test_caption_punctuation_requires_two_independent_recognizers():
    noisy = RegionReading("СУММА _К. ОПЛАТЕ БЕЗ СКИДКИ", 0.92, "iso:ru", "wide")  # noqa: RUF001
    clean = "СУММА К ОПЛАТЕ БЕЗ СКИДКИ"  # noqa: RUF001
    same_model = [
        RegionReading(clean, 0.9, "page", "full"),
        RegionReading(clean, 0.88, "iso:ru", "crop"),
    ]
    assert choose_text_punctuation(noisy, same_model).text == noisy.text
    independent = [*same_model, RegionReading(clean, 0.87, "tesseract-13", "line")]
    assert choose_text_punctuation(noisy, independent).text == clean


@pytest.mark.parametrize(
    "original, proposed",
    [
        ("Report for A. Smith", "Report for A Smith"),
        ("SAVE THE FILE_NAME", "SAVE THE FILE NAME"),
        ("ОПЛАТА ПО СЧЕТУ.", "ОПЛАТА ПО СЧЕТУ"),
        ("СУММА НДС ПО СТАВКЕ 10%", "СУММА НДС ПО СТАВКЕ 10"),  # noqa: RUF001
        ("НАЛОГ ПО СТАВКЕ 10% ИТОГО 23.45", "НАЛОГ ПО СТАВКЕ 10 ИТОГО 23.45%"),  # noqa: RUF001
        ("ДОЛГ ПО СЧЕТУ -23.45", "ДОЛГ ПО СЧЕТУ 23.45-"),
        ("ПЕЧАТНЫЙ ТЕКСТ ПРОВЕРКИ", "Печатный текст проверки"),  # noqa: RUF001
        ("СОХРАНИТЬ _ОТЧЕТ.ДОК ЗДЕСЬ", "СОХРАНИТЬ ОТЧЕТ.ДОК ЗДЕСЬ"),
    ],
)
def test_punctuation_vote_preserves_initials_identifiers_sentence_endings_and_percent(
    original, proposed
):
    reading = RegionReading(original, 0.9, "page", "full")
    candidates = [
        RegionReading(proposed, 0.95, "cyrillic", "clean"),
        RegionReading(proposed, 0.96, "tesseract-13", "line"),
    ]
    assert choose_text_punctuation(reading, candidates).text == original


def test_amount_punctuation_uses_the_value_and_independent_votes():
    amount = RegionReading("=23.45.", 0.9, "page", "full")
    clean = [
        RegionReading("23.45", 0.99, "iso:ru", "clean"),
        RegionReading("23.45", 0.98, "cyrillic", "clean"),
    ]
    assert (
        choose_amount_punctuation(amount, clean, leading_equals=False).text == "=23.45"
    )
    assert (
        choose_amount_punctuation(amount, clean[:1], leading_equals=False).text
        == "=23.45."
    )
    wrong_scale = [
        RegionReading("2345", 0.99, reading.engine, "clean") for reading in clean
    ]
    assert (
        choose_amount_punctuation(amount, wrong_scale, leading_equals=False).text
        == "=23.45."
    )
    missing = RegionReading("*23.45", 0.9, "page", "full")
    assert choose_amount_punctuation(missing, [], leading_equals=False).text == "*23.45"
    assert choose_amount_punctuation(missing, [], leading_equals=True).text == "=23.45"


def test_observed_marker_is_separate_from_the_first_numeric_glyph():
    symbols = [
        SymbolReading("=", 0.99, (3, 5, 12, 10)),
        SymbolReading("2", 0.98, (18, 0, 28, 16)),
    ]
    assert supports_leading_equals(symbols, "=23.45", "23.45")
    # A hallucinated marker cannot reuse the first digit's pixels.
    overlapping = [SymbolReading("=", 0.99, (18, 5, 27, 10)), symbols[1]]
    assert not supports_leading_equals(overlapping, "=23.45", "23.45")
    assert not supports_leading_equals(symbols, "=2345", "23.45")
    assert not supports_leading_equals(symbols, "=23.45", "-23.45")


def test_punctuation_crop_keeps_the_operator_and_excludes_the_next_line():
    image = Image.new("L", (80, 40), 255)
    draw = ImageDraw.Draw(image)
    draw.rectangle((5, 15, 12, 16), fill=0)
    draw.rectangle((5, 20, 12, 21), fill=0)
    draw.rectangle((20, 10, 28, 27), fill=0)
    draw.rectangle((35, 10, 43, 27), fill=0)
    draw.rectangle((20, 37, 60, 39), fill=0)
    cropped = punctuation_line_crop(np.asarray(image))
    # Both leading bars survive; neighboring-line ink is outside the crop.
    import cv2

    _, _, stats, _ = cv2.connectedComponentsWithStats(
        (cropped < 128).astype(np.uint8), connectivity=8
    )
    assert sorted(stats[1:, cv2.CC_STAT_AREA].tolist()) == [16, 16, 162, 162]
    assert cropped.shape[0] == 18
    padded = np.pad(np.asarray(image), ((1, 0), (1, 0)), constant_values=255)
    assert np.array_equal(punctuation_line_crop(padded), cropped)


def test_numeric_refinement_retains_an_observed_asterisk():
    candidates = [
        RegionReading("23.45", 0.9, f"tesseract-{mode}", "wide") for mode in (7, 13)
    ]
    assert choose_line_number("*23.45", candidates) == "*23.45"
