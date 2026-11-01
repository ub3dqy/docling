# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Safety regressions for crop-based corrections of financial numbers."""

import numpy as np
import pytest
from PIL import Image, ImageDraw

from docling.models.stages.ocr.ocr_region_refiner import (
    RegionReading,
    choose_line_number,
    choose_number,
    clean_region,
    compatible_text_reading,
    normalize_mixed_script,
    refine_word_choices,
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
