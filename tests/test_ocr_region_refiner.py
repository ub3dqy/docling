# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Safety regressions for crop-based corrections of financial numbers."""

import numpy as np
import pytest
from PIL import Image, ImageDraw

from docling.models.stages.ocr.ocr_region_refiner import (
    OcrRegionRefiner,
    RegionReading,
    SymbolReading,
    choose_amount_punctuation,
    choose_line_number,
    choose_number,
    choose_receipt_decimal,
    choose_short_code,
    choose_text_punctuation,
    clean_region,
    compatible_text_reading,
    normalize_mixed_script,
    photo_contrast,
    printed_dash,
    printed_decimal,
    punctuation_line_crop,
    refine_word_choices,
    supports_leading_equals,
    supports_photo_label,
    two_bar_marker,
)
from docling.utils.ocr_region_pixels import (
    light_on_dark,
    terminal_exclamation,
    trim_light_text,
)


@pytest.mark.parametrize("mark", ["bang", "bar", "colon", "question"])
def test_terminal_mark_needs_a_separate_full_height_stem_and_lower_dot(mark):
    image = Image.new("L", (190, 42), "white")
    draw = ImageDraw.Draw(image)
    for x in range(10, 140, 18):
        draw.rectangle((x, 8, x + 10, 30), fill=0)
    if mark in {"bang", "bar"}:
        draw.rectangle((165, 8, 168, 25 if mark == "bang" else 34), fill=0)
    if mark in {"bang", "colon", "question"}:
        draw.rectangle((165, 30, 168, 33), fill=0)
    if mark == "colon":
        draw.rectangle((165, 14, 168, 17), fill=0)
    if mark == "question":
        draw.rectangle((159, 8, 172, 14), fill=0)
        draw.rectangle((166, 14, 170, 24), fill=0)
    assert terminal_exclamation(np.asarray(image)) == (mark == "bang")


@pytest.mark.parametrize("confidence", [0.65, 0.83])
@pytest.mark.parametrize(
    "original",
    ["Приходите к нам снова!", "Обращайтесь снова?", "Стоимость, ед."],
)
def test_observed_sentence_ending_needs_stronger_votes_when_already_confident(
    confidence,
    original,
):
    from types import SimpleNamespace

    refiner = object.__new__(OcrRegionRefiner)
    refiner._dictionary = SimpleNamespace(word_is_known=lambda _: True)
    candidates = [
        RegionReading(original[:-1], 0.999, "cyrillic-v6-medium", "row-source"),
        RegionReading(original[:-1], 0.85, "cyrillic-v6-tiny", "row-source"),
    ]
    selected = refiner._choose_text(original, confidence, candidates)
    assert selected.text == (original if confidence >= 0.8 else original[:-1])


def test_multiple_preprocessings_of_one_recognizer_are_not_independent_votes():
    readings = [
        RegionReading("32 795", 0.99, "recognizer-a", "clean"),
        RegionReading("32795", 0.99, "recognizer-a", "wide"),
        RegionReading("32795", 0.99, "recognizer-a", "threshold"),
        RegionReading("32 785", 0.97, "recognizer-b", "clean"),
    ]
    assert choose_number("32 785", readings) == "32 785"


def test_tesseract_segmentation_modes_are_one_numeric_vote():
    readings = [
        RegionReading("32 795", 0.99, "tesseract-7", "source"),
        RegionReading("32 795", 0.99, "tesseract-13", "source"),
    ]
    assert choose_number("32 785", readings) == "32 785"
    readings.append(RegionReading("32 795", 0.99, "iso:ru", "source"))
    assert choose_number("32 785", readings) == "32 795"


@pytest.mark.parametrize("source_confidence", [0.99991, 0.95])
def test_sentence_ending_vote_uses_each_recognizers_strongest_reading(
    source_confidence,
):
    from types import SimpleNamespace

    refiner = object.__new__(OcrRegionRefiner)
    refiner._dictionary = SimpleNamespace(word_is_known=lambda _: True)
    original = "Масса, кг."  # noqa: RUF001 -- Russian unit caption
    readings = [
        RegionReading(original[:-1], 0.99995, "recognizer-a", "clean"),
        RegionReading(original, 0.99979, "recognizer-a", "source"),
        RegionReading(original[:-1], 0.96686, "recognizer-b", "clean"),
        RegionReading(original, source_confidence, "recognizer-b", "source"),
    ]
    assert refiner._choose_text(original, 0.88, readings).text == (
        original if source_confidence > 0.96686 else original[:-1]
    )


@pytest.mark.parametrize("original_ending,correct_ending", [("?", "!"), ("!", "?")])
def test_period_protection_retains_established_question_and_exclamation_corrections(
    original_ending,
    correct_ending,
):
    from types import SimpleNamespace

    refiner = object.__new__(OcrRegionRefiner)
    refiner._dictionary = SimpleNamespace(word_is_known=lambda _: True)
    original = "Read : carefully" + original_ending
    corrected = "Read carefully" + correct_ending
    readings = [
        RegionReading(corrected, 0.99995, "recognizer-a", "clean"),
        RegionReading(original, 0.99979, "recognizer-a", "source"),
        RegionReading(corrected, 0.96686, "recognizer-b", "clean"),
        RegionReading(original, 0.99991, "recognizer-b", "source"),
    ]
    assert refiner._choose_text(original, 0.88, readings).text == corrected


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


@pytest.mark.parametrize("paper", [255, 170])
def test_rule_cleanup_retains_a_decimal_dot_next_to_digits(paper):
    image = Image.new("L", (180, 60), paper)
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


def test_broken_cell_rules_do_not_hide_a_faint_short_dash():
    image = Image.new("L", (450, 150), 255)
    draw = ImageDraw.Draw(image)
    for start, end in ((0, 26), (51, 62), (94, 149)):
        draw.line((449, start, 449, end), fill=80, width=2)
    draw.rectangle((426, 122, 437, 129), fill=155)
    assert printed_dash(np.asarray(image))


def test_skewed_tall_cell_keeps_its_dash_beside_an_inward_border():
    image = Image.new("L", (300, 180), 255)
    draw = ImageDraw.Draw(image)
    draw.line((288, 20, 288, 179), fill=70, width=4)
    draw.line((120, 5, 280, 5), fill=100)
    draw.rectangle((264, 165, 275, 169), fill=80)
    assert printed_dash(np.asarray(image))


def test_raster_rounding_of_a_short_horizontal_dash_is_not_a_round_dot():
    image = Image.new("L", (450, 75), 255)
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((426, 47, 435, 53), radius=2, fill=110)
    assert printed_dash(np.asarray(image))
    draw.rectangle((426, 47, 435, 53), fill=255)
    draw.ellipse((426, 47, 432, 53), fill=110)
    assert not printed_dash(np.asarray(image))


def test_partial_thin_border_does_not_hide_the_dash_in_a_tall_row():
    image = Image.new("L", (680, 320), 255)
    draw = ImageDraw.Draw(image)
    draw.rectangle((2, 10, 4, 102), fill=80)
    draw.rectangle((672, 240, 674, 319), fill=80)
    draw.rectangle((642, 289, 663, 296), fill=100)
    assert printed_dash(np.asarray(image))


@pytest.mark.parametrize("text", ["1", "7", "-1", "-7", "0", "-10", "."])
def test_rule_cleanup_does_not_turn_numeric_ink_into_a_dash(text):
    import cv2

    array = np.full((80, 240), 255, dtype=np.uint8)
    cv2.rectangle(array, (0, 0), (239, 79), 80, 2)
    cv2.putText(array, text, (155, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 0, 2)
    assert not printed_dash(array)


def test_short_photo_labels_need_independent_readings_and_preserve_real_numbers():
    candidate = RegionReading("ИЗО", 0.67, "tesseract-7", "photo")
    readings = [candidate, RegionReading("изо", 0.58, "iso:ru", "photo")]

    def known(word: str) -> bool:
        return word.casefold() == "изо"

    assert supports_photo_label("130", candidate, readings, known)
    assert not supports_photo_label("130", candidate, [candidate], known)
    for original in ("1300", "13.0", "-130", "130%", "130 мм"):
        assert not supports_photo_label(original, candidate, readings, known)


@pytest.mark.parametrize("text,expected", [("-", True), ("-1", False), (".", False)])
def test_tall_rows_keep_a_low_baseline_dash_without_dropping_numeric_glyphs(
    text, expected
):
    import cv2

    array = np.full((200, 300), 255, dtype=np.uint8)
    cv2.rectangle(array, (0, 0), (299, 199), 80, 2)
    cv2.putText(array, text, (235, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 0, 2)
    assert printed_dash(array) == expected


def test_light_letter_detection_uses_contrast_on_medium_gray_material():
    light = np.full((50, 140), 136, dtype=np.uint8)
    light[12:32, 15:125] = 240
    light[47:50, 15:125] = 240
    assert light_on_dark(light)
    cropped = trim_light_text(light)
    assert int((cropped > 180).sum()) == 20 * 110
    paper = np.full((50, 140), 136, dtype=np.uint8)
    paper[12:32, 15:125] = 30
    assert not light_on_dark(paper)


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


def test_punctuation_recheck_preserves_confident_original_amount_digits():
    candidates = [
        RegionReading("171.97", 0.99, f"tesseract-{mode}", "wide") for mode in (7, 13)
    ]
    assert choose_line_number("170.97", candidates, confidence=0.99) == "170.97"
    assert choose_line_number("170.97", candidates, confidence=0.8) == "171.97"


def test_text_refinement_cannot_turn_a_price_decimal_into_two_numbers():
    assert not compatible_text_reading("Цена 94.99", "Цена 94 99")
    assert compatible_text_reading("Цена 94.99", "Цена 94,99")
    assert not compatible_text_reading("Цена 94.99", "Цена 94.98")


def test_right_aligned_dash_survives_cell_frame_and_extra_heading_space():
    image = Image.new("L", (400, 150), 255)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 399, 149), outline=0, width=2)
    draw.rectangle((375, 120, 386, 123), fill=80)
    assert printed_dash(np.asarray(image))
    # A small square ink spot at the same position is not a printed dash.
    draw.rectangle((375, 120, 386, 123), fill=255)
    draw.rectangle((379, 119, 382, 122), fill=0)
    assert not printed_dash(np.asarray(image))


def test_dash_at_clipped_right_edge_is_not_confused_with_bottom_rule():
    image = Image.new("L", (400, 100), 255)
    draw = ImageDraw.Draw(image)
    draw.line((0, 0, 0, 99), fill=0, width=2)
    draw.line((0, 99, 399, 99), fill=0, width=2)
    assert not printed_dash(np.asarray(image))
    draw.rectangle((390, 60, 399, 64), fill=80)
    assert printed_dash(np.asarray(image))


def test_blurred_dash_needs_a_horizontal_core_and_not_just_a_dark_spot():
    image = Image.new("L", (200, 60), 255)
    draw = ImageDraw.Draw(image)
    draw.rectangle((175, 35, 186, 42), fill=150)
    draw.rectangle((175, 39, 186, 42), fill=110)
    draw.rectangle((179, 39, 182, 42), fill=50)
    assert printed_dash(np.asarray(image))
    image = Image.new("L", (200, 60), 255)
    ImageDraw.Draw(image).ellipse((175, 35, 182, 42), fill=110)
    assert not printed_dash(np.asarray(image))


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


def test_expression_marker_can_use_a_separate_confidence_floor_with_pixel_evidence():
    symbols = [
        SymbolReading("=", 0.945, (3, 5, 12, 10)),
        SymbolReading("2", 0.98, (18, 0, 28, 16)),
    ]
    assert not supports_leading_equals(symbols, "=23.45", "23.45")
    assert supports_leading_equals(symbols, "=23.45", "23.45", min_confidence=0.94)


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


def test_receipt_decimal_requires_observed_ink_and_independent_readings():
    readings = [
        RegionReading("103.18", 0.9, "iso:ru", "crop"),
        RegionReading("103.48", 0.95, "tesseract-7", "wide"),
        RegionReading("103.48", 0.95, "tesseract-13", "wide"),
    ]
    assert choose_receipt_decimal("10318", readings, printed_dot=True) == "103.18"
    assert choose_receipt_decimal("10318", readings, printed_dot=False) == "10318"
    assert choose_receipt_decimal("10318", readings[:2], printed_dot=True) == "10318"
    assert choose_receipt_decimal("10319", readings, printed_dot=True) == "10319"
    assert choose_receipt_decimal("-10318", readings, printed_dot=True) == "-10318"


def test_decimal_can_use_consistent_models_when_tesseract_is_uncertain():
    one_model = [
        RegionReading("107.21", 0.94, "iso:ru", variant)
        for variant in ("clean", "wide")
    ]
    assert choose_receipt_decimal("10721", one_model, printed_dot=True) == "10721"
    two_models = [
        *one_model,
        *[
            RegionReading("107.21", 0.94, "ch", variant)
            for variant in ("clean", "wide")
        ],
    ]
    assert choose_receipt_decimal("10721", two_models, printed_dot=True) == "107.21"
    assert choose_receipt_decimal("10721", two_models, printed_dot=False) == "10721"
    assert choose_receipt_decimal("10722", two_models, printed_dot=True) == "10722"


@pytest.mark.parametrize("dot_y,expected", [(25, True), (1, False), (None, False)])
def test_printed_decimal_must_be_in_the_numeric_baseline_gap(dot_y, expected):
    image = Image.new("L", (105, 40), 255)
    draw = ImageDraw.Draw(image)
    for index in range(5):
        draw.rectangle((8 + index * 20, 8, 14 + index * 20, 28), fill=0)
    if dot_y is not None:
        draw.rectangle((59, dot_y, 61, dot_y + 2), fill=0)
    assert printed_decimal(np.asarray(image), 5) == expected


def test_a_printed_dash_is_not_a_decimal_dot_or_an_empty_field():
    image = Image.new("L", (100, 40), 255)
    assert not printed_dash(np.asarray(image))
    ImageDraw.Draw(image).rectangle((35, 18, 65, 20), fill=0)
    assert printed_dash(np.asarray(image))
    image = Image.new("L", (100, 40), 255)
    ImageDraw.Draw(image).rectangle((48, 18, 50, 20), fill=0)
    assert not printed_dash(np.asarray(image))


@pytest.mark.parametrize("mark", ["dash", "dot", "empty"])
def test_photo_contrast_exposes_faint_ink_without_turning_a_dot_into_a_dash(mark):
    image = Image.new("L", (90, 32), 150)
    draw = ImageDraw.Draw(image)
    if mark == "dash":
        draw.rectangle((35, 15, 47, 17), fill=120)
    elif mark == "dot":
        draw.rectangle((41, 15, 43, 17), fill=120)
    assert printed_dash(photo_contrast(np.asarray(image))) == (mark == "dash")


@pytest.mark.parametrize(
    "sigma_x,sigma_y,darkest", [(1.4, 1.0, 80), (1.8, 1.2, 140), (1.8, 1.4, 120)]
)
def test_camera_blur_keeps_a_small_dash_distinct_from_a_round_spot(
    sigma_x, sigma_y, darkest
):
    y, x = np.mgrid[:34, :102]

    def blurred_mark(vertical_sigma):
        signal = np.exp(
            -(
                (x - 50) ** 2 / (2 * sigma_x**2)
                + (y - 17) ** 2 / (2 * vertical_sigma**2)
            )
        )
        return np.rint(210 - (210 - darkest) * signal).astype(np.uint8)

    assert printed_dash(blurred_mark(sigma_y))
    assert not printed_dash(blurred_mark(sigma_x))


def test_a_minus_cannot_pass_the_two_bar_equals_check():
    marker = np.full((7, 10), 255, dtype=np.uint8)
    marker[1:3, :] = 0
    assert not two_bar_marker(marker)
    marker[5:7, :] = 0
    assert two_bar_marker(marker)


def test_equals_marker_ignores_an_upper_row_fragment_and_ink_between_its_bars():
    marker = np.full((30, 14), 255, dtype=np.uint8)
    marker[:4, :12] = 0
    marker[22:25, :13] = 0
    marker[27:30, :] = 0
    marker[25:27, 3:5] = 0
    marker[25:27, 9:13] = 0
    assert two_bar_marker(marker)
    marker[27:30, :] = 255
    assert not two_bar_marker(marker)


def test_an_equals_in_a_previous_row_does_not_turn_a_current_minus_into_equals():
    marker = np.full((40, 14), 255, dtype=np.uint8)
    marker[1:3, :] = 0
    marker[5:7, :] = 0
    marker[30:33, :] = 0
    assert not two_bar_marker(marker)
    marker[30:33, :] = 255
    marker[30:33, 5:9] = 0
    assert not two_bar_marker(marker)


def test_short_code_requires_full_digits_and_independent_engines():
    readings = [
        RegionReading("021", 0.98, "iso:ru", "clean"),
        RegionReading("021", 0.98, "tesseract-7", "clean"),
    ]
    assert choose_short_code("", readings) == "021"
    assert choose_short_code("", readings[:1]) == ""
    assert choose_short_code("1234", readings) == "1234"
    incomplete = [
        RegionReading("21", r.confidence, r.engine, r.variant) for r in readings
    ]
    assert choose_short_code("", incomplete) == ""
