# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT
# ruff: noqa: RUF001 -- Latin/Cyrillic lookalikes are intentional regression inputs.

"""Caption agreement must preserve Latin names and count model families once."""

from docling.models.stages.ocr.ocr_receipt_words import (
    choose_cyrillic_caption,
    choose_observed_text,
    compatible_receipt_text,
    preserves_receipt_codes,
    receipt_word_pattern,
)
from docling.models.stages.ocr.ocr_region_refiner import RegionReading


def test_caption_lookalikes_need_two_independent_observations() -> None:
    candidates = [
        RegionReading("КАССА", 0.62, "iso:ru", "clean"),
        RegionReading("касса", 0.58, "cyrillic", "wide"),
    ]
    assert (
        choose_cyrillic_caption(
            "KACCA", candidates, lambda text: text.casefold() == "касса"
        )
        is not None
    )
    assert choose_cyrillic_caption("KACCA", candidates[:1], lambda _: True) is None
    same_model = [
        RegionReading("КАССА", 0.62, "iso:ru", variant) for variant in ("clean", "wide")
    ]
    assert choose_cyrillic_caption("KACCA", same_model, lambda _: True) is None
    assert choose_cyrillic_caption("COFFEE", candidates, lambda _: True) is None
    assert choose_cyrillic_caption("KACCA12", candidates, lambda _: True) is None
    unrelated = [
        RegionReading("НАС", 0.9, engine, "clean") for engine in ("iso:ru", "cyrillic")
    ]
    assert choose_cyrillic_caption("HAC", unrelated, lambda _: True) is None


def test_a_single_digit_lookalike_in_a_fiscal_word_needs_source_agreement():
    readings = [
        RegionReading("чек", 0.98, "cyrillic-v6-medium", "source"),
        RegionReading("ЧЕК", 0.96, "cyrillic", "clean"),
    ]
    selected = choose_cyrillic_caption("4ek", readings, lambda _: True)
    assert selected is not None and selected.text.casefold() == "чек"
    assert choose_cyrillic_caption("4ek", readings[:1], lambda _: True) is None
    assert choose_cyrillic_caption("4ek12", readings, lambda _: True) is None


def test_long_source_text_cannot_replace_a_real_short_serial_number():
    import numpy as np
    from PIL import Image, ImageDraw

    readings = [
        RegionReading("Спасибо за покупку", 0.96, engine, "source")
        for engine in ("cyrillic-v6-medium", "cyrillic-v6-tiny")
    ]

    def source(glyphs):
        image = Image.new("L", (glyphs * 14 + 12, 32), "white")
        draw = ImageDraw.Draw(image)
        for x in range(6, image.width - 6, 14):
            draw.rectangle((x, 6, x + 7, 25), fill=0)
        return np.asarray(image)

    assert (
        choose_observed_text("10U09", 0.4, source(5), readings, lambda _: True) is None
    )
    selected = choose_observed_text("10U09", 0.4, source(17), readings, lambda _: True)
    assert selected is not None and selected.text == "Спасибо за покупку"
    assert (
        choose_observed_text("10U09", 0.4, source(17), readings[:1], lambda _: True)
        is None
    )


def test_receipt_abbreviation_can_touch_the_next_word_without_matching_a_larger_name() -> (
    None
):
    pattern = receipt_word_pattern("Усл.")
    assert (
        pattern.sub("Сум.", "Усл.оплачены; Предусл.оплачены")
        == "Сум.оплачены; Предусл.оплачены"
    )


def test_text_refinement_cannot_drop_a_letter_from_a_company_code():
    assert preserves_receipt_codes("Клиника И27", "Клиника И-27")
    assert not preserves_receipt_codes("Клиника И27", "Клиника -27")
    assert preserves_receipt_codes("Device USB42", "Device USB-42")
    assert not preserves_receipt_codes("Device USB42", "Device USB43")


def test_caption_reading_cannot_import_arithmetic_from_a_neighboring_amount():
    assert compatible_receipt_text("СУННА БЕЗ НДС", "СУММА БЕЗ НДС")
    for candidate in ("СУММА БЕЗ НДС *", "СУММА БЕЗ НДС =", "СУММА БЕЗ НДС 1"):
        assert not compatible_receipt_text("СУННА БЕЗ НДС", candidate)
    assert compatible_receipt_text("ДОЛГ -23.45", "ДОЛГ -23.45")
    assert compatible_receipt_text("СУММА 23.45", "СУММА =23.45")
