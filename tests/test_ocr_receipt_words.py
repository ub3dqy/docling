# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT
# ruff: noqa: RUF001 -- Latin/Cyrillic lookalikes are intentional regression inputs.

"""Caption agreement must preserve Latin names and count model families once."""

from docling.models.stages.ocr.ocr_receipt_words import (
    choose_cyrillic_caption,
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


def test_receipt_abbreviation_can_touch_the_next_word_without_matching_a_larger_name() -> (
    None
):
    pattern = receipt_word_pattern("Усл.")
    assert (
        pattern.sub("Сум.", "Усл.оплачены; Предусл.оплачены")
        == "Сум.оплачены; Предусл.оплачены"
    )
