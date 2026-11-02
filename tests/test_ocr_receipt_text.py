# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Caption verification cannot invent fiscal words, amounts or tax rates."""

# ruff: noqa: RUF001 -- Deliberately damaged Cyrillic OCR labels.

import pytest

from docling.models.stages.ocr.ocr_receipt_text import (
    choose_amount_marks,
    choose_caption,
)
from docling.models.stages.ocr.ocr_region_refiner import RegionReading


@pytest.mark.parametrize("rate", [2, 5, 18, 22])
def test_percent_glyph_repair_needs_an_independently_observed_same_rate(rate):
    original = f"СУННА НС {rate}7"
    readings = [
        RegionReading(f"СУММА НДС {rate}%", 0.98, "cyrillic-v6-medium", "source"),
        RegionReading(f"СУННА НДС {rate}%", 0.7, "tesseract", "clean"),
    ]

    def known(word):
        return False

    assert choose_caption(original, readings[:1], known) is None
    selected = choose_caption(original, readings, known)
    assert selected is not None and selected.text == f"СУММА НДС {rate}%"
    assert choose_caption(f"СУННА НС {rate + 1}7", readings, known) is None
    assert choose_caption(f"СУННА НС {rate}7.00", readings, known) is None
    assert choose_caption(f"СУННА НС {rate}7%", readings, known) is None
    assert choose_caption(f"код {rate}7", readings, known) is None


def test_known_cash_caption_cannot_be_replaced_with_cashless_caption():
    readings = [
        RegionReading("БЕЗНАЛИЧНЫМИ", 0.99, "cyrillic-v6-medium", "source"),
        RegionReading("БЕЗНАЛИЧНЫМИ", 0.99, "cyrillic-v6-tiny", "source"),
    ]
    assert choose_caption("НАЛИЧНЫМИ", readings, lambda word: True) is None


def test_amount_marker_needs_two_models_and_preserves_values_and_negative_sign():
    readings = [
        RegionReading("=127.30", 0.99, engine, "source")
        for engine in ("cyrillic-v6-medium", "cyrillic-v6-tiny")
    ]
    assert choose_amount_marks("+127.30", readings).text == "=127.30"
    for original in (
        "-127.30",
        "127.31",
        "127.300",
        "12730",
        "20% 127.30",
        "2*127.30",
        "*127.30",
    ):
        assert choose_amount_marks(original, readings) is None
    assert choose_amount_marks("+127.30", readings[:1]) is None
