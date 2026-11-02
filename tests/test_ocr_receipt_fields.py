# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Full-field rereading must retain amounts, tax rates and real symbols."""

import numpy as np
import pytest
from PIL import Image, ImageDraw

from docling.models.stages.ocr.ocr_cyrillic_text import CyrillicTextReader
from docling.models.stages.ocr.ocr_receipt_fields import (
    preserves_known_caption,
    retain_field_numbers,
    triangle_is_line_noise,
)
from docling.utils.ocr_character_paths import CharacterPath


def test_a_complete_caption_keeps_the_existing_value_and_decimal_style():
    assert retain_field_numbers("VAT 7% 456.78", "VAT 7% =456,78") == "VAT 7% =456.78"
    assert retain_field_numbers("Card 1200.00", "*Card =1200.00") == "*Card =1200.00"
    assert retain_field_numbers("Card", "*Card") == "*Card"
    assert retain_field_numbers("Total =123.45", "Total=123,45") == "Total =123.45"


@pytest.mark.parametrize(
    "original,candidate",
    [
        ("VAT 7% 456.78", "VAT 7% =456.79"),
        ("VAT 7% 456.78", "VAT 7% =456.780"),
        ("VAT 7% 456.78", "VAT 17% =456.78"),
        ("VAT 7% 456.78", "VAT 7 =456.78"),
        ("Total -456.78", "Total =456.78"),
        ("Total 456.78", "Total -456.78"),
        ("*Card =456.78", "Card 456.78"),
        ("Card 456.78", "Card *456.78"),
        ("Card 456.78", "**Card =456.78"),
        ("Total 456.78", "Total ==456.78"),
        ("VAT 7% =456.78", "VAT =7% 456.78"),
        ("VAT 7% 456.78", "VAT 7 456.78%"),
        ("Total 2\u00d7456.78", "Total 2 456.78"),
        ("Total: 456.78", "Total =456.78"),
        ("Total 456.78 $", "Total 456.78"),
        ("Total (456.78)", "Total 456.78 ()"),
    ],
)
def test_rereading_cannot_change_values_scale_rates_or_meaningful_operators(
    original, candidate
):
    assert retain_field_numbers(original, candidate) is None


def test_separator_debris_is_not_a_triangle_but_a_real_triangle_is_retained():
    real = Image.new("L", (105, 28), 255)
    draw = ImageDraw.Draw(real)
    for x, y in ((45, 7), (37, 19), (53, 19)):
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=0)
    assert not triangle_is_line_noise(np.asarray(real))
    noise = Image.new("L", (105, 28), 255)
    draw = ImageDraw.Draw(noise)
    draw.line((30, 13, 43, 9, 55, 15, 65, 12), fill=0, width=3)
    draw.rectangle((10, 17, 13, 20), fill=0)
    draw.rectangle((88, 15, 92, 19), fill=0)
    assert triangle_is_line_noise(np.asarray(noise))


@pytest.mark.parametrize(
    "preferred,word_choice,dropped_marker",
    [
        ("VAT7% =456.78", "VAT 7% =456,78", "VAT7%456.78"),
        ("*Плат.картой=456.78", "*ПЛАТ.КАРТОЙ =456.78", "ПЛАТ.КАРТОЙ =456.78"),
        ("Итого -456.78", "ИТОГО -456.78", "ИТОГО =456.78"),
    ],
)
def test_character_confidence_cannot_reward_dropping_a_fiscal_marker(
    monkeypatch, preferred, word_choice, dropped_marker
):
    reader = CyrillicTextReader.__new__(CyrillicTextReader)
    view = [
        ("source", CharacterPath(preferred, 0.89, -1.0)),
        ("source-characters", CharacterPath(word_choice, 0.91, -1.3)),
        ("source-characters", CharacterPath(dropped_marker, 0.97, -1.8)),
    ]
    # Replace only the neural-model boundary; keep the real field-path filter.
    monkeypatch.setattr(reader, "read_views", lambda *args, **kwargs: iter([view]))
    result = reader.read_region(
        np.zeros((8, 40), dtype=np.uint8), known=lambda word: True
    )
    assert [path.text for _, path in result] == [preferred, word_choice]


@pytest.mark.parametrize(
    "original,candidate,expected",
    [
        ("БЕЗНАЛИЧНЫМИ =456.78", "БЕЗНАЛИЧНЫ =456.78", False),
        ("Плат.картой 456.78", "*ПЛАТ.КАРТОЙ =456.78", True),
        ("БЕЗНАЛИЧНЫНИ =456.78", "БЕЗНАЛИЧНЫМИ =456.78", True),
        pytest.param(
            "СУМА НДС 7% 456.78",  # noqa: RUF001 -- Russian fiscal caption
            "СУММА НДС 7% =456.78",  # noqa: RUF001 -- Russian fiscal caption
            True,
        ),
        ("Наличными 456.78", "БЕЗНАЛИЧНЫМИ =456.78", True),
    ],
)
def test_wide_field_reading_keeps_a_known_caption_and_can_repair_an_unknown_one(
    original, candidate, expected
):
    known = {
        "безналичными",
        "безналичны",
        "наличными",
        "плат",
        "картой",
        "\u0441\u0443\u043c\u0430",
    }
    assert (
        preserves_known_caption(
            original, candidate, lambda word: word.casefold() in known
        )
        == expected
    )
