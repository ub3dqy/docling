# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""A numeric disagreement needs repeated source glyphs and unambiguous bounds."""

import numpy as np
import pytest
from PIL import Image, ImageDraw

from docling.utils.ocr_digit_templates import (
    decimal_expression_equals,
    decimal_suffix_region,
    repeated_digit_support,
)

pytest.importorskip("cv2")


def _amount(text: str) -> np.ndarray:
    image = Image.new("L", (len(text) * 19 + 8, 38), 255)
    draw = ImageDraw.Draw(image)
    segments = {
        "4": "bcfg",
        "8": "abcdefg",
        "6": "acdefg",
        "3": "abcdg",
        "9": "abcdfg",
    }
    bars = {
        "a": (3, 5, 14, 7),
        "b": (12, 5, 14, 17),
        "c": (12, 16, 14, 29),
        "d": (3, 27, 14, 29),
        "e": (3, 16, 5, 29),
        "f": (3, 5, 5, 17),
        "g": (3, 16, 14, 18),
    }
    for index, char in enumerate(text):
        offset = index * 19
        if char == ".":
            draw.rectangle((offset + 6, 27, offset + 8, 29), fill=0)
        else:
            for segment in segments[char]:
                left, top, right, bottom = bars[segment]
                draw.rectangle((left + offset, top, right + offset, bottom), fill=0)
    return np.asarray(image)


def test_repeated_source_digits_can_verify_an_observed_one_digit_alternative() -> None:
    assert repeated_digit_support(_amount("48886"), "48836", "48886")
    assert not repeated_digit_support(_amount("48836"), "48836", "48886")
    assert not repeated_digit_support(_amount("48886"), "48836", "48896")
    assert not repeated_digit_support(_amount("48886"), "48836", "488.86")
    assert repeated_digit_support(_amount("69994"), "69934", "69994")
    assert not repeated_digit_support(_amount("69934"), "69934", "69994")


def test_decimal_suffix_needs_the_printed_dot_and_complete_digit_count() -> None:
    amount = _amount("488.86")
    region = decimal_suffix_region(amount, 3)
    assert region is not None and region.size > 0
    assert decimal_suffix_region(_amount("48886"), 3) is None
    assert decimal_suffix_region(amount, 4) is None


@pytest.mark.parametrize("equals", [False, True])
@pytest.mark.parametrize("horizontal_scale", [1, 2, 3])
def test_decimal_expression_preserves_a_real_minus(
    equals: bool, horizontal_scale: int
) -> None:
    amount = _amount("488.86")
    image = Image.new("L", (amount.shape[1] + 20, amount.shape[0]), 255)
    image.paste(Image.fromarray(amount), (20, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle((9, 19, 18, 21), fill=0)
    if equals:
        draw.rectangle((9, 13, 18, 15), fill=0)
    image = image.resize((image.width * horizontal_scale, image.height))
    assert decimal_expression_equals(np.asarray(image), 3) == equals


def test_a_colon_before_an_amount_cannot_be_promoted_to_equals():
    amount = _amount("488.86")
    image = Image.new("L", (amount.shape[1] + 20, amount.shape[0]), 255)
    image.paste(Image.fromarray(amount), (20, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle((9, 9, 15, 15), fill=0)
    draw.rectangle((9, 20, 15, 26), fill=0)
    assert not decimal_expression_equals(np.asarray(image), 3)
    assert not decimal_expression_equals(np.asarray(image), 3, allow_uneven_bars=True)


@pytest.mark.parametrize("marker", ["bleeding-equals", "minus", "colon", "plus"])
def test_bleeding_equals_requires_two_horizontal_bars_before_the_decimal(marker):
    amount = _amount("488.86")
    image = Image.new("L", (amount.shape[1] + 32, 38), 255)
    image.paste(Image.fromarray(amount), (32, 0))
    draw = ImageDraw.Draw(image)
    if marker == "bleeding-equals":
        draw.rectangle((8, 10, 27, 12), fill=0)
        draw.rectangle((8, 15, 27, 27), fill=0)
    elif marker == "minus":
        draw.rectangle((8, 15, 27, 19), fill=0)
    elif marker == "colon":
        draw.rectangle((15, 10, 21, 16), fill=0)
        draw.rectangle((15, 23, 21, 29), fill=0)
    else:
        draw.rectangle((8, 17, 27, 19), fill=0)
        draw.rectangle((16, 9, 19, 27), fill=0)
    assert not decimal_expression_equals(np.asarray(image), 3)
    assert decimal_expression_equals(np.asarray(image), 3, allow_uneven_bars=True) == (
        marker == "bleeding-equals"
    )
