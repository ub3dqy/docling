# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""A numeric disagreement needs repeated source glyphs and unambiguous bounds."""

import numpy as np
import pytest
from PIL import Image, ImageDraw

from docling.utils.ocr_digit_templates import (
    connected_decimal_equals,
    decimal_expression_equals,
    decimal_suffix_region,
    printed_multiplication,
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


@pytest.mark.parametrize("scale", [1, 2, 3])
@pytest.mark.parametrize("split_integer", [False, True])
@pytest.mark.parametrize("marker", ["=", "-", "+", ":", "none"])
def test_touching_thermal_marker_needs_both_bars_and_a_complete_amount(
    marker, scale, split_integer
):
    image = Image.new("L", (104, 40), 255)
    draw = ImageDraw.Draw(image)
    for left in [32, 42, 52, 72, 82]:
        draw.rectangle((left, 8, left + 9, 27), fill=0)
        draw.rectangle((left + 2, 10, left + 7, 15), fill=255)
        draw.rectangle((left + 2, 19, left + 7, 25), fill=255)
    draw.rectangle((66, 25, 68, 27), fill=0)
    if marker in ["=", "-", "+"]:
        draw.rectangle((24, 22, 31, 24), fill=0)
    if marker == "=":
        draw.rectangle((24, 17, 31, 19), fill=0)
    if marker == "+":
        draw.rectangle((27, 17, 29, 27), fill=0)
    if marker == ":":
        draw.rectangle((28, 15, 31, 18), fill=0)
        draw.rectangle((28, 22, 31, 25), fill=0)
    if split_integer:
        # One integer glyph is detached from the other two. The marker still
        # touches the first glyph; component count is not the digit count.
        draw.line((41, 8, 41, 27), fill=255)
    image = image.resize((image.width * scale, image.height * scale))
    gray = np.asarray(image)
    assert connected_decimal_equals(gray, 3) == (marker == "=")
    assert not connected_decimal_equals(gray, 2)
    assert not connected_decimal_equals(gray, 4)
    # Removing the real decimal point cannot be repaired by a marker guess.
    draw = ImageDraw.Draw(image)
    draw.rectangle((66 * scale, 25 * scale, 69 * scale - 1, 28 * scale - 1), fill=255)
    assert not connected_decimal_equals(np.asarray(image), 3)


@pytest.mark.parametrize("scale", [1, 2, 3])
@pytest.mark.parametrize("equals", [True, False])
def test_wide_connected_digits_use_the_fractional_pitch_without_losing_signs(
    scale, equals
):
    image = Image.new("L", (132, 40), 255)
    draw = ImageDraw.Draw(image)
    for left in [32, 48, 64, 92, 108]:
        draw.rectangle((left, 8, left + 13, 27), fill=0)
        draw.rectangle((left + 2, 10, left + 11, 25), fill=255)
    # Two integer digits bleed together; the two fractional glyphs stay separate.
    draw.line((61, 20, 64, 20), fill=0)
    draw.rectangle((83, 25, 86, 27), fill=0)
    draw.rectangle((14, 21, 28, 23), fill=0)
    if equals:
        draw.rectangle((14, 15, 28, 17), fill=0)
        # A narrow ink bridge must not hide the two full horizontal strokes.
        draw.rectangle((18, 17, 25, 21), fill=0)
    image = image.resize((image.width * scale, image.height * scale))
    gray = np.asarray(image)
    assert decimal_suffix_region(gray, 3) is not None
    assert decimal_expression_equals(gray, 3, allow_uneven_bars=True) == equals
    draw = ImageDraw.Draw(image)
    draw.rectangle((83 * scale, 25 * scale, 87 * scale - 1, 28 * scale - 1), fill=255)
    assert decimal_suffix_region(np.asarray(image), 3) is None


@pytest.mark.parametrize("scale", [1, 2, 3])
@pytest.mark.parametrize(
    "kind", ["cross", "plus", "minus", "colon", "digit", "no-quantity"]
)
def test_a_missing_multiplication_needs_its_source_arms_and_quantity(kind, scale):
    image = Image.new("L", (82, 36), 255)
    draw = ImageDraw.Draw(image)
    for left in [8, 16, 24, 40, 48, 64]:
        draw.rectangle((left, 8, left + 7, 25), fill=0)
        draw.rectangle((left + 2, 10, left + 5, 23), fill=255)
    draw.rectangle((34, 23, 36, 25), fill=0)
    if kind in ["cross", "no-quantity"]:
        draw.line((56, 12, 63, 20), fill=0, width=2)
        draw.line((56, 20, 63, 12), fill=0, width=2)
    elif kind == "plus":
        draw.rectangle((56, 16, 63, 17), fill=0)
        draw.rectangle((59, 12, 60, 21), fill=0)
    elif kind == "minus":
        draw.rectangle((56, 16, 63, 17), fill=0)
    elif kind == "colon":
        draw.rectangle((59, 12, 61, 14), fill=0)
        draw.rectangle((59, 20, 61, 22), fill=0)
    else:
        draw.rectangle((59, 8, 61, 25), fill=0)
    if kind == "no-quantity":
        draw.rectangle((64, 8, 71, 25), fill=255)
    image = image.resize((image.width * scale, image.height * scale))
    assert printed_multiplication(np.asarray(image), 3) == (kind == "cross")
