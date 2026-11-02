# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Malformed fiscal numbers need visible separators and independent agreement."""

import pytest

from docling.models.stages.ocr.ocr_receipt_amounts import choose_fiscal_number
from docling.models.stages.ocr.ocr_region_refiner import RegionReading


def _readings(value):
    return [
        RegionReading(value, 0.99, "cyrillic-v6-medium", "source"),
        RegionReading(value, 0.97, "iso:ru", "clean"),
    ]


def test_damaged_digits_and_separator_are_recovered_only_from_independent_pixels():
    readings = _readings("87.90")
    selected = choose_fiscal_number("6626", readings, printed_separator=True)
    assert selected is not None and selected.text == "87.90"
    assert choose_fiscal_number("6626", readings, printed_separator=False) is None
    assert choose_fiscal_number("6626", readings[:1], printed_separator=True) is None
    repeated = [
        RegionReading("87.90", 0.99, "iso:ru", variant)
        for variant in ("source", "clean", "wide")
    ]
    assert choose_fiscal_number("6626", repeated, printed_separator=True) is None


def test_malformed_prefix_is_not_a_reason_to_copy_a_neighboring_field():
    selected = choose_fiscal_number(
        "22(480.00", _readings("2 480.00"), printed_separator=True
    )
    assert selected is not None and selected.text == "2480.00"
    conflict = _readings("2480.00") + [
        RegionReading("22480.00", 0.99, engine, "source")
        for engine in ("cyrillic-v6-tiny", "ch")
    ]
    assert choose_fiscal_number("22(480.00", conflict, printed_separator=True) is None


@pytest.mark.parametrize(
    "original", ["87.90", "-87.90", "(87.90)", "(8790", "5*87.90", "20% 8790"]
)
def test_existing_amounts_signs_and_expressions_remain_protected(original):
    assert (
        choose_fiscal_number(original, _readings("88.90"), printed_separator=True)
        is None
    )
