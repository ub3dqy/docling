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


def test_false_minus_needs_two_source_equals_readings_and_both_printed_marks():
    readings = _readings("=73.28")
    for decimal, equals in [(False, True), (True, False), (False, False)]:
        assert (
            choose_fiscal_number(
                "-73.28", readings, printed_separator=decimal, printed_equals=equals
            )
            is None
        )
    selected = choose_fiscal_number(
        "-73,28", readings, printed_separator=True, printed_equals=True
    )
    assert selected is not None and selected.text == "=73,28"
    assert selected.engine == "fiscal-marker-consensus"


@pytest.mark.parametrize("original", ["(73.28)", "=-73.28", "+73.28", "73.28"])
def test_source_equals_cannot_remove_a_parenthesized_or_already_explicit_sign(original):
    assert (
        choose_fiscal_number(
            original, _readings("=73.28"), printed_separator=True, printed_equals=True
        )
        is None
    )


@pytest.mark.parametrize(
    "failure", ["single-family", "different-digits", "negative-veto"]
)
def test_printed_bars_alone_cannot_rewrite_a_fiscal_sign(failure):
    readings = _readings("=73.28")
    if failure == "single-family":
        readings[1] = RegionReading("=73.28", 0.99, readings[0].engine, "wide")
    elif failure == "different-digits":
        readings = _readings("=73.29")
    else:
        readings.append(RegionReading("-73.28", 0.95, "ch", "source"))
    assert (
        choose_fiscal_number(
            "-73.28", readings, printed_separator=True, printed_equals=True
        )
        is None
    )


def test_printed_equals_can_survive_one_reader_omitting_the_marker():
    readings = [
        RegionReading("=73.28", 0.99, "cyrillic-v6-medium", "source"),
        RegionReading("73.28", 0.98, "ch", "wide"),
    ]
    selected = choose_fiscal_number(
        "-73.28", readings, printed_separator=True, printed_equals=True
    )
    assert selected is not None and selected.text == "=73.28"
    readings[0] = RegionReading("73.28", 0.99, "cyrillic-v6-medium", "source")
    assert (
        choose_fiscal_number(
            "-73.28", readings, printed_separator=True, printed_equals=True
        )
        is None
    )


def test_a_bleeding_source_marker_still_needs_two_strong_numeric_families():
    readings = [
        RegionReading("=73.28", 0.93, "cyrillic-v6-medium", "source"),
        RegionReading("73.28", 0.99, "cyrillic-v6-medium", "clean"),
        RegionReading("73.28", 0.98, "iso:ru", "wide"),
    ]
    selected = choose_fiscal_number(
        "-73.28", readings, printed_separator=True, printed_equals=True
    )
    assert selected is not None and selected.text == "=73.28"
    readings[-1] = RegionReading("73.28", 0.94, "iso:ru", "wide")
    assert (
        choose_fiscal_number(
            "-73.28", readings, printed_separator=True, printed_equals=True
        )
        is None
    )
