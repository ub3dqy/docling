# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Slanted paragraph rows must keep their own words and numeric fragments."""

from docling.models.stages.ocr.rapid_ocr_model import RapidOcrModel


def _line(text: str, left: float, right: float, baseline: float, slope: float):
    return (
        [
            [left, baseline + slope * left],
            [right, baseline + slope * right],
            [right, baseline + slope * right + 10],
            [left, baseline + slope * left + 10],
        ],
        text,
        0.95,
    )


def test_slanted_envelopes_do_not_move_a_date_fragment_to_the_next_row():
    first = _line("First", 0, 80, 20, 0.03)
    first_tail = _line("31 December 2024", 320, 900, 20, 0.03)
    second = _line("Second", 0, 80, 37, 0.03)
    second_tail = _line("12 January 2025", 320, 900, 37, 0.03)
    result = RapidOcrModel._order_and_join(
        [(line, []) for line in (second_tail, first_tail, first, second)]
    )
    assert [line[1] for line in result.lines] == [
        "First",
        "31 December 2024",
        "Second",
        "12 January 2025",
    ]


def test_joining_fragments_retains_the_printed_row_slope_for_later_crops():
    first = _line("Part", 0, 100, 20, 0.04)
    tail = _line("two", 102, 160, 20, 0.04)
    next_row = _line("Next row", 0, 160, 37, 0.04)
    result = RapidOcrModel._order_and_join(
        [(line, []) for line in (next_row, tail, first)]
    )
    assert [line[1] for line in result.lines] == ["Part two", "Next row"]
    joined = result.lines[0]
    assert joined[0][0][1] < joined[0][1][1]
    assert abs(RapidOcrModel._line_band(joined, 80)[1] - 10) < 1e-8
    assert [word[1] for word in result.words][:2] == ["Part", "two"]
