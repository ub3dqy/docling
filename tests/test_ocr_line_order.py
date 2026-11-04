# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Slanted paragraph rows must keep their own words and numeric fragments."""

import pytest

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


def test_complete_word_positions_order_fragments_with_overlapping_padded_boxes():
    start = _line("A", 0, 30, 21, 0)
    middle = _line("short", 20, 95, 20, 0)
    end = _line("caption", 90, 165, 19, 0)
    words = [
        _line("A", 3, 24, 21, 0),
        _line("short", 24, 86, 20, 0),
        _line("caption", 97, 160, 19, 0),
    ]
    result = RapidOcrModel._order_and_join(
        [(end, [words[2]]), (middle, [words[1]]), (start, [words[0]])]
    )
    assert [line[1] for line in result.lines] == ["A short caption"]
    assert [word[1] for word in result.words] == ["A", "short", "caption"]


@pytest.mark.parametrize("kind", ["partial", "outside", "duplicate", "other-row"])
def test_word_positions_cannot_join_partial_duplicate_or_other_row_text(kind):
    first = _line("first", 0, 40, 20, 0)
    second = _line("second", 20, 90, 19, 0)
    before = _line("first", 2, 28, 20, 0)
    after = _line("second", 30, 85, 19, 0)
    if kind == "partial":
        after = after[0], "sec", after[2]
    elif kind == "outside":
        after = _line("second", 41, 96, 19, 0)
    elif kind == "duplicate":
        after = _line("second", 23, 60, 19, 0)
    else:
        second = _line("second", 20, 90, 34, 0)
        after = _line("second", 30, 85, 34, 0)
    result = RapidOcrModel._order_and_join([(second, [after]), (first, [before])])
    assert len(result.lines) == 2


@pytest.mark.parametrize("weak_word", [False, True])
def test_weak_fragment_positions_cannot_expand_a_caption_crop(weak_word):
    heading = _line("Heading", 0, 80, 20, 0)
    heading_word = _line("Heading", 3, 76, 20, 0)
    fragment = _line("2", 70, 90, 21, 0)
    word = _line("2", 80, 86, 21, 0)
    if weak_word:
        word = word[0], word[1], 0.72
    else:
        fragment = fragment[0], fragment[1], 0.72
    result = RapidOcrModel._order_and_join(
        [(fragment, [word]), (heading, [heading_word])]
    )
    assert [line[1] for line in result.lines] == ["Heading", "2"]
    assert result.lines[0][0] == heading[0]


@pytest.mark.parametrize("slope", [0, -0.06, 0.06])
@pytest.mark.parametrize("reverse", [False, True])
def test_portrait_detector_boxes_cannot_join_unrelated_rotated_rows(slope, reverse):
    # Detectors keep image-corner order for these boxes; the first edge is
    # horizontal although the text runs down the tall box. The former join
    # treated that entire span as text height and combined neighbouring rows.
    observed = []
    for index, text in enumerate(["Сумма 73.28", "НДС 12.21", "Оплата 73.28"]):
        left, top, width, height = 20 + index * 16, 30, 11, 132
        quad = [
            [left, top],
            [left + width, top + width * slope],
            [left + width - height * slope, top + height + width * slope],
            [left - height * slope, top + height],
        ]
        observed.append((quad, text, 0.96))
    selected = observed[::-1] if reverse else observed
    result = RapidOcrModel._order_and_join([(line, [line]) for line in selected])
    assert sorted(result.lines, key=lambda line: line[1]) == sorted(
        observed, key=lambda line: line[1]
    )
    assert sorted(result.words, key=lambda line: line[1]) == sorted(
        observed, key=lambda line: line[1]
    )
