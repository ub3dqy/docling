# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Order OCR fragments using their sloped text bands rather than envelopes."""

from collections.abc import Callable

from docling_core.types.doc import BoundingBox

from docling.utils.ocr_word_overlap import OcrWord, deduplicate_line_words

_Band = Callable[[OcrWord, float], tuple[float, float]]
_Item = tuple[OcrWord, list[OcrWord]]


def _box(line: OcrWord) -> BoundingBox:
    return BoundingBox(
        l=min(point[0] for point in line[0]),
        t=min(point[1] for point in line[0]),
        r=max(point[0] for point in line[0]),
        b=max(point[1] for point in line[0]),
    )


def _same_row(first: OcrWord, second: OcrWord, band: _Band) -> bool:
    a, b = _box(first), _box(second)
    if min(a.r, b.r) - max(a.l, b.l) > 0.1 * min(a.width, b.width):
        return False
    x = (max(a.l, b.l) + min(a.r, b.r)) / 2
    ay, ah = band(first, x)
    by, bh = band(second, x)
    overlap = min(ay + ah / 2, by + bh / 2) - max(ay - ah / 2, by - bh / 2)
    return min(ah, bh) > 0 and overlap >= 0.5 * min(ah, bh)


def _merge(previous: OcrWord, line: OcrWord, band: _Band) -> OcrWord:
    before, after = _box(previous), _box(line)
    longer = previous if before.width >= after.width else line
    slope = band(longer, 1)[0] - band(longer, 0)[0]
    points = previous[0] + line[0]
    top = min(y - slope * x for x, y in points)
    bottom = max(y - slope * x for x, y in points)
    left, right = min(before.l, after.l), max(before.r, after.r)
    score = (len(previous[1]) * previous[2] + len(line[1]) * line[2]) / max(
        1, len(previous[1]) + len(line[1])
    )
    return (
        [
            [left, slope * left + top],
            [right, slope * right + top],
            [right, slope * right + bottom],
            [left, slope * left + bottom],
        ],
        f"{previous[1].strip()} {line[1].strip()}",
        score,
    )


def order_ocr_lines(selected: list[_Item], *, line_band: _Band) -> list[_Item]:
    rows: list[list[_Item]] = []
    for item in sorted(selected, key=lambda item: _box(item[0]).t):
        for row in rows:
            if all(_same_row(item[0], other[0], line_band) for other in row):
                row.append(item)
                break
        else:
            rows.append([item])
    ordered: list[_Item] = []
    for row in rows:
        row_start = len(ordered)
        for line, words in sorted(row, key=lambda item: _box(item[0]).l):
            if len(ordered) > row_start:
                previous, previous_words = ordered[-1]
                box, previous_box = _box(line), _box(previous)
                # Membership was established from every original quadrilateral.
                # Retain the slope after merging so later crops do not expand
                # into the printed row above or below a long tilted line.
                if box.l - previous_box.r <= 0.5 * min(
                    line_band(line, box.l)[1],
                    line_band(previous, previous_box.r)[1],
                ):
                    ordered[-1] = (
                        _merge(previous, line, line_band),
                        (previous_words or [previous]) + (words or [line]),
                    )
                    continue
            ordered.append((line, words))
    return [deduplicate_line_words(line, words) for line, words in ordered]
