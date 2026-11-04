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


def _ink_box(item: _Item) -> BoundingBox:
    line, words = item
    bounds = _box(line)
    if (
        line[2] < 0.8
        or not words
        or any(word[2] < 0.8 for word in words)
        or "".join(line[1].split())
        != "".join("".join(word[1].split()) for word in words)
    ):
        return bounds
    boxes = [_box(word) for word in words]
    # Use complete word positions only within their own detector envelope.
    # Padding around two neighbouring fragments can overlap despite distinct
    # printed words. Weak, partial or out-of-frame output is not evidence.
    if any(
        box.width <= 0
        or box.height <= 0
        or box.l < bounds.l
        or box.r > bounds.r
        or box.t < bounds.t
        or box.b > bounds.b
        for box in boxes
    ):
        return bounds
    return BoundingBox(
        l=min(box.l for box in boxes),
        t=min(box.t for box in boxes),
        r=max(box.r for box in boxes),
        b=max(box.b for box in boxes),
    )


def _same_row(first_item: _Item, second_item: _Item, band: _Band) -> bool:
    first, second = first_item[0], second_item[0]
    a, b = _box(first), _box(second)
    if a.height > a.width * 1.5 or b.height > b.width * 1.5:
        # A detector's portrait box can contain a rotated line even when its
        # first edge is horizontal. Its long vertical span is not line height.
        # Keep these observations separate until reading direction is known;
        # horizontal row grouping would join unrelated columns of the page.
        return False
    ink_a, ink_b = _ink_box(first_item), _ink_box(second_item)
    if min(ink_a.r, ink_b.r) - max(ink_a.l, ink_b.l) > 0.1 * min(
        ink_a.width, ink_b.width
    ):
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
            if all(_same_row(item, other, line_band) for other in row):
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
