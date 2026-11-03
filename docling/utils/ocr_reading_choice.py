# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Compare readings without treating an uncertain first number as ground truth."""

import re
from collections.abc import Sequence

from docling.utils.ocr_word_overlap import OcrWord


def _bounds(quad: Sequence[Sequence[float]]) -> tuple[float, float, float, float]:
    return (
        min(point[0] for point in quad),
        min(point[1] for point in quad),
        max(point[0] for point in quad),
        max(point[1] for point in quad),
    )


def _source_bounds(
    line: OcrWord, words: Sequence[OcrWord]
) -> tuple[float, float, float, float]:
    left, top, right, bottom = _bounds(line[0])
    if not words or "".join("".join(word[1].split()) for word in words) != "".join(
        line[1].split()
    ):
        return left, top, right, bottom
    boxes = [_bounds(word[0]) for word in words]
    for x0, y0, x1, y1 in boxes:
        area = (x1 - x0) * (y1 - y0)
        overlap = max(0, min(right, x1) - max(left, x0)) * max(
            0, min(bottom, y1) - max(top, y0)
        )
        if area <= 0 or overlap < area * 0.8:
            return left, top, right, bottom
    # A padded detector frame is not text. Complete, valid word boxes identify
    # the observed glyphs more closely without dropping a left-hand caption.
    return (
        max(left, min(box[0] for box in boxes)),
        max(top, min(box[1] for box in boxes)),
        min(right, max(box[2] for box in boxes)),
        min(bottom, max(box[3] for box in boxes)),
    )


def number_signature(text: str) -> list[str]:
    values = []
    for match in re.finditer(r"\(?[-\u2212]?\d[\d ]*(?:[.,]\d+)?\)?", text):
        value = match.group().replace(" ", "").replace(",", ".").replace("\u2212", "-")
        if value.startswith("(") and value.endswith(")"):
            value = "-" + value[1:-1]
        values.append(value)
    return values


def protected_numbers(line: OcrWord, words: Sequence[OcrWord]) -> list[str]:
    """Keep complete numeric expressions and confident, spatially separate values.

    A weak mixed-script row is not a reliable source of digits. Its individually
    confident numeric word boxes still receive the same protection as clear rows.
    """
    text = line[1].strip()
    value = r"\(?[-\u2212]?\d[\d ]*(?:[.,]\d+)?\)?%?"
    numeric_expression = (
        re.fullmatch(r"[=*:\s]*" + value + r"(?:\s*[*=:]\s*" + value + r")*", text)
        is not None
    )
    if line[2] >= 0.85 or numeric_expression:
        return number_signature(text)
    return [
        value
        for word in words
        if word[2] >= 0.95
        and re.fullmatch(
            r"[=*:()+\-\u2212\s]*\d[\d\s.,=*:()%+\-\u2212]*", word[1].strip()
        )
        is not None
        for value in number_signature(word[1])
    ]


def reading_may_replace(
    previous: Sequence[tuple[OcrWord, Sequence[OcrWord]]],
    candidate: OcrWord,
    *,
    preserve_numbers: bool = True,
) -> bool:
    """Compare complete readings, with digit guards for supplementary detection.

    Scale selection retains its established quality-based numeric comparison.
    A later detector must not overwrite its chosen clear numeric observations.
    """
    if not previous or candidate[2] < 0.65:
        return False
    if candidate[2] < 0.8:
        if any(line[2] >= 0.65 for line, _ in previous) or not any(
            char.isalpha() for line, _ in previous for char in line[1]
        ):
            # The relaxed threshold rescues weak text, not unconfirmed changes
            # to numeric-only observations, such as dates or monetary values.
            return False
    left = min(point[0] for point in candidate[0])
    right = max(point[0] for point in candidate[0])
    for line, words in previous:
        frame_left, top, frame_right, bottom = _bounds(line[0])
        frame_width = frame_right - frame_left
        margin = max(frame_width * 0.15, min(frame_width, bottom - top) * 0.6)
        source_left, _, source_right, _ = _source_bounds(line, words)
        width = source_right - source_left
        if source_right > source_left and (
            width - (min(right, source_right) - max(left, source_left)) > margin
        ):
            # Allow one glyph's border uncertainty in a short line. A right-hand
            # amount must not erase the label and price on the rest of the row.
            return False
    # The additional low-confidence scale candidates need the same digit guard.
    protected = (
        [value for line, words in previous for value in protected_numbers(line, words)]
        if preserve_numbers or candidate[2] < 0.8
        else []
    )
    observed = iter(number_signature(candidate[1]))
    if any(not any(value == expected for value in observed) for expected in protected):
        return False
    if protected:
        old_markers = [
            match.group(1)
            for line, _ in previous
            for match in re.finditer(r"([=*])\s*-?\d", line[1])
        ]
        new_markers = [
            match.group(1) for match in re.finditer(r"([=*])\s*-?\d", candidate[1])
        ]
        if old_markers and old_markers != new_markers:
            return False
    old_characters = sum(len("".join(line[1].split())) for line, _ in previous)
    old_quality = sum(len("".join(line[1].split())) * line[2] for line, _ in previous)
    characters = len("".join(candidate[1].split()))
    return characters * candidate[2] > old_quality * 1.05 or (
        candidate[2] >= max(line[2] for line, _ in previous) + 0.1
        and characters >= 0.85 * old_characters
    )
