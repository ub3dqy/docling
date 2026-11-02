# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Discard duplicate OCR characters whose pixels already belong to a word."""

from collections.abc import Sequence
from typing import TypeAlias

OcrWord: TypeAlias = tuple[list[list[float]], str, float]


def deduplicate_line_words(
    line: OcrWord, words: list[OcrWord], *, neighbors: Sequence[OcrWord] = ()
) -> tuple[OcrWord, list[OcrWord]]:
    if "".join(line[1].split()) != "".join("".join(word[1].split()) for word in words):
        return line, words
    kept = []
    for index, word in enumerate(words):
        points, text, confidence = word
        left, right = min(p[0] for p in points), max(p[0] for p in points)
        top, bottom = min(p[1] for p in points), max(p[1] for p in points)
        duplicate = False
        if len(text.strip()) == 1 and right > left and bottom > top:
            for other_index, (other, other_text, other_confidence) in enumerate(
                [*words, *neighbors]
            ):
                if (
                    other_index == index
                    or other is points
                    or len(other_text) < 2
                    or text not in other_text
                    or other_confidence < confidence
                ):
                    continue
                x0, x1 = min(p[0] for p in other), max(p[0] for p in other)
                y0, y1 = min(p[1] for p in other), max(p[1] for p in other)
                overlap = max(0, min(right, x1) - max(left, x0)) * max(
                    0, min(bottom, y1) - max(top, y0)
                )
                if overlap / ((right - left) * (bottom - top)) < 0.95:
                    continue
                if other_index >= len(words):
                    # Separate detector lines can have overlapping envelopes
                    # on a tilted page. Require the actual quadrilaterals to
                    # cover the same pixels before dropping a character.
                    import cv2
                    import numpy as np

                    shape = np.asarray(points, dtype=np.float32)
                    parent = np.asarray(other, dtype=np.float32)
                    if (
                        shape.shape != (4, 2)
                        or parent.shape != (4, 2)
                        or not np.isfinite(shape).all()
                        or not np.isfinite(parent).all()
                    ):
                        continue
                    area = cv2.contourArea(shape)
                    if (
                        area <= 0
                        or cv2.intersectConvexConvex(shape, parent)[0] < area * 0.95
                    ):
                        continue
                if overlap > 0:
                    duplicate = True
                    break
        if not duplicate:
            kept.append(word)
    if len(kept) == len(words):
        return line, words
    return (line[0], " ".join(word[1] for word in kept), line[2]), kept
