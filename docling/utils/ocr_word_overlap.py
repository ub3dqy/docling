# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Discard duplicate OCR characters whose pixels already belong to a word."""

from typing import TypeAlias

OcrWord: TypeAlias = tuple[list[list[float]], str, float]


def deduplicate_line_words(
    line: OcrWord, words: list[OcrWord]
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
            for other_index, (other, other_text, other_confidence) in enumerate(words):
                if (
                    other_index == index
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
                if overlap / ((right - left) * (bottom - top)) >= 0.95:
                    duplicate = True
                    break
        if not duplicate:
            kept.append(word)
    if len(kept) == len(words):
        return line, words
    return (line[0], " ".join(word[1] for word in kept), line[2]), kept
