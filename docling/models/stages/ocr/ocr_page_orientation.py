# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Choose a page direction without turning separate Cyrillic rows upside down."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable
from math import dist
from typing import TYPE_CHECKING

from docling.models.stages.ocr.rapid_ocr_model import (
    _ORIENTATION_CONFIDENCE_SCORE,
    _ORIENTATION_MIN_LINES,
    _confident_chars,
    _OcrPrediction,
    _reads_well,
)
from docling.utils.ocr_reading_choice import number_signature
from docling.utils.ocr_word_overlap import OcrWord

if TYPE_CHECKING:
    from PIL import Image

    from docling.models.stages.ocr.rapid_ocr_model import RapidOcrModel


def restore_vertical_crop_axes(prediction: _OcrPrediction) -> _OcrPrediction:
    """Retain RapidOCR's actual 90CCW portrait-crop reading direction.

    RapidOCR's get_rotate_crop_image rotates crops with height/width >=1.5,
    but its line boxes and CalRecBoxes word boxes retain canonical corners.
    Rotate their corner order together, without moving any source point.
    Decide from the parent line: a narrow digit in a horizontal row was not
    independently rotated. A 180-degree classifier has a separate, unknown
    direction; callers must disable it and fallback predictions stay intact.
    """
    if prediction.line_classifier_fallback:
        return prediction
    lines, line_words = [], []
    for line, words in zip(prediction.lines, prediction.line_words):
        box = line[0]
        # Match the installed cropper's integer dimensions at the boundary.
        width = int(max(dist(box[0], box[1]), dist(box[2], box[3])))
        height = int(max(dist(box[0], box[3]), dist(box[1], box[2])))
        if width > 0 and height >= width * 1.5:
            lines.append((box[1:] + box[:1], line[1], line[2]))
            line_words.append([(w[0][1:] + w[0][:1], w[1], w[2]) for w in words])
        else:
            lines.append(line)
            line_words.append(words)
    return _OcrPrediction(lines, line_words)


def _known_text(
    lines: list[OcrWord], known: Callable[[str], bool]
) -> tuple[float, int]:
    recognized = 0.0
    letters = 0
    for _, text, confidence in lines:
        if confidence < _ORIENTATION_CONFIDENCE_SCORE:
            continue
        # Latin text and numeric identifiers are not Russian dictionary errors.
        for word in re.findall(r"[\u0400-\u04ff]{3,}", text):
            letters += len(word)
            if known(word.casefold()):
                recognized += len(word) * confidence
    return recognized, letters


def _vertical_labels(model: RapidOcrModel, lines: list[OcrWord]) -> bool:
    labels = [
        line
        for line in lines
        if line[2] >= 0.65 and sum(c.isalpha() for c in line[1]) >= 3
    ]
    return bool(
        len(labels) >= 3
        and sum(
            model._line_bbox(line).height > 1.8 * model._line_bbox(line).width
            for line in labels
        )
        >= len(labels) * 0.7
    )


def _retains_source_numbers(source: _OcrPrediction, candidate: _OcrPrediction) -> bool:
    money = re.compile(r"[=*: ]*[-\u2212]?\d+[.,]\d{2}")
    for line, words in zip(source.lines, source.line_words):
        trusted = (
            [line]
            if line[2] >= 0.95 or money.fullmatch(line[1].strip())
            else [
                word
                for word in words
                if word[2] >= 0.95 or money.fullmatch(word[1].strip())
            ]
        )
        values = Counter(
            value for item in trusted for value in number_signature(item[1])
        )
        if values and not any(
            observed[0] == line[0] and values <= Counter(number_signature(observed[1]))
            for observed in candidate.lines
        ):
            return False
    return True


def _credible_upright_text(
    model: RapidOcrModel, lines: list[OcrWord], known: Callable[[str], bool]
) -> bool:
    """Unknown names do not invalidate several well-read horizontal Russian rows."""
    if not _reads_well(lines):
        return False
    labels = [
        line
        for line in lines
        if line[2] >= 0.65 and sum(char.isalpha() for char in line[1]) >= 3
    ]
    horizontal = [
        line
        for line in labels
        if model._line_bbox(line).width > 1.8 * model._line_bbox(line).height
    ]
    if len(horizontal) < _ORIENTATION_MIN_LINES or len(horizontal) < len(labels) * 0.7:
        return False
    evidence = [
        sum(
            len(word) * line[2]
            for word in re.findall(r"[\u0400-\u04ff]{4,}", line[1])
            if known(word.casefold())
        )
        for line in horizontal
        if line[2] >= _ORIENTATION_CONFIDENCE_SCORE
    ]
    return (
        sum(score > 0 for score in evidence) >= _ORIENTATION_MIN_LINES
        and sum(evidence) >= 20
    )


def read_page_upright(
    model: RapidOcrModel, image: Image.Image, prediction: _OcrPrediction
) -> tuple[_OcrPrediction, int, tuple[int, int]]:
    orientation, size = 0, image.size
    known = model.orientation_word_is_known
    vertical = known is not None and _vertical_labels(model, prediction.lines)
    source_quality, source_letters = (
        _known_text(prediction.lines, known) if known is not None else (0.0, 0)
    )
    weak_language = source_letters >= 20 and source_quality < source_letters * 0.6
    upright = known is not None and _credible_upright_text(
        model, prediction.lines, known
    )
    if (
        not upright
        and (known is not None or len(prediction.lines) >= _ORIENTATION_MIN_LINES)
        and (not _reads_well(prediction.lines) or vertical or weak_language)
    ):
        needed = 2 * _confident_chars(prediction.lines)
        # Upright Cyrillic crops can resemble words even upside down. Check
        # the opposite page before the quarter turns when its rows are level.
        turns = (180, 90, 270) if known is not None and not vertical else (90, 180, 270)
        for turn in turns:
            turned = image.rotate(-turn, expand=True)
            candidate = model._read(turned)
            if candidate is None or not _reads_well(candidate.lines):
                continue
            if known is None:
                accepted = _confident_chars(candidate.lines) >= needed
            else:
                quality, letters = _known_text(candidate.lines, known)
                accepted = not _vertical_labels(model, candidate.lines) and (
                    (
                        vertical
                        and _confident_chars(candidate.lines)
                        >= _confident_chars(prediction.lines) * 0.9
                        and quality >= source_quality * 0.9
                    )
                    or (
                        quality >= max(20, source_quality * 2)
                        and quality >= letters * 0.6
                    )
                )
            if accepted:
                prediction, orientation, size = candidate, turn, turned.size
                break
        if (
            known is not None
            and orientation == 0
            and not _reads_well(prediction.lines)
            and model.options.use_cls is False
            and model.options.use_orientation_fallback
            and (source_letters == 0 or source_quality < source_letters * 0.6)
        ):
            # If no full-page direction has enough reliable source text,
            # retain the legacy per-line path instead of forcing a weak
            # rotated reading. Keep this request-local choice consistent
            # through subsequent detector scales.
            fallback = model._read_configured(image, line_classification=True)
            if (
                fallback is not None
                and fallback.lines
                and _retains_source_numbers(prediction, fallback)
            ):
                prediction = fallback
    kept = [
        (line, words)
        for line, words in zip(prediction.lines, prediction.line_words)
        if line[2] >= model.options.text_score
    ]
    return (
        _OcrPrediction(
            [line for line, _ in kept],
            [words for _, words in kept],
            line_classifier_fallback=prediction.line_classifier_fallback,
        ),
        orientation,
        size,
    )
