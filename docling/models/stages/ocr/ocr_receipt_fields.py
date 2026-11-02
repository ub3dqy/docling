# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Re-read complete fiscal fields after caption/amount geometry is joined."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from docling.datamodel.base_models import Cluster, Page
from docling.models.stages.ocr.ocr_region_refiner import (
    RegionReading,
    normalize_mixed_script,
    number_key,
)
from docling.utils.ocr_digit_templates import decimal_expression_equals
from docling.utils.ocr_receipt_rows import is_receipt_field_caption

if TYPE_CHECKING:
    from docling.models.stages.ocr.ocr_region_refiner import OcrRegionRefiner

_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")


def preserves_known_caption(
    original: str, candidate: str, known: Callable[[str], bool]
) -> bool:
    """Retain readable word letters while allowing missing glyphs to return."""
    words = re.findall(r"[^\W\d_]+", normalize_mixed_script(original))
    # Short fragments and abbreviations still need wider context. Preserve
    # the letters of complete known words, including words beside damaged text.
    protected = [word.casefold() for word in words if len(word) >= 4 and known(word)]
    remaining = iter(
        word.casefold()
        for word in re.findall(r"[^\W\d_]+", normalize_mixed_script(candidate))
    )

    def retains(source: str, observed: str) -> bool:
        characters = iter(observed)
        return all(any(char == expected for char in characters) for expected in source)

    return all(
        any(retains(word, observed) for observed in remaining) for word in protected
    )


def _operator_positions(text: str, marker: str) -> Counter[int]:
    ends = [match.end() for match in _NUMBER.finditer(text)]
    return Counter(
        sum(end <= index for end in ends)
        for index, char in enumerate(text)
        if char == marker
    )


def retain_field_numbers(original: str, candidate: str) -> str | None:
    """Keep the established numeric tokens, scale and operators in a field."""
    before, after = _NUMBER.findall(original), list(_NUMBER.finditer(candidate))
    if [v.replace(",", ".") for v in before] != [
        v.group().replace(",", ".") for v in after
    ]:
        return None
    strict = {
        char
        for char in original + candidate
        if unicodedata.category(char) in {"Sm", "Sc", "Ps", "Pe"}
        or char in "-/%№:\u2013\u2014"
    } - {"="}
    if any(
        _operator_positions(original, mark) != _operator_positions(candidate, mark)
        for mark in strict
    ):
        return None
    for marker in ("=", "*"):
        if not _operator_positions(original, marker) <= _operator_positions(
            candidate, marker
        ):
            return None
        if candidate.count(marker) > max(1, original.count(marker)):
            return None
    if candidate.count("*") > original.count("*") and not candidate.lstrip().startswith(
        "*"
    ):
        return None
    if candidate.count("=") > original.count("=") and len(
        re.findall(r"=\s*\d", candidate)
    ) != candidate.count("="):
        return None
    # Preserve the existing decimal mark in text exports, as well as value.
    for text, match in reversed(list(zip(before, after))):
        candidate = candidate[: match.start()] + text + candidate[match.end() :]
    return re.sub(r"(?<=[\w%])=(?=\d)", " =", candidate)


def triangle_is_line_noise(gray: np.ndarray) -> bool:
    """A wide stroke and scattered debris are not a three-dot triangle glyph."""
    import cv2

    _, ink = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    _, _, stats, _ = cv2.connectedComponentsWithStats(ink)
    body = sorted(
        (s for s in stats[1:] if s[4] >= 2), key=lambda s: int(s[4]), reverse=True
    )[:3]
    if len(body) != 3:
        return False
    width = max(int(s[0] + s[2]) for s in body) - min(int(s[0]) for s in body)
    height = max(int(s[1] + s[3]) for s in body) - min(int(s[1]) for s in body)
    largest = body[0]
    return bool(
        width > 2.5 * height
        and largest[2] > 3 * largest[3]
        and largest[4] > 2.5 * body[-1][4]
    )


def refine_receipt_fields(
    refiner: OcrRegionRefiner, page: Page, image: Image.Image
) -> None:
    assert page.parsed_page is not None
    replacements: dict[int, RegionReading] = {}
    removed: set[int] = set()
    for cell in page.parsed_page.textline_cells:
        if not cell.from_ocr:
            continue
        triangle = cell.text.strip() in {"∴", "∵"} and cell.confidence < 0.85
        numeric = re.fullmatch(r"\d+[.,]\d{2}", cell.text.strip()) is not None
        if not triangle and not numeric and not is_receipt_field_caption(cell.text):
            continue
        gray = refiner._crop(image, cell.rect.to_bounding_box(), page, inset=-1)
        if gray is None:
            continue
        if triangle:
            if triangle_is_line_noise(gray):
                removed.add(cell.index)
            continue
        readings = [
            reading
            for reading in refiner._cyrillic_readings(gray)
            if preserves_known_caption(
                cell.text, reading.text, refiner._dictionary.word_is_known
            )
        ]
        if numeric:
            candidates = [
                reading
                for reading in readings
                if reading.confidence >= 0.95
                and reading.text.strip().startswith("=")
                and number_key(reading.text) == number_key(cell.text)
            ]
            digits = len(re.split(r"[.,]", cell.text.strip())[0])
            if not candidates or not decimal_expression_equals(gray, digits):
                continue
            selected = max(candidates, key=lambda reading: reading.confidence)
        else:
            selected = refiner._choose_text(cell.text, cell.confidence, readings)
        text = retain_field_numbers(cell.text, selected.text)
        if text is None or text == cell.text:
            continue
        replacements[cell.index] = RegionReading(
            text, selected.confidence, selected.engine, selected.variant
        )
        cell.text, cell.confidence = text, selected.confidence
    page.parsed_page.textline_cells = [
        cell for cell in page.parsed_page.textline_cells if cell.index not in removed
    ]

    def update(cluster: Cluster) -> None:
        cluster.cells = [
            cell
            for cell in cluster.cells
            if not cell.from_ocr or cell.index not in removed
        ]
        for cell in cluster.cells:
            if cell.from_ocr and cell.index in replacements:
                reading = replacements[cell.index]
                cell.text, cell.confidence = reading.text, reading.confidence
        for child in cluster.children:
            update(child)

    if page.predictions.layout is not None:
        for cluster in page.predictions.layout.clusters:
            update(cluster)
