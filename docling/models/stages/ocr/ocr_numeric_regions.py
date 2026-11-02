# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Verify uncertain printed integers before ranking their surrounding text."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from PIL import Image

from docling.datamodel.base_models import Page
from docling.models.stages.ocr.ocr_region_refiner import (
    RegionReading,
    choose_number,
    number_key,
    recognizer_family,
)

if TYPE_CHECKING:
    from docling.models.stages.ocr.ocr_region_refiner import OcrRegionRefiner


def replace_word_text(page: Page, word_index: int, replacement: str) -> set[int]:
    """Update only complete tokens in lines containing the source word's box."""
    assert page.parsed_page is not None
    word = next(
        cell for cell in page.parsed_page.word_cells if cell.index == word_index
    )
    original = word.text
    bounds = word.rect.to_bounding_box()
    pattern = re.compile(r"(?<!\w)" + re.escape(original) + r"(?!\w)")
    changed = set()
    for line in page.parsed_page.textline_cells:
        if (
            line.from_ocr
            and bounds.intersection_over_self(line.rect.to_bounding_box()) >= 0.8
            and len(pattern.findall(line.text)) == 1
        ):
            line.text = pattern.sub(lambda _: replacement, line.text)
            changed.add(line.index)
    if changed:
        word.text = replacement
    return changed


def refine_numeric_words(
    refiner: OcrRegionRefiner, page: Page, image: Image.Image
) -> set[int]:
    """Read an uncertain isolated number from its own pixels, not its caption."""
    assert page.parsed_page is not None
    changed = set()
    for word in page.parsed_page.word_cells:
        if (
            not word.from_ocr
            or word.confidence >= 0.95
            or re.fullmatch(r"\d{3,8}", word.text) is None
        ):
            continue
        gray = refiner._crop(image, word.rect.to_bounding_box(), page, inset=-1.5)
        if gray is None:
            continue
        readings = [
            reading
            for reading in refiner._rapid_readings(gray, numeric=True)
            if reading.confidence >= 0.95
        ]
        selected = choose_number(word.text, readings)
        if selected != word.text and number_key(selected) is not None:
            changed |= replace_word_text(page, word.index, selected)
    return changed | refine_numbered_suffixes(refiner, page, image)


def choose_numbered_suffix(original: str, readings: list[RegionReading]) -> str:
    """Recover one clipped trailing digit only with an observed number sign.

    Keep the complete prefix of the original number. Ordinary integers, dates,
    amounts and changed leading digits do not qualify for this recovery path.
    """
    before = re.fullmatch(r"[\u2116\u0418IN]\s*(\d{2,8})", original.strip())
    if before is None:
        return original
    groups: dict[str, list[RegionReading]] = {}
    for reading in readings:
        match = re.fullmatch(r"\u2116\s*(\d{2,9})", reading.text.strip())
        if reading.confidence < 0.95 or match is None:
            continue
        digits = match[1]
        if len(digits) == len(before[1]) + 1 and digits.startswith(before[1]):
            groups.setdefault(digits, []).append(reading)
    supported = [
        group
        for group in groups.values()
        if len({recognizer_family(reading.engine) for reading in group}) >= 2
    ]
    if len(supported) != 1:
        return original
    return max(supported[0], key=lambda reading: reading.confidence).text.strip()


def refine_numbered_suffixes(
    refiner: OcrRegionRefiner, page: Page, image: Image.Image
) -> set[int]:
    """A tall merged caption can clip the number printed below its final words."""
    assert page.parsed_page is not None
    changed = set()
    for line in page.parsed_page.textline_cells:
        match = re.search(r"(?<!\w)([\u2116\u0418IN])\s*(\d{2,8})\s*$", line.text)
        if not line.from_ocr or line.confidence >= 0.95 or match is None:
            continue
        bounds = line.rect.to_bounding_box()
        words = [
            word
            for word in page.parsed_page.word_cells
            if word.from_ocr
            and word.rect.to_bounding_box().intersection_over_self(bounds) >= 0.8
        ]
        prefixes = [word for word in words if word.text == match[1]]
        numbers = [word for word in words if word.text == match[2]]
        if len(prefixes) != 1 or len(numbers) != 1:
            continue
        prefix, number = prefixes[0], numbers[0]
        a, b = prefix.rect.to_bounding_box(), number.rect.to_bounding_box()
        if not 0 <= b.l - a.r <= max(a.height, b.height):
            continue
        top, bottom = min(a.t, b.t), max(a.b, b.b)
        height = bottom - top
        crop = b.model_copy(
            update={
                "l": a.l,
                "r": b.r + height * 0.6,
                "t": top + height * 0.4,
                "b": bottom,
            }
        )
        gray = refiner._crop(image, crop, page, inset=-0.8)
        if gray is None:
            continue
        selected = choose_numbered_suffix(
            match.group().strip(),
            [
                *refiner._rapid_readings(gray, numeric=True),
                *refiner._cyrillic_readings(gray),
            ],
        )
        if selected != match.group().strip():
            line.text = line.text[: match.start()] + selected
            prefix.text = "\u2116"
            number.text = re.sub(r"\D", "", selected)
            changed.add(line.index)
    return changed
