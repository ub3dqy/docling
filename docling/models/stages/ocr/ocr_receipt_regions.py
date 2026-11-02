# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Verify individual receipt prices inside mixed text lines."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageOps

from docling.datamodel.base_models import Page
from docling.models.stages.ocr.ocr_region_refiner import (
    RegionReading,
    choose_number,
    number_key,
)
from docling.utils.ocr_digit_templates import (
    decimal_expression_equals,
    decimal_suffix_region,
    repeated_digit_support,
)

if TYPE_CHECKING:
    from tesserocr import PyTessBaseAPI

    from docling.models.stages.ocr.ocr_region_refiner import OcrRegionRefiner


def _amount_pixels(
    refiner: OcrRegionRefiner, api: PyTessBaseAPI, gray: np.ndarray, amount: str
) -> np.ndarray | None:
    """Locate the amount in an expression from recognized symbol coordinates."""
    region = decimal_suffix_region(gray, len(amount.split(".")[0]))
    if region is not None:
        return region
    api.SetVariable("tessedit_char_whitelist", "0123456789.,=+-*()")
    api.SetPageSegMode(refiner._tesserocr.PSM.SINGLE_LINE)
    sample = ImageOps.expand(Image.fromarray(gray), 12, fill=255)
    api.SetImage(sample)
    if not api.GetUTF8Text().strip():
        return None
    symbols = []
    for item in refiner._tesserocr.iterate_level(
        api.GetIterator(), refiner._tesserocr.RIL.SYMBOL
    ):
        text = item.GetUTF8Text(refiner._tesserocr.RIL.SYMBOL)
        box = item.BoundingBox(refiner._tesserocr.RIL.SYMBOL)
        if len(text) != 1 or box is None:
            return None
        symbols.append((text, box))
    joined = "".join(text for text, _ in symbols)
    length = len(amount.split(".")[0])
    matches = list(
        re.finditer(rf"(?<![\d.,])\d{{{length}}}[.,]\d{{2}}(?![\d.,])", joined)
    )
    if len(matches) != 1:
        return None
    match = matches[0]
    boxes = [box for _, box in symbols[match.start() : match.end()]]
    left, top = min(b[0] for b in boxes) - 14, min(b[1] for b in boxes) - 14
    right, bottom = max(b[2] for b in boxes) - 10, max(b[3] for b in boxes) - 10
    return gray[
        max(0, top) : min(gray.shape[0], bottom),
        max(0, left) : min(gray.shape[1], right),
    ]


def _repeated_digit_amount(
    original: str, readings: list[RegionReading], gray: np.ndarray
) -> str:
    """Require an observed full amount and two matching source digit exemplars."""
    key = number_key(original)
    if key is None:
        return original
    digits = "".join(char for char in key if char.isdigit())
    supported = set()
    for reading in readings:
        candidate = number_key(reading.text)
        if (
            reading.confidence >= 0.85
            and candidate is not None
            and re.fullmatch(r"\d+\.\d{2}", candidate)
            and len(candidate) == len(key)
            and repeated_digit_support(gray, digits, candidate.replace(".", ""))
        ):
            supported.add(candidate)
    return supported.pop() if len(supported) == 1 else original


def refine_receipt_prices(
    refiner: OcrRegionRefiner,
    page: Page,
    image: Image.Image,
    api: PyTessBaseAPI,
    *,
    verified_lines: set[int] | None = None,
) -> set[int]:
    """Verify individual printed amounts inside otherwise mixed receipt lines."""
    from rapidocr.ch_ppocr_rec.typings import TextRecOutput

    assert page.parsed_page is not None
    verified = [
        cell
        for cell in page.parsed_page.textline_cells
        if verified_lines is not None and cell.index in verified_lines
    ]
    changed: set[int] = set()
    for word in page.parsed_page.word_cells:
        if any(
            word.text == cell.text
            and word.rect.to_bounding_box().intersection_over_self(
                cell.rect.to_bounding_box()
            )
            >= 0.8
            for cell in verified
        ):
            continue
        amounts = list(re.finditer(r"(?<![\d.,])\d{1,7}[.,]\d{2}(?![\d.,])", word.text))
        if len(amounts) != 1:
            continue
        amount = amounts[0]
        key = number_key(amount.group())
        if (
            not word.from_ocr
            or key is None
            or re.fullmatch(r"\d{1,7}\.\d{2}", key) is None
        ):
            continue
        box = word.rect.to_bounding_box()
        gray = refiner._crop(image, box, page, inset=-1)
        if gray is None:
            continue
        original_gray = gray
        if number_key(word.text) is None:
            if re.fullmatch(r"[\d.,=+*() -]+", word.text.strip()) is None:
                continue
            gray = _amount_pixels(refiner, api, gray, key)
            if gray is None or gray.size == 0:
                continue
        readings = refiner._rapid_readings(gray, numeric=True)
        sample = np.asarray(
            ImageOps.expand(Image.fromarray(gray).convert("RGB"), 6, fill="white")
        )
        for lang, reader in refiner._readers.items():
            result = reader(sample, use_det=False, use_cls=False, use_rec=True)
            if isinstance(result, TextRecOutput) and result.txts and result.scores:
                readings.append(
                    RegionReading(
                        result.txts[0], float(result.scores[0]), lang, "source"
                    )
                )
        selected = number_key(
            _repeated_digit_amount(
                choose_number(amount.group(), readings), readings, gray
            )
        )
        replacement = refiner._receipt_equals(api, original_gray, word.text)
        if re.search(
            r"\d-$", word.text[: amount.start()]
        ) and decimal_expression_equals(original_gray, len(key.split(".")[0])):
            replacement = (
                replacement[: amount.start() - 1] + "=" + replacement[amount.start() :]
            )
        if selected is None:
            continue
        if selected != key:
            digits = iter(char for char in selected if char.isdigit())
            replacement = (
                replacement[: amount.start()]
                + "".join(
                    next(digits) if char.isdigit() else char for char in amount.group()
                )
                + replacement[amount.end() :]
            )
        if replacement == word.text:
            continue
        pattern = re.compile(r"(?<![\d.,])" + re.escape(word.text) + r"(?![\d.,])")
        for line in page.parsed_page.textline_cells:
            if (
                line.from_ocr
                and box.intersection_over_self(line.rect.to_bounding_box()) >= 0.8
                and len(pattern.findall(line.text)) == 1
            ):
                line.text = pattern.sub(lambda _: replacement, line.text)
                changed.add(line.index)
        word.text = replacement
    return changed
