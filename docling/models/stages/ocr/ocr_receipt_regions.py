# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Verify individual receipt prices inside mixed text lines."""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
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


_MONEY = re.compile(r"(?:[=+\-\u2212]\s*|(?<![\d.,]))\(?\d+[.,]\d{2}\)?(?![\d.,])")


@dataclass(frozen=True)
class VerifiedReceiptAmount:
    ordinal: int
    count: int
    key: str
    equals: bool


@dataclass
class ReceiptPriceRefinements:
    """Keep pixel-verified amounts in their positions during later row reading."""

    changed_lines: set[int] = field(default_factory=set)
    amounts: dict[int, dict[int, VerifiedReceiptAmount]] = field(default_factory=dict)

    def record(self, index: int, text: str, span: tuple[int, int]) -> None:
        matches = list(_MONEY.finditer(text))
        for ordinal, match in enumerate(matches):
            key = number_key(match.group())
            if span[0] <= match.start() and match.end() <= span[1] and key is not None:
                self.amounts.setdefault(index, {})[ordinal] = VerifiedReceiptAmount(
                    ordinal, len(matches), key, match.group().lstrip().startswith("=")
                )

    def preserves(self, index: int, text: str) -> bool:
        verified = self.amounts.get(index, {})
        if not verified:
            return True
        matches = list(_MONEY.finditer(text))
        return all(
            len(matches) == item.count
            and number_key(matches[item.ordinal].group()) == item.key
            and (
                not item.equals
                or matches[item.ordinal].group().lstrip().startswith("=")
            )
            for item in verified.values()
        )

    def readings_for(
        self, index: int, readings: list[RegionReading]
    ) -> list[RegionReading]:
        """Carry a verified local sign into a row reading with identical digits.

        A row can still improve letters when it misreads the same source mark.
        Require the same amount count, positions and decimal digits. Never
        replace a row's different digits with a verified neighbouring amount.
        Keep the original model family and label this combined observation.
        """
        verified = self.amounts.get(index, {})
        if not verified:
            return readings
        retained = []
        for reading in readings:
            text = reading.text
            matches = list(_MONEY.finditer(text))
            for item in sorted(
                verified.values(), key=lambda v: v.ordinal, reverse=True
            ):
                if len(matches) != item.count:
                    break
                match = matches[item.ordinal]
                key = number_key(match.group())
                if key is None or key.lstrip("-") != item.key.lstrip("-"):
                    continue
                if key == item.key and (
                    not item.equals or match.group().lstrip().startswith("=")
                ):
                    continue
                digits = re.search(r"\d+[.,]\d{2}", match.group())
                assert digits is not None
                marker = "=" if item.equals else "-" if item.key.startswith("-") else ""
                text = (
                    text[: match.start()]
                    + marker
                    + digits.group()
                    + text[match.end() :]
                )
            if self.preserves(index, text):
                retained.append(
                    reading
                    if text == reading.text
                    else replace(
                        reading, text=text, variant=reading.variant + "+verified-marker"
                    )
                )
        return retained


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
) -> ReceiptPriceRefinements:
    """Verify individual printed amounts inside otherwise mixed receipt lines."""
    from rapidocr.ch_ppocr_rec.typings import TextRecOutput

    from docling.models.stages.ocr.ocr_receipt_expressions import (
        restore_receipt_separator,
    )

    assert page.parsed_page is not None
    verified = [
        cell
        for cell in page.parsed_page.textline_cells
        if verified_lines is not None and cell.index in verified_lines
    ]
    changed = ReceiptPriceRefinements()
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
        replacement = restore_receipt_separator(replacement, readings, original_gray)
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
                match = pattern.search(line.text)
                assert match is not None
                line.text = pattern.sub(lambda _: replacement, line.text)
                changed.changed_lines.add(line.index)
                changed.record(
                    line.index,
                    line.text,
                    (match.start(), match.start() + len(replacement)),
                )
        word.text = replacement
    return changed
