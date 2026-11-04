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
    printed_decimal,
)
from docling.utils.ocr_digit_templates import (
    decimal_expression_equals,
    decimal_suffix_region,
    printed_multiplication,
    repeated_digit_support,
)
from docling.utils.ocr_text_geometry import extend_text_rect, rectified_text_crop

if TYPE_CHECKING:
    from docling_core.types.doc.page import TextCell
    from tesserocr import PyTessBaseAPI

    from docling.models.stages.ocr.ocr_region_refiner import OcrRegionRefiner


_MONEY = re.compile(
    r"(?:[=+\-\u2212]\s*|(?<!\d)(?<!\d[.,]))\(?\d+[.,]\d{2}\)?(?!\d|[.,]\d)"
)
_PRICE_EXPRESSION = re.compile(
    r"(?<!\d)(?<!\d[.,])(?P<price>\d+[.,]\d{2})\s*[*x\u0445\u00d7]\s*"
    r"(?P<quantity>\d+(?:[.,]\d+)?)(?!\d|[.,]\d)",
    re.IGNORECASE,
)
_EXPRESSION_UNIT = re.compile(
    r"[ \t]*(?P<unit>\u0448\u0442|\u043a\u0433|\u0433|\u043b|\u043c|\u0443\u043f|"
    r"\u043f\u0430\u0447|\u0440\u0443\u043b|\u043a\u043e\u043c\u043f\u043b|\u043c\u0435\u0441)\.?(?!\w)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class VerifiedReceiptAmount:
    ordinal: int
    count: int
    key: str
    equals: bool


@dataclass(frozen=True)
class VerifiedReceiptExpression:
    ordinal: int
    count: int
    price: str
    quantity: str
    unit: str | None


@dataclass
class ReceiptPriceRefinements:
    """Keep verified amounts and complete expressions during later row reading."""

    changed_lines: set[int] = field(default_factory=set)
    amounts: dict[int, dict[int, VerifiedReceiptAmount]] = field(default_factory=dict)
    expressions: dict[int, dict[int, VerifiedReceiptExpression]] = field(
        default_factory=dict
    )

    def record(self, index: int, text: str, span: tuple[int, int]) -> None:
        matches = list(_MONEY.finditer(text))
        for ordinal, match in enumerate(matches):
            key = number_key(match.group())
            if span[0] <= match.start() and match.end() <= span[1] and key is not None:
                self.amounts.setdefault(index, {})[ordinal] = VerifiedReceiptAmount(
                    ordinal, len(matches), key, match.group().lstrip().startswith("=")
                )
        expressions = list(_PRICE_EXPRESSION.finditer(text))
        for ordinal, match in enumerate(expressions):
            if span[0] <= match.start() and match.end() <= span[1]:
                unit = _EXPRESSION_UNIT.match(text, match.end())
                self.expressions.setdefault(index, {})[ordinal] = (
                    VerifiedReceiptExpression(
                        ordinal,
                        len(expressions),
                        match["price"].replace(",", "."),
                        match["quantity"].replace(",", "."),
                        unit["unit"].casefold()
                        if unit is not None and unit.end() <= span[1]
                        else None,
                    )
                )

    def preserves(self, index: int, text: str) -> bool:
        verified = self.amounts.get(index, {})
        verified_expressions = self.expressions.get(index, {})
        if not verified and not verified_expressions:
            return True
        matches = list(_MONEY.finditer(text))
        if not all(
            len(matches) == item.count
            and number_key(matches[item.ordinal].group()) == item.key
            and (
                not item.equals
                or matches[item.ordinal].group().lstrip().startswith("=")
            )
            for item in verified.values()
        ):
            return False
        expressions = list(_PRICE_EXPRESSION.finditer(text))
        for item in verified_expressions.values():
            if len(expressions) != item.count:
                return False
            match = expressions[item.ordinal]
            if (
                match["price"].replace(",", ".") != item.price
                or match["quantity"].replace(",", ".") != item.quantity
            ):
                return False
            if item.unit is not None:
                unit = _EXPRESSION_UNIT.match(text, match.end())
                if unit is None or unit["unit"].casefold() != item.unit:
                    return False
        return True

    def readings_for(
        self, index: int, readings: list[RegionReading]
    ) -> list[RegionReading]:
        """Carry a verified local sign into a row reading with identical digits.

        A row can still improve letters when it misreads the same source mark.
        Require the same amount count, positions and decimal digits. Never
        replace a row's different digits with a verified neighbouring amount.
        Keep the original model family and label this combined observation.
        Reject damage to a verified operator, quantity or observed unit.
        """
        verified = self.amounts.get(index, {})
        if not verified and not self.expressions.get(index):
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


def _source_expression(
    refiner: OcrRegionRefiner,
    page: Page,
    word: TextCell,
    gray_page: np.ndarray,
) -> RegionReading | None:
    """Read an uncertain, complete numeric expression from its own text band."""
    from docling.models.stages.ocr.ocr_receipt_expressions import (
        choose_unitless_expression,
    )

    raw = word.text.strip()
    prefix = re.match(r"\d+[.,]\d{2}", raw)
    if (
        page.size is None
        or prefix is None
        or re.fullmatch(r"[\d.,*x\u0445\u00d7 ]+", raw, re.IGNORECASE) is None
    ):
        return None
    box = word.rect.to_bounding_box()
    if box.height > box.width * 1.5:
        return None
    assert page.parsed_page is not None
    neighbours = [
        other.rect.to_bounding_box()
        for other in page.parsed_page.word_cells
        if other is not word
        and other.from_ocr
        and min(other.rect.to_bounding_box().b, box.b)
        - max(other.rect.to_bounding_box().t, box.t)
        >= min(box.height, other.rect.to_bounding_box().height) * 0.7
    ]
    left_gap = min(
        [box.height * 0.3]
        + [max(0, box.l - b.r - 1) for b in neighbours if b.r <= box.l]
    )
    right_gap = min(
        [box.height * 0.7]
        + [max(0, b.l - box.r - 1) for b in neighbours if b.l >= box.r]
    )
    integer_digits = len(prefix[0].partition(".")[0].partition(",")[0])
    existing_operator = re.search(r"[*x\u0445\u00d7]", raw, re.IGNORECASE) is not None
    for left, right in ((0.0, 0.0), (left_gap, right_gap)):
        gray = rectified_text_crop(
            gray_page, extend_text_rect(word.rect, left=left, right=right), page.size
        )
        if gray is None:
            continue
        observed_operator = existing_operator or printed_multiplication(
            gray, integer_digits
        )
        if not observed_operator:
            continue
        readings = [
            *refiner._rapid_readings(gray, numeric=True),
            *refiner._cyrillic_readings(gray),
        ]
        chosen = choose_unitless_expression(
            raw, readings, printed_operator=observed_operator
        )
        if chosen is not None:
            proposed = re.match(r"\d+[.,]\d{2}", chosen.text)
            assert proposed is not None
            if left > 0 and len(proposed[0]) > len(prefix[0]):
                # Padding must not borrow a preceding item ordinal to fill a
                # missing integer digit. That needs the word's own pixels.
                continue
            return chosen
    return None


def _source_separator(
    refiner: OcrRegionRefiner,
    page: Page,
    word: TextCell,
    gray_page: np.ndarray,
) -> RegionReading | None:
    from docling.models.stages.ocr.ocr_receipt_expressions import (
        choose_printed_separator,
    )

    match = re.fullmatch(r"[= ]*(\d{4,8})", word.text.strip())
    if match is None or page.size is None:
        return None
    box = word.rect.to_bounding_box()
    if box.height > box.width * 1.5:
        return None
    gray = rectified_text_crop(gray_page, word.rect, page.size)
    if gray is None or not printed_decimal(gray, len(match[1])):
        return None
    readings = [
        *refiner._rapid_readings(gray, numeric=True),
        *refiner._cyrillic_readings(gray),
    ]
    return choose_printed_separator(word.text, readings, printed_separator=True)


def _source_damaged_amount(
    refiner: OcrRegionRefiner,
    page: Page,
    word: TextCell,
    gray_page: np.ndarray,
) -> RegionReading | None:
    from docling.models.stages.ocr.ocr_receipt_expressions import (
        choose_damaged_amount,
        is_damaged_amount,
    )

    if not is_damaged_amount(word.text) or page.size is None:
        return None
    box = word.rect.to_bounding_box()
    if box.height > box.width * 1.5:
        return None
    gray = rectified_text_crop(gray_page, word.rect, page.size)
    if gray is None:
        return None
    readings = [
        *refiner._rapid_readings(gray, numeric=True),
        *refiner._cyrillic_readings(gray),
    ]
    return choose_damaged_amount(word.text, readings, gray)


def _source_word_pattern(text: str) -> re.Pattern[str]:
    # A stray join punctuation mark after a letter is not part of a larger
    # numeric value. Still reject matching inside another number or decimal.
    return re.compile(r"(?<!\d)(?<!\d[.,])" + re.escape(text) + r"(?!\d|[.,]\d)")


def _apply_source_word(
    page: Page,
    word: TextCell,
    reading: RegionReading,
    changed: ReceiptPriceRefinements,
) -> bool:
    assert page.parsed_page is not None
    box = word.rect.to_bounding_box()
    pattern = _source_word_pattern(word.text)
    owners = [
        line
        for line in page.parsed_page.textline_cells
        if line.from_ocr
        and box.intersection_over_self(line.rect.to_bounding_box()) >= 0.8
        and len(pattern.findall(line.text)) == 1
    ]
    if not owners:
        return False
    for line in owners:
        match = pattern.search(line.text)
        assert match is not None
        line.text = pattern.sub(lambda _: reading.text, line.text)
        changed.changed_lines.add(line.index)
        changed.record(
            line.index, line.text, (match.start(), match.start() + len(reading.text))
        )
    word.text, word.confidence = reading.text, reading.confidence
    return True


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
    for cell in verified:
        changed.record(cell.index, cell.text, (0, len(cell.text)))
    gray_page = np.asarray(image.convert("L"))
    for word in page.parsed_page.word_cells:
        if not word.from_ocr:
            continue
        if any(
            word.text == cell.text
            and word.rect.to_bounding_box().intersection_over_self(
                cell.rect.to_bounding_box()
            )
            >= 0.8
            for cell in verified
        ):
            continue
        separator = _source_separator(refiner, page, word, gray_page)
        if separator is not None and _apply_source_word(page, word, separator, changed):
            continue
        damaged = _source_damaged_amount(refiner, page, word, gray_page)
        if damaged is not None and _apply_source_word(page, word, damaged, changed):
            continue
        expression = _source_expression(refiner, page, word, gray_page)
        if expression is not None and _apply_source_word(
            page, word, expression, changed
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
            if (
                re.fullmatch(
                    r"[\d.,=+*x\u0445\u00d7() -]+", word.text.strip(), re.IGNORECASE
                )
                is None
            ):
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
            r"[\dx\u0445\u00d7*]-$", word.text[: amount.start()], re.IGNORECASE
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
        pattern = _source_word_pattern(word.text)
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
