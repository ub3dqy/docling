# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Re-read malformed monetary fields from their own complete source quads."""

from __future__ import annotations

import re
from collections import defaultdict
from statistics import mean
from typing import TYPE_CHECKING

import numpy as np

from docling.datamodel.base_models import Page
from docling.models.stages.ocr.ocr_receipt_expressions import (
    choose_damaged_amount,
    is_damaged_amount,
    opaque_amount_marker,
)
from docling.models.stages.ocr.ocr_region_refiner import (
    RegionReading,
    number_key,
    printed_decimal,
    recognizer_family,
)
from docling.utils.ocr_digit_templates import (
    connected_decimal_equals,
    decimal_expression_equals,
)
from docling.utils.ocr_receipt_rows import field_row_distance, is_receipt_field_caption
from docling.utils.ocr_region_pixels import clean_region
from docling.utils.ocr_text_geometry import extend_text_rect, rectified_text_crop

if TYPE_CHECKING:
    from PIL import Image

    from docling.models.stages.ocr.ocr_region_refiner import OcrRegionRefiner


def choose_fiscal_number(
    original: str,
    readings: list[RegionReading],
    *,
    printed_separator: bool,
    printed_equals: bool = False,
) -> RegionReading | None:
    """Require two independent source readings; preserve established amounts."""
    key = number_key(original)
    if key is not None and re.fullmatch(r"-?\d+\.\d{2}", key):
        if (
            not printed_separator
            or not printed_equals
            or re.fullmatch(r"[-\u2212]\d+[.,]\d{2}", original.strip()) is None
        ):
            return None
        # A complete negative amount normally stays immutable. A source marker,
        # independent amount readings and an explicit equals reading can prove
        # that only its separator was mistaken for a minus. Keep every digit,
        # the scale and the original decimal punctuation.
        amount = key[1:]
        votes: dict[str, float] = {}
        explicit_equals = False
        for reading in readings:
            observed = number_key(reading.text)
            if observed == key and reading.confidence >= 0.95:
                return None
            explicit_equals = explicit_equals or (
                observed == amount
                and reading.text.strip().startswith("=")
                and reading.confidence >= 0.9
            )
            if (
                reading.confidence >= 0.95
                and observed == amount
                and re.fullmatch(r"[= ]*\d+[.,]\d{2}", reading.text.strip())
            ):
                family = recognizer_family(reading.engine)
                votes[family] = max(votes.get(family, 0), reading.confidence)
        strongest = sorted(votes.values(), reverse=True)
        if explicit_equals and len(strongest) >= 2 and strongest[0] >= 0.97:
            return RegionReading(
                "=" + original.strip()[1:],
                min(strongest[:2]),
                "fiscal-marker-consensus",
                "source-quad-marker",
            )
        return None
    raw = original.strip()
    if (
        not printed_separator
        or re.fullmatch(r"[= ]*\d[\d (]*[.,]\d{2}|[= ]*\d{4,8}", raw) is None
        or raw.startswith("(")
    ):
        return None
    groups: dict[str, dict[str, RegionReading]] = defaultdict(dict)
    for reading in readings:
        candidate = number_key(reading.text)
        if (
            candidate is None
            or re.fullmatch(r"\d+\.\d{2}", candidate) is None
            or reading.confidence < 0.89
        ):
            continue
        family = recognizer_family(reading.engine)
        previous = groups[candidate].get(family)
        if previous is None or previous.confidence < reading.confidence:
            groups[candidate][family] = reading
    supported = []
    for value, group in groups.items():
        strongest = sorted(group.values(), key=lambda r: r.confidence, reverse=True)
        if (
            len(strongest) >= 2
            and strongest[0].confidence >= 0.97
            and mean(r.confidence for r in strongest[:2]) >= 0.94
        ):
            supported.append((value, strongest))
    if len(supported) != 1:
        return None
    value, strongest = supported[0]
    prefix = "=" if raw.startswith("=") else ""
    return RegionReading(
        prefix + value,
        min(r.confidence for r in strongest[:2]),
        "fiscal-number-consensus",
        "source-quad",
    )


def refine_fiscal_numbers(
    refiner: OcrRegionRefiner, page: Page, image: Image.Image
) -> set[int]:
    """Read only numeric-looking cells next to an observed fiscal caption."""
    assert page.parsed_page is not None and page.size is not None
    lines = [c for c in page.parsed_page.textline_cells if c.from_ocr]
    labels = [c for c in lines if is_receipt_field_caption(c.text)]
    gray_page = np.asarray(image.convert("L"))
    changed = set()
    for cell in lines:
        damaged = is_damaged_amount(cell.text)
        key = number_key(cell.text)
        negative = key is not None and re.fullmatch(r"-\d+\.\d{2}", key) is not None
        if (
            not damaged
            and not negative
            and (
                (key is not None and re.fullmatch(r"\d+\.\d{2}", key))
                or re.fullmatch(
                    r"[= ]*\d[\d (]*[.,]\d{2}|[= ]*\d{4,8}", cell.text.strip()
                )
                is None
            )
        ):
            continue
        box = cell.rect.to_bounding_box()
        contained = [
            word
            for word in page.parsed_page.word_cells
            if word.rect.to_bounding_box().intersection_over_self(box) >= 0.8
        ]
        if damaged and any(not word.from_ocr for word in contained):
            continue
        anchors = [
            label
            for label in labels
            if (
                label.index != cell.index
                and label.rect.to_bounding_box().r <= box.l
                and field_row_distance(label, cell)
                <= min(box.height, label.rect.to_bounding_box().height) * 0.7
            )
        ]
        if not anchors:
            continue
        chosen = None
        seen_padding: set[float] = set()
        for margin in (0.0, 0.3, 0.95) if negative or damaged else (0.0,):
            # First use the complete detector quad. More gutter can include
            # unrelated ink or change a recognizer's treatment of an equals.
            # Extend only unresolved fields, along their printed baseline and
            # within the gap before the caption. Each view must independently
            # meet the same source-marker and multi-recognizer requirements.
            left = min(
                box.height * margin,
                max(0, box.l - max(c.rect.to_bounding_box().r for c in anchors)),
            )
            if left in seen_padding:
                continue
            seen_padding.add(left)
            gray = rectified_text_crop(
                gray_page, extend_text_rect(cell.rect, left=left), page.size
            )
            if gray is None:
                continue
            if damaged:
                readings = [
                    *refiner._rapid_readings(gray, numeric=True),
                    *refiner._cyrillic_readings(gray),
                ]
                chosen = choose_damaged_amount(cell.text, readings, gray)
                if chosen is not None:
                    break
                continue
            printed_equals = False
            if negative:
                assert key is not None
                integer_digits = len(key[1:].partition(".")[0])
                printed_equals = (
                    decimal_expression_equals(
                        gray, integer_digits, allow_uneven_bars=True
                    )
                    or connected_decimal_equals(gray, integer_digits)
                    or opaque_amount_marker(gray)
                )
                if not printed_equals:
                    continue
            # An integer-shaped reading needs a visible decimal point.
            digits = sum(ch.isdigit() for ch in cell.text)
            separator = bool(re.search(r"[.,]\d{2}$", cell.text)) or printed_decimal(
                gray, digits
            )
            if not separator:
                continue
            readings = [
                *refiner._rapid_readings(gray, numeric=True),
                *refiner._cyrillic_readings(gray),
                *refiner._rapid_readings(
                    clean_region(gray, remove_noise=True), numeric=True
                ),
            ]
            chosen = choose_fiscal_number(
                cell.text,
                readings,
                printed_separator=separator,
                printed_equals=printed_equals,
            )
            if chosen is not None:
                break
        if chosen is None:
            continue
        original = cell.text
        cell.text, cell.confidence = chosen.text, chosen.confidence
        if (
            damaged
            and contained
            and "".join(
                w.text for w in sorted(contained, key=lambda w: w.index)
            ).replace(" ", "")
            == original.replace(" ", "")
        ):
            # A damaged line can have been split into a digit and punctuation.
            # Replace only the complete matching OCR source span; native text
            # was excluded above. Keep its observed full line quadrilateral.
            removed = {word.index for word in contained}
            merged = cell.model_copy(deep=True, update={"index": min(removed)})
            page.parsed_page.word_cells = [
                word
                for word in page.parsed_page.word_cells
                if word.index not in removed
            ] + [merged]
        for word in page.parsed_page.word_cells:
            if (
                word.from_ocr
                and word.text == original
                and word.rect.to_bounding_box().intersection_over_self(box) >= 0.8
            ):
                word.text, word.confidence = chosen.text, chosen.confidence
                word.rect = cell.rect
        changed.add(cell.index)
    return changed
