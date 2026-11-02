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
from docling.models.stages.ocr.ocr_region_refiner import (
    RegionReading,
    number_key,
    printed_decimal,
    recognizer_family,
)
from docling.utils.ocr_receipt_rows import field_row_distance, is_receipt_field_caption
from docling.utils.ocr_region_pixels import clean_region
from docling.utils.ocr_text_geometry import rectified_text_crop

if TYPE_CHECKING:
    from PIL import Image

    from docling.models.stages.ocr.ocr_region_refiner import OcrRegionRefiner


def choose_fiscal_number(
    original: str, readings: list[RegionReading], *, printed_separator: bool
) -> RegionReading | None:
    """Require two independent source readings; preserve established amounts."""
    key = number_key(original)
    if key is not None and re.fullmatch(r"-?\d+\.\d{2}", key):
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
        key = number_key(cell.text)
        if (key is not None and re.fullmatch(r"-?\d+\.\d{2}", key)) or re.fullmatch(
            r"[= ]*\d[\d (]*[.,]\d{2}|[= ]*\d{4,8}", cell.text.strip()
        ) is None:
            continue
        box = cell.rect.to_bounding_box()
        if not any(
            label.index != cell.index
            and label.rect.to_bounding_box().r <= box.l
            and field_row_distance(label, cell)
            <= min(box.height, label.rect.to_bounding_box().height) * 0.7
            for label in labels
        ):
            continue
        gray = rectified_text_crop(gray_page, cell.rect, page.size)
        if gray is None:
            continue
        # An integer-shaped reading needs a visible decimal point. A malformed
        # expression already containing a separator still needs model agreement.
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
        chosen = choose_fiscal_number(cell.text, readings, printed_separator=separator)
        if chosen is None:
            continue
        original = cell.text
        cell.text, cell.confidence = chosen.text, chosen.confidence
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
