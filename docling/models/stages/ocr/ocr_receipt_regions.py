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

if TYPE_CHECKING:
    from docling.models.stages.ocr.ocr_region_refiner import OcrRegionRefiner


def refine_receipt_prices(
    refiner: OcrRegionRefiner, page: Page, image: Image.Image
) -> set[int]:
    """Verify individual printed amounts inside otherwise mixed receipt lines."""
    from rapidocr.ch_ppocr_rec.typings import TextRecOutput

    assert page.parsed_page is not None
    changed: set[int] = set()
    for word in page.parsed_page.word_cells:
        key = number_key(word.text)
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
        selected = number_key(choose_number(word.text, readings))
        if selected is None or selected == key:
            continue
        digits = iter(char for char in selected if char.isdigit())
        replacement = "".join(
            next(digits) if char.isdigit() else char for char in word.text
        )
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
