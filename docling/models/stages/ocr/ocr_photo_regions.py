# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Read small photographic table labels from complete source cells."""

from __future__ import annotations

import re
from statistics import median
from typing import TYPE_CHECKING

import numpy as np
from docling_core.types.doc import BoundingBox, TableCell
from PIL import Image

from docling.datamodel.base_models import Page, Table
from docling.models.stages.ocr.ocr_region_refiner import (
    RegionReading,
    normalize_mixed_script,
    number_key,
    recognizer_family,
)

if TYPE_CHECKING:
    from tesserocr import PyTessBaseAPI

    from docling.models.stages.ocr.ocr_region_refiner import OcrRegionRefiner


def _word_region_readings(
    refiner: OcrRegionRefiner,
    api: PyTessBaseAPI,
    page: Page,
    image: Image.Image,
    cell: TableCell,
    anchor: str,
) -> list[RegionReading]:
    """Keep missing word endings without including the next cell or its frame."""
    assert (
        page.parsed_page is not None and page.size is not None and cell.bbox is not None
    )
    box = cell.bbox.to_top_left_origin(page.size.height)
    words = [
        word.rect.to_bounding_box()
        for word in page.parsed_page.word_cells
        if word.from_ocr
        and word.rect.to_bounding_box().intersection_over_self(box) > 0.6
    ]
    if not words:
        return []
    height = max(word.b for word in words) - min(word.t for word in words)
    tight = BoundingBox(
        l=max(box.l + 1, min(word.l for word in words) - height * 0.4),
        r=min(box.r - 1, max(word.r for word in words) + height * 0.8),
        t=max(box.t + 0.3, min(word.t for word in words)),
        b=min(box.b - 0.3, max(word.b for word in words)),
    )
    gray = refiner._crop(image, tight, page, inset=-0.25)
    if gray is None:
        return []
    readings = [
        *refiner._rapid_readings(gray, numeric=False),
        *refiner._tess_readings(api, gray, numeric=False),
    ]
    original_letters = sum(char.isalpha() for char in anchor)
    candidates = [
        reading
        for reading in readings
        if (tokens := re.findall(r"[A-Za-z\u0410-\u044f\u0401\u0451]+", reading.text))
        and all(refiner._dictionary.word_is_known(word) for word in tokens)
        and sum(map(len, tokens)) >= original_letters * 0.6
    ]
    anchor_words = re.findall(r"[A-Za-z\u0410-\u044f\u0401\u0451]+", anchor)
    if original_letters >= 4 and all(
        refiner._dictionary.word_is_known(word) for word in anchor_words
    ):

        def key(text: str) -> str:
            return "".join(normalize_mixed_script(text).casefold().split())

        # A single uncertain crop must not replace an already readable label.
        candidates = [
            reading
            for reading in candidates
            if key(reading.text) == key(anchor)
            or len(
                {
                    recognizer_family(other.engine)
                    for other in candidates
                    if other.confidence >= 0.65 and key(other.text) == key(reading.text)
                }
            )
            >= 2
        ]
    return candidates


def refine_photo_labels(
    refiner: OcrRegionRefiner,
    api: PyTessBaseAPI,
    page: Page,
    image: Image.Image,
    table: Table,
    regions: dict[int, np.ndarray],
) -> None:
    labels = [
        cell
        for cell in table.table_cells
        if cell.bbox is not None
        and cell.start_col_offset_idx == 0
        and cell.col_span == 1
        and not cell.column_header
    ]
    neighbors = [
        cell.bbox.l
        for cell in table.table_cells
        if cell.bbox is not None
        and cell.start_col_offset_idx == 1
        and number_key(cell.text) is not None
    ]
    if (
        sum(sum(char.isalpha() for char in cell.text) >= 4 for cell in labels) < 5
        or not neighbors
        or median(cell.bbox.height for cell in labels if cell.bbox is not None) >= 12
    ):
        return
    left = max(
        table.cluster.bbox.l - 0.5,
        min(cell.bbox.l for cell in labels if cell.bbox is not None),
    )
    right = min(neighbors) - 1
    for cell in labels:
        assert cell.bbox is not None
        gray = regions.get(id(cell))
        if gray is None:
            gray = refiner._crop(
                image,
                cell.bbox.model_copy(update={"l": left, "r": right}),
                page,
                inset=-0.5,
            )
        if gray is None or gray.size == 0:
            continue
        readings = refiner._rapid_readings(gray, numeric=False)
        selected = refiner._choose_text(cell.text, 0.65, readings, photo_label=True)
        words = re.findall(r"[\u0410-\u044f\u0401\u0451]+", selected.text)
        if (
            not words
            or any(
                not refiner._dictionary.word_is_known(word)
                or (len(word) <= 2 and word.casefold() not in {"и", "в", "на"})
                for word in words
            )
            or re.search(r"[A-Za-z]", selected.text)
        ):
            # Keep the actual glyphs near the left rule. Rule removal can
            # otherwise erase the first stroke of a small serif character.
            from PIL import ImageOps

            tight = gray[:, 4:-4] if gray.shape[1] > 20 else gray
            for lang in ("iso:ru", "cyrillic"):
                sample = np.asarray(
                    ImageOps.expand(
                        Image.fromarray(tight).convert("RGB"), 6, fill="white"
                    )
                )
                result = refiner._readers[lang](
                    sample, use_det=False, use_cls=False, use_rec=True
                )
                if result.txts and result.scores:
                    readings.append(
                        RegionReading(
                            result.txts[0], float(result.scores[0]), lang, "source-cell"
                        )
                    )
            readings.extend(refiner._photo_readings(api, gray, table_label=True))
            readings.extend(refiner._tess_readings(api, gray, numeric=False))
            selected = refiner._choose_text(cell.text, 0.65, readings, photo_label=True)
        # Preserve the complete-cell reading before checking clipped word
        # endings. A known label needs independent support for a replacement.
        sources = [image]
        smaller = page.get_image(scale=2)
        if smaller is not None and smaller.size != image.size:
            sources.append(smaller)
        for source in sources:
            words = re.findall(r"[\u0410-\u044f\u0401\u0451]+", selected.text)
            if (
                selected.confidence >= 0.9
                and sum(map(len, words)) >= 4
                and all(refiner._dictionary.word_is_known(word) for word in words)
            ):
                break
            readings.extend(
                _word_region_readings(refiner, api, page, source, cell, selected.text)
            )
            selected = refiner._choose_text(cell.text, 0.65, readings, photo_label=True)
        cell.text = selected.text


def photo_number_readings(
    refiner: OcrRegionRefiner, gray: np.ndarray, text: str
) -> list[RegionReading]:
    """Avoid repeated expensive crops when two models confirm the existing value."""
    from PIL import ImageOps

    from docling.utils.ocr_region_pixels import clean_region

    sample = np.asarray(
        ImageOps.expand(
            Image.fromarray(clean_region(gray)).convert("RGB"), 6, fill="white"
        )
    )
    readings = []
    for lang in ("iso:ru", "cyrillic"):
        result = refiner._readers[lang](
            sample, use_det=False, use_cls=False, use_rec=True
        )
        if result.txts and result.scores:
            readings.append(
                RegionReading(result.txts[0], float(result.scores[0]), lang, "clean")
            )
    if len(readings) == 2 and all(
        reading.confidence >= 0.95 and number_key(reading.text) == number_key(text)
        for reading in readings
    ):
        return readings
    return refiner._rapid_readings(gray, numeric=True)
