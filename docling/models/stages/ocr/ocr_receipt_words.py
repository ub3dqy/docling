# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Verify damaged receipt words before assembling their printed lines."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Callable
from statistics import mean
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageOps

from docling.datamodel.base_models import Page
from docling.models.stages.ocr.ocr_region_refiner import (
    RegionReading,
    normalize_mixed_script,
    recognizer_family,
)
from docling.utils.ocr_receipt_rows import is_receipt_field_caption
from docling.utils.ocr_region_pixels import clean_region

if TYPE_CHECKING:
    from tesserocr import PyTessBaseAPI

    from docling.models.stages.ocr.ocr_region_refiner import OcrRegionRefiner


def choose_cyrillic_caption(
    original: str, readings: list[RegionReading], known: Callable[[str], bool]
) -> RegionReading | None:
    """Resolve Latin lookalikes only when two models observe the same caption."""
    visual = original.translate(
        str.maketrans("ABCEHKMOPTXYaceopxy", "АВСЕНКМОРТХУасеорху")  # noqa: RUF001 -- intended lookalikes
    )
    if not visual.isalpha() or re.search(r"[A-Za-z]", visual):
        return None
    groups: dict[str, dict[str, RegionReading]] = defaultdict(dict)
    for reading in readings:
        text = normalize_mixed_script(reading.text).strip()
        if (
            reading.confidence < 0.5
            or not text.isalpha()
            or re.search(r"[A-Za-z]", text)
            or not (known(text) or (original.isupper() and len(text) <= 5))
            or len(text) != len(visual)
            or sum(a != b for a, b in zip(visual.casefold(), text.casefold())) > 1
            or not (is_receipt_field_caption(text) or "касс" in text.casefold())
        ):
            continue
        family = recognizer_family(reading.engine)
        previous = groups[text.casefold()].get(family)
        if previous is None or previous.confidence < reading.confidence:
            groups[text.casefold()][family] = reading
    supported = [
        group
        for group in groups.values()
        if len(group) >= 2 and mean(r.confidence for r in group.values()) >= 0.55
    ]
    if len(supported) != 1:
        return None
    group = supported[0]
    best = max(group.values(), key=lambda reading: reading.confidence)
    return RegionReading(
        normalize_mixed_script(best.text).strip(),
        min(r.confidence for r in group.values()),
        "caption-consensus",
        "caption-consensus",
    )


def receipt_word_pattern(text: str) -> re.Pattern[str]:
    """Anchor letters while allowing a printed abbreviation beside the next word."""
    before = r"(?<!\w)" if text[0].isalnum() or text[0] == "_" else ""
    after = r"(?!\w)" if text[-1].isalnum() or text[-1] == "_" else ""
    return re.compile(before + re.escape(text) + after)


def refine_receipt_words(
    refiner: OcrRegionRefiner, page: Page, image: Image.Image, api: PyTessBaseAPI
) -> set[int]:
    assert page.parsed_page is not None
    changed = set()
    for word in page.parsed_page.word_cells:
        if (
            not word.from_ocr
            or not 3 <= len(word.text) <= 30
            or any(char.isdigit() for char in word.text)
        ):
            continue
        letters = re.findall(r"[A-Za-z\u0410-\u044f\u0401\u0451]+", word.text)
        if not letters:
            continue
        box = word.rect.to_bounding_box()
        lines = [
            line
            for line in page.parsed_page.textline_cells
            if line.from_ocr
            and box.intersection_over_self(line.rect.to_bounding_box()) >= 0.8
            and word.text in line.text
        ]
        caption = any(
            is_receipt_field_caption(line.text)
            or "безна" in normalize_mixed_script(line.text).casefold()
            or "карт" in normalize_mixed_script(line.text).casefold()
            for line in lines
        )
        if not caption:
            continue
        russian = re.search(r"[\u0410-\u044f\u0401\u0451]", word.text) is not None
        if not russian and not caption:
            continue
        if (
            all(refiner._dictionary.word_is_known(text) for text in letters)
            and word.text.isalpha()
        ):
            continue
        crop_box = box
        if russian and not word.text.isalpha():
            crop_box = box.model_copy(update={"l": box.l - box.height * 0.5})
        if caption and not russian:
            neighbors = [
                other.rect.to_bounding_box()
                for other in page.parsed_page.word_cells
                if other is not word
                and min(other.rect.to_bounding_box().b, box.b)
                - max(other.rect.to_bounding_box().t, box.t)
                >= min(box.height, other.rect.to_bounding_box().height) * 0.5
                and other.rect.to_bounding_box().l >= box.r
            ]
            right = min(
                [box.r + box.height * 1.5] + [other.l - 1 for other in neighbors]
            )
            crop_box = box.model_copy(
                update={"l": box.l - min(6, box.height * 0.3), "r": right}
            )
        gray = refiner._crop(image, crop_box, page, inset=-1.5)
        if gray is None:
            continue
        readings = [
            *refiner._rapid_readings(gray, numeric=False),
            *refiner._tess_readings(api, gray, numeric=False),
        ]
        if caption:
            import cv2

            cleaned = clean_region(gray, remove_noise=True)
            for factor in (2.0, 2.5):
                sample = np.asarray(
                    ImageOps.expand(
                        Image.fromarray(
                            cv2.resize(cleaned, None, fx=factor, fy=1)
                        ).convert("RGB"),
                        6,
                        fill="white",
                    )
                )
                for lang in ("iso:ru", "cyrillic"):
                    result = refiner._readers[lang](
                        sample, use_det=False, use_cls=False, use_rec=True
                    )
                    if result.txts and result.scores:
                        readings.append(
                            RegionReading(
                                result.txts[0],
                                float(result.scores[0]),
                                lang,
                                "wide-word",
                            )
                        )
        if word.text.isalpha():
            readings = [
                RegionReading(
                    reading.text.strip(" .,:;|\"'"),
                    reading.confidence,
                    reading.engine,
                    reading.variant,
                )
                for reading in readings
            ]
            readings = [
                reading
                for reading in readings
                if all(char.isalpha() or char.isspace() for char in reading.text)
                and sum(char.isalpha() for char in reading.text)
                >= len(word.text) * 0.85
            ]
        chosen = refiner._choose_text(word.text, word.confidence, readings)
        if not russian:
            confirmed = choose_cyrillic_caption(
                word.text, readings, refiner._dictionary.word_is_known
            )
            if confirmed is not None:
                chosen = confirmed
        if (
            chosen.text == word.text
            or not chosen.text
            or any(char.isdigit() for char in chosen.text)
        ):
            continue
        if not word.text.isalpha():
            original_letters = "".join(
                char for char in word.text if char.isalpha()
            ).casefold()
            candidate_letters = "".join(
                char for char in chosen.text if char.isalpha()
            ).casefold()
            observed_prefix = (
                chosen.text.startswith(("*", "=", "+", "№"))
                and candidate_letters == original_letters[1:]
            )
            if (original_letters != candidate_letters and not observed_prefix) or sum(
                not char.isalnum() and not char.isspace() for char in chosen.text
            ) < sum(not char.isalnum() and not char.isspace() for char in word.text):
                continue
            if (
                len(
                    {
                        recognizer_family(reading.engine)
                        for reading in readings
                        if reading.confidence >= 0.85
                        and reading.text.strip().casefold() == chosen.text.casefold()
                    }
                )
                < 2
            ):
                continue
        # An English trade name needs agreement before becoming Cyrillic.
        if (
            not russian
            and chosen.variant != "caption-consensus"
            and len(
                {
                    recognizer_family(reading.engine)
                    for reading in readings
                    if reading.confidence >= 0.7
                    and normalize_mixed_script(reading.text).strip().casefold()
                    == chosen.text.casefold()
                }
            )
            < 2
        ):
            continue
        replacement = chosen.text.upper() if word.text.isupper() else chosen.text
        pattern = receipt_word_pattern(word.text)
        for line in lines:
            if len(pattern.findall(line.text)) == 1:
                line.text = pattern.sub(lambda _: replacement, line.text)
                changed.add(line.index)
        word.text = replacement
    return changed
