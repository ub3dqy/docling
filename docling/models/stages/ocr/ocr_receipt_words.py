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
from docling.utils.ocr_receipt_rows import field_row_distance, is_receipt_field_caption
from docling.utils.ocr_region_pixels import clean_region, terminal_exclamation

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
    if sum(ch.isdigit() for ch in original) == 1 and any(
        ch.isalpha() for ch in original
    ):
        visual = visual.translate(str.maketrans("0346k", "озчбк"))
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
            or not (
                is_receipt_field_caption(text)
                or "касс" in text.casefold()
                or text.casefold() == "чек"
            )
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


def _cash_heading(text: str) -> bool:
    # Classify lookalikes without changing a merchant's recognized spelling.
    return "КАСС" in text.upper().translate(str.maketrans("ACEHKMOPTXY", "АСЕНКМОРТХУ"))  # noqa: RUF001


def receipt_word_pattern(text: str) -> re.Pattern[str]:
    """Anchor letters while allowing a printed abbreviation beside the next word."""
    before = r"(?<!\w)" if text[0].isalnum() or text[0] == "_" else ""
    after = r"(?!\w)" if text[-1].isalnum() or text[-1] == "_" else ""
    return re.compile(before + re.escape(text) + after)


def preserves_receipt_codes(original: str, candidate: str) -> bool:
    """Do not lose an observed letter beside digits in a name or identifier."""
    protected = [
        token.casefold()
        for token in re.findall(r"[^\W_]+", normalize_mixed_script(original))
        if any(ch.isdigit() for ch in token) and any(ch.isalpha() for ch in token)
    ]
    content = "".join(ch for ch in candidate.casefold() if ch.isalnum())
    return all(token in content for token in protected)


def compatible_receipt_text(original: str, candidate: str) -> bool:
    """Keep caption-only crops free of invented amounts and calculation marks."""
    if not re.search(r"\d", original) and (
        re.search(r"\d", candidate)
        or any(mark in candidate and mark not in original for mark in "*=+")
    ):
        return False
    return preserves_receipt_codes(original, candidate)


def choose_observed_text(
    original: str,
    confidence: float,
    gray: np.ndarray,
    readings: list[RegionReading],
    known: Callable[[str], bool],
) -> RegionReading | None:
    """Recover a badly under-read text row, retaining genuine short identifiers.

    Two independent recognizers must read the same complete Russian words.
    The source must contain enough separate tall glyphs for that longer text;
    agreement alone cannot turn a five-character ID into a sentence.
    """
    if confidence > 0.75 or not any(ch.isalpha() for ch in original):
        return None
    import cv2

    cleaned = clean_region(gray)
    _, ink = cv2.threshold(cleaned, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    _, _, stats, _ = cv2.connectedComponentsWithStats(ink)
    glyphs = sum(
        h >= cleaned.shape[0] * 0.4 and area >= 4 and width <= h * 1.8
        for _, _, width, h, area in stats[1:]
    )
    old_length = sum(ch.isalnum() for ch in original)
    groups: dict[str, dict[str, RegionReading]] = defaultdict(dict)
    for reading in readings:
        text = normalize_mixed_script(reading.text).strip()
        words = re.findall(r"[\u0410-\u044f\u0401\u0451]+", text)
        length = sum(map(len, words))
        if (
            reading.confidence < 0.8
            or re.search(r"[A-Za-z0-9]", text)
            or len(words) < 2
            or not all(known(word) for word in words)
            or length < old_length * 1.5
            or glyphs < max(old_length * 1.5, length * 0.6)
        ):
            continue
        key = " ".join(word.casefold() for word in words)
        family = recognizer_family(reading.engine)
        previous = groups[key].get(family)
        if previous is None or previous.confidence < reading.confidence:
            groups[key][family] = reading
    supported = [group for group in groups.values() if len(group) >= 2]
    if len(supported) != 1:
        return None
    winner = supported[0]
    best = max(winner.values(), key=lambda r: r.confidence)
    return RegionReading(
        normalize_mixed_script(best.text).strip(),
        min(r.confidence for r in winner.values()),
        "observed-text-consensus",
        "source-glyphs",
    )


def choose_receipt_text(
    original: str,
    confidence: float,
    gray: np.ndarray,
    readings: list[RegionReading],
    selected: RegionReading,
    known: Callable[[str], bool],
) -> RegionReading:
    observed = choose_observed_text(original, confidence, gray, readings, known)
    if observed is not None:
        selected = observed
    if (
        selected.text
        and selected.text[-1].isalpha()
        and not any(ch.isdigit() for ch in selected.text)
        and all(known(word) for word in re.findall(r"[^\W\d_]+", selected.text))
        and terminal_exclamation(gray)
    ):
        selected = RegionReading(
            selected.text + "!",
            selected.confidence,
            selected.engine,
            selected.variant,
        )
    return selected


def receipt_text_readings(
    refiner: OcrRegionRefiner,
    gray: np.ndarray,
    text: str,
    confidence: float,
    *,
    enabled: bool,
) -> list[RegionReading]:
    """Use character alternatives for an uncertain sentence on a receipt."""
    if (
        not enabled
        or confidence >= 0.8
        or any(ch.isdigit() for ch in text)
        or is_receipt_field_caption(text)
        or len(re.findall(r"[^\W\d_]+", text)) < 2
    ):
        return []
    return refiner._cyrillic_readings(gray)


def refine_receipt_words(
    refiner: OcrRegionRefiner, page: Page, image: Image.Image, api: PyTessBaseAPI
) -> set[int]:
    assert page.parsed_page is not None
    changed = set()
    for word in page.parsed_page.word_cells:
        if (
            not word.from_ocr
            or not 3 <= len(word.text) <= 30
            or sum(char.isdigit() for char in word.text) > 1
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
            or _cash_heading(line.text)
            for line in lines
        )
        caption = caption or any(
            _cash_heading(anchor.text)
            and 0 <= box.l - anchor.rect.to_bounding_box().r <= box.height * 2
            and field_row_distance(anchor, word) <= box.height * 0.6
            for anchor in page.parsed_page.textline_cells
            if anchor.from_ocr
        )
        if not caption:
            continue
        if normalize_mixed_script(word.text).upper() in {
            "НДС",
            "ФНС",
            "ККТ",  # noqa: RUF001 -- Cyrillic fiscal acronym
            "ФД",
            "ФН",
            "ИНН",
            "ОФД",
        }:
            continue
        russian = re.search(r"[\u0410-\u044f\u0401\u0451]", word.text) is not None
        if not russian and not caption:
            continue
        if (
            all(refiner._dictionary.word_is_known(text) for text in letters)
            and word.text.isalpha()
            and word.confidence >= 0.95
        ):
            continue
        crop_box = box
        complete = [line for line in lines if line.text.strip() == word.text.strip()]
        if len(complete) == 1:
            outer = complete[0].rect.to_bounding_box()
            if outer.width <= box.width * 2:
                crop_box = outer
        if russian and not word.text.isalpha():
            crop_box = box.model_copy(update={"l": box.l - box.height * 0.5})
        if caption and not russian and not complete:
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
        if len(complete) == 1 and page.size is not None:
            from docling.utils.ocr_text_geometry import rectified_text_crop

            aligned = rectified_text_crop(
                np.asarray(image.convert("L")), complete[0].rect, page.size
            )
            if aligned is not None:
                gray = aligned
        if gray is None:
            continue
        readings = [
            *refiner._rapid_readings(gray, numeric=False),
            *refiner._tess_readings(api, gray, numeric=False),
            *refiner._cyrillic_readings(gray),
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
            any(ch.isdigit() for ch in word.text)
            and chosen.variant != "caption-consensus"
        ):
            continue
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
            if (
                original_letters != candidate_letters
                and not observed_prefix
                and chosen.variant != "caption-consensus"
            ) or sum(
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
