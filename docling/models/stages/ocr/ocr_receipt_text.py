# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Verify damaged fiscal captions against independent source-pixel readings."""

from __future__ import annotations

import re
from collections.abc import Callable
from difflib import SequenceMatcher
from statistics import median
from typing import TYPE_CHECKING

import numpy as np
from docling_core.types.doc import BoundingBox
from docling_core.types.doc.page import BoundingRectangle, TextCell

from docling.datamodel.base_models import Cluster, Page
from docling.models.stages.ocr.ocr_region_refiner import (
    RegionReading,
    normalize_mixed_script,
    recognizer_family,
)
from docling.utils.ocr_receipt_rows import (
    field_row_distance,
    field_row_rect,
    is_receipt_field_caption,
)
from docling.utils.ocr_region_pixels import clean_region
from docling.utils.ocr_text_geometry import extend_text_rect, rectified_text_crop

if TYPE_CHECKING:
    from PIL import Image
    from tesserocr import PyTessBaseAPI

    from docling.models.stages.ocr.ocr_region_refiner import OcrRegionRefiner

_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
_FISCAL = re.compile(
    r"(?:ИТОГ|БЕЗНАЛИЧНЫМИ|ПЛАТ[.]?\s*КАРТОЙ|СУММА\s+НДС\s*\d+%)",  # noqa: RUF001 -- Russian fiscal captions
    re.IGNORECASE,
)


def _letters(text: str) -> str:
    return "".join(
        char for char in normalize_mixed_script(text).casefold() if char.isalpha()
    )


def _fiscal_letters(text: str) -> str:
    # Compare lookalikes in damaged fiscal abbreviations. This key is never
    # exported as recognized text and cannot change an identifier or amount.
    text = (
        re.sub(r"\d+.*$", "", text)
        .upper()
        .translate(
            str.maketrans(
                "ACEHKMOPTXY",
                "\u0410\u0421\u0415\u041d\u041a\u041c\u041e\u0420\u0422\u0425\u0423",
            )
        )
    )
    return _letters(text)


def choose_caption(
    original: str, readings: list[RegionReading], known: Callable[[str], bool]
) -> RegionReading | None:
    """Require actual fiscal words and independent evidence for a percent glyph."""
    for candidate in sorted(readings, key=lambda item: item.confidence, reverse=True):
        text = normalize_mixed_script(candidate.text).strip()
        if candidate.confidence < 0.9 or not _FISCAL.fullmatch(text.lstrip("* ")):
            continue
        before, after = _NUMBER.findall(original), _NUMBER.findall(text)
        if before != after:
            rate = re.search(r"(\d+)%", text)
            # A broken percent may be decoded as a trailing 7. This exception
            # is confined to a caption, never a price, ID or decimal amount.
            if (
                rate is None
                or "%" in original
                or not _fiscal_letters(original).startswith(_fiscal_letters(text)[:2])
                or before != [rate[1] + "7"]
                or not any(
                    other.confidence >= 0.6
                    and recognizer_family(other.engine)
                    != recognizer_family(candidate.engine)
                    and re.search(rf"(?<!\d){re.escape(rate[1])}%", other.text)
                    for other in readings
                )
            ):
                continue
        letters = _fiscal_letters(text)
        support = [
            reading
            for reading in readings
            if reading.confidence >= 0.55
            and recognizer_family(reading.engine) != recognizer_family(candidate.engine)
            and _fiscal_letters(reading.text)[:2] == letters[:2]
            and SequenceMatcher(
                None, letters, _fiscal_letters(reading.text), autojunk=False
            ).ratio()
            >= (0.6 if re.search(r"\d+%", text) else 0.8)
        ]
        if not support:
            continue
        # Complete dictionary words in an existing caption are not replaced
        # merely because a different known payment label has higher confidence.
        old_words = re.findall(r"[^\W\d_]+", normalize_mixed_script(original))
        candidate_words = {word.casefold() for word in re.findall(r"[^\W\d_]+", text)}
        fiscal_words = {
            "итог",
            "итого",
            "наличными",
            "безналичными",
            "картой",
            "сумма",
            "оплата",
            "получено",
        }
        if any(
            len(word) >= 4
            and known(word)
            and word.casefold() in fiscal_words
            and word.casefold() not in candidate_words
            for word in old_words
        ):
            continue
        return RegionReading(
            text, candidate.confidence, candidate.engine, candidate.variant
        )
    return None


def choose_amount_marks(
    original: str, readings: list[RegionReading]
) -> RegionReading | None:
    """Repair an observed leading mark, keeping amount digits and minus signs."""
    match = re.fullmatch(r"[=+.:≡ ]*(\d+[.,]\d{2})", original)
    if match is None:
        return None
    amount = match[1].replace(",", ".")
    candidates = [
        reading
        for reading in readings
        if reading.confidence >= 0.85
        and reading.text.strip().replace(",", ".") == "=" + amount
    ]
    if len({recognizer_family(reading.engine) for reading in candidates}) < 2:
        return None
    best = max(candidates, key=lambda reading: reading.confidence)
    if best.confidence < 0.97:
        return None
    return RegionReading("=" + match[1], best.confidence, best.engine, best.variant)


def _apply(page: Page, replacements: dict[int, TextCell], removed: set[int]) -> None:
    assert page.parsed_page is not None
    page.parsed_page.textline_cells = [
        replacements.get(cell.index, cell)
        for cell in page.parsed_page.textline_cells
        if cell.index not in removed
    ]

    def update(cluster: Cluster) -> None:
        cluster.cells = [
            replacements.get(cell.index, cell) if cell.from_ocr else cell
            for cell in cluster.cells
            if not cell.from_ocr or cell.index not in removed
        ]
        if any(cell.from_ocr and cell.index in replacements for cell in cluster.cells):
            boxes = [
                cluster.bbox,
                *(cell.rect.to_bounding_box() for cell in cluster.cells),
            ]
            cluster.bbox = BoundingBox(
                l=min(b.l for b in boxes),
                t=min(b.t for b in boxes),
                r=max(b.r for b in boxes),
                b=max(b.b for b in boxes),
            )
        for child in cluster.children:
            update(child)

    if page.predictions.layout is not None:
        for cluster in page.predictions.layout.clusters:
            update(cluster)


def remove_duplicate_fiscal_marks(page: Page, verified: set[int]) -> None:
    """Remove a second detection of pixels already read in a verified amount.

    The full amount must have an independently confirmed equals sign. Keep
    standalone operators and every mark outside that same source region.
    """
    assert page.parsed_page is not None
    lines = page.parsed_page.textline_cells
    import cv2

    def polygon(rect: BoundingRectangle) -> np.ndarray:
        return np.asarray(
            [
                (rect.r_x0, rect.r_y0),
                (rect.r_x1, rect.r_y1),
                (rect.r_x2, rect.r_y2),
                (rect.r_x3, rect.r_y3),
            ],
            dtype=np.float32,
        )

    amounts = [
        cell
        for cell in lines
        if cell.from_ocr
        and cell.index in verified
        and re.fullmatch(r"=\d+[.,]\d{2}", cell.text)
    ]
    removed: set[int] = set()
    for cell in lines:
        if not cell.from_ocr or cell.text != "+" or cell.confidence >= 0.6:
            continue
        box = cell.rect.to_bounding_box()
        shape = polygon(cell.rect)
        if not np.isfinite(shape).all() or cv2.contourArea(shape) <= 0:
            continue
        for amount in amounts:
            area = amount.rect.to_bounding_box()
            if (
                box.intersection_over_self(area) < 0.95
                or box.r > area.l + area.width * 0.3
                or box.height >= area.height * 0.65
            ):
                continue
            parent = polygon(amount.rect)
            if (
                np.isfinite(parent).all()
                and cv2.intersectConvexConvex(shape, parent)[0]
                >= cv2.contourArea(shape) * 0.95
            ):
                removed.add(cell.index)
                break
    _apply(page, {}, removed)


def refine_company_heading(
    refiner: OcrRegionRefiner, page: Page, gray: np.ndarray
) -> set[int]:
    assert page.parsed_page is not None and page.size is not None
    lines = [cell for cell in page.parsed_page.textline_cells if cell.from_ocr]
    replacements: dict[int, TextCell] = {}
    removed: set[int] = set()
    for prefix in lines:
        if (
            not re.fullmatch(r"[0\u041eO]{3}", prefix.text)
            or prefix.rect.to_bounding_box().b > page.size.height * 0.3
        ):
            continue
        for name in lines:
            box, first = name.rect.to_bounding_box(), prefix.rect.to_bounding_box()
            if (
                name.index == prefix.index
                or box.l < first.r
                or box.l - first.r > first.height * 3
            ):
                continue
            if (
                field_row_distance(prefix, name) > first.height * 0.6
                or len(re.findall(r"[\u0400-\u04ff]{4,}", name.text)) < 2
            ):
                continue
            rect = extend_text_rect(
                name.rect,
                left=box.l - first.l,
                right=box.height * 0.6,
                vertical=box.height * 0.1,
            )
            crop = rectified_text_crop(gray, rect, page.size)
            if crop is None:
                continue
            readings = refiner._text_row_readings(crop)
            for reading in sorted(
                readings, key=lambda item: item.confidence, reverse=True
            ):
                text = normalize_mixed_script(reading.text).strip()
                if reading.confidence < 0.95 or not re.match(
                    r'^\u041e{3}\s+["«]', text
                ):
                    continue
                if _NUMBER.findall(text[3:]) != _NUMBER.findall(name.text):
                    continue
                if (
                    SequenceMatcher(
                        None, _letters(text[3:]), _letters(name.text), autojunk=False
                    ).ratio()
                    < 0.95
                ):
                    continue
                if not any(
                    other.confidence >= 0.9
                    and other.engine != reading.engine
                    and re.match(r"[0\u041eO]{3}", other.text)
                    and _letters(other.text[3:]) == _letters(text[3:])
                    and _NUMBER.findall(other.text[3:]) == _NUMBER.findall(name.text)
                    for other in readings
                ):
                    continue
                replacements[prefix.index] = prefix.model_copy(
                    update={
                        "text": text,
                        "orig": prefix.orig + " " + name.orig,
                        "rect": rect,
                        "confidence": reading.confidence,
                    }
                )
                removed.add(name.index)
                break
    _apply(page, replacements, removed)
    return set(replacements)


def recover_total_caption(
    refiner: OcrRegionRefiner, page: Page, gray: np.ndarray
) -> None:
    """Look for a missing printed total beside an already observed amount.

    Two nearby fiscal rows must establish the caption column. No amount is
    created, changed or calculated. Both models must see the same label.
    """
    assert page.parsed_page is not None and page.size is not None
    lines = [cell for cell in page.parsed_page.textline_cells if cell.from_ocr]
    labels = [cell for cell in lines if is_receipt_field_caption(cell.text)]
    added: list[TextCell] = []
    for amount in lines:
        if re.fullmatch(r"[=*: ]*-?\d+[.,]\d{2}", amount.text) is None:
            continue
        if (
            min(
                amount.rect.r_x2 - amount.rect.r_x3, amount.rect.r_x1 - amount.rect.r_x0
            )
            <= 0
        ):
            continue
        box = amount.rect.to_bounding_box()
        body_height = min(
            box.height,
            (
                abs(amount.rect.r_y0 - amount.rect.r_y3)
                + abs(amount.rect.r_y1 - amount.rect.r_y2)
            )
            / 2,
        )
        if any(
            field_row_distance(label, amount) < box.height * 0.8
            and label.rect.to_bounding_box().r < box.l
            for label in labels
        ):
            continue
        peers = [
            label
            for label in labels
            if body_height * 0.5
            < abs(label.rect.to_bounding_box().t - box.t)
            < page.size.height * 0.1
            and label.rect.to_bounding_box().r < box.l
            and not re.search(r"\d+[.,]\d{2}", label.text)
        ]
        if len(peers) < 2:
            continue
        lefts = [label.rect.to_bounding_box().l for label in peers]
        if max(lefts) - min(lefts) > box.height:
            continue
        height = min(
            box.height, median(label.rect.to_bounding_box().height for label in peers)
        )
        left = median(lefts) - height * 0.3
        right = (
            left
            + median(label.rect.to_bounding_box().width for label in peers)
            + height
        )
        views = []
        for shift in (0.0, height * 0.1, -height * 0.1):
            sx, sy = gray.shape[1] / page.size.width, gray.shape[0] / page.size.height
            band_left, band_right = (
                round((left + shift) * sx) / sx,
                round((right + shift) * sx) / sx,
            )
            padding = round(height * 0.25 * sy) / sy
            source = amount.rect
            top_slope = (source.r_y2 - source.r_y3) / (source.r_x2 - source.r_x3)
            bottom_slope = (source.r_y1 - source.r_y0) / (source.r_x1 - source.r_x0)
            rect = BoundingRectangle(
                r_x0=band_left,
                r_y0=source.r_y0 + bottom_slope * (band_left - source.r_x0) + padding,
                r_x1=band_right,
                r_y1=source.r_y0 + bottom_slope * (band_right - source.r_x0) + padding,
                r_x2=band_right,
                r_y2=source.r_y3 + top_slope * (band_right - source.r_x3) - padding,
                r_x3=band_left,
                r_y3=source.r_y3 + top_slope * (band_left - source.r_x3) - padding,
                coord_origin=source.coord_origin,
            )
            crop = rectified_text_crop(gray, rect, page.size)
            if crop is not None:
                views.extend(
                    (reading, rect)
                    for reading in refiner._text_row_readings(
                        clean_region(crop, remove_noise=True)
                    )
                )
        readings = [reading for reading, _ in views]
        strong = [
            reading
            for reading in readings
            if reading.confidence >= 0.9 and reading.text.strip().casefold() == "итог"
        ]
        if not strong or not any(
            reading.confidence >= 0.55
            and reading.engine != strong[0].engine
            and _letters(reading.text) == _letters(strong[0].text)
            for reading in readings
        ):
            continue
        reading = strong[0]
        rect = next(rect for observed, rect in views if observed is reading)
        index = max((cell.index for cell in [*lines, *added]), default=-1) + 1
        cell = TextCell(
            index=index,
            text=reading.text,
            orig=reading.text,
            confidence=reading.confidence,
            from_ocr=True,
            rect=rect,
        )
        added.append(cell)
        if page.predictions.layout is not None:
            owners = [
                cluster
                for cluster in page.predictions.layout.clusters
                if any(c.index == amount.index and c.from_ocr for c in cluster.cells)
            ]
            if owners:
                owners[0].cells.append(cell)
    page.parsed_page.textline_cells.extend(added)


def refine_receipt_captions(
    refiner: OcrRegionRefiner, page: Page, image: Image.Image, api: PyTessBaseAPI
) -> set[int]:
    assert page.parsed_page is not None and page.size is not None
    gray = np.asarray(image.convert("L"))
    changed = refine_company_heading(refiner, page, gray)
    replacements: dict[int, TextCell] = {}
    captions = [
        cell
        for cell in page.parsed_page.textline_cells
        if cell.from_ocr and is_receipt_field_caption(cell.text)
    ]
    amounts = [
        cell
        for cell in page.parsed_page.textline_cells
        if cell.from_ocr and re.fullmatch(r"[=*: ]*-?\d+[.,]\d{2}", cell.text)
    ]
    for cell in page.parsed_page.textline_cells:
        if cell.from_ocr and re.fullmatch(r"[+.:≡][+.:≡ ]*\d+[.,]\d{2}", cell.text):
            crop = rectified_text_crop(gray, cell.rect, page.size)
            if crop is not None:
                readings = [
                    *refiner._text_row_readings(crop),
                    *refiner._text_row_readings(clean_region(crop, remove_noise=True)),
                ]
                selected = choose_amount_marks(cell.text, readings)
                if selected is not None:
                    replacements[cell.index] = cell.model_copy(
                        update={
                            "text": selected.text,
                            "confidence": selected.confidence,
                        }
                    )
            continue
        if (
            not cell.from_ocr
            or re.search(r"\d+[.,]\d{2}", cell.text)
            or not (
                is_receipt_field_caption(cell.text)
                or (
                    cell.confidence < 0.7
                    and bool(_letters(cell.text))
                    and len(cell.text) <= 40
                    and (
                        any(
                            abs(
                                cell.rect.to_bounding_box().l
                                - peer.rect.to_bounding_box().l
                            )
                            < cell.rect.to_bounding_box().height * 3
                            and abs(
                                cell.rect.to_bounding_box().t
                                - peer.rect.to_bounding_box().t
                            )
                            < page.size.height * 0.12
                            for peer in captions
                        )
                        or sum(
                            peer.rect.to_bounding_box().l
                            > cell.rect.to_bounding_box().r
                            and abs(
                                cell.rect.to_bounding_box().t
                                - peer.rect.to_bounding_box().t
                            )
                            < page.size.height * 0.12
                            for peer in amounts
                        )
                        >= 2
                    )
                )
            )
        ):
            continue
        if _FISCAL.fullmatch(cell.text.strip().lstrip("* ")):
            continue
        crop = rectified_text_crop(gray, cell.rect, page.size)
        if crop is None:
            continue
        cleaned = clean_region(crop, remove_noise=True)
        readings = [
            *refiner._text_row_readings(crop),
            *refiner._text_row_readings(cleaned),
            *refiner._tess_readings(api, cleaned, numeric=False),
        ]
        selected = choose_caption(
            cell.text, readings, refiner._dictionary.word_is_known
        )
        if selected is not None and selected.text != cell.text:
            replacements[cell.index] = cell.model_copy(
                update={"text": selected.text, "confidence": selected.confidence}
            )
    _apply(page, replacements, set())
    captions = [
        cell
        for cell in page.parsed_page.textline_cells
        if cell.from_ocr and is_receipt_field_caption(cell.text)
    ]
    for cell in page.parsed_page.textline_cells:
        if not cell.from_ocr or re.fullmatch(r"\d+[.,]\d{2}", cell.text) is None:
            continue
        box = cell.rect.to_bounding_box()
        if not any(
            label.rect.to_bounding_box().r < box.l
            and field_row_distance(label, cell)
            <= min(label.rect.to_bounding_box().height, box.height) * 0.6
            for label in captions
        ):
            continue
        crop = rectified_text_crop(gray, cell.rect, page.size)
        if crop is None:
            continue
        selected = choose_amount_marks(cell.text, refiner._text_row_readings(crop))
        if selected is not None:
            replacements[cell.index] = cell.model_copy(
                update={"text": selected.text, "confidence": selected.confidence}
            )
    _apply(page, replacements, set())
    recover_total_caption(refiner, page, gray)
    remove_duplicate_fiscal_marks(page, set(replacements))
    return changed | set(replacements)
