# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Verify printed prose rows before replacing fragmented OCR paragraphs."""

from __future__ import annotations

import csv
import io
import re
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass, replace
from difflib import SequenceMatcher
from statistics import median
from typing import TYPE_CHECKING

import numpy as np
from docling_core.types.doc import BoundingBox, CoordOrigin, DocItemLabel
from docling_core.types.doc.page import BoundingRectangle, TextCell

from docling.datamodel.base_models import Cluster, Page
from docling.models.stages.ocr.ocr_region_refiner import (
    normalize_mixed_script,
    recognizer_family,
)
from docling.utils.ocr_text_geometry import rectified_text_crop

if TYPE_CHECKING:
    from PIL import Image
    from tesserocr import PyTessBaseAPI

    from docling.models.stages.ocr.ocr_region_refiner import (
        OcrRegionRefiner,
        RegionReading,
    )

_WORD = re.compile(r"[A-Za-z\u0400-\u04ff0-9]+")
_NUMBER = re.compile(r"[+-]?\d+(?:[.,]\d+)?")
_IDENTIFIER = re.compile(
    r"(?:\u2116|номер|код|артикул|серия|гост|инн|огрн)\s*[:=]?\s*$", re.IGNORECASE
)


@dataclass(frozen=True)
class ProseRow:
    text: str
    confidence: float
    rect: BoundingRectangle
    words: tuple[TextCell, ...]
    quotes_confirmed: bool = False
    source_confirmed: bool = False


def prose_numbers(text: str, known: Callable[[str], bool]) -> tuple[str, ...]:
    """Keep genuine numbers while distinguishing observed letter-like digits.

    Only isolated 0/3 glyphs inside dictionary words qualify. All-digit tokens,
    decimal values, identifiers and consecutive digits remain protected.
    This classification supplies no replacement text to the recognizers.
    """

    def mask(match: re.Match[str]) -> str:
        word = match.group()
        digits = [char for char in word if char.isdigit()]
        if (
            not digits
            or word.isdigit()
            or any(char not in "03" for char in digits)
            or re.search(r"\d{2}", word)
            or _IDENTIFIER.search(text[max(0, match.start() - 30) : match.start()])
        ):
            return word
        letters = normalize_mixed_script(
            word.translate(str.maketrans("03", "\u041e\u0417"))
        )
        if letters.isalpha() and known(letters):
            return " " * len(word)
        return word

    masked = _WORD.sub(mask, text)
    return tuple(match.group().replace(",", ".") for match in _NUMBER.finditer(masked))


def numbers_have_source_positions(
    rows: list[ProseRow], original_words: list[TextCell], known: Callable[[str], bool]
) -> bool:
    available = [
        (word.index, index, value, word.rect.to_bounding_box())
        for word in original_words
        if word.from_ocr
        for index, value in enumerate(prose_numbers(word.text, known))
    ]
    used: set[tuple[int, int]] = set()
    for row in rows:
        for word in row.words:
            box = word.rect.to_bounding_box()
            for value in prose_numbers(word.text, known):
                candidates = [
                    item
                    for item in available
                    if item[:2] not in used
                    and item[2] == value
                    and box.intersection_over_self(item[3]) >= 0.5
                ]
                if not candidates:
                    return False
                selected = max(
                    candidates, key=lambda item: box.intersection_over_self(item[3])
                )
                used.add(selected[:2])
    return True


def _bad_letters(text: str, known: Callable[[str], bool]) -> int:
    return sum(
        len(word)
        for word in _WORD.findall(normalize_mixed_script(text))
        if any(char.isalpha() for char in word)
        and not (word.isupper() and word.isalpha() and len(word) <= 5)
        and not known(word)
    )


def _needs_reading(
    cluster: Cluster, known: Callable[[str], bool], *, verify_quotes: bool = False
) -> bool:
    if (
        cluster.label
        not in {DocItemLabel.TEXT, DocItemLabel.SECTION_HEADER, DocItemLabel.TITLE}
        or cluster.children
    ):
        return False
    if not cluster.cells or any(not cell.from_ocr for cell in cluster.cells):
        return False
    text = " ".join(cell.text for cell in cluster.cells)
    letters = [char for char in text if char.isalpha()]
    if (
        len(letters) < 15
        or sum("\u0400" <= char <= "\u04ff" for char in letters) < len(letters) * 0.7
    ):
        return False
    return bool(
        verify_quotes
        or _bad_letters(text, known) >= 3
        or (
            len(cluster.cells) >= 2
            and median(cell.rect.to_bounding_box().width for cell in cluster.cells)
            < cluster.bbox.width * 0.65
        )
    )


def _row_rect(words: list[TextCell], padding: float) -> BoundingRectangle:
    boxes = [word.rect.to_bounding_box() for word in words]
    centers = [((box.l + box.r) / 2, (box.t + box.b) / 2) for box in boxes]
    height = median(box.height for box in boxes)
    slopes = [
        (b[1] - a[1]) / (b[0] - a[0])
        for index, a in enumerate(centers)
        for b in centers[index + 1 :]
        if abs(b[0] - a[0]) > height * 5
    ]
    slope = median(slopes) if slopes else 0.0
    if abs(slope) > 0.1:
        slope = 0.0
    left, right = (
        min(box.l for box in boxes) - padding,
        max(box.r for box in boxes) + padding,
    )
    top = min(box.t - slope * (box.l + box.r) / 2 for box in boxes) - padding
    bottom = max(box.b - slope * (box.l + box.r) / 2 for box in boxes) + padding
    return BoundingRectangle(
        r_x0=left,
        r_y0=slope * left + bottom,
        r_x1=right,
        r_y1=slope * right + bottom,
        r_x2=right,
        r_y2=slope * right + top,
        r_x3=left,
        r_y3=slope * left + top,
        coord_origin=CoordOrigin.TOPLEFT,
    )


def _read_rows(
    api: PyTessBaseAPI, image: Image.Image, page: Page, cluster: Cluster
) -> list[ProseRow]:
    assert page.size is not None
    from tesserocr import PSM

    sx, sy = image.width / page.size.width, image.height / page.size.height
    box = cluster.bbox.to_top_left_origin(page.size.height)
    left, top = max(0, round((box.l - 1.5) * sx)), max(0, round((box.t - 1.5) * sy))
    right, bottom = (
        min(image.width, round((box.r + 1.5) * sx)),
        min(image.height, round((box.b + 1.5) * sy)),
    )
    if right - left < 8 or bottom - top < 8:
        return []
    api.SetPageSegMode(PSM.SINGLE_BLOCK if len(cluster.cells) > 1 else PSM.SINGLE_LINE)
    api.SetImage(image.crop((left, top, right, bottom)))
    api.Recognize()
    columns = "level page_num block_num par_num line_num word_num left top width height conf text".split()
    groups: dict[tuple[str, str, str], list[TextCell]] = defaultdict(list)
    for word in csv.DictReader(
        io.StringIO(api.GetTSVText(0)), fieldnames=columns, delimiter="\t"
    ):
        if word["level"] != "5" or not word["text"].strip() or float(word["conf"]) < 0:
            continue
        x, y = int(word["left"]), int(word["top"])
        width, height = int(word["width"]), int(word["height"])
        if width <= 0 or height <= 0:
            continue
        rect = BoundingRectangle.from_bounding_box(
            BoundingBox(
                l=(left + x) / sx,
                t=(top + y) / sy,
                r=(left + x + width) / sx,
                b=(top + y + height) / sy,
            )
        )
        groups[word["block_num"], word["par_num"], word["line_num"]].append(
            TextCell(
                index=0,
                text=word["text"],
                orig=word["text"],
                from_ocr=True,
                confidence=float(word["conf"]) / 100,
                rect=rect,
            )
        )
    return [
        ProseRow(
            text=" ".join(word.text for word in words),
            confidence=sum(word.confidence * len(word.text) for word in words)
            / sum(len(word.text) for word in words),
            rect=_row_rect(words, 3 / sx),
            words=tuple(words),
        )
        for words in groups.values()
    ]


def _key(text: str) -> str:
    return "".join(
        char for char in normalize_mixed_script(text).casefold() if char.isalnum()
    )


def quote_pattern(text: str) -> tuple[tuple[int, str], ...]:
    """Keep source quote positions relative to letters and digits."""
    offset = 0
    result = []
    for char in text:
        if char.isalnum():
            offset += 1
        elif char in "«»":
            result.append((offset, char))
    return tuple(result)


def confirmed_quote_repair(original: str, rows: list[ProseRow], deficit: int) -> bool:
    """Only add independently observed quotes that reduce a page imbalance."""
    candidate = " ".join(row.text for row in rows)
    if (
        not deficit
        or not all(row.quotes_confirmed for row in rows)
        or (
            _key(original) != _key(candidate)
            and not (
                rows
                and all(row.source_confirmed for row in rows)
                and _NUMBER.findall(original) == _NUMBER.findall(candidate)
            )
        )
    ):
        return False
    before, after = Counter(quote_pattern(original)), Counter(quote_pattern(candidate))
    if not before <= after:
        return False
    added = [
        mark for (_, mark), count in (after - before).items() for _ in range(count)
    ]
    return (
        bool(added)
        and all(mark == ("«" if deficit > 0 else "»") for mark in added)
        and len(added) <= abs(deficit)
    )


def prose_operators(text: str) -> Counter[str]:
    """Separate a prose dash between Cyrillic words from numeric operators."""
    text = re.sub(
        r"(?<=[\u0400-\u04ff])\s+[-\u2013\u2014]\s+(?=[\u0400-\u04ff])",
        " ",
        text,
    )
    return Counter(char for char in text if char in "-%=/№()[]")


def confirm_prose_row(
    row: ProseRow, readings: list[RegionReading], known: Callable[[str], bool]
) -> ProseRow:
    """Use a complete row only when two strong source readers agree exactly."""
    groups: dict[str, dict[str, RegionReading]] = defaultdict(dict)
    for reading in readings:
        text = " ".join(normalize_mixed_script(reading.text).split())
        if reading.confidence < 0.95 or prose_numbers(text, known) != prose_numbers(
            row.text, known
        ):
            continue
        family = recognizer_family(reading.engine)
        previous = groups[text.casefold()].get(family)
        if previous is None or previous.confidence < reading.confidence:
            groups[text.casefold()][family] = reading
    supported = [group for group in groups.values() if len(group) >= 2]
    if len(supported) != 1:
        return row
    group = supported[0]
    best = max(group.values(), key=lambda reading: reading.confidence)
    text = " ".join(normalize_mixed_script(best.text).split())
    tokens = text.split()
    if len(tokens) != len(row.words) or any(
        prose_numbers(word.text, known) != prose_numbers(token, known)
        for word, token in zip(row.words, tokens)
    ):
        return row
    return replace(
        row,
        text=text,
        confidence=min(reading.confidence for reading in group.values()),
        words=tuple(
            word.model_copy(update={"text": token})
            for word, token in zip(row.words, tokens)
        ),
        quotes_confirmed=True,
        source_confirmed=True,
    )


def repair_prose_words(
    row: ProseRow, readings: list[RegionReading], known: Callable[[str], bool]
) -> ProseRow:
    """Repair a damaged secondary word only when two models observe it in place."""
    groups: dict[str, list[RegionReading]] = defaultdict(list)
    for reading in readings:
        if reading.confidence >= 0.9:
            groups[_key(reading.text)].append(reading)
    supported = [
        group
        for group in groups.values()
        if len({recognizer_family(reading.engine) for reading in group}) >= 2
    ]
    if len(supported) != 1:
        return row
    candidate = max(supported[0], key=lambda reading: reading.confidence).text.split()
    if len(candidate) != len(row.words):
        return row
    words = []
    for word, observed in zip(row.words, candidate):
        letters = observed.strip("\"'«».,;:!?")
        if (
            _key(word.text) != _key(observed)
            and len([char for char in word.text if char.isalpha()]) >= 3
            and not any(char.isdigit() for char in word.text + observed)
            and letters.isalpha()
            and known(letters)
            and not known(word.text)
        ):
            words.append(word.model_copy(update={"text": observed}))
        else:
            words.append(word)
    return replace(row, text=" ".join(word.text for word in words), words=tuple(words))


def _confirmed_rows(
    refiner: OcrRegionRefiner, rows: list[ProseRow], gray: np.ndarray, page: Page
) -> bool:
    assert page.size is not None
    for index, row in enumerate(rows):
        if row.confidence < 0.8:
            return False
        crop = rectified_text_crop(gray, row.rect, page.size)
        if crop is None:
            return False
        key = _key(row.text)
        readings = [
            reading
            for reading in refiner._text_row_readings(crop)
            if reading.confidence >= 0.9
            and prose_numbers(reading.text, refiner._dictionary.word_is_known)
            == prose_numbers(row.text, refiner._dictionary.word_is_known)
            and SequenceMatcher(None, key, _key(reading.text), autojunk=False).ratio()
            >= 0.97
        ]
        row = confirm_prose_row(row, readings, refiner._dictionary.word_is_known)
        row = repair_prose_words(row, readings, refiner._dictionary.word_is_known)
        key = _key(row.text)
        if len(
            {recognizer_family(reading.engine) for reading in readings}
        ) < 2 or not any(_key(reading.text) == key for reading in readings):
            return False
        rows[index] = replace(
            row,
            quotes_confirmed=len(
                {
                    recognizer_family(reading.engine)
                    for reading in readings
                    if quote_pattern(reading.text) == quote_pattern(row.text)
                }
            )
            >= 2,
        )
    return True


def refine_pdf_prose(refiner: OcrRegionRefiner, page: Page) -> None:
    if page.parsed_page is None or page.size is None or page.predictions.layout is None:
        return
    known = refiner._dictionary.word_is_known
    page_text = " ".join(cell.text for cell in page.parsed_page.textline_cells)
    quote_deficit = page_text.count("»") - page_text.count("«")
    table_boxes = [
        cluster.bbox
        for cluster in page.predictions.layout.clusters
        if cluster.label == DocItemLabel.TABLE
    ]
    clusters = [
        cluster
        for cluster in page.predictions.layout.clusters
        if _needs_reading(cluster, known, verify_quotes=quote_deficit != 0)
        and not any(
            cluster.bbox.intersection_over_self(box) >= 0.2 for box in table_boxes
        )
    ]
    if not clusters:
        return
    image = page.get_image(scale=min(300 / 72, 6144 / max(page.size.as_tuple())))
    if image is None:
        return
    gray = np.asarray(image.convert("L"))
    with (
        refiner._lock,
        refiner._tesserocr.PyTessBaseAPI(
            path=str(refiner.tessdata),
            lang="rus+eng",
            psm=refiner._tesserocr.PSM.SINGLE_BLOCK,
            oem=refiner._tesserocr.OEM.LSTM_ONLY,
        ) as api,
    ):
        for cluster in clusters:
            words = [
                word
                for word in page.parsed_page.word_cells
                if word.rect.to_bounding_box().intersection_over_self(cluster.bbox)
                >= 0.7
            ]
            if any(not word.from_ocr for word in words):
                continue
            original = " ".join(cell.text for cell in cluster.cells)
            rows = _read_rows(api, image, page, cluster)
            candidate = " ".join(row.text for row in rows)
            original_length = len("".join(original.split()))
            if (
                not rows
                or not original_length * 0.75
                <= len("".join(candidate.split()))
                <= original_length * 1.25
            ):
                continue
            if Counter(prose_numbers(original, known)) != Counter(
                prose_numbers(candidate, known)
            ):
                continue
            if not _confirmed_rows(refiner, rows, gray, page):
                continue
            candidate = " ".join(row.text for row in rows)
            if prose_operators(original) != prose_operators(candidate):
                continue
            if (
                _bad_letters(candidate, known) >= _bad_letters(original, known)
                and len(cluster.cells) < len(rows) * 1.5
                and not confirmed_quote_repair(original, rows, quote_deficit)
            ):
                continue
            if not numbers_have_source_positions(rows, words, known):
                continue
            quote_deficit += (candidate.count("»") - candidate.count("«")) - (
                original.count("»") - original.count("«")
            )
            removed = {cell.index for cell in cluster.cells}
            line_index = (
                max(
                    (cell.index for cell in page.parsed_page.textline_cells), default=-1
                )
                + 1
            )
            word_index = (
                max((word.index for word in page.parsed_page.word_cells), default=-1)
                + 1
            )
            new_lines = []
            new_words = []
            for row in rows:
                new_lines.append(
                    TextCell(
                        index=line_index,
                        text=row.text,
                        orig=row.text,
                        confidence=row.confidence,
                        from_ocr=True,
                        rect=row.rect,
                    )
                )
                line_index += 1
                for word in row.words:
                    new_words.append(word.model_copy(update={"index": word_index}))
                    word_index += 1
            cluster.cells = new_lines
            page.parsed_page.textline_cells = [
                cell
                for cell in page.parsed_page.textline_cells
                if cell.index not in removed
            ] + new_lines
            word_ids = {word.index for word in words if word.from_ocr}
            page.parsed_page.word_cells = [
                word
                for word in page.parsed_page.word_cells
                if word.index not in word_ids
            ] + new_words
