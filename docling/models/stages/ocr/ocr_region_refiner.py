# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Read uncertain Russian lines and table cells from their original pixels.

Candidates come only from recognizers. The dictionary ranks existing readings;
it never supplies replacement words. No totals, tax rates or document templates
are used to manufacture digits.
"""

from __future__ import annotations

import re
import threading
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from statistics import mean, median
from typing import TYPE_CHECKING

import numpy as np
from docling_core.types.doc import BoundingBox
from PIL import Image, ImageOps

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import Cluster, Page, PageElement, Table
from docling.datamodel.pipeline_options import RapidOcrOptions
from docling.datamodel.settings import settings
from docling.models.stages.ocr.rapid_ocr_model import RapidOcrModel

if TYPE_CHECKING:
    from tesserocr import PyTessBaseAPI

_CYRILLIC = re.compile(r"[\u0410-\u044f\u0401\u0451]")
_WORDS = re.compile(r"[A-Za-z\u0410-\u044f\u0401\u0451]+")
_CONFUSABLES = str.maketrans(
    "ABCEHKMOPTXYaceopxy",
    "АВСЕНКМОРТХУасеорху",  # noqa: RUF001 -- intended lookalikes
)
_NUMBER = re.compile(r"-?\d[\d ]*(?:[.,]\d+)?")


@dataclass(frozen=True)
class RegionReading:
    text: str
    confidence: float
    engine: str
    variant: str


def normalize_mixed_script(text: str) -> str:
    """Resolve lookalikes inside a predominantly Cyrillic word only."""

    def replace(match: re.Match[str]) -> str:
        word = match.group()
        if len(_CYRILLIC.findall(word)) * 2 >= len(word):
            return word.translate(_CONFUSABLES)
        return word

    return _WORDS.sub(replace, text)


def number_key(text: str) -> str | None:
    """Keep decimal separators significant; never infer a missing decimal mark."""
    value = " ".join(text.replace("\u2212", "-").split()).strip("=*: ")
    if value.endswith(".") and value.count(".") > 1:
        value = value[:-1]
    negative = value.startswith("(") and value.endswith(")")
    if negative:
        value = value[1:-1].strip()
    if _NUMBER.fullmatch(value) is None:
        return None
    key = value.replace(" ", "").replace(",", ".")
    return "-" + key if negative else key


def clean_region(gray: np.ndarray, *, remove_noise: bool = False) -> np.ndarray:
    """Remove long border rules, then crop to glyphs without deleting dots."""
    import cv2

    height, width = gray.shape
    if height < 4 or width < 4:
        return gray
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    horizontal = cv2.morphologyEx(
        ink, cv2.MORPH_OPEN, np.ones((1, max(15, round(width * 0.7))), np.uint8)
    )
    vertical = cv2.morphologyEx(
        ink, cv2.MORPH_OPEN, np.ones((max(10, round(height * 0.7)), 1), np.uint8)
    )
    vertical[:, round(width * 0.15) : round(width * 0.85)] = 0
    mask = cv2.dilate(cv2.bitwise_or(horizontal, vertical), np.ones((2, 2), np.uint8))
    result = gray.copy()
    result[mask > 0] = 255
    _, ink = cv2.threshold(result, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(
        np.asarray(ink, dtype=np.uint8), connectivity=8
    )
    body = [
        stat
        for stat in stats[1:]
        if stat[cv2.CC_STAT_HEIGHT] >= height * 0.25
        and stat[cv2.CC_STAT_AREA] >= 8
        and stat[cv2.CC_STAT_WIDTH] < width * 0.9
    ]
    if not body:
        return result
    char_height = median(int(stat[cv2.CC_STAT_HEIGHT]) for stat in body)
    if remove_noise:
        for label, stat in enumerate(stats[1:], 1):
            if (
                stat[cv2.CC_STAT_AREA] < max(3, char_height**2 * 0.012)
                or stat[cv2.CC_STAT_HEIGHT] < char_height * 0.18
            ):
                result[labels == label] = 255
    padding = max(2, round(char_height * 0.15))
    top = max(0, min(int(stat[cv2.CC_STAT_TOP]) for stat in body) - padding)
    left = max(0, min(int(stat[cv2.CC_STAT_LEFT]) for stat in body) - padding)
    bottom = min(
        height,
        max(int(stat[cv2.CC_STAT_TOP] + stat[cv2.CC_STAT_HEIGHT]) for stat in body)
        + padding,
    )
    right = min(
        width,
        max(int(stat[cv2.CC_STAT_LEFT] + stat[cv2.CC_STAT_WIDTH]) for stat in body)
        + padding,
    )
    return result[top:bottom, left:right]


def choose_number(
    original: str, readings: list[RegionReading], *, allow_decimal_removal: bool = False
) -> str:
    """Require independent recognizers to agree before replacing a table number."""
    groups: dict[str, dict[str, RegionReading]] = defaultdict(dict)
    original_key = number_key(original)
    if original_key is None:
        return original
    for reading in readings:
        key = number_key(reading.text)
        if key is None or reading.confidence < 0.85:
            continue
        if sum(char.isdigit() for char in key) != sum(
            char.isdigit() for char in original_key
        ) or key.startswith("-") != original_key.startswith("-"):
            continue
        original_fraction = original_key.partition(".")[2]
        fraction = key.partition(".")[2]
        if len(original_fraction) != len(fraction) and not (
            allow_decimal_removal and len(original_fraction) > 3 and not fraction
        ):
            continue
        previous = groups[key].get(reading.engine)
        if previous is None or reading.confidence > previous.confidence:
            groups[key][reading.engine] = reading
    supported = [group for group in groups.values() if len(group) >= 2]
    if not supported:
        return original
    winner = max(
        supported,
        key=lambda group: (len(group), mean(r.confidence for r in group.values())),
    )
    best = max(winner.values(), key=lambda reading: reading.confidence)
    return best.text.strip()


def choose_line_number(original: str, readings: list[RegionReading]) -> str:
    """Keep a receipt amount unless two segmentation modes verify one digit."""
    original_key = number_key(original)
    if original_key is None:
        return original
    wide = [
        reading
        for reading in readings
        if reading.variant == "wide"
        and reading.confidence >= 0.78
        and number_key(reading.text) is not None
    ]
    if len(wide) != 2 or wide[0].engine == wide[1].engine:
        return original
    new_key = number_key(wide[0].text)
    if new_key != number_key(wide[1].text) or new_key is None:
        return original
    if (
        len(original_key) != len(new_key)
        or [i for i, char in enumerate(original_key) if char in ".-"]
        != [i for i, char in enumerate(new_key) if char in ".-"]
        or sum(a != b for a, b in zip(original_key, new_key)) > 1
    ):
        return original
    return ("=" if original.startswith("=") else "") + wide[0].text.strip("= ")


def compatible_text_reading(original: str, candidate: str) -> bool:
    """Allow OCR spacing changes without losing decimal marks or printed digits."""
    original_length = len("".join(original.split()))
    candidate_length = len("".join(candidate.split()))
    if not 0.55 * original_length <= candidate_length <= 1.8 * original_length:
        return False
    original_numbers = [
        token.replace(",", ".") for token in re.findall(r"\d+(?:[.,]\d+)?", original)
    ]
    candidate_numbers = [
        token.replace(",", ".") for token in re.findall(r"\d+(?:[.,]\d+)?", candidate)
    ]
    return not original_numbers or original_numbers == candidate_numbers


def refine_word_choices(
    reading: RegionReading,
    candidates: list[RegionReading],
    word_is_known: Callable[[str], bool],
) -> RegionReading:
    """Combine observed words only where matching neighbors anchor a disagreement."""
    if reading.confidence >= 0.95:
        return reading
    matches = list(_WORDS.finditer(reading.text))
    words = [match.group().casefold() for match in matches]
    replacements: dict[int, tuple[str, float]] = {}
    for candidate in candidates:
        if candidate.confidence < 0.7 or not compatible_text_reading(
            reading.text, candidate.text
        ):
            continue
        alternative = _WORDS.findall(normalize_mixed_script(candidate.text))
        opcodes = SequenceMatcher(
            None, words, [word.casefold() for word in alternative], autojunk=False
        ).get_opcodes()
        for position, (tag, start, end, other_start, other_end) in enumerate(opcodes):
            if (
                tag != "replace"
                or end - start != 1
                or not 1 <= other_end - other_start <= 2
                or position == 0
                or position == len(opcodes) - 1
                or opcodes[position - 1][0] != "equal"
                or opcodes[position + 1][0] != "equal"
                or len(words[start]) < 4
                or _CYRILLIC.fullmatch(words[start][0]) is None
                or len(_CYRILLIC.findall(words[start])) != len(words[start])
                or word_is_known(words[start])
            ):
                continue
            proposed = alternative[other_start:other_end]
            if not all(word_is_known(word) for word in proposed):
                continue
            joined = "".join(proposed).casefold()
            if (
                SequenceMatcher(None, words[start], joined, autojunk=False).ratio()
                < 0.75
            ):
                continue
            previous = replacements.get(start)
            if previous is None or candidate.confidence > previous[1]:
                replacements[start] = (" ".join(proposed), candidate.confidence)
    if not replacements:
        return reading
    text = reading.text
    for index in sorted(replacements, reverse=True):
        match = matches[index]
        text = text[: match.start()] + replacements[index][0] + text[match.end() :]
    return RegionReading(
        text,
        min(reading.confidence, *(score for _, score in replacements.values())),
        "region-consensus",
        "anchored-words",
    )


class OcrRegionRefiner:
    """A pipeline-owned refiner with separate, serialized recognition readers."""

    def __init__(
        self, *, artifacts_path: Path | None, accelerator_options: AcceleratorOptions
    ) -> None:
        try:
            import pymorphy3
            import tesserocr
        except ImportError as error:
            raise ImportError(
                "Russian region refinement requires docling-slim[russian]. "
                "Alternatively, set use_region_refinement=False."
            ) from error
        root = artifacts_path or settings.cache_dir / "models"
        self.tessdata = root / "tessdata"
        from docling.utils.russian_ocr_models import validate_tessdata

        validate_tessdata(self.tessdata)
        self._tesserocr = tesserocr
        self._dictionary = pymorphy3.MorphAnalyzer()
        self._lock = threading.Lock()
        self._readers = {
            lang: RapidOcrModel(
                enabled=True,
                artifacts_path=root,
                options=RapidOcrOptions(
                    lang=[lang], use_det=False, use_cls=False, text_score=0
                ),
                accelerator_options=accelerator_options,
            ).reader
            for lang in ("iso:ru", "cyrillic", "ch")
        }

    @staticmethod
    def _crop(
        image: Image.Image, box: BoundingBox, page: Page, *, inset: float
    ) -> np.ndarray | None:
        assert page.size is not None
        box = box.to_top_left_origin(page.size.height)
        sx, sy = image.width / page.size.width, image.height / page.size.height
        left = max(0, round((box.l + inset) * sx))
        top = max(0, round((box.t + inset) * sy))
        right = min(image.width, round((box.r - inset) * sx))
        bottom = min(image.height, round((box.b - inset) * sy))
        if right - left < 4 or bottom - top < 4:
            return None
        return np.asarray(image.crop((left, top, right, bottom)).convert("L"))

    def _rapid_readings(
        self, gray: np.ndarray, *, numeric: bool
    ) -> list[RegionReading]:
        import cv2
        from rapidocr.ch_ppocr_rec.typings import TextRecOutput

        cleaned = clean_region(gray)
        variants = {
            "clean": cleaned,
            "wide": cv2.resize(
                cleaned, None, fx=1.5, fy=1, interpolation=cv2.INTER_CUBIC
            ),
        }
        if not numeric:
            variants["threshold"] = np.where(gray < 130, 0, 255).astype(np.uint8)
        readings: list[RegionReading] = []
        for lang in ("iso:ru", "cyrillic", "ch") if numeric else ("iso:ru", "cyrillic"):
            for name, array in variants.items():
                sample = ImageOps.expand(
                    Image.fromarray(array).convert("RGB"), 6, fill="white"
                )
                # Individual calls prevent a long neighboring crop from changing
                # the batch's padding and therefore this region's prediction.
                result = self._readers[lang](
                    np.asarray(sample), use_det=False, use_cls=False, use_rec=True
                )
                if isinstance(result, TextRecOutput) and result.txts and result.scores:
                    readings.append(
                        RegionReading(
                            result.txts[0], float(result.scores[0]), lang, name
                        )
                    )
        return readings

    def _refine_ambiguous_digit(
        self, api: PyTessBaseAPI, gray: np.ndarray, original: str
    ) -> str:
        """Verify a one-digit disagreement against an isolated glyph.

        Keep the original unless a whole-cell reading and two glyph readings
        agree. Glyph segmentation must account for every digit, so a rule,
        decimal mark or connected pair cannot shift the character positions.
        """
        import cv2

        key = number_key(original)
        if key is None or re.fullmatch(r"\d{4,9}", key) is None:
            return original
        alternatives: dict[int, set[str]] = defaultdict(set)
        for reading in self._tess_readings(api, gray, numeric=True):
            candidate = number_key(reading.text)
            if (
                candidate is None
                or len(candidate) != len(key)
                or not candidate.isdigit()
            ):
                continue
            differences = [i for i, (a, b) in enumerate(zip(key, candidate)) if a != b]
            if len(differences) == 1:
                index = differences[0]
                alternatives[index].add(candidate[index])
        if not alternatives:
            return original
        cleaned = clean_region(gray)
        _, ink = cv2.threshold(cleaned, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        _, _, stats, _ = cv2.connectedComponentsWithStats(
            np.asarray(ink, dtype=np.uint8), connectivity=8
        )
        boxes = sorted(
            (int(x), int(y), int(width), int(height))
            for x, y, width, height, _ in stats[1:]
            if height > cleaned.shape[0] * 0.5 and width > 2
        )
        if len(boxes) != len(key):
            return original
        verified: list[tuple[int, str]] = []
        api.SetVariable("tessedit_char_whitelist", "0123456789")
        api.SetPageSegMode(self._tesserocr.PSM.RAW_LINE)
        for index, digits in alternatives.items():
            x, y, width, height = boxes[index]
            glyph = cleaned[
                max(0, y - 1) : y + height + 1,
                max(0, x - 1) : x + width + 1,
            ]
            readings: list[tuple[str, float]] = []
            for factor in (1.5, 2.0):
                sample = Image.fromarray(
                    cv2.resize(
                        glyph, None, fx=factor, fy=1, interpolation=cv2.INTER_CUBIC
                    )
                )
                api.SetImage(ImageOps.expand(sample, 12, fill=255))
                readings.append((api.GetUTF8Text().strip(), api.MeanTextConf() / 100))
            if (
                readings[0][0] == readings[1][0]
                and readings[0][0] in digits
                and min(confidence for _, confidence in readings) >= 0.6
                and mean(confidence for _, confidence in readings) >= 0.65
            ):
                verified.append((index, readings[0][0]))
        if len(verified) != 1:
            return original
        digit_index, replacement = verified[0]
        position = [i for i, char in enumerate(original) if char.isdigit()][digit_index]
        return original[:position] + replacement + original[position + 1 :]

    def _tess_readings(
        self, api: PyTessBaseAPI, gray: np.ndarray, *, numeric: bool
    ) -> list[RegionReading]:
        import cv2

        cleaned = clean_region(gray)
        wide = cv2.resize(cleaned, None, fx=1.5, fy=1, interpolation=cv2.INTER_CUBIC)
        denoised = clean_region(gray, remove_noise=not numeric)
        variants = [("clean", cleaned, 1), ("denoise", denoised, 2), ("wide", wide, 1)]
        if not numeric:
            variants.append(("wide-upscaled", wide, 2))
        api.SetVariable("tessedit_char_whitelist", "0123456789.,()-" if numeric else "")
        readings: list[RegionReading] = []
        modes = (
            (self._tesserocr.PSM.SINGLE_LINE, self._tesserocr.PSM.RAW_LINE)
            if numeric
            else (self._tesserocr.PSM.SINGLE_LINE,)
        )
        for mode in modes:
            api.SetPageSegMode(mode)
            for name, array, scale in variants:
                sample = Image.fromarray(array)
                if scale != 1:
                    sample = sample.resize(
                        (sample.width * scale, sample.height * scale),
                        Image.Resampling.LANCZOS,
                    )
                api.SetImage(ImageOps.expand(sample, 12, fill=255))
                readings.append(
                    RegionReading(
                        api.GetUTF8Text().strip(),
                        api.MeanTextConf() / 100,
                        f"tesseract-{mode}",
                        name,
                    )
                )
        return readings

    def _choose_text(
        self, original: str, confidence: float, readings: list[RegionReading]
    ) -> RegionReading:
        candidates = [
            RegionReading(original, confidence, "page", "original"),
            *readings,
        ]

        def score(reading: RegionReading) -> float:
            text = normalize_mixed_script(reading.text)
            words = _WORDS.findall(text)
            if not words or (reading.engine != "page" and reading.confidence < 0.65):
                return -1
            if reading.engine != "page" and not compatible_text_reading(original, text):
                return -1
            letters = sum(map(len, words))
            known = sum(
                len(word) for word in words if self._dictionary.word_is_known(word)
            )
            russian = len(_CYRILLIC.findall(text)) / letters
            # Rank actual readings; unknown names remain possible. Penalize
            # extra punctuation and favor a complete line over a clipped word.
            punctuation = sum(
                not char.isalnum() and not char.isspace() for char in text
            )
            return float(
                reading.confidence
                * (0.65 + 0.35 * known / letters)
                * (0.7 + 0.3 * russian)
                * float(letters) ** 0.1
                - 0.015 * punctuation
            )

        best = max(candidates, key=score)
        chosen = RegionReading(
            normalize_mixed_script(best.text).strip(),
            best.confidence,
            best.engine,
            best.variant,
        )
        return refine_word_choices(chosen, candidates, self._dictionary.word_is_known)

    def refine_lines(self, page: Page, *, image_input: bool) -> None:
        if not image_input or page.parsed_page is None or page.size is None:
            return
        lines = [cell for cell in page.parsed_page.textline_cells if cell.from_ocr]
        if not lines:
            return
        image = page.get_image(scale=1)
        if image is None:
            return
        heights = [cell.rect.to_bounding_box().height for cell in lines]
        line_height = median(heights)
        replacements: dict[int, RegionReading] = {}
        with (
            self._lock,
            self._tesserocr.PyTessBaseAPI(
                path=str(self.tessdata),
                lang="rus+eng",
                psm=self._tesserocr.PSM.SINGLE_LINE,
                oem=self._tesserocr.OEM.LSTM_ONLY,
            ) as api,
        ):
            for cell in lines:
                if cell.confidence >= 0.95 or not 4 <= len(cell.text) <= 120:
                    continue
                box = cell.rect.to_bounding_box()
                if box.height > 2.5 * line_height:
                    continue
                gray = self._crop(
                    image, box, page, inset=-3 / (image.width / page.size.width)
                )
                if gray is None:
                    continue
                numeric = number_key(cell.text) is not None
                tess = self._tess_readings(api, gray, numeric=numeric)
                if numeric:
                    text = choose_line_number(cell.text, tess)
                    selected = RegionReading(text, cell.confidence, "page", "original")
                    if text != cell.text:
                        selected = RegionReading(
                            text,
                            min(r.confidence for r in tess if r.variant == "wide"),
                            "tesseract",
                            "wide",
                        )
                else:
                    selected = self._choose_text(
                        cell.text,
                        cell.confidence,
                        [*self._rapid_readings(gray, numeric=False), *tess],
                    )
                if selected.text and selected.text != cell.text:
                    replacements[cell.index] = selected
                    cell.text = selected.text
                    cell.confidence = selected.confidence
        if replacements and page.predictions.layout is not None:

            def update(cluster: Cluster) -> None:
                for cell in cluster.cells:
                    if cell.from_ocr and cell.index in replacements:
                        reading = replacements[cell.index]
                        cell.text = reading.text
                        cell.confidence = reading.confidence
                for child in cluster.children:
                    update(child)

            for cluster in page.predictions.layout.clusters:
                update(cluster)

    def refine_tables(
        self, page: Page, elements: list[PageElement], *, image_input: bool = False
    ) -> None:
        if (
            page.parsed_page is None
            or page.size is None
            or not any(cell.from_ocr for cell in page.parsed_page.textline_cells)
        ):
            return
        tables = [element for element in elements if isinstance(element, Table)]
        if not tables:
            return
        # Render a dense scan at high resolution. This never enlarges photos
        # during line refinement, where the original camera pixels matter.
        image = page.get_image(scale=1 if image_input else 600 / 72)
        if image is None:
            return
        with (
            self._lock,
            self._tesserocr.PyTessBaseAPI(
                path=str(self.tessdata),
                lang="rus+eng",
                psm=self._tesserocr.PSM.SINGLE_LINE,
                oem=self._tesserocr.OEM.LSTM_ONLY,
            ) as api,
        ):
            for table in tables:
                column_keys: dict[int, list[str]] = defaultdict(list)
                for cell in table.table_cells:
                    key = number_key(cell.text)
                    if key is not None:
                        column_keys[cell.start_col_offset_idx].append(key)
                # A row identifier or note reference must not become a new
                # amount. Refine established amount columns, retaining short
                # values inside them as well as large totals and decimals.
                amount_columns = {
                    column
                    for column, keys in column_keys.items()
                    if len(keys) >= 3
                    and any(
                        sum(char.isdigit() for char in key) >= 5
                        or re.fullmatch(r"-?\d+\.\d{2}", key) is not None
                        for key in keys
                    )
                }
                for cell in table.table_cells:
                    if (
                        cell.bbox is None
                        or number_key(cell.text) is None
                        or cell.start_col_offset_idx not in amount_columns
                        or cell.column_header
                        or cell.row_header
                    ):
                        continue
                    box = cell.bbox.to_top_left_origin(page.size.height)
                    source_words = [
                        word
                        for word in page.parsed_page.word_cells
                        if word.rect.to_bounding_box().intersection_over_self(box) > 0.5
                    ]
                    if not source_words or any(
                        not word.from_ocr for word in source_words
                    ):
                        continue
                    # Grid cells include border rules. Neural table boxes
                    # usually hug the glyphs and need a small outer margin.
                    inset = 0.35 if table.from_ocr_grid else -0.7
                    gray = self._crop(image, cell.bbox, page, inset=inset)
                    if gray is not None:
                        keys = column_keys[cell.start_col_offset_idx]
                        cell.text = choose_number(
                            cell.text,
                            self._rapid_readings(gray, numeric=True),
                            allow_decimal_removal=(
                                len(keys) >= 5
                                and sum("." not in key for key in keys) / len(keys)
                                >= 0.9
                            ),
                        )
                        cell.text = self._refine_ambiguous_digit(api, gray, cell.text)
