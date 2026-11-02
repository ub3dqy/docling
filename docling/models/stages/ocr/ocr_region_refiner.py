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
from docling_core.types.doc import BoundingBox, DocItemLabel
from PIL import Image, ImageOps

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import Cluster, Page, PageElement, Table
from docling.datamodel.pipeline_options import RapidOcrOptions
from docling.datamodel.settings import settings
from docling.models.stages.ocr.rapid_ocr_model import RapidOcrModel
from docling.utils.ocr_receipt_rows import (
    is_receipt_field_caption,
    is_receipt_page,
    join_receipt_fields,
)
from docling.utils.ocr_region_pixels import (
    clean_region,
    light_on_dark,
    photo_contrast,
    printed_dash,
    trim_light_text,
    two_bar_marker,
)

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


@dataclass(frozen=True)
class SymbolReading:
    text: str
    confidence: float
    bbox: tuple[int, int, int, int]


def recognizer_family(engine: str) -> str:
    if engine in {"page", "iso:ru"}:
        return "eslav"
    if engine.startswith("tesseract"):
        return "tesseract"
    return engine


def supports_photo_label(
    original: str,
    reading: RegionReading,
    readings: list[RegionReading],
    known: Callable[[str], bool],
) -> bool:
    """Permit observed Cyrillic letters in a damaged short table label."""
    text = normalize_mixed_script(reading.text).strip()
    source_label = re.sub(r"^\d{1,3}[.)]?\s*", "", original).strip()
    candidate_label = re.sub(r"^\d{1,3}[.)]?\s*", "", text).strip()
    candidate_words = _WORDS.findall(candidate_label)
    source_ordinal = re.match(r"^\d{1,3}[.)]?", original)
    candidate_ordinal = re.match(r"^\d{1,3}[.)]?", text)
    complete_label = (
        sum(char.isalpha() for char in source_label) >= 1
        and (source_ordinal is None) == (candidate_ordinal is None)
        and (
            not any(char.isdigit() for char in source_label)
            or (source_label.isalnum() and len(source_label) <= 3)
        )
        and candidate_words
        and all(known(word) for word in candidate_words)
        and not any(char.isdigit() for char in candidate_label)
        and re.search(r"[A-Za-z]", candidate_label) is None
        and len(candidate_label) <= 4 * max(2, len(source_label))
    )
    if not complete_label and (
        not original.isalnum()
        or not any(char.isdigit() for char in original)
        or (original.isdigit() and len(original) > 3)
        or not text.isalpha()
        or not known(text)
        or not 0.65 * len(original) <= len(text) <= 1.5 * len(original)
    ):
        return False
    families: dict[str, float] = {}
    for candidate in readings:
        if (
            candidate.confidence >= 0.55
            and normalize_mixed_script(candidate.text).strip().casefold()
            == text.casefold()
        ):
            family = recognizer_family(candidate.engine)
            families[family] = max(families.get(family, 0.0), candidate.confidence)
    return len(families) >= 2 and mean(families.values()) >= (
        0.55 if complete_label else 0.6
    )


def _significant_punctuation(text: str) -> tuple[tuple[int, str], ...]:
    # Decimal marks are checked separately by compatible_text_reading.
    # Keep identifiers and every operator, percent sign and bracket significant.
    result: list[tuple[int, str]] = []
    latin = re.search(r"[A-Za-z]", text) is not None
    content_offset = 0
    for index, char in enumerate(text):
        if char.isalnum():
            content_offset += 1
            continue
        if char.isspace():
            continue
        if char == ".":
            word = re.search(r"([^\W\d_]+)$", text[:index])
            # Sentence endings and abbreviations remain significant. The
            # caption vote can reject only an isolated internal noise dot.
            if (
                index == len(text.rstrip()) - 1
                or latin
                or (word is not None and len(word.group()) >= 2)
            ):
                result.append((content_offset, char))
            continue
        following_word = _WORDS.match(text, index + 1) if char == "_" else None
        if (
            char == "_"
            and not latin
            and (following_word is None or len(following_word.group()) <= 1)
            and not (
                index > 0
                and index + 1 < len(text)
                and text[index - 1].isalnum()
                and text[index + 1].isalnum()
            )
        ):
            continue
        result.append((content_offset, char))
    return tuple(result)


def choose_text_punctuation(
    original: RegionReading, readings: list[RegionReading]
) -> RegionReading:
    """Change punctuation only when independent readings agree on the same text."""
    if (
        original.confidence >= 0.95
        or len(_WORDS.findall(original.text)) < 3
        or not "".join(char for char in original.text if char.isalpha()).isupper()
    ):
        return original
    content = "".join(char for char in original.text if char.isalnum())
    punctuation = _significant_punctuation(original.text)
    groups: dict[str, dict[str, RegionReading]] = defaultdict(dict)
    for reading in readings:
        text = " ".join(reading.text.split())
        if (
            reading.confidence < 0.7
            or "".join(char for char in text if char.isalnum()) != content
            or _significant_punctuation(text) != punctuation
            or not compatible_text_reading(original.text, text)
        ):
            continue
        family = recognizer_family(reading.engine)
        previous = groups[text].get(family)
        if previous is None or reading.confidence > previous.confidence:
            groups[text][family] = reading
    supported = [(text, group) for text, group in groups.items() if len(group) >= 2]
    if not supported:
        return original
    text, winner = max(
        supported,
        key=lambda item: (
            len(item[1]),
            item[0] == " ".join(original.text.split()),
            mean(reading.confidence for reading in item[1].values()),
        ),
    )
    return RegionReading(
        text,
        min(r.confidence for r in winner.values()),
        "punctuation-consensus",
        "same-content",
    )


def choose_amount_punctuation(
    original: RegionReading, readings: list[RegionReading], *, leading_equals: bool
) -> RegionReading:
    """Restore observed field markers without changing an amount's digits or scale."""
    key = number_key(original.text)
    if key is None or re.fullmatch(r"-?\d+\.\d{2}", key) is None:
        return original
    text = original.text
    if text.endswith("."):
        families = {
            recognizer_family(reading.engine)
            for reading in readings
            if reading.confidence >= 0.8
            and number_key(reading.text) == key
            and re.fullmatch(r"[=*: ]*\(?-?\d[\d ]*[.,]\d{2}\)?", reading.text.strip())
            is not None
        }
        if len(families) >= 2:
            text = text[:-1]
    if leading_equals and not key.startswith("-"):
        text = "=" + text.lstrip("=*: ")
    return RegionReading(text, original.confidence, original.engine, original.variant)


def supports_leading_equals(
    symbols: list[SymbolReading],
    text: str,
    amount: str,
    *,
    min_confidence: float = 0.95,
) -> bool:
    """Verify an actual leading glyph, separately from uncertain numeric glyphs."""
    key = number_key(text)
    if (
        not text.startswith("=")
        or key is None
        or len(symbols) < 2
        or len(key) != len(amount)
        or key.startswith("-")
        or amount.startswith("-")
        or re.fullmatch(r"\d+\.\d{2}", key) is None
        or sum(a != b for a, b in zip(key, amount)) > 1
    ):
        return False
    marker, digit = symbols[:2]
    left, top, right, bottom = marker.bbox
    dl, dt, dr, db = digit.bbox
    height = db - dt
    return (
        marker.text == "="
        and marker.confidence >= min_confidence
        and len(digit.text) == 1
        and digit.text.isdigit()
        and height > 0
        and right > left
        and bottom > top
        and right - left <= 1.5 * height
        and bottom - top <= 1.4 * height
        and (left + right) / 2 + 0.25 * height <= (dl + dr) / 2
        and dt - 0.25 * height <= (top + bottom) / 2 <= db + 0.25 * height
    )


def punctuation_line_crop(gray: np.ndarray) -> np.ndarray:
    """Retain leading marks while excluding neighboring lines and distant speckles."""
    import cv2

    height, width = gray.shape
    if height < 4 or width < 4:
        return gray
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    _, _, stats, _ = cv2.connectedComponentsWithStats(
        np.asarray(ink, dtype=np.uint8), connectivity=8
    )
    body = [
        stat
        for stat in stats[1:]
        if stat[3] >= height * 0.25 and stat[4] >= 8 and stat[2] < width * 0.9
    ]
    if not body:
        return gray
    top = min(int(stat[1]) for stat in body)
    bottom = max(int(stat[1] + stat[3]) for stat in body)
    left = max(0, min(int(stat[0]) for stat in body) - round((bottom - top) * 1.5))
    right = min(width, max(int(stat[0] + stat[2]) for stat in body) + 2)
    line = gray[top:bottom, left:right]
    # Round-to-even page coordinates can change the surrounding white margin
    # by one pixel. Normalize that margin without trimming the printed marker.
    _, line_ink = cv2.threshold(line, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    _, _, components, _ = cv2.connectedComponentsWithStats(
        np.asarray(line_ink, dtype=np.uint8), connectivity=8
    )
    useful = [component for component in components[1:] if component[4] >= 2]
    if not useful:
        return line
    first_ink = min(int(component[0]) for component in useful)
    line = line[:, max(0, first_ink - 2) :]
    return cv2.copyMakeBorder(
        line, 0, 0, round((bottom - top) * 0.5), 0, cv2.BORDER_CONSTANT, value=255
    )


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


def choose_line_number(
    original: str, readings: list[RegionReading], *, confidence: float = 0.0
) -> str:
    """Keep a receipt amount unless two segmentation modes verify one digit."""
    original_key = number_key(original)
    if original_key is None or confidence >= 0.95:
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
    prefix = original[:1] if original[:1] in {"=", "*", ":"} else ""
    return prefix + wide[0].text.strip("=*: ")


def choose_receipt_decimal(
    original: str, readings: list[RegionReading], *, printed_dot: bool
) -> str:
    """Restore an observed separator without replacing any original digit."""
    key = number_key(original)
    if not printed_dot or key is None or re.fullmatch(r"\d{4,8}", key) is None:
        return original
    rapid_support = False
    strong_rapid: dict[str, set[str]] = defaultdict(set)
    tess_modes: set[str] = set()
    for reading in readings:
        candidate = number_key(reading.text)
        if candidate is None or re.fullmatch(r"\d+\.\d{2}", candidate) is None:
            continue
        digits = candidate.replace(".", "")
        if len(digits) != len(key) or reading.confidence < 0.78:
            continue
        if reading.engine.startswith("tesseract"):
            if sum(a != b for a, b in zip(digits, key)) <= 1:
                tess_modes.add(reading.engine)
        elif digits == key:
            rapid_support = True
            if reading.confidence >= 0.9:
                strong_rapid[recognizer_family(reading.engine)].add(reading.variant)
    independent_rapid = (
        sum(len(variants) >= 2 for variants in strong_rapid.values()) >= 2
    )
    if not independent_rapid and (not rapid_support or len(tess_modes) < 2):
        return original
    prefix = original[:1] if original[:1] in {"=", "*", ":"} else ""
    return prefix + key[:-2] + "." + key[-2:]


def printed_decimal(gray: np.ndarray, digits: int) -> bool:
    """Require an isolated baseline dot between the actual last three digits."""
    import cv2

    gray = clean_region(gray)
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    _, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    components = [stat for stat in stats[1:] if stat[4] >= 2]
    if not components:
        return False
    height = max(int(stat[3]) for stat in components)
    body = sorted(
        (stat for stat in components if stat[3] >= height * 0.5),
        key=lambda stat: stat[0],
    )
    if height < 6 or len(body) not in {digits, digits - 1}:
        return False
    widths = [int(stat[2]) for stat in body]
    pairs = [
        index for index, width in enumerate(widths) if width > 1.6 * median(widths)
    ]
    if len(body) == digits - 1 and len(pairs) != 1:
        return False
    count = 0
    boundary = None
    for index, stat in enumerate(body[:-1]):
        count += 2 if len(body) == digits - 1 and index in pairs else 1
        if count == digits - 2:
            boundary = index
            break
    if boundary is None:
        return False
    before, after = body[boundary], body[boundary + 1]
    baseline = median(int(stat[1] + stat[3]) for stat in body)
    return any(
        stat[3] <= height * 0.35
        and stat[2] <= height * 0.4
        and before[0] + before[2] <= stat[0] + stat[2] / 2 <= after[0]
        and baseline - height * 0.3 <= stat[1] + stat[3] <= baseline + height * 0.2
        for stat in components
    )


def choose_short_code(original: str, readings: list[RegionReading]) -> str:
    """Read a three-digit code literally; never pad an incomplete reading."""
    if sum(char.isdigit() for char in original) > 3:
        return original
    groups: dict[str, set[str]] = defaultdict(set)
    for reading in readings:
        text = reading.text.strip()
        if reading.confidence >= 0.85 and re.fullmatch(r"\d{3}", text):
            groups[text].add(recognizer_family(reading.engine))
    winners = [text for text, engines in groups.items() if len(engines) >= 2]
    return winners[0] if len(winners) == 1 else original


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
        from docling.models.stages.ocr.ocr_cyrillic_text import create_cyrillic_readers

        self._cyrillic_readers = create_cyrillic_readers(
            root, num_threads=accelerator_options.num_threads
        )

    def _cyrillic_readings(self, gray: np.ndarray) -> list[RegionReading]:
        return [
            RegionReading(path.text, path.confidence, f"cyrillic-v6-{name}", variant)
            for name, reader in self._cyrillic_readers.items()
            for variant, path in reader.read_region(
                gray, known=self._dictionary.word_is_known
            )
        ]

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
        if not numeric:
            api.SetPageSegMode(self._tesserocr.PSM.RAW_LINE)
            for name, array in (
                ("punctuation-clean", cleaned),
                ("punctuation-wide", wide),
            ):
                api.SetImage(ImageOps.expand(Image.fromarray(array), 2, fill=255))
                readings.append(
                    RegionReading(
                        api.GetUTF8Text().strip(),
                        api.MeanTextConf() / 100,
                        "tesseract-13",
                        name,
                    )
                )
        return readings

    def _leading_equals(
        self,
        api: PyTessBaseAPI,
        gray: np.ndarray,
        amount: str,
        *,
        require_two_bars: bool = False,
    ) -> bool:
        import cv2

        if re.fullmatch(r"\d+\.\d{2}", amount) is None:
            return False
        line = punctuation_line_crop(gray)
        api.SetVariable("tessedit_char_whitelist", "0123456789.,()=+-*:\u2022")
        api.SetPageSegMode(self._tesserocr.PSM.SINGLE_LINE)
        votes = 0
        exact = False
        for factor in (1.0, 1.5):
            array = cv2.resize(
                line, None, fx=factor, fy=1, interpolation=cv2.INTER_CUBIC
            )
            for padding in (6, 12):
                sample = ImageOps.expand(Image.fromarray(array), padding, fill=255)
                api.SetImage(sample)
                text = api.GetUTF8Text().strip()
                if not text:
                    continue
                iterator = api.GetIterator()
                symbols: list[SymbolReading] = []
                for item in self._tesserocr.iterate_level(
                    iterator, self._tesserocr.RIL.SYMBOL
                ):
                    box = item.BoundingBox(self._tesserocr.RIL.SYMBOL)
                    if box is None:
                        break
                    symbols.append(
                        SymbolReading(
                            item.GetUTF8Text(self._tesserocr.RIL.SYMBOL),
                            item.Confidence(self._tesserocr.RIL.SYMBOL) / 100,
                            box,
                        )
                    )
                    if len(symbols) == 2:
                        break
                if not supports_leading_equals(symbols, text, amount):
                    continue
                left, top, right, bottom = symbols[0].bbox
                ink = np.asarray(sample)[top:bottom, left:right]
                if (
                    ink.size == 0
                    or np.mean(ink < 128) < 0.1
                    or (require_two_bars and not two_bar_marker(ink))
                ):
                    continue
                votes += 1
                exact = exact or number_key(text) == amount
        return votes >= 2 and exact

    def _receipt_equals(self, api: PyTessBaseAPI, gray: np.ndarray, text: str) -> str:
        """Correct a minus only when two actual bars are observed before that value."""
        import cv2

        observed_originals = [
            match.group()[1:].replace(",", ".")
            for match in re.finditer(
                r"(?<![^\W\d])(?<![.,])-\d+[.,]\d{2}(?![\d.,])", text
            )
        ]
        amounts = {
            amount
            for amount in observed_originals
            if observed_originals.count(amount) == 1
        }
        if not amounts:
            return text
        gray = punctuation_line_crop(gray)
        api.SetVariable("tessedit_char_whitelist", "0123456789.,()=+-*:")
        api.SetPageSegMode(self._tesserocr.PSM.SINGLE_LINE)
        votes: dict[str, int] = defaultdict(int)
        exact: set[str] = set()
        for factor in (1.0, 1.5):
            array = cv2.resize(
                gray, None, fx=factor, fy=1, interpolation=cv2.INTER_CUBIC
            )
            for padding in (6, 12):
                sample = ImageOps.expand(Image.fromarray(array), padding, fill=255)
                api.SetImage(sample)
                if not api.GetUTF8Text().strip():
                    continue
                symbols = []
                for item in self._tesserocr.iterate_level(
                    api.GetIterator(), self._tesserocr.RIL.SYMBOL
                ):
                    box = item.BoundingBox(self._tesserocr.RIL.SYMBOL)
                    if box is None:
                        break
                    symbols.append(
                        SymbolReading(
                            item.GetUTF8Text(self._tesserocr.RIL.SYMBOL),
                            item.Confidence(self._tesserocr.RIL.SYMBOL) / 100,
                            box,
                        )
                    )
                joined = "".join(symbol.text for symbol in symbols)
                if len(joined) != len(symbols):
                    continue
                supported = set()
                # Only the operator is replaced. A trailing OCR fragment or
                # one uncertain numeric glyph cannot supply new amount digits.
                for match in re.finditer(r"=(\d+[.,]\d{2})", joined):
                    observed = match.group(1).replace(",", ".")
                    candidates = [
                        amount
                        for amount in amounts
                        if supports_leading_equals(
                            symbols[match.start() : match.start() + 2],
                            match.group(),
                            amount,
                            min_confidence=0.94,
                        )
                    ]
                    if observed in candidates:
                        candidates = [observed]
                    if len(candidates) != 1:
                        continue
                    amount = candidates[0]
                    left, top, right, bottom = symbols[match.start()].bbox
                    if two_bar_marker(np.asarray(sample)[top:bottom, left:right]):
                        supported.add(amount)
                        if observed == amount:
                            exact.add(amount)
                for amount in supported:
                    votes[amount] += 1
        return re.sub(
            r"(?<![^\W\d])(?<![.,])-\d+[.,]\d{2}(?![\d.,])",
            lambda match: (
                "=" + match.group()[1:]
                if votes[match.group()[1:].replace(",", ".")] >= 2
                and match.group()[1:].replace(",", ".") in exact
                else match.group()
            ),
            text,
        )

    def _choose_text(
        self,
        original: str,
        confidence: float,
        readings: list[RegionReading],
        *,
        extended_photo_word: bool = False,
        photo_label: bool = False,
    ) -> RegionReading:
        candidates = [
            RegionReading(original, confidence, "page", "original"),
            *readings,
        ]

        def score(reading: RegionReading) -> float:
            text = normalize_mixed_script(reading.text)
            words = _WORDS.findall(text)
            verified_label = photo_label and supports_photo_label(
                original, reading, readings, self._dictionary.word_is_known
            )
            minimum = (
                0.55
                if reading.variant in {"light-character-path", "photo-character-path"}
                or verified_label
                else 0.65
            )
            if not words or (reading.engine != "page" and reading.confidence < minimum):
                return -1
            extended = (
                extended_photo_word
                and reading.variant == "light-character-path"
                and original.isalpha()
                and text.isalpha()
                and self._dictionary.word_is_known(text)
                and len(original) <= len(text) <= 3 * len(original)
                and SequenceMatcher(
                    None, original.translate(_CONFUSABLES), text
                ).ratio()
                >= 0.5
            )
            if (
                reading.engine != "page"
                and not compatible_text_reading(original, text)
                and not (extended or verified_label)
            ):
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
        chosen = refine_word_choices(chosen, candidates, self._dictionary.word_is_known)
        punctuation_candidates = []
        for reading in candidates:
            normalized = RegionReading(
                normalize_mixed_script(reading.text),
                reading.confidence,
                reading.engine,
                reading.variant,
            )
            refined = refine_word_choices(
                normalized, candidates, self._dictionary.word_is_known
            )
            punctuation_candidates.append(
                RegionReading(
                    refined.text, refined.confidence, reading.engine, reading.variant
                )
            )
        return choose_text_punctuation(chosen, punctuation_candidates)

    def _photo_readings(
        self, api: PyTessBaseAPI, gray: np.ndarray, *, table_label: bool = False
    ) -> list[RegionReading]:
        """Read photo-specific contrast variants; never supply replacement text."""
        import cv2

        if light_on_dark(gray):
            gray = trim_light_text(gray)
            # Light, broken strokes on dark textured material need the
            # opposite polarity. Smooth the texture before joining the ink.
            blurred = cv2.GaussianBlur(gray, (0, 0), 1.4)
            variants = [255 - gray, np.where(blurred > 150, 0, 255).astype(np.uint8)]
            from docling.utils.ocr_character_paths import image_character_paths

            ink = (gray > 180).astype(np.uint8) * 255
            closed = 255 - cv2.morphologyEx(
                ink, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8)
            )
            sample = np.asarray(
                ImageOps.expand(
                    Image.fromarray(clean_region(closed)).convert("RGB"),
                    6,
                    fill="white",
                )
            )
            paths = image_character_paths(self._readers["cyrillic"], sample)
            alternatives = [
                RegionReading(
                    path.text, path.confidence, "cyrillic", "light-character-path"
                )
                for path in paths
                if path.text.isalpha()
                and len(path.text) >= 5
                and self._dictionary.word_is_known(path.text)
                and path.log_probability >= paths[0].log_probability - 2.5
            ]
        else:
            alternatives = []
            kernel = max(3, min(31, gray.shape[0] // 2) | 1)
            background = cv2.morphologyEx(
                gray, cv2.MORPH_CLOSE, np.ones((kernel, kernel), np.uint8)
            )
            variants = [cv2.divide(gray, np.maximum(background, 1), scale=255)]
            if table_label:
                from docling.utils.ocr_character_paths import image_character_paths

                clean = clean_region(variants[0])
                wide = cv2.resize(
                    clean, None, fx=1.5, fy=1, interpolation=cv2.INTER_CUBIC
                )
                sample = np.asarray(
                    ImageOps.expand(
                        Image.fromarray(wide).convert("RGB"), 6, fill="white"
                    )
                )
                groups: dict[str, list[RegionReading]] = defaultdict(list)
                for lang in ("iso:ru", "cyrillic"):
                    paths = image_character_paths(self._readers[lang], sample)
                    for path in paths:
                        text = normalize_mixed_script(path.text).strip()
                        words = _WORDS.findall(text)
                        if (
                            words
                            and all(
                                self._dictionary.word_is_known(word) for word in words
                            )
                            and path.log_probability >= paths[0].log_probability - 2.5
                            and path.confidence >= 0.55
                        ):
                            groups[" ".join(text.casefold().split())].append(
                                RegionReading(
                                    text, path.confidence, lang, "photo-character-path"
                                )
                            )
                alternatives = [
                    reading
                    for group in groups.values()
                    if len({recognizer_family(reading.engine) for reading in group})
                    >= 2
                    for reading in group
                ]

        return alternatives + [
            reading
            for variant in variants
            for reading in (
                *self._rapid_readings(variant, numeric=False),
                *self._tess_readings(api, variant, numeric=False),
            )
        ]

    def _receipt_prices(
        self, page: Page, image: Image.Image, api: PyTessBaseAPI
    ) -> set[int]:
        from docling.models.stages.ocr.ocr_receipt_regions import refine_receipt_prices

        return refine_receipt_prices(self, page, image, api)

    def refine_lines(self, page: Page, *, image_input: bool) -> None:
        if not image_input or page.parsed_page is None or page.size is None:
            return
        lines = [cell for cell in page.parsed_page.textline_cells if cell.from_ocr]
        if not lines:
            return
        heights = [cell.rect.to_bounding_box().height for cell in lines]
        line_height = median(heights)
        # Short image text must not lose another generation of camera pixels.
        # Keep the established scale for receipts and other larger type.
        # High-DPI image metadata can make the point-sized page much smaller
        # than its camera pixels. Tiny text needs more than the 144-DPI view.
        scale = 4 if line_height < 6 else 2 if line_height < 12 else 1
        scale = min(scale, 6144 / max(page.size.width, page.size.height))
        image = page.get_image(scale=scale)
        if image is None:
            return
        page_text = " ".join(
            normalize_mixed_script(cell.text).casefold() for cell in lines
        )
        receipt = (
            sum(word in page_text for word in ("итог", "оплат", "ндс", "сумм", "касс"))
            >= 2
        ) or is_receipt_page(cell.text for cell in lines)
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
            if not receipt:
                from docling.models.stages.ocr.ocr_receipt_expressions import (
                    verify_receipt_page,
                )

                receipt = verify_receipt_page(self, page, image)
            if receipt:
                from docling.models.stages.ocr.ocr_receipt_expressions import (
                    refine_receipt_expressions,
                )
                from docling.models.stages.ocr.ocr_receipt_words import (
                    refine_receipt_words,
                )

                for index in (
                    refine_receipt_expressions(self, page, image)
                    | self._receipt_prices(page, image, api)
                    | refine_receipt_words(self, page, image, api)
                ):
                    cell = next(cell for cell in lines if cell.index == index)
                    replacements[index] = RegionReading(
                        cell.text, cell.confidence, "page", "verified-word"
                    )
            else:
                from docling.models.stages.ocr.ocr_numeric_regions import (
                    refine_numeric_words,
                )

                for index in refine_numeric_words(self, page, image):
                    cell = next(cell for cell in lines if cell.index == index)
                    replacements[index] = RegionReading(
                        cell.text, cell.confidence, "page", "verified-number"
                    )
            for cell in lines:
                key = number_key(cell.text)
                receipt_field = receipt and (
                    is_receipt_field_caption(cell.text)
                    or (
                        key is not None
                        and re.fullmatch(r"\d{4,8}|\d+\.\d{2}", key) is not None
                    )
                )
                if (cell.confidence >= 0.95 and not receipt_field) or not 4 <= len(
                    cell.text
                ) <= 120:
                    continue
                box = cell.rect.to_bounding_box()
                if box.height > 2.5 * line_height:
                    continue
                if (
                    line_height < 12
                    and page.predictions.layout is not None
                    and any(
                        cluster.label == DocItemLabel.TABLE
                        and box.intersection_over_self(cluster.bbox) >= 0.8
                        for cluster in page.predictions.layout.clusters
                    )
                ):
                    # Read small table text from complete cell crops below.
                    # Repeating it here doubles the cost and uses clipped boxes.
                    continue
                numeric = number_key(cell.text) is not None
                pixel_scale = image.width / page.size.width
                if (
                    not numeric
                    and re.search(r"\d+[.,]\d{2}", cell.text) is None
                    and (not receipt or _CYRILLIC.search(cell.text) is None)
                ):
                    extra = min(9 / pixel_scale, box.height * 0.35)
                    box = box.model_copy(
                        update={"l": box.l - extra, "r": box.r + extra}
                    )
                gray = self._crop(image, box, page, inset=-3 / pixel_scale)
                if gray is None:
                    continue
                photo = (
                    not numeric
                    and not receipt
                    and (box.height < 12 or light_on_dark(gray))
                )
                tess = [] if photo else self._tess_readings(api, gray, numeric=numeric)
                if numeric:
                    text = choose_line_number(
                        cell.text, tess, confidence=cell.confidence
                    )
                    selected = RegionReading(text, cell.confidence, "page", "original")
                    if text != cell.text:
                        selected = RegionReading(
                            text,
                            min(r.confidence for r in tess if r.variant == "wide"),
                            "tesseract",
                            "wide",
                        )
                    key = number_key(selected.text)
                    if receipt and key is not None and re.fullmatch(r"\d{4,8}", key):
                        selected = RegionReading(
                            choose_receipt_decimal(
                                selected.text,
                                [*tess, *self._rapid_readings(gray, numeric=True)],
                                printed_dot=printed_decimal(gray, len(key)),
                            ),
                            selected.confidence,
                            selected.engine,
                            selected.variant,
                        )
                        key = number_key(selected.text)
                    if key is not None and re.fullmatch(r"\d+\.\d{2}", key):
                        punctuation_readings = (
                            self._rapid_readings(gray, numeric=True)
                            if selected.text.endswith(".")
                            else []
                        )
                        leading_equals = not selected.text.startswith(
                            "="
                        ) and self._leading_equals(api, gray, key)
                        if (
                            receipt
                            and not selected.text.startswith("=")
                            and not leading_equals
                        ):
                            wider = box.model_copy(
                                update={
                                    "l": box.l - min(box.height * 0.9, 28 / pixel_scale)
                                }
                            )
                            extended = self._crop(
                                image, wider, page, inset=-3 / pixel_scale
                            )
                            leading_equals = (
                                extended is not None
                                and self._leading_equals(
                                    api, extended, key, require_two_bars=True
                                )
                            )
                        selected = choose_amount_punctuation(
                            selected,
                            [*tess, *punctuation_readings],
                            leading_equals=leading_equals,
                        )
                else:
                    photo_readings = self._photo_readings(api, gray) if photo else []
                    selected = self._choose_text(
                        cell.text,
                        cell.confidence,
                        [
                            *self._rapid_readings(gray, numeric=False),
                            *tess,
                            *photo_readings,
                        ],
                        extended_photo_word=not receipt and light_on_dark(gray),
                    )
                    if receipt:
                        selected = RegionReading(
                            self._receipt_equals(api, gray, selected.text),
                            selected.confidence,
                            selected.engine,
                            selected.variant,
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
        if receipt:
            from docling.models.stages.ocr.ocr_receipt_fields import (
                refine_receipt_fields,
            )

            with self._lock:
                refine_receipt_fields(self, page, image)
                join_receipt_fields(page)
                refine_receipt_fields(self, page, image)

    def refine_tables(
        self, page: Page, elements: list[PageElement], *, image_input: bool = False
    ) -> None:
        from docling.models.stages.ocr.ocr_table_regions import refine_table_regions

        refine_table_regions(self, page, elements, image_input=image_input)
