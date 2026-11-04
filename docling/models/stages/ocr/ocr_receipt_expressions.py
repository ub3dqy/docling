# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Recover printed receipt expressions without deriving prices or quantities."""

from __future__ import annotations

import re
from collections import defaultdict
from statistics import mean, median
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from docling.datamodel.base_models import Page
from docling.models.stages.ocr.ocr_region_refiner import (
    RegionReading,
    normalize_mixed_script,
    number_key,
    recognizer_family,
)
from docling.utils.ocr_receipt_rows import is_receipt_field_caption, is_receipt_page

if TYPE_CHECKING:
    from docling.models.stages.ocr.ocr_region_refiner import OcrRegionRefiner

_EXPRESSION = re.compile(
    r"(?P<price>\d+[.,]\d{2})\s*[*x\u0445\u00d7]\s*"
    r"(?P<quantity>\d+(?:[.,]\d+)?)\s*"
    r"(?P<unit>\u0448\u0442|\u043a\u0433|\u0433|\u043b|\u043c|\u0443\u043f|\u043f\u0430\u0447|\u0440\u0443\u043b|\u043a\u043e\u043c\u043f\u043b|\u043c\u0435\u0441)\.?",
    re.IGNORECASE,
)

_UNITLESS = re.compile(
    r"(?P<price>\d+[.,]\d{2})\s*[*x\u0445\u00d7]\s*"
    r"(?P<quantity>\d+(?:[.,]\d+)?)",
    re.IGNORECASE,
)


def is_damaged_amount(text: str) -> bool:
    """Nominate a broken numeric token, never a complete signed amount."""
    return (
        re.fullmatch(
            r"[.,][=\-\u2212]?\s*\d{1,7}[.,]\d{2}|[= ]*\d{1,7}[.,]", text.strip()
        )
        is not None
    )


def choose_damaged_amount(
    original: str, readings: list[RegionReading], gray: np.ndarray
) -> RegionReading | None:
    """Recover missing digits only from an observed, source-marked full amount.

    The original must be structurally incomplete. Two independent families
    must agree on the complete amount; one must also read its equals sign.
    Retain the surviving digits and any complete fractional part. Require the
    actual source decimal and leading marker, using the proposed digit count
    rather than the damaged first reading's count. No neighbouring value is
    used and a strong negative reading vetoes the positive candidate.
    """
    from docling.utils.ocr_digit_templates import (
        connected_decimal_equals,
        decimal_expression_equals,
        decimal_suffix_region,
    )

    if not is_damaged_amount(original):
        return None
    raw = original.strip()
    fraction = re.search(r"[.,](\d{2})$", raw)
    integer = re.search(r"\d+", raw)
    assert integer is not None
    groups: dict[str, dict[str, RegionReading]] = defaultdict(dict)
    equals: dict[str, RegionReading] = {}
    negatives = set()
    for reading in readings:
        key = number_key(reading.text)
        if key is not None and key.startswith("-") and reading.confidence >= 0.95:
            negatives.add(key[1:])
        match = re.fullmatch(r"[= ]*(\d{1,7}[.,]\d{2})", reading.text.strip())
        if match is None or reading.confidence < 0.9:
            continue
        amount = match[1].replace(",", ".")
        whole, _, decimal = amount.partition(".")
        surviving = iter(whole)
        if (
            not len(integer[0]) <= len(whole) <= len(integer[0]) + 2
            or not all(char in surviving for char in integer[0])
            or (fraction is not None and fraction[1] != decimal)
        ):
            continue
        if reading.text.strip().startswith("="):
            equals[amount] = reading
        if reading.confidence >= 0.95:
            family = recognizer_family(reading.engine)
            group = groups[amount]
            if family not in group or group[family].confidence < reading.confidence:
                group[family] = reading
    winners = []
    for amount, group in groups.items():
        strongest = sorted(group.values(), key=lambda r: r.confidence, reverse=True)
        if (
            amount in negatives
            or amount not in equals
            or len(strongest) < 2
            or strongest[0].confidence < 0.98
            or mean(r.confidence for r in strongest[:2]) < 0.97
        ):
            continue
        digits = len(amount.partition(".")[0])
        if not (
            decimal_expression_equals(gray, digits, allow_uneven_bars=True)
            or connected_decimal_equals(gray, digits)
            or (
                opaque_amount_marker(gray)
                and decimal_suffix_region(gray, digits) is not None
            )
        ):
            continue
        observed = re.search(r"\d+[.,]\d{2}", equals[amount].text)
        assert observed is not None
        winners.append(
            RegionReading(
                "=" + observed[0],
                min(r.confidence for r in strongest[:2]),
                "damaged-amount-consensus",
                "source-quad",
            )
        )
    return winners[0] if len(winners) == 1 else None


def choose_printed_separator(
    original: str, readings: list[RegionReading], *, printed_separator: bool
) -> RegionReading | None:
    """Restore observed punctuation between unchanged digits, retaining colons.

    A time separator stays a colon. No global numeric parser is changed and
    no decimal position is inferred from a currency format or another value.
    """
    raw = original.strip()
    before = re.fullmatch(r"[= ]*(\d{4,8})", raw)
    if not printed_separator or before is None:
        return None
    groups: dict[str, dict[str, RegionReading]] = defaultdict(dict)
    for reading in readings:
        match = re.fullmatch(r"[= ]*(\d+)([.,:])(\d{2})", reading.text.strip())
        if (
            match is None
            or reading.confidence < 0.9
            or match[1] + match[3] != before[1]
        ):
            continue
        family = recognizer_family(reading.engine)
        group = groups[match[2]]
        if family not in group or group[family].confidence < reading.confidence:
            group[family] = reading
    winners = []
    for group in groups.values():
        strongest = sorted(group.values(), key=lambda r: r.confidence, reverse=True)
        if (
            len(strongest) >= 2
            and strongest[0].confidence >= 0.97
            and mean(r.confidence for r in strongest[:2]) >= 0.95
        ):
            winners.append(strongest)
    if len(winners) != 1:
        return None
    best = winners[0][0]
    match = re.fullmatch(r"[= ]*(\d+[.,:]\d{2})", best.text.strip())
    assert match is not None
    return RegionReading(
        ("=" if raw.startswith("=") else "") + match[1],
        min(r.confidence for r in winners[0][:2]),
        "source-separator-consensus",
        "literal-punctuation",
    )


def choose_unitless_expression(
    original: str, readings: list[RegionReading], *, printed_operator: bool
) -> RegionReading | None:
    """Read an observed full expression, keeping quantity, signs and scale.

    A missing operator needs its source cross, not a currency-format guess.
    An existing operator permits one observed price-digit substitution or one
    missing integer digit; complete quantities remain unchanged. No value is
    derived by multiplication or from another receipt field.
    """
    raw = original.strip()
    prefix = re.match(r"\d+[.,]\d{2}", raw)
    if (
        not printed_operator
        or prefix is None
        or re.fullmatch(r"[\d.,*x\u0445\u00d7 ]+", raw, re.IGNORECASE) is None
    ):
        return None
    before = _UNITLESS.fullmatch(raw)
    price = (before["price"] if before else prefix[0]).replace(",", ".")
    tail = raw[prefix.end() :]
    quantity = before["quantity"].replace(",", ".") if before else None
    if before is None and tail:
        if re.fullmatch(r"\d+(?:[.,]\d+)?", tail) is None:
            return None
        quantity = tail.replace(",", ".")
    groups: dict[tuple[str, str], dict[str, RegionReading]] = defaultdict(dict)
    for reading in readings:
        match = _UNITLESS.fullmatch(reading.text.strip())
        if match is None or reading.confidence < 0.9:
            continue
        proposed = match["price"].replace(",", ".")
        observed_quantity = match["quantity"].replace(",", ".")
        if quantity is not None and quantity != observed_quantity:
            continue
        same_length = (
            len(price) == len(proposed)
            and sum(a != b for a, b in zip(price, proposed)) <= 1
        )
        inserted_integer = len(proposed) == len(price) + 1 and any(
            proposed[:i] + proposed[i + 1 :] == price
            for i in range(proposed.index("."))
        )
        if not same_length and not (before is not None and inserted_integer):
            continue
        family = recognizer_family(reading.engine)
        group = groups[proposed, observed_quantity]
        if family not in group or group[family].confidence < reading.confidence:
            group[family] = reading
    winners = []
    for group in groups.values():
        strongest = sorted(group.values(), key=lambda r: r.confidence, reverse=True)
        if (
            len(strongest) >= 2
            and strongest[0].confidence >= 0.98
            and mean(r.confidence for r in strongest[:2]) >= 0.96
        ):
            winners.append(strongest)
    if len(winners) != 1:
        return None
    best = winners[0][0]
    match = _UNITLESS.fullmatch(best.text.strip())
    assert match is not None
    if before is not None and price == match["price"].replace(",", "."):
        return None
    return RegionReading(
        best.text.strip(),
        min(r.confidence for r in winners[0][:2]),
        "printed-expression-consensus",
        "source-quad",
    )


def verify_receipt_page(
    refiner: OcrRegionRefiner, page: Page, image: Image.Image
) -> bool:
    """Confirm a damaged fiscal caption before enabling receipt-specific repairs.

    A cashier caption and several observed decimal amounts nominate the page.
    Two independent recognizers must then confirm another fiscal caption from
    its own pixels. The coarse candidate spelling never becomes output text.
    """
    assert page.parsed_page is not None
    lines = [line for line in page.parsed_page.textline_cells if line.from_ocr]
    texts = [normalize_mixed_script(line.text).upper() for line in lines]
    if (
        not any("\u041a\u0410\u0421\u0421" in text for text in texts)
        or sum(
            re.fullmatch(r"[=*: ]*-?\d+[.,]\d{2}[.,]?", text.strip()) is not None
            for text in texts
        )
        < 3
    ):
        return False
    for line, text in zip(lines, texts):
        if (
            line.confidence >= 0.95
            or not 5 <= len(text) <= 60
            or re.match(r"\W*\u0421\u0423[\u0410-\u042f]{2,4}\b", text) is None
        ):
            continue
        gray = refiner._crop(image, line.rect.to_bounding_box(), page, inset=-3)
        if gray is None:
            continue
        confirmed = [
            reading
            for reading in refiner._cyrillic_readings(gray)
            if reading.confidence >= 0.8 and is_receipt_field_caption(reading.text)
        ]
        if (
            len({recognizer_family(reading.engine) for reading in confirmed}) >= 2
            and any(reading.confidence >= 0.95 for reading in confirmed)
            and is_receipt_page([*texts, *(reading.text for reading in confirmed)])
        ):
            return True
    return False


def choose_receipt_expression(
    original: str, readings: list[RegionReading], *, confidence: float
) -> str:
    """Require two recognizers to observe the same complete price expression."""
    prefix = re.match(r"\d+[.,]\d{2}", original.strip())
    if prefix is None or confidence >= 0.9:
        return original
    before = _EXPRESSION.fullmatch(original.strip())
    price = prefix.group().replace(",", ".")
    groups: dict[tuple[str, str, str], list[RegionReading]] = defaultdict(list)
    for reading in readings:
        match = _EXPRESSION.fullmatch(reading.text.strip())
        if reading.confidence < 0.95 or match is None:
            continue
        proposed = match["price"].replace(",", ".")
        if (
            len(proposed) != len(price)
            or proposed.index(".") != price.index(".")
            or sum(a != b for a, b in zip(price, proposed)) > 1
        ):
            continue
        quantity = match["quantity"].replace(",", ".")
        unit = match["unit"].casefold()
        if before is not None and (
            quantity != before["quantity"].replace(",", ".")
            or unit != before["unit"].casefold()
        ):
            continue
        groups[proposed, quantity, unit].append(reading)
    supported = [
        readings
        for readings in groups.values()
        if len({recognizer_family(reading.engine) for reading in readings}) >= 2
    ]
    if len(supported) != 1:
        return original
    best = max(
        supported[0],
        key=lambda reading: (
            reading.variant.startswith("v6-source"),
            reading.confidence,
        ),
    )
    return best.text.strip()


def compact_amount_marker(gray: np.ndarray) -> bool:
    """A merged filled separator has a taller source body than a minus stroke.

    Infer glyph size from several neighbouring digits. Only inspect a leading
    component wider than an ordinary digit; a dot, colon or ordinary minus
    cannot supply the required horizontal width and vertical ink coverage.
    This shape alone never authorizes a replacement.
    """
    import cv2

    if gray.ndim != 2 or min(gray.shape) < 6:
        return False
    height, width = gray.shape
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    _, _, stats, _ = cv2.connectedComponentsWithStats(ink)
    middle = [
        stat
        for stat in stats[1:]
        if width * 0.2 <= stat[0] <= width * 0.85
        and height * 0.15 <= stat[3] <= height * 0.6
        and stat[2] >= 3
        and stat[2] <= width * 0.25
        and stat[4] >= 12
    ]
    if len(middle) < 3:
        return False
    glyph_height = median(int(stat[3]) for stat in middle)
    glyph_width = median(int(stat[2]) for stat in middle)
    body = sorted(
        (
            stat
            for stat in stats[1:]
            if glyph_height * 0.7 <= stat[3] <= glyph_height * 1.5
            and stat[2] <= width * 0.4
            and stat[4] >= 12
        ),
        key=lambda stat: stat[0],
    )
    if len(body) < 4:
        return False
    x, y, w, h, _ = (int(value) for value in body[0])
    marker_width = round(w - glyph_width)
    if not glyph_width * 0.6 <= marker_width <= glyph_height * 1.6:
        return False
    marker = ink[y : y + h, x : x + marker_width]
    occupied = (marker > 0).sum(axis=1) >= marker_width * 0.45
    return bool(occupied.sum() >= glyph_height * 0.5)


def opaque_amount_marker(gray: np.ndarray) -> bool:
    """A filled fiscal marker has a substantial body; a minus is a thin stroke.

    Compare with neighbouring digit sizes, so scale and camera foreshortening
    do not turn an ordinary minus into a square. This only nominates a source
    marker; independent numeric readings and an explicit source equals are
    still required before changing any output.
    """
    import cv2

    if gray.ndim != 2 or min(gray.shape) < 6:
        return False
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    _, _, stats, _ = cv2.connectedComponentsWithStats(ink)
    parts = [stat for stat in stats[1:] if stat[4] >= 3 and stat[3] >= 3]
    if len(parts) < 4:
        return False
    maximum = max(int(stat[3]) for stat in parts)
    body = [stat for stat in parts if stat[3] >= maximum * 0.6]
    height = median(int(stat[3]) for stat in body)
    digits = [stat for stat in body if stat[3] >= height * 0.85]
    if len(digits) < 3:
        return False
    width = median(int(stat[2]) for stat in digits)
    first_digit = min(int(stat[0]) for stat in digits)
    return any(
        0 <= first_digit - (stat[0] + stat[2]) <= height
        and width * 0.3 <= stat[2] <= width * 1.4
        and height * 0.3 <= stat[3] <= height * 0.8
        and stat[3] >= stat[2] * 0.6
        and stat[4] >= stat[2] * stat[3] * 0.75
        for stat in parts
    )


def restore_receipt_separator(
    original: str, readings: list[RegionReading], gray: np.ndarray
) -> str:
    """Disambiguate a merged fiscal separator while keeping every amount digit."""
    if re.fullmatch(r"-\d+[.,]\d{2}", original.strip()) is None:
        return original
    key = number_key(original)
    if key is None:
        return original
    amount = key[1:]
    if any(
        reading.confidence >= 0.95 and number_key(reading.text) == "-" + amount
        for reading in readings
    ):
        return original
    equals = [
        reading
        for reading in readings
        if reading.confidence >= 0.98
        and reading.text.strip().startswith("=")
        and number_key(reading.text) == amount
    ]
    positive_families = {
        recognizer_family(reading.engine)
        for reading in readings
        if reading.confidence >= 0.95 and number_key(reading.text) == amount
    }
    if (
        len(positive_families) >= 2
        and any(reading.variant == "source" for reading in equals)
        and opaque_amount_marker(gray)
    ):
        return "=" + original.strip()[1:]
    views = {
        reading.variant.split("-")[1]
        for reading in equals
        if reading.variant.startswith("v6-")
    }
    if len(views) < 2 or len(positive_families) < 2 or not compact_amount_marker(gray):
        return original
    return "=" + original.strip()[1:]


def refine_receipt_expressions(
    refiner: OcrRegionRefiner, page: Page, image: Image.Image
) -> set[int]:
    assert page.parsed_page is not None
    changed = set()
    for line in page.parsed_page.textline_cells:
        negative = re.fullmatch(r"-\d+[.,]\d{2}", line.text.strip()) is not None
        expression = re.search(r"\d+[.,]\d{2}", line.text) is not None
        if not line.from_ocr or line.confidence >= 0.95 or not (negative or expression):
            continue
        if not negative and (
            line.confidence >= 0.9
            or re.fullmatch(r"[=*: ]*\d+[.,]\d{2}[.,]?", line.text.strip()) is not None
        ):
            continue
        bounds = line.rect.to_bounding_box()
        words = [
            word
            for word in page.parsed_page.word_cells
            if word.rect.to_bounding_box().intersection_over_self(bounds) >= 0.8
        ]
        if not words or any(not word.from_ocr for word in words):
            continue
        gray = refiner._crop(image, bounds, page, inset=-min(bounds.height * 0.35, 6))
        if gray is None:
            continue
        readings = refiner._cyrillic_readings(gray)
        if negative:
            readings.extend(refiner._rapid_readings(gray, numeric=True))
            replacement = restore_receipt_separator(line.text, readings, gray)
        else:
            replacement = choose_receipt_expression(
                line.text, readings, confidence=line.confidence
            )
        if replacement != line.text:
            line.text = replacement
            # The previous word split can turn a printed unit into extra
            # decimal digits. Replace that complete observed expression once.
            merged = line.model_copy(
                deep=True, update={"index": min(word.index for word in words)}
            )
            removed = {word.index for word in words}
            page.parsed_page.word_cells = [
                word
                for word in page.parsed_page.word_cells
                if word.index not in removed
            ] + [merged]
            changed.add(line.index)
    return changed
