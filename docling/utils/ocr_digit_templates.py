# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Check a disputed digit against repeated glyphs printed in the same amount."""

from itertools import pairwise

import numpy as np


def repeated_digit_support(gray: np.ndarray, original: str, candidate: str) -> bool:
    """Require two source exemplars and reject ambiguous character segmentation.

    This is additional pixel evidence for an observed recognizer alternative,
    never a way to calculate or supply new amount digits.
    """
    import cv2

    if (
        not original.isdigit()
        or not candidate.isdigit()
        or len(original) != len(candidate)
    ):
        return False
    differences = [
        i for i, pair in enumerate(zip(original, candidate)) if pair[0] != pair[1]
    ]
    if len(differences) != 1:
        return False
    index = differences[0]
    exemplars = [
        i
        for i, digit in enumerate(original)
        if digit == candidate[index] and i != index
    ]
    if len(exemplars) < 2:
        return False
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    _, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    body = sorted(
        (int(x), int(y), int(w), int(h))
        for x, y, w, h, _ in stats[1:]
        if h > gray.shape[0] * 0.5 and w > 2
    )
    if not body:
        return False
    pitch = sum(w for _, _, w, _ in body) / len(original)
    counts = [round(w / pitch) for _, _, w, _ in body]
    if sum(counts) != len(original) or any(
        n < 1 or n > 3 or abs(w / pitch - n) > 0.25
        for (_, _, w, _), n in zip(body, counts)
    ):
        return False
    glyphs = []
    for (x, y, w, h), count in zip(body, counts):
        for position in range(count):
            left = max(0, round(x + w * position / count) - 1)
            right = min(gray.shape[1], round(x + w * (position + 1) / count) + 1)
            glyph = ink[max(0, y - 1) : y + h + 1, left:right]
            ys, xs = np.nonzero(glyph)
            if len(xs) == 0:
                return False
            glyphs.append(
                cv2.resize(
                    glyph[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1], (24, 48)
                ).astype(np.float32)
            )
    scores = []
    for glyph in glyphs:
        padded = cv2.copyMakeBorder(glyph, 4, 4, 3, 3, cv2.BORDER_CONSTANT, value=0)
        scores.append(
            float(cv2.matchTemplate(padded, glyphs[index], cv2.TM_CCOEFF_NORMED).max())
        )
    matches = [scores[i] for i in exemplars]
    others = [
        score for i, score in enumerate(scores) if i != index and i not in exemplars
    ]
    # Similar outlines are insufficient if a required vertical stem is absent.
    # For example, a real 3 can correlate with 8 or 9 despite its open left side.
    for left, right in ((0, 9), (15, 24)):
        for top, bottom in ((9, 22), (26, 39)):
            strokes = [
                float((glyphs[i][top:bottom, left:right] > 127).any(axis=1).mean())
                for i in exemplars
            ]
            observed = float(
                (glyphs[index][top:bottom, left:right] > 127).any(axis=1).mean()
            )
            if min(strokes) >= 0.8 and observed < 0.45:
                return False
    return (
        min(matches) >= 0.65
        and sum(matches) / len(matches) >= 0.7
        and bool(others)
        and min(matches) - max(others) >= 0.15
    )


def _decimal_suffix_bounds(
    gray: np.ndarray, integer_digits: int, fraction_digits: int = 2
) -> tuple[int, int, int, int] | None:
    """Locate a complete numeric suffix from digit bodies and its real decimal dot."""
    import cv2

    # Thermal print often connects adjacent digits. Use the decimal dot and
    # a complete, equal-height suffix before falling back to symbol boxes.
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    _, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    body = sorted(
        (int(x), int(y), int(w), int(h))
        for x, y, w, h, _ in stats[1:]
        if h > gray.shape[0] * 0.4 and w > 2
    )
    if body:
        from statistics import median

        height = median(h for _, _, _, h in body)
        count = 0
        suffix = []
        for box in reversed(body):
            x, y, w, h = box
            n = round(w / (height * 0.5))
            if not 1 <= n <= 3 or abs(w / (height * 0.5) - n) > 0.3:
                break
            suffix.insert(0, box)
            count += n
            if count >= integer_digits + fraction_digits:
                break
        if count == integer_digits + fraction_digits:
            left, right = suffix[0][0], suffix[-1][0] + suffix[-1][2]
            top, bottom = (
                min(y for _, y, _, _ in suffix),
                max(y + h for _, y, _, h in suffix),
            )
            decimals = [
                stat
                for stat in stats[1:]
                if left < stat[0] < right
                and stat[3] <= height * 0.3
                and bottom - height * 0.3 <= stat[1] < bottom
                and stat[4] >= 3
            ]
            integer_count = 0
            for first, second in pairwise(suffix):
                integer_count += round(first[2] / (height * 0.5))
                if integer_count == integer_digits and any(
                    first[0] + first[2] <= dot[0] and dot[0] + dot[2] <= second[0]
                    for dot in decimals
                ):
                    return left, top, right, bottom
    return None


def decimal_suffix_region(
    gray: np.ndarray, integer_digits: int, fraction_digits: int = 2
) -> np.ndarray | None:
    bounds = _decimal_suffix_bounds(gray, integer_digits, fraction_digits)
    if bounds is None:
        return None
    left, top, right, bottom = bounds
    return gray[
        max(0, top - 4) : min(gray.shape[0], bottom + 3),
        max(0, left - 6) : min(gray.shape[1], right + 3),
    ]


def decimal_expression_equals(gray: np.ndarray, integer_digits: int) -> bool:
    """Require a source two-bar glyph directly before a verified decimal suffix."""
    import cv2

    from docling.utils.ocr_region_pixels import two_bar_marker

    bounds = _decimal_suffix_bounds(gray, integer_digits)
    if bounds is None:
        return False
    left, top, _, bottom = bounds
    height = bottom - top
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    _, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    markers = [
        (int(x), int(y), int(w), int(h))
        for x, y, w, h, _ in stats[1:]
        if left - height <= x + w <= left
        and height * 0.25 <= w <= height
        and 1 <= h <= height * 0.6
        and top + height * 0.2 <= y + h / 2 <= top + height * 0.8
    ]
    groups: list[list[tuple[int, int, int, int]]] = []
    for marker in sorted(markers):
        if (
            groups
            and abs(marker[0] + marker[2] / 2 - groups[-1][0][0] - groups[-1][0][2] / 2)
            <= height * 0.25
        ):
            groups[-1].append(marker)
        else:
            groups.append([marker])
    supported = 0
    for group in groups:
        x0, y0 = min(m[0] for m in group), min(m[1] for m in group)
        x1, y1 = max(m[0] + m[2] for m in group), max(m[1] + m[3] for m in group)
        supported += two_bar_marker(gray[y0:y1, x0:x1])
    return supported == 1
