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
        if (
            len(body) == integer_digits + fraction_digits
            and 0 < integer_digits < len(body)
            and all(
                0.75 * height <= h <= 1.25 * height and w <= 1.5 * height
                for _, _, w, h in body
            )
        ):
            # Double-width receipt fonts need no assumed half-height pitch
            # when every digit is already a distinct source component.
            left, right = body[0][0], body[-1][0] + body[-1][2]
            top = min(y for _, y, _, _ in body)
            bottom = max(y + h for _, y, _, h in body)
            before, after = body[integer_digits - 1], body[integer_digits]
            if any(
                before[0] + before[2] <= x
                and x + w <= after[0]
                and h <= height * 0.3
                and area >= 3
                and bottom - height * 0.3 <= y < bottom
                for x, y, w, h, area in stats[1:]
            ):
                return left, top, right, bottom
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
    return _thermal_decimal_bounds(body, stats, integer_digits, fraction_digits)


def _thermal_decimal_bounds(
    bodies: list[tuple[int, int, int, int]],
    stats: np.ndarray,
    integer_digits: int,
    fraction_digits: int,
) -> tuple[int, int, int, int] | None:
    """Use the printed fractional glyph pitch for connected, wide digits."""
    if fraction_digits != 2 or not 1 <= integer_digits <= 7 or len(bodies) < 2:
        return None
    from statistics import median

    height = median(box[3] for box in bodies)
    if (
        any(not height * 0.75 <= box[3] <= height * 1.25 for box in bodies)
        or max(box[1] for box in bodies) - min(box[1] for box in bodies) > height * 0.25
    ):
        return None
    supported: set[tuple[int, int, int, int]] = set()
    for fraction_parts in (1, 2):
        if fraction_parts >= len(bodies):
            continue
        integer, fraction = bodies[:-fraction_parts], bodies[-fraction_parts:]
        pitch = (
            fraction[0][2] / 2
            if fraction_parts == 1
            else fraction[1][0]
            + fraction[1][2] / 2
            - fraction[0][0]
            - fraction[0][2] / 2
        )
        if not height * 0.3 <= pitch <= height * 1.2:
            continue
        counts = [round(box[2] / pitch) for box in integer]
        if (
            sum(counts) != integer_digits
            or any(
                n < 1 or abs(box[2] / pitch - n) > 0.3
                for box, n in zip(integer, counts)
            )
            or (fraction_parts == 2 and any(box[2] > pitch * 1.3 for box in fraction))
        ):
            continue
        top = min(box[1] for box in bodies)
        bottom = max(box[1] + box[3] for box in bodies)
        before, after = integer[-1], fraction[0]
        if any(
            before[0] + before[2] <= x
            and x + width <= after[0]
            and h <= height * 0.3
            and area >= 3
            and bottom - height * 0.3 <= y < bottom
            for x, y, width, h, area in stats[1:]
        ):
            supported.add((bodies[0][0], top, bodies[-1][0] + bodies[-1][2], bottom))
    return supported.pop() if len(supported) == 1 else None


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


def decimal_expression_equals(
    gray: np.ndarray, integer_digits: int, *, allow_uneven_bars: bool = False
) -> bool:
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
        and height * 0.25 <= w <= height * 1.5
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
        # A colon has two vertically stacked dots as well. A printed equals
        # sign must have a horizontal outline, not just two occupied bands.
        wide = x1 - x0 > y1 - y0 or (allow_uneven_bars and x1 - x0 >= (y1 - y0) * 0.8)
        supported += wide and two_bar_marker(
            gray[y0:y1, x0:x1], allow_uneven_bars=allow_uneven_bars
        )
    return supported == 1


def connected_decimal_equals(gray: np.ndarray, integer_digits: int) -> bool:
    """Check a thermal equals sign touching a connected, fixed-pitch amount.

    Two connected fractional digits establish the local pitch. A real decimal
    point must separate them from the integer bodies. Account for detached
    integer groups before inspecting the excess prefix of the first group.
    This is pixel evidence only; independent OCR agreement is still required.
    """
    import cv2

    from docling.utils.ocr_region_pixels import two_bar_marker

    if gray.ndim != 2 or min(gray.shape) < 6 or not 1 <= integer_digits <= 7:
        return False
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    _, _, stats, _ = cv2.connectedComponentsWithStats(ink)
    bodies = sorted(
        (int(x), int(y), int(w), int(h))
        for x, y, w, h, _ in stats[1:]
        if h > gray.shape[0] * 0.4 and w > 2
    )
    if len(bodies) < 2:
        return False
    first, fraction = bodies[0], bodies[-1]
    x, y, width, height = first
    fx, fy, fw, fh = fraction
    pitch = fw / 2
    if not height * 0.3 <= pitch <= height * 0.9:
        return False
    remaining = bodies[1:-1]
    counts = [round(w / pitch) for _, _, w, _ in remaining]
    if any(
        count < 1 or abs(w / pitch - count) > 0.25
        for (_, _, w, _), count in zip(remaining, counts)
    ):
        return False
    first_count = integer_digits - sum(counts)
    if first_count < 1:
        return False
    prefix = round(width - first_count * pitch)
    integer_right = bodies[-2][0] + bodies[-2][2]
    if (
        abs(height - fh) > height * 0.15
        or abs(y - fy) > height * 0.15
        or any(
            abs(h - height) > height * 0.15 or abs(iy - y) > height * 0.15
            for _, iy, _, h in remaining
        )
        or not pitch * 0.5 <= prefix <= pitch * 1.6
        or not any(
            integer_right <= dx
            and dx + dw <= fx
            and dh <= height * 0.3
            and y + height * 0.7 <= dy < y + height
            and area >= 3
            for dx, dy, dw, dh, area in stats[1:]
        )
    ):
        return False
    marker = ink[y : y + height, x : x + prefix]
    ys, xs = np.where(marker > 0)
    if not len(xs):
        return False
    marker = marker[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    return two_bar_marker(255 - marker, allow_uneven_bars=True)


def printed_multiplication(gray: np.ndarray, integer_digits: int) -> bool:
    """Verify a small printed cross between a two-decimal price and quantity.

    A decimal dot and the integer span establish the local glyph pitch. The
    cross can touch adjacent digits. Its diagonal arms must fit between two
    complete fractional glyphs and a tall quantity glyph. A third decimal
    digit, plus, minus, colon or trailing ornament cannot supply that evidence.
    """
    import cv2

    if gray.ndim != 2 or min(gray.shape) < 6 or not 1 <= integer_digits <= 7:
        return False
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    _, _, stats, _ = cv2.connectedComponentsWithStats(ink)
    bodies = [s for s in stats[1:] if s[3] > gray.shape[0] * 0.4 and s[2] > 1]
    if len(bodies) < 2:
        return False
    from statistics import median

    height = median(int(s[3]) for s in bodies)
    top = min(int(s[1]) for s in bodies)
    bottom = max(int(s[1] + s[3]) for s in bodies)
    for x, y, width, h, area in stats[1:]:
        if h > height * 0.3 or area < 3 or not bottom - height * 0.35 <= y < bottom:
            continue
        before = [s for s in bodies if s[0] + s[2] <= x]
        after = [s for s in bodies if s[0] >= x + width]
        if not before or not after:
            continue
        pitch = (
            max(int(s[0] + s[2]) for s in before) - min(int(s[0]) for s in before)
        ) / integer_digits
        if not height * 0.3 <= pitch <= height * 1.2:
            continue
        position = min(int(s[0]) for s in after) + 2 * pitch
        for offset in (-0.15, 0.0, 0.15):
            left = max(0, round(position + pitch * (0.1 + offset)))
            right = min(ink.shape[1], round(position + pitch * (0.9 + offset)))
            if right <= left:
                continue
            marker = ink[top:bottom, left:right]
            ys, xs = np.where(marker > 0)
            if len(xs) == 0:
                continue
            marker = marker[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
            mh, mw = marker.shape
            if not height * 0.25 <= mh <= height * 0.75 or mw < mh * 0.4:
                continue
            cy, cx = max(1, round(mh * 0.35)), max(1, round(mw * 0.35))
            if not all(
                part.any()
                for part in (
                    marker[:cy, :cx],
                    marker[:cy, -cx:],
                    marker[-cy:, :cx],
                    marker[-cy:, -cx:],
                )
            ):
                continue
            quantity = ink[top:bottom, min(ink.shape[1], round(position + pitch)) :]
            qy, _ = np.where(quantity > 0)
            if len(qy) > 0 and qy.max() - qy.min() + 1 >= height * 0.75:
                return True
    return False
