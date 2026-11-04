# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Clean original OCR regions without changing their printed characters."""

from statistics import median

import numpy as np


def light_on_dark(gray: np.ndarray) -> bool:
    """Distinguish bright lettering from dark type on uneven gray paper."""
    if gray.size == 0:
        return False
    low, middle, high = np.percentile(gray, (10, 50, 95))
    return bool(middle < 180 and high - middle > max(20, (middle - low) * 1.2))


def trim_light_text(gray: np.ndarray) -> np.ndarray:
    """Exclude fragments of neighboring rows from a light-on-dark line."""
    ink = (gray > 180).sum(axis=1) >= max(2, gray.shape[1] * 0.02)
    bands: list[tuple[int, int]] = []
    for row in np.flatnonzero(ink):
        if bands and row - bands[-1][1] <= 3:
            bands[-1] = bands[-1][0], int(row) + 1
        else:
            bands.append((int(row), int(row) + 1))
    body = [band for band in bands if band[1] - band[0] >= gray.shape[0] * 0.25]
    if not body:
        return gray
    top, bottom = min(body, key=lambda band: abs(sum(band) - gray.shape[0]))
    return gray[max(0, top - 2) : min(gray.shape[0], bottom + 2)]


def clean_region(gray: np.ndarray, *, remove_noise: bool = False) -> np.ndarray:
    """Remove long border rules, then crop to glyphs without deleting dots."""
    import cv2

    height, width = gray.shape
    if height < 4 or width < 4:
        return gray
    threshold, ink = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )
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
    # Keep the source threshold. Whitened rules must not turn gray paper
    # into foreground when a second Otsu threshold sees the new white pixels.
    _, ink = cv2.threshold(result, threshold, 255, cv2.THRESH_BINARY_INV)
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


def terminal_exclamation(gray: np.ndarray) -> bool:
    """Recognize a final separate stem and dot without inventing punctuation."""
    import cv2

    if gray.ndim != 2 or min(gray.shape) < 5:
        return False
    source = clean_region(gray)
    _, ink = cv2.threshold(source, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    _, _, stats, _ = cv2.connectedComponentsWithStats(ink)
    body = [s for s in stats[1:] if s[3] >= source.shape[0] * 0.35 and s[4] >= 4]
    if len(body) < 4:
        return False
    height = median(int(s[3]) for s in body)
    for stem in body:
        x, y, width, h, _ = map(int, stem)
        if (
            x < source.shape[1] * 0.75
            or not height * 0.45 <= h <= height * 1.2
            or width > h * 0.45
        ):
            continue
        for dot in stats[1:]:
            dx, dy, dw, dh, area = map(int, dot)
            if (
                area >= 1
                and dw <= max(2, height * 0.35)
                and dh <= max(2, height * 0.3)
                and 0 <= dy - (y + h) <= height * 0.5
                and abs((dx + dw / 2) - (x + width / 2)) <= max(2, width)
                and not any(
                    other[0] > x + width + height * 0.4 and other[3] >= height * 0.5
                    for other in body
                )
            ):
                return True
    return False


def printed_dash(gray: np.ndarray) -> bool:
    """Distinguish a printed dash in a numeric cell from a dot or an empty cell."""
    import cv2

    height, width = gray.shape
    if height < 6 or width < 6:
        return False
    # Remove only long, nearly straight rules on the outer cell boundary.
    # A dash can touch a skewed frame; discarding its connected component
    # would discard the printed glyph as well. Keep tall interior glyphs so
    # that a digit or a negative number cannot become a lone dash.
    supported = 0
    strong = False
    # Camera blur can leave an elongated core only between two coarse
    # thresholds. Check finer intensity steps while retaining the same shape
    # and repeated-support requirements that reject dots and digit fragments.
    for threshold in range(64, 209, 16):
        _, ink = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
        segments = cv2.HoughLinesP(
            ink,
            1,
            np.pi / 1800,
            threshold=max(8, height // 4),
            minLineLength=max(6, height * 0.6),
            maxLineGap=max(3, round(height * 0.25)),
        )
        mask = np.zeros_like(ink)
        if segments is not None:
            for x1, y1, x2, y2 in segments.reshape(-1, 4):
                vertical = (
                    abs(y2 - y1) >= height * 0.7
                    and abs(x2 - x1) <= height * 0.12
                    and (
                        (x1 + x2) / 2 < max(3, width * 0.015)
                        or (x1 + x2) / 2 > width - max(3, width * 0.015)
                    )
                )
                horizontal = (
                    abs(x2 - x1) >= width * 0.8
                    and abs(y2 - y1) <= width * 0.03
                    and ((y1 + y2) / 2 < height * 0.25 or (y1 + y2) / 2 > height * 0.75)
                )
                if vertical or horizontal:
                    cv2.line(
                        mask,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        255,
                        max(3, min(6, round(min(height, width) * 0.06))),
                    )
        ink[mask > 0] = 0
        # At a clipped cell boundary a rule can leave a one-pixel remnant.
        # Retain the rest of an edge-aligned dash, including its horizontal core.
        ink[:, :2] = 0
        ink[:, -2:] = 0
        _, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
        central = [
            stat
            for stat in stats[1:]
            if stat[4] >= 3 and height * 0.1 < stat[1] + stat[3] / 2 < height * 0.9
        ]
        if len(central) == 1:
            stat = central[0]
            if (
                stat[2] >= max(4, 1.4 * stat[3])
                and stat[2] < min(width * 0.7, height * 1.5)
                and stat[3] <= height * 0.3
                and stat[4] >= stat[2] * stat[3] * 0.6
            ):
                supported += 1
                strong = strong or bool(stat[2] >= 1.4 * stat[3])
                continue
        body = [
            stat
            for stat in stats[1:]
            if stat[4] >= 6
            and not (
                stat[3] >= max(height * 0.6, 3 * stat[2])
                and (
                    stat[0] + stat[2] <= max(3, width * 0.05)
                    or stat[0] >= width - max(3, width * 0.05)
                )
                and (stat[1] <= height * 0.08 or stat[1] + stat[3] >= height * 0.92)
            )
            and not (
                stat[2] <= max(2, width * 0.008)
                and stat[3] >= max(height * 0.12, 6 * stat[2])
                and (
                    stat[0] <= max(3, width * 0.015)
                    or stat[0] >= width - max(3, width * 0.015)
                )
                and (
                    stat[1] <= max(3, height * 0.04)
                    or stat[1] + stat[3] >= height - max(3, height * 0.04)
                )
            )
            and not (
                stat[2] >= width * 0.5
                and stat[3] <= height * 0.35
                and (stat[1] < height * 0.2 or stat[1] > height * 0.8)
            )
            and not (
                stat[2] >= width * 0.4
                and stat[3] >= height * 0.7
                and stat[4] < stat[2] * stat[3] * 0.2
                and (stat[0] < width * 0.02 or stat[0] + stat[2] > width * 0.98)
                and (stat[1] < height * 0.04 or stat[1] + stat[3] > height * 0.96)
            )
            and 2 < stat[1] + stat[3] / 2 < height - 2
        ]
        bars = [
            stat
            for stat in body
            if stat[2] >= max(4, 1.4 * stat[3])
            and stat[2] < min(width * 0.7, height * 1.5)
            and stat[3] <= height * 0.3
            and stat[4] >= stat[2] * stat[3] * 0.6
            and stat[1] + stat[3] / 2 > height * 0.1
            and (
                height * 0.1 <= stat[1] + stat[3] / 2 <= height * 0.9
                or stat[2] < width * 0.15
            )
        ]
        if len(bars) != 1:
            continue
        dash = bars[0]
        if any(
            stat is not dash
            and stat[2] > 2
            and stat[3] > max(1.5 * dash[3], height * 0.06)
            for stat in body
        ):
            continue
        supported += 1
        strong = strong or bool(dash[2] >= 1.4 * dash[3])
    return supported >= 2 and strong


def photo_contrast(gray: np.ndarray) -> np.ndarray:
    """Expose faint photographic ink while retaining its source shape."""
    import cv2

    kernel = max(3, min(31, gray.shape[0] // 2) | 1)
    background = cv2.morphologyEx(
        gray, cv2.MORPH_CLOSE, np.ones((kernel, kernel), np.uint8)
    )
    difference = np.maximum(background.astype(np.float32) - gray, 0)
    maximum = float(difference.max())
    if maximum < 20:
        return gray
    return np.clip(255 - difference * 255 / maximum, 0, 255).astype(np.uint8)


def two_bar_marker(gray: np.ndarray, *, allow_uneven_bars: bool = False) -> bool:
    if gray.size == 0 or gray.shape[1] < 2:
        return False
    ink = gray < 128
    return any(
        _two_bar_shape(ink, coverage, allow_uneven_bars=allow_uneven_bars)
        for coverage in (
            (0.4, 0.55, 0.65, 0.9) if allow_uneven_bars else (0.4, 0.55, 0.65)
        )
    )


def _two_bar_shape(
    ink: np.ndarray, coverage: float, *, allow_uneven_bars: bool = False
) -> bool:
    # Tesseract can include unrelated horizontal padding inside a glyph box.
    occupied = ink.sum(axis=1) >= max(2, ink.shape[1] * coverage)
    bands: list[tuple[int, int]] = []
    for index, value in enumerate(occupied):
        if value:
            if bands and bands[-1][1] == index:
                bands[-1] = (bands[-1][0], index + 1)
            else:
                bands.append((index, index + 1))
    groups: list[list[tuple[int, int]]] = []
    for band in bands:
        if groups and band[0] - groups[-1][-1][1] <= max(
            2, 1.5 * max(band[1] - band[0], groups[-1][-1][1] - groups[-1][-1][0])
        ):
            groups[-1].append(band)
        else:
            groups.append([band])
    # A fragment above the current line must not become a second minus bar.
    if not groups or len(groups[-1]) != 2 or groups[-1][-1][1] < ink.shape[0] * 0.8:
        return False
    first, second = groups[-1]
    heights = (first[1] - first[0], second[1] - second[0])
    if not allow_uneven_bars and max(heights) > 2 * min(heights):
        return False
    first_ink = ink[first[0] : first[1]].mean(axis=0) >= 0.5
    second_ink = ink[second[0] : second[1]].mean(axis=0) >= 0.5
    if min(first_ink.sum(), second_ink.sum()) < max(2, ink.shape[1] * 0.3):
        return False
    if allow_uneven_bars and (
        min(first_ink.sum(), second_ink.sum())
        < max(first_ink.sum(), second_ink.sum()) * 0.85
        or first_ink.sum() < heights[0] * 1.2
        or second_ink.sum() < heights[1] * 1.2
    ):
        return False
    return bool(
        (first_ink & second_ink).sum() >= 0.75 * min(first_ink.sum(), second_ink.sum())
    )
