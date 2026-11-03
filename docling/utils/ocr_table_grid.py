# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Recover dense ruled tables from their printed grid and OCR word coordinates."""

from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass
from itertools import pairwise
from math import ceil
from statistics import median

import numpy as np
from docling_core.types.doc import BoundingBox, DocItemLabel, TableCell
from docling_core.types.doc.page import TextCell
from PIL import Image

from docling.datamodel.base_models import Cluster, Table
from docling.utils.ocr_photo_grid import table_from_photo_grid
from docling.utils.ocr_table_content import record_ocr_table_coverage
from docling.utils.ocr_table_orientation import label_first_rows, sideways_grid_views


def _centers(positions: list[float], tolerance: float) -> list[float]:
    groups: list[list[float]] = []
    for position in sorted(positions):
        if groups and position - groups[-1][0] <= tolerance:
            groups[-1].append(position)
        else:
            groups.append([position])
    return [median(group) for group in groups]


def _row_centers(positions: list[float]) -> list[float]:
    groups: list[list[float]] = []
    for position in positions:
        if groups and position - groups[-1][-1] <= 2:
            groups[-1].append(position)
        else:
            groups.append([position])
    return [median(group) for group in groups]


def _fragmented_verticals(
    binary_views: list[np.ndarray], width: int, height: int
) -> list[float]:
    """Find ruling tracks whose short pieces jointly cover the table height.

    A faint separator can be broken at every horizontal rule. Requiring one
    segment to span half the table loses real columns. Character stems alone
    must not qualify: require long pieces and substantial nonduplicated coverage.
    """
    import cv2

    tracks: list[tuple[float, float, float]] = []
    for binary in binary_views:
        segments = cv2.HoughLinesP(
            binary,
            1,
            np.pi / 1800,
            threshold=max(20, height // 30),
            minLineLength=max(24, height * 0.1),
            maxLineGap=max(3, height // 80),
        )
        if segments is None:
            continue
        for x1, y1, x2, y2 in segments.reshape(-1, 4):
            dy = float(y2 - y1)
            if abs(dy) < max(24, height * 0.1) or abs(x2 - x1) > 0.025 * abs(dy):
                continue
            anchor = float(x1) + (height / 2 - float(y1)) * float(x2 - x1) / dy
            tracks.append((anchor, float(min(y1, y2)), float(max(y1, y2))))
    groups: list[list[tuple[float, float, float]]] = []
    for track in sorted(tracks):
        if groups and track[0] - groups[-1][0][0] <= max(3, width * 0.008):
            groups[-1].append(track)
        else:
            groups.append([track])
    boundaries = []
    for group in groups:
        bands: list[tuple[float, float]] = []
        for _, top, bottom in sorted(group, key=lambda track: track[1]):
            if bands and top <= bands[-1][1]:
                bands[-1] = bands[-1][0], max(bottom, bands[-1][1])
            else:
                bands.append((top, bottom))
        if sum(bottom - top for top, bottom in bands) >= height * 0.65:
            boundaries.append(median(track[0] for track in group))
    return boundaries


def table_from_ocr_grid(
    image: Image.Image,
    cluster: Cluster,
    words: list[TextCell],
    *,
    page_no: int,
    scale: float,
    allow_open_labels: bool = False,
    adaptive_thresholds: bool = False,
) -> Table | None:
    words = [word for word in words if word.text.strip()]
    if not words or not all(word.from_ocr for word in words):
        return None
    views = sideways_grid_views(image, cluster, words, scale=scale)
    if views and all(word.from_ocr for word in words):
        candidates = []
        for view in views:
            table = _table_from_ocr_grid(
                view.image,
                view.cluster,
                view.words,
                page_no=page_no,
                scale=scale,
                allow_open_labels=True,
                adaptive_thresholds=adaptive_thresholds,
            )
            if table is not None:
                candidates.append((label_first_rows(table), view, table))
        candidates.sort(key=lambda candidate: candidate[0], reverse=True)
        if (
            candidates
            and candidates[0][0] >= 5
            and (len(candidates) == 1 or candidates[0][0] >= candidates[1][0] + 3)
        ):
            _, view, table = candidates[0]
            return view.restore(table, cluster)
        # Do not silently materialize a sideways table as a transposed grid.
        return None
    if adaptive_thresholds:
        photographed = table_from_photo_grid(
            image, cluster, words, page_no=page_no, scale=scale
        )
        if photographed is not None:
            return photographed
    table = _table_from_ocr_grid(
        image,
        cluster,
        words,
        page_no=page_no,
        scale=scale,
        allow_open_labels=allow_open_labels,
        adaptive_thresholds=adaptive_thresholds,
    )
    if table is None and not adaptive_thresholds:
        # A scanned PDF can have the same skew as a camera image. Keep the
        # existing straight-grid path first, then require a complete ruled
        # grid in rectified coordinates before replacing predicted cells.
        return table_from_photo_grid(
            image, cluster, words, page_no=page_no, scale=scale
        )
    return table


def _table_from_ocr_grid(
    image: Image.Image,
    cluster: Cluster,
    words: list[TextCell],
    *,
    page_no: int,
    scale: float,
    allow_open_labels: bool = False,
    adaptive_thresholds: bool = False,
) -> Table | None:
    """Prefer a complete, dense printed grid over merged model-predicted rows.

    This fallback needs OCR word coordinates, at least three ruled columns and
    five rows. Each row boundary must cross a majority of the columns. Ordinary
    text and sparse/partly ruled tables stay with the structure model. OpenCV
    is optional in Docling and is installed by the RapidOCR extra.
    """
    if not words or not all(word.from_ocr for word in words):
        return None
    try:
        import cv2
    except ImportError:
        return None

    box = cluster.bbox.scaled(scale)
    padding = max(
        3,
        round((min(box.width, box.height) if allow_open_labels else box.width) * 0.02),
    )
    left = max(0, int(box.l) - padding)
    top = max(0, int(box.t) - padding)
    right = min(image.width, ceil(box.r) + padding)
    bottom = min(image.height, ceil(box.b) + padding)
    if right - left < 100 or bottom - top < 100:
        return None
    gray = np.asarray(image.crop((left, top, right, bottom)).convert("L"))
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    binary_views = [ink]
    if allow_open_labels or adaptive_thresholds:
        binary_views.extend(
            cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY_INV,
                window,
                offset,
            )
            for window, offset in ((31, 4), (31, 10), (51, 4))
        )
        ink = binary_views[0].copy()
        for binary in binary_views[1:]:
            ink |= binary
    height, width = ink.shape
    detected = [
        cv2.HoughLinesP(
            binary,
            1,
            np.pi / 1800,
            threshold=max(30, height // 12),
            minLineLength=height * 0.45,
            maxLineGap=max(3, height // (40 if allow_open_labels else 80)),
        )
        for binary in binary_views
    ]
    valid_segments = [
        segment.reshape(-1, 4) for segment in detected if segment is not None
    ]
    if not valid_segments:
        return None
    segments = np.concatenate(valid_segments)
    verticals = [
        float(x1 + x2) / 2
        for x1, y1, x2, y2 in segments.reshape(-1, 4)
        if abs(y2 - y1) >= height * 0.45
        and abs(x2 - x1) <= (0.05 if allow_open_labels else 0.025) * abs(y2 - y1)
    ]
    column_tolerance = max(3, width * 0.008)
    columns = _centers(verticals, column_tolerance)
    # Short fragments can supply a missing rule. They must not shift an
    # already supported long rule and change an otherwise valid source crop.
    for position in _fragmented_verticals(binary_views, width, height):
        if all(abs(position - column) > column_tolerance for column in columns):
            columns.append(position)
    if len(columns) < 4:
        # A wholly unsupported grid still needs the skew-aware fallback below.
        return None
    if len(binary_views) == 1:
        # A global threshold can erase one faint interior separator while the
        # other rules still form a plausible but incomplete grid. Add locally
        # supported tracks only after preserving every established boundary.
        local_ink = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 4
        )
        for position in _fragmented_verticals([local_ink], width, height):
            if all(abs(position - column) > column_tolerance for column in columns):
                columns.append(position)
    columns.sort()
    if allow_open_labels and columns[0] > width * 0.06:
        # Some financial forms have no left rule around the label column.
        # Repeated full-width row rules and real labels must support that column.
        horizontals = [
            (float(y1 + y2) / 2)
            for x1, y1, x2, y2 in segments.reshape(-1, 4)
            if abs(x2 - x1) >= width * 0.8 and abs(y2 - y1) <= 0.025 * abs(x2 - x1)
        ]
        labels = [
            word
            for word in words
            if sum(char.isalpha() for char in word.text) >= 4
            and (word.rect.to_bounding_box().l + word.rect.to_bounding_box().r)
            * scale
            / 2
            - left
            < columns[0]
        ]
        if len(_centers(horizontals, 4)) >= 5 and len(labels) >= 5:
            columns.insert(0, 0.0)
    # All outer borders must be visible, not a set of incidental interior lines.
    if columns[0] > width * 0.06 or columns[-1] < width * 0.94:
        return None
    tolerance = max(1, round(width / 280))
    signals = []
    for start, end in pairwise(columns):
        inset = max(2, round((end - start) * 0.08))
        if end - start <= 2 * inset:
            return None
        signal = (ink[:, round(start) + inset : round(end) - inset] > 0).mean(axis=1)
        signals.append(
            np.convolve(signal >= 0.65, np.ones(2 * tolerance + 1), mode="same") > 0
        )
    ruled = np.sum(signals, axis=0) >= ceil(len(signals) * 0.6)
    positions = np.flatnonzero(ruled).astype(float).tolist()
    rows = (
        _row_centers(positions)
        if allow_open_labels
        else _centers(positions, 2 * tolerance + 2)
    )
    # Retain the capture margin needed by slightly clipped model boxes.
    # In a wide table that margin can exceed six percent of its short side.
    row_margin = max(height * 0.06, padding + height * 0.04)
    if len(rows) < 6 or (
        not allow_open_labels
        and (rows[0] > row_margin or rows[-1] < height - row_margin)
    ):
        return None
    if any(end - start <= 2 * tolerance for start, end in pairwise(rows)):
        return None

    vertical_slopes = [
        float(x2 - x1) / float(y2 - y1)
        for x1, y1, x2, y2 in segments
        if abs(y2 - y1) >= height * 0.45 and abs(x2 - x1) <= 0.025 * abs(y2 - y1)
    ]
    rectified = None
    if vertical_slopes and abs(median(vertical_slopes)) * height > max(
        3, min(end - start for start, end in pairwise(columns)) * 0.1
    ):
        # Retain established numeric cells and their original source pixels.
        # A supported warp can locate short dashes near a slanted border without
        # rebuilding every cell or interpolating faint numeric glyphs again.
        rectified = table_from_photo_grid(
            image, cluster, words, page_no=page_no, scale=scale
        )
        if rectified is not None and (
            rectified.num_cols != len(columns) - 1
            or rectified.num_rows != len(rows) - 1
        ):
            rectified = None
    geometry = rectified.ocr_grid_geometry if rectified is not None else None
    rectified_cells = (
        {
            (cell.start_row_offset_idx, cell.start_col_offset_idx): cell
            for cell in rectified.table_cells
        }
        if rectified is not None
        else {}
    )
    moved_markers: set[tuple[int, int]] = set()

    contents: dict[tuple[int, int], list[TextCell]] = defaultdict(list)
    placed: set[int] = set()
    outside: list[TextCell] = []
    for word_index, word in enumerate(words):
        word_box = word.rect.to_bounding_box()
        x = (word_box.l + word_box.r) * scale / 2 - left
        y = (word_box.t + word_box.b) * scale / 2 - top
        column, row = bisect_right(columns, x) - 1, bisect_right(rows, y) - 1
        if (
            geometry is not None
            and not any(char.isdigit() for char in word.text)
            and sum(char.isalpha() for char in word.text) <= 1
            and any(char in "-_\u2013\u2014" for char in word.text)
        ):
            (a, b, c), (d, e, f) = geometry.transform
            source_x, source_y = (x + left) / scale, (y + top) / scale
            marker_column = (
                bisect_right(geometry.columns, a * source_x + b * source_y + c) - 1
            )
            marker_row = (
                bisect_right(geometry.rows, d * source_x + e * source_y + f) - 1
            )
            if (
                0 <= marker_column < len(columns) - 1
                and 0 <= marker_row < len(rows) - 1
                and (marker_row, marker_column) != (row, column)
            ):
                column, row = marker_column, marker_row
                moved_markers.add((row, column))
        if 0 <= column < len(columns) - 1 and 0 <= row < len(rows) - 1:
            contents[row, column].append(word)
            placed.add(word_index)
        elif allow_open_labels and not rows[0] <= y <= rows[-1]:
            outside.append(word)
            placed.add(word_index)
    if len(placed) - len(outside) < (len(words) - len(outside)) * 0.9:
        return None
    offset = int(bool(outside))
    cells = [
        TableCell(
            text=" ".join(
                word.text
                for word in (
                    sorted(
                        contents[row, column],
                        key=lambda word: word.rect.to_bounding_box().l,
                    )
                    if allow_open_labels
                    and not any(
                        char.isalpha()
                        for word in contents[row, column]
                        for char in word.text
                    )
                    else contents[row, column]
                )
            ),
            start_row_offset_idx=row + offset,
            end_row_offset_idx=row + 1 + offset,
            start_col_offset_idx=column,
            end_col_offset_idx=column + 1,
            column_header=row == 0,
            bbox=rectified_cells[row, column].bbox
            if (row, column) in moved_markers
            else BoundingBox(
                l=(left + columns[column]) / scale,
                r=(left + columns[column + 1]) / scale,
                t=(top + rows[row]) / scale,
                b=(top + rows[row + 1]) / scale,
            ),
        )
        for row in range(len(rows) - 1)
        for column in range(len(columns) - 1)
    ]
    if outside:
        boxes = [word.rect.to_bounding_box() for word in outside]
        cells.insert(
            0,
            TableCell(
                text=" ".join(word.text for word in outside),
                start_row_offset_idx=0,
                end_row_offset_idx=1,
                start_col_offset_idx=0,
                end_col_offset_idx=len(columns) - 1,
                col_span=len(columns) - 1,
                column_header=True,
                bbox=BoundingBox(
                    l=min(box.l for box in boxes),
                    t=min(box.t for box in boxes),
                    r=max(box.r for box in boxes),
                    b=max(box.b for box in boxes),
                ),
            ),
        )
    table = Table(
        from_ocr_grid=True,
        ocr_grid_rectified=geometry is not None and not outside,
        ocr_grid_geometry=geometry if not outside else None,
        id=cluster.id,
        label=cluster.label,
        page_no=page_no,
        cluster=cluster,
        otsl_seq=[],
        table_cells=cells,
        num_rows=len(rows) - 1 + offset,
        num_cols=len(columns) - 1,
    )
    record_ocr_table_coverage(table, words, matched_token_ids=placed)
    return table


@dataclass(frozen=True)
class RecoveredOcrTable:
    table: Table
    cell_indices: frozenset[int]


def table_from_aligned_ocr(
    image: Image.Image,
    lines: list[TextCell],
    *,
    page_no: int,
    cluster_id: int,
    scale: float,
) -> RecoveredOcrTable | None:
    """Recover an overlooked two-column table with a visible vertical divider.

    Require repeated left alignment, at least eight paired rows, short values,
    a consistent progression down both columns and a long printed divider.
    The row progression may differ between columns on a curved photograph.
    """
    try:
        import cv2
    except ImportError:
        return None
    candidates = [
        cell
        for cell in lines
        if cell.from_ocr and cell.text.strip() and cell.confidence >= 0.5
    ]
    if len(candidates) < 16:
        return None
    tolerance = image.width / scale * 0.04
    groups: list[list[TextCell]] = []
    for cell in sorted(candidates, key=lambda item: item.rect.to_bounding_box().l):
        x = cell.rect.to_bounding_box().l
        if groups and x - groups[-1][0].rect.to_bounding_box().l <= tolerance:
            groups[-1].append(cell)
        else:
            groups.append([cell])
    groups = [
        sorted(
            group,
            key=lambda item: (
                (item.rect.to_bounding_box().t + item.rect.to_bounding_box().b) / 2
            ),
        )
        for group in groups
        if len(group) >= 8
    ]
    if len(groups) < 2:
        return None
    pairs = []
    for left_group, right_group in pairwise(groups):
        if abs(len(left_group) - len(right_group)) > 2:
            continue
        if median(len(cell.text) for cell in right_group) > 24:
            continue
        gap_left = max(cell.rect.to_bounding_box().r for cell in left_group) * scale
        gap_right = min(cell.rect.to_bounding_box().l for cell in right_group) * scale
        if gap_right - gap_left >= image.width * 0.01:
            pairs.append((left_group, right_group, gap_left, gap_right))
    if not pairs:
        return None
    gray = np.asarray(image.convert("L"))
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    segments = cv2.HoughLinesP(
        ink,
        1,
        np.pi / 1800,
        threshold=max(50, image.height // 6),
        minLineLength=image.height * 0.4,
        maxLineGap=max(3, image.height // 50),
    )
    if segments is None:
        return None
    for left_group, right_group, gap_left, gap_right in pairs:
        count = min(len(left_group), len(right_group))
        if not any(
            abs(y2 - y1) >= image.height * 0.4
            and abs(x2 - x1) <= 0.15 * abs(y2 - y1)
            and gap_left <= (x1 + x2) / 2 <= gap_right
            for x1, y1, x2, y2 in segments.reshape(-1, 4)
        ):
            continue
        best: tuple[float, list[TextCell], list[TextCell]] | None = None
        for left_start in range(len(left_group) - count + 1):
            for right_start in range(len(right_group) - count + 1):
                first = left_group[left_start : left_start + count]
                second = right_group[right_start : right_start + count]
                ys = [
                    np.array(
                        [
                            (
                                cell.rect.to_bounding_box().t
                                + cell.rect.to_bounding_box().b
                            )
                            / 2
                            for cell in column
                        ]
                    )
                    for column in (first, second)
                ]
                step = float(median(np.diff(ys[1]).tolist()))
                if step <= 0:
                    continue
                # A missing middle row must not shift later associations.
                if any(
                    np.min(np.diff(centers)) < 0.35 * median(np.diff(centers).tolist())
                    or np.max(np.diff(centers))
                    > 1.7 * median(np.diff(centers).tolist())
                    for centers in ys
                ):
                    continue
                fit = np.polyfit(ys[0], ys[1], 1)
                slope, intercept = float(fit[0]), float(fit[1])
                residual = (
                    float(np.sqrt(np.mean((ys[1] - slope * ys[0] - intercept) ** 2)))
                    / step
                )
                if (
                    0.5 <= slope <= 2
                    and residual <= 0.3
                    and (best is None or residual < best[0])
                ):
                    best = residual, first, second
        if best is None:
            continue
        _, first, second = best
        selected = first + second
        boxes = [cell.rect.to_bounding_box() for cell in selected]
        bbox = BoundingBox(
            l=min(b.l for b in boxes),
            t=min(b.t for b in boxes),
            r=max(b.r for b in boxes),
            b=max(b.b for b in boxes),
        )
        cluster = Cluster(
            id=cluster_id, label=DocItemLabel.TABLE, bbox=bbox, cells=selected
        )
        table = Table(
            id=cluster_id,
            label=DocItemLabel.TABLE,
            page_no=page_no,
            cluster=cluster,
            otsl_seq=[],
            num_rows=count,
            num_cols=2,
            table_cells=[
                TableCell(
                    text=cell.text,
                    bbox=cell.rect.to_bounding_box(),
                    start_row_offset_idx=row,
                    end_row_offset_idx=row + 1,
                    start_col_offset_idx=column,
                    end_col_offset_idx=column + 1,
                )
                for row, pair in enumerate(zip(first, second))
                for column, cell in enumerate(pair)
            ],
        )
        return RecoveredOcrTable(table, frozenset(cell.index for cell in selected))
    return None
