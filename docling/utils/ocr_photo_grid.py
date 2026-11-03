# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Recover photographed grids without treating skewed rules as extra rows."""

from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass
from itertools import pairwise
from math import ceil
from statistics import median

import numpy as np
from docling_core.types.doc import BoundingBox, TableCell
from docling_core.types.doc.page import TextCell
from PIL import Image

from docling.datamodel.base_models import Cluster, OcrGridGeometry, Table
from docling.utils.ocr_table_content import record_ocr_table_coverage


@dataclass(frozen=True)
class PhotoGrid:
    columns: list[float]
    rows: list[float]
    horizontal_slope: float
    vertical_slope: float
    width: int
    height: int

    def project(self, x: float, y: float) -> tuple[float, float]:
        return (
            x - self.vertical_slope * (y - self.height / 2),
            y - self.horizontal_slope * (x - self.width / 2),
        )

    def restore(self, x: float, y: float) -> tuple[float, float]:
        x -= self.vertical_slope * self.height / 2
        y -= self.horizontal_slope * self.width / 2
        determinant = 1 - self.vertical_slope * self.horizontal_slope
        return (
            (x + self.vertical_slope * y) / determinant,
            (y + self.horizontal_slope * x) / determinant,
        )


def photo_grid(gray: np.ndarray, *, padding: int = 0) -> PhotoGrid | None:
    import cv2

    height, width = gray.shape
    ink = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 10
    )
    segments = cv2.HoughLinesP(
        ink,
        1,
        np.pi / 1800,
        threshold=max(30, min(height, width) // 12),
        minLineLength=min(height, width) * 0.4,
        maxLineGap=max(3, round(min(height, width) * 0.02)),
    )
    if segments is None:
        return None
    horizontal = []
    vertical = []
    for x1, y1, x2, y2 in segments.reshape(-1, 4):
        dx, dy = int(x2 - x1), int(y2 - y1)
        if abs(dx) >= width * 0.6 and abs(dy) <= abs(dx) * 0.06:
            horizontal.append(dy / dx)
        if abs(dy) >= height * 0.6 and abs(dx) <= abs(dy) * 0.06:
            vertical.append(dx / dy)
    if len(horizontal) < 5 or len(vertical) < 4:
        return None
    horizontal_slope, vertical_slope = median(horizontal), median(vertical)
    transform = np.array(
        [
            [1, -vertical_slope, vertical_slope * height / 2],
            [-horizontal_slope, 1, horizontal_slope * width / 2],
        ],
        dtype=np.float64,
    )
    rectified = cv2.warpAffine(ink, transform, (width, height))
    boundaries = []
    for axis in (0, 1):
        # A three-pixel strip tolerates the width of a printed rule. It does
        # not join neighboring rows or invent a missing boundary.
        kernel = np.ones((1, 3) if axis == 0 else (3, 1), np.uint8)
        supported = (cv2.dilate(rectified, kernel) > 127).mean(axis=axis) >= 0.65
        groups: list[list[int]] = []
        for position in np.flatnonzero(supported):
            if groups and position - groups[-1][-1] <= 2:
                groups[-1].append(int(position))
            else:
                groups.append([int(position)])
        boundaries.append([float(median(group)) for group in groups])
    columns, rows = boundaries
    # Rectifying a sloped outer rule leaves corner clearance in the original
    # axis-aligned canvas. Account for that geometric clearance separately
    # from the tolerated margin; a wide table magnifies the vertical offset.
    column_margin = (
        max(width * 0.06, padding + width * 0.04) + abs(vertical_slope) * height / 2
    )
    row_margin = (
        max(height * 0.06, padding + height * 0.04) + abs(horizontal_slope) * width / 2
    )
    if (
        len(columns) < 4
        or len(rows) < 6
        or columns[0] > column_margin
        or columns[-1] < width - column_margin
        or rows[0] > row_margin
        or rows[-1] < height - row_margin
        or any(b - a < 6 for a, b in pairwise(columns))
        or any(b - a < 6 for a, b in pairwise(rows))
    ):
        return None
    return PhotoGrid(columns, rows, horizontal_slope, vertical_slope, width, height)


def _cell_regions(
    straight: np.ndarray,
    *,
    table: Table,
    columns: list[float],
    rows: list[float],
    scale: float,
) -> dict[int, np.ndarray]:
    margin = max(1, round(scale * 0.5))
    return {
        id(cell): straight[
            round(rows[cell.start_row_offset_idx]) + margin : round(
                rows[cell.end_row_offset_idx]
            )
            - margin,
            round(columns[cell.start_col_offset_idx]) + margin : round(
                columns[cell.end_col_offset_idx]
            )
            - margin,
        ]
        for cell in table.table_cells
    }


def accepted_grid_regions(
    image: Image.Image, table: Table, *, scale: float
) -> dict[int, np.ndarray]:
    """Sample an accepted grid at reading DPI without detecting its rules again."""
    import cv2

    geometry = table.ocr_grid_geometry
    if (
        not table.from_ocr_grid
        or geometry is None
        or len(geometry.rows) != table.num_rows + 1
        or len(geometry.columns) != table.num_cols + 1
    ):
        return {}
    transform = np.array(geometry.transform, dtype=float)
    transform[:, 2] *= scale
    straight = cv2.warpAffine(
        np.asarray(image.convert("L")),
        transform,
        (ceil(geometry.width * scale), ceil(geometry.height * scale)),
        borderValue=255,
    )
    return _cell_regions(
        straight,
        table=table,
        columns=[position * scale for position in geometry.columns],
        rows=[position * scale for position in geometry.rows],
        scale=scale,
    )


def photo_table_regions(
    image: Image.Image, table: Table, *, scale: float
) -> dict[int, np.ndarray]:
    """Read a recovered photo grid in its own straightened coordinates."""
    import cv2

    if not table.from_ocr_grid:
        return {}
    box = table.cluster.bbox.scaled(scale)
    padding = max(3, round(box.width * 0.02))
    left, top = max(0, int(box.l) - padding), max(0, int(box.t) - padding)
    right = min(image.width, ceil(box.r) + padding)
    bottom = min(image.height, ceil(box.b) + padding)
    if right - left < 100 or bottom - top < 100:
        return {}
    gray = np.asarray(image.crop((left, top, right, bottom)).convert("L"))
    grid = photo_grid(gray, padding=padding)
    if (
        grid is None
        or len(grid.rows) != table.num_rows + 1
        or len(grid.columns) != table.num_cols + 1
    ):
        return {}
    transform = np.array(
        [
            [1, -grid.vertical_slope, grid.vertical_slope * grid.height / 2],
            [-grid.horizontal_slope, 1, grid.horizontal_slope * grid.width / 2],
        ],
        dtype=float,
    )
    straight = cv2.warpAffine(
        gray, transform, (grid.width, grid.height), borderValue=255
    )
    return _cell_regions(
        straight, table=table, columns=grid.columns, rows=grid.rows, scale=scale
    )


def table_from_photo_grid(
    image: Image.Image,
    cluster: Cluster,
    words: list[TextCell],
    *,
    page_no: int,
    scale: float,
) -> Table | None:
    """Use only complete, dense grids supported by source rules and OCR words."""
    try:
        import cv2
    except ImportError:
        return None
    box = cluster.bbox.scaled(scale)
    padding = max(3, round(box.width * 0.02))
    left, top = max(0, int(box.l) - padding), max(0, int(box.t) - padding)
    right = min(image.width, ceil(box.r) + padding)
    bottom = min(image.height, ceil(box.b) + padding)
    if right - left < 100 or bottom - top < 100:
        return None
    gray = np.asarray(image.crop((left, top, right, bottom)).convert("L"))
    grid = photo_grid(gray, padding=padding)
    if grid is None:
        return None
    words = [word for word in words if word.text.strip()]
    contents: dict[tuple[int, int], list[TextCell]] = defaultdict(list)
    placed: set[int] = set()
    for word_index, word in enumerate(words):
        bounds = word.rect.to_bounding_box()
        x, y = grid.project(
            (bounds.l + bounds.r) * scale / 2 - left,
            (bounds.t + bounds.b) * scale / 2 - top,
        )
        col = bisect_right(grid.columns, x) - 1
        row = bisect_right(grid.rows, y) - 1
        if 0 <= col < len(grid.columns) - 1 and 0 <= row < len(grid.rows) - 1:
            contents[row, col].append(word)
            placed.add(word_index)
    if sum(map(len, contents.values())) < len(words) * 0.9:
        return None
    cells = []
    for row in range(len(grid.rows) - 1):
        for col in range(len(grid.columns) - 1):
            corners = [
                grid.restore(x, y)
                for x in grid.columns[col : col + 2]
                for y in grid.rows[row : row + 2]
            ]
            cells.append(
                TableCell(
                    text=" ".join(
                        word.text
                        for word in sorted(
                            contents[row, col],
                            key=lambda word: word.rect.to_bounding_box().l,
                        )
                    ),
                    bbox=BoundingBox(
                        l=(left + min(x for x, _ in corners)) / scale,
                        r=(left + max(x for x, _ in corners)) / scale,
                        t=(top + min(y for _, y in corners)) / scale,
                        b=(top + max(y for _, y in corners)) / scale,
                    ),
                    start_row_offset_idx=row,
                    end_row_offset_idx=row + 1,
                    start_col_offset_idx=col,
                    end_col_offset_idx=col + 1,
                    column_header=row == 0,
                )
            )
    table = Table(
        from_ocr_grid=True,
        ocr_grid_rectified=True,
        ocr_grid_geometry=OcrGridGeometry(
            transform=(
                (
                    1,
                    -grid.vertical_slope,
                    (-left + grid.vertical_slope * (top + grid.height / 2)) / scale,
                ),
                (
                    -grid.horizontal_slope,
                    1,
                    (-top + grid.horizontal_slope * (left + grid.width / 2)) / scale,
                ),
            ),
            columns=[position / scale for position in grid.columns],
            rows=[position / scale for position in grid.rows],
            width=grid.width / scale,
            height=grid.height / scale,
        ),
        id=cluster.id,
        label=cluster.label,
        page_no=page_no,
        cluster=cluster,
        otsl_seq=[],
        table_cells=cells,
        num_rows=len(grid.rows) - 1,
        num_cols=len(grid.columns) - 1,
    )
    record_ocr_table_coverage(table, words, matched_token_ids=placed)
    return table
