# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Commit verified prose without retaining an overlapping copy of its pixels."""

from __future__ import annotations

import unicodedata
from typing import TYPE_CHECKING

from docling_core.types.doc import BoundingBox, CoordOrigin, DocItemLabel
from docling_core.types.doc.page import BoundingRectangle, TextCell

from docling.datamodel.base_models import Cluster, Page

if TYPE_CHECKING:
    from docling.models.stages.ocr.ocr_prose_regions import ProseRow


def _text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def _band(rect: BoundingRectangle, x: float) -> tuple[float, float] | None:
    left = (rect.r_x0 + rect.r_x3) / 2
    right = (rect.r_x1 + rect.r_x2) / 2
    if abs(right - left) < 1e-6:
        return None
    a = (rect.r_y0 + rect.r_y3) / 2
    b = (rect.r_y1 + rect.r_y2) / 2
    height = (abs(rect.r_y0 - rect.r_y3) + abs(rect.r_y1 - rect.r_y2)) / 2
    return a + (x - left) * (b - a) / (right - left), height


def _same_band(first: BoundingRectangle, second: BoundingRectangle, x: float) -> bool:
    if first.coord_origin != second.coord_origin:
        return False
    a, b = _band(first, x), _band(second, x)
    return bool(
        a is not None
        and b is not None
        and min(a[1], b[1]) > 0
        and max(a[1], b[1]) <= 2.5 * min(a[1], b[1])
        and abs(a[0] - b[0]) <= 0.5 * min(a[1], b[1])
    )


def _same_pixels(cell: TextCell, row: ProseRow, words: tuple[TextCell, ...]) -> bool:
    a = cell.rect.to_bounding_box()
    boxes = [word.rect.to_bounding_box() for word in words]
    b = BoundingBox(
        l=min(box.l for box in boxes),
        t=min(box.t for box in boxes),
        r=max(box.r for box in boxes),
        b=max(box.b for box in boxes),
    )
    if min(a.width, b.width) <= 0 or max(a.width, b.width) > 1.5 * min(
        a.width, b.width
    ):
        return False
    x = (max(a.l, b.l) + min(a.r, b.r)) / 2
    return bool(
        _same_band(cell.rect, row.rect, x)
        and max(a.intersection_over_self(b), b.intersection_over_self(a)) >= 0.85
    )


def _covered(cell: TextCell, rows: list[ProseRow]) -> bool:
    # Numeric fields and native PDF text keep their existing ownership guards.
    # Exact punctuation is required; balanced quotes do not alter the page's
    # quote deficit when the redundant cell is removed.
    if (
        not cell.from_ocr
        or cell.rect.coord_origin != CoordOrigin.TOPLEFT
        or cell.confidence < 0.8
        or any(char.isdigit() for char in cell.text)
        or cell.text.count("«") != cell.text.count("»")
    ):
        return False
    wanted = _text(cell.text)
    if len(wanted) < 12:
        return False
    for row in rows:
        if not row.source_confirmed or row.rect.coord_origin != CoordOrigin.TOPLEFT:
            continue
        for start in range(len(row.words)):
            for end in range(start + 1, len(row.words) + 1):
                words = row.words[start:end]
                observed = _text(" ".join(word.text for word in words))
                if observed == wanted and _same_pixels(cell, row, words):
                    return True
                if len(observed) >= len(wanted):
                    break
    return False


def replace_verified_prose(page: Page, cluster: Cluster, rows: list[ProseRow]) -> None:
    """Replace owned cells and exact, independently confirmed overlapping copies.

    A paragraph crop can include the first row assigned to another text cluster.
    Consume that donor only when its whole text and its physical row are present
    in a verified new row. Identical text printed on another row stays intact.
    """
    assert page.parsed_page is not None and page.predictions.layout is not None
    owned = list(cluster.cells)

    def donors(item: Cluster) -> None:
        if (
            item is not cluster
            and item.label == DocItemLabel.TEXT
            and not item.children
        ):
            owned.extend(cell for cell in item.cells if _covered(cell, rows))
        for child in item.children:
            donors(child)

    for item in page.predictions.layout.clusters:
        donors(item)
    removed = {cell.index for cell in owned}
    line_index = max((cell.index for cell in page.cells), default=-1) + 1
    word_index = (
        max((word.index for word in page.parsed_page.word_cells), default=-1) + 1
    )
    word_ids = {
        word.index
        for word in page.parsed_page.word_cells
        if word.from_ocr
        and _text(word.text)
        and any(
            _text(word.text).replace(" ", "") in _text(cell.text).replace(" ", "")
            and _same_band(
                word.rect,
                cell.rect,
                (word.rect.to_bounding_box().l + word.rect.to_bounding_box().r) / 2,
            )
            and word.rect.to_bounding_box().intersection_over_self(
                cell.rect.to_bounding_box()
            )
            >= 0.7
            for cell in owned
        )
    }
    new_lines, new_words = [], []
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
        cell for cell in page.cells if cell.index not in removed
    ] + new_lines
    page.parsed_page.word_cells = [
        word for word in page.parsed_page.word_cells if word.index not in word_ids
    ] + new_words

    def retain(item: Cluster) -> bool:
        before = len(item.cells)
        item.cells = [cell for cell in item.cells if cell.index not in removed]
        item.children = [child for child in item.children if retain(child)]
        return not (
            before > 0
            and not item.cells
            and not item.children
            and item.label == DocItemLabel.TEXT
        )

    page.predictions.layout.clusters = [
        item for item in page.predictions.layout.clusters if retain(item)
    ]
