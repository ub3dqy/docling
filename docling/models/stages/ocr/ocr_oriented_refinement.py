# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Refine rotated image text in its reading frame, then restore provenance."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from math import atan2, degrees
from typing import TYPE_CHECKING

from docling_core.types.doc import BoundingBox, CoordOrigin, Size
from docling_core.types.doc.page import (
    BoundingRectangle,
    PdfPageBoundaryType,
    PdfPageGeometry,
    SegmentedPdfPage,
    TextCell,
)
from PIL import Image
from pydantic import PrivateAttr
from typing_extensions import override

from docling.datamodel.base_models import Cluster, Page, PagePredictions

if TYPE_CHECKING:
    from docling.models.stages.ocr.ocr_region_refiner import OcrRegionRefiner


def _reading_turn(page: Page) -> int:
    if page.parsed_page is None:
        return 0
    cells = [*page.cells, *page.parsed_page.word_cells, *page.parsed_page.char_cells]
    if any(
        not cell.from_ocr or cell.rect.coord_origin != CoordOrigin.TOPLEFT
        for cell in cells
    ):
        return 0
    labels = [cell for cell in page.cells if sum(c.isalpha() for c in cell.text) >= 4]
    if len(labels) < 5:
        return 0
    turns: Counter[int] = Counter()
    for cell in labels:
        rect = cell.rect
        dx, dy = rect.r_x2 - rect.r_x3, rect.r_y2 - rect.r_y3
        if dx == 0 and dy == 0:
            continue
        angle = degrees(atan2(dy, dx)) % 360
        quarter = round(angle / 90) * 90 % 360
        if abs((angle - quarter + 180) % 360 - 180) <= 15:
            turns[quarter] += 1
    if not turns:
        return 0
    turn, count = turns.most_common(1)[0]
    return turn if count >= len(labels) * 0.8 else 0


def _point(x: float, y: float, *, angle: int, size: Size) -> tuple[float, float]:
    if angle == 90:
        return y, size.width - x
    if angle == 180:
        return size.width - x, size.height - y
    if angle == 270:
        return size.height - y, x
    raise ValueError("A refinement turn must be 90, 180 or 270 degrees.")


def _turn_rect(rect: BoundingRectangle, *, angle: int, size: Size) -> BoundingRectangle:
    points = [
        _point(x, y, angle=angle, size=size)
        for x, y in [
            (rect.r_x0, rect.r_y0),
            (rect.r_x1, rect.r_y1),
            (rect.r_x2, rect.r_y2),
            (rect.r_x3, rect.r_y3),
        ]
    ]
    return BoundingRectangle(
        r_x0=points[0][0],
        r_y0=points[0][1],
        r_x1=points[1][0],
        r_y1=points[1][1],
        r_x2=points[2][0],
        r_y2=points[2][1],
        r_x3=points[3][0],
        r_y3=points[3][1],
        coord_origin=CoordOrigin.TOPLEFT,
    )


def _turn_text(page: Page, *, angle: int, size: Size) -> None:
    assert page.parsed_page is not None
    seen: set[int] = set()
    seen_clusters: set[int] = set()

    def cell(text: TextCell) -> None:
        if id(text) not in seen:
            seen.add(id(text))
            text.rect = _turn_rect(text.rect, angle=angle, size=size)

    def cluster(item: Cluster) -> None:
        if id(item) in seen_clusters:
            return
        seen_clusters.add(id(item))
        item.bbox = _turn_rect(
            BoundingRectangle.from_bounding_box(
                item.bbox.to_top_left_origin(size.height)
            ),
            angle=angle,
            size=size,
        ).to_bounding_box()
        for text in item.cells:
            cell(text)
        for child in item.children:
            cluster(child)

    for text in [
        *page.parsed_page.char_cells,
        *page.parsed_page.word_cells,
        *page.cells,
    ]:
        cell(text)
    if page.predictions.layout is not None:
        for item in page.predictions.layout.clusters:
            cluster(item)


class _RefinementPage(Page):
    """An OCR-only working view; source images and non-text data stay unchanged."""

    _source: Page | None = PrivateAttr(default=None)
    _angle: int = PrivateAttr(default=0)

    @override
    def get_image(
        self,
        scale: float = 1.0,
        max_size: int | None = None,
        cropbox: BoundingBox | None = None,
    ) -> Image.Image | None:
        assert self._source is not None and self.size is not None
        if max_size is not None and max_size > 0:
            scale = min(scale, max_size / max(self.size.width, self.size.height))
        image = self._source.get_image(scale=scale)
        if image is None:
            return None
        image = image.rotate(self._angle, expand=True)
        if cropbox is not None:
            image = image.crop(
                cropbox.to_top_left_origin(self.size.height).scaled(scale).as_tuple()
            )
        return image


def refine_image_lines(refiner: OcrRegionRefiner, page: Page) -> None:
    """Use an accepted OCR reading direction, without rotating the source page.

    The OCR quadrilateral retains its reading axis after source-coordinate
    mapping. Require a clear, shared direction across several text labels.
    Run all existing refinement rules on a separate upright text/layout view.
    Commit its text only after success and map every resulting box back.
    """
    angle = _reading_turn(page)
    if angle == 0 or page.size is None or page.parsed_page is None:
        refiner.refine_lines(page, image_input=True)
        return
    source = page.parsed_page
    chars, words, lines, layout = deepcopy(
        (
            source.char_cells,
            source.word_cells,
            source.textline_cells,
            page.predictions.layout,
        )
    )
    size = (
        page.size.model_copy()
        if angle == 180
        else Size(width=page.size.height, height=page.size.width)
    )
    box = BoundingBox(l=0, t=0, r=size.width, b=size.height)
    dimension = PdfPageGeometry(
        angle=0,
        rect=BoundingRectangle.from_bounding_box(box),
        boundary_type=PdfPageBoundaryType.CROP_BOX,
        art_bbox=box,
        bleed_bbox=box,
        crop_bbox=box,
        media_bbox=box,
        trim_bbox=box,
    )
    view = _RefinementPage(
        page_no=page.page_no,
        size=size,
        parsed_page=SegmentedPdfPage(
            dimension=dimension,
            char_cells=chars,
            word_cells=words,
            textline_cells=lines,
            has_chars=source.has_chars,
            has_words=source.has_words,
            has_lines=source.has_lines,
        ),
        predictions=PagePredictions(layout=layout),
    )
    view._source, view._angle = page, angle
    _turn_text(view, angle=angle, size=page.size)
    refiner.refine_lines(view, image_input=True)
    _turn_text(view, angle=360 - angle, size=size)
    assert view.parsed_page is not None
    source.char_cells = view.parsed_page.char_cells
    source.word_cells = view.parsed_page.word_cells
    source.textline_cells = view.parsed_page.textline_cells
    source.has_chars = view.parsed_page.has_chars
    source.has_words = view.parsed_page.has_words
    source.has_lines = view.parsed_page.has_lines
    page.predictions.layout = view.predictions.layout
