# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import logging
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
from docling_core.types.doc import BoundingBox, DocItemLabel
from pydantic import AnyUrl, BaseModel, ValidationError

from docling.datamodel.base_models import (
    AssembledUnit,
    Cluster,
    ContainerElement,
    FieldRegionElement,
    FigureElement,
    FormFieldPrediction,
    InputFormat,
    Page,
    PageElement,
    Table,
    TextElement,
)
from docling.datamodel.document import ConversionResult
from docling.models.base_layout_model import (
    CONTAINER_LABELS,
    FIGURE_LABEL,
    PAGE_HEADER_LABELS,
    TABLE_LABELS,
    TEXT_ELEM_LABELS,
)
from docling.models.base_model import BasePageModel
from docling.models.stages.ocr.ocr_region_refiner import OcrRegionRefiner
from docling.utils.ocr_table_grid import table_from_aligned_ocr
from docling.utils.profiling import TimeRecorder
from docling.utils.text_normalization import (
    cell_text_lines,
    normalize_extracted_text,
)

_log = logging.getLogger(__name__)


class PageAssembleOptions(BaseModel):
    recover_ocr_tables: bool = False


class PageAssembleModel(BasePageModel):
    # Minimum fraction of a cluster's area that a hyperlink rect must cover
    # to be considered a match (avoids false positives from adjacent links).
    _HYPERLINK_COVERAGE_THRESHOLD = 0.5

    def __init__(
        self, options: PageAssembleOptions, ocr_refiner: OcrRegionRefiner | None = None
    ):
        self.options = options
        self.ocr_refiner = ocr_refiner

    @staticmethod
    def _match_hyperlink(
        cluster_bbox: BoundingBox,
        page: Page,
    ) -> Optional[Union[AnyUrl, Path]]:
        """Pick the hyperlink annotation with the highest spatial overlap on cluster_bbox.

        Hyperlink rects are BOTTOMLEFT-origin; cluster bboxes are TOPLEFT-origin.
        """
        if page.parsed_page is None or not page.parsed_page.hyperlinks:
            return None

        if page.size is None:
            return None

        page_height = page.size.height

        # Accumulate coverage per URI — a single hyperlink may span multiple
        # annotation rectangles (e.g. a URL that wraps across lines).
        coverage_by_uri: Dict[str, float] = {}

        for hl in page.parsed_page.hyperlinks:
            if hl.uri is None:
                continue

            uri_str = str(hl.uri)
            hl_bbox = hl.rect.to_bounding_box().to_top_left_origin(page_height)
            coverage_by_uri[uri_str] = coverage_by_uri.get(
                uri_str, 0.0
            ) + cluster_bbox.intersection_over_self(hl_bbox)

        if not coverage_by_uri:
            return None

        best_uri = max(coverage_by_uri.items(), key=lambda x: x[1])[0]
        if coverage_by_uri[best_uri] < PageAssembleModel._HYPERLINK_COVERAGE_THRESHOLD:
            return None

        try:
            return AnyUrl(best_uri)
        except ValidationError:
            return Path(best_uri)

    def sanitize_text(self, lines: List[str]) -> str:
        """Join the text lines of a cluster into a single normalized string.

        Lines are joined with a space, except across a hyphen that splits a
        word over the line break ("algo-" / "rithms"), which is dropped so
        that the two halves close up. A hyphen only splits a word when it is
        attached to the word it splits; one that follows whitespace is a
        literal character -- a separator dash, a bullet marker, or a wrapped
        hyphen-prefixed token -- so it is kept and its lines are joined like
        any other line break.

        The joined text is then normalized: a few typographic characters are
        replaced by ASCII equivalents and ligatures are expanded.
        """
        if len(lines) == 0:
            return ""

        for ix, line in enumerate(lines[1:]):
            prev_line = lines[ix]

            if prev_line.endswith("-"):
                prev_words = re.findall(r"\b[\w]+\b", prev_line)
                line_words = re.findall(r"\b[\w]+\b", line)

                hyphen_attached_to_word = len(prev_line) > 1 and prev_line[-2].isalnum()

                if (
                    hyphen_attached_to_word
                    and len(prev_words)
                    and len(line_words)
                    and prev_words[-1].isalnum()
                    and line_words[0].isalnum()
                ):
                    lines[ix] = prev_line[:-1]
                elif not hyphen_attached_to_word:
                    lines[ix] += " "
            else:
                lines[ix] += " "

        sanitized_text = "".join(lines)

        sanitized_text = normalize_extracted_text(sanitized_text)

        return sanitized_text.strip()  # Strip any leading or trailing whitespace

    def _recover_picture_only_ocr(
        self, elements: list[PageElement], page: Page, *, image_input: bool = False
    ) -> list[PageElement]:
        """Expose OCR paragraphs when a photographed page is labelled a picture.

        Keep ordinary figure labels inside their picture. This fallback applies
        only when a page has almost no readable body text or populated table
        and the picture children contain actual OCR output. A short date or
        scanner mark outside a full-page picture must not hide the document.
        """
        if not image_input and any(
            isinstance(element, Table)
            and any(cell.text.strip() for cell in element.table_cells)
            for element in elements
        ):
            return elements
        body_characters = sum(
            len(element.text.strip())
            for element in elements
            if isinstance(element, TextElement)
            and element.label not in PAGE_HEADER_LABELS
        )
        recovered: list[PageElement] = []
        for element in elements:
            if not isinstance(element, FigureElement):
                recovered.append(element)
                continue
            text_children = [
                child
                for child in element.cluster.children
                if child.label in TEXT_ELEM_LABELS
                and any(cell.from_ocr and cell.text.strip() for cell in child.cells)
            ]
            if not text_children:
                recovered.append(element)
                continue
            picture_characters = sum(
                len(cell.text.strip())
                for child in text_children
                for cell in child.cells
            )
            if (
                body_characters
                and not image_input
                and (
                    page.size is None
                    or element.cluster.bbox.area()
                    < 0.75 * page.size.width * page.size.height
                    or picture_characters < 5 * body_characters
                )
            ):
                recovered.append(element)
                continue
            promoted_ids = {child.id for child in text_children}
            picture_cluster = element.cluster.model_copy(
                update={
                    "children": [
                        child
                        for child in element.cluster.children
                        if child.id not in promoted_ids
                    ]
                }
            )
            recovered.append(element.model_copy(update={"cluster": picture_cluster}))
            recovered.extend(
                TextElement(
                    label=child.label,
                    id=child.id,
                    text=self.sanitize_text(cell_text_lines(child.cells)),
                    hyperlink=self._match_hyperlink(child.bbox, page),
                    page_no=page.page_no,
                    cluster=child,
                )
                for child in text_children
            )
        return recovered

    def _recover_ocr_table(
        self, elements: list[PageElement], page: Page
    ) -> list[PageElement]:
        if page.parsed_page is None or any(
            isinstance(element, Table) for element in elements
        ):
            return elements
        if sum(cell.from_ocr for cell in page.parsed_page.textline_cells) < 16:
            return elements
        if not any(word.from_ocr for word in page.parsed_page.word_cells):
            return elements
        cluster_ids = (
            [cluster.id for cluster in page.predictions.layout.clusters]
            if page.predictions.layout is not None
            else []
        )
        cluster_ids.extend(
            child.id for element in elements for child in element.cluster.children
        )
        image = page.get_image(scale=2)
        if image is None:
            return elements
        recovered = table_from_aligned_ocr(
            image,
            page.parsed_page.textline_cells,
            page_no=page.page_no,
            cluster_id=max(
                [element.id for element in elements] + cluster_ids, default=-1
            )
            + 1,
            scale=2,
        )
        if recovered is None:
            return elements
        result: list[PageElement] = []
        for element in elements:
            if not isinstance(element, TextElement):
                result.append(element)
                continue
            cells = [
                cell
                for cell in element.cluster.cells
                if cell.index not in recovered.cell_indices
            ]
            if len(cells) == len(element.cluster.cells):
                result.append(element)
            elif cells:
                result.append(
                    element.model_copy(
                        update={
                            "cluster": element.cluster.model_copy(
                                update={"cells": cells}
                            ),
                            "text": self.sanitize_text(cell_text_lines(cells)),
                        }
                    )
                )
        result.append(recovered.table)
        return result

    def __call__(
        self, conv_res: ConversionResult, page_batch: Iterable[Page]
    ) -> Iterable[Page]:
        for page in page_batch:
            assert page._backend is not None
            if not page._backend.is_valid():
                yield page
            else:
                with TimeRecorder(conv_res, "page_assemble"):
                    assert page.predictions.layout is not None
                    if self.ocr_refiner is not None:
                        self.ocr_refiner.refine_lines(
                            page, image_input=conv_res.input.format == InputFormat.IMAGE
                        )

                    # assembles some JSON output page by page.

                    elements: List[PageElement] = []
                    headers: List[PageElement] = []
                    body: List[PageElement] = []
                    form_fields = page.predictions.form_fields or FormFieldPrediction()

                    for cluster in page.predictions.layout.clusters:
                        # _log.info("Cluster label seen:", cluster.label)
                        if cluster.label in TEXT_ELEM_LABELS:
                            text = self.sanitize_text(cell_text_lines(cluster.cells))
                            hyperlink = self._match_hyperlink(cluster.bbox, page)
                            # A paragraph that inlines AcroForm widgets carries a
                            # single keyed field item; attach it so it materializes
                            # as a field_item in place (see TextElement.field_item)
                            # instead of being pulled into a separate field_region.
                            field_item = form_fields.inline_items.get(cluster.id)
                            text_el = TextElement(
                                label=cluster.label,
                                id=cluster.id,
                                text=text,
                                hyperlink=hyperlink,
                                page_no=page.page_no,
                                cluster=cluster,
                                field_item=field_item,
                            )
                            elements.append(text_el)

                            if cluster.label in PAGE_HEADER_LABELS:
                                headers.append(text_el)
                            else:
                                body.append(text_el)
                        elif cluster.label in TABLE_LABELS:
                            tbl = None
                            if page.predictions.tablestructure:
                                tbl = page.predictions.tablestructure.table_map.get(
                                    cluster.id, None
                                )
                            if not tbl:  # fallback: add table without structure, if it isn't present
                                tbl = Table(
                                    label=cluster.label,
                                    id=cluster.id,
                                    text="",
                                    otsl_seq=[],
                                    table_cells=[],
                                    cluster=cluster,
                                    page_no=page.page_no,
                                )
                            elements.append(tbl)
                            body.append(tbl)
                        elif cluster.label == FIGURE_LABEL:
                            fig = None
                            if page.predictions.figures_classification:
                                fig = page.predictions.figures_classification.figure_map.get(
                                    cluster.id, None
                                )
                            if not fig:  # fallback: add figure without classification, if it isn't present
                                fig = FigureElement(
                                    label=cluster.label,
                                    id=cluster.id,
                                    text="",
                                    data=None,
                                    cluster=cluster,
                                    page_no=page.page_no,
                                )
                            elements.append(fig)
                            body.append(fig)
                        elif cluster.label in CONTAINER_LABELS:
                            field_items = form_fields.form_items.get(cluster.id)
                            if field_items is None:
                                container_el = ContainerElement(
                                    label=cluster.label,
                                    id=cluster.id,
                                    page_no=page.page_no,
                                    cluster=cluster,
                                )
                            else:
                                container_el = FieldRegionElement(
                                    label=DocItemLabel.FIELD_REGION,
                                    id=cluster.id,
                                    page_no=page.page_no,
                                    cluster=cluster,
                                    items=field_items,
                                )
                            elements.append(container_el)
                            body.append(container_el)

                    elements = self._recover_picture_only_ocr(
                        elements,
                        page,
                        image_input=conv_res.input.format == InputFormat.IMAGE,
                    )
                    if self.options.recover_ocr_tables:
                        elements = self._recover_ocr_table(elements, page)
                    if self.ocr_refiner is not None:
                        self.ocr_refiner.refine_tables(
                            page,
                            elements,
                            image_input=conv_res.input.format == InputFormat.IMAGE,
                        )
                    headers = [
                        element
                        for element in elements
                        if element.label in PAGE_HEADER_LABELS
                    ]
                    body = [
                        element
                        for element in elements
                        if element.label not in PAGE_HEADER_LABELS
                    ]
                    page.assembled = AssembledUnit(
                        elements=elements, headers=headers, body=body
                    )

                yield page
