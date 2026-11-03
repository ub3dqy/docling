# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Refine assigned OCR lines before table tokens are materialized."""

import re
from collections.abc import Iterable

from docling_core.types.doc import DocItemLabel

from docling.datamodel.base_models import InputFormat, Page
from docling.datamodel.document import ConversionResult
from docling.models.base_model import BasePageModel
from docling.models.stages.ocr.ocr_region_refiner import (
    OcrRegionRefiner,
    normalize_mixed_script,
)
from docling.utils.ocr_receipt_rows import is_receipt_page
from docling.utils.profiling import TimeRecorder


class OcrLineRefinementModel(BasePageModel):
    def __init__(self, refiner: OcrRegionRefiner) -> None:
        self.refiner = refiner

    @staticmethod
    def route_fiscal_text(page: Page) -> None:
        """A fiscal receipt with a printed total is not a document index.

        Use the block's observed content, rather than its filename or a template.
        Genuine tables retain table extraction and its source-coverage safeguard.
        """
        if page.predictions.layout is None:
            return
        for cluster in page.predictions.layout.clusters:
            if cluster.label != DocItemLabel.DOCUMENT_INDEX:
                continue
            texts = [
                normalize_mixed_script(cell.text).upper()
                for cell in cluster.cells
                if cell.from_ocr
            ]
            text = " ".join(texts)
            if (
                is_receipt_page(texts)
                and len(re.findall(r"\d+[.,]\d{2}(?!\d)", text)) >= 3
                and re.search(
                    r"КАСС.{0,12}ЧЕК|ФН\W*\d|ФПД\W*\d|ККТ\W*\d",  # noqa: RUF001 -- Russian fiscal captions
                    text,
                )
                and re.search(
                    r"\bИТОГ(?:О)?\b[^\d]{0,60}\d+[.,:]\d{2}(?!\d)",  # noqa: RUF001 -- Russian total caption
                    text,
                )
            ):
                cluster.label = DocItemLabel.TEXT

    def __call__(
        self, conv_res: ConversionResult, page_batch: Iterable[Page]
    ) -> Iterable[Page]:
        for page in page_batch:
            if page._backend is not None and page._backend.is_valid():
                with TimeRecorder(conv_res, "ocr_line_refinement"):
                    self.refiner.refine_lines(
                        page, image_input=conv_res.input.format == InputFormat.IMAGE
                    )
                    self.route_fiscal_text(page)
            yield page
