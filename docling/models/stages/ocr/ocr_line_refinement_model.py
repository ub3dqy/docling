# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Refine assigned OCR lines before table tokens are materialized."""

import re
from collections.abc import Iterable

from docling_core.types.doc import DocItemLabel

from docling.datamodel.base_models import InputFormat, Page
from docling.datamodel.document import ConversionResult
from docling.models.base_model import BasePageModel
from docling.models.stages.ocr.ocr_oriented_refinement import refine_image_lines
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
        """A confirmed fiscal payment record is not a document index.

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
            total = re.search(
                r"\bИТОГ(?:О)?\b[^\d]{0,60}\d+[.,:]\d{2}(?!\d)",  # noqa: RUF001 -- Russian total caption
                text,
            )
            # Large or stamped total captions can be unreadable even though
            # the payment, tax and explicit amount separators survive. Do not
            # require that one word to avoid sending a whole receipt through
            # the document-index table decoder. This changes its label only;
            # source text and genuine TABLE clusters remain untouched.
            payment_record = (
                re.search(
                    r"\b(?:БЕЗНАЛИЧНЫМИ|НАЛИЧНЫМИ|ПЛАТ[.]?\s*КАРТОЙ)\b",
                    text,
                )
                and re.search(r"\bНДС\s*\d{1,2}\s*%", text)  # noqa: RUF001
                and len(re.findall(r"=\s*\d+[.,]\d{2}(?!\d)", text)) >= 2
            )
            if (
                is_receipt_page(texts)
                and len(re.findall(r"\d+[.,]\d{2}(?!\d)", text)) >= 3
                and re.search(
                    r"КАСС.{0,12}ЧЕК|ФН\W*\d|ФПД\W*\d|ККТ\W*\d",  # noqa: RUF001 -- Russian fiscal captions
                    text,
                )
                and (total or payment_record)
            ):
                cluster.label = DocItemLabel.TEXT

    def __call__(
        self, conv_res: ConversionResult, page_batch: Iterable[Page]
    ) -> Iterable[Page]:
        for page in page_batch:
            if page._backend is not None and page._backend.is_valid():
                with TimeRecorder(conv_res, "ocr_line_refinement"):
                    if conv_res.input.format == InputFormat.IMAGE:
                        refine_image_lines(self.refiner, page)
                    else:
                        self.refiner.refine_lines(page, image_input=False)
                    self.route_fiscal_text(page)
            yield page
