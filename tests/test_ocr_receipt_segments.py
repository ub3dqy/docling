# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT
# ruff: noqa: RUF001 -- Mixed-script OCR evidence is intentional.

"""Separate neighboring papers without losing numbers or native text."""

from docling_core.types.doc import BoundingBox, DocItemLabel, Size
from docling_core.types.doc.page import BoundingRectangle, SegmentedPdfPage, TextCell

from docling.datamodel.base_models import Cluster, LayoutPrediction, Page
from docling.utils.ocr_receipt_segments import split_receipt_lines


def _cell(index, text, left, right, *, native=False):
    return TextCell(
        index=index,
        text=text,
        orig=text,
        from_ocr=not native,
        confidence=0.95,
        rect=BoundingRectangle.from_bounding_box(
            BoundingBox(l=left, t=20, r=right, b=45)
        ),
    )


def _page(lines, words):
    page = Page(page_no=1, size=Size(width=600, height=800))
    page.parsed_page = SegmentedPdfPage.model_construct(
        textline_cells=lines,
        word_cells=words,
        char_cells=[],
    )
    page.predictions.layout = LayoutPrediction(
        clusters=[
            Cluster(
                id=0,
                label=DocItemLabel.TEXT,
                bbox=BoundingBox(l=10, t=10, r=590, b=60),
                cells=list(lines),
            )
        ]
    )
    return page


def test_company_and_bank_on_neighboring_papers_keep_separate_source_regions():
    words = [
        _cell(0, "о00", 20, 60),
        _cell(1, '"Первая', 70, 140),
        _cell(2, 'компания"', 150, 245),
        _cell(3, "БАНК", 350, 420),
    ]
    native = _cell(0, "Native text", 450, 590, native=True)
    page = _page([_cell(0, 'о00 "Первая компания" БАНК', 20, 420), native], words)
    split_receipt_lines(page)
    assert page.parsed_page is not None
    assert [c.text for c in page.cells] == [
        "о00",
        '"Первая компания"',
        "БАНК",
        "Native text",
    ]
    assert page.cells[-1] is native
    assert page.predictions.layout is not None
    assert page.predictions.layout.clusters[0].cells[-1] is native
    assert page.cells[1].rect.to_bounding_box().r == 245
    split_receipt_lines(page)
    assert len(page.cells) == 4


def test_split_does_not_append_neighboring_bank_digits_to_the_amount():
    words = [_cell(0, "513.88", 100, 170), _cell(1, "77ID", 200, 245)]
    page = _page([_cell(0, "513.8877ID", 100, 245)], words)
    split_receipt_lines(page)
    assert [c.text for c in page.cells] == ["513.88", "77ID"]
    assert page.cells[0].rect.to_bounding_box().r == 170


def test_partial_word_output_cannot_remove_an_unrepresented_part_of_a_line():
    page = _page([_cell(0, "513.8877ID", 100, 245)], [_cell(0, "513.88", 100, 170)])
    split_receipt_lines(page)
    assert [c.text for c in page.cells] == ["513.8877ID"]


def test_sparse_ctc_alignment_keeps_the_whole_sentence_for_source_rereading():
    words = [
        _cell(0, "1", 120, 126),
        _cell(1, "0U", 190, 202),
        _cell(2, "09", 250, 262),
    ]
    page = _page([_cell(0, "10U09", 120, 410)], words)
    split_receipt_lines(page)
    assert [c.text for c in page.cells] == ["10U09"]
    assert page.cells[0].rect.to_bounding_box().r == 410
