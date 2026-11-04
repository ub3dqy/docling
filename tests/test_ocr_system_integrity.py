# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Preserve source readings across selection and table assembly boundaries."""

from unittest.mock import MagicMock

import pytest
from docling_core.types.doc import BoundingBox, DocItemLabel, TableCell
from docling_core.types.doc.page import BoundingRectangle, TextCell

from docling.datamodel.base_models import (
    Cluster,
    InputFormat,
    LayoutPrediction,
    Page,
    Table,
    TextElement,
)
from docling.datamodel.document import ConversionResult
from docling.models.stages.ocr.ocr_line_refinement_model import OcrLineRefinementModel
from docling.models.stages.page_assemble.page_assemble_model import (
    PageAssembleModel,
    PageAssembleOptions,
)
from docling.models.stages.table_structure.table_structure_model import (
    _words_and_ocr_cells,
)
from docling.utils.ocr_reading_choice import reading_may_replace
from docling.utils.ocr_table_content import record_ocr_table_coverage


def cell(
    text: str, *, index: int, left: float, top: float, right: float, bottom: float
) -> TextCell:
    return TextCell(
        text=text,
        orig=text,
        index=index,
        from_ocr=True,
        confidence=0.9,
        rect=BoundingRectangle.from_bounding_box(
            BoundingBox(l=left, t=top, r=right, b=bottom)
        ),
    )


def table_for(source: list[TextCell], values: list[str]) -> Table:
    cluster = Cluster(
        id=4,
        label=DocItemLabel.TABLE,
        bbox=BoundingBox(l=0, t=0, r=200, b=200),
        cells=source,
    )
    return Table(
        id=4,
        label=DocItemLabel.TABLE,
        cluster=cluster,
        page_no=1,
        otsl_seq=[],
        num_cols=1,
        num_rows=len(values),
        table_cells=[
            TableCell(
                text=text,
                start_row_offset_idx=index,
                end_row_offset_idx=index + 1,
                start_col_offset_idx=0,
                end_col_offset_idx=1,
                bbox=BoundingBox(l=0, t=0, r=200, b=200),
            )
            for index, text in enumerate(values)
        ],
    )


def box(left: float, right: float) -> list[list[float]]:
    return [[left, 10], [right, 10], [right, 30], [left, 30]]


def test_weak_mixed_reading_does_not_block_a_better_complete_row() -> None:
    original = (box(0, 200), "ab03X75472", 0.45)
    candidate = (box(0, 200), "Товар 87.40*2 =174.80", 0.75)
    assert reading_may_replace([(original, [])], candidate)


@pytest.mark.parametrize("confidence,preserve_numbers", [(0.99, True), (0.75, False)])
@pytest.mark.parametrize(
    "before,after", [("49.20", "49.70"), ("10%", "20%"), ("+49.20", "-49.20")]
)
def test_confident_numeric_word_survives_a_weak_surrounding_caption(
    confidence: float, preserve_numbers: bool, before: str, after: str
) -> None:
    original = (box(0, 200), f"Damaged label {before}", 0.45)
    amount = (box(150, 200), before, 0.99)
    candidate = (box(0, 200), f"Corrected label {after}", confidence)
    assert not reading_may_replace(
        [(original, [amount])], candidate, preserve_numbers=preserve_numbers
    )


def test_better_right_hand_amount_cannot_erase_the_left_side_of_a_row() -> None:
    original = (box(0, 200), "Damaged whole row", 0.4)
    candidate = (box(150, 200), "An apparently confident replacement", 0.99)
    assert not reading_may_replace([(original, [])], candidate)


def test_partial_word_output_cannot_hide_the_left_side_of_a_source_row() -> None:
    original = (box(0, 200), "Damaged whole row", 0.4)
    only_right_word = (box(150, 200), "row", 0.9)
    candidate = (box(150, 200), "An apparently confident replacement", 0.99)
    assert not reading_may_replace([(original, [only_right_word])], candidate)


def test_out_of_frame_word_box_cannot_hide_the_source_caption() -> None:
    original = (box(0, 200), "Damaged whole row", 0.4)
    misplaced_word = (box(250, 300), "Damaged whole row", 0.9)
    candidate = (box(150, 200), "An apparently confident replacement", 0.99)
    assert not reading_may_replace([(original, [misplaced_word])], candidate)


def test_incomplete_weak_decimal_does_not_lock_out_the_whole_amount() -> None:
    original = (box(0, 200), ".00", 0.42)
    candidate = (box(0, 200), "12.00", 0.99)
    assert reading_may_replace([(original, [])], candidate)


def test_incomplete_word_is_not_a_confident_complete_numeric_value() -> None:
    original = (box(0, 200), ".00", 0.42)
    incomplete_word = (box(0, 200), ".00", 0.99)
    candidate = (box(0, 200), "12.00", 0.99)
    assert reading_may_replace([(original, [incomplete_word])], candidate)


def test_fiscal_total_can_be_in_a_separate_ocr_box() -> None:
    texts = ["Кассовый чек", "58.40*2", "=116.80", "ИТОГ", "=116.80", "НДС 5%"]
    source = [
        cell(
            text, index=index, left=0, top=index * 20, right=200, bottom=index * 20 + 15
        )
        for index, text in enumerate(texts)
    ]
    cluster = Cluster(
        id=0,
        label=DocItemLabel.DOCUMENT_INDEX,
        bbox=BoundingBox(l=0, t=0, r=200, b=200),
        cells=source,
    )
    page = Page(page_no=1)
    page.predictions.layout = LayoutPrediction(clusters=[cluster])
    OcrLineRefinementModel.route_fiscal_text(page)
    assert cluster.label == DocItemLabel.TEXT


def test_financial_contents_is_not_relabelled_as_a_receipt() -> None:
    source = [
        cell(
            text, index=index, left=0, top=index * 20, right=200, bottom=index * 20 + 15
        )
        for index, text in enumerate(
            ["Оглавление", "Учет НДС 10.00", "Суммы 20.00", "Итоги 30.00"]
        )
    ]
    cluster = Cluster(
        id=0,
        label=DocItemLabel.DOCUMENT_INDEX,
        bbox=BoundingBox(l=0, t=0, r=200, b=200),
        cells=source,
    )
    page = Page(page_no=1)
    page.predictions.layout = LayoutPrediction(clusters=[cluster])
    OcrLineRefinementModel.route_fiscal_text(page)
    assert cluster.label == DocItemLabel.DOCUMENT_INDEX


@pytest.mark.parametrize(
    "label,native,lines,routed",
    [
        (
            DocItemLabel.DOCUMENT_INDEX,
            False,
            [
                "Кассовый чек",
                "58.40*2",
                "=116.80",
                "ЯОГ",
                "=116.80",
                "БЕЗНАЛИЧНЫМИ",
                "СУММА НДС 5% =5.56",  # noqa: RUF001 -- Russian fiscal caption
            ],
            True,
        ),
        (
            DocItemLabel.DOCUMENT_INDEX,
            False,
            ["Кассовый чек", "58.40*2", "116.80", "БЕЗНАЛИЧНЫМИ", "СУММА НДС 5% 5.56"],  # noqa: RUF001 -- Russian fiscal caption
            False,
        ),
        (
            DocItemLabel.DOCUMENT_INDEX,
            False,
            ["Кассовый чек", "=116.80", "=116.80", "СУММА НДС 5% =5.56"],  # noqa: RUF001 -- Russian fiscal caption
            False,
        ),
        (
            DocItemLabel.DOCUMENT_INDEX,
            False,
            ["Оглавление", "=116.80", "=116.80", "БЕЗНАЛИЧНЫМИ", "СУММА НДС 5% =5.56"],  # noqa: RUF001 -- Russian fiscal caption
            False,
        ),
        (
            DocItemLabel.TABLE,
            False,
            [
                "Кассовый чек",
                "=116.80",
                "=116.80",
                "БЕЗНАЛИЧНЫМИ",
                "СУММА НДС 5% =5.56",  # noqa: RUF001 -- Russian fiscal caption
            ],
            False,
        ),
        (
            DocItemLabel.DOCUMENT_INDEX,
            True,
            [
                "Кассовый чек",
                "=116.80",
                "=116.80",
                "БЕЗНАЛИЧНЫМИ",
                "СУММА НДС 5% =5.56",  # noqa: RUF001 -- Russian fiscal caption
            ],
            False,
        ),
    ],
)
def test_payment_and_tax_source_evidence_survives_an_unreadable_total_caption(
    label: DocItemLabel, native: bool, lines: list[str], routed: bool
) -> None:
    source = [
        cell(
            text, index=index, left=0, top=index * 20, right=250, bottom=index * 20 + 15
        )
        for index, text in enumerate(lines)
    ]
    for item in source:
        item.from_ocr = not native
    cluster = Cluster(
        id=0, label=label, bbox=BoundingBox(l=0, t=0, r=250, b=200), cells=source
    )
    page = Page(page_no=1)
    page.predictions.layout = LayoutPrediction(clusters=[cluster])
    OcrLineRefinementModel.route_fiscal_text(page)
    assert cluster.label == (DocItemLabel.TEXT if routed else label)
    assert [item.text for item in cluster.cells] == lines


def test_refined_line_replaces_stale_words_before_table_matching() -> None:
    line = cell("Кассовый чек", index=0, left=0, top=0, right=200, bottom=20)
    old_words = [
        cell("Кассовы", index=0, left=0, top=0, right=130, bottom=20),
        cell("чек", index=1, left=140, top=0, right=200, bottom=20),
    ]
    assert _words_and_ocr_cells(old_words, [line]) == [line]


@pytest.mark.parametrize("scrambled_clusters", [False, True])
def test_refined_line_keeps_its_place_between_original_word_tokens(
    scrambled_clusters: bool,
) -> None:
    lines = [
        cell(text, index=row, left=0, top=row * 30, right=120, bottom=row * 30 + 20)
        for row, text in enumerate(["First row", "Fixed middle", "Last row"])
    ]
    words = [
        cell(text, index=0, left=left, top=row * 30, right=right, bottom=row * 30 + 20)
        for row, text, left, right in [
            (0, "First", 0, 40),
            (0, "row", 50, 80),
            (1, "Wrong", 0, 50),
            (1, "middle", 60, 110),
            (2, "Last", 0, 40),
            (2, "row", 50, 80),
        ]
    ]
    clusters = [lines[2], lines[0], lines[1]] if scrambled_clusters else lines
    assert [c.text for c in _words_and_ocr_cells(words, clusters)] == [
        "First",
        "row",
        "Fixed middle",
        "Last",
        "row",
    ]


def test_matching_ids_are_input_positions_even_when_cell_indices_overlap() -> None:
    source = [
        cell("39.50", index=0, left=0, top=10, right=40, bottom=30),
        cell("39.50", index=0, left=50, top=10, right=90, bottom=30),
    ]
    table = table_for(source, ["39.50"])
    record_ocr_table_coverage(table, source, matched_token_ids={0})
    assert table.unmatched_ocr_cells == [source[1]]
    assembler = PageAssembleModel(PageAssembleOptions())
    retained, recovered = assembler._recover_unmatched_table_text(table, 10)
    assert retained is None
    assert recovered is not None and recovered.text.count("39.50") == 2


def test_small_unmatched_tail_is_preserved_beside_a_populated_table() -> None:
    source = [
        cell(
            str(100 + index),
            index=index,
            left=0,
            top=index * 10,
            right=40,
            bottom=index * 10 + 8,
        )
        for index in range(10)
    ]
    table = table_for(source, [word.text for word in source[:-1]])
    record_ocr_table_coverage(table, source, matched_token_ids=set(range(9)))
    retained, recovered = PageAssembleModel(
        PageAssembleOptions()
    )._recover_unmatched_table_text(table, 20)
    assert retained is table
    assert recovered is not None and recovered.text == "109"
    assert recovered.id != table.id


def test_one_unplaced_ocr_token_does_not_replace_a_native_text_table() -> None:
    native = [
        cell(
            str(100 + index),
            index=index,
            left=0,
            top=index * 10,
            right=40,
            bottom=index * 10 + 8,
        ).model_copy(update={"from_ocr": False})
        for index in range(10)
    ]
    extra = cell("39.50", index=10, left=0, top=110, right=40, bottom=118)
    table = table_for([*native, extra], [c.text for c in native])
    record_ocr_table_coverage(table, [*native, extra], matched_token_ids=set(range(10)))
    retained, recovered = PageAssembleModel(
        PageAssembleOptions()
    )._recover_unmatched_table_text(table, 20)
    assert retained is table
    assert recovered is not None and recovered.text == "39.50"


def test_missing_table_prediction_keeps_its_ocr_source_in_the_assembled_page() -> None:
    source = [cell("Product 39.50", index=0, left=0, top=10, right=140, bottom=30)]
    cluster = table_for(source, []).cluster
    page = Page(page_no=1)
    page.predictions.layout = LayoutPrediction(clusters=[cluster])
    page._backend = MagicMock()
    page._backend.is_valid.return_value = True
    conversion = MagicMock(spec=ConversionResult)
    conversion.input = MagicMock(format=InputFormat.PDF)
    conversion.timings = {}
    pages = list(PageAssembleModel(PageAssembleOptions())(conversion, [page]))
    assert pages[0].assembled is not None
    elements = pages[0].assembled.elements
    assert len(elements) == 1 and isinstance(elements[0], TextElement)
    assert elements[0].text == "Product 39.50"
