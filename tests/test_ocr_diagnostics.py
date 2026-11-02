# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Keep stage diagnostics independent from later mutations and final exports."""

from docling_core.types.doc import BoundingBox, DocItemLabel, Size
from docling_core.types.doc.page import BoundingRectangle, TextCell

from docling.datamodel.base_models import Cluster, LayoutPrediction, Page
from docling.models.base_ocr_model import _empty_segmented_page
from scripts.diagnose_russian_ocr import Snapshot, Trace, snapshot_score
from scripts.qualify_real_russian_ocr import Case, FieldReference, Line


def test_stage_snapshot_retains_cells_before_the_shared_cluster_is_modified() -> None:
    page = Page(page_no=1, size=Size(width=100, height=200))
    page.parsed_page = _empty_segmented_page(page)
    cell = TextCell(
        index=0,
        text="НДС 24.15",
        orig="НДС 24.15",
        confidence=0.8,
        from_ocr=True,
        rect=BoundingRectangle.from_bounding_box(BoundingBox(l=10, t=100, r=90, b=120)),
    )
    page.parsed_page.textline_cells = [cell]
    page.predictions.layout = LayoutPrediction(
        clusters=[
            Cluster(
                id=0,
                label=DocItemLabel.TEXT,
                bbox=BoundingBox(l=0, t=0, r=100, b=200),
                cells=[cell],
            )
        ]
    )
    trace = Trace()
    before = trace.page("before", page)
    cell.text = "НДС -24.15"
    page.predictions.layout.clusters[0].cells.clear()
    page.parsed_page.textline_cells.clear()
    trace.page("after", page)
    assert before.lines[0].text == "НДС 24.15"
    assert before.clusters[0].cells[0].text == "НДС 24.15"
    assert before.lines[0].bbox == (0.1, 0.5, 0.9, 0.6)
    assert trace.events[1].lines == []
    assert trace.events[1].clusters[0].cells == []


def test_diagnostic_distinguishes_a_reading_failure_from_a_later_export_loss() -> None:
    case = Case(
        id="receipt",
        file="receipt.jpg",
        page=1,
        kind="receipt",
        source_url="https://example.org/receipt.jpg",
        fields=[FieldReference(name="vat", roi=(0, 0, 1, 1), value="24.15")],
        texts=[],
        rows=[],
    )
    correct = Snapshot(
        stage="ocr",
        lines=[
            Line(
                text="НДС =24.15",
                confidence=0.9,
                bbox=(0.1, 0.2, 0.9, 0.3),
            )
        ],
    )
    broken = Snapshot(
        stage="later",
        lines=[
            Line(
                text="НДС -24.15",
                confidence=0.99,
                bbox=(0.1, 0.2, 0.9, 0.3),
            )
        ],
    )
    lost = snapshot_score(case, correct, "Итог 507.00").fields[0]
    misread = snapshot_score(case, broken, "Карта 24.15").fields[0]
    assert lost.roi_number_correct
    assert not lost.available_in_export
    assert not misread.roi_number_correct
    assert not misread.passed
