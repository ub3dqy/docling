# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""OCR text on a photographed page must survive ordinary document export."""

from pathlib import PurePath

import pytest
from docling_core.types.doc import BoundingBox, DocItemLabel, Size
from docling_core.types.doc.page import BoundingRectangle, TextCell

from docling.datamodel.base_models import (
    AssembledUnit,
    Cluster,
    FigureElement,
    InputFormat,
    Page,
    PageElement,
    TextElement,
)
from docling.datamodel.document import ConversionResult, InputDocument
from docling.models.stages.page_assemble.page_assemble_model import (
    PageAssembleModel,
    PageAssembleOptions,
)
from docling.models.stages.reading_order.readingorder_model import (
    ReadingOrderModel,
    ReadingOrderOptions,
)


def _picture(*, from_ocr: bool) -> FigureElement:
    children = []
    for index, text in enumerate(["ИТОГО К ОПЛАТЕ", "170.97"], start=2):  # noqa: RUF001
        bbox = BoundingBox(l=30, t=index * 30, r=230, b=index * 30 + 20)
        children.append(
            Cluster(
                id=index,
                label=DocItemLabel.TEXT,
                bbox=bbox,
                cells=[
                    TextCell(
                        index=index,
                        text=text,
                        orig=text,
                        from_ocr=from_ocr,
                        rect=BoundingRectangle.from_bounding_box(bbox),
                    )
                ],
            )
        )
    cluster = Cluster(
        id=1,
        label=DocItemLabel.PICTURE,
        bbox=BoundingBox(l=0, t=0, r=300, b=400),
        children=children,
    )
    return FigureElement(id=1, label=cluster.label, page_no=1, cluster=cluster)


@pytest.mark.parametrize("outside_text", ["", "06.11."])
def test_picture_only_ocr_is_available_once_in_text_and_markdown(
    outside_text: str,
) -> None:
    picture = _picture(from_ocr=True)
    page = Page(page_no=1, size=Size(width=300, height=400))
    source: list[PageElement] = [picture]
    if outside_text:
        cluster = Cluster(
            id=4, label=DocItemLabel.TEXT, bbox=BoundingBox(l=10, t=10, r=60, b=25)
        )
        source.append(
            TextElement(
                id=4, label=cluster.label, text=outside_text, page_no=1, cluster=cluster
            )
        )
    elements = PageAssembleModel(PageAssembleOptions())._recover_picture_only_ocr(
        source,
        page,
        image_input=True,
    )
    conv_res = ConversionResult(
        input=InputDocument.model_construct(
            file=PurePath("receipt.jpg"),
            document_hash="0" * 64,
            valid=True,
            format=InputFormat.IMAGE,
        ),
        pages=[page],
        assembled=AssembledUnit(elements=elements, body=elements),
    )
    document = ReadingOrderModel(ReadingOrderOptions())(conv_res)
    for exported in (document.export_to_text(), document.export_to_markdown()):
        assert exported.count("ИТОГО К ОПЛАТЕ") == 1  # noqa: RUF001
        assert exported.count("170.97") == 1
    assert not document.pictures[0].children
    # Do not mutate the earlier stage's clusters or lose the source picture.
    assert len(picture.cluster.children) == 2
    assert len(document.pictures) == 1


@pytest.mark.parametrize("from_ocr,with_body", [(False, False), (True, True)])
def test_ordinary_figure_children_stay_inside_the_picture(
    from_ocr: bool, with_body: bool
) -> None:
    picture = _picture(from_ocr=from_ocr)
    elements: list[PageElement] = [picture]
    if with_body:
        cluster = Cluster(
            id=4,
            label=DocItemLabel.TEXT,
            bbox=BoundingBox(l=10, t=410, r=290, b=430),
        )
        elements.append(
            TextElement(
                id=4,
                label=cluster.label,
                page_no=1,
                cluster=cluster,
                text="Ordinary page with a figure",
            )
        )
    result = PageAssembleModel(PageAssembleOptions())._recover_picture_only_ocr(
        elements, Page(page_no=1, size=Size(width=300, height=450))
    )
    assert result == elements
    assert len(result[0].cluster.children) == 2
