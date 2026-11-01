# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Keep printed receipt field captions and their right-aligned amounts together."""

import re
from collections.abc import Iterable

from docling_core.types.doc import BoundingBox
from docling_core.types.doc.page import BoundingRectangle, TextCell

from docling.datamodel.base_models import Cluster, Page


def is_receipt_field_caption(text: str) -> bool:
    # Classify mixed-script captions without changing the recognized text.
    label = text.upper().translate(str.maketrans("ACEHKMOPTXY", "АСЕНКМОРТХУ"))  # noqa: RUF001
    return any(word in label for word in ("ИТОГ", "ОПЛАТ", "НАЛИЧ", "НДС", "СУМ"))  # noqa: RUF001


def is_receipt_page(texts: Iterable[str]) -> bool:
    """Require distinct fiscal cues, including OCR Latin/Cyrillic lookalikes."""
    mapping = str.maketrans("ACEHKMOPTXY", "АСЕНКМОРТХУ")  # noqa: RUF001
    labels = " ".join(texts).upper().translate(mapping)
    cues = ("ИТОГ", "ОПЛАТ", "НАЛИЧ", "НДС", "СУМ", "КАСС")  # noqa: RUF001
    return sum(word in labels for word in cues) >= 2


def join_receipt_fields(page: Page) -> None:
    if page.parsed_page is None or page.size is None:
        return
    lines = [cell for cell in page.parsed_page.textline_cells if cell.from_ocr]
    amounts = [
        cell
        for cell in lines
        if re.fullmatch(r"[=*: ]*-?\d[\d ]*[.,]\d{2}\.?", cell.text)
    ]
    used: set[int] = set()
    merged: dict[int, TextCell] = {}
    for label in sorted(lines, key=lambda cell: cell.rect.to_bounding_box().t):
        if not is_receipt_field_caption(label.text) or re.search(
            r"\d+[.,]\d{2}", label.text
        ):
            continue
        box = label.rect.to_bounding_box()
        candidates = []
        for amount in amounts:
            if amount.index in used:
                continue
            other = amount.rect.to_bounding_box()
            overlap = min(box.b, other.b) - max(box.t, other.t)
            distance = abs((box.t + box.b - other.t - other.b) / 2)
            if (
                other.l >= box.r - min(box.height, other.height) * 0.1
                and other.l - box.r <= page.size.width * 0.35
                and overlap >= min(box.height, other.height) * 0.3
                and distance <= max(box.height, other.height) * 0.6
                and max(box.height, other.height) <= 3 * min(box.height, other.height)
            ):
                candidates.append((distance, amount))
        if not candidates:
            continue
        _, amount = min(candidates, key=lambda item: item[0])
        other = amount.rect.to_bounding_box()
        merged[label.index] = label.model_copy(
            update={
                "text": f"{label.text} {amount.text}",
                "orig": f"{label.orig} {amount.orig}",
                "confidence": min(label.confidence, amount.confidence),
                "rect": BoundingRectangle.from_bounding_box(
                    BoundingBox(
                        l=box.l, t=min(box.t, other.t), r=other.r, b=max(box.b, other.b)
                    )
                ),
            }
        )
        used.add(amount.index)
    if not merged:
        return
    page.parsed_page.textline_cells = [
        merged.get(cell.index, cell)
        for cell in page.parsed_page.textline_cells
        if cell.index not in used
    ]

    def update(cluster: Cluster) -> None:
        cluster.cells = [
            merged.get(cell.index, cell) if cell.from_ocr else cell
            for cell in cluster.cells
            if not cell.from_ocr or cell.index not in used
        ]
        if cluster.cells:
            boxes = [
                cluster.bbox,
                *(cell.rect.to_bounding_box() for cell in cluster.cells),
            ]
            cluster.bbox = BoundingBox(
                l=min(box.l for box in boxes),
                t=min(box.t for box in boxes),
                r=max(box.r for box in boxes),
                b=max(box.b for box in boxes),
            )
        for child in cluster.children:
            update(child)

    if page.predictions.layout is not None:
        for cluster in page.predictions.layout.clusters:
            update(cluster)
