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
    return any(
        word in label
        for word in ("ИТОГ", "ОПЛАТ", "НАЛИЧ", "НДС", "СУМ")  # noqa: RUF001
    ) or bool(
        re.match(r"\W*ПЛАТ[.,]?\s*КАРТ", label)  # noqa: RUF001 -- Russian payment caption
    )


def is_receipt_page(texts: Iterable[str]) -> bool:
    """Require distinct fiscal cues, including OCR Latin/Cyrillic lookalikes."""
    mapping = str.maketrans("ACEHKMOPTXY", "АСЕНКМОРТХУ")  # noqa: RUF001
    labels = " ".join(texts).upper().translate(mapping)
    cues = ("ИТОГ", "ОПЛАТ", "НАЛИЧ", "НДС", "СУМ", "КАСС")  # noqa: RUF001
    return sum(word in labels for word in cues) >= 2


def field_row_distance(label: TextCell, amount: TextCell) -> float:
    """Compare baselines at the amount column, including tilted photographs."""
    # The amount often spans more source pixels than a damaged short caption.
    # Its baseline therefore gives the more stable tilt measurement.
    rect, other = amount.rect, label.rect
    left = (rect.r_x0 + rect.r_x3) / 2
    right = (rect.r_x1 + rect.r_x2) / 2
    y_left = (rect.r_y0 + rect.r_y3) / 2
    y_right = (rect.r_y1 + rect.r_y2) / 2
    x = sum((other.r_x0, other.r_x1, other.r_x2, other.r_x3)) / 4
    y = sum((other.r_y0, other.r_y1, other.r_y2, other.r_y3)) / 4
    slope = (y_right - y_left) / (right - left) if right > left else 0.0
    return abs(y - (y_left + slope * (x - left)))


def field_row_rect(label: TextCell, amount: TextCell) -> BoundingRectangle:
    """Keep the source baseline when joining a distant label and amount."""
    rect, other = label.rect, amount.rect

    def center_x(r: BoundingRectangle) -> float:
        return (r.r_x0 + r.r_x1 + r.r_x2 + r.r_x3) / 4

    def center_y(r: BoundingRectangle) -> float:
        return (r.r_y0 + r.r_y1 + r.r_y2 + r.r_y3) / 4

    dx = center_x(other) - center_x(rect)
    slope = (center_y(other) - center_y(rect)) / dx if dx > 0 else 0.0
    if all(abs(r.r_y2 - r.r_y3) <= abs(r.r_x2 - r.r_x3) * 0.01 for r in (rect, other)):
        slope = 0.0
    points = [
        point
        for r in (rect, other)
        for point in (
            (r.r_x0, r.r_y0),
            (r.r_x1, r.r_y1),
            (r.r_x2, r.r_y2),
            (r.r_x3, r.r_y3),
        )
    ]
    left, right = min(x for x, _ in points), max(x for x, _ in points)
    top = min(y - slope * x for x, y in points)
    bottom = max(y - slope * x for x, y in points)
    return BoundingRectangle(
        r_x0=left,
        r_y0=bottom + slope * left,
        r_x1=right,
        r_y1=bottom + slope * right,
        r_x2=right,
        r_y2=top + slope * right,
        r_x3=left,
        r_y3=top + slope * left,
        coord_origin=rect.coord_origin,
    )


def _supported_wide_field(
    label: TextCell,
    *,
    amount: TextCell,
    lines: list[TextCell],
    amounts: list[TextCell],
    page_height: float,
) -> bool:
    """Use a wide gutter only when two other fiscal rows confirm its columns."""
    box, right = label.rect.to_bounding_box(), amount.rect.to_bounding_box()
    row_centers: list[float] = []
    center = (box.t + box.b) / 2
    for peer in lines:
        if peer.index == label.index or not is_receipt_field_caption(peer.text):
            continue
        other = peer.rect.to_bounding_box()
        peer_center = (other.t + other.b) / 2
        height = max(box.height, other.height)
        if (
            abs(other.l - box.l) > height
            or not height * 0.8 < abs(peer_center - center) <= page_height * 0.15
            or any(abs(peer_center - seen) < height * 0.8 for seen in row_centers)
        ):
            continue
        if (
            re.search(r"\d+[.,]\d{2}", peer.text) is not None
            and abs(other.r - right.r) <= height
        ) or any(
            candidate.index != amount.index
            and abs(candidate.rect.to_bounding_box().r - right.r) <= height
            and field_row_distance(peer, candidate)
            <= min(other.height, candidate.rect.to_bounding_box().height) * 0.6
            for candidate in amounts
        ):
            row_centers.append(peer_center)
    return len(row_centers) >= 2


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
            distance = field_row_distance(label, amount)
            height = min(box.height, other.height)
            same_row = (
                other.l >= box.r - height * 0.1
                and distance <= height * 0.6
                and max(box.height, other.height) <= 3 * height
            )
            if same_row and (
                other.l - box.r <= page.size.width * 0.35
                or (
                    other.l - box.r <= page.size.width * 0.6
                    and _supported_wide_field(
                        label,
                        amount=amount,
                        lines=lines,
                        amounts=amounts,
                        page_height=page.size.height,
                    )
                )
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
                "rect": field_row_rect(label, amount),
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
