# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Keep separate source columns when a detector joins neighboring papers."""

import re
from statistics import median

from docling_core.types.doc import BoundingBox
from docling_core.types.doc.page import BoundingRectangle, TextCell

from docling.datamodel.base_models import Cluster, Page
from docling.utils.ocr_receipt_rows import is_receipt_field_caption


def split_receipt_lines(page: Page) -> None:
    """Split only complete word evidence, preserving every source word and digit."""
    if page.parsed_page is None:
        return
    lines, words = page.parsed_page.textline_cells, page.parsed_page.word_cells
    next_index = max((c.index for c in lines), default=-1) + 1
    replacements: dict[int, list[TextCell]] = {}
    for line in lines:
        if not line.from_ocr:
            continue
        box = line.rect.to_bounding_box()
        contained = sorted(
            (
                w
                for w in words
                if w.from_ocr
                and w.rect.to_bounding_box().intersection_over_self(box) >= 0.8
            ),
            key=lambda w: w.rect.to_bounding_box().l,
        )
        if len(contained) < 2 or "".join(line.text.split()) != "".join(
            "".join(w.text.split()) for w in contained
        ):
            continue
        company_prefix = re.fullmatch(r"[0oO\u041e\u043e]{3}", contained[0].text)
        if (
            company_prefix is None
            and not any(ch.isdigit() for ch in line.text)
            and not is_receipt_field_caption(line.text)
        ) or sum(w.rect.to_bounding_box().width for w in contained) < box.width * 0.55:
            # A bad CTC alignment can place a few short "words" inside a long
            # sentence. Retain its complete source window for rereading.
            continue
        heights = [
            min(
                w.rect.to_bounding_box().height,
                w.rect.to_bounding_box().width / max(1, len(w.text)) * 1.7,
            )
            for w in contained
        ]
        gap = max(4, median(heights))
        groups = [[contained[0]]]
        for word in contained[1:]:
            previous = groups[-1][-1].rect.to_bounding_box()
            amount = re.fullmatch(
                r"[=* ]*\d+[.,]\d{2}", word.text
            ) is not None and is_receipt_field_caption(
                " ".join(w.text for w in groups[-1])
            )
            if (
                word.rect.to_bounding_box().l - previous.r > gap
                or (company_prefix is not None and len(groups) == 1)
                or amount
            ):
                groups.append([])
            groups[-1].append(word)
        if len(groups) == 1:
            continue
        separated = []
        for index, group in enumerate(groups):
            boxes = [w.rect.to_bounding_box() for w in group]
            bounds = BoundingBox(
                l=min(b.l for b in boxes),
                t=min(b.t for b in boxes),
                r=max(b.r for b in boxes),
                b=max(b.b for b in boxes),
                coord_origin=box.coord_origin,
            )
            separated.append(
                line.model_copy(
                    update={
                        "index": line.index if index == 0 else next_index,
                        "text": " ".join(w.text for w in group),
                        "orig": " ".join(w.orig for w in group),
                        "rect": (
                            group[0].rect
                            if len(group) == 1
                            else BoundingRectangle.from_bounding_box(bounds)
                        ),
                        "confidence": min(w.confidence for w in group),
                    }
                )
            )
            if index:
                next_index += 1
        replacements[line.index] = separated
    if not replacements:
        return
    page.parsed_page.textline_cells = [
        part
        for cell in lines
        for part in (replacements.get(cell.index, [cell]) if cell.from_ocr else [cell])
    ]

    def update(cluster: Cluster) -> None:
        cluster.cells = [
            part
            for cell in cluster.cells
            for part in (
                replacements.get(cell.index, [cell]) if cell.from_ocr else [cell]
            )
        ]
        for child in cluster.children:
            update(child)

    if page.predictions.layout is not None:
        for cluster in page.predictions.layout.clusters:
            update(cluster)
