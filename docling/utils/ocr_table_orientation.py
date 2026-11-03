# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Read a sideways OCR grid while retaining original-page provenance."""

from dataclasses import dataclass
from statistics import median

from docling_core.types.doc import BoundingBox
from docling_core.types.doc.document import Orientation
from docling_core.types.doc.page import BoundingRectangle, TextCell
from PIL import Image

from docling.datamodel.base_models import Cluster, Table


def turn_box(box: BoundingBox, angle: int, size: tuple[float, float]) -> BoundingBox:
    """Map a top-left box through a counter-clockwise quarter turn."""
    box = box.to_top_left_origin(size[1])
    if angle == 90:
        return BoundingBox(l=box.t, t=size[0] - box.r, r=box.b, b=size[0] - box.l)
    if angle == 270:
        return BoundingBox(l=size[1] - box.b, t=box.l, r=size[1] - box.t, b=box.r)
    raise ValueError("An OCR grid turn must be 90 or 270 degrees.")


@dataclass(frozen=True)
class OrientedGrid:
    image: Image.Image
    cluster: Cluster
    words: list[TextCell]
    angle: int
    size: tuple[float, float]

    def restore(self, table: Table, original: Cluster) -> Table:
        inverse_size = (self.size[1], self.size[0])
        geometry = table.ocr_grid_geometry
        if geometry is not None:
            (a, b, c), (d, e, f) = geometry.transform
            # Compose the original-page quarter turn with the accepted warp.
            # The grid's own coordinates stay upright at every reading DPI.
            transform = (
                ((-b, a, b * self.size[0] + c), (-e, d, e * self.size[0] + f))
                if self.angle == 90
                else ((b, -a, a * self.size[1] + c), (e, -d, d * self.size[1] + f))
            )
            geometry = geometry.model_copy(update={"transform": transform})
        return table.model_copy(
            update={
                "cluster": original,
                "ocr_grid_geometry": geometry,
                "orientation": Orientation.ROT_90
                if self.angle == 90
                else Orientation.ROT_270,
                "table_cells": [
                    cell.model_copy(
                        update={
                            "bbox": turn_box(cell.bbox, 360 - self.angle, inverse_size)
                        }
                    )
                    if cell.bbox is not None
                    else cell
                    for cell in table.table_cells
                ],
                "unmatched_ocr_cells": [
                    cell.model_copy(
                        update={
                            "rect": BoundingRectangle.from_bounding_box(
                                turn_box(
                                    cell.rect.to_bounding_box(),
                                    360 - self.angle,
                                    inverse_size,
                                )
                            )
                        }
                    )
                    for cell in table.unmatched_ocr_cells
                ],
            }
        )


def sideways_grid_views(
    image: Image.Image, cluster: Cluster, words: list[TextCell], *, scale: float
) -> list[OrientedGrid]:
    """Try quarter turns only for a clear majority of tall multi-letter words."""
    labels = [word for word in words if sum(c.isalpha() for c in word.text) >= 4]
    if len(labels) < 5 or sum(
        word.rect.to_bounding_box().height > 1.8 * word.rect.to_bounding_box().width
        for word in labels
    ) < 0.7 * len(labels):
        return []
    size = (image.width / scale, image.height / scale)
    views = []
    for angle in (90, 270):
        turned_words = [
            word.model_copy(
                update={
                    "rect": BoundingRectangle.from_bounding_box(
                        turn_box(word.rect.to_bounding_box(), angle, size)
                    )
                }
            )
            for word in words
        ]
        turned_words.sort(
            key=lambda word: (
                word.rect.to_bounding_box().t,
                word.rect.to_bounding_box().l,
            )
        )
        views.append(
            OrientedGrid(
                image.rotate(angle, expand=True),
                cluster.model_copy(
                    update={"bbox": turn_box(cluster.bbox, angle, size)}
                ),
                turned_words,
                angle,
                size,
            )
        )
    return views


def label_first_rows(table: Table) -> int:
    """Require repeated labels on the left and numeric cells on the right."""
    rows: list[list[str]] = [[""] * table.num_cols for _ in range(table.num_rows)]
    for cell in table.table_cells:
        rows[cell.start_row_offset_idx][cell.start_col_offset_idx] = cell.text
    return sum(
        any(sum(char.isalpha() for char in value) >= 4 for value in row[:2])
        and all(not any(char.isalpha() for char in value) for value in row[-2:])
        and any(char.isdigit() for value in row[-2:] for char in value)
        for row in rows
    )


def upright_table_view(
    image: Image.Image, cluster: Cluster, words: list[TextCell], *, scale: float
) -> OrientedGrid | None:
    """Choose a supported reading direction, including forms with open borders."""
    if not words or not all(word.from_ocr for word in words):
        return None
    supported = []
    for view in sideways_grid_views(image, cluster, words, scale=scale):
        labels = [
            word.rect.to_bounding_box()
            for word in view.words
            if sum(char.isalpha() for char in word.text) >= 4
        ]
        numbers = [
            word.rect.to_bounding_box()
            for word in view.words
            if sum(char.isdigit() for char in word.text) >= 3
            and not any(char.isalpha() for char in word.text)
        ]
        if len(numbers) < 8:
            continue
        gap = median((box.l + box.r) / 2 for box in numbers) - median(
            (box.l + box.r) / 2 for box in labels
        )
        if gap > view.cluster.bbox.width * 0.2:
            supported.append(view)
    return supported[0] if len(supported) == 1 else None
