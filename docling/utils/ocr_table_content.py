# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Account for OCR tokens before a table replaces their source text."""

from collections.abc import Sequence

from docling_core.types.doc.page import TextCell

from docling.datamodel.base_models import Table


def record_ocr_table_coverage(
    table: Table,
    source_cells: Sequence[TextCell],
    *,
    matched_token_ids: set[int],
) -> None:
    """Retain source tokens that did not reach the predictor/grid's output.

    IDs refer to positions in the nonempty token list passed to TableFormer,
    never to TextCell.index, which can overlap between OCR and native text.
    Track placement before numeric refinement, which can change the cell text.
    """
    source = [cell for cell in source_cells if cell.text.strip()]
    table.source_cell_count = len(source)
    table.unmatched_ocr_cells = [
        cell
        for index, cell in enumerate(source)
        if cell.from_ocr and index not in matched_token_ids
    ]
