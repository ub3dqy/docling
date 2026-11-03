# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Read numeric table cells from their complete original grid regions."""

from __future__ import annotations

import re
from collections import defaultdict
from statistics import median
from typing import TYPE_CHECKING

import numpy as np

from docling.datamodel.base_models import Page, PageElement, Table
from docling.models.stages.ocr.ocr_region_refiner import (
    choose_number,
    choose_short_code,
    number_key,
)
from docling.utils.ocr_region_pixels import photo_contrast, printed_dash

if TYPE_CHECKING:
    from docling.models.stages.ocr.ocr_region_refiner import OcrRegionRefiner


def _rectified_dash(gray: np.ndarray, table: Table, quarter_turns: int) -> bool:
    if quarter_turns and table.ocr_grid_geometry is None:
        gray = np.rot90(gray, quarter_turns).copy()
    return printed_dash(gray)


def refine_table_regions(
    refiner: OcrRegionRefiner,
    page: Page,
    elements: list[PageElement],
    *,
    image_input: bool = False,
) -> None:
    if (
        page.parsed_page is None
        or page.size is None
        or not any(cell.from_ocr for cell in page.parsed_page.textline_cells)
    ):
        return
    tables = [element for element in elements if isinstance(element, Table)]
    if not tables:
        return
    # Dense PDF scans need 600 DPI. Short photographic cells also need
    # enough detail to avoid another reduction to a 72 DPI fallback image.
    cell_heights = [
        cell.bbox.height
        for table in tables
        for cell in table.table_cells
        if cell.bbox is not None
    ]
    photo_scale = 4 if cell_heights and median(cell_heights) < 12 else 1
    # Bound the full-page canvas before reading individual small cells.
    photo_scale = min(photo_scale, 6144 / max(page.size.width, page.size.height))
    image = page.get_image(scale=photo_scale if image_input else 600 / 72)
    if image is None:
        return
    with (
        refiner._lock,
        refiner._tesserocr.PyTessBaseAPI(
            path=str(refiner.tessdata),
            lang="rus+eng",
            psm=refiner._tesserocr.PSM.SINGLE_LINE,
            oem=refiner._tesserocr.OEM.LSTM_ONLY,
        ) as api,
    ):
        for table in tables:
            from docling.models.stages.ocr.ocr_photo_regions import (
                photo_number_readings,
                refine_photo_labels,
            )
            from docling.utils.ocr_photo_grid import (
                accepted_grid_regions,
                photo_table_regions,
            )

            # Recover skewed source cells in either a photo or a scanned PDF.
            # Axis-aligned bounds can otherwise include the neighboring rule.
            # Straight PDF cells retain the established source crop. An
            # unnecessary warp can join a faint digit to its neighbouring rule.
            if image_input:
                regions = photo_table_regions(image, table, scale=photo_scale)
            elif table.ocr_grid_rectified:
                regions = (
                    accepted_grid_regions(image, table, scale=600 / 72)
                    if table.ocr_grid_geometry is not None
                    else photo_table_regions(image, table, scale=600 / 72)
                )
            else:
                regions = {}
            if image_input:
                refine_photo_labels(refiner, api, page, image, table, regions)
            column_keys: dict[int, list[str]] = defaultdict(list)
            for cell in table.table_cells:
                key = number_key(cell.text)
                if key is not None:
                    column_keys[cell.start_col_offset_idx].append(key)
            # A row identifier or note reference must not become a new
            # amount. Refine established amount columns, retaining short
            # values inside them as well as large totals and decimals.
            amount_columns = {
                column
                for column, keys in column_keys.items()
                if len(keys) >= 3
                and any(
                    sum(char.isdigit() for char in key) >= (4 if image_input else 5)
                    or re.fullmatch(r"-?\d+\.\d{2}", key) is not None
                    for key in keys
                )
            }
            numeric_columns = {
                column for column, keys in column_keys.items() if len(keys) >= 3
            }
            if table.from_ocr_grid:
                # Short financial tables can have only two small amounts in
                # a year column. Printed dashes in its other physical cells
                # still need inspection; neither large totals nor OCR words
                # are required to see a real source dash.
                for column, keys in column_keys.items():
                    body = [
                        cell
                        for cell in table.table_cells
                        if cell.start_col_offset_idx == column
                        and not cell.column_header
                        and not cell.row_header
                    ]
                    if len(keys) >= 2 and not any(
                        char.isalpha() for cell in body for char in cell.text
                    ):
                        numeric_columns.add(column)
                # A fully ruled amount column can contain only printed dashes.
                # Its empty OCR output is not evidence of an empty source cell.
                amount_rows: dict[int, int] = defaultdict(int)
                for cell in table.table_cells:
                    if (
                        cell.start_col_offset_idx in amount_columns
                        and number_key(cell.text) is not None
                    ):
                        amount_rows[cell.start_row_offset_idx] += 1
                for column in range(table.num_cols):
                    body = [
                        cell
                        for cell in table.table_cells
                        if cell.start_col_offset_idx == column
                        and amount_rows[cell.start_row_offset_idx] >= 2
                        and not cell.column_header
                        and not cell.row_header
                    ]
                    if len(body) >= 3 and not any(
                        char.isalpha() for cell in body for char in cell.text
                    ):
                        numeric_columns.add(column)
            code_columns = {
                column
                for column, keys in column_keys.items()
                if len(keys) >= 3
                and sum(re.fullmatch(r"\d{3}", key) is not None for key in keys)
                >= 0.7 * len(keys)
            } - amount_columns
            if image_input:
                code_columns = set()
            for cell in table.table_cells:
                if (
                    cell.bbox is None
                    or cell.start_col_offset_idx not in numeric_columns
                    or cell.column_header
                    or cell.row_header
                ):
                    continue
                box = cell.bbox.to_top_left_origin(page.size.height)
                source_words = [
                    word
                    for word in page.parsed_page.word_cells
                    if word.rect.to_bounding_box().intersection_over_self(box) > 0.5
                ]
                if any(not word.from_ocr for word in source_words) or (
                    not source_words and not table.from_ocr_grid
                ):
                    continue
                # Grid cells include border rules. Neural table boxes
                # usually hug the glyphs and need a small outer margin.
                inset = 0.35 if table.from_ocr_grid else -0.7
                rectified_gray = regions.get(id(cell))
                # Preserve the original shapes of faint PDF digits. A second
                # interpolation can close a 6 into an 8. The recovered grid
                # still supplies correct bounds and independent dash evidence.
                gray = rectified_gray if image_input else None
                if gray is None:
                    gray = refiner._crop(image, cell.bbox, page, inset=inset)
                if gray is not None:
                    quarter_turns = {
                        "rot_90": 1,
                        "rot_180": 2,
                        "rot_270": 3,
                    }.get(table.orientation.value, 0)
                    if quarter_turns:
                        gray = np.rot90(gray, quarter_turns).copy()
                    empty_or_marker = cell.text.strip() in {
                        "",
                        ".",
                        "-",
                        "_",
                        "\u2014",
                        "\u2013",
                    }
                    if empty_or_marker or number_key(cell.text) is None:
                        if (
                            printed_dash(gray)
                            or (
                                not image_input
                                and rectified_gray is not None
                                and _rectified_dash(
                                    rectified_gray, table, quarter_turns
                                )
                            )
                            or (image_input and printed_dash(photo_contrast(gray)))
                        ):
                            cell.text = "-"
                            continue
                    if empty_or_marker:
                        if (
                            cell.text.strip()
                            or cell.start_col_offset_idx not in code_columns
                        ):
                            continue
                    if cell.start_col_offset_idx in code_columns:
                        cell.text = choose_short_code(
                            cell.text,
                            [
                                *refiner._rapid_readings(gray, numeric=True),
                                *refiner._tess_readings(api, gray, numeric=True),
                            ],
                        )
                        continue
                    if (
                        cell.start_col_offset_idx not in amount_columns
                        or number_key(cell.text) is None
                    ):
                        continue
                    keys = column_keys[cell.start_col_offset_idx]
                    cell.text = choose_number(
                        cell.text,
                        photo_number_readings(refiner, gray, cell.text)
                        if image_input
                        else refiner._rapid_readings(gray, numeric=True),
                        allow_decimal_removal=(
                            len(keys) >= 5
                            and sum("." not in key for key in keys) / len(keys) >= 0.9
                        ),
                    )
                    cell.text = refiner._refine_ambiguous_digit(api, gray, cell.text)
