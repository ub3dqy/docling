# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Keep real-document quality checks independent of conversion success."""

from scripts.qualify_real_russian_ocr import (
    FieldReference,
    RowReference,
    TableEvidence,
    check_field,
    check_row,
)


def test_receipt_amount_requires_a_complete_token_in_the_region_and_export() -> None:
    price = FieldReference(name="price", roi=(0, 0, 1, 1), value="56.99")
    assert not check_field(price, "56.993", "56.993").passed
    assert not check_field(price, "56.99", "").passed
    assert check_field(price, "56.99.", "56,99").passed


def test_merged_table_rows_do_not_count_as_a_correct_financial_row() -> None:
    reference = RowReference(key="1100", values=["1 434 493", "1 481 626"])
    merged = TableEvidence(
        rows=[["1100 1200", "1 434 493", "1 481 626"]], num_rows=1, num_cols=3
    )
    assert not check_row(reference, [merged]).passed
    separated = TableEvidence(
        rows=[["1100", "1 434 493", "1 481 626"]], num_rows=1, num_cols=3
    )
    assert check_row(reference, [separated]).passed
