# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Keep real-document quality checks independent of conversion success."""

from pathlib import Path

import pytest

from scripts.qualify_real_russian_ocr import (
    FieldReference,
    ReferenceAdjudications,
    References,
    RowReference,
    SourceManifest,
    TableEvidence,
    check_field,
    check_row,
    reviewed_references,
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


@pytest.fixture
def frozen_source_review():
    directory = Path("tests/data/russian_ocr_real")
    return (
        References.model_validate_json(
            (directory / "references.json").read_text(encoding="utf-8")
        ),
        SourceManifest.model_validate_json(
            (directory / "sources.json").read_text(encoding="utf-8")
        ),
        ReferenceAdjudications.model_validate_json(
            (directory / "reference_adjudications.json").read_text(encoding="utf-8")
        ),
    )


def test_source_review_keeps_the_original_reference_and_control_count(
    frozen_source_review,
):
    original, manifest, adjudications = frozen_source_review
    reviewed = reviewed_references(original, manifest, adjudications)
    assert original.cases[1].rows[1].values[2] == "76 393"
    assert reviewed.cases[1].rows[1].values[2] == "75 393"
    assert sum(len(row.values) for case in reviewed.cases for row in case.rows) == sum(
        len(row.values) for case in original.cases for row in case.rows
    )


@pytest.mark.parametrize(
    "tampering",
    [
        {"source_sha256": "0" * 64},
        {"source_page": 6},
        {"original": "75 393"},
        {"value_index": 50},
        {"row_key": "missing"},
    ],
)
def test_source_review_rejects_a_different_source_or_reference_value(
    frozen_source_review, tampering
):
    original, manifest, adjudications = frozen_source_review
    adjudications.corrections[0] = adjudications.corrections[0].model_copy(
        update=tampering
    )
    with pytest.raises(ValueError):
        reviewed_references(original, manifest, adjudications)
