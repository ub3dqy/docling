# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Real, offline OCR regression with prefetched Russian models.

Set DOCLING_TEST_RUSSIAN_ARTIFACTS to the directory created by docling-tools.
These fixtures are synthetic printed scans, not handwriting or scanner tests.
"""

import os
from pathlib import Path

import pytest

from scripts.qualify_russian_ocr import PageScore, qualify

pytestmark = pytest.mark.ml_ocr


@pytest.fixture(scope="module")
def russian_scores(tmp_path_factory: pytest.TempPathFactory) -> list[PageScore]:
    artifacts = os.environ.get("DOCLING_TEST_RUSSIAN_ARTIFACTS")
    if artifacts is None:
        pytest.skip("Set DOCLING_TEST_RUSSIAN_ARTIFACTS to prefetched Russian models.")
    configured_output = os.environ.get("DOCLING_TEST_RUSSIAN_OUTPUT")
    output = (
        Path(configured_output)
        if configured_output is not None
        else tmp_path_factory.mktemp("russian-ocr")
    )
    return qualify(Path("tests/data/russian_ocr"), Path(artifacts), output)


@pytest.mark.parametrize("case", ["russian", "mixed", "table", "degraded"])
@pytest.mark.parametrize("source_kind", ["image", "pdf"])
def test_russian_ocr_preserves_scan_text_and_table(
    russian_scores: list[PageScore], case: str, source_kind: str
) -> None:
    matches = [
        score
        for score in russian_scores
        if score.case == case
        and (Path(score.source).suffix == ".pdf") == (source_kind == "pdf")
    ]
    assert len(matches) == 1, "Every scan must be checked as an image and a PDF page."
    score = matches[0]
    assert score.passed, score.model_dump_json(indent=2)
