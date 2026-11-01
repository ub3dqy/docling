# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""An interrupted or corrupt model update must preserve the trusted checkpoint."""

import hashlib
from pathlib import Path

import pytest

from docling.models.stages.ocr.rapid_ocr_model import RapidOcrModel
from docling.utils import russian_ocr_models


def test_corrupt_download_keeps_the_previous_model_and_removes_temporary_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    previous = b"previous trusted checkpoint"
    directory = tmp_path / "tessdata"
    directory.mkdir()
    checkpoint = directory / "rus.traineddata"
    checkpoint.write_bytes(previous)
    monkeypatch.setattr(
        russian_ocr_models, "_MODELS", {"rus": hashlib.sha256(previous).hexdigest()}
    )
    monkeypatch.setattr(RapidOcrModel, "download_models", lambda **kwargs: None)

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size):
            yield b"untrusted or incomplete checkpoint"

    monkeypatch.setattr(
        russian_ocr_models.requests, "get", lambda *args, **kwargs: Response()
    )
    with pytest.raises(ValueError, match="checksum mismatch"):
        russian_ocr_models.download_refinement_models(tmp_path, force=True)
    russian_ocr_models.validate_tessdata(directory)
    assert checkpoint.read_bytes() == previous
    assert list(directory.iterdir()) == [checkpoint]
