# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Model corruption and interrupted copies must not replace trusted bytes."""

import hashlib
from pathlib import Path

import pytest

from docling.utils import russian_ocr_text_models as models


@pytest.fixture
def trusted_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    payloads = {
        "model.onnx": b"trusted model",
        "rec_keys.txt": "\u0430\n\u0431\n".encode(),
    }
    assets = tuple(
        models.CyrillicModelAsset(name, len(data), hashlib.sha256(data).hexdigest())
        for name, data in payloads.items()
    )
    spec = models.CyrillicModelSpec("medium", "owner/model", "a" * 40, assets)
    monkeypatch.setattr(models, "CYRILLIC_MODELS", (spec,))
    directory = tmp_path / "CyrillicOcrV6" / "medium"
    directory.mkdir(parents=True)
    for name, data in payloads.items():
        (directory / name).write_bytes(data)
    return tmp_path


@pytest.mark.parametrize("payload", [b"truncated", b"changed model"])
def test_untrusted_fetch_preserves_the_previous_valid_model(
    trusted_model: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, payload: bytes
) -> None:
    checkpoint = trusted_model / "CyrillicOcrV6" / "medium" / "model.onnx"
    previous = checkpoint.read_bytes()
    fetched = tmp_path / "fetched.onnx"
    fetched.write_bytes(payload)
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download", lambda **kwargs: str(fetched)
    )
    with pytest.raises(ValueError, match="checksum mismatch"):
        models.download_cyrillic_models(trusted_model, force=True)
    models.validate_cyrillic_models(trusted_model)
    assert checkpoint.read_bytes() == previous


def test_interrupted_staging_preserves_checkpoint_and_removes_partial_copy(
    trusted_model: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = trusted_model / "CyrillicOcrV6" / "medium"
    checkpoint = directory / "model.onnx"
    previous = {path.name: path.read_bytes() for path in directory.iterdir()}
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download", lambda **kwargs: str(checkpoint)
    )

    def interrupt_copy(incoming, destination):
        destination.write(incoming.read(3))
        raise OSError("interrupted copy")

    monkeypatch.setattr(models.shutil, "copyfileobj", interrupt_copy)
    with pytest.raises(OSError, match="interrupted copy"):
        models.download_cyrillic_models(trusted_model, force=True)
    models.validate_cyrillic_models(trusted_model)
    assert {path.name: path.read_bytes() for path in directory.iterdir()} == previous


def test_complete_verified_local_models_need_no_network(
    trusted_model: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse_network(**kwargs):
        pytest.fail("Verified models must be reusable offline.")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", refuse_network)
    models.download_cyrillic_models(trusted_model)
    models.validate_cyrillic_models(trusted_model)


def test_an_altered_alphabet_is_rejected_even_if_weights_are_unchanged(
    trusted_model: Path,
) -> None:
    keys = trusted_model / "CyrillicOcrV6" / "medium" / "rec_keys.txt"
    keys.write_text("\u0431\n\u0430\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match=r"rec_keys\.txt"):
        models.validate_cyrillic_models(trusted_model)
