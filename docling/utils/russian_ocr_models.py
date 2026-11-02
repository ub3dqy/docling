# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Pinned, integrity-checked Tesseract models for Russian line refinement."""

import hashlib
import tempfile
from pathlib import Path

import requests

_REVISION = "e12c65a915945e4c28e237a9b52bc4a8f39a0cec"
_MODELS = {
    "rus": "b617eb6830ffabaaa795dd87ea7fd251adfe9cf0efe05eb9a2e8128b7728d6b6",
    "eng": "8280aed0782fe27257a68ea10fe7ef324ca0f8d85bd2fd145d1c2b560bcb66ba",
}


def _sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def validate_tessdata(directory: Path) -> None:
    for language, digest in _MODELS.items():
        path = directory / f"{language}.traineddata"
        if not path.is_file() or _sha256(path) != digest:
            raise FileNotFoundError(
                f"Missing or changed Russian refinement model: {path}. "
                "Run docling-tools models download russian_ocr_refinement "
                f"--output-dir {directory.parent}, or disable use_region_refinement."
            )


def download_refinement_models(
    directory: Path, *, force: bool = False, progress: bool = False
) -> None:
    from docling.models.stages.ocr.rapid_ocr_model import RapidOcrModel

    for language in ("iso:ru", "cyrillic", "ch"):
        RapidOcrModel.download_models(
            backend="onnxruntime",
            lang=language,
            local_dir=directory / "RapidOcr",
            force=force,
            progress=progress,
        )
    tessdata = directory / "tessdata"
    tessdata.mkdir(parents=True, exist_ok=True)
    for language, digest in _MODELS.items():
        path = tessdata / f"{language}.traineddata"
        if not force and path.is_file() and _sha256(path) == digest:
            continue
        with tempfile.NamedTemporaryFile(dir=tessdata, delete=False) as file:
            temporary = Path(file.name)
            try:
                url = f"https://raw.githubusercontent.com/tesseract-ocr/tessdata_best/{_REVISION}/{language}.traineddata"
                with requests.get(url, stream=True, timeout=(5, 30)) as response:
                    response.raise_for_status()
                    downloaded_bytes = 0
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        downloaded_bytes += len(chunk)
                        if downloaded_bytes > 32 * 1024 * 1024:
                            raise ValueError(f"Oversized Tesseract model: {language}")
                        file.write(chunk)
                file.close()
                if _sha256(temporary) != digest:
                    raise ValueError(f"Tesseract model checksum mismatch: {language}")
                temporary.replace(path)
            finally:
                file.close()
                temporary.unlink(missing_ok=True)
    from docling.utils.russian_ocr_text_models import download_cyrillic_models

    download_cyrillic_models(directory, force=force)
