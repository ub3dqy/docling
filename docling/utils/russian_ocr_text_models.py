# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Pinned Cyrillic recognizers. Downloads are explicit; inference is offline."""

import hashlib
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CyrillicModelAsset:
    name: str
    size: int
    sha256: str


@dataclass(frozen=True)
class CyrillicModelSpec:
    name: str
    repository: str
    revision: str
    assets: tuple[CyrillicModelAsset, ...]


_KEYS = CyrillicModelAsset(
    "rec_keys.txt",
    3631,
    "54d92ec6d2873654d1def513c7a95951402eac2f3b90f5c9c382640bb3f6bbb5",
)
_NOTICE = CyrillicModelAsset(
    "NOTICE", 115, "5a4a4b6c9e20d1a1be7e3a8ae6d0de82e7bbdd983d3c49404fc70c7b0e2e97e5"
)
# Apache-2.0 model cards and upstream attribution travel with the weights.
CYRILLIC_MODELS = (
    CyrillicModelSpec(
        "medium",
        "metrika-labs/PP-OCRv6_medium_rec_cyrillic_onnx",
        "b7adf385100d30fe27c3aac598e2a7623539060c",
        (
            CyrillicModelAsset(
                "model.onnx",
                62762162,
                "7e3412339fdb50e26149cce226cca2b0788377fa58a297cdb9df9d2353344b1b",
            ),
            _KEYS,
            _NOTICE,
            CyrillicModelAsset(
                "README.md",
                4589,
                "2b3928023d614d9d83c7d777ac6819ad31eff333071b61422c6cc81c29f928de",
            ),
        ),
    ),
    CyrillicModelSpec(
        "tiny",
        "metrika-labs/PP-OCRv6_tiny_rec_cyrillic_onnx",
        "375752a0865949066f6d49828ee927faf9f4577c",
        (
            CyrillicModelAsset(
                "model.onnx",
                2497057,
                "87ad7e9c5681417b314ced82fccc8f703e05ae084edaf7a8aa31948a71a1a773",
            ),
            _KEYS,
            _NOTICE,
            CyrillicModelAsset(
                "README.md",
                4583,
                "2fa5ed810b9fb451016c9419ffea2398e1c91e575db9582cbab38935e1fc4a85",
            ),
        ),
    ),
)


def _matches(path: Path, asset: CyrillicModelAsset) -> bool:
    if not path.is_file() or path.stat().st_size != asset.size:
        return False
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest() == asset.sha256


def validate_cyrillic_models(directory: Path) -> None:
    for model in CYRILLIC_MODELS:
        for asset in model.assets:
            path = directory / "CyrillicOcrV6" / model.name / asset.name
            if not _matches(path, asset):
                raise FileNotFoundError(
                    f"Missing or changed Cyrillic refinement asset: {path}. "
                    "Run docling-tools models download russian_ocr_refinement "
                    f"--output-dir {directory}, or disable use_region_refinement."
                )


def download_cyrillic_models(directory: Path, *, force: bool = False) -> None:
    from huggingface_hub import hf_hub_download

    for model in CYRILLIC_MODELS:
        target = directory / "CyrillicOcrV6" / model.name
        target.mkdir(parents=True, exist_ok=True)
        for asset in model.assets:
            path = target / asset.name
            if not force and _matches(path, asset):
                continue
            source = Path(
                hf_hub_download(
                    repo_id=model.repository,
                    filename=asset.name,
                    revision=model.revision,
                    force_download=force,
                )
            )
            if not _matches(source, asset):
                raise ValueError(
                    f"Cyrillic model checksum mismatch: {model.name}/{asset.name}"
                )
            # Verify the staged bytes as well. A failed fetch or copy cannot
            # replace a trusted model with an incomplete or mismatched file.
            with tempfile.NamedTemporaryFile(dir=target, delete=False) as file:
                temporary = Path(file.name)
                try:
                    with source.open("rb") as incoming:
                        shutil.copyfileobj(incoming, file)
                    file.close()
                    if not _matches(temporary, asset):
                        raise ValueError(
                            f"Cyrillic model copy checksum mismatch: {path}"
                        )
                    temporary.replace(path)
                finally:
                    file.close()
                    temporary.unlink(missing_ok=True)
