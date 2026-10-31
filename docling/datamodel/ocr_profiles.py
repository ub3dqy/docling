# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Local OCR presets shared by the CLI and Python applications."""

from enum import Enum

from docling.datamodel.pipeline_options import OcrMode, RapidOcrOptions


class OcrProfile(str, Enum):
    RUSSIAN = "ru"


def ocr_options_for_profile(profile: OcrProfile) -> RapidOcrOptions:
    """Return a fresh OCR configuration for a supported profile.

    Russian selects PP-OCRv5's East Slavic recognizer through ONNX Runtime.
    The Torch fallback uses an older recognizer and cannot serve ``iso:ru``.
    """
    if profile != OcrProfile.RUSSIAN:
        raise ValueError(f"Unsupported OCR profile: {profile!r}")
    # Read a scan as a page: tight layout crops can clip headings and final lines.
    # 144 DPI keeps this recognizer from unnecessarily rescaling large scans.
    return RapidOcrOptions(
        backend="onnxruntime", lang=["iso:ru"], mode=OcrMode.FULL_PAGE, scale=2.0
    )
