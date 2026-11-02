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
    # Preserve small digits and headings at 300 DPI. RapidOCR's default 2000px
    # limit otherwise reduces a dense A4 scan back to about 170 DPI.
    return RapidOcrOptions(
        backend="onnxruntime",
        lang=["iso:ru"],
        mode=OcrMode.FULL_PAGE,
        scale=300 / 72,
        image_scale=2.0,
        preserve_native_resolution=True,
        use_word_boxes=True,
        use_multiscale=True,
        use_region_refinement=True,
        text_score=0.25,
        rapidocr_params={
            "Global.max_side_len": 4096,
            "Det.limit_side_len": 736,
            "Det.limit_type": "min",
        },
    )
