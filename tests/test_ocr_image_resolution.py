# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Image DPI must not discard source detail or cause unbounded compensation."""

import pytest
from PIL import Image

from docling.backend.image_backend import _ImagePageBackend
from docling.datamodel.base_models import InputFormat, Page
from docling.datamodel.pipeline_options import RapidOcrOptions
from docling.models.stages.ocr.rapid_ocr_model import RapidOcrModel


@pytest.mark.parametrize(
    ("preserve", "expected"), [(False, (317, 413)), (True, (660, 861))]
)
def test_high_dpi_photo_retains_source_pixels_only_when_requested(preserve, expected):
    backend = _ImagePageBackend(Image.new("RGB", (660, 861)), 0, (300, 300))
    page = Page(page_no=1, size=backend.get_size())
    page._backend = backend
    model = RapidOcrModel.__new__(RapidOcrModel)
    model.options = RapidOcrOptions(image_scale=2, preserve_native_resolution=preserve)
    model.scale = 300 / 72

    image_scale = model._render_scale(page, InputFormat.IMAGE)
    assert backend.get_page_image(scale=image_scale).size == expected
    # A PDF continues to use its configured rendering scale.
    assert model._render_scale(page, InputFormat.PDF) == model.scale


def test_anisotropic_dpi_cannot_expand_automatic_compensation_beyond_canvas():
    backend = _ImagePageBackend(Image.new("RGB", (100, 60)), 0, (1_000_000, 1))
    page = Page(page_no=1, size=backend.get_size())
    page._backend = backend
    model = RapidOcrModel.__new__(RapidOcrModel)
    model.options = RapidOcrOptions(image_scale=0.1, preserve_native_resolution=True)
    model.scale = 300 / 72

    scale = model._render_scale(page, InputFormat.IMAGE)
    assert max(backend.get_page_image(scale=scale).size) == 1536

    model.options.image_scale = 2
    assert model._render_scale(page, InputFormat.IMAGE) == 2


def test_dense_high_dpi_page_keeps_its_already_sufficient_configured_view():
    backend = _ImagePageBackend(Image.new("RGB", (2480, 3508)), 0, (300, 300))
    page = Page(page_no=1, size=backend.get_size())
    page._backend = backend
    model = RapidOcrModel.__new__(RapidOcrModel)
    model.options = RapidOcrOptions(image_scale=2, preserve_native_resolution=True)
    model.scale = 300 / 72

    assert model._render_scale(page, InputFormat.IMAGE) == 2
