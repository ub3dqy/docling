# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Primary Russian OCR must keep the direction of all source rows."""

import os
from pathlib import Path

import pytest
from docling_core.types.doc import BoundingBox, DocItemLabel, Size
from docling_core.types.doc.page import (
    BoundingRectangle,
    PdfPageBoundaryType,
    PdfPageGeometry,
    SegmentedPdfPage,
    TextCell,
)
from PIL import Image

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import Cluster, LayoutPrediction, Page
from docling.datamodel.ocr_profiles import OcrProfile, ocr_options_for_profile
from docling.datamodel.pipeline_options import RapidOcrOptions
from docling.models.stages.ocr.ocr_line_classifier import LineClassifierView
from docling.models.stages.ocr.ocr_oriented_refinement import refine_image_lines
from docling.models.stages.ocr.ocr_region_refiner import OcrRegionRefiner
from docling.models.stages.ocr.rapid_ocr_model import (
    RapidOcrModel,
    _OcrPrediction,
    _reads_well,
)
from docling.utils.orientation import ocr_quad_to_bounding_rectangle

pytest.importorskip("rapidocr")

pytestmark = pytest.mark.ml_ocr


@pytest.mark.parametrize("evidence", ["several-rows", "one-row", "weak", "vertical"])
def test_known_horizontal_rows_do_not_trigger_rotations_for_unknown_names(
    monkeypatch, evidence
):
    model = object.__new__(RapidOcrModel)
    model.options = RapidOcrOptions(
        text_score=0.25, use_cls=False, use_orientation_fallback=False
    )
    known = {"подтверждение", "документ", "приложение"}
    model.orientation_word_is_known = known.__contains__
    words = ["Подтверждение", "Документ", "Приложение"]
    if evidence == "one-row":
        words = [" ".join(words), "Неизвестноеслово", "Другоенеизвестноеслово"]
    lines = []
    for index, word in enumerate(words):
        y = index * 40
        box = (
            [[0, y], [20, y], [20, y + 250], [0, y + 250]]
            if evidence == "vertical"
            else [[0, y], [400, y], [400, y + 20], [0, y + 20]]
        )
        lines.append(
            (box, word + " " + "ЁХЖЩ" * 12, 0.75 if evidence == "weak" else 0.9)
        )
    source = _OcrPrediction(lines, [[] for _ in lines])
    calls = []

    def read(image, *, reader=None):
        calls.append(image.size)
        return None

    monkeypatch.setattr(model, "_read", read)
    result, turn, _ = model._read_upright(Image.new("RGB", (500, 400)), source)
    assert turn == 0
    assert result.lines == source.lines
    assert len(calls) == (0 if evidence == "several-rows" else 3)


def _turned_page(angle):
    upright = Image.new("RGB", (300, 500), "white")
    upright.putpixel((10, 20), (255, 0, 0))
    image = upright.rotate(-angle, expand=True)
    box = BoundingBox(l=0, t=0, r=image.width, b=image.height)
    lines = [
        TextCell(
            index=index,
            text="Пример",
            orig="Пример",
            confidence=0.9,
            from_ocr=True,
            rect=ocr_quad_to_bounding_rectangle(
                [[40, y], [220, y], [220, y + 20], [40, y + 20]],
                original_offset=box,
                scale=1,
                orientation=(360 - angle) % 360,
                im_size=upright.size,
            ),
        )
        for index, y in enumerate(range(50, 300, 50))
    ]
    dimension = PdfPageGeometry(
        angle=0,
        rect=BoundingRectangle.from_bounding_box(box),
        boundary_type=PdfPageBoundaryType.CROP_BOX,
        art_bbox=box,
        bleed_bbox=box,
        crop_bbox=box,
        media_bbox=box,
        trim_bbox=box,
    )
    page = Page(
        page_no=1,
        size=Size(width=image.width, height=image.height),
        parsed_page=SegmentedPdfPage(
            dimension=dimension,
            char_cells=[],
            word_cells=lines,
            textline_cells=lines,
            has_chars=False,
            has_words=True,
            has_lines=True,
        ),
    )
    page.predictions.layout = LayoutPrediction(
        clusters=[
            Cluster(
                id=0,
                label=DocItemLabel.TEXT,
                bbox=box,
                cells=list(lines),
            )
        ]
    )
    page._image_cache = {1.0: image}
    return page, upright


@pytest.mark.parametrize("angle", [90, 180, 270])
def test_upright_refinement_keeps_source_images_and_restores_all_text_boxes(
    monkeypatch, angle
):
    page, upright = _turned_page(angle)
    original_size = page.size
    original = page.cells[0]
    rectangles = [cell.rect for cell in page.cells]
    dimension = page.parsed_page.dimension
    image_bytes = page.get_image().tobytes()
    refiner = object.__new__(OcrRegionRefiner)

    def replace(view, *, image_input):
        assert image_input
        assert view is not page
        assert view.size == Size(width=300, height=500)
        assert view.get_image().tobytes() == upright.tobytes()
        assert view.cells[0].rect.to_bounding_box().as_tuple() == pytest.approx(
            (40, 50, 220, 70)
        )
        assert view.cells[0] is view.parsed_page.word_cells[0]
        assert view.cells[0] is view.predictions.layout.clusters[0].cells[0]
        view.cells[0].text = "Обновлено"
        extra = view.cells[1].model_copy(
            deep=True, update={"index": 99, "text": "Добавлено"}
        )
        view.parsed_page.textline_cells.append(extra)
        view.parsed_page.word_cells.append(extra)
        view.predictions.layout.clusters[0].cells.append(extra)
        assert original.text == "Пример"

    monkeypatch.setattr(refiner, "refine_lines", replace)
    refine_image_lines(refiner, page)
    assert page.size == original_size
    assert page.parsed_page.dimension is dimension
    assert page.get_image().tobytes() == image_bytes
    assert page.cells[0].text == "Обновлено"
    assert len(page.cells) == 6
    assert len(page.parsed_page.word_cells) == 6
    assert page.cells[0] is page.parsed_page.word_cells[0]
    assert page.cells[0] is page.predictions.layout.clusters[0].cells[0]
    assert [cell.rect for cell in page.cells[:5]] == rectangles
    assert page.cells[-1].text == "Добавлено"
    assert page.cells[-1].rect == rectangles[1]
    assert page.predictions.layout.clusters[0].bbox == BoundingBox(
        l=0,
        t=0,
        r=original_size.width,
        b=original_size.height,
    )


def test_failed_upright_refinement_cannot_leave_source_text_or_boxes_rotated(
    monkeypatch,
):
    page, _ = _turned_page(90)
    cells, layout, size = page.cells, page.predictions.layout, page.size
    original = page.cells[0].model_copy(deep=True)
    refiner = object.__new__(OcrRegionRefiner)

    def fail(view, *, image_input):
        view.cells[0].text = "Незавершённая замена"
        raise RuntimeError("reader failed")

    monkeypatch.setattr(refiner, "refine_lines", fail)
    with pytest.raises(RuntimeError, match="reader failed"):
        refine_image_lines(refiner, page)
    assert page.cells is cells
    assert page.cells[0] == original
    assert page.predictions.layout is layout
    assert page.size is size


@pytest.mark.parametrize("angle,native", [(0, False), (90, True)])
def test_upright_or_native_text_keeps_the_original_refinement_frame(
    monkeypatch, angle, native
):
    page, _ = _turned_page(angle)
    if native:
        page.cells[0].from_ocr = False
    refiner = object.__new__(OcrRegionRefiner)
    calls = []

    def read(view, *, image_input):
        assert image_input
        calls.append(view)

    monkeypatch.setattr(refiner, "refine_lines", read)
    refine_image_lines(refiner, page)
    assert calls == [page]
    assert calls[0] is page


@pytest.mark.parametrize("allow", [True, False])
def test_weak_orientation_evidence_keeps_only_an_explicit_legacy_fallback(
    monkeypatch, allow
):
    model = object.__new__(RapidOcrModel)
    model.options = RapidOcrOptions(
        text_score=0.25, use_cls=False, use_orientation_fallback=allow
    )
    model.reader = object()
    model._line_classifier_readers = {id(model.reader): LineClassifierView(object())}
    known = {"пример", "русского", "текста"}
    model.orientation_word_is_known = known.__contains__
    boxes = [[[0, y], [250, y], [250, y + 20], [0, y + 20]] for y in range(0, 280, 40)]
    weak = [(box, "яфвцз", 0.4) for box in boxes[2:]]
    source = _OcrPrediction(weak, [[] for _ in weak])
    lines = [
        (boxes[0], "Пример русского", 0.99),
        (boxes[1], "Пример текста", 0.99),
        *weak,
    ]
    candidate = _OcrPrediction(lines, [[] for _ in lines])
    assert not _reads_well(candidate.lines)
    fallback = _OcrPrediction(lines[:2], [[], []], line_classifier_fallback=True)

    def read(image, *, reader=None):
        return fallback if isinstance(reader, LineClassifierView) else candidate

    monkeypatch.setattr(model, "_read", read)
    result, turn, _ = model._read_upright(Image.new("RGB", (400, 400)), source)
    assert turn == 0
    assert result.lines == (fallback.lines if allow else source.lines)
    assert result.line_classifier_fallback is allow


@pytest.mark.parametrize("text,confidence", [("Пример", 0.99), ("данные", 0.4)])
def test_fallback_cannot_erase_source_text_or_flip_a_clear_short_russian_word(
    monkeypatch, text, confidence
):
    model = object.__new__(RapidOcrModel)
    model.options = RapidOcrOptions(
        text_score=0.25, use_cls=False, use_orientation_fallback=True
    )
    model.reader = object()
    model._line_classifier_readers = {id(model.reader): LineClassifierView(object())}
    model.orientation_word_is_known = lambda word: word == "пример"
    line = ([[0, 0], [100, 0], [100, 20], [0, 20]], text, confidence)
    source = _OcrPrediction([line], [[]])
    wrong = _OcrPrediction([([*line[0]], "0078", 0.99)], [[]], True)

    def read(image, *, reader=None):
        if not isinstance(reader, LineClassifierView):
            return None
        return wrong if confidence >= 0.95 else _OcrPrediction([], [], True)

    monkeypatch.setattr(model, "_read", read)
    result, turn, _ = model._read_upright(Image.new("RGB", (200, 100)), source)
    assert result.lines == source.lines
    assert turn == 0
    assert not result.line_classifier_fallback


@pytest.mark.parametrize("whole_line", [True, False])
def test_line_fallback_cannot_overwrite_a_complete_source_amount(
    monkeypatch, whole_line
):
    model = object.__new__(RapidOcrModel)
    model.options = RapidOcrOptions(
        text_score=0.25, use_cls=False, use_orientation_fallback=True
    )
    model.reader = object()
    model._line_classifier_readers = {id(model.reader): LineClassifierView(object())}
    model.orientation_word_is_known = lambda word: False
    box = [[0, 0], [100, 0], [100, 20], [0, 20]]
    source = _OcrPrediction(
        [(box, "=73.60" if whole_line else "данные =73.60", 0.7)],
        [[]] if whole_line else [[(box, "=73.60", 0.99)]],
    )
    wrong = _OcrPrediction([(box, "0637", 0.8)], [[]], True)
    monkeypatch.setattr(model, "_read", lambda image, **kwargs: wrong)
    result, turn, _ = model._read_upright(Image.new("RGB", (200, 100)), source)
    assert result.lines == source.lines
    assert turn == 0


def test_fallback_line_direction_is_retained_by_both_detector_scales(monkeypatch):
    model = object.__new__(RapidOcrModel)
    model.options = RapidOcrOptions(use_cls=False, use_region_refinement=True)
    model._multiscale_reader = object()
    model._refinement_reader = object()
    model._line_classifier_readers = {
        id(reader): LineClassifierView(object())
        for reader in (model._multiscale_reader, model._refinement_reader)
    }
    calls = []

    def read(image, *, reader=None):
        calls.append(isinstance(reader, LineClassifierView))
        return None

    monkeypatch.setattr(model, "_read", read)
    model._read_multiscale(
        Image.new("RGB", (3072, 1000)),
        _OcrPrediction([], []),
        line_classification=True,
    )
    assert calls == [True, True]


@pytest.fixture(scope="module")
def primary_model():
    artifacts = os.environ.get("DOCLING_TEST_RUSSIAN_ARTIFACTS")
    if artifacts is None:
        pytest.skip("Set DOCLING_TEST_RUSSIAN_ARTIFACTS to prefetched Russian models")
    return RapidOcrModel(
        enabled=True,
        artifacts_path=Path(artifacts),
        options=ocr_options_for_profile(OcrProfile.RUSSIAN),
        accelerator_options=AcceleratorOptions(
            device=AcceleratorDevice.CPU, num_threads=4
        ),
    )


@pytest.fixture(scope="module")
def source_reading(primary_model):
    image = Image.open("tests/data/russian_ocr/sources/russian.png").convert("RGB")
    original = primary_model._read(image)
    assert original is not None
    return image, original


@pytest.mark.parametrize("turn", [0, 90, 180, 270])
def test_page_direction_keeps_cyrillic_words_without_flipping_individual_rows(
    primary_model, source_reading, turn
):
    image, original = source_reading
    assert primary_model.options.use_cls is False
    source = image.rotate(turn, expand=True)
    prediction = primary_model._read(source)
    assert prediction is not None
    result, _, _ = primary_model._read_upright(source, prediction)
    assert sorted(line[1].strip() for line in result.lines) == sorted(
        line[1].strip()
        for line in original.lines
        if line[2] >= primary_model.options.text_score
    )


def test_an_empty_primary_reading_does_not_block_page_direction_search(
    primary_model, source_reading
):
    image, original = source_reading
    source = image.rotate(180, expand=True)
    result, turn, size = primary_model._read_upright(source, _OcrPrediction([], []))
    assert turn == 180
    assert size == image.size
    assert sorted(line[1].strip() for line in result.lines) == sorted(
        line[1].strip()
        for line in original.lines
        if line[2] >= primary_model.options.text_score
    )


def test_fallback_reader_does_not_change_the_normal_readers_direction_settings(
    primary_model,
):
    fallback = primary_model._line_classifier_readers[id(primary_model.reader)]
    assert fallback.reader is not primary_model.reader
    assert fallback.reader.cfg is not primary_model.reader.cfg
    image = Image.open("tests/data/russian_ocr/sources/russian.png").convert("RGB")
    crop = image.crop((100, 160, 2100, 255))
    primary_model._read(crop)
    assert primary_model.reader.use_cls is False
    result = primary_model._read_configured(crop, line_classification=True)
    if result is not None:
        assert result.line_classifier_fallback
    assert primary_model.reader.use_cls is False
    assert fallback.reader.use_cls is True
