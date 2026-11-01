# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""The RapidOCR boundary must retain fine table tokens and full-line fallbacks."""

import numpy as np
import pytest
from docling_core.types.doc import BoundingBox
from PIL import Image

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.pipeline_options import RapidOcrOptions
from docling.models.stages.ocr.rapid_ocr_model import RapidOcrModel, _OcrPrediction
from docling.models.stages.table_structure.table_structure_model import (
    _words_and_ocr_cells,
)

pytest.importorskip("rapidocr")


def _box(left: float, top: float, right: float, bottom: float) -> list[list[float]]:
    return [[left, top], [right, top], [right, bottom], [left, bottom]]


def _model() -> RapidOcrModel:
    return RapidOcrModel(
        enabled=False,
        artifacts_path=None,
        options=RapidOcrOptions(use_word_boxes=True),
        accelerator_options=AcceleratorOptions(),
    )


@pytest.mark.parametrize("partial_words", [False, True])
def test_table_tokens_keep_full_lines_when_word_output_is_incomplete(
    monkeypatch: pytest.MonkeyPatch, partial_words: bool
) -> None:
    from rapidocr.utils.output import RapidOCROutput

    output = RapidOCROutput(
        boxes=np.array([_box(20, 40, 200, 60)]),
        txts=("1150 1 383 225",),
        scores=(0.95,),
        word_results=(
            (("1150", 0.96, _box(20, 40, 60, 60)),)
            if partial_words
            else (
                ("1150", 0.96, _box(20, 40, 60, 60)),
                ("1 383 225", 0.94, _box(120, 40, 200, 60)),
            ),
        ),
    )
    model = _model()
    monkeypatch.setattr(model, "reader", lambda *args, **kwargs: output, raising=False)
    prediction = model._read(Image.new("RGB", (240, 100)))
    assert prediction is not None
    offset = BoundingBox(l=100, t=200, r=220, b=250)
    lines = model._to_cells(prediction.lines, offset, 0, (240, 100), 2)
    words = model._to_cells(prediction.words, offset, 0, (240, 100), 2)
    tokens = _words_and_ocr_cells(words, lines)
    if partial_words:
        assert [token.text for token in tokens] == ["1150 1 383 225"]
    else:
        assert [token.text for token in tokens] == ["1150", "1 383 225"]
        assert tokens[1].rect.to_bounding_box() == BoundingBox(
            l=160, t=220, r=200, b=230
        )


def test_multiscale_replaces_incomplete_line_with_its_own_words(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model()
    image = Image.new("RGB", (3072, 1000))
    initial = _OcrPrediction([(_box(100, 100, 1000, 180), "We audit", 0.55)], [[]])
    alternative = _OcrPrediction(
        [(_box(50, 50, 500, 90), "We performed the audit", 0.96)],
        [[(_box(50, 50, 500, 90), "We performed the audit", 0.96)]],
    )
    monkeypatch.setattr(model, "_read", lambda image, **kwargs: alternative)
    prediction = model._read_multiscale(image, initial)
    assert [line[1] for line in prediction.lines] == ["We performed the audit"]
    assert prediction.words[0][0] == _box(100, 100, 1000, 180)
    assert prediction.words[0][1] == "We performed the audit"


def test_close_ocr_fragments_keep_printed_order_and_fine_table_tokens() -> None:
    right = (_box(298, 98, 470, 123), "отчетности", 0.98)
    left = (_box(100, 100, 300, 121), "бухгалтерской", 0.97)
    next_row = (_box(100, 150, 420, 175), "Следующая строка", 0.99)
    result = RapidOcrModel._order_and_join(
        [(right, [right]), (left, [left]), (next_row, [next_row])]
    )
    assert [line[1] for line in result.lines] == [
        "бухгалтерской отчетности",
        "Следующая строка",
    ]
    assert [word[1] for word in result.line_words[0]] == ["бухгалтерской", "отчетности"]


def test_fragment_join_never_crosses_an_established_row_boundary() -> None:
    lines = [
        (_box(100, 100, 500, 180), "Long next label", 0.95),
        (_box(600, 100, 680, 120), "5", 0.95),
        (_box(100, 105, 300, 140), "Previous label", 0.95),
    ]
    result = RapidOcrModel._order_and_join([(line, [line]) for line in lines])
    assert len(result.lines) == 3
    assert {line[1] for line in result.lines} == {line[1] for line in lines}


def test_multiscale_does_not_replace_a_neighbouring_tilted_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model()
    initial = _OcrPrediction(
        [([[0, 40], [1000, 100], [1000, 130], [0, 70]], "Last sentence", 0.95)], [[]]
    )
    alternative = _OcrPrediction(
        [
            (
                [[0, 0], [500, 30], [500, 45], [0, 15]],
                "A longer sentence in the preceding printed row",
                0.96,
            )
        ],
        [[]],
    )
    monkeypatch.setattr(model, "_read", lambda image, **kwargs: alternative)
    result = model._read_multiscale(Image.new("RGB", (3072, 1000)), initial)
    assert {line[1] for line in result.lines} == {
        "Last sentence",
        "A longer sentence in the preceding printed row",
    }


@pytest.mark.parametrize(
    "orientation,image_size,expected",
    [
        (0, (400, 300), (110, 220, 130, 230)),
        (90, (300, 400), (120, 320, 130, 340)),
        (180, (400, 300), (270, 320, 290, 330)),
        (270, (300, 400), (270, 210, 280, 230)),
    ],
)
def test_word_cells_return_to_original_page_after_rotation_and_crop(
    orientation: int, image_size: tuple[int, int], expected: tuple[float, ...]
) -> None:
    cells = _model()._to_cells(
        [(_box(20, 40, 60, 60), "123.45", 0.95)],
        BoundingBox(l=100, t=200, r=300, b=350),
        orientation,
        image_size,
        2,
    )
    assert cells[0].rect.to_bounding_box().as_tuple() == expected
