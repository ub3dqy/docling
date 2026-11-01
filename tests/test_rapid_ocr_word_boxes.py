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


def test_confident_sideways_form_is_reread_as_a_whole_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model()
    image = Image.new("RGB", (800, 1200))
    initial_lines = [
        (_box(20 + i * 30, 40, 35 + i * 30, 300), "Article", 0.98) for i in range(6)
    ]
    initial_lines.extend(
        (_box(20 + i * 30, 800, 35 + i * 30, 1100), "12345.67", 0.98) for i in range(9)
    )
    initial = _OcrPrediction(initial_lines, [[] for _ in initial_lines])
    upright_lines = [
        (_box(10, 10 + i * 30, 270, 25 + i * 30), "Article", 0.98) for i in range(6)
    ]
    upright_lines.extend(
        (_box(600, 10 + i * 30, 900, 25 + i * 30), "12345.67", 0.98) for i in range(9)
    )
    upright = _OcrPrediction(upright_lines, [[] for _ in upright_lines])
    seen = []

    def read(turned: Image.Image) -> _OcrPrediction:
        seen.append(turned.size)
        return upright

    monkeypatch.setattr(model, "_read", read)
    actual, turn = model._read_sideways_page(image, initial)
    assert actual is upright
    assert turn == 270
    assert seen == [(1200, 800)]
    # Ordinary upright text must not be reread or turned by this path.
    assert model._read_sideways_page(image, upright) == (upright, 0)
    assert len(seen) == 1


@pytest.mark.parametrize(
    "original,candidate",
    [
        ("1 565387", "1:565387"),
        ("-123.45", "123.45"),
        ("12345", "123.45"),
        ("99.99", "99.89"),
        ("=123.45", "123.45"),
        ("*123.45", "=123.45"),
        ("99.99 *5 =499.95", "99.99 5 =499.95"),
    ],
)
def test_supplementary_detector_cannot_replace_conflicting_numeric_values(
    monkeypatch: pytest.MonkeyPatch, original: str, candidate: str
) -> None:
    model = _model()
    model.options.use_region_refinement = True
    monkeypatch.setattr(model, "_refinement_reader", object(), raising=False)
    initial = _OcrPrediction([(_box(20, 40, 200, 60), original, 0.7)], [[]])
    alternative = _OcrPrediction([(_box(20, 40, 200, 60), candidate, 0.99)], [[]])
    monkeypatch.setattr(model, "_read", lambda *args, **kwargs: alternative)
    actual = model._refine_detection(Image.new("RGB", (240, 100)), initial)
    assert [line[1] for line in actual.lines] == [original]


def test_split_detector_boxes_cannot_bypass_numeric_replacement_protection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model()
    model.options.use_region_refinement = True
    monkeypatch.setattr(model, "_refinement_reader", object(), raising=False)
    initial = _OcrPrediction([(_box(20, 20, 220, 120), "123.45", 0.6)], [[]])
    alternative = _OcrPrediction(
        [
            (_box(20, 25, 220, 60), "123.46", 0.99),
            (_box(20, 80, 220, 115), "234.56", 0.99),
        ],
        [[], []],
    )
    monkeypatch.setattr(model, "_read", lambda *args, **kwargs: alternative)
    result = model._refine_detection(Image.new("RGB", (240, 150)), initial)
    assert [line[1] for line in result.lines] == ["123.45"]


def test_supplementary_detector_retains_small_digits_without_adding_lone_letter_noise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model()
    model.options.use_region_refinement = True
    monkeypatch.setattr(model, "_refinement_reader", object(), raising=False)
    alternative = _OcrPrediction(
        [(_box(20, 20, 30, 40), "в", 0.99), (_box(60, 20, 70, 40), "5", 0.99)], [[], []]
    )
    monkeypatch.setattr(model, "_read", lambda *args, **kwargs: alternative)
    result = model._refine_detection(
        Image.new("RGB", (100, 80)), _OcrPrediction([], [])
    )
    assert [line[1] for line in result.lines] == ["5"]


@pytest.mark.parametrize("longer,expanded", [("POCCWORD", True), ("RANDOMXX", False)])
def test_light_word_added_by_a_detector_gets_its_complete_source_region(
    monkeypatch: pytest.MonkeyPatch, longer: str, expanded: bool
) -> None:
    from PIL import ImageDraw

    model = _model()
    model.options.use_region_refinement = True
    monkeypatch.setattr(model, "_refinement_reader", object(), raising=False)
    image = Image.new("RGB", (600, 180), (70, 85, 100))
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 23, 220, 27), fill="white")
    draw.rectangle((300, 115, 430, 122), fill="white")
    anchor = (_box(300, 110, 430, 140), "Other", 0.99)
    partial = (_box(20, 20, 100, 40), "POCCX", 0.86)
    full = (_box(10, 10, 110, 20), longer, 0.55)

    def read(image: Image.Image, **kwargs) -> _OcrPrediction:
        lines = [partial, anchor] if image.width == 600 else [full]
        return _OcrPrediction(lines, [[line] for line in lines])

    monkeypatch.setattr(model, "_read", read)
    result = model._refine_detection(
        image,
        _OcrPrediction([anchor], [[anchor]]),
        detector_reference=image.resize((300, 90)),
    )
    index = next(i for i, line in enumerate(result.lines) if line[1] == "POCCX")
    assert model._line_bbox(result.lines[index]).r == (220 if expanded else 100)
    # Expanding the crop must not inject the weaker detector's proposed text.
    assert result.line_words[index][0][1] == "POCCX"


def test_overlapping_character_does_not_duplicate_a_price_in_the_joined_line() -> None:
    price = (_box(20, 20, 100, 50), "87.99", 0.96)
    duplicate = (_box(88, 21, 99, 49), "9", 0.7)
    quantity = (_box(125, 20, 200, 50), "2=175.98", 0.95)
    line = (_box(20, 20, 200, 50), "87.99 92=175.98", 0.9)
    result = RapidOcrModel._order_and_join([(line, [price, duplicate, quantity])])
    assert result.lines[0][1] == "87.99 2=175.98"
    assert [word[1] for word in result.words] == ["87.99", "2=175.98"]
    # An adjacent quantity uses separate pixels and must remain present.
    separate = (_box(104, 20, 119, 50), "9", 0.7)
    result = RapidOcrModel._order_and_join([(line, [price, separate, quantity])])
    assert [word[1] for word in result.words] == ["87.99", "9", "2=175.98"]
