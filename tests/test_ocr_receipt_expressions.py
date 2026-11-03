# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Receipt repairs need complete independent readings and actual source ink."""

import numpy as np
import pytest
from docling_core.types.doc import BoundingBox, CoordOrigin
from docling_core.types.doc.page import BoundingRectangle, TextCell
from PIL import Image, ImageDraw

from docling.backend.image_backend import _ImagePageBackend
from docling.datamodel.base_models import Page
from docling.models.stages.ocr.ocr_receipt_expressions import (
    choose_receipt_expression,
    compact_amount_marker,
    opaque_amount_marker,
    restore_receipt_separator,
    verify_receipt_page,
)
from docling.models.stages.ocr.ocr_receipt_regions import ReceiptPriceRefinements
from docling.models.stages.ocr.ocr_region_refiner import OcrRegionRefiner, RegionReading


def _readings(text: str) -> list[RegionReading]:
    return [
        RegionReading(text, 0.99, "cyrillic-v6-medium", "v6-source-1.0"),
        RegionReading(text, 0.99, "cyrillic-v6-tiny", "v6-clean-1.0"),
    ]


def test_quantity_and_unit_are_observed_before_recovering_a_merged_expression():
    assert (
        choose_receipt_expression("425.0017.", _readings("425.00*1шт."), confidence=0.8)
        == "425.00*1шт."
    )
    assert (
        choose_receipt_expression("325.0017.", _readings("425.00*1шт."), confidence=0.8)
        == "425.00*1шт."
    )


@pytest.mark.parametrize(
    "original,candidate",
    [
        ("25.00*2\u043a\u0433", "25.00*1\u043a\u0433"),
        ("25.00*2\u043a\u0433", "25.00*2м"),
        ("25.00*0.500\u043a\u0433", "25.00*0.5\u043a\u0433"),
        ("25.00*2\u043a\u0433", "250.00*2\u043a\u0433"),
        ("25.00*2\u043a\u0433", "35.10*2\u043a\u0433"),
        ("-25.00*2\u043a\u0433", "25.00*2\u043a\u0433"),
        ("25.0017.", "25.00"),
        ("25.0017.", "25.00*1"),
    ],
)
def test_expression_repair_retains_quantity_unit_sign_and_decimal_positions(
    original, candidate
):
    assert (
        choose_receipt_expression(original, _readings(candidate), confidence=0.8)
        == original
    )


def test_two_preprocessings_of_one_model_cannot_rewrite_an_expression():
    readings = [
        RegionReading("425.00*1шт.", 0.99, "cyrillic-v6-medium", variant)
        for variant in ("v6-source-1.0", "v6-clean-1.0")
    ]
    assert (
        choose_receipt_expression("325.0017.", readings, confidence=0.8) == "325.0017."
    )


def _marker(kind: str) -> np.ndarray:
    image = Image.new("L", (190, 52), 255)
    draw = ImageDraw.Draw(image)
    for left in (28, 45, 62, 79, 96, 128, 145):
        draw.rectangle((left, 24, left + 12, 37), fill=0)
    draw.line((0, 16, 170, 21), fill=0, width=2)
    draw.line((0, 42, 170, 47), fill=0, width=2)
    if kind == "filled":
        draw.rectangle((11, 26, 28, 34), fill=0)
    elif kind == "minus":
        draw.rectangle((11, 29, 28, 31), fill=0)
    elif kind == "colon":
        draw.rectangle((14, 26, 17, 28), fill=0)
        draw.rectangle((14, 32, 17, 34), fill=0)
    return np.asarray(image)


@pytest.mark.parametrize("kind", ["minus", "colon", "empty"])
def test_a_real_minus_colon_or_missing_mark_cannot_be_a_filled_separator(kind):
    assert not compact_amount_marker(_marker(kind))
    readings = [
        RegionReading("=425.00", 0.999, "cyrillic-v6-medium", variant)
        for variant in ("v6-source-1.0", "v6-clean-1.0")
    ] + [RegionReading("425.00", 0.99, "ch", "clean")]
    assert restore_receipt_separator("-425.00", readings, _marker(kind)) == "-425.00"


def test_a_filled_marker_needs_two_views_independent_digits_and_no_negative_vote():
    image = _marker("filled")
    assert compact_amount_marker(image)
    readings = [
        RegionReading("=425.00", 0.999, "cyrillic-v6-medium", variant)
        for variant in ("v6-source-1.0", "v6-clean-1.0")
    ]
    assert restore_receipt_separator("-425.00", readings, image) == "-425.00"
    readings.append(RegionReading("425.00", 0.99, "ch", "clean"))
    assert restore_receipt_separator("-425.00", readings, image) == "=425.00"
    assert restore_receipt_separator("-425.01", readings, image) == "-425.01"
    readings.append(RegionReading("-425.00", 0.99, "cyrillic-v6-tiny", "source"))
    assert restore_receipt_separator("-425.00", readings, image) == "-425.00"


def _detached_marker(kind: str, scale: int) -> np.ndarray:
    image = Image.new("L", (130, 40), 255)
    draw = ImageDraw.Draw(image)
    for left in (30, 47, 64, 88, 105):
        draw.rectangle((left, 10, left + 11, 29), fill=0)
    if kind == "filled":
        draw.rectangle((15, 14, 23, 24), fill=0)
    elif kind == "compact":
        draw.rectangle((15, 17, 23, 23), fill=0)
    elif kind == "minus":
        draw.rectangle((12, 19, 25, 21), fill=0)
    elif kind == "bold-minus":
        draw.rectangle((5, 16, 24, 23), fill=0)
    elif kind == "colon":
        draw.rectangle((19, 14, 21, 16), fill=0)
        draw.rectangle((19, 22, 21, 24), fill=0)
    return np.asarray(image.resize((130 * scale, 40 * scale)))


@pytest.mark.parametrize("scale", [1, 2, 3])
@pytest.mark.parametrize(
    "kind", ["filled", "compact", "minus", "bold-minus", "colon", "empty"]
)
def test_detached_source_mark_requires_independent_digits_and_source_equals(
    kind, scale
):
    image = _detached_marker(kind, scale)
    filled = kind in {"filled", "compact"}
    assert opaque_amount_marker(image) is filled
    readings = [
        RegionReading("=317.42", 0.99, "ch", "source"),
        RegionReading("317.42", 0.99, "cyrillic", "clean"),
    ]
    expected = "=317.42" if filled else "-317.42"
    assert restore_receipt_separator("-317.42", readings, image) == expected


def test_detached_mark_cannot_override_negative_or_disagreeing_source_readings():
    image = _detached_marker("filled", 2)
    source_equals = RegionReading("=317.42", 0.99, "ch", "source")
    independent = RegionReading("317.42", 0.99, "cyrillic", "clean")
    for readings in (
        [source_equals],
        [source_equals, RegionReading("317.42", 0.99, "ch", "clean")],
        [source_equals, RegionReading("317.43", 0.99, "cyrillic", "clean")],
        [RegionReading("=317.42", 0.99, "ch", "clean"), independent],
        [
            source_equals,
            independent,
            RegionReading("-317.42", 0.96, "iso:ru", "source"),
        ],
    ):
        assert restore_receipt_separator("-317.42", readings, image) == "-317.42"


def test_later_row_reading_preserves_verified_amount_without_freezing_other_fields():
    original = "Товар 55.30*2 =110.60 НДС 102"
    start = original.index("=")
    verified = ReceiptPriceRefinements()
    verified.record(7, original, (start, start + len("=110.60")))
    assert verified.preserves(7, "Описание 56,30*2 =110,60 НДС 10%")
    assert verified.preserves(7, "Описание 55.30*2 = 110.60 НДС 10%")
    assert verified.preserves(8, "Иная строка -110.60")
    for damaged in (
        "Товар 55.30*2 -110.60 НДС 102",
        "Товар 55.30*2 110.60 НДС 102",
        "Товар 55.30*2 =110.61 НДС 102",
        "Товар 110.60*2 =55.30 НДС 102",
        "Товар =110.60 НДС 102",
    ):
        assert not verified.preserves(7, damaged)


def test_verified_negative_amount_keeps_its_sign_and_each_repeated_position():
    original = "Возврат -36.80 НДС 6.13 Оплата -36.80"
    verified = ReceiptPriceRefinements()
    for start in (original.index("-"), original.rindex("-")):
        verified.record(4, original, (start, start + len("-36.80")))
    assert verified.preserves(4, "Возврат (36,80) НДС 7.14 Оплата \u221236.80")
    assert not verified.preserves(4, "Возврат 36.80 НДС 6.13 Оплата -36.80")
    assert not verified.preserves(4, "Возврат -36.80 НДС 6.13 Оплата 36.80")


def test_verified_separator_attached_to_quantity_is_not_dropped():
    original = "Товар 21.50*2=43.00"
    verified = ReceiptPriceRefinements()
    start = original.index("=")
    verified.record(2, original, (start, len(original)))
    assert verified.preserves(2, "Описание 21,50*2=43,00")
    assert not verified.preserves(2, "Товар 21.50*2 43.00")
    assert not verified.preserves(2, "Товар 21.50*2-43.00")


def test_row_letters_can_improve_with_the_verified_source_marker_and_unchanged_digits():
    original = "Товар 55.30*2 =110.60 НДС 102"
    start = original.index("=")
    verified = ReceiptPriceRefinements()
    verified.record(7, original, (start, start + len("=110.60")))
    readings = [
        RegionReading("Название 55.30*2 -110.60 НДС 10%", 0.9, "cyrillic", "source"),
        RegionReading("Название 55.30*2 110,60 НДС 10%", 0.9, "iso:ru", "source"),
        RegionReading("Название 55.30*2 -110.61 НДС 10%", 0.99, "ch", "source"),
        RegionReading("Название -110.60 НДС 10%", 0.99, "ch", "wide"),
    ]
    retained = verified.readings_for(7, readings)
    assert [r.text for r in retained] == [
        "Название 55.30*2 =110.60 НДС 10%",
        "Название 55.30*2 =110,60 НДС 10%",
    ]
    assert [r.engine for r in retained] == ["cyrillic", "iso:ru"]
    assert all(
        r.confidence == 0.9 and "+verified-marker" in r.variant for r in retained
    )
    assert readings[0].text == "Название 55.30*2 -110.60 НДС 10%"
    assert verified.readings_for(8, readings) == readings


@pytest.mark.parametrize("independent", [False, True])
def test_a_damaged_fiscal_caption_needs_independent_confirmation(
    monkeypatch, independent
):
    backend = _ImagePageBackend(Image.new("RGB", (400, 240)), 0, (72, 72))
    parsed = backend.get_segmented_page()
    texts = [
        "\u041a\u0410\u0421\u0421\u0418\u0420",
        "25.00",
        "70.00",
        "=45.00",
        "\u0421\u0423\u041d\u041d\u0410 \u041d\u0421 107",
    ]
    parsed.textline_cells = [
        TextCell(
            index=index,
            text=text,
            orig=text,
            from_ocr=True,
            confidence=0.8,
            rect=BoundingRectangle.from_bounding_box(
                BoundingBox.from_tuple(
                    (20, 20 + 30 * index, 200, 40 + 30 * index), CoordOrigin.TOPLEFT
                )
            ),
        )
        for index, text in enumerate(texts)
    ]
    page = Page(page_no=1, size=backend.get_size(), parsed_page=parsed)
    refiner = OcrRegionRefiner.__new__(OcrRegionRefiner)
    readings = [
        RegionReading(
            "\u0421\u0423\u041c\u041c\u0410 \u041d\u0414\u0421 10%",
            0.99,
            engine,
            variant,
        )
        for engine, variant in (
            ("cyrillic-v6-medium", "v6-source-1.0"),
            (
                "cyrillic-v6-tiny" if independent else "cyrillic-v6-medium",
                "v6-clean-1.0",
            ),
        )
    ]
    monkeypatch.setattr(refiner, "_cyrillic_readings", lambda gray: readings)
    assert verify_receipt_page(refiner, page, backend.get_page_image()) is independent
    # Classification does not replace damaged text with a guessed caption.
    assert page.parsed_page.textline_cells[-1].text == texts[-1]
    parsed.textline_cells = parsed.textline_cells[3:]
    assert not verify_receipt_page(refiner, page, backend.get_page_image())
