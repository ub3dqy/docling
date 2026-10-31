# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

# /// script
# requires-python = ">=3.11"
# dependencies = ["pillow>=13,<14", "reportlab>=4"]
# ///

# ruff: noqa: RUF001 -- Cyrillic characters are intentional OCR reference text.

"""Create reproducible image-only Russian OCR samples with known text.

Requires a font with Cyrillic glyphs. Font files are read locally and are not
embedded or redistributed. Existing sample files are never overwritten.
"""

import argparse
import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


@dataclass(frozen=True)
class ScanCase:
    name: str
    filename: str
    title: str
    lines: tuple[str, ...]
    table_rows: tuple[tuple[str, ...], ...]
    max_character_error_rate: float

    def reference_text(self) -> str:
        parts = [self.title, *self.lines]
        parts.extend(" ".join(cell for cell in row if cell) for row in self.table_rows)
        return "\n".join(parts)


def sample_cases() -> tuple[ScanCase, ...]:
    russian_lines = (
        "Система распознаёт печатные документы на русском языке.",
        "Важные буквы: Ё ё, Й й, Ж ж, Щ щ, Ъ ъ, Ы ы, Ь ь, Э э, Ю ю, Я я.",
        "Счёт № 2026-001 от 1 ноября 2026 года.",
        "Сумма к оплате: 12 345,67 руб.",
        "Договор действует до 31 декабря 2026 года.",
        "Это искусственный скан для проверки качества распознавания.",
    )
    return (
        ScanCase(
            name="russian",
            filename="russian.png",
            title="ПРОВЕРКА РУССКОГО ТЕКСТА",
            lines=russian_lines,
            table_rows=(),
            max_character_error_rate=0.025,
        ),
        ScanCase(
            name="mixed",
            filename="mixed.png",
            title="РУССКИЙ И АНГЛИЙСКИЙ",
            lines=(
                "Документ: Invoice 2026-042",
                "Получатель: Тестовая организация",
                "Product: Office Paper A4",
                "Количество: 25 units",
                "Цена: 380,00 RUB",
                "Итого: 9 500,00 RUB",
                "Contact: test@example.invalid",
                "Latin and Cyrillic text on one page.",
                "Все данные вымышлены и используются только для проверки.",
            ),
            table_rows=(),
            max_character_error_rate=0.05,
        ),
        ScanCase(
            name="table",
            filename="table.png",
            title="ТАБЛИЦА ЗАКУПОК",
            lines=("Тестовый документ. Все данные вымышлены.",),
            table_rows=(
                ("Наименование", "Количество", "Цена, руб.", "Сумма, руб."),
                ("Бумага", "5", "320,00", "1600,00"),
                ("Карандаш", "12", "25,00", "300,00"),
                ("Папка", "3", "90,00", "270,00"),
                ("Итого", "", "", "2170,00"),
            ),
            max_character_error_rate=0.05,
        ),
        ScanCase(
            name="degraded",
            filename="degraded.jpg",
            title="СКАН С НЕБОЛЬШИМ НАКЛОНОМ",
            lines=russian_lines,
            table_rows=(),
            max_character_error_rate=0.07,
        ),
    )


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    text: str,
    *,
    font: ImageFont.FreeTypeFont,
    x: int,
    y: int,
    width: int,
    line_height: int,
) -> int:
    line = ""
    for word in text.split():
        candidate = f"{line} {word}".strip()
        if line and draw.textlength(candidate, font=font) > width:
            draw.text((x, y), line, font=font, fill="black")
            y += line_height
            line = word
        else:
            line = candidate
    draw.text((x, y), line, font=font, fill="black")
    return y + line_height


def render_scan(case: ScanCase, font_path: Path, bold_font_path: Path) -> Image.Image:
    image = Image.new("RGB", (2480, 3508), "white")
    draw = ImageDraw.Draw(image)
    heading_font = ImageFont.truetype(str(bold_font_path), 66)
    body_font = ImageFont.truetype(str(font_path), 52)
    y = (
        draw_wrapped(
            draw,
            case.title,
            font=heading_font,
            x=200,
            y=280,
            width=2080,
            line_height=92,
        )
        + 130
    )
    for line in case.lines:
        y = (
            draw_wrapped(
                draw, line, font=body_font, x=200, y=y, width=2080, line_height=80
            )
            + 32
        )
    if case.table_rows:
        y += 110
        columns = (200, 1050, 1450, 1850, 2280)
        row_height = 150
        table_font = ImageFont.truetype(str(font_path), 46)
        header_font = ImageFont.truetype(str(bold_font_path), 44)
        for row_index, row in enumerate(case.table_rows):
            top = y + row_index * row_height
            for column, text in enumerate(row):
                draw.rectangle(
                    (columns[column], top, columns[column + 1], top + row_height),
                    outline="black",
                    width=4,
                )
                draw.text(
                    (columns[column] + 20, top + 48),
                    text,
                    font=header_font if row_index == 0 else table_font,
                    fill="black",
                )
    if case.name == "degraded":
        image = ImageEnhance.Contrast(image).enhance(0.72)
        image = image.filter(ImageFilter.GaussianBlur(radius=0.4))
        image = image.rotate(
            2.0, resample=Image.Resampling.BICUBIC, expand=False, fillcolor="white"
        )
        # A small fixed set of dust marks keeps the input reproducible and small.
        rng = random.Random(20261101)
        draw = ImageDraw.Draw(image)
        for _ in range(1500):
            point = (rng.randrange(image.width), rng.randrange(image.height))
            draw.point(point, fill=(170, 170, 170))
    return image


def generate_samples(output_dir: Path, font_path: Path, bold_font_path: Path) -> None:
    cases = sample_cases()
    planned = [output_dir / "sources" / case.filename for case in cases]
    planned.extend([output_dir / "sources/scans.pdf", output_dir / "expected.json"])
    existing = [str(path) for path in planned if path.exists()]
    if existing:
        raise FileExistsError(f"Sample files already exist: {', '.join(existing)}")
    if not font_path.is_file() or not bold_font_path.is_file():
        raise FileNotFoundError("Both font files must exist and support Cyrillic.")
    (output_dir / "sources").mkdir(parents=True, exist_ok=True)
    rows = []
    for page_no, case in enumerate(cases, start=1):
        image = render_scan(case, font_path, bold_font_path)
        path = output_dir / "sources" / case.filename
        if path.suffix == ".jpg":
            image.save(path, quality=78, dpi=(300, 300))
        else:
            image.save(path, dpi=(300, 300), optimize=True)
        image.close()
        rows.append(
            {
                **asdict(case),
                "page_no": page_no,
                "expected_text": case.reference_text(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    pdf_path = output_dir / "sources/scans.pdf"
    pdf = canvas.Canvas(str(pdf_path), pagesize=A4, pageCompression=1, invariant=1)
    pdf.setTitle("Synthetic Russian OCR scans")
    pdf.setAuthor("Test fixture")
    for case in cases:
        pdf.drawImage(
            str(output_dir / "sources" / case.filename),
            0,
            0,
            width=A4[0],
            height=A4[1],
        )
        pdf.showPage()
    pdf.save()
    manifest = {
        "version": 1,
        "description": "Synthetic printed scans, not physical-scanner evidence.",
        "dpi": 300,
        "font": font_path.name,
        "bold_font": bold_font_path.name,
        "pdf": "scans.pdf",
        "pdf_sha256": hashlib.sha256(pdf_path.read_bytes()).hexdigest(),
        "cases": rows,
    }
    (output_dir / "expected.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"Created {len(cases)} scan images and {pdf_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--font", required=True, type=Path)
    parser.add_argument("--bold-font", required=True, type=Path)
    args = parser.parse_args()
    generate_samples(args.output_dir, args.font, args.bold_font)


if __name__ == "__main__":
    main()
