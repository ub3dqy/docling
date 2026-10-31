# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Check Russian OCR against image-only fixtures without Python network access.

Run from the checkout after installing its ``russian`` extra and downloading
the models. Reports describe synthetic printed scans, not physical scanners.
"""

import argparse
import hashlib
import importlib.metadata
import socket
import unicodedata
from collections.abc import Sequence
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import pypdfium2
from docling_core.types.doc import DoclingDocument, TableItem, TextItem
from pydantic import BaseModel, SerializeAsAny

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.ocr_profiles import OcrProfile, ocr_options_for_profile
from docling.datamodel.pipeline_options import OcrOptions, PdfPipelineOptions
from docling.document_converter import (
    DocumentConverter,
    ImageFormatOption,
    PdfFormatOption,
)


class ReferenceCase(BaseModel):
    name: str
    filename: str
    page_no: int
    expected_text: str
    sha256: str
    table_rows: list[list[str]]
    max_character_error_rate: float


class ReferenceSet(BaseModel):
    pdf: str
    pdf_sha256: str
    cases: list[ReferenceCase]


class PageScore(BaseModel):
    source: str
    page_no: int
    case: str
    status: str
    character_error_rate: float
    word_error_rate: float
    character_edits: int
    reference_characters: int
    table_matches: bool
    expected_text: str
    recognized_text: str
    passed: bool


class QualificationReport(BaseModel):
    generated_utc: datetime
    scope: str = "Synthetic printed scans; no physical-scanner or handwriting claim."
    normalization: str = "NFC, casefold, collapsed whitespace; no script/ё correction."
    python_network_connections_blocked: bool = True
    native_network_capture_performed: bool = False
    ocr_engine: str
    ocr_options: SerializeAsAny[OcrOptions]
    versions: dict[str, str]
    model_sha256: dict[str, str]
    durations_seconds: dict[str, float]
    weighted_character_error_rate: float
    passed: bool
    pages: list[PageScore]


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, hashlib.sha256).hexdigest()


def normalize_text(text: str) -> str:
    """Ignore case and whitespace only; keep ё, punctuation and script errors."""
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def edit_distance(reference: Sequence[str], actual: Sequence[str]) -> int:
    previous: list[int] = list(range(len(actual) + 1))
    for row, expected in enumerate(reference, start=1):
        current: list[int] = [row]
        for column, observed in enumerate(actual, start=1):
            current.append(
                min(
                    previous[column] + 1,
                    current[column - 1] + 1,
                    previous[column - 1] + (expected != observed),
                )
            )
        previous = current
    return previous[-1]


def page_content(
    document: DoclingDocument, page_no: int
) -> tuple[str, list[list[list[str]]]]:
    parts: list[str] = []
    tables: list[list[list[str]]] = []
    for item, _ in document.iterate_items(page_no=page_no):
        if isinstance(item, TextItem):
            parts.append(item.text)
        elif isinstance(item, TableItem):
            rows = [[""] * item.data.num_cols for _ in range(item.data.num_rows)]
            for cell in item.data.table_cells:
                rows[cell.start_row_offset_idx][cell.start_col_offset_idx] = cell.text
            tables.append(rows)
            parts.extend(" ".join(cell for cell in row if cell) for row in rows)
    return "\n".join(parts), tables


def score_page(
    reference: ReferenceCase,
    document: DoclingDocument,
    *,
    source: str,
    page_no: int,
    status: ConversionStatus,
) -> PageScore:
    text, tables = page_content(document, page_no)
    expected = normalize_text(reference.expected_text)
    actual = normalize_text(text)
    edits = edit_distance(expected, actual)
    cer = edits / len(expected)
    wer = edit_distance(expected.split(), actual.split()) / len(expected.split())
    expected_table = [
        [normalize_text(cell) for cell in row] for row in reference.table_rows
    ]
    table_matches = not expected_table or any(
        [[normalize_text(cell) for cell in row] for row in table] == expected_table
        for table in tables
    )
    return PageScore(
        source=source,
        page_no=page_no,
        case=reference.name,
        status=status.value,
        character_error_rate=cer,
        word_error_rate=wer,
        character_edits=edits,
        reference_characters=len(expected),
        table_matches=table_matches,
        expected_text=reference.expected_text,
        recognized_text=text,
        passed=(
            status == ConversionStatus.SUCCESS
            and cer <= reference.max_character_error_rate
            and table_matches
        ),
    )


def verify_sources(fixtures: Path, reference: ReferenceSet) -> None:
    sources = fixtures / "sources"
    hashes = {reference.pdf: reference.pdf_sha256}
    hashes.update({case.filename: case.sha256 for case in reference.cases})
    for filename, expected in hashes.items():
        path = sources / filename
        if path.resolve().parent != sources.resolve():
            raise ValueError(f"Fixture must be directly inside sources: {filename}")
        if sha256_file(path) != expected:
            raise ValueError(f"Fixture hash differs: {filename}")
    with pypdfium2.PdfDocument(sources / reference.pdf) as pdf:
        if len(pdf) != len(reference.cases):
            raise ValueError("The PDF page count differs from the reference.")
        for page_no in range(len(pdf)):
            with (
                closing(pdf[page_no]) as page,
                closing(page.get_textpage()) as text_page,
            ):
                if text_page.count_chars() != 0:
                    raise ValueError("OCR fixtures must not contain native PDF text.")


def deny_network(*args: object, **kwargs: object) -> None:
    raise RuntimeError("Network access is disabled during OCR qualification.")


def qualify(
    fixtures: Path,
    artifacts: Path,
    output: Path,
    *,
    ocr_options: OcrOptions | None = None,
) -> list[PageScore]:
    reference = ReferenceSet.model_validate_json(
        (fixtures / "expected.json").read_text(encoding="utf-8")
    )
    verify_sources(fixtures, reference)
    artifacts = artifacts.resolve(strict=True)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use an empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    selected_ocr = (
        ocr_options
        if ocr_options is not None
        else ocr_options_for_profile(OcrProfile.RUSSIAN)
    )
    options = PdfPipelineOptions(
        artifacts_path=artifacts,
        ocr_options=selected_ocr,
        accelerator_options=AcceleratorOptions(
            device=AcceleratorDevice.CPU, num_threads=4
        ),
        document_timeout=180,
    )
    converter = DocumentConverter(
        allowed_formats=[InputFormat.PDF, InputFormat.IMAGE],
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=options),
            InputFormat.IMAGE: ImageFormatOption(pipeline_options=options),
        },
    )
    sources = [case.filename for case in reference.cases] + [reference.pdf]
    scores: list[PageScore] = []
    durations: dict[str, float] = {}
    with (
        patch.dict("os.environ", {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}),
        patch.object(socket.socket, "connect", deny_network),
        patch.object(socket.socket, "connect_ex", deny_network),
        patch.object(socket, "create_connection", deny_network),
    ):
        for source in sources:
            started = perf_counter()
            result = converter.convert(fixtures / "sources" / source)
            durations[source] = perf_counter() - started
            stem = Path(source).stem
            result.document.save_as_json(output / f"{stem}.json")
            result.document.save_as_markdown(output / f"{stem}.md")
            (output / f"{stem}.txt").write_text(
                result.document.export_to_text(), encoding="utf-8"
            )
            cases = (
                reference.cases
                if source == reference.pdf
                else [case for case in reference.cases if case.filename == source]
            )
            for case in cases:
                page_no = case.page_no if source == reference.pdf else 1
                scores.append(
                    score_page(
                        case,
                        result.document,
                        source=source,
                        page_no=page_no,
                        status=result.status,
                    )
                )
                print(
                    f"{source} page {page_no}: CER={scores[-1].character_error_rate:.2%}, "
                    f"tables={scores[-1].table_matches}, passed={scores[-1].passed}",
                    flush=True,
                )
    report = QualificationReport(
        generated_utc=datetime.now(timezone.utc),
        ocr_engine=selected_ocr.kind,
        ocr_options=selected_ocr,
        versions={
            name: importlib.metadata.version(name)
            for name in (
                "docling-slim",
                "docling-core",
                "torch",
                "rapidocr",
                "onnxruntime",
                "pypdfium2",
            )
        },
        model_sha256={
            str(path.relative_to(artifacts)): sha256_file(path)
            for path in sorted(artifacts.rglob("*"))
            if path.suffix in {".onnx", ".safetensors", ".pth"}
        },
        durations_seconds=durations,
        weighted_character_error_rate=(
            sum(score.character_edits for score in scores)
            / sum(score.reference_characters for score in scores)
        ),
        passed=all(score.passed for score in scores),
        pages=scores,
    )
    (output / "qualification.json").write_text(
        report.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, default=Path("tests/data/russian_ocr"))
    parser.add_argument("--artifacts-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scores = qualify(args.fixtures, args.artifacts_path, args.output)
    raise SystemExit(0 if all(score.passed for score in scores) else 1)


if __name__ == "__main__":
    main()
