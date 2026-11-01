# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Qualify the Russian profile against frozen public scans and camera documents."""

import argparse
import hashlib
import importlib.metadata
import json
import re
import socket
import subprocess
import unicodedata
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

from docling_core.types.doc import TableItem
from pydantic import BaseModel, Field

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import ConversionStatus, InputFormat, Page
from docling.datamodel.ocr_profiles import OcrProfile, ocr_options_for_profile
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import (
    DocumentConverter,
    ImageFormatOption,
    PdfFormatOption,
)
from scripts.qualify_russian_ocr import edit_distance, normalize_text, sha256_file

FIXTURES = Path(__file__).resolve().parents[1] / "tests/data/russian_ocr_real"
ROI = tuple[float, float, float, float]


class TextReference(BaseModel):
    name: str
    roi: ROI
    max_cer: float = Field(ge=0, le=1)
    text: str = Field(min_length=1)


class RowReference(BaseModel):
    key: str
    values: list[str] = Field(min_length=1)


class FieldReference(BaseModel):
    name: str
    roi: ROI
    value: str


class Case(BaseModel):
    id: str = Field(pattern=r"^[a-z0-9-]+$")
    file: str
    page: int = Field(ge=1)
    kind: str
    source_url: str
    texts: list[TextReference]
    rows: list[RowReference]
    fields: list[FieldReference]


class References(BaseModel):
    scope: str
    exclusions: list[str]
    cases: list[Case] = Field(min_length=1)


class SourceReference(BaseModel):
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(alias="bytes", gt=0)
    url: str


class SourceManifest(BaseModel):
    reference_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    files: dict[str, SourceReference]


class ReferenceCorrection(BaseModel):
    case_id: str
    row_key: str
    value_index: int = Field(ge=0)
    original: str
    reviewed: str
    source_page: int = Field(ge=1)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence: str = Field(min_length=1)


class ReferenceAdjudications(BaseModel):
    reference_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    corrections: list[ReferenceCorrection] = Field(min_length=1)


def reviewed_references(
    references: References,
    manifest: SourceManifest,
    adjudications: ReferenceAdjudications,
) -> References:
    """Apply explicit source reviews to a copy, keeping the frozen answer intact."""
    if adjudications.reference_sha256 != manifest.reference_sha256:
        raise ValueError("Source reviews do not match the frozen reference version.")
    reviewed = references.model_copy(deep=True)
    reviewed.scope += (
        " Source transcription corrections were reviewed after the initial diagnostics."
        " This remains a development regression sample, not a blind evaluation."
    )
    seen: set[tuple[str, str, int]] = set()
    for correction in adjudications.corrections:
        identity = (correction.case_id, correction.row_key, correction.value_index)
        cases = [case for case in reviewed.cases if case.id == correction.case_id]
        if len(cases) != 1 or identity in seen:
            raise ValueError("Source review must identify one unique reference value.")
        case = cases[0]
        if (
            case.page != correction.source_page
            or manifest.files[case.file].sha256 != correction.source_sha256
        ):
            raise ValueError("Source review points to a different document or page.")
        rows = [row for row in case.rows if row.key == correction.row_key]
        if (
            len(rows) != 1
            or correction.value_index >= len(rows[0].values)
            or rows[0].values[correction.value_index] != correction.original
        ):
            raise ValueError(
                "Source review does not match the original expected value."
            )
        rows[0].values[correction.value_index] = correction.reviewed
        seen.add(identity)
    return reviewed


class Line(BaseModel):
    text: str
    confidence: float
    bbox: ROI


class TableEvidence(BaseModel):
    rows: list[list[str]]
    num_rows: int
    num_cols: int


class TextScore(BaseModel):
    name: str
    expected: str
    actual: str
    roi: ROI
    edits: int
    reference_characters: int
    cer: float
    wer: float
    limit: float
    passed: bool


class RowScore(BaseModel):
    key: str
    expected: list[str]
    actual: list[str]
    actual_row: list[str]
    table_index: int | None
    value_matches: list[bool]
    passed: bool


class FieldScore(BaseModel):
    name: str
    expected: str
    actual_region: str
    roi_number_correct: bool
    available_in_export: bool
    passed: bool


class CaseResult(BaseModel):
    id: str
    kind: str
    source_url: str
    file_sha256: str
    source_page: int
    status: str
    elapsed_seconds: float
    text_scores: list[TextScore]
    row_scores: list[RowScore]
    field_scores: list[FieldScore]
    tables: list[TableEvidence]
    passed: bool
    text_export_characters: int = 0
    error: str | None = None


def compact(value: str) -> str:
    return "".join(unicodedata.normalize("NFC", value).casefold().split())


def decimal_value(value: str) -> Decimal | None:
    candidate = compact(value)
    if candidate.startswith("(") and candidate.endswith(")"):
        candidate = "-" + candidate[1:-1]
    if re.fullmatch(r"-?\d+(?:[.,]\d+)?", candidate) is None:
        return None
    return Decimal(candidate.replace(",", "."))


def equal_value(expected: str, actual: str) -> bool:
    number = decimal_value(expected)
    if number is not None:
        return decimal_value(actual) == number
    return compact(expected) == compact(actual)


def region_text(lines: list[Line], roi: ROI) -> str:
    left, top, right, bottom = roi
    if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
        raise ValueError(f"Invalid control region: {roi}")
    selected = [
        line
        for line in lines
        if top <= (line.bbox[1] + line.bbox[3]) / 2 <= bottom
        and line.bbox[0] < right
        and line.bbox[2] > left
    ]
    # Use the same row grouping as the audited baseline. References and limits
    # stay frozen; right-aligned amounts are read after their row labels.
    selected.sort(key=lambda line: (line.bbox[1], line.bbox[0]))
    groups: list[list[Line]] = []
    for line in selected:
        for group in groups:
            if all(
                min(line.bbox[3], other.bbox[3]) - max(line.bbox[1], other.bbox[1])
                >= 0.5 * min(line.bbox[3] - line.bbox[1], other.bbox[3] - other.bbox[1])
                and min(line.bbox[2], other.bbox[2]) <= max(line.bbox[0], other.bbox[0])
                for other in group
            ):
                group.append(line)
                break
        else:
            groups.append([line])
    return " ".join(
        line.text
        for group in groups
        for line in sorted(group, key=lambda item: item.bbox[0])
    )


def check_field(
    reference: FieldReference, actual_region: str, exported: str
) -> FieldScore:
    # Terminal punctuation is allowed. Extra decimal digits are not a match.
    pattern = r"(?<!\d)(?<!\d[.,])-?\d+(?:[.,]\d+)?(?!\d|[.,]\d)"
    roi_correct = any(
        equal_value(reference.value, token)
        for token in re.findall(pattern, actual_region)
    )
    in_export = any(
        equal_value(reference.value, token) for token in re.findall(pattern, exported)
    )
    return FieldScore(
        name=reference.name,
        expected=reference.value,
        actual_region=actual_region,
        roi_number_correct=roi_correct,
        available_in_export=roi_correct and in_export,
        passed=roi_correct and in_export,
    )


def collect_lines(page: Page) -> list[Line]:
    if page.size is None or page.parsed_page is None:
        raise ValueError("The conversion did not retain OCR lines.")
    lines: list[Line] = []
    for cell in page.parsed_page.textline_cells:
        box = cell.rect.to_bounding_box().to_top_left_origin(page.size.height)
        lines.append(
            Line(
                text=cell.text,
                confidence=cell.confidence,
                bbox=(
                    box.l / page.size.width,
                    box.t / page.size.height,
                    box.r / page.size.width,
                    box.b / page.size.height,
                ),
            )
        )
    return lines


def check_row(reference: RowReference, tables: list[TableEvidence]) -> RowScore:
    best_values: list[str] = []
    best_row: list[str] = []
    best_matches = [False] * len(reference.values)
    best_table = None
    for table_index, table in enumerate(tables):
        for row in table.rows:
            for column, value in enumerate(row):
                if compact(value) != compact(reference.key):
                    continue
                trailing = [value for value in row[column + 1 :] if value.strip()]
                actual = trailing[-len(reference.values) :]
                matches = [
                    index < len(actual) and equal_value(expected, actual[index])
                    for index, expected in enumerate(reference.values)
                ]
                if best_table is None or sum(matches) > sum(best_matches):
                    best_values, best_row = actual, row
                    best_matches, best_table = matches, table_index
    return RowScore(
        key=reference.key,
        expected=reference.values,
        actual=best_values,
        actual_row=best_row,
        table_index=best_table,
        value_matches=best_matches,
        passed=all(best_matches),
    )


def deny_network(*args: object, **kwargs: object) -> None:
    raise RuntimeError("Python network connections are blocked during the OCR run.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--artifacts-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--use-source-review",
        action="store_true",
        help="Apply documented visual source corrections and retain both sets of scores.",
    )
    args = parser.parse_args()
    sources = args.sources.resolve(strict=True)
    references_path = FIXTURES / "references.json"
    expected_manifest = SourceManifest.model_validate_json(
        (FIXTURES / "sources.json").read_text(encoding="utf-8")
    )
    if sha256_file(references_path) != expected_manifest.reference_sha256:
        raise ValueError("The frozen reference bytes have changed.")
    references = References.model_validate_json(
        references_path.read_text(encoding="utf-8")
    )
    original_references = references.model_copy(deep=True)
    adjudications_path = FIXTURES / "reference_adjudications.json"
    adjudications = (
        ReferenceAdjudications.model_validate_json(
            adjudications_path.read_text(encoding="utf-8")
        )
        if args.use_source_review
        else None
    )
    if adjudications is not None:
        references = reviewed_references(references, expected_manifest, adjudications)
    effective_reference_bytes = references.model_dump_json(indent=2).encode("utf-8")
    destination = args.output
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("Existing results will not be overwritten.")
    destination.mkdir(parents=True, exist_ok=True)
    artifacts = args.artifacts_path.resolve(strict=True)
    files: dict[str, str] = {}
    for case in references.cases:
        source = sources / case.file
        if source.resolve().parent != sources:
            raise ValueError("Source must be directly inside the sources directory.")
        files[case.file] = sha256_file(source)
        expected_source = expected_manifest.files[case.file]
        if (
            files[case.file] != expected_source.sha256
            or source.stat().st_size != expected_source.size_bytes
        ):
            raise ValueError(f"Source bytes differ from the frozen sample: {case.file}")
    manifest = {
        "sealed_before_inference_utc": datetime.now(timezone.utc).isoformat(),
        "reference_sha256": sha256_file(references_path),
        "effective_reference_sha256": hashlib.sha256(
            effective_reference_bytes
        ).hexdigest(),
        "assessment_basis": "source_review"
        if adjudications is not None
        else "original_references",
        "adjudications_sha256": sha256_file(adjudications_path)
        if adjudications is not None
        else None,
        "runner_sha256": sha256_file(Path(__file__)),
        "code_sha256": {
            str(path.relative_to(FIXTURES.parents[2])): sha256_file(path)
            for path in sorted((FIXTURES.parents[2] / "docling").rglob("*.py"))
        },
        "file_sha256": files,
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "scope": references.scope,
        "python_network_connections_blocked": True,
        "native_network_capture_performed": False,
        "profile": "ru",
        "ocr_options": ocr_options_for_profile(OcrProfile.RUSSIAN).model_dump(
            mode="json"
        ),
        "versions": {
            name: importlib.metadata.version(name)
            for name in (
                "docling-slim",
                "docling-core",
                "docling-parse",
                "rapidocr",
                "onnxruntime",
                "torch",
                "tesserocr",
                "pymorphy3",
                "pymorphy3-dicts-ru",
            )
        },
    }
    (destination / "input-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (destination / "runner-used.py").write_bytes(Path(__file__).read_bytes())
    (destination / "references-used.json").write_bytes(references_path.read_bytes())
    (destination / "effective-references-used.json").write_bytes(
        effective_reference_bytes
    )
    if adjudications is not None:
        (destination / "reference-adjudications-used.json").write_bytes(
            adjudications_path.read_bytes()
        )
    options = PdfPipelineOptions(
        artifacts_path=artifacts,
        ocr_options=ocr_options_for_profile(OcrProfile.RUSSIAN),
        accelerator_options=AcceleratorOptions(
            device=AcceleratorDevice.CPU, num_threads=4
        ),
        generate_parsed_pages=True,
        document_timeout=180,
    )
    converter = DocumentConverter(
        allowed_formats=[InputFormat.PDF, InputFormat.IMAGE],
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=options),
            InputFormat.IMAGE: ImageFormatOption(pipeline_options=options),
        },
    )
    results: list[CaseResult] = []
    with (
        patch.dict("os.environ", {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}),
        patch.object(socket.socket, "connect", deny_network),
        patch.object(socket.socket, "connect_ex", deny_network),
        patch.object(socket, "create_connection", deny_network),
    ):
        for case in references.cases:
            case_dir = destination / case.id
            case_dir.mkdir()
            started = perf_counter()
            print(f"Start {case.id}, source page {case.page}", flush=True)
            try:
                conversion = converter.convert(
                    sources / case.file, page_range=(case.page, case.page)
                )
                elapsed = perf_counter() - started
                conversion.document.save_as_json(case_dir / "document.json")
                conversion.document.save_as_markdown(case_dir / "document.md")
                exported = conversion.document.export_to_text()
                (case_dir / "document.txt").write_text(exported, encoding="utf-8")
                if len(conversion.pages) != 1:
                    raise ValueError("Each control case must convert exactly one page.")
                lines = collect_lines(conversion.pages[0])
                (case_dir / "ocr-lines.json").write_text(
                    json.dumps(
                        [line.model_dump() for line in lines],
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                tables: list[TableEvidence] = []
                for item, _ in conversion.document.iterate_items():
                    if isinstance(item, TableItem):
                        rows = [
                            [""] * item.data.num_cols for _ in range(item.data.num_rows)
                        ]
                        for cell in item.data.table_cells:
                            rows[cell.start_row_offset_idx][
                                cell.start_col_offset_idx
                            ] = cell.text
                        tables.append(
                            TableEvidence(
                                rows=rows,
                                num_rows=item.data.num_rows,
                                num_cols=item.data.num_cols,
                            )
                        )
                text_scores: list[TextScore] = []
                for reference in case.texts:
                    actual = region_text(lines, reference.roi)
                    expected_normalized = normalize_text(reference.text)
                    actual_normalized = normalize_text(actual)
                    edits = edit_distance(expected_normalized, actual_normalized)
                    cer = edits / len(expected_normalized)
                    text_scores.append(
                        TextScore(
                            name=reference.name,
                            expected=reference.text,
                            actual=actual,
                            roi=reference.roi,
                            edits=edits,
                            reference_characters=len(expected_normalized),
                            cer=cer,
                            wer=edit_distance(
                                expected_normalized.split(), actual_normalized.split()
                            )
                            / len(expected_normalized.split()),
                            limit=reference.max_cer,
                            passed=cer <= reference.max_cer,
                        )
                    )
                row_scores = [check_row(reference, tables) for reference in case.rows]
                field_scores: list[FieldScore] = []
                for reference in case.fields:
                    actual = region_text(lines, reference.roi)
                    field_scores.append(check_field(reference, actual, exported))
                score = CaseResult(
                    id=case.id,
                    kind=case.kind,
                    source_url=case.source_url,
                    file_sha256=files[case.file],
                    source_page=case.page,
                    status=conversion.status.value,
                    elapsed_seconds=elapsed,
                    text_scores=text_scores,
                    row_scores=row_scores,
                    field_scores=field_scores,
                    tables=tables,
                    text_export_characters=len(exported),
                    passed=(
                        conversion.status == ConversionStatus.SUCCESS
                        and (bool(exported.strip()) or not (case.fields or case.texts))
                        and all(
                            check.passed
                            for check in [*text_scores, *row_scores, *field_scores]
                        )
                    ),
                )
            except Exception as error:
                # A diagnostic must record a failed case and continue with the other documents.
                score = CaseResult(
                    id=case.id,
                    kind=case.kind,
                    source_url=case.source_url,
                    file_sha256=files[case.file],
                    source_page=case.page,
                    status="exception",
                    elapsed_seconds=perf_counter() - started,
                    text_scores=[],
                    row_scores=[],
                    field_scores=[],
                    tables=[],
                    passed=False,
                    error=f"{type(error).__name__}: {error}",
                )
            results.append(score)
            (case_dir / "score.json").write_text(
                score.model_dump_json(indent=2), encoding="utf-8"
            )
            print(
                f"{case.id}: status={score.status}, CER={[round(item.cer * 100, 2) for item in score.text_scores]}, "
                f"table_values={sum(sum(row.value_matches) for row in score.row_scores)}/{sum(len(row.expected) for row in score.row_scores)}, "
                f"fields={sum(field.passed for field in score.field_scores)}/{len(score.field_scores)}, passed={score.passed}, error={score.error}",
                flush=True,
            )
    report = {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "scope": references.scope,
        "reference_sha256": manifest["reference_sha256"],
        "effective_reference_sha256": manifest["effective_reference_sha256"],
        "assessment_basis": manifest["assessment_basis"],
        "adjudications_sha256": manifest["adjudications_sha256"],
        "normalization": "NFC, casefold, collapsed whitespace for CER; no ё/e or Latin/Cyrillic correction. Numeric table comparisons ignore whitespace and accept decimal comma/point and equivalent negative notation.",
        "exclusions": references.exclusions,
        "passed_cases": sum(case.passed for case in results),
        "total_cases": len(results),
        "cases": [case.model_dump() for case in results],
    }
    if adjudications is not None:
        original_by_id = {case.id: case for case in original_references.cases}
        original_results: list[CaseResult] = []
        for result in results:
            original_rows = [
                check_row(row, result.tables) for row in original_by_id[result.id].rows
            ]
            original_results.append(
                result.model_copy(
                    update={
                        "row_scores": original_rows,
                        "passed": (
                            result.status == ConversionStatus.SUCCESS.value
                            and (
                                result.text_export_characters > 0
                                or not (
                                    original_by_id[result.id].fields
                                    or original_by_id[result.id].texts
                                )
                            )
                            and all(
                                check.passed
                                for check in [
                                    *result.text_scores,
                                    *original_rows,
                                    *result.field_scores,
                                ]
                            )
                        ),
                    }
                )
            )
        report["original_reference_results"] = {
            "scope": original_references.scope,
            "passed_cases": sum(case.passed for case in original_results),
            "cases": [case.model_dump() for case in original_results],
        }
        report["reference_corrections"] = adjudications.model_dump()
    (destination / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"Completed {len(results)} cases. Passed {report['passed_cases']}.",
        flush=True,
    )
    raise SystemExit(0 if all(case.passed for case in results) else 1)


if __name__ == "__main__":
    main()
