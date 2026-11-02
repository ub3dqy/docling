# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Trace OCR stage boundaries without changing models, references, or predictions.

Run one conversion at a time in a dedicated process: the temporary instrumentation
patches stage methods process-wide. Outputs are diagnostic, not a new blind score.
"""

import argparse
import json
import socket
import subprocess
from collections.abc import Generator
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

from docling_core.types.doc import BoundingBox, DocItemLabel, TableItem
from pydantic import BaseModel, Field

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import (
    Cluster,
    InputFormat,
    LayoutPrediction,
    Page,
    Table,
)
from docling.datamodel.ocr_profiles import OcrProfile, ocr_options_for_profile
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableStructureOptions
from docling.document_converter import (
    DocumentConverter,
    ImageFormatOption,
    PdfFormatOption,
)
from docling.models.stages.layout.layout_object_detection_model import (
    LayoutObjectDetectionModel,
)
from docling.models.stages.layout.layout_postprocessing_model import (
    LayoutPostprocessingModel,
)
from docling.models.stages.ocr.ocr_region_refiner import OcrRegionRefiner
from docling.models.stages.ocr.rapid_ocr_model import RapidOcrModel
from docling.models.stages.page_assemble.page_assemble_model import PageAssembleModel
from docling.models.stages.table_structure.table_structure_model import (
    TableStructureModel,
)
from scripts.qualify_real_russian_ocr import (
    Case,
    FieldScore,
    Line,
    References,
    RowScore,
    SourceManifest,
    TableEvidence,
    check_field,
    check_row,
    collect_lines,
    deny_network,
    region_text,
)
from scripts.qualify_russian_ocr import edit_distance, normalize_text, sha256_file

PROJECT = Path(__file__).resolve().parents[1]


class Snapshot(BaseModel):
    stage: str
    lines: list[Line] = Field(default_factory=list)
    word_count: int = 0
    image_size: tuple[int, int] | None = None
    clusters: list[Cluster] = Field(default_factory=list)
    tables: list[Table] = Field(default_factory=list)
    assembled_text: str | None = None


class TextDiagnosticScore(BaseModel):
    name: str
    expected: str
    actual: str
    edits: int
    characters: int
    passed: bool


class StageScore(BaseModel):
    stage: str
    lines: int
    word_count: int
    fields: list[FieldScore]
    texts: list[TextDiagnosticScore]
    rows: list[RowScore]
    table_shapes: list[tuple[int, int]]


class Trace:
    def __init__(self) -> None:
        self.events: list[Snapshot] = []

    def page(self, stage: str, page: Page) -> Snapshot:
        snapshot = Snapshot(
            stage=stage,
            lines=collect_lines(page)
            if page.parsed_page is not None and page.size is not None
            else [],
            word_count=len(page.parsed_page.word_cells) if page.parsed_page else 0,
            clusters=page.predictions.layout.clusters
            if page.predictions.layout
            else [],
            tables=(
                list(page.predictions.tablestructure.table_map.values())
                if page.predictions.tablestructure
                else []
            ),
        )
        # Every stage must be frozen before subsequent in-place mutations.
        frozen = snapshot.model_copy(deep=True)
        self.events.append(frozen)
        return frozen

    def prediction(self, stage: str, image_size: tuple[int, int], prediction) -> None:
        if prediction is None:
            self.events.append(Snapshot(stage=stage, image_size=image_size))
            return
        width, height = image_size
        self.events.append(
            Snapshot(
                stage=stage,
                image_size=image_size,
                word_count=len(prediction.words),
                lines=[
                    Line(
                        text=text,
                        confidence=float(confidence),
                        bbox=(
                            min(x for x, _ in box) / width,
                            min(y for _, y in box) / height,
                            max(x for x, _ in box) / width,
                            max(y for _, y in box) / height,
                        ),
                    )
                    for box, text, confidence in prediction.lines
                ],
            )
        )


def evidence(tables: list[Table]) -> list[TableEvidence]:
    result = []
    for table in tables:
        rows = [[""] * table.num_cols for _ in range(table.num_rows)]
        for cell in table.table_cells:
            rows[cell.start_row_offset_idx][cell.start_col_offset_idx] = cell.text
        result.append(
            TableEvidence(rows=rows, num_rows=table.num_rows, num_cols=table.num_cols)
        )
    return result


def snapshot_score(case: Case, snapshot: Snapshot, exported: str) -> StageScore:
    fields = [
        check_field(item, region_text(snapshot.lines, item.roi), exported)
        for item in case.fields
    ]
    texts = []
    for item in case.texts:
        actual = region_text(snapshot.lines, item.roi)
        expected = normalize_text(item.text)
        edits = edit_distance(expected, normalize_text(actual))
        texts.append(
            TextDiagnosticScore(
                name=item.name,
                expected=item.text,
                actual=actual,
                edits=edits,
                characters=len(expected),
                passed=edits / len(expected) <= item.max_cer,
            )
        )
    rows = [check_row(item, evidence(snapshot.tables)) for item in case.rows]
    return StageScore(
        stage=snapshot.stage,
        lines=len(snapshot.lines),
        word_count=snapshot.word_count,
        fields=fields,
        texts=texts,
        rows=rows,
        table_shapes=[(item.num_rows, item.num_cols) for item in snapshot.tables],
    )


@contextmanager
def instrument(trace: Trace) -> Generator[None, None, None]:
    with ExitStack() as stack:
        original_read = RapidOcrModel._read

        @wraps(original_read)
        def read(model, image, *, reader=None):
            result = original_read(model, image, reader=reader)
            name = (
                "read.primary"
                if reader is None
                else "read.alternate-detector"
                if reader is model._refinement_reader
                else "read.reduced"
            )
            trace.prediction(name, image.size, result)
            return result

        stack.enter_context(patch.object(RapidOcrModel, "_read", read))
        original_detector = RapidOcrModel._refine_detection

        @wraps(original_detector)
        def detector(model, image, prediction, **kwargs):
            trace.prediction("multiscale.before-detector", image.size, prediction)
            result = original_detector(model, image, prediction, **kwargs)
            trace.prediction("multiscale.after-detector", image.size, result)
            return result

        stack.enter_context(patch.object(RapidOcrModel, "_refine_detection", detector))

        def wrap_pages(cls, original, stage: str) -> None:
            @wraps(original)
            def wrapped(model, conversion, pages):
                for page in original(model, conversion, pages):
                    event = trace.page(stage, page)
                    if page.assembled is not None:
                        event.tables = [
                            element.model_copy(deep=True)
                            for element in page.assembled.elements
                            if isinstance(element, Table)
                        ]
                        event.assembled_text = "\n".join(
                            element.text or "" for element in page.assembled.elements
                        )
                    yield page

            stack.enter_context(patch.object(cls, original.__name__, wrapped))

        def wrap_predictions(cls, original, stage: str) -> None:
            @wraps(original)
            def wrapped(model, conversion, pages):
                pages = list(pages)
                result = original(model, conversion, pages)
                for page, prediction in zip(pages, result, strict=True):
                    if stage == "layout.raw":
                        # Raw layout is returned before the caller attaches it.
                        snapshot = Snapshot(
                            stage=stage,
                            lines=collect_lines(page)
                            if page.parsed_page is not None and page.size is not None
                            else [],
                            clusters=prediction.clusters,
                        )
                        trace.events.append(snapshot.model_copy(deep=True))
                    else:
                        event = trace.page(stage, page)
                        if stage == "layout.assigned":
                            event.clusters = [
                                cluster.model_copy(deep=True)
                                for cluster in prediction.clusters
                            ]
                return result

            stack.enter_context(patch.object(cls, original.__name__, wrapped))

        wrap_pages(RapidOcrModel, RapidOcrModel.__call__, "ocr.cells")
        wrap_predictions(
            LayoutObjectDetectionModel,
            LayoutObjectDetectionModel.predict_layout,
            "layout.raw",
        )
        wrap_predictions(
            LayoutPostprocessingModel,
            LayoutPostprocessingModel.postprocess_layout,
            "layout.assigned",
        )
        wrap_predictions(
            TableStructureModel, TableStructureModel.predict_tables, "table.structured"
        )
        wrap_pages(PageAssembleModel, PageAssembleModel.__call__, "page.assembled")
        original_lines = OcrRegionRefiner.refine_lines

        @wraps(original_lines)
        def lines(refiner, page, *, image_input):
            trace.page("refine-lines.before", page)
            result = original_lines(refiner, page, image_input=image_input)
            trace.page("refine-lines.after", page)
            return result

        stack.enter_context(patch.object(OcrRegionRefiner, "refine_lines", lines))
        original_tables = OcrRegionRefiner.refine_tables

        @wraps(original_tables)
        def tables(refiner, page, elements, *, image_input):
            before = trace.page("refine-tables.before", page)
            before.tables = [
                element.model_copy(deep=True)
                for element in elements
                if isinstance(element, Table)
            ]
            result = original_tables(refiner, page, elements, image_input=image_input)
            after = trace.page("refine-tables.after", page)
            after.tables = [
                element.model_copy(deep=True)
                for element in elements
                if isinstance(element, Table)
            ]
            return result

        stack.enter_context(patch.object(OcrRegionRefiner, "refine_tables", tables))
        yield


def fingerprint(
    references: Path, manifest: Path, sources: Path, artifacts: Path
) -> dict[str, object]:
    def tree(root: Path, pattern: str = "*") -> dict[str, str]:
        return {
            path.relative_to(root).as_posix(): sha256_file(path)
            for path in sorted(root.rglob(pattern))
            if path.is_file() and ".cache" not in path.parts
        }

    return {
        "runtime": tree(PROJECT / "docling", "*.py"),
        "models": tree(artifacts),
        "references": sha256_file(references),
        "manifest": sha256_file(manifest),
        "sources": tree(sources),
        "evaluator": sha256_file(PROJECT / "scripts/qualify_real_russian_ocr.py"),
        "normalizer": sha256_file(PROJECT / "scripts/qualify_russian_ocr.py"),
        "dependencies": {
            name: sha256_file(PROJECT / name) for name in ("pyproject.toml", "uv.lock")
        },
        "diagnostic_script": sha256_file(Path(__file__)),
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True
        ).strip(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--artifacts-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--variant",
        choices=["full", "tableformer", "linear-layout"],
        default="full",
    )
    parser.add_argument("--case", action="append", default=[])
    args = parser.parse_args()
    references = References.model_validate_json(args.references.read_bytes())
    manifest = SourceManifest.model_validate_json(args.manifest.read_bytes())
    if sha256_file(args.references) != manifest.reference_sha256:
        raise ValueError("References do not match the sealed manifest")
    if any(case.file not in manifest.files for case in references.cases):
        raise ValueError("A reference source is missing from the sealed manifest")
    for name, entry in manifest.files.items():
        path = (args.sources / name).resolve(strict=True)
        if (
            path.parent != args.sources.resolve(strict=True)
            or sha256_file(path) != entry.sha256
        ):
            raise ValueError(f"Invalid or changed source: {name}")
    cases = [case for case in references.cases if not args.case or case.id in args.case]
    if not cases or set(args.case) - {case.id for case in cases}:
        raise ValueError("Unknown or empty case selection")
    args.output.mkdir(parents=True, exist_ok=False)
    baseline = fingerprint(
        args.references, args.manifest, args.sources, args.artifacts_path
    )
    (args.output / "baseline.json").write_text(
        json.dumps(baseline, indent=2), encoding="utf-8"
    )
    options = PdfPipelineOptions(
        artifacts_path=args.artifacts_path.resolve(strict=True),
        ocr_options=ocr_options_for_profile(OcrProfile.RUSSIAN),
        accelerator_options=AcceleratorOptions(
            device=AcceleratorDevice.CPU, num_threads=4
        ),
        generate_parsed_pages=True,
        document_timeout=180,
        table_structure_options=TableStructureOptions(
            prefer_ocr_grid=args.variant != "tableformer"
        ),
    )
    trace = Trace()
    summaries = []
    started = datetime.now(timezone.utc).isoformat()
    try:
        with (
            patch.dict(
                "os.environ", {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
            ),
            patch.object(socket.socket, "connect", deny_network),
            patch.object(socket.socket, "connect_ex", deny_network),
            patch.object(socket, "create_connection", deny_network),
            instrument(trace),
            ExitStack() as experiments,
        ):
            if args.variant == "linear-layout":
                # Controlled ablation, not a production routing rule: retain
                # the same OCR and remove all learned structural labels.
                def linear_layout(model, conversion, pages):
                    predictions = []
                    for page in pages:
                        if page.size is None:
                            raise ValueError("A page must have physical dimensions")
                        prediction = LayoutPrediction(
                            clusters=[
                                Cluster(
                                    id=0,
                                    label=DocItemLabel.TEXT,
                                    bbox=BoundingBox(
                                        l=0, t=0, r=page.size.width, b=page.size.height
                                    ),
                                )
                            ]
                        )
                        page.predictions.layout = prediction
                        predictions.append(prediction)
                    return predictions

                experiments.enter_context(
                    patch.object(
                        LayoutObjectDetectionModel, "predict_layout", linear_layout
                    )
                )
            converter = DocumentConverter(
                allowed_formats=[InputFormat.PDF, InputFormat.IMAGE],
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=options),
                    InputFormat.IMAGE: ImageFormatOption(pipeline_options=options),
                },
            )
            for case in cases:
                trace.events.clear()
                destination = args.output / case.id
                destination.mkdir()
                print(f"START {args.variant} {case.id}", flush=True)
                clock = perf_counter()
                try:
                    conversion = converter.convert(
                        args.sources / case.file, page_range=(case.page, case.page)
                    )
                finally:
                    # Retain the last completed stage if a conversion fails.
                    for index, event in enumerate(trace.events):
                        (destination / f"{index:02}-{event.stage}.json").write_text(
                            event.model_dump_json(indent=2), encoding="utf-8"
                        )
                if len(conversion.pages) != 1:
                    raise RuntimeError("Each diagnostic must return one source page")
                exported = conversion.document.export_to_text()
                conversion.document.save_as_json(destination / "document.json")
                (destination / "document.txt").write_text(exported, encoding="utf-8")
                final_tables = [
                    TableEvidence(
                        rows=[
                            ["" for _ in range(item.data.num_cols)]
                            for _ in range(item.data.num_rows)
                        ],
                        num_rows=item.data.num_rows,
                        num_cols=item.data.num_cols,
                    )
                    for item, _ in conversion.document.iterate_items()
                    if isinstance(item, TableItem)
                ]
                table_index = 0
                for item, _ in conversion.document.iterate_items():
                    if isinstance(item, TableItem):
                        for cell in item.data.table_cells:
                            final_tables[table_index].rows[cell.start_row_offset_idx][
                                cell.start_col_offset_idx
                            ] = cell.text
                        table_index += 1
                summary = {
                    "case_id": case.id,
                    "status": conversion.status.value,
                    "elapsed_seconds": perf_counter() - clock,
                    "stages": [
                        snapshot_score(case, event, exported).model_dump()
                        for event in trace.events
                    ],
                    "exported_row_scores": [
                        check_row(item, final_tables).model_dump() for item in case.rows
                    ],
                }
                summaries.append(summary)
                (destination / "summary.json").write_text(
                    json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                print(
                    f"DONE {case.id} {summary['elapsed_seconds']:.1f}s, {len(trace.events)} stages",
                    flush=True,
                )
    finally:
        current = fingerprint(
            args.references, args.manifest, args.sources, args.artifacts_path
        )
        checks = {name: current[name] == value for name, value in baseline.items()}
        verification = {
            "started_utc": started,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "variant": args.variant,
            "expected_cases": [case.id for case in cases],
            "completed_cases": [item["case_id"] for item in summaries],
            "checks": checks,
        }
        (args.output / "verification.json").write_text(
            json.dumps(verification, indent=2), encoding="utf-8"
        )
        print("INTEGRITY", checks, flush=True)
        if not all(checks.values()):
            raise RuntimeError("Diagnostic inputs changed while running")
    (args.output / "summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
