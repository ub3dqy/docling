# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Compare existing recognizers on frozen image regions or detected lines.

The region mode supplies human control ROIs; the detected-line mode freezes
automatic quadrilaterals and rectifies their source pixels. Both bypass final
document assembly. Neither measures end-to-end accuracy or blind generalization.
"""

import argparse
import json
import socket
from pathlib import Path
from unittest.mock import patch

import numpy as np
from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.page import TextCell
from PIL import Image, ImageOps
from pydantic import BaseModel
from rapidocr.ch_ppocr_rec.typings import TextRecOutput

from docling.backend.image_backend import _ImagePageBackend, _oriented_frame
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import Cluster
from docling.datamodel.pipeline_options import RapidOcrOptions
from docling.models.stages.ocr.ocr_cyrillic_text import create_cyrillic_readers
from docling.models.stages.ocr.rapid_ocr_model import RapidOcrModel
from docling.utils.ocr_text_geometry import rectified_text_crop
from scripts.diagnose_russian_ocr import Snapshot, fingerprint
from scripts.qualify_real_russian_ocr import (
    FieldReference,
    Line,
    References,
    SourceManifest,
    check_field,
    deny_network,
    region_text,
)
from scripts.qualify_russian_ocr import sha256_file


class Reading(BaseModel):
    text: str
    confidence: float


class RegionProbe(BaseModel):
    case_id: str
    reference: FieldReference
    crop_size: tuple[int, int]
    source_pixels: tuple[int, int, int, int]
    readings: dict[str, Reading]


class LineProbe(BaseModel):
    case_id: str
    index: int
    bbox: tuple[float, float, float, float]
    crop_size: tuple[int, int]
    readings: dict[str, Reading]


class ScoredReading(BaseModel):
    text: str
    confidence: float | None = None
    number_correct: bool


class FieldProbeScore(BaseModel):
    case_id: str
    field: str
    expected: str
    readings: dict[str, ScoredReading]


class ProbeResult(BaseModel):
    scope: str
    total_fields: int
    total_lines: int | None = None
    correct: dict[str, int]
    rows: list[FieldProbeScore]


def detected_cells(snapshot: Snapshot) -> list[TextCell]:
    cells: dict[int, TextCell] = {}

    def visit(cluster: Cluster) -> None:
        for cell in cluster.cells:
            if cell.index in cells and cells[cell.index] != cell:
                raise ValueError(
                    "One detected cell index identifies different readings"
                )
            cells[cell.index] = cell
        for child in cluster.children:
            visit(child)

    for cluster in snapshot.clusters:
        visit(cluster)
    if len(cells) != len(snapshot.lines):
        raise ValueError("The trace does not cover every detected line")
    return [cells[index] for index in sorted(cells)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--artifacts-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--trace-run",
        type=Path,
        action="append",
        default=[],
        help="Read fixed automatic line quadrilaterals from these trace runs instead of control ROIs",
    )
    args = parser.parse_args()
    refs = References.model_validate_json(args.references.read_bytes())
    manifest = SourceManifest.model_validate_json(args.manifest.read_bytes())
    if sha256_file(args.references) != manifest.reference_sha256:
        raise ValueError("References do not match the frozen manifest")
    if any(case.file not in manifest.files for case in refs.cases):
        raise ValueError("A reference source is missing from the frozen manifest")
    for name, entry in manifest.files.items():
        path = (args.sources / name).resolve(strict=True)
        if (
            path.parent != args.sources.resolve(strict=True)
            or sha256_file(path) != entry.sha256
        ):
            raise ValueError(f"Invalid or changed source: {name}")
    cases = [case for case in refs.cases if case.fields]
    if not cases or any(Path(case.file).suffix.lower() == ".pdf" for case in cases):
        raise ValueError("This probe requires numeric fields on image sources")
    args.output.mkdir(parents=True, exist_ok=False)
    traces = {}
    for case in cases:
        if args.trace_run:
            candidates = [
                path
                for run in args.trace_run
                for path in (run / case.id).glob("*-layout.assigned.json")
            ]
            if len(candidates) != 1:
                raise ValueError(f"The case needs exactly one stage trace: {case.id}")
            traces[case.id] = candidates[0]

    def inputs() -> dict[str, object]:
        return fingerprint(
            args.references, args.manifest, args.sources, args.artifacts_path
        ) | {
            "probe_script": sha256_file(Path(__file__)),
            "trace_inputs": {
                case_id: {
                    "layout": sha256_file(path),
                    "document": sha256_file(path.parent / "document.json"),
                }
                for case_id, path in traces.items()
            },
        }

    before = inputs()
    (args.output / "baseline.json").write_text(
        json.dumps(before, indent=2), encoding="utf-8"
    )
    probes: list[RegionProbe] = []
    line_probes: list[LineProbe] = []
    completed_cases: set[str] = set()
    try:
        with (
            patch.dict(
                "os.environ", {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
            ),
            patch.object(socket.socket, "connect", deny_network),
            patch.object(socket.socket, "connect_ex", deny_network),
            patch.object(socket, "create_connection", deny_network),
        ):
            primary = RapidOcrModel(
                enabled=True,
                artifacts_path=args.artifacts_path,
                options=RapidOcrOptions(
                    lang=["iso:ru"], use_det=False, use_cls=False, text_score=0
                ),
                accelerator_options=AcceleratorOptions(
                    device=AcceleratorDevice.CPU, num_threads=4
                ),
            ).reader
            cyrillic = create_cyrillic_readers(args.artifacts_path, num_threads=4)

            def recognize(
                crop: Image.Image, *, pad_primary: bool
            ) -> dict[str, Reading]:
                result = primary(
                    np.asarray(
                        ImageOps.expand(crop, 6, fill="white") if pad_primary else crop
                    ),
                    use_det=False,
                    use_cls=False,
                    use_rec=True,
                )
                if (
                    not isinstance(result, TextRecOutput)
                    or not result.txts
                    or not result.scores
                ):
                    raise RuntimeError(
                        "The primary recognizer returned no text reading"
                    )
                readings = {
                    "v5-eslav": Reading(
                        text=result.txts[0], confidence=float(result.scores[0])
                    )
                }
                for name, reader in cyrillic.items():
                    path = reader.read_text_row(np.asarray(crop.convert("L")))
                    readings[f"v6-{name}"] = Reading(
                        text=path.text if path else "",
                        confidence=path.confidence if path else 0,
                    )
                return readings

            for case in cases:
                with Image.open(args.sources / case.file) as opened:
                    oriented, dpi = _oriented_frame(opened)
                    source = oriented.convert("RGB")
                if args.trace_run:
                    snapshot = Snapshot.model_validate_json(
                        traces[case.id].read_bytes()
                    )
                    document = DoclingDocument.model_validate_json(
                        (traces[case.id].parent / "document.json").read_bytes()
                    )
                    backend = _ImagePageBackend(source, 0, dpi)
                    size = backend.get_size()
                    if size != document.pages[case.page].size:
                        raise ValueError("Trace and source physical dimensions differ")
                    # Match the existing 144 DPI image view; keep every detected
                    # quadrilateral fixed across recognizers and rectify once.
                    gray = np.asarray(backend.get_page_image(scale=2).convert("L"))
                    for cell in detected_cells(snapshot):
                        aligned = rectified_text_crop(gray, cell.rect, size)
                        if aligned is None:
                            raise ValueError("A detected line could not be rectified")
                        box = cell.rect.to_bounding_box().to_top_left_origin(
                            size.height
                        )
                        crop = Image.fromarray(aligned).convert("RGB")
                        line_probes.append(
                            LineProbe(
                                case_id=case.id,
                                index=cell.index,
                                bbox=(
                                    box.l / size.width,
                                    box.t / size.height,
                                    box.r / size.width,
                                    box.b / size.height,
                                ),
                                crop_size=crop.size,
                                readings=recognize(crop, pad_primary=False),
                            )
                        )
                    completed_cases.add(case.id)
                    print("DONE LINES", case.id, flush=True)
                    continue
                for field_index, field in enumerate(case.fields):
                    left, top, right, bottom = field.roi
                    if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
                        raise ValueError(
                            f"Invalid region for {case.id}, field {field_index}"
                        )
                    pixels = (
                        int(left * source.width),
                        int(top * source.height),
                        int(right * source.width),
                        int(bottom * source.height),
                    )
                    crop = source.crop(pixels)
                    crop.save(args.output / f"{case.id}-{field_index:03}.png")
                    readings = recognize(crop, pad_primary=True)
                    probes.append(
                        RegionProbe(
                            case_id=case.id,
                            reference=field,
                            crop_size=crop.size,
                            source_pixels=pixels,
                            readings=readings,
                        )
                    )
                print("DONE", case.id, flush=True)
                completed_cases.add(case.id)
            # Freeze all predictions before consulting expected numeric values.
            (args.output / "readings.json").write_text(
                json.dumps(
                    [
                        probe.model_dump()
                        for probe in (line_probes if args.trace_run else probes)
                    ],
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
    finally:
        after = inputs()
        checks = {name: after[name] == value for name, value in before.items()}
        (args.output / "verification.json").write_text(
            json.dumps(
                {
                    "checks": checks,
                    "completed_fields": sum(
                        len(case.fields) for case in cases if case.id in completed_cases
                    ),
                    "completed_lines": len(line_probes),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        if not all(checks.values()):
            raise RuntimeError("Diagnostic inputs changed while reading the regions")
    rows: list[FieldProbeScore] = []
    if args.trace_run:
        if not line_probes:
            raise ValueError("The trace contains no readable line inputs")
        families = tuple(line_probes[0].readings)
        for case in cases:
            actual = {
                family: [
                    Line(
                        text=probe.readings[family].text,
                        confidence=probe.readings[family].confidence,
                        bbox=probe.bbox,
                    )
                    for probe in line_probes
                    if probe.case_id == case.id
                ]
                for family in families
            }
            for field in case.fields:
                rows.append(
                    FieldProbeScore(
                        case_id=case.id,
                        field=field.name,
                        expected=field.value,
                        readings={
                            family: ScoredReading(
                                text=region_text(lines, field.roi),
                                number_correct=check_field(
                                    field,
                                    region_text(lines, field.roi),
                                    region_text(lines, field.roi),
                                ).passed,
                            )
                            for family, lines in actual.items()
                        },
                    )
                )
        totals = {
            family: sum(row.readings[family].number_correct for row in rows)
            for family in families
        }
        output = ProbeResult(
            scope="Fixed automatically detected and rectified line quadrilaterals; every recognizer reads every line. No layout, table assembly, export, expected-answer selection, or retraining.",
            total_fields=len(rows),
            total_lines=len(line_probes),
            correct=totals,
            rows=rows,
        )
        (args.output / "result.json").write_text(
            output.model_dump_json(indent=2, exclude_none=True), encoding="utf-8"
        )
        print("CORRECT LINES", totals, "of", len(rows), flush=True)
        return
    rows = [
        FieldProbeScore(
            case_id=probe.case_id,
            field=probe.reference.name,
            expected=probe.reference.value,
            readings={
                name: ScoredReading(
                    text=reading.text,
                    confidence=reading.confidence,
                    number_correct=check_field(
                        probe.reference, reading.text, reading.text
                    ).passed,
                )
                for name, reading in probe.readings.items()
            },
        )
        for probe in probes
    ]
    totals = {
        name: sum(row.readings[name].number_correct for row in rows)
        for name in probes[0].readings
    }
    output = ProbeResult(
        scope="Frozen human-provided image control ROIs; no automatic localization, assembly, or export. Not an end-to-end or blind evaluation.",
        total_fields=len(probes),
        correct=totals,
        rows=rows,
    )
    (args.output / "result.json").write_text(
        output.model_dump_json(indent=2, exclude_none=True), encoding="utf-8"
    )
    print("CORRECT", totals, "of", len(probes), flush=True)


if __name__ == "__main__":
    main()
