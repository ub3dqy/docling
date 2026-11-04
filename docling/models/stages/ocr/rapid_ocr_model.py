# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import logging
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from statistics import median
from typing import TYPE_CHECKING, Type, cast, get_args

import numpy
from docling_core.types.doc import BoundingBox, CoordOrigin
from docling_core.types.doc.page import TextCell
from PIL import Image

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat, Page
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import (
    OcrOptions,
    RapidOcrBackend,
    RapidOcrOptions,
)
from docling.datamodel.settings import settings
from docling.exceptions import (
    OcrLanguageNotSupportedError,
    RapidOcrModelSizeNotSupportedError,
)
from docling.models.base_ocr_model import BaseOcrModel
from docling.models.stages.ocr.ocr_line_classifier import (
    LineClassifierView,
    RapidReader as _RapidOCRReader,
)
from docling.utils.accelerator_utils import decide_device
from docling.utils.ocr_language import (
    OcrLanguage,
    OcrLanguageResolver,
    OcrLanguageSupport,
)
from docling.utils.ocr_reading_choice import (
    number_signature as _number_signature,
    protected_numbers,
    reading_may_replace,
)
from docling.utils.ocr_region_pixels import light_on_dark
from docling.utils.ocr_word_overlap import deduplicate_line_words
from docling.utils.orientation import ocr_quad_to_bounding_rectangle
from docling.utils.profiling import TimeRecorder
from docling.utils.utils import download_url_with_progress, safe_version

if TYPE_CHECKING:
    from rapidocr.utils.typings import EngineType, OCRVersion

_log = logging.getLogger(__name__)

# Default OCR language as a canonical tag, for the prefetch entry points.
_RAPIDOCR_DEFAULT_LANGUAGE = "ch"

# Docling's pre-canonicalization spellings for the two default recognizers, kept so that configs
# written before OCR languages were canonicalized keep working
_RAPIDOCR_LEGACY_ALIASES = {"chinese": "ch", "english": "en"}

# Recognition/detection model size for the PP-OCRv6 path; v4/v5 use "mobile".
_RAPIDOCR_DET_MODEL_LANG = "ch"
_RAPIDOCR_CLS_MODEL_LANG = "ch"
_RAPIDOCR_MODEL_TYPE = "small"
_RAPIDOCR_V4V5_MODEL_TYPE = "mobile"

# Inference backends docling supports, derived from the RapidOcrOptions.backend
# annotation so the two cannot drift.
_RAPIDOCR_BACKENDS: frozenset[str] = frozenset(get_args(RapidOcrBackend))

# detect_orientation: a line with at least this score is read with confidence, and a
# region with fewer lines is too small to judge its orientation.
_ORIENTATION_CONFIDENCE_SCORE = 0.8
_ORIENTATION_MIN_LINES = 3

# A line read by RapidOCR: its box as four corner points, its text and its score
_OcrLine = tuple[list[list[float]], str, float]
_RapidWord = tuple[str, float, list[list[float]] | None]
_RapidWordLine = tuple[_RapidWord, ...] | _RapidWord


@dataclass(frozen=True)
class _OcrPrediction:
    lines: list[_OcrLine]
    line_words: list[list[_OcrLine]]
    line_classifier_fallback: bool = False

    @property
    def words(self) -> list[_OcrLine]:
        return [word for group in self.line_words for word in group]


# Canonical BCP-47 to PP-OCR recognizer codes. The PP-OCRv6 set comes from the
# installed `rapidocr`; the v4/v5 sets below have no upstream equivalent to read,
# so they mirror the PP-OCR release notes summarised in `docs/concepts/OCR.md`.

# Recognition languages served by the PP-OCRv4 backbone (the torch fallback).
_PPOCRV4_CODES = frozenset(
    {"arabic", "cyrillic", "devanagari", "ka", "korean", "latin", "ta", "te"}
)

# Recognition languages served by the PP-OCRv5 backbone.
_PPOCRV5_CODES = frozenset(
    {
        "arabic",
        "ch",
        "cyrillic",
        "devanagari",
        "el",
        "en",
        "eslav",
        "korean",
        "latin",
        "ta",
        "te",
        "th",
    }
)

# Canonical tag -> PP-OCR code, for the languages whose code is not simply the primary subtag.
# `None` marks a tag that must *not* fall through to the generic rules below,
# because the code that looks right means something else.
_PPOCR_CANONICAL_TO_CODE_DEVIATIONS: dict[str, str | None] = {
    "zh-Hans": "ch",
    "zh-Hant": "chinese_cht",
    "ja-Jpan": "japan",
    "ko-Kore": "korean",
    "sr-Latn": "rs_latin",
    # `tl` is PP-OCR's code; BCP-47 canonicalizes Tagalog to `fil`.
    "fil-Latn": "tl",
    # PP-OCR serves East Slavic with a narrower recognizer than `cyrillic`.
    "ru-Cyrl": "eslav",
    "uk-Cyrl": "eslav",
    "be-Cyrl": "eslav",
    # PP-OCR's `ka` is Kannada; BCP-47 `ka` is Georgian.
    "kn-Knda": "ka",
    "ka-Geor": None,
}

# For PPOCRv4 and PPOCRv5 some languages are described not individually but using their scripts
# This is a mapping from the canonical ISO 15924 script to PP-OCR script-family code
_PPOCR_SCRIPT_TO_CODE: dict[str, str] = {
    "Latn": "latin",
    "Cyrl": "cyrillic",
    "Arab": "arabic",
    "Deva": "devanagari",
}

# Reverse of the language table
_PPOCR_CODE_TO_CANONICAL_DEVIATIONS: dict[str, list[str]] = {}
for tag, token in _PPOCR_CANONICAL_TO_CODE_DEVIATIONS.items():
    if token is not None:
        _PPOCR_CODE_TO_CANONICAL_DEVIATIONS.setdefault(token, []).append(tag)

# PP-OCRv6 codes that duplicate a language already reachable by its subtag
_PPOCR_REDUNDANT_CODES = frozenset({"french", "german"})


@dataclass(frozen=True)
class _RapidOcrArtifact:
    """One resolved checkpoint: where it lives locally and where it comes from"""

    # What gets handed to RapidOCR: the checkpoint file, or its directory for paddle.
    model_path: Path

    # Every local file the checkpoint needs, mapped to the URL it is downloaded from.
    files: dict[Path, str]

    # Recognition-keys file, for the entries that ship one (v6/v5 onnx embed the charset).
    dict_path: Path | None


@dataclass(frozen=True)
class _RapidOcrModelSpec:
    """One RapidOCR checkpoint set, as requested or as resolved"""

    # Docling inference backend name, one of _RAPIDOCR_BACKENDS.
    backend: str

    # Language exactly as the user wrote it
    user_lang: str | None = None

    # PP-OCR code the rapidocr registry expects, after normalization and aliasing.
    rapidocr_code: str | None = None

    # PP-OCR backbone that the (backend, language) pair resolves to.
    ppocr_version: "OCRVersion | None" = None


@lru_cache(maxsize=len(_RAPIDOCR_BACKENDS))
def _ppocr_supported_languages(vocabulary: frozenset[str]) -> OcrLanguageSupport:
    """What a PP-OCR code vocabulary can serve, as tags and as native codes."""
    tags: set[str] = set()
    native: set[str] = set()
    for code in vocabulary:
        if code in _PPOCR_REDUNDANT_CODES:
            continue
        # A code several tags deviate onto -- `eslav` is `ru`, `uk` and `be` --
        # is advertised as every one of them.
        deviations = _PPOCR_CODE_TO_CANONICAL_DEVIATIONS.get(code, [code])
        for deviation in deviations:
            language = OcrLanguageResolver.canonicalize_bcp47(
                deviation, raise_exception=False
            )
            if language is not None and _ppocr_code(language, vocabulary) == code:
                tags.add(language.short_tag())
            else:
                # A recognizer no tag can name -- the `latin`, `cyrillic`,
                # `arabic` and `devanagari` script models -- is offered as the
                # code itself, which is the only spelling that reaches it.
                native.add(code)
    return OcrLanguageSupport(bcp47=sorted(tags), native=sorted(native))


@lru_cache(maxsize=1)
def _installed_ppocrv6_codes() -> frozenset[str]:
    """The PP-OCRv6 recognition languages, read from the installed rapidocr.

    Asks `list_supported_langs`, which every rapidocr in the supported range
    provides; the `PP_OCRV6_LANGS` constant it replaces was removed in 3.10.0.
    A version that moves it has to say so rather than leave docling guessing
    at the vocabulary.
    """
    from rapidocr.utils.typings import OCRVersion, TaskType

    try:
        from rapidocr.utils.model_resolver import list_supported_langs
    except ImportError as err:
        raise ImportError(
            f"The installed rapidocr ({safe_version('rapidocr')}) does not expose "
            "list_supported_langs, which docling needs to resolve PP-OCR "
            "recognizers. Install a version in the supported range: "
            "rapidocr>=3.9.1,<4.0.0."
        ) from err
    return frozenset(list_supported_langs(TaskType.REC, OCRVersion.PPOCRV6))


@lru_cache(maxsize=len(_RAPIDOCR_BACKENDS))
def _rapidocr_vocabulary(backend: str) -> frozenset[str]:
    """PP-OCR codes a backend can serve: v6 plus its own v4/v5 fallback sets."""
    fallback = _PPOCRV4_CODES if backend == "torch" else _PPOCRV5_CODES | _PPOCRV4_CODES
    return _installed_ppocrv6_codes() | fallback


def _ppocr_code(language: OcrLanguage, vocabulary: frozenset[str]) -> str | None:
    """Map a canonical tag onto a PP-OCR code, or `None` if there is no model.

    `vocabulary` is the union of code sets the caller can actually reach, so
    the resolution never returns a code the backend cannot serve.
    """
    if language.is_passthrough():
        native = language.native
        alias = _RAPIDOCR_LEGACY_ALIASES.get(native or "")
        if alias is not None:
            _log.warning(
                "%r is a docling alias for the PP-OCR code %r. Write %r instead.",
                native,
                alias,
                alias,
            )
            native = alias
        return native if native in vocabulary else None

    if language.bcp47() in _PPOCR_CANONICAL_TO_CODE_DEVIATIONS:
        code = _PPOCR_CANONICAL_TO_CODE_DEVIATIONS[language.bcp47()]
        return code if code is not None and code in vocabulary else None

    # The bcp47 language matches the PPOCR code only when it uses its default script
    if language.has_default_script() and language.bcp47_language in vocabulary:
        return language.bcp47_language

    # Otherwise try to match the bcp47 script with the PPOCR script-based languages
    family = _PPOCR_SCRIPT_TO_CODE.get(language.bcp47_script or "")
    if family is not None and family in vocabulary:
        return family
    return None


def _parse_rapidocr_model_spec(value: str) -> _RapidOcrModelSpec:
    """Parse a `<backend>:<lang>` prefetch spec into its requested form.

    The split is on the first colon alone: a BCP-47 language carries one of its
    own, `onnxruntime:iso:th-Thai`, which is the spelling the model hands the user.
    """
    backend, separator, lang = value.partition(":")
    if not separator or not backend or not lang:
        raise ValueError(
            f"Invalid RapidOCR model spec {value!r}. "
            "Expected '<backend>:<lang>', e.g. 'onnxruntime:ch'."
        )
    if backend not in _RAPIDOCR_BACKENDS:
        raise ValueError(
            f"Unknown RapidOCR backend {backend!r} in {value!r}. "
            f"Supported: {sorted(_RAPIDOCR_BACKENDS)}."
        )
    try:
        _resolve_rapidocr(lang, backend)
    except (ValueError, OcrLanguageNotSupportedError) as err:
        raise ValueError(f"Invalid RapidOCR model spec {value!r}: {err}") from err
    return _RapidOcrModelSpec(backend=backend, user_lang=lang)


def _ppocr_version_for_code(code: str, backend: str) -> "OCRVersion":
    """Which PP-OCR backbone serves a code on this backend.

    Prefers PP-OCRv6 (whose recognizer covers ~52 codes). Torch then falls back
    to PP-OCRv4; the other backends try PP-OCRv5 first and PP-OCRv4 for the
    codes v5 lacks -- `ka`, PP-OCR's Kannada, is the only one.
    """
    from rapidocr.utils.typings import OCRVersion

    if code in _installed_ppocrv6_codes():
        return OCRVersion.PPOCRV6
    if backend == "torch":
        return OCRVersion.PPOCRV4
    return OCRVersion.PPOCRV5 if code in _PPOCRV5_CODES else OCRVersion.PPOCRV4


def _resolve_rapidocr(lang: str, backend: str) -> _RapidOcrModelSpec:
    """Map one language + backend onto a fully populated _RapidOcrModelSpec.

    `lang` may be one of PP-OCR's own codes or a BCP-47 tag behind `iso:`

    Raises:
        ValueError: `lang` is neither a PP-OCR code nor a valid BCP-47 tag.
        OcrLanguageNotSupportedError: No PP-OCR recognizer serves it on `backend`.

    Callers pass a single language; reducing a multi-language request is up to them.
    """
    language = OcrLanguageResolver.canonicalize_ocr_language(lang)
    code = _ppocr_code(language, _rapidocr_vocabulary(backend))
    if code is None:
        raise OcrLanguageNotSupportedError(
            f"RapidOCR (backend={backend})",
            language.tag(),
            supported=_ppocr_supported_languages(_rapidocr_vocabulary(backend)),
        )
    version = _ppocr_version_for_code(code, backend)

    _log.debug(
        "RapidOCR resolved lang=%r backend=%r -> version=%s rec_code=%r",
        lang,
        backend,
        version.value,
        code,
    )
    return _RapidOcrModelSpec(
        backend=backend,
        user_lang=lang,
        rapidocr_code=code,
        ppocr_version=version,
    )


def _backend_to_engine_type(backend: str) -> "EngineType":
    """Map a docling backend name onto the rapidocr EngineType it stands for."""
    from rapidocr.utils.typings import EngineType

    engine_types = {
        "onnxruntime": EngineType.ONNXRUNTIME,
        "openvino": EngineType.OPENVINO,
        "paddle": EngineType.PADDLE,
        "torch": EngineType.TORCH,
    }
    if backend not in engine_types:
        raise ValueError(
            f"Unknown RapidOCR backend {backend!r}. Supported: {sorted(_RAPIDOCR_BACKENDS)}."
        )
    return engine_types[backend]


def _rapidocr_artifacts(
    target_dir: Path,
    engine: "EngineType",
    version: "OCRVersion",
    rec_code: str,
    *,
    model_size: str = _RAPIDOCR_MODEL_TYPE,
    need_det: bool = True,
    need_cls: bool = True,
    need_rec: bool = True,
) -> dict[str, _RapidOcrArtifact]:
    """Resolve the det/cls/rec checkpoints for RapidOCR keyed by their task.

    `model_size` only applies to PP-OCRv6 det/rec; cls and non-v6 versions
    always use `_RAPIDOCR_V4V5_MODEL_TYPE` ("mobile").

    This is a pure registry lookup: no network and no filesystem I/O
    """
    from rapidocr.inference_engine.base import FileInfo, InferSession
    from rapidocr.utils.typings import EngineType, ModelType, OCRVersion, TaskType

    size = ModelType(
        model_size if version == OCRVersion.PPOCRV6 else _RAPIDOCR_V4V5_MODEL_TYPE
    )
    cls_size = ModelType(_RAPIDOCR_V4V5_MODEL_TYPE)

    file_infos: dict[str, FileInfo] = {}
    if need_det:
        file_infos["det"] = FileInfo(
            engine, version, TaskType.DET, _RAPIDOCR_DET_MODEL_LANG, size
        )
    if need_cls:
        file_infos["cls"] = FileInfo(
            engine,
            OCRVersion.PPOCRV4,
            TaskType.CLS,
            _RAPIDOCR_CLS_MODEL_LANG,
            cls_size,
        )
    if need_rec:
        file_infos["rec"] = FileInfo(engine, version, TaskType.REC, rec_code, size)

    artifacts: dict[str, _RapidOcrArtifact] = {}
    for task, file_info in file_infos.items():
        # Use RapidOCR's InferSession to get the URL for that FileInfo
        info = InferSession.get_model_url(file_info)
        model_url = info["model_dir"]

        files: dict[Path, str] = {}
        if engine == EngineType.PADDLE:
            # paddle ships a directory bundle; the "model path" is that directory.
            model_url = model_url.rstrip("/")
            model_path = target_dir / Path(model_url).name
            for name in info:
                if name in ("model_dir", "dict_url"):
                    continue
                files[model_path / name] = f"{model_url}/{name}"
        else:
            model_path = target_dir / Path(model_url).name
            files[model_path] = model_url

        dict_path: Path | None = None
        dict_url = info.get("dict_url")
        if dict_url:
            # v6/v5 onnx embed the charset, so only some entries ship a keys file.
            dict_path = target_dir / Path(dict_url).name
            files[dict_path] = dict_url

        artifacts[task] = _RapidOcrArtifact(
            model_path=model_path, files=files, dict_path=dict_path
        )
    return artifacts


def _validate_rapidocr_model_size(
    *,
    backend: str,
    language: str,
    ppocr_version: "OCRVersion",
    rec_code: str,
    model_size: str,
) -> None:
    """Confirm `model_size` has a PP-OCRv6 det/rec checkpoint for this language.

    Non-PP-OCRv6 versions ignore `model_size`; a non-default value just logs a
    warning. Uses rapidocr's own registry lookup rather than a hardcoded table.

    Raises:
        RapidOcrModelSizeNotSupportedError: no PP-OCRv6 det or rec checkpoint
            for this language at `model_size`.
    """
    from rapidocr.inference_engine.base import FileInfo, InferSession
    from rapidocr.utils.typings import ModelType, OCRVersion, TaskType

    if ppocr_version != OCRVersion.PPOCRV6:
        if model_size != "small":
            _log.warning(
                "RapidOCR model_size=%r only applies to PP-OCRv6; language %r "
                "resolves to %s, so the normal %s model assets will be used instead.",
                model_size,
                language,
                ppocr_version.value,
                ppocr_version.value,
            )
        return

    engine = _backend_to_engine_type(backend)
    size = ModelType(model_size)
    for task, code in (
        (TaskType.DET, _RAPIDOCR_DET_MODEL_LANG),
        (TaskType.REC, rec_code),
    ):
        try:
            InferSession.get_model_url(
                FileInfo(engine, ppocr_version, task, code, size)
            )
        except ValueError as err:
            raise RapidOcrModelSizeNotSupportedError(
                backend=backend,
                language=language,
                ppocr_version=ppocr_version.value,
                model_size=model_size,
                detail=str(err),
            ) from err


def _reads_well(lines: list[_OcrLine]) -> bool:
    """Whether enough lines were found and at least half are read with confidence."""
    confident = sum(
        1 for _, _, score in lines if score >= _ORIENTATION_CONFIDENCE_SCORE
    )
    return len(lines) >= _ORIENTATION_MIN_LINES and 2 * confident >= len(lines)


def _confident_chars(lines: list[_OcrLine]) -> int:
    return sum(
        len(text) for _, text, score in lines if score >= _ORIENTATION_CONFIDENCE_SCORE
    )


class RapidOcrModel(BaseOcrModel):
    _model_repo_folder = "RapidOcr"
    orientation_word_is_known: Callable[[str], bool] | None = None

    multiple_languages = False

    def __init__(
        self,
        enabled: bool,
        artifacts_path: Path | None,
        options: RapidOcrOptions,
        accelerator_options: AcceleratorOptions,
    ):
        super().__init__(
            enabled=enabled,
            artifacts_path=artifacts_path,
            options=options,
            accelerator_options=accelerator_options,
        )
        self.options: RapidOcrOptions
        self._multiscale_reader: _RapidOCRReader | None = None
        self._refinement_reader: _RapidOCRReader | None = None
        self._line_classifier_readers: dict[int, LineClassifierView] = {}

        # multiplier for 72 dpi; the default 3.0 == 216 dpi.
        self.scale = self.options.scale
        self._native_codes: list[str] = []

        if self.enabled:
            try:
                from rapidocr import ModelType, OCRVersion, RapidOCR  # type: ignore
            except ImportError:
                raise ImportError(
                    "RapidOCR is not installed. Please install it via `pip install rapidocr onnxruntime` to use this OCR engine. "
                    "Alternatively, Docling has support for other OCR engines. See the documentation."
                )

            # Decide the accelerator devices
            device = decide_device(accelerator_options.device)
            use_cuda = str(AcceleratorDevice.CUDA.value).lower() in device
            use_dml = accelerator_options.device == AcceleratorDevice.AUTO
            intra_op_num_threads = accelerator_options.num_threads
            gpu_id = 0
            if use_cuda and ":" in device:
                gpu_id = int(device.split(":")[1])
            backend_enum = _backend_to_engine_type(self.options.backend)

            # One language, warn-and-truncate and coverage checks all happen here.
            self._native_codes = self.resolve_ocr_languages()
            rec_code = self._native_codes[0]
            lang = (
                self.languages[0].tag()
                if self.languages
                else _RAPIDOCR_DEFAULT_LANGUAGE
            )
            ppocr_version = _ppocr_version_for_code(rec_code, self.options.backend)
            _validate_rapidocr_model_size(
                backend=self.options.backend,
                language=lang,
                ppocr_version=ppocr_version,
                rec_code=rec_code,
                model_size=self.options.model_size,
            )

            det_model_path = self.options.det_model_path
            cls_model_path = self.options.cls_model_path
            rec_model_path = self.options.rec_model_path
            rec_keys_path = self.options.rec_keys_path
            font_path = self.options.font_path

            # A pinned path that does not exist is a configuration error
            missing_pinned = [
                model_path
                for model_path in (
                    det_model_path,
                    cls_model_path,
                    rec_model_path,
                    rec_keys_path,
                    font_path,
                )
                if model_path is not None and not Path(model_path).exists()
            ]
            if missing_pinned:
                listed = "\n".join(f"  - {path}" for path in missing_pinned)
                raise FileNotFoundError(
                    f"The following RapidOCR paths do not exist:\n{listed}"
                )

            # Params forwarded to RapidOCR only in the library-managed flow (no artifacts_path)
            lang_params: dict[str, object] = {}

            if artifacts_path is not None:
                # artifacts_path means fully-offline operation
                target_dir = artifacts_path / self._model_repo_folder
                artifacts: dict[str, _RapidOcrArtifact] = _rapidocr_artifacts(
                    target_dir,
                    backend_enum,
                    ppocr_version,
                    rec_code,
                    model_size=self.options.model_size,
                    need_det=det_model_path is None,
                    need_cls=cls_model_path is None,
                    need_rec=rec_model_path is None,
                )
                missing = [
                    dest
                    for artifact in artifacts.values()
                    for dest in artifact.files
                    if not dest.is_file()
                ]
                if missing:
                    listed = "\n".join(f"  - {path}" for path in missing)
                    # `lang` is the canonical tag, which is what the prefetcher takes.
                    model_size_flag = (
                        f" --rapidocr-model-size {self.options.model_size}"
                        if self.options.model_size != "small"
                        else ""
                    )
                    raise FileNotFoundError(
                        "RapidOCR artifacts not found or incomplete in artifacts_path.\n"
                        f"Expected under: {target_dir}\n"
                        f"Resolved: backend={self.options.backend} "
                        f"ppocr_version={ppocr_version.value} rec_code={rec_code} "
                        f"model_size={self.options.model_size}\n"
                        f"Missing files:\n{listed}\n"
                        "Prefetch them with:\n"
                        f"  docling-tools models download rapidocr "
                        f"--rapidocr-backend-lang {self.options.backend}:{lang}"
                        f"{model_size_flag} "
                        f"-o {artifacts_path}\n"
                        "Or unset artifacts_path to let RapidOCR resolve and download "
                        "the checkpoints itself."
                    )

                if "det" in artifacts:
                    det_model_path = str(artifacts["det"].model_path)
                if "cls" in artifacts:
                    cls_model_path = str(artifacts["cls"].model_path)
                if "rec" in artifacts:
                    rec_model_path = str(artifacts["rec"].model_path)
                    if rec_keys_path is None and artifacts["rec"].dict_path is not None:
                        rec_keys_path = str(artifacts["rec"].dict_path)
            else:
                # Let RapidOCR resolve and cache the checkpoints itself
                size = ModelType(
                    self.options.model_size
                    if ppocr_version == OCRVersion.PPOCRV6
                    else _RAPIDOCR_V4V5_MODEL_TYPE
                )
                if det_model_path is None:
                    lang_params["Det.ocr_version"] = ppocr_version
                    lang_params["Det.lang_type"] = _RAPIDOCR_DET_MODEL_LANG
                    lang_params["Det.model_type"] = size
                if cls_model_path is None:
                    lang_params["Cls.ocr_version"] = OCRVersion.PPOCRV4
                    lang_params["Cls.lang_type"] = _RAPIDOCR_CLS_MODEL_LANG
                    lang_params["Cls.model_type"] = ModelType(_RAPIDOCR_V4V5_MODEL_TYPE)
                if rec_model_path is None:
                    lang_params["Rec.ocr_version"] = ppocr_version
                    lang_params["Rec.lang_type"] = rec_code
                    lang_params["Rec.model_type"] = size

            params = {
                # Global settings (these are still correct)
                # detect_orientation judges a reading by all its lines, so it applies
                # text_score itself, after that.
                "Global.text_score": 0.0
                if self.options.detect_orientation
                else self.options.text_score,
                "Global.font_path": font_path,
                "Global.log_level": "info" if self.options.print_verbose else "warning",
                # Engine-level ONNXRuntime settings
                "EngineConfig.onnxruntime.intra_op_num_threads": intra_op_num_threads,
                "EngineConfig.onnxruntime.use_cuda": use_cuda,
                "EngineConfig.onnxruntime.cuda_ep_cfg.device_id": gpu_id,
                # Engine-level OpenVINO settings
                "EngineConfig.openvino.inference_num_threads": intra_op_num_threads,
                # Detection model settings
                "Det.model_path": det_model_path,
                "Det.use_cuda": use_cuda,
                "Det.use_dml": use_dml,
                # Classification model settings
                "Cls.model_path": cls_model_path,
                "Cls.use_cuda": use_cuda,
                "Cls.use_dml": use_dml,
                # Recognition model settings
                "Rec.model_path": rec_model_path,
                "Rec.font_path": font_path,
                "Rec.rec_keys_path": rec_keys_path,
                "Rec.use_cuda": use_cuda,
                "Rec.use_dml": use_dml,
                "Det.engine_type": backend_enum,
                "Cls.engine_type": backend_enum,
                "Rec.engine_type": backend_enum,
                "EngineConfig.paddle.cpu_math_library_num_threads": intra_op_num_threads,
                "EngineConfig.paddle.use_cuda": use_cuda,
                "EngineConfig.paddle.cuda_ep_cfg.device_id": gpu_id,
                "EngineConfig.torch.use_cuda": use_cuda,
                "EngineConfig.torch.cuda_ep_cfg.device_id": gpu_id,
            }

            if self.options.rec_font_path is not None:
                _log.warning(
                    "The 'rec_font_path' option for RapidOCR is deprecated. Please use 'font_path' instead."
                )

            # Library-managed model-resolution params
            params.update(lang_params)

            user_params = self.options.rapidocr_params
            if user_params:
                _log.debug("Overwriting RapidOCR params with user-provided values.")
                params.update(user_params)

            def create_reader(config: Mapping[str, object]) -> RapidOCR:
                reader = RapidOCR(params=dict(config))
                if self.options.use_orientation_fallback:
                    # RapidOCR mutates Global call flags. Keep fallback flags
                    # in a separate lazy reader, never in the normal reader.
                    self._line_classifier_readers[id(reader)] = LineClassifierView(
                        RapidOCR(params=dict(config))
                    )
                return reader

            self.reader = create_reader(params)
            if (
                self.languages
                and self.languages[0].bcp47_language == "ru"
                and self.options.detect_orientation
                and self.options.use_region_refinement
            ):
                import pymorphy3

                self.orientation_word_is_known = pymorphy3.MorphAnalyzer().word_is_known
            if self.options.use_multiscale:
                # A small original scan needs the regular detector scale. The
                # reduced second reading needs larger detection glyphs, while
                # recognition still crops the original reduced image.
                self._multiscale_reader = create_reader(
                    params | {"Det.limit_side_len": 1536}
                )
                alternate_detector = (
                    (artifacts_path or settings.cache_dir / "models")
                    / self._model_repo_folder
                    / "PP-OCRv6_det_small.onnx"
                )
                if (
                    rec_code == "eslav"
                    and self.options.use_region_refinement
                    and self.options.use_det is not False
                    and self.options.det_model_path is None
                    and "Det.model_path" not in (self.options.rapidocr_params or {})
                    and alternate_detector.is_file()
                ):
                    self._refinement_reader = create_reader(
                        params | {"Det.model_path": str(alternate_detector)}
                    )

    def supported_ocr_languages(self) -> OcrLanguageSupport:
        return _ppocr_supported_languages(_rapidocr_vocabulary(self.options.backend))

    def resolve_ocr_languages(self) -> list[str]:
        # An empty `lang` list means "the engine's own default", which for PP-OCR
        # is the Simplified Chinese recognizer. Resolved the same way a request
        # is, so the recognizer this loads and the one the prefetch hint names
        # cannot drift apart.
        if not self.languages:
            default = OcrLanguageResolver.canonicalize_ocr_language(
                _RAPIDOCR_DEFAULT_LANGUAGE
            )
            code = _ppocr_code(default, _rapidocr_vocabulary(self.options.backend))
            assert code is not None, (
                f"the RapidOCR default {_RAPIDOCR_DEFAULT_LANGUAGE!r} has no "
                f"recognizer on backend {self.options.backend!r}"
            )
            return [code]
        return super().resolve_ocr_languages()

    def map_ocr_language(self, language: OcrLanguage) -> str:
        code = _ppocr_code(language, _rapidocr_vocabulary(self.options.backend))
        if code is None:
            raise OcrLanguageNotSupportedError(
                f"RapidOCR (backend={self.options.backend})",
                language.tag(),
                supported=self.supported_ocr_languages(),
            )
        return code

    @classmethod
    def download_models(
        cls,
        backend: str,
        local_dir: Path | None = None,
        force: bool = False,
        progress: bool = False,
        lang: str = _RAPIDOCR_DEFAULT_LANGUAGE,
        model_size: str = _RAPIDOCR_MODEL_TYPE,
    ) -> Path:
        if local_dir is None:
            local_dir = settings.cache_dir / "models" / cls._model_repo_folder
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)

        resolved = _resolve_rapidocr(lang, backend)
        assert resolved.ppocr_version is not None
        assert resolved.rapidocr_code is not None
        _validate_rapidocr_model_size(
            backend=backend,
            language=lang,
            ppocr_version=resolved.ppocr_version,
            rec_code=resolved.rapidocr_code,
            model_size=model_size,
        )

        engine = _backend_to_engine_type(backend)
        for artifact in _rapidocr_artifacts(
            local_dir,
            engine,
            resolved.ppocr_version,
            resolved.rapidocr_code,
            model_size=model_size,
        ).values():
            for dest, url in artifact.files.items():
                if dest.exists() and not force:
                    continue
                # paddle checkpoints are directory bundles, so dest may be nested.
                dest.parent.mkdir(parents=True, exist_ok=True)
                buf = download_url_with_progress(url, progress=progress)
                with dest.open("wb") as fw:
                    fw.write(buf.read())
        return local_dir

    def __call__(
        self, conv_res: ConversionResult, page_batch: Iterable[Page]
    ) -> Iterable[Page]:
        if not self.enabled:
            yield from page_batch
            return

        for page in page_batch:
            assert page._backend is not None
            if not page._backend.is_valid():
                yield page
            else:
                with TimeRecorder(conv_res, "ocr"):
                    ocr_rects = self.get_ocr_rects(page)
                    scale = self._render_scale(page, conv_res.input.format)

                    all_ocr_cells = []
                    all_ocr_words = []
                    for ocr_rect in ocr_rects:
                        # Skip zero area boxes
                        if ocr_rect.area() == 0:
                            continue
                        high_res_image = page._backend.get_page_image(
                            scale=scale, cropbox=ocr_rect
                        )
                        result = self._read(high_res_image)
                        if result is None:
                            if (
                                self.options.detect_orientation
                                and self.orientation_word_is_known is not None
                            ):
                                # An upside-down Russian page can produce no
                                # lines. Keep it eligible for page comparison.
                                result = _OcrPrediction([], [])
                            else:
                                _log.warning("RapidOCR returned empty result!")
                                continue
                        orientation, im_size = 0, high_res_image.size
                        if self.options.use_region_refinement:
                            result, orientation = self._read_sideways_page(
                                high_res_image, result
                            )
                            if orientation:
                                high_res_image = high_res_image.rotate(
                                    -orientation, expand=True
                                )
                                im_size = high_res_image.size
                        if (
                            self.options.detect_orientation
                            and not orientation
                            and self.orientation_word_is_known is not None
                        ):
                            result, orientation, im_size = self._read_upright(
                                high_res_image, result
                            )
                            if orientation:
                                high_res_image = high_res_image.rotate(
                                    -orientation, expand=True
                                )
                        if self.options.use_multiscale:
                            reference = (
                                page._backend.get_page_image(scale=1, cropbox=ocr_rect)
                                if conv_res.input.format == InputFormat.IMAGE
                                and self.options.use_region_refinement
                                else None
                            )
                            if reference is not None and orientation:
                                reference = reference.rotate(-orientation, expand=True)
                            result = self._read_multiscale(
                                high_res_image,
                                result,
                                refine_detection=conv_res.input.format
                                == InputFormat.IMAGE,
                                detector_reference=reference,
                                line_classification=True
                                if result.line_classifier_fallback
                                else None,
                            )
                        if (
                            self.options.detect_orientation
                            and not orientation
                            and self.orientation_word_is_known is None
                        ):
                            result, orientation, im_size = self._read_upright(
                                high_res_image, result
                            )

                        del high_res_image

                        if self.options.use_cls is False:
                            from docling.models.stages.ocr.ocr_page_orientation import (
                                restore_vertical_crop_axes,
                            )

                            result = restore_vertical_crop_axes(result)
                        all_ocr_cells.extend(
                            self._to_cells(
                                result.lines, ocr_rect, orientation, im_size, scale
                            )
                        )
                        all_ocr_words.extend(
                            self._to_cells(
                                result.words, ocr_rect, orientation, im_size, scale
                            )
                        )

                    # Post-process the cells
                    self.post_process_cells(all_ocr_cells, page, conv_res)
                    if self.options.use_word_boxes:
                        assert page.parsed_page is not None
                        # Do not share a last-result cache: one engine may serve
                        # concurrent conversions. Keep words with this page only.
                        words = [
                            cell
                            for cell in page.parsed_page.word_cells
                            if not cell.from_ocr
                        ]
                        words.extend(all_ocr_words)
                        for index, word in enumerate(words):
                            word.index = index
                        page.parsed_page.word_cells = words
                        page.parsed_page.has_words = bool(words)

                # DEBUG code:
                if settings.debug.visualize_ocr:
                    self.draw_ocr_rects_and_cells(conv_res, page, ocr_rects)

                yield page

    def _render_scale(self, page: Page, input_format: InputFormat) -> float:
        scale = (
            self.options.image_scale
            if input_format == InputFormat.IMAGE
            and self.options.image_scale is not None
            else self.scale
        )
        if (
            input_format == InputFormat.IMAGE
            and self.options.preserve_native_resolution
            and page._backend is not None
            and page.size is not None
        ):
            native = page._backend.get_native_image_size()
            if native is not None:
                required = max(
                    native[0] / page.size.width, native[1] / page.size.height
                )
                # Compensate small high-DPI photos without forcing a dense
                # full-page scan beyond the established 1536-pixel OCR view.
                # The explicitly configured scale is still a lower bound.
                scale = max(scale, min(required, 1536 / max(page.size.as_tuple())))
        return scale

    def _to_cells(
        self,
        observations: list[_OcrLine],
        offset: BoundingBox,
        orientation: int,
        image_size: tuple[int, int],
        scale: float,
    ) -> list[TextCell]:
        return [
            TextCell(
                index=index,
                text=text,
                orig=text,
                confidence=score,
                from_ocr=True,
                rect=ocr_quad_to_bounding_rectangle(
                    box,
                    original_offset=offset,
                    scale=scale,
                    orientation=orientation,
                    im_size=image_size,
                ),
            )
            for index, (box, text, score) in enumerate(observations)
        ]

    def _read(
        self,
        image: Image.Image,
        *,
        reader: "_RapidOCRReader | None" = None,
    ) -> _OcrPrediction | None:
        from rapidocr.utils.output import RapidOCROutput

        selected_reader = self.reader if reader is None else reader
        result = selected_reader(
            numpy.asarray(image),
            use_det=self.options.use_det,
            use_cls=self.options.use_cls,
            use_rec=self.options.use_rec,
            return_word_box=self.options.use_word_boxes,
        )
        if (
            not isinstance(result, RapidOCROutput)
            or result.boxes is None
            or result.txts is None
            or result.scores is None
        ):
            return None
        lines = list(zip(result.boxes.tolist(), result.txts, result.scores))
        line_words: list[list[_OcrLine]] = [[] for _ in lines]
        # RapidOCR 3.10 annotates word_results as a flat triple, but returns
        # one group of triples per line (or a flat empty-output sentinel).
        word_results = cast("tuple[_RapidWordLine, ...]", result.word_results)
        if self.options.use_word_boxes and len(word_results) == len(lines):
            for index, (line, word_line) in enumerate(zip(lines, word_results)):
                # RapidOCR's empty-output sentinel is a flat triple. It has no
                # word coordinates; leave that line available as a fallback.
                if (
                    line[2] < self.options.text_score
                    or not word_line
                    or isinstance(word_line[0], str)
                ):
                    continue
                complete_words = [
                    (numpy.asarray(box).tolist(), text, float(score))
                    for text, score, box in cast("tuple[_RapidWord, ...]", word_line)
                    if box is not None
                ]
                # Partial word output must not replace a complete table line.
                if "".join(
                    "".join(word[1].split()) for word in complete_words
                ) == "".join(line[1].split()):
                    lines[index], line_words[index] = deduplicate_line_words(
                        line, complete_words
                    )
        return _OcrPrediction(
            lines=lines,
            line_words=line_words,
            line_classifier_fallback=isinstance(selected_reader, LineClassifierView),
        )

    def _read_configured(
        self,
        image: Image.Image,
        *,
        reader: _RapidOCRReader | None = None,
        line_classification: bool | None = None,
    ) -> _OcrPrediction | None:
        if line_classification is True:
            selected = self.reader if reader is None else reader
            fallback = self._line_classifier_readers.get(id(selected))
            if fallback is None:
                raise RuntimeError("Line-direction fallback reader is not configured")
            return self._read(image, reader=fallback)
        return self._read(image, reader=reader)

    def _read_multiscale(
        self,
        image: Image.Image,
        prediction: _OcrPrediction,
        *,
        refine_detection: bool = True,
        detector_reference: Image.Image | None = None,
        line_classification: bool | None = None,
    ) -> _OcrPrediction:
        ratio = 1536 / max(image.size)
        if ratio >= 0.9:
            ordered = self._order_and_join(
                list(zip(prediction.lines, prediction.line_words))
            )
            return (
                self._refine_detection(
                    image,
                    ordered,
                    detector_reference=detector_reference,
                    line_classification=line_classification,
                )
                if refine_detection
                else ordered
            )
        reduced = image.resize(
            (max(1, round(image.width * ratio)), max(1, round(image.height * ratio))),
            Image.Resampling.LANCZOS,
        )
        alternative = self._read_configured(
            reduced,
            reader=self._multiscale_reader,
            line_classification=line_classification,
        )
        if alternative is None:
            return (
                self._refine_detection(
                    image,
                    prediction,
                    detector_reference=detector_reference,
                    line_classification=line_classification,
                )
                if refine_detection
                else prediction
            )
        sx, sy = image.width / reduced.width, image.height / reduced.height

        def rescale(observation: _OcrLine) -> _OcrLine:
            box, text, score = observation
            return [[x * sx, y * sy] for x, y in box], text, score

        selected = list(zip(prediction.lines, prediction.line_words))
        for candidate, words in zip(alternative.lines, alternative.line_words):
            candidate = rescale(candidate)
            conflicts = [
                index
                for index, (line, _) in enumerate(selected)
                if self._same_text_region(candidate, line)
            ]
            if conflicts:
                if not reading_may_replace(
                    [selected[index] for index in conflicts],
                    candidate,
                    preserve_numbers=False,
                ):
                    continue
                selected = [
                    item
                    for index, item in enumerate(selected)
                    if index not in conflicts
                ]
            elif candidate[2] < 0.8:
                continue
            selected.append((candidate, [rescale(word) for word in words]))
        merged = self._order_and_join(selected)
        return (
            self._refine_detection(
                image,
                merged,
                detector_reference=detector_reference,
                line_classification=line_classification,
            )
            if refine_detection
            else merged
        )

    def _refine_detection(
        self,
        image: Image.Image,
        prediction: _OcrPrediction,
        *,
        detector_reference: Image.Image | None = None,
        line_classification: bool | None = None,
    ) -> _OcrPrediction:
        """Recover small missed lines with a second local detector, retaining words."""
        if not self.options.use_region_refinement or self._refinement_reader is None:
            return prediction
        alternative = self._read_configured(
            image,
            reader=self._refinement_reader,
            line_classification=line_classification,
        )
        if alternative is None:
            return prediction
        if len(prediction.lines) <= 12 and any(
            len(line[1]) >= 4
            and line[1].isalpha()
            and light_on_dark(
                numpy.asarray(image.crop(self._line_bbox(line).as_tuple()).convert("L"))
            )
            for line in prediction.lines
        ):
            reduced = (
                detector_reference
                if detector_reference is not None
                else image.resize(
                    (max(1, image.width // 2), max(1, image.height // 2)),
                    Image.Resampling.LANCZOS,
                )
            )
            smaller = self._read_configured(
                reduced,
                reader=self._refinement_reader,
                line_classification=line_classification,
            )
            if smaller is not None:
                sx, sy = image.width / reduced.width, image.height / reduced.height
                for line, words in zip(smaller.lines, smaller.line_words):
                    scaled = [
                        ([[x * sx, y * sy] for x, y in item[0]], item[1], item[2])
                        for item in [line, *words]
                    ]
                    alternative.lines.append(scaled[0])
                    alternative.line_words.append(scaled[1:])
        selected = list(zip(prediction.lines, prediction.line_words))
        additions = [
            (line, words)
            for line, words in zip(alternative.lines, alternative.line_words)
            if line[2] >= 0.8
            and (len("".join(line[1].split())) >= 2 or line[1].strip().isdecimal())
        ]
        # A detector may merge two neighboring receipt rows into one poor box.
        # Replace that box only if several precise readings cover its span.
        retained = []
        for line, words in selected:
            box = self._line_bbox(line)
            concatenated = re.search(r"\d+[.,]\d{2}\d+[.,]\d", line[1]) is not None
            fragments = additions
            if concatenated:
                # A damaged row can contain two prices without a separator.
                # Retain its label too: labels are usually less confident than
                # prices, and dropping them would turn a split into text loss.
                fragments = [
                    (candidate, candidate_words)
                    for candidate, candidate_words in zip(
                        alternative.lines, alternative.line_words
                    )
                    if candidate[2] >= 0.7
                ]
            contained = [
                candidate
                for candidate, _ in fragments
                if self._line_bbox(candidate).intersection_over_self(box) >= 0.7
                and self._line_bbox(candidate).height <= box.height * 0.8
            ]
            if (line[2] < 0.65 or concatenated) and len(contained) >= 2:
                original_values = protected_numbers(line, words)
                observed_values = [
                    value
                    for candidate in contained
                    for value in _number_signature(candidate[1])
                ]
                if not concatenated and any(
                    observed_values.count(value) < original_values.count(value)
                    for value in set(original_values)
                ):
                    retained.append((line, words))
                    continue
                boxes = [self._line_bbox(candidate) for candidate in contained]
                complete_split = concatenated and (
                    sum(
                        re.fullmatch(r"\d+[.,]\d{2}", candidate[1].strip()) is not None
                        for candidate in contained
                    )
                    >= 2
                    and any(
                        sum(char.isalpha() for char in candidate[1]) >= 4
                        for candidate in contained
                    )
                )
                if (
                    max(b.r for b in boxes) - min(b.l for b in boxes)
                    >= box.width * (0.6 if complete_split else 0.7)
                    and max(b.b for b in boxes) - min(b.t for b in boxes)
                    >= box.height * 0.65
                ):
                    additions.extend(
                        (candidate, candidate_words)
                        for candidate, candidate_words in fragments
                        if candidate in contained
                        and not any(candidate is item[0] for item in additions)
                    )
                    continue
            retained.append((line, words))
        selected = retained
        for candidate, words in additions:
            conflicts = [
                index
                for index, (line, _) in enumerate(selected)
                if self._same_text_region(candidate, line)
                or (
                    self._line_bbox(candidate).intersection_over_self(
                        self._line_bbox(line)
                    )
                    >= 0.85
                    and bool(_number_signature(line[1]))
                )
            ]
            if conflicts:
                if not reading_may_replace(
                    [selected[index] for index in conflicts], candidate
                ):
                    continue
                selected = [
                    item
                    for index, item in enumerate(selected)
                    if index not in conflicts
                ]
            selected.append((candidate, words))
        expanded = []
        for line, words in selected:
            box = self._line_bbox(line)
            for candidate in alternative.lines:
                candidate_box = self._line_bbox(candidate)
                if (
                    line[1].isalpha()
                    and len(line[1]) >= 4
                    and candidate[2] >= 0.5
                    and candidate[1][:3]
                    .upper()
                    .translate(
                        str.maketrans("АВСЕНКМОРТХУ", "ABCEHKMOPTXY")  # noqa: RUF001
                    )
                    == line[1][:3]
                    .upper()
                    .translate(
                        str.maketrans("АВСЕНКМОРТХУ", "ABCEHKMOPTXY")  # noqa: RUF001
                    )
                    and len(candidate[1]) >= 1.5 * len(line[1])
                    and candidate_box.width >= 1.5 * box.width
                    and candidate_box.height <= 1.3 * box.height
                    and box.intersection_over_self(candidate_box) >= 0.8
                    and light_on_dark(
                        numpy.asarray(image.crop(candidate_box.as_tuple()).convert("L"))
                    )
                ):
                    # A second detector can see the rest of a light word.
                    # Expand its pixels for regional rereading, retaining the
                    # original text until recognition verifies a replacement.
                    line = candidate[0], line[1], line[2]
                    break
            expanded.append((line, words))
        return self._order_and_join(expanded)

    @classmethod
    def _order_and_join(
        cls, selected: list[tuple[_OcrLine, list[_OcrLine]]]
    ) -> _OcrPrediction:
        from docling.utils.ocr_line_order import order_ocr_lines

        neighbors = [word for _, words in selected for word in words]
        selected = [
            deduplicate_line_words(line, words, neighbors=neighbors)
            for line, words in selected
        ]
        ordered = order_ocr_lines(selected, line_band=cls._line_band)
        return _OcrPrediction(
            [line for line, _ in ordered], [words for _, words in ordered]
        )

    @staticmethod
    def _same_printed_row(first: BoundingBox, second: BoundingBox) -> bool:
        overlap_x = min(first.r, second.r) - max(first.l, second.l)
        overlap_y = min(first.b, second.b) - max(first.t, second.t)
        return overlap_y >= 0.5 * min(
            first.height, second.height
        ) and overlap_x <= 0.1 * min(first.width, second.width)

    @staticmethod
    def _line_bbox(line: _OcrLine) -> BoundingBox:
        return BoundingBox(
            l=min(point[0] for point in line[0]),
            t=min(point[1] for point in line[0]),
            r=max(point[0] for point in line[0]),
            b=max(point[1] for point in line[0]),
        )

    @staticmethod
    def _line_band(line: _OcrLine, x: float) -> tuple[float, float]:
        # Compare the text baselines at the same x coordinate. Axis-aligned
        # envelopes of long tilted lines also cover neighbouring printed rows.
        points = line[0]
        left_x = (points[0][0] + points[3][0]) / 2
        right_x = (points[1][0] + points[2][0]) / 2
        left_y = (points[0][1] + points[3][1]) / 2
        right_y = (points[1][1] + points[2][1]) / 2
        if abs(right_x - left_x) < 1e-6:
            return (left_y + right_y) / 2, abs(right_y - left_y)
        center = left_y + (x - left_x) * (right_y - left_y) / (right_x - left_x)
        height = (
            abs(points[3][1] - points[0][1]) + abs(points[2][1] - points[1][1])
        ) / 2
        return center, height

    @classmethod
    def _same_text_region(cls, first_line: _OcrLine, second_line: _OcrLine) -> bool:
        first, second = cls._line_bbox(first_line), cls._line_bbox(second_line)
        overlap_x = min(first.r, second.r) - max(first.l, second.l)
        if overlap_x <= 0.5 * min(first.width, second.width):
            return False
        x = (max(first.l, second.l) + min(first.r, second.r)) / 2
        first_y, first_height = cls._line_band(first_line, x)
        second_y, second_height = cls._line_band(second_line, x)
        height = min(first_height, second_height)
        return (
            height > 0
            and max(first_height, second_height) <= 2 * height
            and abs(first_y - second_y) <= 0.5 * height
        )

    def _read_upright(
        self, image: Image.Image, prediction: _OcrPrediction
    ) -> tuple[_OcrPrediction, int, tuple[int, int]]:
        """Compare actual page readings without changing individual row directions."""
        from docling.models.stages.ocr.ocr_page_orientation import read_page_upright

        return read_page_upright(self, image, prediction)

    def _read_sideways_page(
        self, image: Image.Image, prediction: _OcrPrediction
    ) -> tuple[_OcrPrediction, int]:
        """Reread a sideways form even if its individual rotated crops look confident."""
        # A line detector may merge a label with several amount columns.
        # Individual word coordinates still reveal the page's reading direction.
        observations = prediction.words or prediction.lines
        labels = [
            line for line in observations if sum(c.isalpha() for c in line[1]) >= 4
        ]
        numbers = [
            line
            for line in observations
            if sum(c.isdigit() for c in line[1]) >= 3
            and not any(c.isalpha() for c in line[1])
        ]
        if (
            len(labels) < 5
            or len(numbers) < 8
            or sum(
                self._line_bbox(line).height > 1.8 * self._line_bbox(line).width
                for line in labels
            )
            < len(labels) * 0.7
        ):
            return prediction, 0
        separation = median(
            (self._line_bbox(line).t + self._line_bbox(line).b) / 2 for line in numbers
        ) - median(
            (self._line_bbox(line).t + self._line_bbox(line).b) / 2 for line in labels
        )
        if abs(separation) < image.height * 0.2:
            return prediction, 0
        turn = 270 if separation > 0 else 90
        rotated = image.rotate(-turn, expand=True)
        candidate = self._read(rotated)
        if candidate is None:
            return prediction, 0
        candidate_labels = [
            line for line in candidate.lines if sum(c.isalpha() for c in line[1]) >= 4
        ]
        if (
            len(candidate_labels) < 5
            or sum(
                self._line_bbox(line).width > self._line_bbox(line).height
                for line in candidate_labels
            )
            < len(candidate_labels) * 0.7
            or _confident_chars(candidate.lines)
            < 0.9 * _confident_chars(prediction.lines)
        ):
            return prediction, 0
        return candidate, turn

    @classmethod
    def get_options_type(cls) -> Type[OcrOptions]:
        return RapidOcrOptions
