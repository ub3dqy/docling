# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Read difficult Cyrillic regions with pinned PP-OCRv6 ONNX models."""

import re
from collections.abc import Callable, Iterator
from math import ceil
from pathlib import Path

import numpy as np

from docling.utils.ocr_character_paths import CharacterPath, character_paths
from docling.utils.ocr_region_pixels import clean_region, photo_contrast
from docling.utils.russian_ocr_text_models import (
    CYRILLIC_MODELS,
    validate_cyrillic_models,
)


class CyrillicTextReader:
    def __init__(self, directory: Path, *, num_threads: int) -> None:
        import onnxruntime

        settings = onnxruntime.SessionOptions()
        settings.intra_op_num_threads = num_threads
        settings.inter_op_num_threads = 1
        settings.log_severity_level = 3
        # These readers alternate with other models. Idle pools must yield CPU.
        settings.add_session_config_entry("session.intra_op.allow_spinning", "0")
        settings.add_session_config_entry("session.inter_op.allow_spinning", "0")
        self._session = onnxruntime.InferenceSession(
            str(directory / "model.onnx"),
            sess_options=settings,
            providers=["CPUExecutionProvider"],
        )
        self._alphabet = [
            "",
            *(directory / "rec_keys.txt").read_text(encoding="utf-8").splitlines(),
            " ",
        ]
        self._input = self._session.get_inputs()[0].name

    def _paths(
        self, gray: np.ndarray, *, known: Callable[[str], bool]
    ) -> list[CharacterPath]:
        import cv2

        if gray.ndim != 2 or gray.shape[0] < 4 or gray.shape[1] < 4:
            return []
        # Match the publisher's 48-pixel normalization and trailing context.
        # Keep each region separate so batch padding cannot change its result.
        scaled_width = ceil(48 * gray.shape[1] / gray.shape[0])
        width = max(640, ceil(scaled_width * 1.5))
        if width > 4096:
            return []
        sample = cv2.resize(gray, (scaled_width, 48), interpolation=cv2.INTER_LINEAR)
        sample = cv2.cvtColor(sample, cv2.COLOR_GRAY2BGR)
        normalized = sample.transpose(2, 0, 1).astype(np.float32) / 127.5 - 1
        padded = np.zeros((1, 3, 48, width), dtype=np.float32)
        padded[0, :, :, :scaled_width] = normalized
        probabilities = self._session.run(None, {self._input: padded})[0][0]
        greedy = character_paths(probabilities, self._alphabet, beam_width=1)
        words = re.findall(
            r"[A-Za-z\u0410-\u044f\u0401\u0451]+(?:-[A-Za-z\u0410-\u044f\u0401\u0451]+)*",
            greedy[0].text,
        )
        if greedy[0].confidence >= 0.95 and all(known(word) for word in words):
            return greedy
        return character_paths(probabilities, self._alphabet)

    def read_views(
        self,
        gray: np.ndarray,
        *,
        known: Callable[[str], bool],
        prefer_clean: bool = False,
    ) -> Iterator[list[tuple[str, CharacterPath]]]:
        import cv2

        variants = {
            "source": gray[:, 2:-2] if gray.shape[1] > 20 else gray,
            "clean": clean_region(gray),
            "contrast": clean_region(photo_contrast(gray)),
        }
        names = ("clean", "source", "contrast") if prefer_clean else tuple(variants)
        cache: dict[tuple[tuple[int, ...], bytes], list[CharacterPath]] = {}
        for name in names:
            array = variants[name]
            for factor in (1.0, 1.5):
                sample = cv2.resize(array, None, fx=factor, fy=1)
                key = (sample.shape, sample.tobytes())
                if key not in cache:
                    cache[key] = self._paths(sample, known=known)
                paths = cache[key]
                if not paths:
                    yield []
                    continue
                readings = [(f"v6-{name}-{factor}", paths[0])]
                # The dictionary only ranks paths from real model scores.
                # It cannot insert an unseen letter, digit or punctuation.
                readings.extend(
                    (f"v6-{name}-{factor}-characters", path)
                    for path in paths[1:]
                    if path.confidence >= 0.8
                    and path.log_probability >= paths[0].log_probability - 3
                    and (
                        words := re.findall(
                            r"[A-Za-z\u0410-\u044f\u0401\u0451]+(?:-[A-Za-z\u0410-\u044f\u0401\u0451]+)*",
                            path.text,
                        )
                    )
                    and all(known(word) for word in words)
                )
                yield readings

    def read_region(
        self, gray: np.ndarray, *, known: Callable[[str], bool]
    ) -> list[tuple[str, CharacterPath]]:
        def numeric_content(text: str) -> str:
            text = re.sub(r"(?<=\d),(?=\d)", ".", text)
            return "".join(
                char for char in text if not (char.isalpha() or char.isspace())
            )

        readings = []
        for view in self.read_views(gray, known=known):
            if not view:
                continue
            preferred = view[0][1]
            numeric = any(char.isdigit() for char in preferred.text)
            content = numeric_content(preferred.text)
            # A shorter CTC path can have higher mean character confidence
            # simply because it dropped an uncertain operator. Word choices
            # must retain the best path's numeric and punctuation content.
            readings.extend(
                (variant, path)
                for variant, path in view
                if not numeric or numeric_content(path.text) == content
            )
        return readings


def create_cyrillic_readers(
    directory: Path, *, num_threads: int
) -> dict[str, CyrillicTextReader]:
    validate_cyrillic_models(directory)
    return {
        spec.name: CyrillicTextReader(
            directory / "CyrillicOcrV6" / spec.name, num_threads=num_threads
        )
        for spec in CYRILLIC_MODELS
    }
