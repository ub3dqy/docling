# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Retain alternative CTC readings supported by the actual character scores."""

from dataclasses import dataclass
from math import log
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from rapidocr import RapidOCR


@dataclass(frozen=True)
class CharacterPath:
    text: str
    confidence: float
    log_probability: float


@dataclass(frozen=True)
class _State:
    total: float
    best: float
    characters: tuple[float, ...]


def image_character_paths(
    reader: "RapidOCR", sample: np.ndarray
) -> list[CharacterPath]:
    """Use the installed RapidOCR recognizer's own normalization and alphabet."""
    # RapidOCR creates text_rec lazily on its first recognition call.
    reader(sample, use_det=False, use_cls=False, use_rec=True)
    recognizer = reader.text_rec
    assert recognizer is not None
    _, height, width = recognizer.rec_image_shape
    ratio = max(width / height, sample.shape[1] / sample.shape[0])
    normalized = recognizer.resize_norm_img(sample, ratio)
    return character_paths(
        recognizer.session(normalized[np.newaxis].astype(np.float32))[0],
        recognizer.postprocess_op.character,
    )


def character_paths(
    probabilities: np.ndarray,
    alphabet: list[str],
    *,
    beam_width: int = 64,
) -> list[CharacterPath]:
    """Decode bounded alternatives without inserting dictionary characters.

    Token zero is CTC blank. Each new character must be among the four
    strongest actual predictions at its position and have probability >= .03.
    Scores retain the CTC blank/repetition paths; confidence uses the emitted
    character probabilities, as in the recognizer's ordinary decoder.
    """
    if (
        probabilities.ndim != 2
        or probabilities.shape[1] != len(alphabet)
        or not np.isfinite(probabilities).all()
        or np.any(probabilities < 0)
        or np.any(probabilities > 1)
        or beam_width < 1
    ):
        raise ValueError(
            "Expected finite CTC character probabilities and a positive beam."
        )
    states = {("", 0): _State(0.0, 0.0, ())}
    for row in probabilities:
        strongest = int(row.argmax())
        tokens = [
            int(token)
            for token in np.argsort(row)[-4:]
            if row[token] >= 0.03 or token == strongest
        ]
        updated: dict[tuple[str, int], _State] = {}
        for token in tokens:
            probability = float(row[token])
            increment = log(max(1e-30, probability))
            for (text, previous), state in states.items():
                emitted = token != 0 and token != previous
                key = (text + alphabet[token] if emitted else text, token)
                characters = (
                    (*state.characters, probability) if emitted else state.characters
                )
                best = state.best + increment
                total = state.total + increment
                existing = updated.get(key)
                if existing is not None:
                    total = float(np.logaddexp(existing.total, total))
                    if existing.best > best:
                        best, characters = existing.best, existing.characters
                updated[key] = _State(total, best, characters)
        states = dict(
            sorted(updated.items(), key=lambda item: item[1].total, reverse=True)[
                :beam_width
            ]
        )
    results: dict[str, _State] = {}
    for (text, _), state in states.items():
        existing = results.get(text)
        if existing is None:
            results[text] = state
        else:
            best = state if state.best > existing.best else existing
            results[text] = _State(
                float(np.logaddexp(existing.total, state.total)),
                best.best,
                best.characters,
            )
    return [
        CharacterPath(
            text=text,
            confidence=float(np.mean(state.characters)) if state.characters else 0.0,
            log_probability=state.total,
        )
        for text, state in sorted(
            results.items(), key=lambda item: item[1].total, reverse=True
        )
    ]
