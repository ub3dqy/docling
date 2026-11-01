# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Check character alternatives against exhaustive CTC path probabilities."""

from itertools import groupby, product
from math import exp

import numpy as np
import pytest

from docling.utils.ocr_character_paths import character_paths


def test_blank_repetition_and_alternative_probabilities_match_all_alignments():
    alphabet = ["", "a", "b"]
    probabilities = np.array(
        [[0.05, 0.85, 0.10], [0.75, 0.20, 0.05], [0.10, 0.70, 0.20]]
    )
    exhaustive: dict[str, float] = {}
    for alignment in product(range(len(alphabet)), repeat=len(probabilities)):
        text = "".join(alphabet[token] for token, _ in groupby(alignment) if token)
        probability = float(
            np.prod(
                [probabilities[index, token] for index, token in enumerate(alignment)]
            )
        )
        exhaustive[text] = exhaustive.get(text, 0.0) + probability
    readings = character_paths(probabilities, alphabet, beam_width=128)
    assert readings[0].text == "aa"
    assert {reading.text for reading in readings} == set(exhaustive)
    for reading in readings:
        assert exp(reading.log_probability) == pytest.approx(exhaustive[reading.text])
        assert 0 <= reading.confidence <= 1


def test_a_dictionary_cannot_supply_an_unobserved_character():
    alphabet = ["", "a", "b", "c"]
    probabilities = np.array([[0.05, 0.88, 0.06, 0.01], [0.90, 0.05, 0.04, 0.01]])
    readings = character_paths(probabilities, alphabet)
    assert any(reading.text == "b" for reading in readings)
    assert all("c" not in reading.text for reading in readings)
