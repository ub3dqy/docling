# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Fixed line-classification views for the RapidOCR call interface."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class RapidReader(Protocol):
    def __call__(
        self,
        image: np.ndarray,
        /,
        *,
        use_det: bool | None = None,
        use_cls: bool | None = None,
        use_rec: bool | None = None,
        return_word_box: bool = False,
    ) -> object: ...


@dataclass(frozen=True)
class LineClassifierView:
    """Enable the legacy classifier on its separately configured reader."""

    reader: RapidReader

    def __call__(
        self,
        image: np.ndarray,
        /,
        *,
        use_det: bool | None = None,
        use_cls: bool | None = None,
        use_rec: bool | None = None,
        return_word_box: bool = False,
    ) -> object:
        return self.reader(
            image,
            use_det=use_det,
            use_cls=True,
            use_rec=use_rec,
            return_word_box=return_word_box,
        )
