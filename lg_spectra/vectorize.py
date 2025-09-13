"""Vectorisation utilities for matching."""
from __future__ import annotations

from typing import Iterable, Sequence
import numpy as np


def sticks_to_vector(sticks: Iterable, wl_min: int = 360, wl_max: int = 800) -> np.ndarray:
    """Convert a list of sticks to a 1-nm binned vector.

    ``sticks`` can be a sequence of ``(lambda, intensity)`` tuples or
    objects with ``lambda_nm``/``intensity`` attributes.
    """

    vec = np.zeros(wl_max - wl_min + 1, dtype=float)
    for s in sticks:
        if hasattr(s, 'lambda_nm'):
            lam = getattr(s, 'lambda_nm')
            inten = getattr(s, 'intensity', getattr(s, 'rel_intensity', 0.0))
        else:
            lam, inten = s  # type: ignore[misc]
        idx = int(round(lam)) - wl_min
        if 0 <= idx < len(vec):
            vec[idx] = max(vec[idx], float(inten))
    return vec
