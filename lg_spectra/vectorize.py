"""Vectorisation utilities for matching."""

from __future__ import annotations

from typing import List
import numpy as np

from .sticks import Stick


def sticks_to_vector(
    sticks: List[Stick], wl_min: int = 360, wl_max: int = 800
) -> np.ndarray:
    """Convert stick list to 1-nm binned vector."""
    vec = np.zeros(wl_max - wl_min + 1, dtype=float)
    for s in sticks:
        idx = int(round(s.lambda_nm)) - wl_min
        if 0 <= idx < len(vec):
            vec[idx] = max(vec[idx], s.rel_intensity)
    return vec
