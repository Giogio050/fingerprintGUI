"""Spectral preprocessing pipelines."""
from __future__ import annotations

from typing import Dict, Iterable, Sequence, Tuple, List
import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import minimum_filter1d


def rolling_min(y: np.ndarray, win: int) -> np.ndarray:
    return y - minimum_filter1d(y, size=win)


def snv(y: np.ndarray) -> np.ndarray:
    y = y - np.mean(y)
    s = np.std(y)
    return y / s if s else y


def area_norm(y: np.ndarray) -> np.ndarray:
    area = np.trapz(y)
    return y / area if area else y


def moving_avg(y: np.ndarray, n: int) -> np.ndarray:
    c = np.convolve(y, np.ones(n) / n, mode='same')
    return c


def sanitize_spectrum(
    lam: Sequence[float],
    y: Sequence[float],
    grid: Sequence[float] | None = None,
    saturate: float | None = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Clip negatives, handle saturation and interpolate to ``grid``.

    Returns the possibly modified wavelength and intensity arrays along with
    a list of textual notes describing applied corrections.
    """
    notes: List[str] = []
    lam_arr = np.asarray(lam, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if lam_arr.max() < 100:  # assume data in µm
        lam_arr = lam_arr * 1000
        notes.append('lambda scaled from um to nm')
    if np.any(y_arr < 0):
        y_arr = np.clip(y_arr, 0, None)
        notes.append('negative intensities clipped')
    if saturate is not None and np.any(y_arr > saturate):
        y_arr = np.clip(y_arr, None, saturate)
        notes.append('saturation clipping applied')
    if grid is not None and not np.array_equal(lam_arr, grid):
        y_arr = np.interp(grid, lam_arr, y_arr)
        lam_arr = np.asarray(grid, dtype=float)
        notes.append('interpolated to common grid')
    return lam_arr, y_arr, notes


_OPERATORS = {
    'rolling_min': rolling_min,
    'snv': lambda y: snv(y),
    'area_norm': lambda y: area_norm(y),
    'savgol': lambda y, win=7, poly=2: savgol_filter(y, win, poly),
    'moving_avg': lambda y, n=3: moving_avg(y, n),
}


def apply_pipeline(lam: np.ndarray, y: np.ndarray, config: Iterable[Dict]) -> np.ndarray:
    """Apply a sequence of preprocessing steps described by ``config``.

    Each element in ``config`` is a mapping with key ``op`` specifying
    the operator name and optional keyword arguments.
    """
    y_hat = np.asarray(y, dtype=float)
    for step in config:
        op = step['op']
        func = _OPERATORS.get(op)
        if func is None:
            raise ValueError(f'Unknown operator {op}')
        kwargs = {k: v for k, v in step.items() if k != 'op'}
        y_hat = func(y_hat, **kwargs)
    return y_hat
