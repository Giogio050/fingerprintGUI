"""Spectral preprocessing pipelines."""

from __future__ import annotations

from typing import Dict, Iterable
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
    c = np.convolve(y, np.ones(n) / n, mode="same")
    return c


_OPERATORS = {
    "rolling_min": rolling_min,
    "snv": lambda y: snv(y),
    "area_norm": lambda y: area_norm(y),
    "savgol": lambda y, win=7, poly=2: savgol_filter(y, win, poly),
    "moving_avg": lambda y, n=3: moving_avg(y, n),
}


def apply_pipeline(
    lam: np.ndarray, y: np.ndarray, config: Iterable[Dict]
) -> np.ndarray:
    """Apply a sequence of preprocessing steps described by ``config``.

    Each element in ``config`` is a mapping with key ``op`` specifying
    the operator name and optional keyword arguments.
    """
    y_hat = np.asarray(y, dtype=float)
    for step in config:
        op = step["op"]
        func = _OPERATORS.get(op)
        if func is None:
            raise ValueError(f"Unknown operator {op}")
        kwargs = {k: v for k, v in step.items() if k != "op"}
        y_hat = func(y_hat, **kwargs)
    return y_hat
