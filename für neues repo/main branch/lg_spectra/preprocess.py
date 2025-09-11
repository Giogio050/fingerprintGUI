"""Spectral preprocessing pipelines."""
from __future__ import annotations

from typing import Dict, Iterable
import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import minimum_filter1d, median_filter, grey_opening, grey_closing
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve


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


def quantile_norm(y: np.ndarray, q: float = 0.95) -> np.ndarray:
    """Scale ``y`` so that its ``q``-quantile equals 1."""
    anchor = np.quantile(y, q)
    return y / anchor if anchor else y


def median_filt(y: np.ndarray, size: int = 5) -> np.ndarray:
    return median_filter(y, size=size)


def savgol_deriv(y: np.ndarray, win: int = 7, poly: int = 2, deriv: int = 1) -> np.ndarray:
    return savgol_filter(y, win, poly, deriv=deriv)


def morph_open(y: np.ndarray, size: int = 5) -> np.ndarray:
    return y - grey_opening(y, size=size)


def morph_close(y: np.ndarray, size: int = 5) -> np.ndarray:
    return y - grey_closing(y, size=size)


def asls(y: np.ndarray, lam: float = 1e5, p: float = 0.01, niter: int = 10) -> np.ndarray:
    """Asymmetric least squares baseline correction."""
    y = np.asarray(y, float)
    L = len(y)
    D = diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
    w = np.ones(L)
    for _ in range(niter):
        W = diags(w, 0)
        Z = W + lam * (D @ D.T)
        z = spsolve(Z.tocsr(), w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return y - z


_OPERATORS = {
    'rolling_min': rolling_min,
    'snv': lambda y: snv(y),
    'area_norm': lambda y: area_norm(y),
    'savgol': lambda y, win=7, poly=2: savgol_filter(y, win, poly),
    'moving_avg': lambda y, n=3: moving_avg(y, n),
    'quantile_norm': lambda y, q=0.95: quantile_norm(y, q),
    'median': lambda y, size=5: median_filt(y, size),
    'savgol_deriv1': lambda y, win=7, poly=2: savgol_deriv(y, win, poly, deriv=1),
    'savgol_deriv2': lambda y, win=7, poly=2: savgol_deriv(y, win, poly, deriv=2),
    'morph_open': lambda y, size=5: morph_open(y, size),
    'morph_close': lambda y, size=5: morph_close(y, size),
    'asls': lambda y, lam=1e5, p=0.01, niter=10: asls(y, lam, p, niter),
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
