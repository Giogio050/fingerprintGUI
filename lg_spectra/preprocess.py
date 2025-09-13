"""Spectral preprocessing pipelines and ensembles."""
from __future__ import annotations

from itertools import product
from typing import Dict, Iterable, List, Tuple, Callable

import numpy as np
from scipy import sparse
from scipy.signal import savgol_filter
from scipy.sparse.linalg import spsolve


def baseline_asls(lam: np.ndarray, y: np.ndarray, lam_s: float = 1e5,
                  p: float = 0.01, niter: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """Asymmetric least squares baseline correction.

    Parameters
    ----------
    lam, y:
        Wavelength axis and spectrum.
    lam_s:
        Smoothness parameter :math:`\lambda`.
    p:
        Asymmetry parameter.
    niter:
        Number of iterations.
    """
    L = y.size
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L - 2, L))
    w = np.ones(L)
    for _ in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam_s * D.T @ D
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y <= z)
    return lam, y - z


def snv(lam: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    y0 = y - y.mean()
    s = y0.std()
    return lam, (y0 / s if s else y0)


def sg(lam: np.ndarray, y: np.ndarray, window: int = 7,
       poly: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    return lam, savgol_filter(y, window_length=window, polyorder=poly)


def deriv(lam: np.ndarray, y: np.ndarray, order: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    dy = np.gradient(y, lam)
    if order == 2:
        dy = np.gradient(dy, lam)
    return lam, dy


def area_norm(lam: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    area = np.trapz(y, lam)
    return lam, (y / area if area else y)


def rolling_min(lam: np.ndarray, y: np.ndarray, width_nm: float = 10.0) -> Tuple[np.ndarray, np.ndarray]:
    step = np.mean(np.diff(lam)) or 1.0
    win = max(1, int(round(width_nm / step)))
    baseline = np.minimum.accumulate(y)
    baseline = np.minimum(baseline, np.minimum.accumulate(y[::-1])[::-1])
    from scipy.ndimage import minimum_filter1d
    baseline = minimum_filter1d(baseline, size=win)
    return lam, y - baseline


def peak_align(lam: np.ndarray, y: np.ndarray, ref: np.ndarray,
               max_shift_nm: float = 5.0) -> Tuple[np.ndarray, np.ndarray]:
    step = np.mean(np.diff(lam)) or 1.0
    max_shift = int(round(max_shift_nm / step))
    corr = np.correlate(ref, y, mode='full')
    shift = np.argmax(corr) - (len(y) - 1)
    shift = int(np.clip(shift, -max_shift, max_shift))
    lam2 = lam - shift * step
    y2 = np.roll(y, shift)
    return lam2, y2


Operator = Callable[..., Tuple[np.ndarray, np.ndarray]]


_OPERATORS: Dict[str, Operator] = {
    'baseline_asls': baseline_asls,
    'snv': snv,
    'sg': sg,
    'deriv': deriv,
    'area_norm': area_norm,
    'rolling_min': rolling_min,
    'peak_align': peak_align,
}


def apply_pipeline(lam: np.ndarray, y: np.ndarray,
                   spec: Iterable[Dict | Tuple[str, Dict] | str]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Apply a sequence of preprocessing steps.

    Parameters
    ----------
    lam, y:
        Input wavelength axis and spectrum.
    spec:
        Sequence of operations. Each element may be a string, a
        ``(op, params)`` tuple or a mapping containing ``op`` and
        keyword arguments.

    Returns
    -------
    lam2, y2, info:
        Processed axis and spectrum together with a list of applied
        operator names.
    """
    lam2 = np.asarray(lam, dtype=float)
    y2 = np.asarray(y, dtype=float)
    info: List[str] = []
    for step in spec:
        if isinstance(step, str):
            op, params = step, {}
        elif isinstance(step, tuple):
            op, params = step
        elif isinstance(step, dict):
            op = step.get('op')
            params = {k: v for k, v in step.items() if k != 'op'}
        else:  # pragma: no cover - defensive
            raise TypeError('Invalid pipeline step')
        func = _OPERATORS.get(op)
        if func is None:
            raise ValueError(f'Unknown operator {op}')
        lam2, y2 = func(lam2, y2, **params)
        info.append(op)
    return lam2, y2, info

