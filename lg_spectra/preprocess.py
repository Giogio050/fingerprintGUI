"""Spectral preprocessing pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy import sparse
from scipy.fft import dct as fft_dct, idct
from scipy.sparse.linalg import spsolve
from scipy.ndimage import grey_closing, grey_opening, minimum_filter1d
from scipy.signal import medfilt, savgol_filter

from .io import MASTER_GRID


ArrayLike = np.ndarray
OperatorFunc = Callable[[ArrayLike, ArrayLike, Dict[str, float]], Tuple[ArrayLike, ArrayLike] | ArrayLike]


@dataclass(frozen=True)
class PipelineStep:
    """Descriptor for a single preprocessing step."""

    op: str
    params: Dict[str, float]


def _ensure_odd(window: int) -> int:
    return window if window % 2 else window + 1


def _interp_master(lam: ArrayLike, y: ArrayLike, params: Dict[str, float]) -> Tuple[ArrayLike, ArrayLike]:
    grid = params.get("grid")
    grid = np.asarray(grid if grid is not None else MASTER_GRID, dtype=float)
    if y.ndim > 1:
        interp = np.stack(
            [np.interp(grid, lam, row, left=0.0, right=0.0) for row in np.asarray(y)],
            axis=0,
        )
    else:
        interp = np.interp(grid, lam, y, left=0.0, right=0.0)
    return grid, interp


def _rolling_min(lam: ArrayLike, y: ArrayLike, params: Dict[str, float]) -> ArrayLike:
    win = int(params.get("win", 31))
    return y - minimum_filter1d(y, size=max(win, 3))


def _asls(lam: ArrayLike, y: ArrayLike, params: Dict[str, float]) -> ArrayLike:
    lam_s = float(params.get("lam", 1e5))
    p = float(params.get("p", 0.001))
    iters = int(params.get("iters", 10))
    L = y.size
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
    D = D @ D.T
    w = np.ones(L)
    for _ in range(iters):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam_s * D
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return y - z


def _morph_open(lam: ArrayLike, y: ArrayLike, params: Dict[str, float]) -> ArrayLike:
    size = int(params.get("struct", 5))
    return y - grey_opening(y, size=size)


def _morph_close(lam: ArrayLike, y: ArrayLike, params: Dict[str, float]) -> ArrayLike:
    size = int(params.get("struct", 5))
    return y - grey_closing(y, size=size)


def _snv(lam: ArrayLike, y: ArrayLike, params: Dict[str, float]) -> ArrayLike:
    y_centered = y - np.mean(y)
    std = np.std(y_centered)
    return y_centered / std if std else y_centered


def _vector_norm(lam: ArrayLike, y: ArrayLike, params: Dict[str, float]) -> ArrayLike:
    norm = np.linalg.norm(y)
    return y / norm if norm else y


def _area_norm(lam: ArrayLike, y: ArrayLike, params: Dict[str, float]) -> ArrayLike:
    area = np.trapz(y, lam) if lam.size == y.size else np.trapz(y)
    return y / area if area else y


def _quantile_norm(lam: ArrayLike, y: ArrayLike, params: Dict[str, float]) -> ArrayLike:
    q = float(params.get("quantile", 0.95))
    anchor = np.quantile(np.abs(y), q)
    return y / anchor if anchor else y


def _moving_avg(lam: ArrayLike, y: ArrayLike, params: Dict[str, float]) -> ArrayLike:
    n = int(params.get("n", 5))
    kernel = np.ones(max(n, 1)) / max(n, 1)
    return np.convolve(y, kernel, mode="same")


def _savgol(lam: ArrayLike, y: ArrayLike, params: Dict[str, float]) -> ArrayLike:
    win = _ensure_odd(int(params.get("win", 7)))
    poly = int(params.get("poly", 2))
    return savgol_filter(y, win, poly)


def _median(lam: ArrayLike, y: ArrayLike, params: Dict[str, float]) -> ArrayLike:
    size = _ensure_odd(int(params.get("size", 5)))
    return medfilt(y, kernel_size=size)


def _wavelet_soft(lam: ArrayLike, y: ArrayLike, params: Dict[str, float]) -> ArrayLike:
    if y.size < 8:
        return y
    coeffs = fft_dct(y, norm="ortho")
    tail = coeffs[int(coeffs.size / 2) :]
    sigma = np.median(np.abs(tail)) / 0.6745 if tail.size else np.std(coeffs)
    thresh = sigma * np.sqrt(2 * np.log(y.size)) * float(params.get("scale", 1.0))
    soft = np.sign(coeffs) * np.maximum(np.abs(coeffs) - thresh, 0.0)
    return idct(soft, norm="ortho")


def _derivative(lam: ArrayLike, y: ArrayLike, params: Dict[str, float]) -> ArrayLike:
    order = int(params.get("order", 1))
    if order == 1:
        return np.gradient(y, lam)
    if order == 2:
        return np.gradient(np.gradient(y, lam), lam)
    raise ValueError("Unsupported derivative order")


def _gradient_clamp(lam: ArrayLike, y: ArrayLike, params: Dict[str, float]) -> ArrayLike:
    limit = float(params.get("limit", 0.15))
    grad = np.gradient(y, lam)
    cap = limit * np.max(np.abs(grad)) if grad.size else 0.0
    if cap == 0:
        return y
    grad_clamped = np.clip(grad, -cap, cap)
    y0 = y[0]
    reconstructed = y0 + np.cumsum(grad_clamped[:-1])
    return np.concatenate([[y0], reconstructed])


_OPERATORS: Dict[str, OperatorFunc] = {
    "interp_master": _interp_master,
    "rolling_min": _rolling_min,
    "asls": _asls,
    "morph_open": _morph_open,
    "morph_close": _morph_close,
    "snv": _snv,
    "vector_norm": _vector_norm,
    "area_norm": _area_norm,
    "quantile_norm": _quantile_norm,
    "moving_avg": _moving_avg,
    "savgol": _savgol,
    "median": _median,
    "wavelet_soft": _wavelet_soft,
    "derivative": _derivative,
    "gradient_clamp": _gradient_clamp,
}


def apply_pipeline(
    lam: ArrayLike,
    y: ArrayLike,
    config: Iterable[Dict[str, float] | PipelineStep],
    *,
    return_wavelength: bool = False,
) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
    """Apply a sequence of preprocessing steps described by *config*.

    Each element in *config* must either be a :class:`PipelineStep` or a
    dictionary containing the key ``op`` and optional operator parameters.
    ``interp_master`` operations may change the wavelength axis.
    """

    lam_work = np.asarray(lam, dtype=float)
    y_work = np.asarray(y, dtype=float)
    for step in config:
        if isinstance(step, PipelineStep):
            name = step.op
            params = dict(step.params)
        else:
            step = dict(step)
            name = str(step.pop("op"))
            params = step
        func = _OPERATORS.get(name)
        if func is None:
            raise ValueError(f"Unknown operator '{name}'")
        result = func(lam_work, y_work, params)
        if isinstance(result, tuple):
            lam_work, y_work = result
        else:
            y_work = result
    y_out = np.asarray(y_work, dtype=float)
    if return_wavelength:
        return lam_work, y_out
    return y_out


@lru_cache(maxsize=16)
def preset(name: str) -> List[PipelineStep]:
    """Return predefined preprocessing presets used by the CLI and GUI."""

    presets = {
        "NoiseMax": [
            PipelineStep("interp_master", {}),
            PipelineStep("rolling_min", {"win": 51}),
            PipelineStep("wavelet_soft", {"level": 3}),
            PipelineStep("savgol", {"win": 11, "poly": 3}),
            PipelineStep("snv", {}),
        ],
        "DerivSharp": [
            PipelineStep("interp_master", {}),
            PipelineStep("asls", {"lam": 1e6, "p": 0.001}),
            PipelineStep("savgol", {"win": 9, "poly": 2}),
            PipelineStep("derivative", {"order": 1}),
            PipelineStep("quantile_norm", {"quantile": 0.9}),
        ],
        "BaselineHard": [
            PipelineStep("interp_master", {}),
            PipelineStep("asls", {"lam": 1e6, "p": 0.01}),
            PipelineStep("median", {"size": 7}),
            PipelineStep("vector_norm", {}),
        ],
    }
    if name not in presets:
        raise KeyError(name)
    return presets[name]
