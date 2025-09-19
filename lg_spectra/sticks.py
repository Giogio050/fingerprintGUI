"""Peak picking and stick fingerprint generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy.signal import find_peaks, peak_widths


@dataclass(slots=True)
class Stick:
    lambda_nm: float
    rel_intensity: float
    width_nm: float
    prominence: float


@dataclass(slots=True)
class StickSelection:
    sticks: List[Stick]
    diagnostics: Dict[str, float | str]


def _normalize(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    span = np.ptp(y)
    if span == 0:
        return np.zeros_like(y)
    return (y - np.min(y)) / span


def _strategy_prominence(y: np.ndarray, params: Dict[str, float]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    prominence = float(params.get("prominence", 0.002))
    distance = int(params.get("distance", 6))
    height = params.get("height")
    peaks, props = find_peaks(y, prominence=prominence, distance=distance, height=height)
    return peaks, props


def _strategy_derivative(lam: np.ndarray, y: np.ndarray, params: Dict[str, float]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    grad = np.gradient(y, lam)
    grad2 = np.gradient(grad, lam)
    zero_cross = np.where(np.diff(np.signbit(grad)))[0]
    candidates = zero_cross[(grad2[zero_cross] < 0)]
    prominence = float(params.get("prominence", 0.0015))
    distance = int(params.get("distance", 5))
    peaks, props = find_peaks(y, prominence=prominence, distance=distance)
    mask = np.isin(peaks, candidates)
    peaks = peaks[mask]
    for key, arr in props.items():
        props[key] = arr[mask]
    return peaks, props


def _strategy_adaptive(lam: np.ndarray, y: np.ndarray, params: Dict[str, float]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    window = int(params.get("window", 11))
    window = window if window % 2 else window + 1
    pad = window // 2
    padded = np.pad(y, pad, mode="reflect")
    strides = np.lib.stride_tricks.sliding_window_view(padded, window)
    loc_mad = np.median(np.abs(strides - np.median(strides, axis=1, keepdims=True)), axis=1)
    threshold = np.median(loc_mad) * float(params.get("k", 4.0))
    prominence = max(threshold, float(params.get("prominence", 0.001)))
    peaks, props = find_peaks(y, prominence=prominence, distance=int(params.get("distance", 6)))
    return peaks, props


_STRATEGIES = {
    "prominence": lambda lam, y, p: _strategy_prominence(y, p),
    "derivative": _strategy_derivative,
    "adaptive": _strategy_adaptive,
}


def pick_sticks(
    lam: np.ndarray,
    y: np.ndarray,
    params: Dict[str, float] | None = None,
    *,
    return_info: bool = False,
) -> List[Stick] | StickSelection:
    """Detect spectral peaks and convert them into stick fingerprints."""

    params = dict(params or {})
    strategy = params.pop("strategy", "prominence")
    k = int(params.pop("k", 6))
    y = np.asarray(y, dtype=float)
    lam = np.asarray(lam, dtype=float)
    if y.size != lam.size:
        raise ValueError("Spectrum and wavelength axes must match in size")

    if np.allclose(y, y[0]):
        sticks: List[Stick] = []
        info = {"warning": "feature-poor", "dynamic_range": 0.0, "snr": 0.0}
        return StickSelection(sticks, info) if return_info else sticks

    strategy_func = _STRATEGIES.get(strategy, _STRATEGIES["prominence"])
    peaks, props = strategy_func(lam, y, params)
    if len(peaks) == 0 and strategy != "prominence":
        peaks, props = _strategy_prominence(y, params)
    if len(peaks) == 0:
        sticks = []
        info = {"warning": "no-peaks", "dynamic_range": float(np.ptp(y)), "snr": 0.0}
        return StickSelection(sticks, info) if return_info else sticks

    widths, _, left_ips, right_ips = peak_widths(y, peaks, rel_height=0.5)
    rel = y[peaks]
    rel_norm = rel / rel.max() * 100 if rel.max() else rel
    prominences = props.get("prominences", np.ones_like(rel))

    if np.ptp(y) < float(params.get("flat_threshold", 1e-3)):
        k = min(k, 3)
        warning = "feature-poor"
    else:
        warning = ""

    delta = float(np.mean(np.diff(lam))) if lam.size > 1 else 1.0
    sticks = [
        Stick(
            lambda_nm=float(lam[idx]),
            rel_intensity=float(rel_norm[i]),
            width_nm=float((right_ips[i] - left_ips[i]) * delta),
            prominence=float(prominences[i]),
        )
        for i, idx in enumerate(peaks)
    ]
    sticks.sort(key=lambda s: s.rel_intensity, reverse=True)
    sticks = sticks[:k]

    noise = np.std(_normalize(y))
    snr = float(rel_norm.max() / (noise + 1e-9)) if sticks else 0.0
    info = {
        "snr": snr,
        "dynamic_range": float(np.ptp(y)),
        "n_peaks": float(len(peaks)),
        "warning": warning,
    }
    return StickSelection(sticks, info) if return_info else sticks
