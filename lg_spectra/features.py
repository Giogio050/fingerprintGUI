"""Feature extraction from spectra and sticks."""

from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Iterable, List, Sequence

import numpy as np
from scipy.fft import dct
from scipy.stats import entropy, kurtosis, skew

from .sticks import Stick
from .vectorize import sticks_to_vector


BANDS: Sequence[tuple[float, float]] = (
    (360, 400),
    (400, 450),
    (450, 500),
    (500, 600),
    (600, 700),
    (700, 800),
)


def _safe_norm(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    y -= y.min()
    max_v = y.max()
    return y / max_v if max_v else y


def _bandpower(lam: np.ndarray, y: np.ndarray) -> List[float]:
    lam = np.asarray(lam)
    y = np.asarray(y)
    totals: List[float] = []
    for lo, hi in BANDS:
        mask = (lam >= lo) & (lam < hi)
        if mask.any():
            totals.append(float(np.trapezoid(y[mask], lam[mask])))
        else:
            totals.append(0.0)
    total = sum(totals) or 1.0
    return [p / total for p in totals]


def _peak_complexity(lam: np.ndarray, sticks: Sequence[Stick]) -> Dict[str, float]:
    counts = []
    for lo, hi in BANDS:
        counts.append(sum(1 for s in sticks if lo <= s.lambda_nm < hi))
    total = sum(counts) or 1
    return {
        "per_band": counts,
        "density": total / (BANDS[-1][1] - BANDS[0][0]),
    }


def _phash(vector: np.ndarray) -> str:
    coarse = np.interp(np.linspace(0, vector.size - 1, 32), np.arange(vector.size), vector)
    coeffs = dct(coarse, norm="ortho")[:16]
    median = np.median(coeffs[1:])
    bits = "".join("1" if c > median else "0" for c in coeffs[1:])
    return "phash_v1:" + format(int(bits, 2), "016x")


def _ratios(sticks: Sequence[Stick]) -> List[float]:
    if len(sticks) < 2:
        return []
    intensities = np.array([s.rel_intensity for s in sticks], dtype=float)
    intensities = intensities / intensities.max() if intensities.max() else intensities
    base = intensities[0]
    if base == 0:
        return []
    return (intensities[1:] / base).tolist()


def _global_stats(lam: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    weights = y / (y.sum() + 1e-9)
    lambda_mean = float(np.average(lam, weights=weights))
    lambda_median = float(np.interp(0.5, np.cumsum(weights), lam, left=float(lam[0]), right=float(lam[-1])))
    return {
        "lambda_mean": lambda_mean,
        "lambda_median": lambda_median,
        "skew": float(skew(y)),
        "kurt": float(kurtosis(y)),
        "lambda_std": float(np.sqrt(np.average((lam - lambda_mean) ** 2, weights=weights))),
    }


def compute_features(
    lam: np.ndarray,
    y: np.ndarray,
    sticks: Sequence[Stick],
    *,
    meta: Dict[str, object] | None = None,
    rt_min: float | None = None,
) -> Dict[str, object]:
    """Compute the fingerprint feature dictionary."""

    lam = np.asarray(lam, dtype=float)
    y = np.asarray(y, dtype=float)
    if y.ndim != 1:
        y = np.asarray(y).reshape(-1)
    y_norm = _safe_norm(y)

    vector = sticks_to_vector(list(sticks))
    ratios = _ratios(sticks)
    dct16 = dct(y_norm, norm="ortho")[:16]

    rel = np.array([s.rel_intensity for s in sticks]) if sticks else np.array([])
    purity = float(rel.max() / (rel.sum() + 1e-9)) if rel.size else 0.0
    snr = float(rel.max() / (np.std(y_norm) + 1e-9)) if rel.size else 0.0

    features: Dict[str, object] = {
        "unit": "nm",
        "sticks": [asdict(s) for s in sticks],
        "ratios": ratios,
        "entropy": float(entropy(y_norm + 1e-12)),
        "bandpower": _bandpower(lam, y_norm),
        "dct16": dct16.tolist(),
        "hash": _phash(y_norm),
        "global": _global_stats(lam, y_norm),
        "quality": {
            "snr": snr,
            "purity": purity,
            "n_sticks": int(len(sticks)),
        },
        "peak_complexity": _peak_complexity(lam, sticks),
        "stick_vector": vector.tolist(),
    }
    if rt_min is not None:
        features["rt_min"] = float(rt_min)
    if meta is not None:
        features["meta"] = dict(meta)
    if rel.size >= 1:
        features["top_lambda"] = float(sticks[0].lambda_nm)
    return features
