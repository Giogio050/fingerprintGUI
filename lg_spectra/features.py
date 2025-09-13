"""Feature extraction from processed spectra and their sticks."""
from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy.fft import dct
from scipy.stats import kurtosis, skew


# Wavelength windows for band power calculation (nm)
BANDS: List[Tuple[int, int]] = [
    (360, 400),
    (400, 450),
    (450, 500),
    (500, 600),
    (600, 700),
    (700, 800),
]


RATIO_PAIRS: List[Tuple[int, int]] = [(0, 1), (0, 2), (1, 2)]


def _bandpower(lam: np.ndarray, y: np.ndarray) -> List[float]:
    """Integrate spectral power over predefined wavelength bands."""
    out: List[float] = []
    for lo, hi in BANDS:
        mask = (lam >= lo) & (lam < hi)
        if mask.any():
            out.append(float(np.trapz(y[mask], lam[mask])))
        else:
            out.append(0.0)
    total = sum(out) or 1.0
    return [p / total for p in out]


def _dct16(lam: np.ndarray, y: np.ndarray) -> List[float]:
    grid = np.linspace(lam.min(), lam.max(), 256)
    y_res = np.interp(grid, lam, y)
    coeffs = dct(y_res, norm='ortho')[:16]
    return coeffs.tolist()


def _phash(lam: np.ndarray, y: np.ndarray) -> str:
    """Return a 64‑bit perceptual hash of the spectrum."""
    grid = np.linspace(lam.min(), lam.max(), 256)
    y_res = np.interp(grid, lam, y)
    coeffs = dct(y_res, norm='ortho')[:65]
    med = np.median(coeffs[1:])
    bits = ''.join('1' if c > med else '0' for c in coeffs[1:65])
    return 'phash_v1:' + format(int(bits, 2), '016x')


def _purity(lam: np.ndarray, y: np.ndarray) -> float:
    """Simple peak symmetry metric around the global maximum."""
    idx = int(np.argmax(y))
    step = np.mean(np.diff(lam)) or 1.0
    half = int(round(5.0 / step))  # 5 nm window on each side
    left = y[max(0, idx - half): idx]
    right = y[idx + 1: idx + 1 + half]
    a_left = float(np.sum(left))
    a_right = float(np.sum(right))
    denom = a_left + a_right + 1e-12
    return 1.0 - abs(a_left - a_right) / denom


def compute_features(lam: np.ndarray, y_proc: np.ndarray,
                     sticks: Sequence[Tuple[float, float]],
                     rt: float | None = None) -> Dict:
    """Compute fingerprint features for a spectrum.

    Parameters
    ----------
    lam, y_proc:
        Wavelength axis and processed spectrum.
    sticks:
        Sequence of ``(lambda_nm, intensity)`` tuples representing
        detected peaks. The following intensity ratios are computed:
        (0,1), (0,2) and (1,2) with indices referring to the sorted peak
        list (descending intensity).
    rt:
        Optional retention time.
    """

    lam = np.asarray(lam, dtype=float)
    y = np.asarray(y_proc, dtype=float)
    rel_vals = []
    for s in sticks:
        if isinstance(s, (tuple, list)):
            rel_vals.append(s[1])
        else:
            rel_vals.append(getattr(s, 'intensity', getattr(s, 'rel_intensity', 0.0)))
    rel = np.array(rel_vals, dtype=float)
    rel_sorted = np.sort(rel)[::-1]
    ratios: List[float] = []
    for i, j in RATIO_PAIRS:
        if len(rel_sorted) > max(i, j) and rel_sorted[j] != 0:
            ratios.append(float(rel_sorted[i] / rel_sorted[j]))
        else:
            ratios.append(0.0)

    y_norm = y / (np.max(y) or 1.0)
    feats: Dict[str, object] = {
        'ratio': ratios,
        'bandpower': _bandpower(lam, y_norm),
        'dct16': _dct16(lam, y_norm),
        'phash': _phash(lam, y_norm),
        'lambda_mean': float(np.average(lam, weights=y_norm)),
        'skew': float(skew(y_norm)),
        'kurt': float(kurtosis(y_norm)),
        'snr': float(np.max(y_norm) / (np.std(y_norm) + 1e-12)),
        'n_sticks': int(len(sticks)),
        'purity': _purity(lam, y_norm),
    }
    feats['sticks'] = [
        {'lambda_nm': float(s[0] if isinstance(s, (tuple, list)) else getattr(s, 'lambda_nm', 0.0)),
         'intensity': float(s[1] if isinstance(s, (tuple, list)) else getattr(s, 'intensity', getattr(s, 'rel_intensity', 0.0)))}
        for s in sticks
    ]
    if rt is not None:
        feats['rt'] = float(rt)
    return feats


SCHEMA: Dict[str, object] = {
    'type': 'object',
    'properties': {
        'ratio': {'type': 'array', 'items': {'type': 'number'}},
        'bandpower': {'type': 'array', 'items': {'type': 'number'}},
        'dct16': {'type': 'array', 'items': {'type': 'number'}},
        'phash': {'type': 'string'},
        'lambda_mean': {'type': 'number'},
        'skew': {'type': 'number'},
        'kurt': {'type': 'number'},
        'snr': {'type': 'number'},
        'n_sticks': {'type': 'integer'},
        'purity': {'type': 'number'},
        'rt': {'type': ['number', 'null']},
        'sticks': {'type': 'array'},
    },
    'required': ['ratio', 'bandpower', 'dct16', 'phash', 'lambda_mean',
                 'skew', 'kurt', 'snr', 'n_sticks', 'purity'],
}


__all__ = ['compute_features', 'SCHEMA', 'BANDS']

