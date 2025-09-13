"""Feature extraction from spectra and sticks."""
from __future__ import annotations

from typing import Dict, List
import numpy as np
from scipy.stats import entropy, skew, kurtosis
from scipy.fft import dct

from .sticks import Stick
from .quality import check_warnings


BANDS = [(360, 400), (400, 450), (450, 500), (500, 600), (600, 700), (700, 800)]


def _bandpower(lam: np.ndarray, y: np.ndarray) -> List[float]:
    powers = []
    for lo, hi in BANDS:
        mask = (lam >= lo) & (lam < hi)
        if mask.any():
            powers.append(float(np.trapz(y[mask], lam[mask])))
        else:
            powers.append(0.0)
    total = sum(powers) or 1.0
    return [p / total for p in powers]


def _phash(y: np.ndarray) -> str:
    small = np.interp(np.linspace(0, len(y) - 1, 32), np.arange(len(y)), y)
    coeffs = dct(small, norm='ortho')[:16]
    med = np.median(coeffs[1:])
    bits = ''.join('1' if c > med else '0' for c in coeffs[1:])
    return 'phash_v1:' + hex(int(bits, 2))[2:]


def compute_features(lam: np.ndarray, y: np.ndarray, sticks: List[Stick]) -> Dict:
    """Compute fingerprint features from processed spectrum and its sticks."""
    rel = np.array([s.rel_intensity for s in sticks])
    ratios = (rel[1:] / rel[0]) if len(rel) > 1 else np.array([])
    y_norm = y / np.max(y) if np.max(y) else y
    max_rel = float(rel.max()) if rel.size else 0.0
    feats = {
        'sticks': [s.__dict__ for s in sticks],
        'ratios': ratios.tolist(),
        'entropy': float(entropy(np.abs(y_norm) + 1e-12)),
        'bandpower': _bandpower(lam, y_norm),
        'dct16': dct(y_norm, norm='ortho')[:16].tolist(),
        'hash': _phash(y_norm),
        'global': {
            'lambda_mean': float(np.average(lam, weights=y_norm)),
            'skew': float(skew(y_norm)),
            'kurt': float(kurtosis(y_norm)),
        },
        'quality': {
            'snr': float(max_rel / (np.std(y_norm) + 1e-9)),
            'purity': float(max_rel / (np.sum(rel) + 1e-9)) if rel.size else 0.0,
            'n_sticks': int(len(sticks)),
        },
    }
    feats['warnings'] = check_warnings(feats, lam=lam, y=y_norm)
    return feats
