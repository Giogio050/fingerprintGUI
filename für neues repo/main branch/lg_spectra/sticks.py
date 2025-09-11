"""Peak picking and stick fingerprint generation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List
import numpy as np
from scipy.signal import find_peaks, peak_widths


@dataclass
class Stick:
    lambda_nm: float
    rel_intensity: float
    width_nm: float
    prominence: float


def pick_sticks(lam: np.ndarray, y: np.ndarray, params: dict | None = None) -> List[Stick]:
    """Detect peaks and return the top-K as sticks.

    Parameters
    ----------
    lam, y:
        Wavelength and preprocessed spectrum.
    params:
        ``dict`` with keys ``k`` (top-N peaks) and ``prominence`` and
        ``distance`` forwarded to :func:`find_peaks`.
    """
    params = params or {}
    k = params.get('k', 6)
    peaks, props = find_peaks(y, prominence=params.get('prominence', 0.001),
                              distance=params.get('distance', 5))
    if len(peaks) == 0:
        return []
    widths = peak_widths(y, peaks, rel_height=0.5)
    rel = y[peaks] / y[peaks].max() * 100.0
    sticks = [
        Stick(lambda_nm=float(lam[p]), rel_intensity=float(r),
              width_nm=float(w), prominence=float(props['prominences'][i]))
        for i, (p, r, w) in enumerate(zip(peaks, rel, widths[0]))
    ]
    sticks.sort(key=lambda s: s.rel_intensity, reverse=True)
    return sticks[:k]
