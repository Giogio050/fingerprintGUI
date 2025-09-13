"""Peak detection and stick representation utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from scipy.signal import find_peaks as _find_peaks, peak_widths


@dataclass
class Stick:
    """Representation of a spectral peak."""

    lambda_nm: float
    intensity: float
    width_nm: float = 0.0
    prominence: float = 0.0


def find_peaks(lam: np.ndarray, y: np.ndarray,
               min_prominence: float = 0.001,
               min_distance_nm: float = 1.0) -> List[Tuple[float, float]]:
    """Locate spectral peaks.

    Parameters
    ----------
    lam, y:
        Wavelength axis and spectrum.
    min_prominence:
        Minimum prominence forwarded to :func:`scipy.signal.find_peaks`.
    min_distance_nm:
        Minimum distance between peaks in nanometres.
    """

    step = np.mean(np.diff(lam)) or 1.0
    distance = int(round(min_distance_nm / step))
    peaks, _ = _find_peaks(y, prominence=min_prominence, distance=distance)
    return [(float(lam[p]), float(y[p])) for p in peaks]


def to_sticks(lam: np.ndarray, y: np.ndarray, bin_nm: float = 1.0) -> np.ndarray:
    """Quantise a spectrum to a stick representation.

    The output is an ``(N,2)`` array with wavelength/intensity pairs on a
    regular grid defined by ``bin_nm``.
    """

    grid = np.arange(np.floor(lam.min()), np.ceil(lam.max()) + bin_nm, bin_nm)
    y_res = np.interp(grid, lam, y)
    return np.column_stack([grid, y_res])


def pick_sticks(lam: np.ndarray, y: np.ndarray, params: dict | None = None) -> List[Stick]:
    """Detect peaks and return the top-K as :class:`Stick` objects."""

    params = params or {}
    k = params.get('k', 6)
    min_prom = params.get('prominence', 0.001)
    min_dist = params.get('distance', 5.0)
    step = np.mean(np.diff(lam)) or 1.0
    distance = int(round(min_dist / step))
    peaks, props = _find_peaks(y, prominence=min_prom, distance=distance)
    if len(peaks) == 0:
        return []
    widths = peak_widths(y, peaks, rel_height=0.5)[0] * step
    sticks: List[Stick] = [
        Stick(lambda_nm=float(lam[p]), intensity=float(y[p]),
              width_nm=float(w), prominence=float(props['prominences'][i]))
        for i, (p, w) in enumerate(zip(peaks, widths))
    ]
    sticks.sort(key=lambda s: s.intensity, reverse=True)
    return sticks[:k]


__all__ = ['Stick', 'find_peaks', 'to_sticks', 'pick_sticks']

