"""Selected ion monitoring style traces from spectral data."""
from __future__ import annotations

from typing import Dict, Sequence
import numpy as np


def sim_traces(
    lam: np.ndarray,
    frames: np.ndarray,
    target_lambdas: Sequence[float],
    band: int = 2,
) -> Dict[str, np.ndarray]:
    """Extract narrow-band traces around ``target_lambdas``.

    Parameters
    ----------
    lam:
        Wavelength axis.
    frames:
        2-D array (time x wavelength) of spectra.
    target_lambdas:
        Iterable of central wavelengths to integrate around.
    band:
        Half width of the band in nm.
    """
    traces: Dict[str, np.ndarray] = {}
    for tlam in target_lambdas:
        mask = (lam >= tlam - band) & (lam <= tlam + band)
        if not mask.any():
            continue
        traces[f"{tlam:.1f}"] = frames[:, mask].mean(axis=1)
    return traces
