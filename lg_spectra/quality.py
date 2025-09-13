"""Quality assessment helpers for spectral fingerprints."""
from __future__ import annotations

from typing import Dict, List, Sequence
import numpy as np


def check_warnings(fp: Dict, lam: Sequence[float] | None = None,
                   y: Sequence[float] | None = None,
                   k_min: int = 3, snr_min: float = 10.0) -> List[str]:
    """Return human readable warnings about fingerprint quality.

    Parameters
    ----------
    fp:
        Fingerprint dictionary containing a ``quality`` section.
    lam, y:
        Optional wavelength axis and corresponding intensities to
        perform additional checks such as absorption above 360 nm.
    k_min:
        Minimum required number of sticks.
    snr_min:
        Minimum acceptable signal-to-noise ratio.
    """
    warnings: List[str] = []
    q = fp.get('quality', {})
    if q.get('n_sticks', 0) < k_min:
        warnings.append(f"n_sticks < {k_min}")
    if q.get('snr', 0.0) < snr_min:
        warnings.append(f"snr below {snr_min}")
    if lam is not None and y is not None:
        lam_arr = np.asarray(lam)
        y_arr = np.asarray(y)
        if not np.any((lam_arr >= 360) & (y_arr > 0)):
            warnings.append("no absorption >= 360 nm")
    return warnings
