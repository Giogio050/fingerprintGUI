"""Selected ion monitoring style traces from spectral data."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


def sim_traces(
    lam: np.ndarray,
    frames: np.ndarray,
    target_lambdas: Sequence[float],
    band: int = 2,
) -> Dict[str, Sequence[float]]:
    """Extract narrow-band traces around ``target_lambdas``.

    Returns a dictionary with the synthetic chromatogram for each target
    wavelength and a ``time`` axis representing the frame indices.
    """

    lam = np.asarray(lam, dtype=float)
    frames = np.asarray(frames, dtype=float)
    if frames.ndim != 2:
        raise ValueError("frames must be a 2-D array")

    time_axis = np.arange(frames.shape[0], dtype=float)
    traces: Dict[str, Sequence[float]] = {"time": time_axis.tolist()}
    for tlam in sorted(set(round(float(x), 3) for x in target_lambdas)):
        mask = (lam >= tlam - band) & (lam <= tlam + band)
        if not mask.any():
            continue
        trace = frames[:, mask].mean(axis=1)
        traces[f"{tlam:.3f}"] = trace.tolist()
    return traces
