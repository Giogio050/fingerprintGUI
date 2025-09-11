"""Data loading utilities for Little Garden HPLC/DAD spectra.

The :func:`load_any` function accepts paths to NPY/NPZ/CSV files and
returns wavelength and spectral arrays along with a meta dictionary.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple
import json
import numpy as np


def _load_npy_pair(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load *_lam.npy and *_spec.npy files."""
    if path.name.endswith('_lam.npy'):
        lam_path = path
        spec_path = path.with_name(path.name.replace('_lam.npy', '_spec.npy'))
    elif path.name.endswith('_spec.npy'):
        spec_path = path
        lam_path = path.with_name(path.name.replace('_spec.npy', '_lam.npy'))
    else:
        raise ValueError('Numpy files must end with _lam.npy or _spec.npy')

    lam = np.load(lam_path)
    spec = np.load(spec_path)
    return lam.astype(float), spec.astype(float)


def load_any(path: str | Path) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Load spectral data from ``path``.

    Parameters
    ----------
    path:
        Path to ``.npy``, ``.npz`` or ``.csv`` file. For NPY files the
        companion *_lam.npy/*_spec.npy pair is detected automatically.

    Returns
    -------
    lam, spec, meta
        ``lam`` is a 1-D wavelength array, ``spec`` can be 1-D or 2-D
        spectral data, and ``meta`` contains information about the
        origin of the data (currently only file name and time stamp).
    """
    p = Path(path)
    meta = {"file": p.name}
    if p.suffix == '.npz':
        data = np.load(p)
        lam = data['wavelength']
        key = 'absorbance_mau' if 'absorbance_mau' in data else 'intensity'
        spec = data[key]
    elif p.suffix == '.npy':
        lam, spec = _load_npy_pair(p)
    elif p.suffix == '.csv':
        arr = np.loadtxt(p, delimiter=',', skiprows=1)
        lam, spec = arr[:, 0], arr[:, 1]
    else:
        raise ValueError(f'Unsupported file type: {p.suffix}')

    meta['mtime'] = p.stat().st_mtime
    return lam, spec, meta
