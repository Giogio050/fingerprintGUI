"""Data I/O helpers for Little Garden spectral fingerprints."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List
import json
import time

import numpy as np


_WL_ALIASES = ["wavelength", "lam", "lambda_nm"]
_INT_ALIASES = ["intensity", "i"]
_ABS_ALIASES = ["absorbance_mau", "absorbance", "a_mau"]


def _find_key(keys: Iterable[str], aliases: List[str]) -> str | None:
    for a in aliases:
        if a in keys:
            return a
    return None


def _load_npy_pair(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load *_lam.npy/*_spec.npy files."""
    if path.name.endswith("_lam.npy"):
        lam_path = path
        spec_path = path.with_name(path.name.replace("_lam.npy", "_spec.npy"))
    elif path.name.endswith("_spec.npy"):
        spec_path = path
        lam_path = path.with_name(path.name.replace("_spec.npy", "_lam.npy"))
    else:
        raise ValueError("Numpy files must end with _lam.npy or _spec.npy")

    if not lam_path.exists() or not spec_path.exists():
        raise FileNotFoundError(
            f"Missing companion file for '{path.name}'. Expected '{lam_path.name}' and '{spec_path.name}'."
        )
    lam = np.load(lam_path).astype(float)
    spec = np.load(spec_path).astype(float)
    return lam, spec


def load_any(path: str | Path) -> Dict:
    """Load spectral data from ``path``.

    Returns a dict with keys ``wavelength``, ``intensity`` and ``absorbance_mau``
    (latter two may be ``None``) plus a ``meta`` dictionary.
    """

    p = Path(path)
    meta = {"filename": p.name, "mtime": p.stat().st_mtime}

    if p.suffix == ".npz":
        data = np.load(p, allow_pickle=True)
        key_wl = _find_key(data.files, _WL_ALIASES)
        key_int = _find_key(data.files, _INT_ALIASES)
        key_abs = _find_key(data.files, _ABS_ALIASES)
        if key_wl is None:
            raise KeyError("wavelength key missing in NPZ")
        lam = data[key_wl].astype(float)
        intensity = data[key_int].astype(float) if key_int else None
        absorb = data[key_abs].astype(float) if key_abs else None
        if "meta" in data:
            try:
                meta.update(json.loads(str(data["meta"].item())))
            except Exception:
                pass

    elif p.suffix == ".npy":
        if p.name.endswith("_lam.npy"):
            lam, spec = _load_npy_pair(p)
            intensity = spec
            absorb = None
        elif p.name.endswith("_spec.npy"):
            lam_path = p.with_name(p.name.replace("_spec.npy", "_lam.npy"))
            if lam_path.exists():
                lam, spec = _load_npy_pair(p)
                intensity = spec
                absorb = None
            else:
                arr = np.load(p).astype(float)
                if arr.ndim == 2 and 2 in arr.shape:
                    if arr.shape[0] == 2:
                        lam, spec = arr[0], arr[1]
                    elif arr.shape[1] == 2:
                        lam, spec = arr[:, 0], arr[:, 1]
                    else:
                        raise ValueError(
                            f"Standalone _spec.npy must contain wavelength as first row or column: '{p.name}'"
                        )
                    intensity = spec
                    absorb = None
                else:
                    raise FileNotFoundError(
                        f"Missing companion _lam.npy for '{p.name}' and no embedded wavelength"
                    )
        else:
            arr = np.load(p).astype(float)
            if arr.ndim == 2 and 2 in arr.shape:
                if arr.shape[0] == 2:
                    lam, spec = arr[0], arr[1]
                elif arr.shape[1] == 2:
                    lam, spec = arr[:, 0], arr[:, 1]
                else:
                    raise ValueError(
                        f"Standalone _spec.npy must contain wavelength as first row or column: '{p.name}'"
                    )
                intensity = spec
                absorb = None
            else:
                raise ValueError(f"Unsupported NPY format for '{p.name}'")

    elif p.suffix == ".csv":
        arr = np.genfromtxt(p, delimiter=",", names=True, dtype=float)
        names = {n.lower(): n for n in arr.dtype.names or []}
        key_wl = _find_key(names, _WL_ALIASES)
        if key_wl is None:
            raise KeyError("wavelength column missing in CSV")
        lam = arr[names[key_wl]].astype(float)
        key_int = _find_key(names, _INT_ALIASES)
        key_abs = _find_key(names, _ABS_ALIASES)
        intensity = arr[names[key_int]].astype(float) if key_int else None
        absorb = arr[names[key_abs]].astype(float) if key_abs else None
    else:
        raise ValueError(f"Unsupported file type: {p.suffix}")

    return {
        "wavelength": lam,
        "intensity": intensity,
        "absorbance_mau": absorb,
        "meta": meta,
    }


def load_folder(
    path: str | Path,
    patterns: Iterable[str] = ("*.npz", "*.csv", "*_spec.npy", "*_lam.npy"),
) -> List[Dict]:
    """Load all spectra found in ``path`` matching ``patterns``."""

    folder = Path(path)
    files: List[Path] = []
    for pat in patterns:
        files.extend(folder.glob(pat))

    results: List[Dict] = []
    handled: set[Path] = set()
    for f in sorted(files):
        if f in handled:
            continue
        if f.name.endswith("_lam.npy"):
            spec = f.with_name(f.name.replace("_lam.npy", "_spec.npy"))
            handled.add(f)
            if spec.exists():
                handled.add(spec)
            results.append(load_any(f))
        elif f.name.endswith("_spec.npy"):
            lam = f.with_name(f.name.replace("_spec.npy", "_lam.npy"))
            handled.add(f)
            if lam.exists():
                handled.add(lam)
            results.append(load_any(f))
        else:
            handled.add(f)
            results.append(load_any(f))
    return results


def save_spec(
    path: str | Path,
    wavelength: np.ndarray,
    *,
    intensity: np.ndarray | None = None,
    absorbance_mau: np.ndarray | None = None,
    exposure: float = 0.0,
    gain: float = 0.0,
    stack_ms: float = 0.0,
    stack_n: int = 1,
    snr: float | None = None,
    flicker: float | None = None,
    source: str = "",
    matrix: str | None = None,
    roi: str | None = None,
) -> None:
    """Save spectral data to a compressed NPZ file."""

    p = Path(path)
    meta = {
        "filename": p.name,
        "timestamp": time.time(),
        "source": source,
    }
    if matrix is not None:
        meta["matrix"] = matrix
    if roi is not None:
        meta["roi"] = roi

    np.savez_compressed(
        p,
        wavelength=np.asarray(wavelength, dtype=float),
        intensity=None if intensity is None else np.asarray(intensity, dtype=float),
        absorbance_mau=(
            None if absorbance_mau is None else np.asarray(absorbance_mau, dtype=float)
        ),
        exposure=float(exposure),
        gain=float(gain),
        stack_ms=float(stack_ms),
        stack_n=int(stack_n),
        snr=None if snr is None else float(snr),
        flicker=None if flicker is None else float(flicker),
        meta=json.dumps(meta),
    )
