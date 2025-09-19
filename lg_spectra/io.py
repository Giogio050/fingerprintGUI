"""Data I/O helpers for Little Garden spectral fingerprints."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple
import time

import numpy as np


__all__ = ["MASTER_GRID", "SpectralRecord", "load_any", "load_folder", "save_spec"]


MASTER_GRID = np.arange(360.0, 801.0, 1.0)

_WL_ALIASES = ["wavelength", "lam", "lambda_nm"]
_INT_ALIASES = ["intensity", "i", "absorbance", "signal"]
_ABS_ALIASES = ["absorbance_mau", "absorbance", "a_mau"]


@dataclass(slots=True)
class SpectralRecord:
    """Container for spectra loaded from disk."""

    wavelength: np.ndarray
    spectrum: np.ndarray
    meta: Dict[str, object]

    def on_master_grid(self) -> "SpectralRecord":
        """Return a copy interpolated onto :data:`MASTER_GRID`."""

        lam = np.asarray(self.wavelength, dtype=float)
        if np.allclose(lam, MASTER_GRID):
            return self
        spectrum = np.asarray(self.spectrum, dtype=float)
        if spectrum.ndim == 1:
            interp = np.interp(MASTER_GRID, lam, spectrum, left=0.0, right=0.0)
        else:
            interp = np.stack(
                [
                    np.interp(MASTER_GRID, lam, frame, left=0.0, right=0.0)
                    for frame in spectrum
                ],
                axis=0,
            )
        return SpectralRecord(MASTER_GRID.copy(), interp, dict(self.meta))


def _find_key(keys: Sequence[str], aliases: Sequence[str]) -> str | None:
    for alias in aliases:
        if alias in keys:
            return alias
    return None


def _load_npy_pair(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load paired ``*_lam.npy``/``*_spec.npy`` arrays."""

    if path.name.endswith("_lam.npy"):
        lam_path = path
        spec_path = path.with_name(path.name.replace("_lam.npy", "_spec.npy"))
    elif path.name.endswith("_spec.npy"):
        spec_path = path
        lam_path = path.with_name(path.name.replace("_spec.npy", "_lam.npy"))
    else:  # pragma: no cover - guarded by caller
        raise ValueError("Unexpected file suffix for numpy pair")

    if not lam_path.exists() or not spec_path.exists():
        raise FileNotFoundError(
            "Missing companion file for '" + path.name + "' in pair loading"
        )
    lam = np.load(lam_path).astype(float)
    spec = np.load(spec_path).astype(float)
    return lam, spec


def _coerce_spectrum(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 1:
        return arr.astype(float)
    if arr.ndim == 2:
        return arr.astype(float)
    raise ValueError("Spectral array must be 1-D or 2-D")


def _load_embedded(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.load(path).astype(float)
    if arr.ndim != 2 or 2 not in arr.shape:
        raise ValueError(
            f"Standalone numpy file '{path.name}' must contain wavelength/spec columns"
        )
    if arr.shape[0] == 2:
        lam, spec = arr[0], arr[1]
    else:
        lam, spec = arr[:, 0], arr[:, 1]
    return lam.astype(float), np.asarray(spec, dtype=float)


def _load_csv(path: Path) -> Tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    arr = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
    names = [n or "" for n in (arr.dtype.names or ())]
    lower = {n.lower(): n for n in names}
    key_wl = _find_key(lower.keys(), _WL_ALIASES)
    if key_wl is None:
        raise KeyError("wavelength column missing in CSV")
    wl = arr[lower[key_wl]].astype(float)
    key_int = _find_key(lower.keys(), _INT_ALIASES)
    key_abs = _find_key(lower.keys(), _ABS_ALIASES)
    intensity = arr[lower[key_int]].astype(float) if key_int else None
    absorb = arr[lower[key_abs]].astype(float) if key_abs else None
    return wl, intensity, absorb


def _npz_meta(data: np.lib.npyio.NpzFile) -> Dict[str, object]:
    meta: Dict[str, object] = {}
    if "meta" in data.files:
        try:
            meta_obj = data["meta"].item()
            if isinstance(meta_obj, bytes):
                meta.update(json.loads(meta_obj.decode("utf8")))
            elif isinstance(meta_obj, str):
                meta.update(json.loads(meta_obj))
            elif isinstance(meta_obj, dict):
                meta.update(meta_obj)
        except Exception:  # pragma: no cover - defensive parsing
            pass
    return meta


def load_any(path: str | Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """Load spectral data from *path*.

    Parameters
    ----------
    path:
        File to load. Supported formats are NPY/NPZ/CSV according to the
        project specification.

    Returns
    -------
    tuple
        A tuple ``(wavelength, spectrum, meta)`` where the spectrum is either a
        1-D trace or a 2-D array of frames x wavelengths.
    """

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    meta: Dict[str, object] = {
        "filename": p.name,
        "mtime": p.stat().st_mtime,
    }

    if p.suffix == ".npz":
        with np.load(p, allow_pickle=True) as data:
            key_wl = _find_key(data.files, _WL_ALIASES)
            if key_wl is None:
                raise KeyError("wavelength key missing in NPZ")
            wl = np.asarray(data[key_wl], dtype=float)
            key_int = _find_key(data.files, _INT_ALIASES)
            key_abs = _find_key(data.files, _ABS_ALIASES)
            intensity = (
                np.asarray(data[key_int], dtype=float) if key_int else None
            )
            absorb = np.asarray(data[key_abs], dtype=float) if key_abs else None
            spectrum = intensity if intensity is not None else absorb
            if spectrum is None:
                raise KeyError("No intensity/absorbance data in NPZ")
            meta.update(_npz_meta(data))
    elif p.suffix == ".npy":
        if p.name.endswith("_lam.npy"):
            wl, spectrum = _load_npy_pair(p)
        elif p.name.endswith("_spec.npy"):
            companion = p.with_name(p.name.replace("_spec.npy", "_lam.npy"))
            if companion.exists():
                wl, spectrum = _load_npy_pair(p)
            else:
                try:
                    wl, spectrum = _load_embedded(p)
                except ValueError as exc:
                    raise FileNotFoundError(
                        f"Missing companion file for '{p.name}'"
                    ) from exc
        else:
            wl, spectrum = _load_embedded(p)
    elif p.suffix.lower() == ".csv":
        wl, intensity, absorb = _load_csv(p)
        spectrum = intensity if intensity is not None else absorb
        if spectrum is None:
            raise KeyError("CSV must contain intensity or absorbance column")
    else:
        raise ValueError(f"Unsupported file format: {p.suffix}")

    spectrum = _coerce_spectrum(np.asarray(spectrum))
    wl = np.asarray(wl, dtype=float)
    if wl.ndim != 1:
        raise ValueError("Wavelength axis must be one-dimensional")

    if spectrum.ndim == 2 and spectrum.shape[-1] != wl.shape[0]:
        raise ValueError("Spectrum frame dimension does not match wavelength axis")

    return wl, spectrum, meta


def _iter_folder_matches(folder: Path, patterns: Iterable[str]) -> Iterator[Path]:
    for pattern in patterns:
        yield from folder.glob(pattern)


def load_folder(
    path: str | Path,
    patterns: Iterable[str] = ("*.npz", "*.csv", "*_spec.npy", "*_lam.npy"),
) -> List[Dict[str, object]]:
    """Load all matching spectra from *path*.

    The return value is a list of dictionaries with ``wavelength``,
    ``spectrum`` and ``meta`` keys. Duplicate spectrum/lam pairs are
    handled automatically.
    """

    folder = Path(path)
    if not folder.exists():
        raise FileNotFoundError(str(folder))

    files = sorted(set(_iter_folder_matches(folder, patterns)))
    handled: set[Path] = set()
    result: List[Dict[str, object]] = []
    for file in files:
        if file in handled:
            continue
        if file.name.endswith("_lam.npy"):
            companion = file.with_name(file.name.replace("_lam.npy", "_spec.npy"))
            handled.add(file)
            if companion.exists():
                handled.add(companion)
        elif file.name.endswith("_spec.npy"):
            companion = file.with_name(file.name.replace("_spec.npy", "_lam.npy"))
            handled.add(file)
            if companion.exists():
                handled.add(companion)
        else:
            handled.add(file)
        wl, spec, meta = load_any(file)
        result.append({"wavelength": wl, "spectrum": spec, "meta": meta})
    return result


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
