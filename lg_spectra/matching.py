"""Fingerprint matching utilities."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from scipy.stats import kendalltau, wasserstein_distance

from .sticks import Stick
from .vectorize import sticks_to_vector


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _kendall_rank(sample: Sequence[Stick], library: Sequence[Stick]) -> float:
    if not sample or not library:
        return 0.0
    s = np.array([s.rel_intensity for s in sample])
    l = np.array([s.rel_intensity for s in library])
    min_len = min(len(s), len(l))
    tau, _ = kendalltau(np.argsort(-s)[:min_len], np.argsort(-l)[:min_len])
    return float(tau) if tau == tau else 0.0


def _earth_mover(a: np.ndarray, b: np.ndarray) -> float:
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    a_pdf = a / a.sum()
    b_pdf = b / b.sum()
    x = np.arange(a.size)
    dist = wasserstein_distance(x, x, a_pdf, b_pdf)
    return float(1.0 / (1.0 + dist))


def _ratio_score(sample: Sequence[float], library: Sequence[float], std: Sequence[float] | None = None) -> float:
    if not sample or not library:
        return 0.0
    n = min(len(sample), len(library))
    sample = np.array(sample[:n])
    library = np.array(library[:n])
    if std is not None and len(std) >= n:
        tolerance = np.array(std[:n]) + 0.05
    else:
        tolerance = np.full(n, 0.1)
    diff = np.abs(sample - library)
    return float(np.clip(1 - (diff / (tolerance + 1e-9)), 0, 1).mean())


def _rt_penalty(sample_rt: float | None, library_rt: float | None, tol_pct: float) -> float:
    if sample_rt is None or library_rt is None:
        return 0.5
    diff = abs(sample_rt - library_rt)
    tol = max(library_rt * tol_pct / 100.0, 0.1)
    score = max(0.0, 1 - diff / tol)
    return float(score)


def _hash_cosine(sample_dct: Sequence[float], library_dct: Sequence[float]) -> float:
    if not sample_dct or not library_dct:
        return 0.0
    return _cosine(np.array(sample_dct[:16]), np.array(library_dct[:16]))


def score(sample_fp: Dict, library_entry: Dict) -> Dict[str, float]:
    """Compute similarity scores between a sample fingerprint and a library entry."""

    sample_sticks = [Stick(**s) for s in sample_fp.get("sticks", [])]
    library_sticks = [Stick(**s) for s in library_entry.get("sticks", [])]
    vec_sample = sticks_to_vector(sample_sticks)
    vec_library = sticks_to_vector(library_sticks)

    s_cos = _cosine(vec_sample, vec_library)
    s_ratio = _ratio_score(
        sample_fp.get("ratios", []),
        library_entry.get("ratios_mean", library_entry.get("ratios", [])),
        library_entry.get("ratios_std"),
    )
    s_hash = _hash_cosine(sample_fp.get("dct16", []), library_entry.get("dct16_mean", []))
    tolerances = library_entry.get("tolerances", {})
    s_rt = _rt_penalty(
        sample_fp.get("rt_min"),
        library_entry.get("rt_min"),
        float(tolerances.get("rt_rel_pct", 15.0)),
    )
    purity = float(sample_fp.get("quality", {}).get("purity", 0.0))
    snr = float(sample_fp.get("quality", {}).get("snr", 0.0))

    total = 0.5 * s_cos + 0.2 * s_ratio + 0.15 * s_rt + 0.1 * purity + 0.05 * s_hash

    return {
        "S_cos": s_cos,
        "S_ratio": s_ratio,
        "S_rt": s_rt,
        "S_hash": s_hash,
        "Purity": purity,
        "S": total,
        "S_kendall": _kendall_rank(sample_sticks, library_sticks),
        "S_emd": _earth_mover(vec_sample, vec_library),
        "SNR": snr,
        "n_sticks": float(len(sample_sticks)),
    }
