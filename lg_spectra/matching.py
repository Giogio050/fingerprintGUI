"""Matching utilities for spectral fingerprints."""
from __future__ import annotations

from typing import Dict, List

import numpy as np


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if not na or not nb:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _hash_sim(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    ha = int(a.split(':', 1)[1], 16)
    hb = int(b.split(':', 1)[1], 16)
    dist = bin(ha ^ hb).count('1')
    return 1.0 - dist / 64.0


def score(sample: Dict, entry: Dict) -> Dict[str, float]:
    """Compute individual similarity components and total score."""

    s_cos = _cosine(np.array(sample.get('dct16', [])),
                    np.array(entry.get('dct16_mean', [])))

    ratio_s = 0.0
    a = np.array(sample.get('ratio', []), dtype=float)
    b = np.array(entry.get('ratio_mean', []), dtype=float)
    if a.size and b.size:
        diff = np.linalg.norm(a[:min(len(a), len(b))] - b[:min(len(a), len(b))])
        denom = np.linalg.norm(entry.get('ratio_std', np.ones_like(b))) or 1.0
        ratio_s = 1.0 - min(1.0, diff / (denom * np.sqrt(len(a))))

    s_rt = 1.0
    if sample.get('rt') is not None and entry.get('rt_mean') is not None:
        sigma = entry.get('rt_std', 0.0) or 1.0
        z = abs(sample['rt'] - entry['rt_mean']) / sigma
        s_rt = 1.0 / (1.0 + z)

    s_purity = 1.0 - abs(sample.get('purity', 0.0) - entry.get('purity_mean', 0.0))
    s_purity = max(0.0, min(1.0, s_purity))
    s_hash = 0.0
    if entry.get('hashes'):
        s_hash = max(_hash_sim(sample.get('phash', ''), h) for h in entry['hashes'])

    total = (0.50 * s_cos + 0.20 * ratio_s + 0.15 * s_rt +
             0.10 * s_purity + 0.05 * s_hash)

    return {
        'S_cos': s_cos,
        'S_ratio': ratio_s,
        'S_rt': s_rt,
        'Purity': s_purity,
        'S_hash': s_hash,
        'S': total,
    }


def match_spectrum(sample_features: Dict, library: 'Library') -> List[Dict]:
    """Score ``sample_features`` against all entries in ``library``.

    Returns a list of dictionaries sorted by descending total score. Each
    dictionary contains the library ``id`` and an ``ampel`` key indicating
    match quality (``green``, ``yellow`` or ``red``).
    """

    results: List[Dict] = []
    for id, entry in library.entries.items():
        sc = score(sample_features, entry)
        sc['id'] = id
        if sc['S'] >= 0.85:
            sc['ampel'] = 'green'
        elif sc['S'] >= 0.70:
            sc['ampel'] = 'yellow'
        else:
            sc['ampel'] = 'red'
        results.append(sc)
    results.sort(key=lambda r: r['S'], reverse=True)
    return results


from .vectorize import sticks_to_vector  # noqa: F401  (re-export if needed)

__all__ = ['score', 'match_spectrum']

