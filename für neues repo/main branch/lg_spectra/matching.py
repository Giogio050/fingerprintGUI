"""Fingerprint matching utilities."""
from __future__ import annotations

from typing import Dict
import numpy as np

from .vectorize import sticks_to_vector


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if not na or not nb:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _ratio_score(a: list, b: list) -> float:
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    diff = np.abs(np.array(a[:n]) - np.array(b[:n]))
    return float(1 - np.mean(diff))


def score(sample_fp: Dict, library_entry: Dict) -> Dict[str, float]:
    """Compute similarity scores between a sample fingerprint and a library entry."""
    vec_sample = sticks_to_vector([Stick(**s) for s in sample_fp['sticks']]) if 'sticks' in sample_fp else np.array([])
    vec_lib = sticks_to_vector([Stick(**s) for s in library_entry['sticks']]) if 'sticks' in library_entry else np.array([])
    s_cos = _cosine(vec_sample, vec_lib)
    s_ratio = _ratio_score(sample_fp.get('ratios', []), library_entry.get('ratios', []))
    s_hash = 1.0 if sample_fp.get('hash') == library_entry.get('hash') else 0.0
    purity = sample_fp.get('quality', {}).get('purity', 0.0)
    rt_pen = library_entry.get('rt_min') and sample_fp.get('rt_min')
    s_rt = 1 - abs(sample_fp.get('rt_min', 0) - library_entry.get('rt_min', 0)) / max(library_entry.get('rt_min', 1), 1)
    score_total = 0.5 * s_cos + 0.2 * s_ratio + 0.1 * s_rt + 0.1 * purity + 0.1 * s_hash
    return {
        'S_cos': s_cos,
        'S_ratio': s_ratio,
        'S_rt': s_rt,
        'Purity': purity,
        'S_hash': s_hash,
        'S': score_total,
    }

# needed dataclass import at top
from .sticks import Stick
