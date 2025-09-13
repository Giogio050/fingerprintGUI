"""Generation of preprocessing pipeline ensembles."""
from __future__ import annotations

from itertools import product
from typing import Dict, List


def generate_pipelines(n: int = 32) -> List[List[Dict]]:
    """Create a collection of preprocessing pipeline configurations.

    Parameters
    ----------
    n:
        Maximum number of pipelines to return. The generator will yield
        between 32 and 64 combinations based on a small parameter grid.
    """

    baselines = [[], [{'op': 'baseline_asls', 'lam_s': 1e5}]]
    snvs = [[{'op': 'snv'}]]
    sgs = [[{'op': 'sg', 'window': w, 'poly': p}] for w, p in product([5, 7, 9, 11], [2, 3])]
    derivs = [[], [{'op': 'deriv', 'order': 1}]]
    norms = [[{'op': 'area_norm'}], []]

    pipelines: List[List[Dict]] = []
    for base, snv_step, sg_step, deriv_step, norm in product(baselines, snvs, sgs, derivs, norms):
        pipe = base + snv_step + sg_step + deriv_step + norm
        if pipe:
            pipelines.append(pipe)
    return pipelines[: max(32, min(64, n))]


__all__ = ['generate_pipelines']

