"""Library management utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

from .sticks import Stick


def add_replicate(fp: Dict, out_path: str | Path) -> None:
    """Save a fingerprint dictionary to ``out_path`` in JSON format."""

    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf8") as fh:
        json.dump(fp, fh, indent=2)


def _load_replicates(folder: Path) -> List[Dict]:
    return [json.loads(p.read_text()) for p in sorted(folder.glob("*.json"))]


def _aggregate_sticks(replicates: Iterable[Dict]) -> List[Dict[str, float]]:
    sticks_per_rep = [[Stick(**s) for s in rep.get("sticks", [])] for rep in replicates]
    max_len = max((len(s) for s in sticks_per_rep), default=0)
    aggregated: List[Dict[str, float]] = []
    for idx in range(max_len):
        lambdas = []
        rel = []
        widths = []
        prominences = []
        for sticks in sticks_per_rep:
            if idx < len(sticks):
                stick = sticks[idx]
                lambdas.append(stick.lambda_nm)
                rel.append(stick.rel_intensity)
                widths.append(stick.width_nm)
                prominences.append(stick.prominence)
        if lambdas:
            aggregated.append(
                {
                    "lambda_nm": float(np.mean(lambdas)),
                    "rel_intensity": float(np.mean(rel)),
                    "width_nm": float(np.mean(widths)),
                    "prominence": float(np.mean(prominences)),
                }
            )
    return aggregated


def _agg_array(values: List[List[float]]) -> Dict[str, List[float]]:
    if not values:
        return {"mean": [], "std": []}
    min_len = min(len(v) for v in values)
    if min_len == 0:
        return {"mean": [], "std": []}
    arr = np.array([v[:min_len] for v in values], dtype=float)
    return {"mean": arr.mean(axis=0).tolist(), "std": arr.std(axis=0).tolist()}


def build_index(db_root: str | Path, out_file: str | Path | None = None) -> Dict[str, Dict]:
    """Build an aggregated feature index for all analytes in ``db_root``."""

    db_root = Path(db_root)
    index: Dict[str, Dict] = {}
    for analyte_dir in sorted(db_root.iterdir()):
        if not analyte_dir.is_dir():
            continue
        replicate_dir = analyte_dir / "replicates"
        if not replicate_dir.exists():
            continue
        reps = _load_replicates(replicate_dir)
        if not reps:
            continue
        ratios = [rep.get("ratios", []) for rep in reps]
        bandpower = [rep.get("bandpower", []) for rep in reps]
        dct16 = [rep.get("dct16", []) for rep in reps]
        rt_vals = [rep.get("rt_min") for rep in reps if rep.get("rt_min") is not None]
        hash_values = [rep.get("hash") for rep in reps if rep.get("hash")]
        qualities = [rep.get("quality", {}) for rep in reps]

        ratio_stats = _agg_array([r for r in ratios if r])

        entry: Dict[str, object] = {
            "ratios_mean": ratio_stats["mean"],
            "ratios_std": ratio_stats["std"],
            "bandpower_mean": np.mean(np.vstack(bandpower), axis=0).tolist() if bandpower else [],
            "bandpower_std": np.std(np.vstack(bandpower), axis=0).tolist() if bandpower else [],
            "dct16_mean": np.mean(np.vstack(dct16), axis=0).tolist() if dct16 else [],
            "sticks": _aggregate_sticks(reps),
            "hash": max(set(hash_values), key=hash_values.count) if hash_values else "",
            "rt_min": float(np.mean(rt_vals)) if rt_vals else None,
            "quality_mean": {
                "snr": float(np.mean([q.get("snr", 0.0) for q in qualities])),
                "purity": float(np.mean([q.get("purity", 0.0) for q in qualities])),
            },
        }
        meta_file = analyte_dir / "meta.json"
        if meta_file.exists():
            entry["meta"] = json.loads(meta_file.read_text())
            entry["tolerances"] = entry["meta"].get("tolerances", {})
        index[analyte_dir.name] = entry

    if out_file is None:
        out_path = db_root / "library_index.json"
    else:
        out_path = Path(out_file)
    out_path.write_text(json.dumps(index, indent=2))
    return index
