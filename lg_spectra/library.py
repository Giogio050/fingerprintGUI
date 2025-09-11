"""Library management utilities."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict
import numpy as np


def add_replicate(fp: Dict, out_path: str | Path) -> None:
    """Save fingerprint ``fp`` to ``out_path`` in JSON format."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w', encoding='utf8') as fh:
        json.dump(fp, fh, indent=2)


def build_index(db_root: str | Path) -> Dict:
    """Build an index of mean/std features for each analyte folder."""
    db_root = Path(db_root)
    index: Dict[str, Dict] = {}
    for analyte_dir in db_root.iterdir():
        if not analyte_dir.is_dir():
            continue
        reps = list((analyte_dir / 'replicates').glob('*.json'))
        if not reps:
            continue
        feats = [json.loads(p.read_text()) for p in reps]
        ratios = np.array([f.get('ratios', []) for f in feats], dtype=float)
        bandpower = np.array([f.get('bandpower', []) for f in feats], dtype=float)
        index[analyte_dir.name] = {
            'ratios_mean': ratios.mean(axis=0).tolist() if ratios.size else [],
            'ratios_std': ratios.std(axis=0).tolist() if ratios.size else [],
            'bandpower_mean': bandpower.mean(axis=0).tolist() if bandpower.size else [],
            'bandpower_std': bandpower.std(axis=0).tolist() if bandpower.size else [],
            'sticks': feats[0].get('sticks', []),
        }
    out_file = db_root / 'library_index.json'
    with out_file.open('w', encoding='utf8') as fh:
        json.dump(index, fh, indent=2)
    return index
