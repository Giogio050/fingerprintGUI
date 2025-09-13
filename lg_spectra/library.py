"""Fingerprint library management."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


@dataclass
class _Accum:
    dct16: List[List[float]] = field(default_factory=list)
    ratio: List[List[float]] = field(default_factory=list)
    purity: List[float] = field(default_factory=list)
    hashes: set = field(default_factory=set)
    rt: List[float] = field(default_factory=list)


class Library:
    """Collection of reference fingerprints."""

    def __init__(self) -> None:
        self._accum: Dict[str, _Accum] = {}
        self.entries: Dict[str, Dict] = {}

    # ------------------------------------------------------------------
    def add(self, id: str, features: Dict) -> None:
        """Add a fingerprint to the library under ``id``."""

        acc = self._accum.setdefault(id, _Accum())
        acc.dct16.append(list(features.get('dct16', [])))
        acc.ratio.append(list(features.get('ratio', [])))
        acc.purity.append(float(features.get('purity', 0.0)))
        if 'phash' in features:
            acc.hashes.add(features['phash'])
        if features.get('rt') is not None:
            acc.rt.append(float(features['rt']))

    # ------------------------------------------------------------------
    def _finalise(self) -> None:
        self.entries = {}
        for id, acc in self._accum.items():
            entry: Dict[str, object] = {
                'dct16_mean': np.mean(acc.dct16, axis=0).tolist(),
                'dct16_std': np.std(acc.dct16, axis=0).tolist(),
                'ratio_mean': np.mean(acc.ratio, axis=0).tolist(),
                'ratio_std': np.std(acc.ratio, axis=0).tolist(),
                'purity_mean': float(np.mean(acc.purity)) if acc.purity else 0.0,
                'hashes': list(acc.hashes),
            }
            if acc.rt:
                entry['rt_mean'] = float(np.mean(acc.rt))
                entry['rt_std'] = float(np.std(acc.rt)) if len(acc.rt) > 1 else 0.0
            self.entries[id] = entry

    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        """Save aggregated library to ``path`` as JSON."""

        if not self.entries:
            self._finalise()
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open('w', encoding='utf8') as fh:
            json.dump(self.entries, fh, indent=2)

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path) -> 'Library':
        """Load library from JSON file."""

        inst = cls()
        with Path(path).open('r', encoding='utf8') as fh:
            inst.entries = json.load(fh)
        return inst


def add_replicate(fp: Dict, out_path: str | Path) -> None:
    """Utility to save a single fingerprint as JSON."""

    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w', encoding='utf8') as fh:
        json.dump(fp, fh, indent=2)


def build_index(db_root: str | Path) -> Library:
    """Build a library from a directory tree of JSON fingerprints."""

    lib = Library()
    db_root = Path(db_root)
    for analyte_dir in db_root.iterdir():
        if not analyte_dir.is_dir():
            continue
        for fp_file in (analyte_dir / 'replicates').glob('*.json'):
            features = json.loads(fp_file.read_text())
            lib.add(analyte_dir.name, features)
    lib._finalise()
    out_file = db_root / 'library.json'
    lib.save(out_file)
    return lib


__all__ = ['Library', 'add_replicate', 'build_index']

