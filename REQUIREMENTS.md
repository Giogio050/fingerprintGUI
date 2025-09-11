# Projektanforderungen für DIY-HPLC/DAD Fingerprint App

Dieses Dokument fasst den ursprünglichen Grundprompt zusammen und dient als Referenz für alle Kernfunktionen des Projekts.

## Ziel
Entwicklung einer Windows-freundlichen Python-Anwendung (GUI + CLI) zur Visualisierung und Identifikation von Spektren eines DIY-HPLC/DAD (Little-Garden, 360–1050 nm). Hauptfokus ist eine hohe Identifikationssensitivität für kleine Peaks durch ein Ensemble von 50–100 Preprocessing- und Matching-Methoden.

## Rahmenbedingungen
- Python ≥3.10 (32/64-bit)
- Erlaubte Libraries: `numpy`, `scipy` (signal), `pyqtgraph`, `PyQt6`, `scikit-learn` (optional für PCA/DTW), `json`, `pathlib`
- Dateiein-Import (NPY/NPZ/CSV), keine Kameraanbindung
- Saubere Modulstruktur mit Type Hints, Docstrings und CLI-Entrypoints

## Datenformate
- NPY-Paare: `*_lam.npy` (Wellenlängen) + `*_spec.npy` (Spektren)
- NPZ (optional) mit Keys `wavelength`, `absorbance_mau` oder `intensity`
- CSV: zwei Spalten `lambda_nm,value`
- Automatische Erkennung von Dateipaaren und Erzeugung eines `meta`-Blocks (Dateiname, Zeit, Matrix)

## Ordnerbasierte Datenbank
```
/Database/
  AnalyteName_A/
    meta.json
    replicates/
      A_2025-09-10_run1.pms.json
      A_2025-09-10_run2.pms.json
```
- `meta.json` enthält: `analyte`, `matrix`, `detector`, `prep`, `notes`, `tolerances` (`d_lambda_nm`, `rt_rel_pct`)
- `pms.json` speichert Fingerprint und Feature-Vektoren eines Replikats

## GUI (PyQt6 + pyqtgraph)
Drei dockbare Panels:
1. **Spectra** – Rohdaten, Preprocessing-Vergleich, Overlay, Cursor-Readout
2. **Pseudo-MS** – Stick-Plot und Tabellenansicht
3. **Match/DB** – Kandidatenliste mit Score, Ampel-Anzeige und Detailvergleich

Toolbar-Funktionen: Ordner laden, Batch-Verarbeitung, Fingerprint-Export, Library-Match, SIM-Kanäle, Preset-Dropdowns.

## Pipeline & Multi-Method-Runner
- Engine testet 50–100 Methoden-Kombinationen aus Baseline-Korrektur, Glättung, Ableitungen, Peak-Picking, Alignment, Feature-Bildung und Matching.
- Qualitätsmetriken: `S_cos`, `S_ratio`, `S_rt`, `S_hash`, `Purity`, `#Sticks`, `SNR`
- Gesamtscore: `S = 0.5*S_cos + 0.2*S_ratio + 0.15*S_rt + 0.1*Purity + 0.05*S_hash`
- Ausgabe der Top-N (z.B. 3) Pipelines in der GUI

## Fingerprint-Schema (`pms.json`)
```json
{
  "unit": "nm",
  "sticks": [
    {"lambda_nm": 372.1, "rel_intensity": 100.0, "width_nm": 9.8, "prominence": 0.031},
    {"lambda_nm": 398.4, "rel_intensity": 46.2, "width_nm": 7.1, "prominence": 0.014}
  ],
  "ratios": [0.462, 0.215],
  "entropy": 0.53,
  "bandpower": [0.31, 0.27, 0.18, 0.12, 0.07, 0.05],
  "dct16": [0.91, -0.12, 0.03, ...],
  "hash": "phash_v1:…",
  "global": {"lambda_mean": 412.6, "skew": -0.21, "kurt": 2.7},
  "quality": {"snr": 12.3, "purity": 0.88, "n_sticks": 8},
  "rt_min": 4.82,
  "meta": {"file": "sample_001_spec.npy", "matrix": "MeOH", "detector": "LG-CCD 360-1050"}
}
```

## Matching & Kandidatenliste
- `library_index.json` enthält Mittelwert/Streuung je Feature
- Scores: `S_cos`, `S_ratio`, `S_rt`, `S_hash` → Gesamt-Score `S`
- Kandidatenliste (Top-10) mit Ampel: grün ≥0.85, gelb 0.70–0.85, sonst rot

## SIM-Kanäle (optional)
- Erzeugt Chromatogramme aus ±2 nm um Library-Top-λ
- Peak-Erkennung per Prominenz-Suche und anschließender Fingerprint/Merging

## CLI-Kommandos
- `fp make <input_folder> --preset NoiseMax --out <out_folder>`
- `lib build <db_folder> --out library_index.json`
- `id match <sample> --db <db_folder> --top 10`
- `sim export <sample_npz> --library library_index.json --out sim.csv`

## Code-Skeleton
Modulübersicht: `io.py`, `preprocess.py`, `sticks.py`, `features.py`, `vectorize.py`, `matching.py`, `library.py`, `sim.py`, `gui.py`.

## Performance
- Vektorisiert (NumPy), Caching von Baselines/Ableitungen
- Deterministische Seeds, Progress-Bar, Abbruchknopf

## Edge-Cases
- Flache Spektren → Warnung „Feature-arm“, `K=3`
- Fehlende Absorption ≥360 nm → Hinweis auf Derivatisierung/Zweitkanal
- Mismatched λ-Gitter → Interpolation auf Integer-Gitter 360–800 nm

Diese Datei dient als Grundlage für weitere Implementierungen und Merge-Aktivitäten.
