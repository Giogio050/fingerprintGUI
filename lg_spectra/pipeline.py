"""Multi-method preprocessing and fingerprint generation."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable, Dict, List, Sequence

import numpy as np

from .features import compute_features
from .matching import _cosine as cosine_metric  # reuse internal helper
from .preprocess import PipelineStep, apply_pipeline
from .sticks import Stick, pick_sticks
from .vectorize import sticks_to_vector


@dataclass(frozen=True)
class MethodConfig:
    name: str
    steps: Sequence[PipelineStep]
    peak_params: Dict[str, float]
    alignment: str = "none"


@dataclass
class PipelineResult:
    name: str
    steps: Sequence[PipelineStep]
    wavelength: np.ndarray
    processed: np.ndarray
    sticks: List[Stick]
    diagnostics: Dict[str, float | str]
    features: Dict[str, object]
    scores: Dict[str, float]


@dataclass(frozen=True)
class PresetDefinition:
    base: Sequence[PipelineStep]
    baselines: Sequence[Sequence[PipelineStep]]
    smoothings: Sequence[Sequence[PipelineStep]]
    normalisations: Sequence[Sequence[PipelineStep]]
    derivatives: Sequence[Sequence[PipelineStep]]
    alignments: Sequence[str]
    peak_sets: Sequence[Dict[str, float]]


PIPELINE_PRESETS: Dict[str, PresetDefinition] = {
    "NoiseMax": PresetDefinition(
        base=(PipelineStep("interp_master", {}),),
        baselines=(
            (PipelineStep("rolling_min", {"win": 31}),),
            (PipelineStep("asls", {"lam": 1e5, "p": 0.001}),),
            (PipelineStep("morph_open", {"struct": 11}),),
        ),
        smoothings=(
            (PipelineStep("wavelet_soft", {"level": 3}),),
            (PipelineStep("savgol", {"win": 11, "poly": 3}),),
            (PipelineStep("median", {"size": 7}),),
        ),
        normalisations=(
            (PipelineStep("snv", {}),),
            (PipelineStep("quantile_norm", {"quantile": 0.95}),),
            (PipelineStep("vector_norm", {}),),
        ),
        derivatives=(
            tuple(),
            (PipelineStep("gradient_clamp", {"limit": 0.12}),),
        ),
        alignments=("none", "integer"),
        peak_sets=(
            {"k": 6, "prominence": 0.002, "distance": 6, "strategy": "prominence"},
            {"k": 8, "prominence": 0.0015, "distance": 8, "strategy": "adaptive"},
            {"k": 6, "prominence": 0.001, "distance": 6, "strategy": "derivative"},
        ),
    ),
    "DerivSharp": PresetDefinition(
        base=(PipelineStep("interp_master", {}),),
        baselines=(
            (PipelineStep("asls", {"lam": 1e6, "p": 0.001}),),
            (PipelineStep("morph_close", {"struct": 9}),),
        ),
        smoothings=(
            (PipelineStep("savgol", {"win": 9, "poly": 2}),),
            (PipelineStep("moving_avg", {"n": 5}),),
        ),
        normalisations=(
            (PipelineStep("snv", {}), PipelineStep("quantile_norm", {"quantile": 0.9})),
            (PipelineStep("area_norm", {}),),
        ),
        derivatives=(
            (PipelineStep("derivative", {"order": 1}),),
            (PipelineStep("derivative", {"order": 2}),),
        ),
        alignments=("none", "integer", "dtw"),
        peak_sets=(
            {"k": 6, "prominence": 0.0015, "distance": 5, "strategy": "derivative"},
            {"k": 6, "prominence": 0.002, "distance": 6, "strategy": "prominence"},
        ),
    ),
    "BaselineHard": PresetDefinition(
        base=(PipelineStep("interp_master", {}),),
        baselines=(
            (PipelineStep("asls", {"lam": 1e6, "p": 0.01}), PipelineStep("rolling_min", {"win": 51})),
            (PipelineStep("morph_open", {"struct": 7}), PipelineStep("morph_close", {"struct": 11})),
        ),
        smoothings=(
            (PipelineStep("wavelet_soft", {"level": 2}),),
            (PipelineStep("median", {"size": 5}),),
        ),
        normalisations=(
            (PipelineStep("vector_norm", {}),),
            (PipelineStep("snv", {}),),
        ),
        derivatives=(tuple(),),
        alignments=("none", "integer"),
        peak_sets=(
            {"k": 6, "prominence": 0.0025, "distance": 8, "strategy": "prominence"},
            {"k": 10, "prominence": 0.0018, "distance": 10, "strategy": "adaptive"},
        ),
    ),
}


def _generate_methods(preset: str, max_methods: int) -> List[MethodConfig]:
    definition = PIPELINE_PRESETS[preset]
    combos = product(
        definition.baselines,
        definition.smoothings,
        definition.normalisations,
        definition.derivatives,
        definition.alignments,
        definition.peak_sets,
    )
    methods: List[MethodConfig] = []
    for idx, (baseline, smoothing, norm, deriv, alignment, peak) in enumerate(combos):
        steps: List[PipelineStep] = list(definition.base)
        steps.extend(baseline)
        steps.extend(smoothing)
        steps.extend(norm)
        steps.extend(deriv)
        name = f"{preset}-{idx:03d}"
        methods.append(MethodConfig(name, tuple(steps), dict(peak), alignment))
        if len(methods) >= max_methods:
            break
    return methods


def _ratio_consistency(sample: Sequence[float], reference: Sequence[float]) -> float:
    if not sample or not reference:
        return 0.0
    n = min(len(sample), len(reference))
    sample = np.array(sample[:n], dtype=float)
    reference = np.array(reference[:n], dtype=float)
    diff = np.abs(sample - reference)
    return float(np.clip(1 - diff / (0.15 + 1e-9), 0, 1).mean())


def _align_signal(y: np.ndarray, reference: np.ndarray, mode: str) -> np.ndarray:
    if mode == "integer":
        max_shift = 3
        best = y
        best_score = -np.inf
        for shift in range(-max_shift, max_shift + 1):
            if shift == 0:
                candidate = y
            elif shift > 0:
                candidate = np.concatenate([np.zeros(shift), y[:-shift]])
            else:
                candidate = np.concatenate([y[-shift:], np.zeros(-shift)])
            score = float(np.dot(candidate, reference))
            if score > best_score:
                best_score = score
                best = candidate
        return best
    if mode == "dtw":
        return _dtw_warp(y, reference)
    return y


def _dtw_warp(y: np.ndarray, reference: np.ndarray, window: int = 6) -> np.ndarray:
    n = len(y)
    m = len(reference)
    window = max(window, abs(n - m))
    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        j_start = max(1, i - window)
        j_end = min(m, i + window)
        for j in range(j_start, j_end + 1):
            dist = abs(y[i - 1] - reference[j - 1])
            cost[i, j] = dist + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    i, j = n, m
    path: List[tuple[int, int]] = []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        steps = (
            (cost[i - 1, j], i - 1, j),
            (cost[i, j - 1], i, j - 1),
            (cost[i - 1, j - 1], i - 1, j - 1),
        )
        _, i, j = min(steps, key=lambda x: x[0])
    path.append((0, 0))
    path = sorted(set(path), key=lambda p: (p[1], p[0]))
    mapped_j = np.array([p[1] for p in path], dtype=float)
    mapped_i = np.array([p[0] for p in path], dtype=float)
    unique_j, indices = np.unique(mapped_j, return_index=True)
    samples = y[mapped_i[indices].astype(int)]
    return np.interp(np.arange(m), unique_j, samples, left=y[0], right=y[-1])


def run_pipelines(
    lam: np.ndarray,
    spectrum: np.ndarray,
    *,
    preset: str = "NoiseMax",
    max_methods: int = 90,
    top_n: int = 3,
    progress: Callable[[int, int], None] | None = None,
) -> List[PipelineResult]:
    """Run a preset ensemble of pipelines and return the best results."""

    lam = np.asarray(lam, dtype=float)
    spectrum = np.asarray(spectrum, dtype=float)
    if spectrum.ndim == 2:
        base = np.median(spectrum, axis=0)
    else:
        base = spectrum

    methods = _generate_methods(preset, max_methods)
    if not methods:
        return []

    ref_method = methods[0]
    ref_lam, ref_processed = apply_pipeline(
        lam, base, ref_method.steps, return_wavelength=True
    )
    ref_sticks_sel = pick_sticks(ref_lam, ref_processed, {"k": 6}, return_info=True)
    ref_features = compute_features(ref_lam, ref_processed, ref_sticks_sel.sticks)
    ref_vector = np.array(ref_features.get("stick_vector", sticks_to_vector(ref_sticks_sel.sticks)))
    ref_ratios = ref_features.get("ratios", [])
    ref_dct = np.array(ref_features.get("dct16", []))

    results: List[PipelineResult] = []
    total_methods = len(methods)
    for idx, method in enumerate(methods, 1):
        lam_proc, y_proc = apply_pipeline(lam, base, method.steps, return_wavelength=True)
        y_aligned = _align_signal(y_proc, ref_processed, method.alignment)
        if method.alignment == "dtw":
            lam_proc = ref_lam
        selection = pick_sticks(lam_proc, y_aligned, method.peak_params, return_info=True)
        sticks = selection.sticks
        if not sticks:
            if progress:
                progress(idx, total_methods)
            continue
        features = compute_features(lam_proc, y_aligned, sticks)
        vec = np.array(features.get("stick_vector", sticks_to_vector(sticks)))
        ratios = features.get("ratios", [])
        dct16 = np.array(features.get("dct16", []))
        purity = float(features.get("quality", {}).get("purity", 0.0))
        scores = {
            "S_cos": cosine_metric(vec, ref_vector),
            "S_ratio": _ratio_consistency(ratios, ref_ratios),
            "S_rt": 1.0,
            "Purity": purity,
            "S_hash": cosine_metric(dct16[:16], ref_dct[:16]) if dct16.size and ref_dct.size else 0.0,
        }
        scores["S"] = (
            0.5 * scores["S_cos"]
            + 0.2 * scores["S_ratio"]
            + 0.15 * scores["S_rt"]
            + 0.1 * scores["Purity"]
            + 0.05 * scores["S_hash"]
        )
        diagnostics = dict(selection.diagnostics)
        diagnostics["alignment"] = method.alignment
        diagnostics["preset"] = preset
        results.append(
            PipelineResult(
                name=method.name,
                steps=method.steps,
                wavelength=lam_proc,
                processed=y_aligned,
                sticks=sticks,
                diagnostics=diagnostics,
                features=features,
                scores=scores,
            )
        )
        if progress:
            progress(idx, total_methods)

    results.sort(key=lambda r: r.scores.get("S", 0.0), reverse=True)
    return results[:top_n]
