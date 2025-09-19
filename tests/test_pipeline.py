import numpy as np

from lg_spectra.features import compute_features
from lg_spectra.pipeline import run_pipelines
from lg_spectra.sticks import pick_sticks


def _synthetic_spectrum() -> tuple[np.ndarray, np.ndarray]:
    lam = np.linspace(360, 800, 200)
    peak1 = np.exp(-0.5 * ((lam - 420) / 8) ** 2)
    peak2 = 0.6 * np.exp(-0.5 * ((lam - 510) / 10) ** 2)
    peak3 = 0.4 * np.exp(-0.5 * ((lam - 620) / 12) ** 2)
    spectrum = peak1 + peak2 + peak3 + 0.02 * np.random.RandomState(0).normal(size=lam.size)
    return lam, spectrum


def test_compute_features_produces_expected_keys() -> None:
    lam, spectrum = _synthetic_spectrum()
    sticks = pick_sticks(lam, spectrum, {"k": 5, "prominence": 0.05})
    feats = compute_features(lam, spectrum, sticks)
    assert "bandpower" in feats
    assert len(feats["bandpower"]) == 6
    assert "dct16" in feats and len(feats["dct16"]) == 16
    assert feats["quality"]["n_sticks"] <= 5


def test_run_pipelines_returns_ordered_results() -> None:
    lam, spectrum = _synthetic_spectrum()
    results = run_pipelines(lam, spectrum, preset="NoiseMax", max_methods=6, top_n=2)
    assert results
    assert results[0].scores["S"] >= results[-1].scores["S"]
    assert results[0].features["sticks"]
