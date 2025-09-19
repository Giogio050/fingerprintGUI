from pathlib import Path

import numpy as np

from lg_spectra.io import load_any, load_folder, save_spec


def test_save_and_load_spec_npz(tmp_path: Path) -> None:
    lam = np.linspace(400, 700, 5)
    inten = np.random.rand(5)
    out = tmp_path / "test.npz"
    save_spec(out, lam, intensity=inten, source="unit")
    wl, spec, meta = load_any(out)
    assert np.allclose(wl, lam)
    assert np.allclose(spec, inten)
    assert meta["filename"] == "test.npz"


def test_load_folder_with_various_files(tmp_path: Path) -> None:
    lam = np.linspace(400, 700, 5)
    inten = np.random.rand(5)

    # NPY pair
    np.save(tmp_path / "sample_lam.npy", lam)
    np.save(tmp_path / "sample_spec.npy", inten)

    # CSV with aliases
    csv_path = tmp_path / "sample.csv"
    data = np.column_stack([lam, inten])
    header = "lambda_nm,I"
    np.savetxt(csv_path, data, delimiter=",", header=header, comments="")

    # Embedded spec
    np.save(tmp_path / "embedded_spec.npy", np.vstack([lam, inten]))

    items = load_folder(tmp_path)
    assert len(items) == 3
    bases = {d["meta"]["filename"] for d in items}
    assert {"sample_lam.npy", "sample.csv", "embedded_spec.npy"} <= bases


def test_orphan_spec_raises(tmp_path: Path) -> None:
    inten = np.random.rand(5)
    orphan = tmp_path / "orphan_spec.npy"
    np.save(orphan, inten)
    try:
        load_any(orphan)
    except FileNotFoundError as e:
        assert "Missing companion" in str(e)
    else:  # pragma: no cover - should not happen
        assert False
