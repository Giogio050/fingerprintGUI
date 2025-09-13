import json

import numpy as np

from lg_spectra.features import compute_features
from lg_spectra.sticks import pick_sticks


def test_compute_features_json_serializable():
    lam = np.linspace(360, 400, 100)
    y = np.sin(np.linspace(0, 6.28, 100)) + 1
    sticks = pick_sticks(lam, y)
    fp1 = compute_features(lam, y, sticks, rt=1.23)
    fp2 = compute_features(lam, y, sticks, rt=1.23)
    assert fp1['phash'] == fp2['phash']
    json.dumps(fp1)  # should not raise

