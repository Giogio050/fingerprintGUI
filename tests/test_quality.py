import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from lg_spectra.features import compute_features
from lg_spectra.sticks import Stick
import unittest


class QualityWarnTest(unittest.TestCase):
    def test_quality_warnings(self) -> None:
        lam = np.linspace(300, 350, 50)
        y = np.ones_like(lam) * 0.1
        sticks = []
        fp = compute_features(lam, y, sticks)
        warnings = fp.get('warnings', [])
        self.assertTrue(any('n_sticks' in w for w in warnings))
        self.assertTrue(any('snr' in w for w in warnings))
        self.assertTrue(any('360' in w for w in warnings))


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
