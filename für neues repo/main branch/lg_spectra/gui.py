"""PyQt6 GUI application for spectral analysis."""
from __future__ import annotations

from pathlib import Path
from typing import List

from PyQt6 import QtWidgets
import pyqtgraph as pg
import numpy as np

from .io import load_any
from .preprocess import apply_pipeline
from .sticks import pick_sticks
from .features import compute_features


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle('Little Garden HPLC/DAD')
        self._init_ui()

    def _init_ui(self) -> None:
        self.spectra_plot = pg.PlotWidget(title='Spectra')
        self.ms_plot = pg.PlotWidget(title='Pseudo-MS')
        self.match_table = QtWidgets.QTableWidget()

        dock1 = QtWidgets.QDockWidget('Spectra', self)
        dock1.setWidget(self.spectra_plot)
        dock2 = QtWidgets.QDockWidget('Pseudo-MS', self)
        dock2.setWidget(self.ms_plot)
        dock3 = QtWidgets.QDockWidget('Match', self)
        dock3.setWidget(self.match_table)

        self.addDockWidget(QtWidgets.Qt.DockWidgetArea.LeftDockWidgetArea, dock1)
        self.addDockWidget(QtWidgets.Qt.DockWidgetArea.RightDockWidgetArea, dock2)
        self.addDockWidget(QtWidgets.Qt.DockWidgetArea.BottomDockWidgetArea, dock3)

        open_action = QtWidgets.QAction('Ordner laden', self)
        open_action.triggered.connect(self.open_folder)
        toolbar = self.addToolBar('Main')
        toolbar.addAction(open_action)

    def open_folder(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, 'Ordner wählen')
        if not folder:
            return
        self.load_folder(Path(folder))

    def load_folder(self, folder: Path) -> None:
        files = list(folder.glob('*_lam.npy'))
        if not files:
            return
        lam, spec, meta = load_any(files[0])
        y = spec[0] if spec.ndim > 1 else spec
        y_hat = apply_pipeline(lam, y, [{'op': 'snv'}, {'op': 'savgol', 'win': 7, 'poly': 2}])
        self.spectra_plot.clear()
        self.spectra_plot.plot(lam, y, pen='r')
        self.spectra_plot.plot(lam, y_hat, pen='g')
        sticks = pick_sticks(lam, y_hat, {'k': 6})
        self.ms_plot.clear()
        for s in sticks:
            self.ms_plot.plot([s.lambda_nm, s.lambda_nm], [0, s.rel_intensity], pen='b')
        fp = compute_features(lam, y_hat, sticks, meta=meta)
        rows: List[tuple[str, str]] = []
        rows.append(('File', meta.get('file', '')))
        rows.append(('Entropy', f'{fp["entropy"]:.3f}'))
        rows.append(('λ̄', f'{fp["global"]["lambda_mean"]:.2f}'))
        rows.append(('Skew', f'{fp["global"]["skew"]:.2f}'))
        rows.append(('Kurt', f'{fp["global"]["kurt"]:.2f}'))
        rows.append(('SNR', f'{fp["quality"]["snr"]:.2f}'))
        rows.append(('Purity', f'{fp["quality"]["purity"]:.2f}'))
        rows.append(('#Sticks', str(fp["quality"]["n_sticks"])) )
        for idx, r in enumerate(fp['ratios'][:3]):
            rows.append((f'Ratio {idx+2}/1', f'{r:.3f}'))
        for i, bp in enumerate(fp['bandpower']):
            rows.append((f'Band {i}', f'{bp:.3f}'))
        self.match_table.setColumnCount(2)
        self.match_table.setRowCount(len(rows))
        for i, (name, val) in enumerate(rows):
            self.match_table.setItem(i, 0, QtWidgets.QTableWidgetItem(name))
            self.match_table.setItem(i, 1, QtWidgets.QTableWidgetItem(val))


def run() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = MainWindow()
    win.resize(800, 600)
    win.show()
    app.exec()


if __name__ == '__main__':  # pragma: no cover - manual run
    run()
