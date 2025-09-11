"""PyQt6 GUI application for spectral analysis."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import json
import numpy as np
import pyqtgraph as pg
from PyQt6 import QtWidgets
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction

from .io import load_any
from .preprocess import apply_pipeline
from .sticks import pick_sticks
from .features import compute_features
from .library import build_index
from .matching import score


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle('Little Garden HPLC/DAD')
        self.current_fp: Dict | None = None
        self.library_index: Dict[str, Dict] = {}
        self.current_folder: Optional[Path] = None
        self.loaded_spectra: Dict[str, Tuple[np.ndarray, np.ndarray, Dict]] = {}
        self._init_ui()

    def _init_ui(self) -> None:
        self.spectra_plot = pg.PlotWidget(title='Spectra')
        self.ms_plot = pg.PlotWidget(title='Pseudo-MS')
        self.feature_table = QtWidgets.QTableWidget()
        self.match_table = QtWidgets.QTableWidget()
        self.file_list = QtWidgets.QListWidget()

        dock1 = QtWidgets.QDockWidget('Spectra', self)
        dock1.setWidget(self.spectra_plot)
        dock2 = QtWidgets.QDockWidget('Pseudo-MS', self)
        dock2.setWidget(self.ms_plot)
        dock3 = QtWidgets.QDockWidget('Features', self)
        dock3.setWidget(self.feature_table)
        dock4 = QtWidgets.QDockWidget('Match', self)
        dock4.setWidget(self.match_table)
        dock5 = QtWidgets.QDockWidget('Dateien', self)
        dock5.setWidget(self.file_list)

        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock1)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock2)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock3)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock4)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock5)

        self.file_list.itemSelectionChanged.connect(self._display_selected)

        open_action = QAction('Ordner laden', self)
        open_action.triggered.connect(self.open_folder)
        add_action = QAction('Spektren hinzufügen', self)
        add_action.triggered.connect(self.add_spectra)
        lib_action = QAction('Bibliothek laden', self)
        lib_action.triggered.connect(self.open_library)
        match_action = QAction('Abgleichen', self)
        match_action.triggered.connect(self.match_spectrum)
        export_action = QAction('Fingerprint exportieren', self)
        export_action.triggered.connect(self.export_fingerprint)
        toolbar = self.addToolBar('Main')
        toolbar.addAction(open_action)
        toolbar.addAction(add_action)
        toolbar.addAction(lib_action)
        toolbar.addAction(match_action)
        toolbar.addAction(export_action)

    def open_folder(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, 'Ordner wählen')
        if not folder:
            return
        self.load_folder(Path(folder))

    def open_library(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, 'Bibliothek wählen')
        if not folder:
            return
        self.load_library(Path(folder))

    def load_folder(self, folder: Path) -> None:
        self.current_folder = folder
        files = sorted(folder.glob('*_lam.npy'))
        if not files:
            return
        self.loaded_spectra.clear()
        self.file_list.clear()
        for p in files:
            lam, spec, meta = load_any(p)
            base = p.name.replace('_lam.npy', '')
            self.loaded_spectra[base] = (lam, spec, meta)
            self.file_list.addItem(base)
        if self.file_list.count():
            self.file_list.setCurrentRow(0)

    def add_spectra(self, paths: Optional[List[Path]] = None) -> None:
        if paths is None:
            files, _ = QtWidgets.QFileDialog.getOpenFileNames(
                self,
                'Spektren wählen',
                filter='Spektren (*.npy *.npz *.csv)'
            )
            if not files:
                return
            paths = [Path(f) for f in files]
        for p in paths:
            lam, spec, meta = load_any(p)
            name = p.name
            if name.endswith('_lam.npy'):
                base = name[:-8]
            elif name.endswith('_spec.npy'):
                base = name[:-9]
            else:
                base = p.stem
            self.loaded_spectra[base] = (lam, spec, meta)
            if not self.file_list.findItems(base, Qt.MatchFlag.MatchExactly):
                self.file_list.addItem(base)
        if self.file_list.count() and self.file_list.currentRow() == -1:
            self.file_list.setCurrentRow(0)

    def _display_selected(self) -> None:
        items = self.file_list.selectedItems()
        if not items:
            return
        base = items[0].text()
        lam, spec, _ = self.loaded_spectra[base]
        y = spec[0] if spec.ndim > 1 else spec
        y_hat = apply_pipeline(lam, y, [{'op': 'snv'}, {'op': 'savgol', 'win': 7, 'poly': 2}])
        self.spectra_plot.clear()
        self.spectra_plot.plot(lam, y, pen='r')
        self.spectra_plot.plot(lam, y_hat, pen='g')
        sticks = pick_sticks(lam, y_hat, {'k': 6})
        self.ms_plot.clear()
        for s in sticks:
            self.ms_plot.plot([s.lambda_nm, s.lambda_nm], [0, s.rel_intensity], pen='b')
        fp = compute_features(lam, y_hat, sticks)
        self.current_fp = fp
        self.feature_table.setColumnCount(2)
        self.feature_table.setRowCount(len(fp['bandpower']))
        for i, bp in enumerate(fp['bandpower']):
            self.feature_table.setItem(i, 0, QtWidgets.QTableWidgetItem(f'Band {i}'))
            self.feature_table.setItem(i, 1, QtWidgets.QTableWidgetItem(f'{bp:.3f}'))

    def load_library(self, folder: Path) -> None:
        index_file = folder / 'library_index.json'
        if index_file.exists():
            self.library_index = json.loads(index_file.read_text())
        else:
            self.library_index = build_index(folder)

    def match_spectrum(self) -> None:
        if not self.current_fp or not self.library_index:
            return
        rows = []
        for name, entry in self.library_index.items():
            lib_entry = {
                'sticks': entry.get('sticks', []),
                'ratios': entry.get('ratios_mean', []),
                'bandpower': entry.get('bandpower_mean', []),
                'hash': entry.get('hash')
            }
            sc = score(self.current_fp, lib_entry)
            rows.append((name, sc['S'], sc['S_cos'], sc['S_ratio']))
        rows.sort(key=lambda r: r[1], reverse=True)
        self.match_table.setColumnCount(4)
        self.match_table.setHorizontalHeaderLabels(['Analyte', 'Score', 'Cos', 'Ratio'])
        self.match_table.setRowCount(len(rows))
        for i, (name, s_total, s_cos, s_ratio) in enumerate(rows):
            self.match_table.setItem(i, 0, QtWidgets.QTableWidgetItem(name))
            self.match_table.setItem(i, 1, QtWidgets.QTableWidgetItem(f'{s_total:.3f}'))
            self.match_table.setItem(i, 2, QtWidgets.QTableWidgetItem(f'{s_cos:.3f}'))
            self.match_table.setItem(i, 3, QtWidgets.QTableWidgetItem(f'{s_ratio:.3f}'))

    def export_fingerprint(self, path: Optional[Path] = None) -> None:
        if not self.current_fp:
            return
        if path is None:
            file, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Fingerprint speichern', filter='JSON (*.json)')
            if not file:
                return
            path = Path(file)
        with Path(path).open('w', encoding='utf8') as fh:
            json.dump(self.current_fp, fh, indent=2)


def run() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = MainWindow()
    win.resize(800, 600)
    win.show()
    app.exec()


if __name__ == '__main__':  # pragma: no cover - manual run
    run()
