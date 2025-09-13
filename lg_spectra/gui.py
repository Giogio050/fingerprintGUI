"""PyQt6 GUI application for spectral analysis."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import json
import numpy as np
import pyqtgraph as pg
from qt_material import apply_stylesheet
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QColor, QCloseEvent

from .io import load_any
from .preprocess import apply_pipeline, sanitize_spectrum
from .sticks import pick_sticks
from .features import compute_features
from .library import build_index
from .matching import score
from .sim import sim_traces


class MatchWorker(QtCore.QObject):
    progress = QtCore.pyqtSignal(int)
    finished = QtCore.pyqtSignal(list)
    message = QtCore.pyqtSignal(str)

    def __init__(self, fp: Dict, library: Dict[str, Dict]) -> None:
        super().__init__()
        self.fp = fp
        self.library = library

    def run(self) -> None:
        rows = []
        total = len(self.library) or 1
        for i, (name, entry) in enumerate(self.library.items()):
            lib_entry = {
                'sticks': entry.get('sticks', []),
                'ratios': entry.get('ratios_mean', []),
                'bandpower': entry.get('bandpower_mean', []),
                'hash': entry.get('hash'),
            }
            sc = score(self.fp, lib_entry)
            rows.append((name, sc['S'], sc['S_cos'], sc['S_ratio']))
            self.progress.emit(int((i + 1) / total * 100))
        self.finished.emit(rows)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle('Little Garden HPLC/DAD')
        self.current_fp: Dict | None = None
        self.library_index: Dict[str, Dict] = {}
        self.current_folder: Optional[Path] = None
        self.loaded_spectra: Dict[str, Tuple[np.ndarray, np.ndarray, Dict]] = {}
        self.settings = QtCore.QSettings('lg_spectra', 'app')
        self.dark = self.settings.value('dark', True, bool)
        self.accent = self.settings.value('accent', 'teal')
        self._apply_theme()
        geom = self.settings.value('geometry')
        if geom:
            self.restoreGeometry(geom)
        self._init_ui()
        last = self.settings.value('last_folder')
        if last:
            self.load_folder(Path(str(last)))

    def _init_ui(self) -> None:
        self.spectra_plot = pg.PlotWidget(title='Spectra')
        self.ms_plot = pg.PlotWidget(title='Pseudo-MS')
        self.feature_table = QtWidgets.QTableWidget()
        self.match_table = QtWidgets.QTableWidget()
        self.file_list = QtWidgets.QListWidget()
        self.progress = QtWidgets.QProgressBar()
        self.log_console = QtWidgets.QTextEdit()
        self.log_console.setReadOnly(True)

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
        dock6 = QtWidgets.QDockWidget('Log', self)
        dock6.setWidget(self.log_console)

        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock1)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock2)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock3)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock4)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock5)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock6)

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
        sim_action = QAction('Export SIM', self)
        sim_action.triggered.connect(self.export_sim)
        toolbar = self.addToolBar('Main')
        toolbar.addAction(open_action)
        toolbar.addAction(add_action)
        toolbar.addAction(lib_action)
        toolbar.addAction(match_action)
        toolbar.addAction(export_action)
        toolbar.addAction(sim_action)
        self.k_spin = QtWidgets.QSpinBox()
        self.k_spin.setRange(1, 12)
        self.k_spin.setValue(6)
        toolbar.addWidget(self.k_spin)
        self.pipeline_combo = QtWidgets.QComboBox()
        self.pipeline_combo.addItems(['snv+savgol', 'snv'])
        toolbar.addWidget(self.pipeline_combo)
        self.statusBar().addPermanentWidget(self.progress)

        view_menu = self.menuBar().addMenu('Darstellung')
        theme_action = QAction('Theme wechseln', self)
        theme_action.triggered.connect(self.toggle_theme)
        view_menu.addAction(theme_action)
        accent_menu = view_menu.addMenu('Akzentfarbe')
        for col in ['teal', 'blue', 'red', 'orange']:
            act = QAction(col, self)
            act.triggered.connect(lambda _, c=col: self.set_accent(c))
            accent_menu.addAction(act)

    def open_folder(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, 'Ordner wählen')
        if not folder:
            return
        self.settings.setValue('last_folder', folder)
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
        self.statusBar().showMessage(f'{len(files)} Dateien geladen', 5000)
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
        self.statusBar().showMessage(f'{len(paths)} Dateien hinzugefügt', 5000)
        if self.file_list.count() and self.file_list.currentRow() == -1:
            self.file_list.setCurrentRow(0)

    def _display_selected(self) -> None:
        items = self.file_list.selectedItems()
        if not items:
            return
        base = items[0].text()
        lam, spec, _ = self.loaded_spectra[base]
        y = spec[0] if spec.ndim > 1 else spec
        lam, y, _ = sanitize_spectrum(lam, y)
        cfg = [{'op': 'snv'}]
        if self.pipeline_combo.currentText() == 'snv+savgol':
            cfg.append({'op': 'savgol', 'win': 7, 'poly': 2})
        y_hat = apply_pipeline(lam, y, cfg)
        self.spectra_plot.clear()
        self.spectra_plot.plot(lam, y, pen='r')
        self.spectra_plot.plot(lam, y_hat, pen='g')
        sticks = pick_sticks(lam, y_hat, {'k': self.k_spin.value()})
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
        self.progress.setValue(0)
        self.thread = QtCore.QThread(self)
        self.worker = MatchWorker(self.current_fp, self.library_index)
        self.worker.moveToThread(self.thread)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.finished.connect(self._match_finished)
        self.thread.started.connect(self.worker.run)
        self.thread.start()

    def _match_finished(self, rows: List[Tuple[str, float, float, float]]) -> None:
        self.thread.quit()
        self.thread.wait()
        self.progress.setValue(100)
        rows.sort(key=lambda r: r[1], reverse=True)
        self.match_table.setColumnCount(4)
        self.match_table.setHorizontalHeaderLabels(['Analyte', 'Score', 'Cos', 'Ratio'])
        self.match_table.setRowCount(len(rows))
        for i, (name, s_total, s_cos, s_ratio) in enumerate(rows):
            self.match_table.setItem(i, 0, QtWidgets.QTableWidgetItem(name))
            score_item = QtWidgets.QTableWidgetItem(f'{s_total:.3f}')
            color = QColor('red')
            if s_total >= 0.8:
                color = QColor('green')
            elif s_total >= 0.5:
                color = QColor('yellow')
            score_item.setBackground(color)
            self.match_table.setItem(i, 1, score_item)
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

    def export_sim(self) -> None:
        items = self.file_list.selectedItems()
        if not items:
            return
        base = items[0].text()
        lam, spec, _ = self.loaded_spectra[base]
        if spec.ndim != 2:
            return
        file, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'SIM export', filter='CSV (*.csv)')
        if not file:
            return
        channels, ok = QtWidgets.QInputDialog.getText(self, 'Kanäle', 'Wellenlängen (comma separated)')
        if not ok or not channels:
            return
        chans = [float(c.strip()) for c in channels.split(',') if c.strip()]
        traces = sim_traces(lam, spec, chans)
        if not traces:
            return
        arr = np.column_stack([traces[k] for k in traces])
        header = ','.join(traces.keys())
        np.savetxt(file, arr, delimiter=',', header=header, comments='')

    def _apply_theme(self) -> None:
        self.current_theme = f"{'dark' if self.dark else 'light'}_{self.accent}.xml"
        apply_stylesheet(QtWidgets.QApplication.instance(), theme=self.current_theme)

    def toggle_theme(self) -> None:
        self.dark = not self.dark
        self._apply_theme()

    def set_accent(self, accent: str) -> None:
        self.accent = accent
        self._apply_theme()

    def closeEvent(self, event: QCloseEvent) -> None:
        self.settings.setValue('geometry', self.saveGeometry())
        self.settings.setValue('dark', self.dark)
        self.settings.setValue('accent', self.accent)
        if self.current_folder:
            self.settings.setValue('last_folder', str(self.current_folder))
        super().closeEvent(event)


def run() -> None:
    QtWidgets.QApplication.setAttribute(QtCore.Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    apply_stylesheet(app, theme='dark_teal.xml')
    win = MainWindow()
    win.resize(800, 600)
    win.show()
    app.exec()


if __name__ == '__main__':  # pragma: no cover - manual run
    run()
