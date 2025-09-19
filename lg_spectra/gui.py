"""PyQt6 GUI application for spectral analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtGui, QtWidgets
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction

from .io import load_any, load_folder
from .library import build_index
from .matching import score
from .pipeline import PIPELINE_PRESETS, PipelineResult, run_pipelines
from .sim import sim_traces


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Little Garden HPLC/DAD")
        self.current_results: List[PipelineResult] = []
        self.current_folder: Optional[Path] = None
        self.library_index: Dict[str, Dict] = {}
        self.loaded_spectra: Dict[str, Tuple[np.ndarray, np.ndarray, Dict]] = {}
        self._init_ui()

    def _init_ui(self) -> None:
        self.spectra_plot = pg.PlotWidget(title="Spectra")
        self.spectra_plot.showGrid(x=True, y=True, alpha=0.2)
        self.ms_plot = pg.PlotWidget(title="Pseudo-MS")
        self.pipeline_table = QtWidgets.QTableWidget()
        self.pipeline_table.setColumnCount(5)
        self.pipeline_table.setHorizontalHeaderLabels(["Pipeline", "Score", "SNR", "Purity", "Warn"])
        self.match_table = QtWidgets.QTableWidget()
        self.match_table.setColumnCount(4)
        self.match_table.setHorizontalHeaderLabels(["Analyte", "Score", "Cos", "Ratio"])
        self.file_list = QtWidgets.QListWidget()
        self.feature_table = QtWidgets.QTableWidget()
        self.feature_table.setColumnCount(2)
        self.feature_table.setHorizontalHeaderLabels(["Feature", "Value"])

        dock_spectra = QtWidgets.QDockWidget("Spectra", self)
        dock_spectra.setWidget(self.spectra_plot)
        dock_ms = QtWidgets.QDockWidget("Pseudo-MS", self)
        dock_ms.setWidget(self.ms_plot)
        dock_pipeline = QtWidgets.QDockWidget("Pipelines", self)
        dock_pipeline.setWidget(self.pipeline_table)
        dock_match = QtWidgets.QDockWidget("Match", self)
        dock_match.setWidget(self.match_table)
        dock_features = QtWidgets.QDockWidget("Features", self)
        dock_features.setWidget(self.feature_table)
        dock_files = QtWidgets.QDockWidget("Dateien", self)
        dock_files.setWidget(self.file_list)

        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock_files)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock_spectra)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock_ms)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock_pipeline)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock_match)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock_features)

        self.file_list.itemSelectionChanged.connect(self._display_selected)
        self.pipeline_table.itemSelectionChanged.connect(self._display_pipeline)

        open_action = QAction("Ordner laden", self)
        open_action.triggered.connect(self.open_folder)
        add_action = QAction("Dateien hinzufügen", self)
        add_action.triggered.connect(self.add_spectra)
        batch_action = QAction("Batch verarbeiten", self)
        batch_action.triggered.connect(self.batch_process)
        export_action = QAction("Fingerprint exportieren", self)
        export_action.triggered.connect(self.export_fingerprint)
        library_action = QAction("Bibliothek laden", self)
        library_action.triggered.connect(self.open_library)
        match_action = QAction("Library matchen", self)
        match_action.triggered.connect(self.match_spectrum)
        sim_action = QAction("SIM", self)
        sim_action.triggered.connect(self.show_sim_dialog)

        toolbar = self.addToolBar("Main")
        toolbar.addAction(open_action)
        toolbar.addAction(add_action)
        toolbar.addAction(batch_action)
        toolbar.addAction(export_action)
        toolbar.addAction(library_action)
        toolbar.addAction(match_action)
        toolbar.addAction(sim_action)

        toolbar.addSeparator()
        self.preset_combo = QtWidgets.QComboBox()
        for name in PIPELINE_PRESETS:
            self.preset_combo.addItem(name)
        self.preset_combo.currentTextChanged.connect(self._preset_changed)
        toolbar.addWidget(QtWidgets.QLabel("Preset:"))
        toolbar.addWidget(self.preset_combo)

        self.statusBar().showMessage("Bereit")

    # --- data loading -------------------------------------------------
    def open_folder(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Ordner wählen")
        if not folder:
            return
        self.load_folder(Path(folder))

    def load_folder(self, folder: Path) -> None:
        self.current_folder = folder
        self.loaded_spectra.clear()
        self.file_list.clear()
        for record in load_folder(folder):
            lam = record["wavelength"]
            spec = record["spectrum"]
            meta = record.get("meta", {})
            base = Path(meta.get("filename", "sample")).stem
            self.loaded_spectra[base] = (lam, spec, meta)
            self.file_list.addItem(base)
        if self.file_list.count():
            self.file_list.setCurrentRow(0)

    def add_spectra(self) -> None:
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Spektren wählen", filter="Spektren (*.npy *.npz *.csv)"
        )
        if not files:
            return
        for f in files:
            lam, spec, meta = load_any(f)
            base = Path(f).stem
            meta.setdefault("filename", Path(f).name)
            self.loaded_spectra[base] = (lam, spec, meta)
            if not self.file_list.findItems(base, Qt.MatchFlag.MatchExactly):
                self.file_list.addItem(base)
        if self.file_list.count() and self.file_list.currentRow() == -1:
            self.file_list.setCurrentRow(0)

    # --- pipeline execution ------------------------------------------
    def _preset_changed(self, name: str) -> None:
        if self.file_list.currentItem():
            self._display_selected()

    def _display_selected(self) -> None:
        items = self.file_list.selectedItems()
        if not items:
            return
        base = items[0].text()
        lam, spec, meta = self.loaded_spectra[base]
        preset = self.preset_combo.currentText()
        self.statusBar().showMessage(f"Verarbeite {base} mit {preset}…")
        QtWidgets.QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self.current_results = run_pipelines(lam, spec, preset=preset, max_methods=60, top_n=3)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self.statusBar().showMessage(f"{len(self.current_results)} Pipelines berechnet")
        self._populate_pipeline_table()
        if self.current_results:
            self.pipeline_table.selectRow(0)
            self._render_pipeline(self.current_results[0], lam, spec)

    def _populate_pipeline_table(self) -> None:
        self.pipeline_table.setRowCount(len(self.current_results))
        for row, result in enumerate(self.current_results):
            scores = result.scores
            diag = result.diagnostics
            self.pipeline_table.setItem(row, 0, QtWidgets.QTableWidgetItem(result.name))
            self.pipeline_table.setItem(row, 1, QtWidgets.QTableWidgetItem(f"{scores['S']:.3f}"))
            self.pipeline_table.setItem(
                row, 2, QtWidgets.QTableWidgetItem(f"{diag.get('snr', 0.0):.2f}")
            )
            self.pipeline_table.setItem(
                row, 3, QtWidgets.QTableWidgetItem(f"{scores['Purity']:.3f}")
            )
            self.pipeline_table.setItem(
                row,
                4,
                QtWidgets.QTableWidgetItem(diag.get("warning", "")),
            )
        self.pipeline_table.resizeColumnsToContents()

    def _display_pipeline(self) -> None:
        items = self.pipeline_table.selectedItems()
        if not items:
            return
        row = items[0].row()
        if row < len(self.current_results):
            base = self.file_list.currentItem().text()
            lam, spec, _ = self.loaded_spectra[base]
            self._render_pipeline(self.current_results[row], lam, spec)

    def _render_pipeline(self, result: PipelineResult, lam: np.ndarray, spec: np.ndarray) -> None:
        raw = np.median(spec, axis=0) if spec.ndim == 2 else spec
        self.spectra_plot.clear()
        self.spectra_plot.plot(lam, raw, pen=pg.mkPen("r", width=1), name="Raw")
        self.spectra_plot.plot(result.wavelength, result.processed, pen=pg.mkPen("g", width=2), name="Processed")
        self.ms_plot.clear()
        for stick in result.sticks:
            self.ms_plot.plot(
                [stick.lambda_nm, stick.lambda_nm],
                [0, stick.rel_intensity],
                pen=pg.mkPen("b", width=2),
            )
        self.ms_plot.setLabel("bottom", "Wavelength", units="nm")
        self.ms_plot.setLabel("left", "rel. Intensity", units="%")
        self._populate_feature_table(result.features)

    def _populate_feature_table(self, features: Dict[str, object]) -> None:
        rows = [
            ("Entropy", features.get("entropy")),
            ("Top λ", features.get("top_lambda")),
            ("Purity", features.get("quality", {}).get("purity")),
            ("SNR", features.get("quality", {}).get("snr")),
        ]
        self.feature_table.setRowCount(len(rows))
        for idx, (name, value) in enumerate(rows):
            self.feature_table.setItem(idx, 0, QtWidgets.QTableWidgetItem(str(name)))
            self.feature_table.setItem(idx, 1, QtWidgets.QTableWidgetItem(f"{value}"))
        self.feature_table.resizeColumnsToContents()

    # --- actions ------------------------------------------------------
    def batch_process(self) -> None:
        if not self.loaded_spectra:
            return
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Ausgabeordner wählen")
        if not folder:
            return
        out_dir = Path(folder)
        preset = self.preset_combo.currentText()
        for base, (lam, spec, meta) in self.loaded_spectra.items():
            results = run_pipelines(lam, spec, preset=preset, max_methods=60, top_n=1)
            if not results:
                continue
            fp = dict(results[0].features)
            fp["meta"] = meta | {"pipeline": results[0].name}
            fp["diagnostics"] = results[0].diagnostics
            path = out_dir / f"{base}.pms.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(fp, indent=2))
        self.statusBar().showMessage("Batch abgeschlossen")

    def export_fingerprint(self) -> None:
        if not self.current_results:
            return
        file, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Fingerprint speichern", filter="JSON (*.json)"
        )
        if not file:
            return
        fp = dict(self.current_results[0].features)
        base = self.file_list.currentItem().text()
        _, _, meta = self.loaded_spectra[base]
        fp["meta"] = meta | {"pipeline": self.current_results[0].name}
        fp["diagnostics"] = self.current_results[0].diagnostics
        Path(file).write_text(json.dumps(fp, indent=2))
        self.statusBar().showMessage(f"Fingerprint gespeichert: {file}")

    def open_library(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Bibliothek wählen")
        if not folder:
            return
        folder_path = Path(folder)
        index_file = folder_path / "library_index.json"
        if index_file.exists():
            self.library_index = json.loads(index_file.read_text())
        else:
            self.library_index = build_index(folder_path)
        self.statusBar().showMessage(f"Bibliothek geladen ({len(self.library_index)} Analyte)")

    def match_spectrum(self) -> None:
        if not self.current_results or not self.library_index:
            return
        fp = dict(self.current_results[0].features)
        fp["meta"] = fp.get("meta", {}) | {"pipeline": self.current_results[0].name}
        rows = []
        for name, entry in self.library_index.items():
            sc = score(fp, entry)
            rows.append((name, sc))
        rows.sort(key=lambda x: x[1]["S"], reverse=True)
        self.match_table.setRowCount(min(len(rows), 10))
        for row, (name, sc) in enumerate(rows[:10]):
            self.match_table.setItem(row, 0, QtWidgets.QTableWidgetItem(name))
            self.match_table.setItem(row, 1, QtWidgets.QTableWidgetItem(f"{sc['S']:.3f}"))
            self.match_table.setItem(row, 2, QtWidgets.QTableWidgetItem(f"{sc['S_cos']:.3f}"))
            self.match_table.setItem(row, 3, QtWidgets.QTableWidgetItem(f"{sc['S_ratio']:.3f}"))
            color = QtGui.QColor("red")
            if sc["S"] >= 0.85:
                color = QtGui.QColor("green")
            elif sc["S"] >= 0.7:
                color = QtGui.QColor("yellow")
            for col in range(4):
                self.match_table.item(row, col).setBackground(color)
        self.match_table.resizeColumnsToContents()

    def show_sim_dialog(self) -> None:
        if not self.current_results:
            return
        fp = self.current_results[0].features
        sticks = fp.get("sticks", [])
        base = self.file_list.currentItem().text()
        lam, spec, _ = self.loaded_spectra[base]
        if spec.ndim != 2:
            QtWidgets.QMessageBox.information(self, "SIM", "Keine Zeitscheiben vorhanden")
            return
        lambdas = [s["lambda_nm"] for s in sticks[:5]]
        traces = sim_traces(lam, spec, lambdas, band=2)
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("SIM-Traces")
        layout = QtWidgets.QVBoxLayout(dialog)
        text = QtWidgets.QPlainTextEdit(dialog)
        text.setPlainText(json.dumps(traces, indent=2))
        layout.addWidget(text)
        dialog.resize(400, 300)
        dialog.exec()


def run() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = MainWindow()
    win.resize(1200, 800)
    win.show()
    app.exec()
