from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QFileDialog,
    QGridLayout,
    QHeaderView,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from xrdviz.calibration import auto_calibrate_phases
from xrdviz.io import load_reference_peaks_csv_many, load_rigaku_peaks_csv
from xrdviz.publication import make_peak_table_rows


class ReferencePeaksMixin:
    """Reference-phase import, calibration, table, and export-facing UI."""

    def _reference_peaks_tab(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        controls = QGridLayout()
        controls.setHorizontalSpacing(4)
        controls.setVerticalSpacing(4)
        ref_button = QPushButton("Reference CSV")
        ref_button.clicked.connect(self.import_reference_peaks_dialog)
        rigaku_button = QPushButton("Rigaku Peaks")
        rigaku_button.clicked.connect(self.import_rigaku_peaks_dialog)
        cif_button = QPushButton("CIF")
        cif_button.clicked.connect(self.open_cif)
        self.auto_fit_button = QPushButton("Auto fit phase peaks")
        self.auto_fit_button.clicked.connect(self.auto_fit_phase_peaks)
        export_button = QPushButton("Export peak table")
        export_button.clicked.connect(self.export_peak_table_dialog)
        for index, button in enumerate(
            (ref_button, rigaku_button, cif_button, self.auto_fit_button, export_button)
        ):
            controls.addWidget(button, index // 3, index % 3)
        layout.addLayout(controls)

        self.peak_table = QTableWidget(0, 8)
        self.peak_table.setHorizontalHeaderLabels(
            ["Phase", "2theta", "d", "Q", "Intensity", "hkl", "Label", "Source"]
        )
        self.peak_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents
        )
        self.peak_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        layout.addWidget(self.peak_table, stretch=1)
        return panel

    def import_reference_peaks_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open reference peaks CSV", "", "CSV files (*.csv);;All files (*.*)"
        )
        if path:
            self.import_reference_peaks(Path(path))

    def import_reference_peaks(self, path: str | Path) -> None:
        try:
            text = Path(path).read_text(encoding="utf-8-sig")
            self.state.phases.extend(
                load_reference_peaks_csv_many(text, source_path=str(path))
            )
        except Exception as exc:  # noqa: BLE001 - surfaced to user
            QMessageBox.warning(self, "Reference peaks import failed", str(exc))
            return
        self.refresh_layers()
        self.refresh_peak_table()
        self.render()

    def import_rigaku_peaks_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Rigaku peaks CSV", "", "CSV files (*.csv);;All files (*.*)"
        )
        if path:
            self.import_rigaku_peaks(Path(path))

    def import_rigaku_peaks(self, path: str | Path) -> None:
        try:
            self.state.phases.append(
                load_rigaku_peaks_csv(
                    Path(path).read_text(encoding="utf-8-sig"), source_path=str(path)
                )
            )
        except Exception as exc:  # noqa: BLE001 - surfaced to user
            QMessageBox.warning(self, "Rigaku peaks import failed", str(exc))
            return
        self.refresh_layers()
        self.refresh_peak_table()
        self.render()

    def auto_fit_phase_peaks(self) -> None:
        try:
            self.state.settings = self._settings_from_controls()
            auto_calibrate_phases(self.state)
        except Exception as exc:  # noqa: BLE001 - surfaced to user
            QMessageBox.warning(self, "Auto fit failed", str(exc))
            return
        self.refresh_layers()
        self.refresh_peak_table()
        self.render()

    def refresh_peak_table(self) -> None:
        try:
            self.state.settings = self._settings_from_controls()
        except Exception:
            return
        rows = make_peak_table_rows(self.state)
        self.peak_table.setRowCount(0)
        for data in rows:
            row = self.peak_table.rowCount()
            self.peak_table.insertRow(row)
            values = [
                data["phase"],
                f"{float(data['two_theta']):.4g}",
                f"{float(data['d']):.4g}",
                f"{float(data['q']):.4g}",
                f"{float(data['intensity']):.4g}",
                data["hkl"],
                data["label"],
                data["source"],
            ]
            for column, value in enumerate(values):
                self.peak_table.setItem(row, column, QTableWidgetItem(str(value)))
