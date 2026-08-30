from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QFileDialog, QMessageBox

from xrdviz.plot.renderer import export_project
from xrdviz.plot.style import apply_publication_preset
from xrdviz.publication import export_publication_bundle
from xrdviz.project import load_project, save_project


class ProjectIoMixin:
    """Project, figure, and publication-bundle I/O for the desktop window."""

    def apply_preset(self, preset: str) -> None:
        try:
            settings = self._settings_from_controls()
            self.state.settings = apply_publication_preset(settings, preset)
        except Exception as exc:  # noqa: BLE001 - surfaced to user
            self._show_status(f"Preset not applied: {exc}", timeout=8000)
            return
        self._sync_controls_from_settings()
        self.render()

    def open_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open XRDViz project", "", "XRDViz JSON (*.json);;All files (*.*)"
        )
        if not path:
            return
        try:
            self.state = load_project(path)
        except Exception as exc:  # noqa: BLE001 - surfaced to user
            QMessageBox.critical(self, "Open project failed", str(exc))
            return
        self._sync_controls_from_settings()
        self.refresh_layers()
        self.render()

    def save_project(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save XRDViz project",
            "project.xrdviz.json",
            "XRDViz JSON (*.json);;All files (*.*)",
        )
        if not path:
            return
        try:
            self.state.settings = self._settings_from_controls()
            save_project(self.state, path)
        except Exception as exc:  # noqa: BLE001 - surfaced to user
            QMessageBox.critical(self, "Save project failed", str(exc))
        else:
            self._show_status(f"Project saved: {Path(path).name}")

    def export_figure(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export figure",
            "xrd_figure.pdf",
            "PDF (*.pdf);;SVG (*.svg);;PNG (*.png);;TIFF (*.tif *.tiff)",
        )
        if not path:
            return
        try:
            self.state.settings = self._settings_from_controls()
            export_project(self.state, path)
        except Exception as exc:  # noqa: BLE001 - surfaced to user
            self._show_status(f"Figure export failed: {exc}", timeout=8000)
            QMessageBox.critical(self, "Export failed", str(exc))
        else:
            self._show_status(f"Figure exported: {Path(path).name}")

    def export_publication_bundle_dialog(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Export publication bundle")
        if not folder:
            return
        try:
            self.export_publication_bundle_to(Path(folder))
        except Exception as exc:  # noqa: BLE001 - surfaced to user
            self._show_status(f"Publication bundle export failed: {exc}", timeout=8000)
            QMessageBox.critical(self, "Publication bundle export failed", str(exc))

    def export_publication_bundle_to(self, output_dir: str | Path):
        """Export to ``output_dir``; dialog callers surface any raised error."""

        self.state.settings = self._settings_from_controls()
        outputs = export_publication_bundle(self.state, output_dir)
        self._show_status(
            f"Publication bundle exported: {Path(output_dir).name or Path(output_dir)}"
        )
        return outputs

    def export_peak_table_dialog(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Export peak table", "reference_peak_table.csv", "CSV files (*.csv)"
        )
        if not path:
            return
        from xrdviz.publication import export_peak_table

        try:
            self.state.settings = self._settings_from_controls()
            output = export_peak_table(self.state, Path(path).parent)
            if output != Path(path):
                Path(path).write_text(
                    output.read_text(encoding="utf-8"), encoding="utf-8"
                )
        except Exception as exc:  # noqa: BLE001 - surfaced to user
            self._show_status(f"Peak table export failed: {exc}", timeout=8000)
            QMessageBox.critical(self, "Peak table export failed", str(exc))
        else:
            self._show_status(f"Peak table exported: {Path(path).name}")
