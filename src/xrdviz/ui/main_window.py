from __future__ import annotations

from pathlib import Path
from typing import Callable

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from xrdviz.calibration import auto_calibrate_phases
from xrdviz.batch import apply_batch_metadata
from xrdviz.cif import load_cif_phase
from xrdviz.io import (
    apply_sample_metadata,
    load_reference_peaks_csv_many,
    load_rigaku_peaks_csv,
    load_sample_labels_csv,
    load_spectrum,
)
from xrdviz.models import OKABE_ITO, PlotSettings, ProjectState, default_axis_label
from xrdviz.plot.renderer import export_project, render_project
from xrdviz.plot.style import apply_publication_preset
from xrdviz.publication import export_publication_bundle, make_peak_table_rows
from xrdviz.project import load_project, save_project

SPECTRUM_SUFFIXES = {".txt", ".csv", ".xy", ".dat"}
CIF_SUFFIXES = {".cif"}


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("XRDViz")
        self.setAcceptDrops(True)
        self.state = ProjectState()
        self.figure = Figure(figsize=(self.state.settings.figure_width_in, self.state.settings.figure_height_in))
        self.canvas = FigureCanvas(self.figure)
        self._refreshing_layers = False

        self._build_actions()
        self._build_layout()
        self._connect_controls()
        self._sync_controls_from_settings()
        self.refresh_layers()
        self.render()

    def dragEnterEvent(self, event) -> None:  # noqa: N802 - Qt API
        if any(_is_supported_path(Path(url.toLocalFile())) for url in event.mimeData().urls()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event) -> None:  # noqa: N802 - Qt API
        paths = [Path(url.toLocalFile()) for url in event.mimeData().urls()]
        self.add_files(paths)
        event.acceptProposedAction()

    def _build_actions(self) -> None:
        file_menu = self.menuBar().addMenu("&File")

        self.open_spectrum_action = _action("Open spectrum...", self, self.open_spectra)
        self.open_folder_action = _action("Import spectra folder...", self, self.open_spectra_folder)
        self.open_cif_action = _action("Open CIF...", self, self.open_cif)
        self.open_reference_action = _action("Open reference peaks CSV...", self, self.import_reference_peaks_dialog)
        self.open_rigaku_peaks_action = _action("Open Rigaku peaks CSV...", self, self.import_rigaku_peaks_dialog)
        self.open_metadata_action = _action("Open sample labels CSV...", self, self.import_sample_metadata_dialog)
        self.open_project_action = _action("Open project...", self, self.open_project)
        self.save_project_action = _action("Save project...", self, self.save_project)
        self.export_action = _action("Export figure...", self, self.export_figure)
        self.export_bundle_action = _action("Export publication bundle...", self, self.export_publication_bundle_dialog)
        self.quit_action = _action("Quit", self, self.close)

        for action in (
            self.open_spectrum_action,
            self.open_folder_action,
            self.open_cif_action,
            self.open_reference_action,
            self.open_rigaku_peaks_action,
            self.open_metadata_action,
            self.open_project_action,
            self.save_project_action,
            self.export_action,
            self.export_bundle_action,
        ):
            file_menu.addAction(action)
        file_menu.addSeparator()
        file_menu.addAction(self.quit_action)

        toolbar = QToolBar("Main")
        toolbar.setMovable(False)
        self.addToolBar(Qt.TopToolBarArea, toolbar)
        for action in (self.open_spectrum_action, self.open_cif_action, self.open_reference_action, self.export_action):
            toolbar.addAction(action)

    def _build_layout(self) -> None:
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._layers_panel())
        splitter.addWidget(self.canvas)
        splitter.addWidget(self._properties_panel())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([250, 780, 300])
        self.setCentralWidget(splitter)

    def _layers_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.addWidget(QLabel("Layers"))

        self.layer_table = QTableWidget(0, 7)
        self.layer_table.setHorizontalHeaderLabels(["Show", "Type", "Label", "Axis", "Color", "Offset", "Width/a"])
        self.layer_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.layer_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.layer_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.layer_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        layout.addWidget(self.layer_table, stretch=1)

        buttons = QHBoxLayout()
        add_spectrum = QPushButton("Spectrum")
        add_spectrum.clicked.connect(self.open_spectra)
        add_cif = QPushButton("CIF")
        add_cif.clicked.connect(self.open_cif)
        remove = QPushButton("Remove")
        remove.clicked.connect(self.remove_selected_layer)
        toggle = QPushButton("Toggle")
        toggle.clicked.connect(self.toggle_selected_visibility)
        up = QPushButton("Up")
        up.clicked.connect(lambda: self.move_selected_layer(-1))
        down = QPushButton("Down")
        down.clicked.connect(lambda: self.move_selected_layer(1))
        buttons.addWidget(add_spectrum)
        buttons.addWidget(add_cif)
        buttons.addWidget(remove)
        buttons.addWidget(toggle)
        buttons.addWidget(up)
        buttons.addWidget(down)
        layout.addLayout(buttons)

        color_button = QPushButton("Set color")
        color_button.clicked.connect(self.set_selected_color)
        layout.addWidget(color_button)
        return panel

    def _properties_panel(self) -> QWidget:
        tabs = QTabWidget()
        tabs.addTab(self._plot_properties_tab(), "Plot")
        tabs.addTab(self._batch_properties_tab(), "Batch")
        tabs.addTab(self._reference_peaks_tab(), "Reference Peaks")
        return tabs

    def _plot_properties_tab(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        form = QFormLayout(container)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

        self.input_axis_combo = _axis_combo(include_auto=True)
        form.addRow("New spectrum axis", self.input_axis_combo)

        self.x_axis_combo = _axis_combo()
        form.addRow("Display x axis", self.x_axis_combo)

        self.energy_spin = _double_spin(1.0, 200.0, 4)
        self.energy_spin.setSuffix(" keV")
        form.addRow("Energy", self.energy_spin)

        self.x_label_edit = QLineEdit()
        self.y_label_edit = QLineEdit()
        self.panel_title_edit = QLineEdit()
        form.addRow("X title", self.x_label_edit)
        form.addRow("Y title", self.y_label_edit)
        form.addRow("Panel title", self.panel_title_edit)

        self.x_min_edit = QLineEdit()
        self.x_max_edit = QLineEdit()
        self.x_min_edit.setPlaceholderText("auto")
        self.x_max_edit.setPlaceholderText("auto")
        form.addRow("X min", self.x_min_edit)
        form.addRow("X max", self.x_max_edit)

        self.normalize_check = QCheckBox("Normalize each spectrum")
        self.log_check = QCheckBox("Log10 display")
        self.stack_check = QCheckBox("Stack spectra")
        form.addRow(self.normalize_check)
        form.addRow(self.log_check)
        form.addRow(self.stack_check)

        self.stack_spacing_spin = _double_spin(0.0, 10.0, 3)
        form.addRow("Stack spacing", self.stack_spacing_spin)

        self.width_spin = _double_spin(1.0, 20.0, 3)
        self.width_spin.setSuffix(" in")
        self.height_spin = _double_spin(1.0, 20.0, 3)
        self.height_spin.setSuffix(" in")
        self.dpi_spin = QSpinBox()
        self.dpi_spin.setRange(72, 2400)
        self.dpi_spin.setSingleStep(50)
        form.addRow("Figure width", self.width_spin)
        form.addRow("Figure height", self.height_spin)
        form.addRow("DPI", self.dpi_spin)

        self.font_size_spin = _double_spin(4.0, 24.0, 1)
        self.axis_label_size_spin = _double_spin(4.0, 30.0, 1)
        self.tick_label_size_spin = _double_spin(4.0, 24.0, 1)
        self.line_width_spin = _double_spin(0.1, 8.0, 2)
        self.bragg_height_spin = _double_spin(0.05, 1.0, 2)
        self.legend_check = QCheckBox("Show legend")
        self.direct_labels_check = QCheckBox("Direct curve labels")
        self.phase_legend_check = QCheckBox("Show phase legend")
        self.y_tick_labels_check = QCheckBox("Show y tick labels")
        self.legend_location_combo = _field_combo(
            [
                ("Upper right", "upper right"),
                ("Upper left", "upper left"),
                ("Lower right", "lower right"),
                ("Lower left", "lower left"),
                ("Best", "best"),
                ("Outside right", "outside right"),
                ("None", "none"),
            ]
        )
        self.template_combo = _field_combo(
            [
                ("Nature single", "nature_single"),
                ("Nature double", "nature_double"),
                ("Science single", "science_single"),
                ("Science double", "science_double"),
                ("Custom", "custom"),
            ]
        )
        form.addRow("Font size", self.font_size_spin)
        form.addRow("Axis label size", self.axis_label_size_spin)
        form.addRow("Tick label size", self.tick_label_size_spin)
        form.addRow("Line width", self.line_width_spin)
        form.addRow("Bragg band height", self.bragg_height_spin)
        form.addRow("Legend position", self.legend_location_combo)
        form.addRow("Template", self.template_combo)
        form.addRow(self.legend_check)
        form.addRow(self.direct_labels_check)
        form.addRow(self.phase_legend_check)
        form.addRow(self.y_tick_labels_check)

        preset_bar = QHBoxLayout()
        single = QPushButton("Nature single")
        double = QPushButton("Nature double")
        science_single = QPushButton("Science single")
        science_double = QPushButton("Science double")
        single.clicked.connect(lambda: self.apply_preset("nature_single"))
        double.clicked.connect(lambda: self.apply_preset("nature_double"))
        science_single.clicked.connect(lambda: self.apply_preset("science_single"))
        science_double.clicked.connect(lambda: self.apply_preset("science_double"))
        preset_bar.addWidget(single)
        preset_bar.addWidget(double)
        preset_bar.addWidget(science_single)
        preset_bar.addWidget(science_double)
        form.addRow(preset_bar)

        refresh = QPushButton("Refresh plot")
        refresh.clicked.connect(self.render)
        form.addRow(refresh)

        divider = QFrame()
        divider.setFrameShape(QFrame.HLine)
        form.addRow(divider)
        form.addRow(QLabel("Tip: drag spectra or CIF files into the window."))

        scroll.setWidget(container)
        return scroll

    def _batch_properties_tab(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        form = QFormLayout(container)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

        self.view_mode_combo = _field_combo(
            [
                ("Overlay", "overlay"),
                ("Stack", "stack"),
                ("Gradient stack", "gradient_stack"),
                ("Heatmap", "heatmap"),
            ]
        )
        self.sort_by_combo = _field_combo(
            [
                ("Frame", "frame"),
                ("Time", "time"),
                ("Temperature", "temperature"),
                ("Order", "order"),
            ]
        )
        self.color_by_combo = _field_combo(
            [
                ("Frame", "frame"),
                ("Time", "time"),
                ("Temperature", "temperature"),
                ("Order", "order"),
            ]
        )
        self.colormap_combo = _field_combo(
            [
                ("Blue rose", "blue_rose"),
                ("Viridis", "viridis"),
                ("Plasma", "plasma"),
                ("Magma", "magma"),
                ("Cividis", "cividis"),
                ("Turbo", "turbo"),
            ]
        )
        self.show_colorbar_check = QCheckBox("Show colorbar")
        self.show_every_n_spin = QSpinBox()
        self.show_every_n_spin.setRange(1, 1000)
        self.heatmap_points_spin = QSpinBox()
        self.heatmap_points_spin.setRange(16, 10000)
        self.heatmap_points_spin.setSingleStep(64)

        form.addRow("View mode", self.view_mode_combo)
        form.addRow("Sort by", self.sort_by_combo)
        form.addRow("Color by", self.color_by_combo)
        form.addRow("Colormap", self.colormap_combo)
        form.addRow(self.show_colorbar_check)
        form.addRow("Show every N", self.show_every_n_spin)
        form.addRow("Heatmap points", self.heatmap_points_spin)

        apply_button = QPushButton("Apply metadata and sort")
        apply_button.clicked.connect(self.apply_batch_metadata_to_layers)
        form.addRow(apply_button)

        scroll.setWidget(container)
        return scroll

    def _reference_peaks_tab(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        controls = QHBoxLayout()
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
        controls.addWidget(ref_button)
        controls.addWidget(rigaku_button)
        controls.addWidget(cif_button)
        controls.addWidget(self.auto_fit_button)
        controls.addWidget(export_button)
        layout.addLayout(controls)

        self.peak_table = QTableWidget(0, 8)
        self.peak_table.setHorizontalHeaderLabels(["Phase", "2theta", "d", "Q", "Intensity", "hkl", "Label", "Source"])
        self.peak_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.peak_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        layout.addWidget(self.peak_table, stretch=1)
        return panel

    def _connect_controls(self) -> None:
        controls = [
            self.x_axis_combo,
            self.energy_spin,
            self.x_label_edit,
            self.y_label_edit,
            self.panel_title_edit,
            self.x_min_edit,
            self.x_max_edit,
            self.normalize_check,
            self.log_check,
            self.stack_check,
            self.stack_spacing_spin,
            self.width_spin,
            self.height_spin,
            self.dpi_spin,
            self.font_size_spin,
            self.axis_label_size_spin,
            self.tick_label_size_spin,
            self.line_width_spin,
            self.bragg_height_spin,
            self.legend_location_combo,
            self.template_combo,
            self.legend_check,
            self.direct_labels_check,
            self.phase_legend_check,
            self.y_tick_labels_check,
            self.view_mode_combo,
            self.sort_by_combo,
            self.color_by_combo,
            self.colormap_combo,
            self.show_colorbar_check,
            self.show_every_n_spin,
            self.heatmap_points_spin,
        ]
        for control in controls:
            if isinstance(control, QLineEdit):
                control.editingFinished.connect(self.render)
            elif control is self.x_axis_combo:
                control.currentIndexChanged.connect(lambda *_args: self._axis_changed())
            elif control is self.template_combo:
                control.currentIndexChanged.connect(lambda *_args: self._template_changed())
            elif control in (self.sort_by_combo, self.color_by_combo, self.colormap_combo):
                control.currentIndexChanged.connect(lambda *_args: self._batch_controls_changed())
            elif isinstance(control, QComboBox):
                control.currentIndexChanged.connect(lambda *_args: self.render())
            elif isinstance(control, (QCheckBox, QSpinBox, QDoubleSpinBox)):
                signal = control.stateChanged if isinstance(control, QCheckBox) else control.valueChanged
                signal.connect(lambda *_args: self.render())
        self.layer_table.itemChanged.connect(self._layer_item_changed)

    def _axis_changed(self) -> None:
        axis_kind = self.x_axis_combo.currentData()
        self.x_label_edit.setText(default_axis_label(axis_kind))
        self.render()

    def _template_changed(self) -> None:
        preset = self.template_combo.currentData()
        if preset and preset != "custom":
            self.apply_preset(preset)
        else:
            self.render()

    def _batch_controls_changed(self) -> None:
        if self.state.spectra:
            self.apply_batch_metadata_to_layers()
        else:
            self.render()

    def open_spectra(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Open spectrum files",
            "",
            "Spectrum files (*.txt *.csv *.xy *.dat);;All files (*.*)",
        )
        self.add_files([Path(path) for path in paths])

    def open_spectra_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Import spectra folder")
        if not folder:
            return
        paths = sorted(path for path in Path(folder).iterdir() if path.suffix.lower() in SPECTRUM_SUFFIXES)
        self.add_files(paths)

    def open_cif(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Open CIF files", "", "CIF files (*.cif);;All files (*.*)")
        self.add_files([Path(path) for path in paths])

    def import_sample_metadata_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open sample labels CSV", "", "CSV files (*.csv);;All files (*.*)")
        if path:
            self.import_sample_metadata(Path(path))

    def import_sample_metadata(self, path: str | Path) -> None:
        try:
            metadata = load_sample_labels_csv(Path(path).read_text(encoding="utf-8-sig"))
            apply_sample_metadata(self.state.spectra, metadata)
        except Exception as exc:  # noqa: BLE001 - surfaced to user
            QMessageBox.warning(self, "Sample metadata import failed", str(exc))
            return
        self.refresh_layers()
        self.render()

    def apply_batch_metadata_to_layers(self) -> None:
        self._apply_batch_metadata_to_state()
        self.refresh_layers()
        self.render()

    def import_reference_peaks_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open reference peaks CSV", "", "CSV files (*.csv);;All files (*.*)")
        if path:
            self.import_reference_peaks(Path(path))

    def import_reference_peaks(self, path: str | Path) -> None:
        try:
            text = Path(path).read_text(encoding="utf-8-sig")
            self.state.phases.extend(load_reference_peaks_csv_many(text, source_path=str(path)))
        except Exception as exc:  # noqa: BLE001 - surfaced to user
            QMessageBox.warning(self, "Reference peaks import failed", str(exc))
            return
        self.refresh_layers()
        self.refresh_peak_table()
        self.render()

    def import_rigaku_peaks_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open Rigaku peaks CSV", "", "CSV files (*.csv);;All files (*.*)")
        if path:
            self.import_rigaku_peaks(Path(path))

    def import_rigaku_peaks(self, path: str | Path) -> None:
        try:
            self.state.phases.append(load_rigaku_peaks_csv(Path(path).read_text(encoding="utf-8-sig"), source_path=str(path)))
        except Exception as exc:  # noqa: BLE001 - surfaced to user
            QMessageBox.warning(self, "Rigaku peaks import failed", str(exc))
            return
        self.refresh_layers()
        self.refresh_peak_table()
        self.render()

    def add_files(self, paths: list[Path]) -> None:
        spectrum_paths = [path for path in paths if path.suffix.lower() in SPECTRUM_SUFFIXES]
        cif_paths = [path for path in paths if path.suffix.lower() in CIF_SUFFIXES]
        self._add_spectra(spectrum_paths)
        self._add_cifs(cif_paths)
        if spectrum_paths or cif_paths:
            self.refresh_layers()
            self.refresh_peak_table()
            self.render()

    def remove_selected_layer(self) -> None:
        selected = self.layer_table.selectionModel().selectedRows()
        if not selected:
            return
        kind, index = self.layer_table.item(selected[0].row(), 0).data(Qt.UserRole)
        if kind == "spectrum":
            del self.state.spectra[index]
        else:
            del self.state.phases[index]
        self.refresh_layers()
        self.refresh_peak_table()
        self.render()

    def set_selected_color(self) -> None:
        selected = self.layer_table.selectionModel().selectedRows()
        if not selected:
            return
        kind, index = self.layer_table.item(selected[0].row(), 0).data(Qt.UserRole)
        current = self.state.spectra[index].color if kind == "spectrum" else self.state.phases[index].color
        color = QColorDialog.getColor(QColor(current), self, "Choose layer color")
        if not color.isValid():
            return
        if kind == "spectrum":
            self.state.spectra[index].color = color.name()
        else:
            self.state.phases[index].color = color.name()
        self.refresh_layers()
        self.refresh_peak_table()
        self.render()

    def toggle_selected_visibility(self) -> None:
        selected = self.layer_table.selectionModel().selectedRows()
        if not selected:
            return
        row = selected[0].row()
        kind, index = self.layer_table.item(selected[0].row(), 0).data(Qt.UserRole)
        if kind == "spectrum":
            self.state.spectra[index].visible = not self.state.spectra[index].visible
        else:
            self.state.phases[index].visible = not self.state.phases[index].visible
        self.refresh_layers()
        if self.layer_table.rowCount():
            self.layer_table.selectRow(min(row, self.layer_table.rowCount() - 1))
        self.refresh_peak_table()
        self.render()

    def move_selected_layer(self, direction: int) -> None:
        selected = self.layer_table.selectionModel().selectedRows()
        if not selected:
            return
        kind, index = self.layer_table.item(selected[0].row(), 0).data(Qt.UserRole)
        collection = self.state.spectra if kind == "spectrum" else self.state.phases
        new_index = index + direction
        if new_index < 0 or new_index >= len(collection):
            return
        collection[index], collection[new_index] = collection[new_index], collection[index]
        if kind == "spectrum":
            for order, layer in enumerate(self.state.spectra):
                layer.order = order
        self.refresh_layers()
        self.layer_table.selectRow(new_index)
        self.refresh_peak_table()
        self.render()

    def apply_preset(self, preset: str) -> None:
        settings = self._settings_from_controls()
        self.state.settings = apply_publication_preset(settings, preset)
        self._sync_controls_from_settings()
        self.render()

    def open_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open XRDViz project", "", "XRDViz JSON (*.json);;All files (*.*)")
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
        self.state.settings = self._settings_from_controls()
        try:
            save_project(self.state, path)
        except Exception as exc:  # noqa: BLE001 - surfaced to user
            QMessageBox.critical(self, "Save project failed", str(exc))

    def export_figure(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export figure",
            "xrd_figure.pdf",
            "PDF (*.pdf);;SVG (*.svg);;PNG (*.png);;TIFF (*.tif *.tiff)",
        )
        if not path:
            return
        self.state.settings = self._settings_from_controls()
        try:
            export_project(self.state, path)
        except Exception as exc:  # noqa: BLE001 - surfaced to user
            QMessageBox.critical(self, "Export failed", str(exc))

    def export_publication_bundle_dialog(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Export publication bundle")
        if folder:
            self.export_publication_bundle_to(Path(folder))

    def export_publication_bundle_to(self, output_dir: str | Path):
        self.state.settings = self._settings_from_controls()
        return export_publication_bundle(self.state, output_dir)

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

    def export_peak_table_dialog(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export peak table", "reference_peak_table.csv", "CSV files (*.csv)")
        if not path:
            return
        from xrdviz.publication import export_peak_table

        output = export_peak_table(self.state, Path(path).parent)
        if output != Path(path):
            Path(path).write_text(output.read_text(encoding="utf-8"), encoding="utf-8")

    def refresh_layers(self) -> None:
        self._refreshing_layers = True
        self.layer_table.setRowCount(0)
        for index, spectrum in enumerate(self.state.spectra):
            row = self.layer_table.rowCount()
            self.layer_table.insertRow(row)
            self._set_layer_row(row, "spectrum", index, spectrum.visible, spectrum.name, spectrum.axis_kind, spectrum.color, spectrum.offset, spectrum.linewidth)
        for index, phase in enumerate(self.state.phases):
            row = self.layer_table.rowCount()
            self.layer_table.insertRow(row)
            self._set_layer_row(row, phase.source_type, index, phase.visible, phase.name, phase.source_axis, phase.color, "", phase.lattice_a)
        self._refreshing_layers = False

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

    def _set_layer_row(self, row: int, kind: str, index: int, visible: bool, label: str, axis_kind: str, color: str, offset: object, linewidth: object) -> None:
        show_item = QTableWidgetItem("yes" if visible else "no")
        show_item.setData(Qt.UserRole, (kind, index))
        self.layer_table.setItem(row, 0, show_item)
        for column, value in enumerate([kind, label, axis_kind, color, _format_table_value(offset), _format_table_value(linewidth)], start=1):
            item = QTableWidgetItem(str(value))
            if column == 4:
                item.setForeground(QColor(color))
            self.layer_table.setItem(row, column, item)

    def _layer_item_changed(self, item: QTableWidgetItem) -> None:
        if self._refreshing_layers:
            return
        role_item = self.layer_table.item(item.row(), 0)
        if role_item is None:
            return
        kind, index = role_item.data(Qt.UserRole)
        text = item.text().strip()
        try:
            if kind == "spectrum":
                self._update_spectrum_from_table(index, item.column(), text)
            else:
                self._update_phase_from_table(index, item.column(), text)
        except ValueError:
            self.refresh_layers()
            return
        self.refresh_peak_table()
        self.render()

    def _update_spectrum_from_table(self, index: int, column: int, text: str) -> None:
        layer = self.state.spectra[index]
        if column == 2 and text:
            layer.name = text
        elif column == 4 and QColor(text).isValid():
            layer.color = QColor(text).name()
        elif column == 5:
            layer.offset = float(text)
        elif column == 6:
            layer.linewidth = float(text)

    def _update_phase_from_table(self, index: int, column: int, text: str) -> None:
        phase = self.state.phases[index]
        if column == 2 and text:
            phase.name = text
            phase.phase = text
        elif column == 4 and QColor(text).isValid():
            phase.color = QColor(text).name()
        elif column == 6:
            phase.lattice_a = phase.reference_lattice_a if not text else float(text)

    def render(self) -> None:
        try:
            self.state.settings = self._settings_from_controls()
            render_project(self.state, self.figure)
        except Exception as exc:  # noqa: BLE001 - surfaced to user
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, str(exc), ha="center", va="center", wrap=True)
            ax.set_axis_off()
        self.canvas.draw_idle()
        self.refresh_peak_table()

    def _add_spectra(self, paths: list[Path]) -> None:
        axis_kind = self.input_axis_combo.currentData()
        for path in paths:
            try:
                color = OKABE_ITO[len(self.state.spectra) % len(OKABE_ITO)]
                self.state.spectra.append(load_spectrum(path, axis_kind=axis_kind, color=color))
            except Exception as exc:  # noqa: BLE001 - surfaced to user
                QMessageBox.warning(self, "Spectrum import failed", f"{path.name}: {exc}")
        if paths:
            self._apply_batch_metadata_to_state()

    def _apply_batch_metadata_to_state(self) -> None:
        if not self.state.spectra:
            return
        sort_by = self.sort_by_combo.currentData() if hasattr(self, "sort_by_combo") else "frame"
        color_by = self.color_by_combo.currentData() if hasattr(self, "color_by_combo") else "frame"
        colormap = self.colormap_combo.currentData() if hasattr(self, "colormap_combo") else "blue_rose"
        apply_batch_metadata(self.state.spectra, sort_by=sort_by, color_by=color_by, colormap=colormap)

    def _add_cifs(self, paths: list[Path]) -> None:
        settings = self._settings_from_controls()
        for path in paths:
            try:
                color = OKABE_ITO[(len(self.state.spectra) + len(self.state.phases)) % len(OKABE_ITO)]
                self.state.phases.append(load_cif_phase(path, energy_kev=settings.energy_kev, color=color))
            except Exception as exc:  # noqa: BLE001 - surfaced to user
                QMessageBox.warning(self, "CIF import failed", f"{path.name}: {exc}")

    def _settings_from_controls(self) -> PlotSettings:
        return PlotSettings(
            x_axis=self.x_axis_combo.currentData(),
            energy_kev=self.energy_spin.value(),
            x_label=self.x_label_edit.text().strip() or default_axis_label(self.x_axis_combo.currentData()),
            y_label=self.y_label_edit.text().strip() or "Intensity (a.u.)",
            panel_title=self.panel_title_edit.text().strip(),
            x_min=_optional_float(self.x_min_edit.text()),
            x_max=_optional_float(self.x_max_edit.text()),
            normalize=self.normalize_check.isChecked(),
            log_scale=self.log_check.isChecked(),
            stack_enabled=self.stack_check.isChecked(),
            stack_spacing=self.stack_spacing_spin.value(),
            figure_width_in=self.width_spin.value(),
            figure_height_in=self.height_spin.value(),
            dpi=self.dpi_spin.value(),
            font_size=self.font_size_spin.value(),
            axis_label_size=self.axis_label_size_spin.value(),
            tick_label_size=self.tick_label_size_spin.value(),
            line_width=self.line_width_spin.value(),
            bragg_band_height=self.bragg_height_spin.value(),
            show_legend=self.legend_check.isChecked(),
            direct_labels=self.direct_labels_check.isChecked(),
            show_phase_legend=self.phase_legend_check.isChecked(),
            show_y_tick_labels=self.y_tick_labels_check.isChecked(),
            view_mode=self.view_mode_combo.currentData(),
            sort_by=self.sort_by_combo.currentData(),
            color_by=self.color_by_combo.currentData(),
            colormap=self.colormap_combo.currentData(),
            show_colorbar=self.show_colorbar_check.isChecked(),
            show_every_n=self.show_every_n_spin.value(),
            heatmap_points=self.heatmap_points_spin.value(),
            legend_location=self.legend_location_combo.currentData(),
            template_name=self.template_combo.currentData(),
        )

    def _sync_controls_from_settings(self) -> None:
        settings = self.state.settings
        setters: list[tuple[QWidget, Callable[[], None]]] = [
            (self.x_axis_combo, lambda: _set_combo_data(self.x_axis_combo, settings.x_axis)),
            (self.energy_spin, lambda: self.energy_spin.setValue(settings.energy_kev)),
            (self.x_label_edit, lambda: self.x_label_edit.setText(settings.x_label)),
            (self.y_label_edit, lambda: self.y_label_edit.setText(settings.y_label)),
            (self.panel_title_edit, lambda: self.panel_title_edit.setText(settings.panel_title)),
            (self.x_min_edit, lambda: self.x_min_edit.setText("" if settings.x_min is None else f"{settings.x_min:g}")),
            (self.x_max_edit, lambda: self.x_max_edit.setText("" if settings.x_max is None else f"{settings.x_max:g}")),
            (self.normalize_check, lambda: self.normalize_check.setChecked(settings.normalize)),
            (self.log_check, lambda: self.log_check.setChecked(settings.log_scale)),
            (self.stack_check, lambda: self.stack_check.setChecked(settings.stack_enabled)),
            (self.stack_spacing_spin, lambda: self.stack_spacing_spin.setValue(settings.stack_spacing)),
            (self.width_spin, lambda: self.width_spin.setValue(settings.figure_width_in)),
            (self.height_spin, lambda: self.height_spin.setValue(settings.figure_height_in)),
            (self.dpi_spin, lambda: self.dpi_spin.setValue(settings.dpi)),
            (self.font_size_spin, lambda: self.font_size_spin.setValue(settings.font_size)),
            (self.axis_label_size_spin, lambda: self.axis_label_size_spin.setValue(settings.axis_label_size)),
            (self.tick_label_size_spin, lambda: self.tick_label_size_spin.setValue(settings.tick_label_size)),
            (self.line_width_spin, lambda: self.line_width_spin.setValue(settings.line_width)),
            (self.bragg_height_spin, lambda: self.bragg_height_spin.setValue(settings.bragg_band_height)),
            (self.legend_location_combo, lambda: _set_combo_data(self.legend_location_combo, settings.legend_location)),
            (self.template_combo, lambda: _set_combo_data(self.template_combo, settings.template_name)),
            (self.legend_check, lambda: self.legend_check.setChecked(settings.show_legend)),
            (self.direct_labels_check, lambda: self.direct_labels_check.setChecked(settings.direct_labels)),
            (self.phase_legend_check, lambda: self.phase_legend_check.setChecked(settings.show_phase_legend)),
            (self.y_tick_labels_check, lambda: self.y_tick_labels_check.setChecked(settings.show_y_tick_labels)),
            (self.view_mode_combo, lambda: _set_combo_data(self.view_mode_combo, settings.view_mode)),
            (self.sort_by_combo, lambda: _set_combo_data(self.sort_by_combo, settings.sort_by)),
            (self.color_by_combo, lambda: _set_combo_data(self.color_by_combo, settings.color_by)),
            (self.colormap_combo, lambda: _set_combo_data(self.colormap_combo, settings.colormap)),
            (self.show_colorbar_check, lambda: self.show_colorbar_check.setChecked(settings.show_colorbar)),
            (self.show_every_n_spin, lambda: self.show_every_n_spin.setValue(settings.show_every_n)),
            (self.heatmap_points_spin, lambda: self.heatmap_points_spin.setValue(settings.heatmap_points)),
        ]
        for widget, setter in setters:
            was_blocked = widget.blockSignals(True)
            setter()
            widget.blockSignals(was_blocked)


def _axis_combo(*, include_auto: bool = False) -> QComboBox:
    combo = QComboBox()
    if include_auto:
        combo.addItem("Auto", "auto")
    combo.addItem("2theta", "two_theta")
    combo.addItem("d-spacing", "d")
    combo.addItem("Q", "q")
    return combo


def _field_combo(items: list[tuple[str, str]]) -> QComboBox:
    combo = QComboBox()
    for label, value in items:
        combo.addItem(label, value)
    return combo


def _set_combo_data(combo: QComboBox, data: str) -> None:
    for index in range(combo.count()):
        if combo.itemData(index) == data:
            combo.setCurrentIndex(index)
            return


def _double_spin(minimum: float, maximum: float, decimals: int) -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setRange(minimum, maximum)
    spin.setDecimals(decimals)
    spin.setSingleStep(10 ** -decimals)
    return spin


def _optional_float(text: str) -> float | None:
    stripped = text.strip()
    if not stripped:
        return None
    return float(stripped)


def _format_table_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (float, int)):
        return f"{value:g}"
    return str(value)


def _action(text: str, parent: QWidget, slot: Callable[[], None]) -> QAction:
    action = QAction(text, parent)
    action.triggered.connect(slot)
    return action


def _is_supported_path(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SPECTRUM_SUFFIXES | CIF_SUFFIXES
