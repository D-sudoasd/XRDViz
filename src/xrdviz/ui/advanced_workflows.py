from __future__ import annotations

import math
from threading import Event
from pathlib import Path
from typing import Callable

import numpy as np
from PySide6.QtCore import QObject, QThread, Qt, Signal, Slot
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from xrdviz.models import OKABE_ITO, PLOT_TEXT_COLOR, PlotAnnotation, SpectrumLayer
from xrdviz.ui.analysis_dialogs import (
    AnnotationDialog,
    DerivedAnalysisDialog,
    DetectorImportDialog,
    MapImportDialog,
    PeakDecompositionDialog,
)


def _action(text: str, parent: QWidget, slot: Callable[[], None]) -> QAction:
    action = QAction(text, parent)
    action.triggered.connect(slot)
    return action


def _set_combo_data(combo, data: str) -> None:
    for index in range(combo.count()):
        if combo.itemData(index) == data:
            combo.setCurrentIndex(index)
            return


class PeakFitWorker(QObject):
    """Run one peak decomposition outside the Qt GUI thread.

    The numerical API remains synchronous, but this worker supplies the GUI
    boundary with a cooperative cancellation event and an explicit evaluation
    budget.  It intentionally emits plain result/error signals so the mixin
    can decide how to update project state.
    """

    finished = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        seeds: tuple[object, ...],
        *,
        profile: str,
        baseline_order: int,
        max_nfev: int,
    ) -> None:
        super().__init__()
        self._x = np.asarray(x, dtype=float).copy()
        self._y = np.asarray(y, dtype=float).copy()
        self._seeds = tuple(seeds)
        self._profile = profile
        self._baseline_order = baseline_order
        self._max_nfev = max_nfev
        self._cancel_event = Event()

    @Slot()
    def run(self) -> None:
        from xrdviz.peakfit import PeakFitCancelled, fit_peaks

        try:
            result = fit_peaks(
                self._x,
                self._y,
                self._seeds,
                profile=self._profile,
                baseline_order=self._baseline_order,
                max_nfev=self._max_nfev,
                cancel_check=self._cancel_event.is_set,
            )
        except PeakFitCancelled:
            self.cancelled.emit()
        except Exception as exc:  # noqa: BLE001 - worker boundary
            self.failed.emit(str(exc))
        else:
            self.finished.emit(result)

    def cancel(self) -> None:
        """Request cancellation at the next numerical evaluation boundary."""

        self._cancel_event.set()


class AdvancedWorkflowMixin:
    """Advanced XRD UI workflows kept separate from the core window shell."""

    def _build_analysis_actions(self) -> None:
        analysis_menu = self.menuBar().addMenu("&Analysis")
        self.peak_decomposition_action = _action(
            "Decompose selected spectrum...", self, self.peak_decomposition_dialog
        )
        self.cancel_peak_decomposition_action = _action(
            "Cancel running peak decomposition",
            self,
            self.cancel_peak_decomposition,
        )
        self.detector_action = _action(
            "Open detector image / integrate...", self, self.detector_import_dialog
        )
        self.rsm_action = _action(
            "Open reciprocal-space map CSV...", self, self.import_rsm_dialog
        )
        self.pole_figure_action = _action(
            "Open pole-figure CSV...", self, self.import_pole_figure_dialog
        )
        self.peak_width_action = _action(
            "Scherrer / Williamson-Hall from peak CSV...",
            self,
            self.derived_analysis_dialog,
        )
        self.rocking_curve_action = _action(
            "Open rocking-curve CSV...", self, self.import_rocking_curve_dialog
        )
        self.annotation_action = _action(
            "Add vertical annotation...", self, self.add_annotation_dialog
        )
        for action in (
            self.peak_decomposition_action,
            self.cancel_peak_decomposition_action,
            self.detector_action,
            self.rsm_action,
            self.pole_figure_action,
            self.peak_width_action,
            self.rocking_curve_action,
            self.annotation_action,
        ):
            analysis_menu.addAction(action)

    def _analysis_tab(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        intro = QLabel(
            "Advanced workflows keep their source data and assumptions in the project and publication bundle. "
            "Detector integration uses an explicit flat-detector preview model; external refinement display is not a Rietveld solver."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)
        for label, slot in (
            ("Peak decomposition", self.peak_decomposition_dialog),
            ("Detector image / radial / cake", self.detector_import_dialog),
            ("Import RSM CSV", self.import_rsm_dialog),
            ("Import pole-figure CSV", self.import_pole_figure_dialog),
            ("Scherrer / Williamson-Hall", self.derived_analysis_dialog),
            ("Rocking curve", self.import_rocking_curve_dialog),
            ("Add plot annotation", self.add_annotation_dialog),
        ):
            button = QPushButton(label)
            button.clicked.connect(slot)
            layout.addWidget(button)
        self.cancel_peak_decomposition_button = QPushButton(
            "Cancel running peak decomposition"
        )
        self.cancel_peak_decomposition_button.clicked.connect(
            self.cancel_peak_decomposition
        )
        layout.addWidget(self.cancel_peak_decomposition_button)
        self.analysis_summary_label = QLabel("No advanced result loaded.")
        self.analysis_summary_label.setWordWrap(True)
        layout.addWidget(self.analysis_summary_label)
        layout.addStretch(1)
        return panel

    def import_pattern_fit_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open observed/calculated fit CSV",
            "",
            "CSV files (*.csv);;All files (*.*)",
        )
        if path:
            self.import_pattern_fit(Path(path))

    def import_pattern_fit(self, path: str | Path) -> None:
        from xrdviz.fit import load_pattern_fit

        try:
            fit = load_pattern_fit(path)
        except Exception as exc:  # noqa: BLE001 - surfaced to user
            QMessageBox.warning(self, "Fit import failed", str(exc))
            return
        self.state.fit = fit
        self._clear_x_range_controls()
        _set_combo_data(self.view_mode_combo, "refinement")
        _set_combo_data(self.x_axis_combo, fit.axis_kind)
        self.normalize_check.setChecked(False)
        if fit.sigma:
            _set_combo_data(self.uncertainty_mode_combo, "band")
        self.render()
        self._show_status(f"Fit imported: {Path(path).name}")

    def peak_decomposition_dialog(self) -> None:
        if not self.state.spectra:
            QMessageBox.information(
                self, "Peak decomposition", "Load at least one spectrum first."
            )
            return
        dialog = PeakDecompositionDialog(self)
        if dialog.exec() != QDialog.Accepted:
            return
        try:
            self.decompose_spectrum_async(**dialog.parameters())
        except Exception as exc:  # noqa: BLE001 - surfaced to user
            QMessageBox.warning(self, "Peak decomposition failed", str(exc))

    def decompose_spectrum(
        self,
        *,
        spectrum_index: int | None = None,
        centers: list[float] | None = None,
        max_peaks: int = 3,
        profile: str = "pseudo_voigt",
        baseline_order: int = 1,
        max_nfev: int | None = None,
    ):
        from xrdviz.peakfit import DEFAULT_PEAK_FIT_MAX_NFEV, fit_peaks

        layer, x, y, seeds = self._prepare_peak_decomposition(
            spectrum_index=spectrum_index,
            centers=centers,
            max_peaks=max_peaks,
        )
        result = fit_peaks(
            x,
            y,
            seeds,
            profile=profile,
            baseline_order=baseline_order,
            max_nfev=(
                DEFAULT_PEAK_FIT_MAX_NFEV
                if max_nfev is None
                else max_nfev
            ),
        )
        self._apply_peak_decomposition_result(layer, result)
        return result

    def decompose_spectrum_async(
        self,
        *,
        spectrum_index: int | None = None,
        centers: list[float] | None = None,
        max_peaks: int = 3,
        profile: str = "pseudo_voigt",
        baseline_order: int = 1,
        max_nfev: int | None = None,
        finished: Callable[[object], None] | None = None,
    ) -> PeakFitWorker:
        """Start peak fitting on a worker thread and return immediately."""

        from xrdviz.peakfit import DEFAULT_PEAK_FIT_MAX_NFEV

        current_thread = getattr(self, "_peak_fit_thread", None)
        if current_thread is not None and current_thread.isRunning():
            raise RuntimeError("A peak decomposition is already running")
        layer, x, y, seeds = self._prepare_peak_decomposition(
            spectrum_index=spectrum_index,
            centers=centers,
            max_peaks=max_peaks,
        )
        thread = QThread(self)
        worker = PeakFitWorker(
            x,
            y,
            tuple(seeds),
            profile=profile,
            baseline_order=baseline_order,
            max_nfev=(
                DEFAULT_PEAK_FIT_MAX_NFEV
                if max_nfev is None
                else max_nfev
            ),
        )
        worker.moveToThread(thread)
        self._peak_fit_thread = thread
        self._peak_fit_worker = worker
        self._peak_fit_layer = layer
        self._peak_fit_finished_callback = finished
        thread.started.connect(worker.run)
        worker.finished.connect(self._peak_fit_finished)
        worker.failed.connect(self._peak_fit_failed)
        worker.cancelled.connect(self._peak_fit_cancelled)
        # Quit must be requested directly from the worker thread.  A queued
        # connection would wait for the GUI event loop, which is intentionally
        # blocked by ``shutdown_peak_decomposition().wait()`` during close.
        worker.finished.connect(thread.quit, Qt.DirectConnection)
        worker.failed.connect(thread.quit, Qt.DirectConnection)
        worker.cancelled.connect(thread.quit, Qt.DirectConnection)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        worker.cancelled.connect(worker.deleteLater)
        thread.finished.connect(self._peak_fit_thread_finished)
        thread.finished.connect(thread.deleteLater)
        thread.start()
        self._show_status(
            f"Peak decomposition running (max {worker._max_nfev:g} evaluations)...",
            timeout=9000,
        )
        return worker

    def cancel_peak_decomposition(self) -> bool:
        """Request cancellation of the active worker, if one exists."""

        worker = getattr(self, "_peak_fit_worker", None)
        thread = getattr(self, "_peak_fit_thread", None)
        if worker is None or thread is None or not thread.isRunning():
            self._show_status("No peak decomposition is running")
            return False
        worker.cancel()
        self._show_status("Peak decomposition cancellation requested", timeout=9000)
        return True

    def shutdown_peak_decomposition(self, timeout_ms: int = 2000) -> bool:
        """Cancel the active fit and wait for its worker thread to finish.

        ``MainWindow.closeEvent`` should call this before allowing the window
        to close.  A timeout is reported as ``False`` so the caller can keep
        the window alive and retry; the thread is deliberately never forced
        to terminate while SciPy may still be executing.
        """

        if isinstance(timeout_ms, bool):
            raise ValueError("timeout_ms must be a non-negative integer")
        try:
            resolved_timeout = int(timeout_ms)
        except (TypeError, ValueError) as exc:
            raise ValueError("timeout_ms must be a non-negative integer") from exc
        if resolved_timeout != timeout_ms or resolved_timeout < 0:
            raise ValueError("timeout_ms must be a non-negative integer")

        thread = getattr(self, "_peak_fit_thread", None)
        if thread is None:
            return True
        worker = getattr(self, "_peak_fit_worker", None)
        if thread.isRunning():
            if worker is not None:
                worker.cancel()
            if not thread.wait(resolved_timeout):
                self._show_status(
                    "Peak decomposition is still running; close cancelled",
                    timeout=9000,
                )
                return False

        # ``thread.finished`` may be queued behind a close event.  Clear the
        # references synchronously after wait so a later close does not try to
        # reuse a finished worker; Qt's existing finished connections still
        # perform QObject cleanup when the event loop resumes.
        self._peak_fit_thread = None
        self._peak_fit_worker = None
        self._peak_fit_layer = None
        self._peak_fit_finished_callback = None
        return True

    def _prepare_peak_decomposition(
        self,
        *,
        spectrum_index: int | None,
        centers: list[float] | None,
        max_peaks: int,
    ) -> tuple[SpectrumLayer, np.ndarray, np.ndarray, list[object]]:
        from xrdviz.peakfit import PeakSeed, guess_peak_seeds

        if not self.state.spectra:
            raise ValueError("Load at least one spectrum before peak decomposition")
        index = (
            self._selected_spectrum_index()
            if spectrum_index is None
            else int(spectrum_index)
        )
        if index < 0 or index >= len(self.state.spectra):
            raise ValueError("Selected spectrum index is out of range")
        layer = self.state.spectra[index]
        x = np.asarray(layer.x, dtype=float)
        y = np.asarray(layer.y, dtype=float)
        if (
            x.size < 3
            or x.size != y.size
            or not np.all(np.isfinite(x))
            or not np.all(np.isfinite(y))
        ):
            raise ValueError(
                "Selected spectrum must contain at least three finite x/y points"
            )
        if np.any(np.diff(x) <= 0.0):
            raise ValueError("Selected spectrum x values must be strictly increasing")

        requested_centers = (
            [] if centers is None else [float(value) for value in centers]
        )
        if requested_centers:
            if any(not math.isfinite(value) for value in requested_centers):
                raise ValueError("Peak centres must be finite")
            requested_centers = sorted(requested_centers)
            if len(set(requested_centers)) != len(requested_centers):
                raise ValueError("Peak centres must be unique")
            baseline = float(np.percentile(y, 20.0))
            axis_step = float(np.median(np.diff(x)))
            initial_width = max(float(np.ptp(x)) / 80.0, axis_step * 4.0)
            seeds: list[object] = []
            for peak_index, center in enumerate(requested_centers, start=1):
                if not x[0] <= center <= x[-1]:
                    raise ValueError(
                        f"Peak centre {center:g} lies outside the selected spectrum"
                    )
                nearest = int(np.argmin(np.abs(x - center)))
                amplitude = max(float(y[nearest] - baseline), np.finfo(float).eps)
                seeds.append(
                    PeakSeed(
                        center=center,
                        amplitude=amplitude,
                        width=initial_width,
                        name=f"peak_{peak_index}",
                    )
                )
        else:
            seeds = list(guess_peak_seeds(x, y, max_peaks=max_peaks))
        return layer, x, y, seeds

    def _apply_peak_decomposition_result(
        self, layer: SpectrumLayer, result: object
    ) -> None:
        from xrdviz.peakfit import decomposition_to_pattern_fit

        self.state.fit = decomposition_to_pattern_fit(
            result,
            name=f"{layer.name} peak decomposition",
            source_path=layer.source_path,
            axis_kind=layer.axis_kind,
            wavelength_angstrom=layer.wavelength_angstrom,
        )
        self._clear_x_range_controls()
        _set_combo_data(self.view_mode_combo, "refinement")
        _set_combo_data(self.x_axis_combo, layer.axis_kind)
        self.normalize_check.setChecked(False)
        self.fit_components_check.setChecked(True)
        self.fit_background_check.setChecked(True)
        self.render()
        status = (
            "converged" if result.converged else "not converged — inspect before export"
        )
        self._show_status(
            f"Peak decomposition {status}: {len(result.summaries)} component(s)",
            timeout=9000,
        )

    def _peak_fit_finished(self, result: object) -> None:
        layer = getattr(self, "_peak_fit_layer", None)
        if layer is None:
            self._show_status("Peak decomposition finished without a source layer")
            return
        try:
            self._apply_peak_decomposition_result(layer, result)
        except Exception as exc:  # noqa: BLE001 - async result boundary
            self._show_status(f"Peak decomposition failed: {exc}", timeout=9000)
            return
        callback = getattr(self, "_peak_fit_finished_callback", None)
        if callback is not None:
            try:
                callback(result)
            except Exception as exc:  # noqa: BLE001 - caller callback boundary
                self._show_status(f"Peak decomposition callback failed: {exc}", timeout=9000)

    def _peak_fit_failed(self, message: str) -> None:
        self._show_status(f"Peak decomposition failed: {message}", timeout=9000)

    def _peak_fit_cancelled(self) -> None:
        self._show_status("Peak decomposition cancelled", timeout=9000)

    def _peak_fit_thread_finished(self) -> None:
        self._peak_fit_thread = None
        self._peak_fit_worker = None
        self._peak_fit_layer = None
        self._peak_fit_finished_callback = None

    def detector_import_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open detector image",
            "",
            "Detector data (*.npy *.npz *.tif *.tiff *.png *.jpg *.jpeg *.bmp);;All files (*.*)",
        )
        if not path:
            return
        from xrdviz.detector import load_detector_image

        try:
            image = load_detector_image(path)
        except Exception as exc:  # noqa: BLE001 - surfaced to user
            QMessageBox.warning(self, "Detector import failed", str(exc))
            return
        dialog = DetectorImportDialog((int(image.shape[0]), int(image.shape[1])), self)
        if dialog.exec() != QDialog.Accepted:
            return
        try:
            self.import_detector_image(Path(path), image=image, **dialog.parameters())
        except Exception as exc:  # noqa: BLE001 - surfaced to user
            QMessageBox.warning(self, "Detector workflow failed", str(exc))

    def import_detector_image(
        self,
        path: str | Path,
        *,
        mode: str = "raw",
        center: tuple[float, float] | None = None,
        pixel_size: tuple[float, float] | None = None,
        distance: float | None = None,
        wavelength: float | None = None,
        radial_unit: str = "two_theta",
        radial_bins: int = 720,
        chi_bins: int = 360,
        image=None,
    ):
        from xrdviz.detector import (
            DetectorGeometry,
            generate_cake,
            integrate_radial,
            load_detector_image,
        )
        from xrdviz.maps import MapData

        source = Path(path)
        detector_image = (
            load_detector_image(source) if image is None else np.asarray(image)
        )
        normalized_mode = str(mode).strip().lower()
        if normalized_mode == "raw":
            self.state.map_data = MapData.from_detector_raw(
                detector_image,
                source_path=str(source),
                metadata={
                    "processing": "raw_detector_view",
                    "calibrated": False,
                    "geometry_assumptions": [],
                },
            )
            self._activate_map_view()
            self.render()
            self._show_status(f"Raw detector image loaded: {source.name}")
            return self.state.map_data
        if (
            center is None
            or pixel_size is None
            or distance is None
            or wavelength is None
        ):
            raise ValueError(
                "Radial/cake processing requires beam centre, pixel size, distance, and wavelength"
            )
        geometry = DetectorGeometry(
            center=center,
            pixel_size=pixel_size,
            distance=distance,
            wavelength=wavelength,
        )
        if normalized_mode == "radial":
            result = integrate_radial(
                detector_image,
                geometry,
                unit=radial_unit,
                n_bins=radial_bins,
            )
            valid = (
                (result.counts > 0)
                & np.isfinite(result.intensity)
                & np.isfinite(result.bin_centers)
            )
            if int(np.count_nonzero(valid)) < 2:
                raise ValueError(
                    "Radial integration produced fewer than two populated bins"
                )
            layer = SpectrumLayer(
                name=f"{source.stem} radial",
                x=result.bin_centers[valid].tolist(),
                y=result.intensity[valid].tolist(),
                axis_kind=result.unit,
                source_path=str(source),
                color=OKABE_ITO[len(self.state.spectra) % len(OKABE_ITO)],
                order=len(self.state.spectra),
                wavelength_angstrom=geometry.wavelength,
                warnings=[
                    "Flat untilted detector preview; no distortion, polarization, solid-angle, or instrument calibration correction.",
                    (
                        f"Detector geometry: center=({geometry.center_x:g}, {geometry.center_y:g}) pixel; "
                        f"pixel_size=({geometry.pixel_size_x:g}, {geometry.pixel_size_y:g}) mm; "
                        f"distance={geometry.distance:g} mm; wavelength={geometry.wavelength:g} A."
                    ),
                ],
            )
            self.state.spectra.append(layer)
            self._clear_x_range_controls()
            _set_combo_data(self.view_mode_combo, "overlay")
            _set_combo_data(self.x_axis_combo, result.unit)
            self.normalize_check.setChecked(False)
            self.refresh_layers()
            self.render()
            self._show_status(
                f"Radial integration added: {len(layer.x)} populated bins"
            )
            return layer
        if normalized_mode == "cake":
            cake = generate_cake(
                detector_image,
                geometry,
                n_two_theta=radial_bins,
                n_chi=chi_bins,
            )
            self.state.map_data = MapData.from_cake(
                cake,
                source_path=str(source),
                metadata={
                    "processing": "flat_detector_cake_preview",
                    "beam_center_pixel": [geometry.center_x, geometry.center_y],
                    "pixel_size_mm": [geometry.pixel_size_x, geometry.pixel_size_y],
                    "distance_mm": geometry.distance,
                    "wavelength_angstrom": geometry.wavelength,
                    "two_theta_bins": int(radial_bins),
                    "chi_bins": int(chi_bins),
                    "corrections": {
                        "distortion": False,
                        "polarization": False,
                        "solid_angle": False,
                        "instrument_calibration": False,
                    },
                },
            )
            self._activate_map_view()
            self.render()
            self._show_status(f"2theta-chi cake generated: {source.name}")
            return self.state.map_data
        raise ValueError("Detector mode must be raw, radial, or cake")

    def import_rsm_dialog(self) -> None:
        self._import_map_dialog("rsm", "Open reciprocal-space map CSV")

    def import_pole_figure_dialog(self) -> None:
        self._import_map_dialog("pole_figure", "Open pole-figure CSV")

    def _import_map_dialog(self, kind: str, title: str) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, title, "", "CSV files (*.csv *.tsv);;All files (*.*)"
        )
        if not path:
            return
        dialog = MapImportDialog(kind, self)
        if dialog.exec() != QDialog.Accepted:
            return
        try:
            self.import_map_csv(path, kind=kind, **dialog.parameters())
        except Exception as exc:  # noqa: BLE001 - surfaced to user
            QMessageBox.warning(self, "Map import failed", str(exc))

    def import_map_csv(
        self,
        path: str | Path,
        *,
        kind: str,
        labels: dict[str, str] | None = None,
        units: dict[str, str] | None = None,
    ):
        from xrdviz.maps import load_map_csv

        self.state.map_data = load_map_csv(path, kind=kind, labels=labels, units=units)
        self._activate_map_view()
        self.render()
        self._show_status(f"{self.state.map_data.kind} map imported: {Path(path).name}")
        return self.state.map_data

    def _activate_map_view(self) -> None:
        self._clear_x_range_controls()
        _set_combo_data(self.view_mode_combo, "map")
        self.normalize_check.setChecked(False)
        self.log_check.setChecked(False)
        self.show_colorbar_check.setChecked(True)

    def derived_analysis_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open peak measurements CSV",
            "",
            "CSV files (*.csv);;All files (*.*)",
        )
        if not path:
            return
        dialog = DerivedAnalysisDialog(self)
        if dialog.exec() != QDialog.Accepted:
            return
        try:
            self.import_peak_measurements(path, **dialog.parameters())
        except Exception as exc:  # noqa: BLE001 - surfaced to user
            QMessageBox.warning(self, "Peak-width analysis failed", str(exc))

    def import_peak_measurements(
        self,
        path: str | Path,
        *,
        kind: str,
        wavelength: float,
        wavelength_unit: str = "nm",
        k: float = 0.9,
        instrument_fwhm: float = 0.0,
        instrument_fwhm_unit: str = "deg",
        output_unit: str = "nm",
    ):
        from xrdviz.analysis import (
            build_scherrer_plot,
            build_williamson_hall_plot,
            load_peak_measurements_csv,
        )

        peaks = load_peak_measurements_csv(path, angle_unit="deg")
        kwargs = {
            "k": k,
            "wavelength": wavelength,
            "wavelength_unit": wavelength_unit,
            "instrument_fwhm": instrument_fwhm,
            "instrument_fwhm_unit": instrument_fwhm_unit,
            "output_unit": output_unit,
            "source": str(path),
        }
        if kind == "scherrer":
            result = build_scherrer_plot(peaks, **kwargs)
        elif kind == "williamson_hall":
            result = build_williamson_hall_plot(peaks, **kwargs)
        else:
            raise ValueError(
                "Derived analysis kind must be scherrer or williamson_hall"
            )
        self.state.derived_plot = result
        self._activate_derived_view()
        self.render()
        self._show_status(f"{kind} analysis imported: {Path(path).name}")
        return result

    def import_rocking_curve_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open rocking-curve CSV", "", "CSV files (*.csv);;All files (*.*)"
        )
        if not path:
            return
        try:
            self.import_rocking_curve(path)
        except Exception as exc:  # noqa: BLE001 - surfaced to user
            QMessageBox.warning(self, "Rocking curve import failed", str(exc))

    def import_rocking_curve(self, path: str | Path, *, x_unit: str = "deg"):
        from xrdviz.analysis import build_rocking_curve_plot, load_rocking_curve_csv

        omega, intensity = load_rocking_curve_csv(path, x_unit=x_unit)
        result = build_rocking_curve_plot(
            omega, intensity, x_unit=x_unit, source=str(path)
        )
        self.state.derived_plot = result
        self._activate_derived_view()
        self.render()
        self._show_status(f"Rocking curve imported: {Path(path).name}")
        return result

    def _activate_derived_view(self) -> None:
        self._clear_x_range_controls()
        _set_combo_data(self.view_mode_combo, "derived")
        self.normalize_check.setChecked(False)
        self.log_check.setChecked(False)

    def add_annotation_dialog(self) -> None:
        dialog = AnnotationDialog(self)
        if dialog.exec() != QDialog.Accepted:
            return
        try:
            self.add_annotation(**dialog.parameters())
        except Exception as exc:  # noqa: BLE001 - surfaced to user
            QMessageBox.warning(self, "Annotation not added", str(exc))

    def add_annotation(
        self, x: float, text: str, *, color: str = PLOT_TEXT_COLOR
    ) -> PlotAnnotation:
        annotation = PlotAnnotation(x=x, text=text, color=color)
        self.state.annotations.append(annotation)
        self.render()
        self._show_status(f"Annotation added: {annotation.text}")
        return annotation

    def _selected_spectrum_index(self) -> int:
        selected = self.layer_table.selectionModel().selectedRows()
        if selected:
            role_item = self.layer_table.item(selected[0].row(), 0)
            if role_item is not None:
                kind, index = role_item.data(Qt.UserRole)
                if kind == "spectrum":
                    return int(index)
        return next(
            (index for index, layer in enumerate(self.state.spectra) if layer.visible),
            0,
        )

    def _clear_x_range_controls(self) -> None:
        self.x_min_edit.setText("")
        self.x_max_edit.setText("")

    def _update_analysis_summary(self) -> None:
        if not hasattr(self, "analysis_summary_label"):
            return
        parts = []
        if self.state.fit is not None:
            convergence = (
                "unknown convergence"
                if self.state.fit.converged is None
                else ("converged" if self.state.fit.converged else "not converged")
            )
            parts.append(
                f"Fit: {self.state.fit.name} ({len(self.state.fit.components)} components; {convergence})"
            )
        if self.state.map_data is not None:
            parts.append(
                f"Map: {self.state.map_data.kind} "
                f"({len(self.state.map_data.y)} x {len(self.state.map_data.x)} grid)"
            )
        if self.state.derived_plot is not None:
            parts.append(
                f"Derived: {self.state.derived_plot.kind} ({len(self.state.derived_plot.x)} points)"
            )
        if self.state.annotations:
            parts.append(f"Annotations: {len(self.state.annotations)}")
        self.analysis_summary_label.setText(
            "\n".join(parts) if parts else "No advanced result loaded."
        )
