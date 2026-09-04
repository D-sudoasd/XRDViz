from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QVBoxLayout,
)

from xrdviz.detector import MAX_CAKE_CELLS


_MIN_CAKE_BINS = 8
# Keep the 1-D radial workflow at the resolution supported by the public
# integration API.  Cake output has a separate product budget and couples the
# two axes dynamically below; using sqrt(MAX_CAKE_CELLS) as an unconditional
# per-axis maximum would silently cap ordinary radial plots at 2,000 bins.
_MAX_RADIAL_BINS = 100_000
_MAX_CAKE_BINS_PER_AXIS = min(
    _MAX_RADIAL_BINS,
    MAX_CAKE_CELLS // _MIN_CAKE_BINS,
)


def _combo(items: list[tuple[str, str]]) -> QComboBox:
    combo = QComboBox()
    for label, value in items:
        combo.addItem(label, value)
    return combo


def _double(
    minimum: float, maximum: float, value: float, decimals: int = 6
) -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setRange(minimum, maximum)
    spin.setDecimals(decimals)
    spin.setValue(value)
    spin.setKeyboardTracking(False)
    return spin


class _AcceptedDialog(QDialog):
    def _finish(self, form: QFormLayout, *, help_text: str = "") -> None:
        layout = QVBoxLayout(self)
        if help_text:
            help_label = QLabel(help_text)
            help_label.setWordWrap(True)
            layout.addWidget(help_label)
        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


class PeakDecompositionDialog(_AcceptedDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Peak decomposition")
        form = QFormLayout()
        self.profile = _combo(
            [
                ("Pseudo-Voigt", "pseudo_voigt"),
                ("Gaussian", "gaussian"),
                ("Lorentzian", "lorentzian"),
            ]
        )
        self.baseline_order = QSpinBox()
        self.baseline_order.setRange(0, 2)
        self.baseline_order.setValue(1)
        self.centers = QLineEdit()
        self.centers.setPlaceholderText("e.g. 35.2, 40.1; leave blank for suggestions")
        self.max_peaks = QSpinBox()
        self.max_peaks.setRange(1, 20)
        self.max_peaks.setValue(3)
        form.addRow("Peak profile", self.profile)
        form.addRow("Baseline order", self.baseline_order)
        form.addRow("Peak centres", self.centers)
        form.addRow("Suggested peak limit", self.max_peaks)
        self._finish(
            form,
            help_text=(
                "Peak centres are fit seeds, not phase assignments. Blank centres use a deterministic "
                "prominence-based suggestion that remains visible in the fitted component output."
            ),
        )

    def parameters(self) -> dict[str, object]:
        text = self.centers.text().strip().replace(";", ",")
        centers = (
            []
            if not text
            else [float(value.strip()) for value in text.split(",") if value.strip()]
        )
        return {
            "profile": self.profile.currentData(),
            "baseline_order": self.baseline_order.value(),
            "centers": centers,
            "max_peaks": self.max_peaks.value(),
        }


class DetectorImportDialog(_AcceptedDialog):
    def __init__(self, image_shape: tuple[int, int], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Detector image workflow")
        height, width = image_shape
        form = QFormLayout()
        self.mode = _combo(
            [
                ("Raw detector image", "raw"),
                ("Radial integration (1D)", "radial"),
                ("2theta-chi cake", "cake"),
            ]
        )
        self.center_x = _double(0.0, max(width - 1.0, 0.0), (width - 1.0) / 2.0, 3)
        self.center_y = _double(0.0, max(height - 1.0, 0.0), (height - 1.0) / 2.0, 3)
        self.pixel_x = _double(1.0e-9, 1.0e6, 0.1)
        self.pixel_y = _double(1.0e-9, 1.0e6, 0.1)
        self.distance = _double(1.0e-9, 1.0e9, 100.0)
        self.wavelength = _double(1.0e-9, 1.0e6, 1.5406)
        self.radial_unit = _combo(
            [("2θ (deg)", "two_theta"), ("Q (Å⁻¹)", "q"), ("d (Å)", "d")]
        )
        self._updating_bin_budget = False
        self.radial_bins = QSpinBox()
        self.radial_bins.setRange(_MIN_CAKE_BINS, _MAX_RADIAL_BINS)
        self.radial_bins.setValue(720)
        self.chi_bins = QSpinBox()
        self.chi_bins.setRange(_MIN_CAKE_BINS, _MAX_CAKE_BINS_PER_AXIS)
        self.chi_bins.setValue(360)
        form.addRow("Output", self.mode)
        form.addRow("Beam centre x (pixel)", self.center_x)
        form.addRow("Beam centre y (pixel)", self.center_y)
        form.addRow("Pixel size x (mm)", self.pixel_x)
        form.addRow("Pixel size y (mm)", self.pixel_y)
        form.addRow("Detector distance (mm)", self.distance)
        form.addRow("Wavelength (A)", self.wavelength)
        form.addRow("Radial axis", self.radial_unit)
        form.addRow("Radial / 2theta bins", self.radial_bins)
        form.addRow("Chi bins", self.chi_bins)
        self.mode.currentIndexChanged.connect(self._update_enabled)
        self.radial_bins.valueChanged.connect(self._update_cake_budget)
        self.chi_bins.valueChanged.connect(self._update_cake_budget)
        self._finish(
            form,
            help_text=(
                "Raw view needs no geometry. Radial/cake previews use a flat, untilted detector model; "
                "pixel size and distance share mm, while wavelength is in angstrom. No distortion, "
                "polarization, solid-angle, or instrument calibration is invented. Cake output is "
                f"limited to {MAX_CAKE_CELLS:,} cells."
            ),
        )
        self._update_enabled()

    def _update_enabled(self) -> None:
        calibrated = self.mode.currentData() != "raw"
        for widget in (
            self.center_x,
            self.center_y,
            self.pixel_x,
            self.pixel_y,
            self.distance,
            self.wavelength,
            self.radial_bins,
        ):
            widget.setEnabled(calibrated)
        self.radial_unit.setEnabled(self.mode.currentData() == "radial")
        self.chi_bins.setEnabled(self.mode.currentData() == "cake")
        self._update_cake_budget()

    def _update_cake_budget(self, *_args: object) -> None:
        """Keep cake bins within the dense result budget without capping 1-D plots."""

        if self._updating_bin_budget:
            return
        self._updating_bin_budget = True
        try:
            # Restore the full per-axis range whenever the chi axis is not
            # active.  This is what makes 100k-bin radial plots available.
            self.radial_bins.setMaximum(_MAX_RADIAL_BINS)
            self.chi_bins.setMaximum(_MAX_CAKE_BINS_PER_AXIS)
            if self.mode.currentData() != "cake":
                return

            radial = self.radial_bins.value()
            chi = self.chi_bins.value()
            sender = self.sender()

            # Preserve the axis the user just changed where possible by
            # reducing the other axis first.  If the requested axis alone is
            # too large, clamp it to the largest value compatible with the
            # minimum of the other axis.
            if radial * chi > MAX_CAKE_CELLS:
                if sender is self.radial_bins:
                    chi = min(chi, MAX_CAKE_CELLS // radial)
                    if chi < _MIN_CAKE_BINS:
                        radial = min(radial, MAX_CAKE_CELLS // _MIN_CAKE_BINS)
                        chi = min(chi, MAX_CAKE_CELLS // radial)
                elif sender is self.chi_bins:
                    radial = min(radial, MAX_CAKE_CELLS // chi)
                    if radial < _MIN_CAKE_BINS:
                        chi = min(chi, MAX_CAKE_CELLS // _MIN_CAKE_BINS)
                        radial = min(radial, MAX_CAKE_CELLS // chi)
                else:
                    # On entering cake mode there is no user-changed axis;
                    # retain the current radial resolution and reduce chi.
                    chi = min(chi, MAX_CAKE_CELLS // radial)
                    if chi < _MIN_CAKE_BINS:
                        radial = min(radial, MAX_CAKE_CELLS // _MIN_CAKE_BINS)
                        chi = min(chi, MAX_CAKE_CELLS // radial)

                self.radial_bins.setValue(max(_MIN_CAKE_BINS, radial))
                self.chi_bins.setValue(max(_MIN_CAKE_BINS, chi))
                radial = self.radial_bins.value()
                chi = self.chi_bins.value()

            # Couple the maxima to the current opposite-axis values.  These
            # maxima are only a UI guard; generate_cake remains the fail-closed
            # backend authority for callers outside this dialog.
            self.radial_bins.setMaximum(
                min(_MAX_CAKE_BINS_PER_AXIS, MAX_CAKE_CELLS // chi)
            )
            self.chi_bins.setMaximum(
                min(_MAX_CAKE_BINS_PER_AXIS, MAX_CAKE_CELLS // radial)
            )
        finally:
            self._updating_bin_budget = False

    def parameters(self) -> dict[str, object]:
        mode = self.mode.currentData()
        radial_bins = self.radial_bins.value()
        chi_bins = self.chi_bins.value()
        if mode == "cake" and radial_bins * chi_bins > MAX_CAKE_CELLS:
            raise ValueError(
                "cake output exceeds the cell limit: "
                f"{radial_bins:,} x {chi_bins:,} > {MAX_CAKE_CELLS:,}"
            )
        return {
            "mode": mode,
            "center": (self.center_x.value(), self.center_y.value()),
            "pixel_size": (self.pixel_x.value(), self.pixel_y.value()),
            "distance": self.distance.value(),
            "wavelength": self.wavelength.value(),
            "radial_unit": self.radial_unit.currentData(),
            "radial_bins": radial_bins,
            "chi_bins": chi_bins,
        }


class DerivedAnalysisDialog(_AcceptedDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Peak-width analysis")
        form = QFormLayout()
        self.kind = _combo(
            [("Scherrer sizes", "scherrer"), ("Williamson-Hall", "williamson_hall")]
        )
        self.wavelength = _double(1.0e-9, 1.0e6, 0.15406)
        self.wavelength_unit = _combo(
            [("nm", "nm"), ("angstrom", "angstrom"), ("pm", "pm")]
        )
        self.shape_factor = _double(1.0e-6, 10.0, 0.9)
        self.instrument_fwhm = _double(0.0, 180.0, 0.0)
        self.output_unit = _combo(
            [("nm", "nm"), ("angstrom", "angstrom"), ("um", "um")]
        )
        form.addRow("Analysis", self.kind)
        form.addRow("Wavelength", self.wavelength)
        form.addRow("Wavelength unit", self.wavelength_unit)
        form.addRow("Shape factor K", self.shape_factor)
        form.addRow("Instrument FWHM (deg)", self.instrument_fwhm)
        form.addRow("Size output unit", self.output_unit)
        self._finish(
            form,
            help_text=(
                "Input CSV must explicitly contain 2theta and FWHM in degrees. Instrument broadening "
                "is removed in quadrature. The result has no uncertainty unless the source supplies one."
            ),
        )

    def parameters(self) -> dict[str, object]:
        return {
            "kind": self.kind.currentData(),
            "wavelength": self.wavelength.value(),
            "wavelength_unit": self.wavelength_unit.currentData(),
            "k": self.shape_factor.value(),
            "instrument_fwhm": self.instrument_fwhm.value(),
            "instrument_fwhm_unit": "deg",
            "output_unit": self.output_unit.currentData(),
        }


class MapImportDialog(_AcceptedDialog):
    def __init__(self, kind: str, parent=None) -> None:
        super().__init__(parent)
        self.kind = kind
        self.setWindowTitle("Map axis metadata")
        form = QFormLayout()
        if kind == "pole_figure":
            defaults = ("phi", "chi", "deg", "deg", "Intensity")
        else:
            defaults = ("q_parallel", "q_perp", "A^-1", "A^-1", "Intensity")
        self.x_label = QLineEdit(defaults[0])
        self.y_label = QLineEdit(defaults[1])
        self.x_unit = QLineEdit(defaults[2])
        self.y_unit = QLineEdit(defaults[3])
        self.intensity_label = QLineEdit(defaults[4])
        self.intensity_unit = QLineEdit("a.u.")
        form.addRow("Horizontal label", self.x_label)
        form.addRow("Vertical label", self.y_label)
        form.addRow("Horizontal unit", self.x_unit)
        form.addRow("Vertical unit", self.y_unit)
        form.addRow("Intensity label", self.intensity_label)
        form.addRow("Intensity unit", self.intensity_unit)
        self._finish(
            form,
            help_text=(
                "Axis units are recorded exactly as declared. XRDViz does not infer reciprocal-space "
                "units or convert raw goniometer coordinates."
            ),
        )

    def parameters(self) -> dict[str, object]:
        labels = {
            "x": self.x_label.text().strip(),
            "y": self.y_label.text().strip(),
            "intensity": self.intensity_label.text().strip(),
        }
        units = {
            "x": self.x_unit.text().strip(),
            "y": self.y_unit.text().strip(),
            "intensity": self.intensity_unit.text().strip(),
        }
        if not labels["x"] or not labels["y"] or not labels["intensity"]:
            raise ValueError("Map axis and intensity labels must not be empty")
        if not units["x"] or not units["y"]:
            raise ValueError("Map horizontal and vertical units must be declared")
        return {"labels": labels, "units": units}


class AnnotationDialog(_AcceptedDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Add plot annotation")
        form = QFormLayout()
        self.x_value = _double(-1.0e12, 1.0e12, 0.0)
        self.text = QLineEdit()
        form.addRow("Display x", self.x_value)
        form.addRow("Label", self.text)
        self._finish(
            form, help_text="The x value is interpreted in the current display axis."
        )

    def parameters(self) -> dict[str, object]:
        return {"x": self.x_value.value(), "text": self.text.text().strip()}
