from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from xrdviz.analysis import DerivedPlot
    from xrdviz.fit import PatternFit
    from xrdviz.maps import MapData


AXIS_KINDS = {"two_theta", "d", "q"}
VIEW_MODES = {
    "overlay",
    "stack",
    "gradient_stack",
    "heatmap",
    "refinement",
    "small_multiples",
    "map",
    "derived",
}
BATCH_FIELDS = {"order", "frame", "time", "temperature", "color_value"}
UNCERTAINTY_MODES = {"none", "band", "bars"}
LEGEND_LOCATIONS = {
    "upper right",
    "upper left",
    "lower right",
    "lower left",
    "best",
    "outside right",
    "none",
}

PUBLICATION_PALETTE = [
    "#45A7E6",
    "#D62F53",
    "#2B9C8F",
    "#7A5CC7",
    "#E2A23A",
    "#286FB7",
    "#B05A7A",
    "#555555",
]
OKABE_ITO = PUBLICATION_PALETTE
PLOT_TEXT_COLOR = "#2C2C2C"
PLOT_AXIS_COLOR = "#3F3F3F"
# Muted guide/grid elements retain the original neutral publication tone.
PLOT_MUTED_COLOR = "#9A9A9A"


def normalize_axis_kind(axis_kind: str) -> str:
    aliases = {
        "2theta": "two_theta",
        "2_theta": "two_theta",
        "two-theta": "two_theta",
        "two_theta": "two_theta",
        "theta2": "two_theta",
        "d": "d",
        "d-spacing": "d",
        "d_spacing": "d",
        "q": "q",
        "Q": "q",
    }
    key = axis_kind.strip()
    normalized = aliases.get(key, aliases.get(key.lower()))
    if normalized not in AXIS_KINDS:
        raise ValueError(f"Unsupported axis kind: {axis_kind!r}")
    return normalized


def default_axis_label(axis_kind: str) -> str:
    axis_kind = normalize_axis_kind(axis_kind)
    if axis_kind == "two_theta":
        return r"2$\theta$ (deg)"
    if axis_kind == "d":
        return r"$d$-spacing ($\AA$)"
    return r"$Q$ ($\AA^{-1}$)"


@dataclass(slots=True)
class SpectrumLayer:
    name: str
    x: list[float]
    y: list[float]
    axis_kind: str = "two_theta"
    color: str = PUBLICATION_PALETTE[0]
    visible: bool = True
    offset: float = 0.0
    linewidth: float = 1.0
    source_path: str = ""
    raw_x: list[float] = field(default_factory=list)
    raw_y: list[float] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    removed_rows: int = 0
    order: int = 0
    frame_index: int | None = None
    time_s: float | None = None
    temperature: float | None = None
    temperature_unit: str = ""
    group: str = ""
    color_value: float | None = None
    # Appended after the legacy fields to preserve positional construction.
    y_error: list[float] = field(default_factory=list)
    wavelength_angstrom: float | None = None

    def __post_init__(self) -> None:
        self.axis_kind = normalize_axis_kind(self.axis_kind)
        self.x = [float(value) for value in self.x]
        self.y = [float(value) for value in self.y]
        self.raw_x = [float(value) for value in self.raw_x] if self.raw_x else list(self.x)
        self.raw_y = [float(value) for value in self.raw_y] if self.raw_y else list(self.y)
        self.y_error = [float(value) for value in self.y_error]
        self.wavelength_angstrom = (
            None if self.wavelength_angstrom is None else float(self.wavelength_angstrom)
        )
        if len(self.x) != len(self.y):
            raise ValueError("Spectrum x and y arrays must have the same length")
        if len(self.raw_x) != len(self.raw_y):
            raise ValueError("Raw spectrum x and y arrays must have the same length")
        if self.y_error and len(self.y_error) != len(self.y):
            raise ValueError("Spectrum uncertainty values must match the spectrum length")
        if any(not math.isfinite(value) or value < 0 for value in self.y_error):
            raise ValueError("Spectrum uncertainty values must be finite and non-negative")
        if self.wavelength_angstrom is not None and (
            not math.isfinite(self.wavelength_angstrom) or self.wavelength_angstrom <= 0
        ):
            raise ValueError("Spectrum wavelength must be finite and positive")
        self.order = int(self.order)
        self.frame_index = None if self.frame_index is None else int(self.frame_index)
        self.time_s = None if self.time_s is None else float(self.time_s)
        self.temperature = None if self.temperature is None else float(self.temperature)
        self.temperature_unit = str(self.temperature_unit or "")
        self.group = str(self.group or "")
        self.color_value = None if self.color_value is None else float(self.color_value)


@dataclass(slots=True)
class PhasePeak:
    two_theta: float
    intensity: float
    hkl: str = ""
    label: str = ""
    source_axis: str = "two_theta"
    relative_intensity: float | None = None

    def __post_init__(self) -> None:
        self.source_axis = normalize_axis_kind(self.source_axis)
        self.two_theta = float(self.two_theta)
        self.intensity = float(self.intensity)
        if self.relative_intensity is not None:
            self.relative_intensity = float(self.relative_intensity)


@dataclass(slots=True)
class PhaseLayer:
    name: str
    source_path: str
    color: str = PUBLICATION_PALETTE[1]
    peaks: list[PhasePeak] = field(default_factory=list)
    visible: bool = True
    tick_height: float = 0.8
    source_type: str = "cif"
    source_axis: str = "two_theta"
    phase: str = ""
    card_id: str = ""
    marker_shape: str = "line"
    show_guides: bool = False
    label_policy: str = "none"
    reference_lattice_a: float | None = None
    lattice_a: float | None = None
    auto_calibrated: bool = False
    calibration_confidence: str = ""
    calibration_error: float | None = None
    calibration_notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.source_axis = normalize_axis_kind(self.source_axis)
        if not self.phase:
            self.phase = self.name
        if self.reference_lattice_a is not None:
            self.reference_lattice_a = float(self.reference_lattice_a)
        if self.lattice_a is not None:
            self.lattice_a = float(self.lattice_a)
        if self.reference_lattice_a is not None and self.reference_lattice_a <= 0:
            raise ValueError("reference_lattice_a must be positive")
        if self.lattice_a is not None and self.lattice_a <= 0:
            raise ValueError("lattice_a must be positive")
        if self.calibration_error is not None:
            self.calibration_error = float(self.calibration_error)
        self.calibration_notes = [str(note) for note in self.calibration_notes]


@dataclass(slots=True)
class PlotAnnotation:
    """A user-declared vertical guide in the current display coordinate."""

    x: float
    text: str
    color: str = PLOT_TEXT_COLOR
    y_fraction: float = 0.92
    show_line: bool = True

    def __post_init__(self) -> None:
        self.x = float(self.x)
        self.y_fraction = float(self.y_fraction)
        if not math.isfinite(self.x):
            raise ValueError("Annotation x must be finite")
        if not math.isfinite(self.y_fraction) or not 0.0 <= self.y_fraction <= 1.0:
            raise ValueError("Annotation y_fraction must be in [0, 1]")
        self.text = str(self.text).strip()
        if not self.text:
            raise ValueError("Annotation text must not be empty")
        self.color = str(self.color or PLOT_TEXT_COLOR)
        self.show_line = bool(self.show_line)


@dataclass(slots=True)
class PlotSettings:
    x_axis: str = "two_theta"
    energy_kev: float = 8.0478
    x_label: str = r"2$\theta$ (deg)"
    y_label: str = "Intensity (a.u.)"
    panel_title: str = ""
    x_min: float | None = None
    x_max: float | None = None
    normalize: bool = True
    log_scale: bool = False
    log_epsilon: float = 1e-9
    stack_enabled: bool = False
    stack_spacing: float = 0.3
    figure_width_in: float = 89.0 / 25.4
    figure_height_in: float = 2.35
    dpi: int = 600
    font_family: str = "Arial"
    font_size: float = 7.0
    axis_label_size: float = 7.0
    tick_label_size: float = 6.0
    line_width: float = 0.75
    bragg_band_height: float = 0.16
    show_legend: bool = True
    direct_labels: bool = False
    show_phase_legend: bool = True
    show_y_tick_labels: bool = False
    view_mode: str = "overlay"
    sort_by: str = "frame"
    color_by: str = "frame"
    # ``cividis`` is the default continuous map for quantitative views.
    # ``blue_rose`` remains available as the legacy publication palette.
    colormap: str = "cividis"
    show_colorbar: bool = False
    show_every_n: int = 1
    heatmap_points: int = 600
    legend_location: str = "best"
    template_name: str = "nature_single"
    margin_left: float = 0.16
    margin_right: float = 0.98
    margin_top: float = 0.96
    margin_bottom: float = 0.16
    # New options are appended after all legacy fields to preserve positional APIs.
    uncertainty_mode: str = "none"
    uncertainty_alpha: float = 0.22
    errorbar_stride: int = 1
    show_fit_components: bool = True
    show_fit_background: bool = True
    show_fit_metrics: bool = True
    small_multiples_columns: int = 2
    show_panel_labels: bool = True
    inset_enabled: bool = False
    inset_x_min: float | None = None
    inset_x_max: float | None = None

    def __post_init__(self) -> None:
        self.x_axis = normalize_axis_kind(self.x_axis)
        if not math.isfinite(float(self.energy_kev)) or self.energy_kev <= 0:
            raise ValueError("Energy must be positive")
        if not math.isfinite(float(self.log_epsilon)) or self.log_epsilon <= 0:
            raise ValueError("Log epsilon must be positive")
        if (
            not math.isfinite(float(self.figure_width_in))
            or not math.isfinite(float(self.figure_height_in))
            or self.figure_width_in <= 0
            or self.figure_height_in <= 0
        ):
            raise ValueError("Figure dimensions must be positive")
        if isinstance(self.dpi, bool):
            raise ValueError("DPI must be a positive integer")
        try:
            dpi_value = float(self.dpi)
        except (TypeError, ValueError) as exc:
            raise ValueError("DPI must be a positive integer") from exc
        if not math.isfinite(dpi_value) or dpi_value <= 0 or not dpi_value.is_integer():
            raise ValueError("DPI must be a positive integer")
        self.dpi = int(dpi_value)
        if self.x_min is not None and not math.isfinite(float(self.x_min)):
            raise ValueError("x_min must be finite")
        if self.x_max is not None and not math.isfinite(float(self.x_max)):
            raise ValueError("x_max must be finite")
        if self.x_min is not None and self.x_max is not None and self.x_min >= self.x_max:
            raise ValueError("x_min must be smaller than x_max")
        for field_name, value in (
            ("stack_spacing", self.stack_spacing),
            ("font_size", self.font_size),
            ("axis_label_size", self.axis_label_size),
            ("tick_label_size", self.tick_label_size),
            ("line_width", self.line_width),
            ("bragg_band_height", self.bragg_band_height),
        ):
            if not math.isfinite(float(value)):
                raise ValueError(f"{field_name} must be finite")
        if self.view_mode not in VIEW_MODES:
            raise ValueError(f"Unsupported view mode: {self.view_mode!r}")
        if self.uncertainty_mode not in UNCERTAINTY_MODES:
            raise ValueError(f"Unsupported uncertainty mode: {self.uncertainty_mode!r}")
        if not math.isfinite(float(self.uncertainty_alpha)) or not 0.0 < self.uncertainty_alpha <= 1.0:
            raise ValueError("uncertainty_alpha must be finite and in (0, 1]")
        errorbar_stride = float(self.errorbar_stride)
        if not math.isfinite(errorbar_stride) or not errorbar_stride.is_integer() or errorbar_stride < 1:
            raise ValueError("errorbar_stride must be a positive integer")
        self.errorbar_stride = int(errorbar_stride)
        if self.sort_by not in BATCH_FIELDS:
            raise ValueError(f"Unsupported sort field: {self.sort_by!r}")
        if self.color_by not in BATCH_FIELDS:
            raise ValueError(f"Unsupported color field: {self.color_by!r}")
        show_every_n = float(self.show_every_n)
        if not math.isfinite(show_every_n) or not show_every_n.is_integer() or show_every_n < 1:
            raise ValueError("show_every_n must be at least 1")
        self.show_every_n = int(show_every_n)
        heatmap_points = float(self.heatmap_points)
        if not math.isfinite(heatmap_points) or not heatmap_points.is_integer() or heatmap_points < 2:
            raise ValueError("heatmap_points must be at least 2")
        self.heatmap_points = int(heatmap_points)
        panel_columns = float(self.small_multiples_columns)
        if not math.isfinite(panel_columns) or not panel_columns.is_integer() or not 1 <= panel_columns <= 6:
            raise ValueError("small_multiples_columns must be an integer from 1 to 6")
        self.small_multiples_columns = int(panel_columns)
        if (self.inset_x_min is None) != (self.inset_x_max is None):
            raise ValueError("Inset x min and max must be provided together")
        if self.inset_enabled and self.inset_x_min is None:
            raise ValueError("Inset x min and max are required when the inset is enabled")
        if self.inset_x_min is not None and self.inset_x_max is not None:
            self.inset_x_min = float(self.inset_x_min)
            self.inset_x_max = float(self.inset_x_max)
            if not math.isfinite(self.inset_x_min) or not math.isfinite(self.inset_x_max):
                raise ValueError("Inset x range must be finite")
            if self.inset_x_min >= self.inset_x_max:
                raise ValueError("Inset x min must be smaller than inset x max")
        if self.legend_location not in LEGEND_LOCATIONS:
            raise ValueError(f"Unsupported legend location: {self.legend_location!r}")
        if not (0.0 <= self.margin_left < self.margin_right <= 1.0):
            raise ValueError("Horizontal margins must satisfy 0 <= left < right <= 1")
        if not (0.0 <= self.margin_bottom < self.margin_top <= 1.0):
            raise ValueError("Vertical margins must satisfy 0 <= bottom < top <= 1")


@dataclass(slots=True)
class ProjectState:
    spectra: list[SpectrumLayer] = field(default_factory=list)
    phases: list[PhaseLayer] = field(default_factory=list)
    settings: PlotSettings = field(default_factory=PlotSettings)
    # New analysis state is appended after all legacy fields for positional compatibility.
    fit: PatternFit | None = None
    map_data: MapData | None = None
    derived_plot: DerivedPlot | None = None
    annotations: list[PlotAnnotation] = field(default_factory=list)


def project_to_dict(state: ProjectState) -> dict[str, Any]:
    return {
        "spectra": [asdict(layer) for layer in state.spectra],
        "phases": [asdict(layer) for layer in state.phases],
        "fit": asdict(state.fit) if state.fit is not None else None,
        "map_data": state.map_data.to_dict() if state.map_data is not None else None,
        "derived_plot": state.derived_plot.to_dict() if state.derived_plot is not None else None,
        "annotations": [asdict(annotation) for annotation in state.annotations],
        "settings": asdict(state.settings),
    }


def project_from_dict(data: dict[str, Any]) -> ProjectState:
    settings = PlotSettings(**data.get("settings", {}))
    spectra = [SpectrumLayer(**item) for item in data.get("spectra", [])]
    phases = []
    for item in data.get("phases", []):
        peaks = [PhasePeak(**peak) for peak in item.get("peaks", [])]
        phase_data = dict(item)
        phase_data["peaks"] = peaks
        phases.append(PhaseLayer(**phase_data))
    fit = None
    fit_data = data.get("fit")
    if fit_data is not None:
        from xrdviz.fit import FitComponent, PatternFit

        parsed_fit = dict(fit_data)
        parsed_fit["components"] = [FitComponent(**component) for component in fit_data.get("components", [])]
        fit = PatternFit(**parsed_fit)
    map_data = None
    if data.get("map_data") is not None:
        from xrdviz.maps import MapData

        map_data = MapData.from_dict(data["map_data"])
    derived_plot = None
    if data.get("derived_plot") is not None:
        from xrdviz.analysis import DerivedPlot

        derived_plot = DerivedPlot.from_dict(data["derived_plot"])
    annotations = [PlotAnnotation(**item) for item in data.get("annotations", [])]
    return ProjectState(
        spectra=spectra,
        phases=phases,
        fit=fit,
        map_data=map_data,
        derived_plot=derived_plot,
        annotations=annotations,
        settings=settings,
    )
