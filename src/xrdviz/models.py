from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


AXIS_KINDS = {"two_theta", "d", "q"}

OKABE_ITO = [
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#000000",
    "#F0E442",
]


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
    return r"$Q$ (A$^{-1}$)"


@dataclass(slots=True)
class SpectrumLayer:
    name: str
    x: list[float]
    y: list[float]
    axis_kind: str = "two_theta"
    color: str = "#0072B2"
    visible: bool = True
    offset: float = 0.0
    linewidth: float = 1.0
    source_path: str = ""
    raw_x: list[float] = field(default_factory=list)
    raw_y: list[float] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    removed_rows: int = 0
    order: int = 0

    def __post_init__(self) -> None:
        self.axis_kind = normalize_axis_kind(self.axis_kind)
        self.x = [float(value) for value in self.x]
        self.y = [float(value) for value in self.y]
        self.raw_x = [float(value) for value in self.raw_x] if self.raw_x else list(self.x)
        self.raw_y = [float(value) for value in self.raw_y] if self.raw_y else list(self.y)
        if len(self.x) != len(self.y):
            raise ValueError("Spectrum x and y arrays must have the same length")
        if len(self.raw_x) != len(self.raw_y):
            raise ValueError("Raw spectrum x and y arrays must have the same length")


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
    color: str = "#D55E00"
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
    figure_width_in: float = 3.50394
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

    def __post_init__(self) -> None:
        self.x_axis = normalize_axis_kind(self.x_axis)
        if self.energy_kev <= 0:
            raise ValueError("Energy must be positive")
        if self.log_epsilon <= 0:
            raise ValueError("Log epsilon must be positive")
        if self.figure_width_in <= 0 or self.figure_height_in <= 0:
            raise ValueError("Figure dimensions must be positive")
        if self.dpi <= 0:
            raise ValueError("DPI must be positive")
        if self.x_min is not None and self.x_max is not None and self.x_min >= self.x_max:
            raise ValueError("x_min must be smaller than x_max")


@dataclass(slots=True)
class ProjectState:
    spectra: list[SpectrumLayer] = field(default_factory=list)
    phases: list[PhaseLayer] = field(default_factory=list)
    settings: PlotSettings = field(default_factory=PlotSettings)


def project_to_dict(state: ProjectState) -> dict[str, Any]:
    return asdict(state)


def project_from_dict(data: dict[str, Any]) -> ProjectState:
    settings = PlotSettings(**data.get("settings", {}))
    spectra = [SpectrumLayer(**item) for item in data.get("spectra", [])]
    phases = []
    for item in data.get("phases", []):
        peaks = [PhasePeak(**peak) for peak in item.get("peaks", [])]
        phase_data = dict(item)
        phase_data["peaks"] = peaks
        phases.append(PhaseLayer(**phase_data))
    return ProjectState(spectra=spectra, phases=phases, settings=settings)
