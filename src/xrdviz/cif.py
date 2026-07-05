from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from xrdviz.axes import convert_x, wavelength_from_energy
from xrdviz.models import PhaseLayer, PhasePeak, PlotSettings


@dataclass(slots=True)
class DisplayPeak:
    x: float
    intensity: float
    hkl: str


def phase_peaks_for_axis(phase: PhaseLayer, settings: PlotSettings) -> list[DisplayPeak]:
    x_values = [phase_peak_position_for_axis(phase, peak, settings.x_axis, settings.energy_kev) for peak in phase.peaks]
    return [
        DisplayPeak(x=x_value, intensity=peak.intensity, hkl=peak.hkl)
        for x_value, peak in zip(x_values, phase.peaks)
    ]


def peak_position_for_axis(peak: PhasePeak, axis_kind: str, energy_kev: float) -> float:
    return convert_x([peak.two_theta], peak.source_axis, axis_kind, energy_kev)[0]


def phase_peak_position_for_axis(phase: PhaseLayer, peak: PhasePeak, axis_kind: str, energy_kev: float) -> float:
    base_d = peak_position_for_axis(peak, "d", energy_kev)
    scale = lattice_scale_for_phase(phase)
    if scale == 1.0:
        return peak_position_for_axis(peak, axis_kind, energy_kev)
    return convert_x([base_d * scale], "d", axis_kind, energy_kev)[0]


def lattice_scale_for_phase(phase: PhaseLayer) -> float:
    if phase.reference_lattice_a is None or phase.lattice_a is None:
        return 1.0
    if phase.reference_lattice_a <= 0 or phase.lattice_a <= 0:
        return 1.0
    return phase.lattice_a / phase.reference_lattice_a


def load_cif_phase(path: str | Path, *, energy_kev: float, color: str = "#D55E00") -> PhaseLayer:
    file_path = Path(path)
    peaks = calculate_cif_peaks(file_path, energy_kev=energy_kev)
    lattice_a = read_cif_lattice_a(file_path)
    return PhaseLayer(
        name=file_path.stem,
        source_path=str(file_path),
        color=color,
        peaks=peaks,
        reference_lattice_a=lattice_a,
        lattice_a=lattice_a,
    )


def read_cif_lattice_a(path: str | Path) -> float | None:
    structure = _read_structure(path)
    value = float(structure.lattice.a)
    return value if value > 0 else None


def calculate_cif_peaks(path: str | Path, *, energy_kev: float, two_theta_range: tuple[float, float] = (0.0, 120.0)) -> list[PhasePeak]:
    try:
        from pymatgen.analysis.diffraction.xrd import XRDCalculator
    except ImportError as exc:
        raise RuntimeError("pymatgen is required to calculate CIF Bragg positions") from exc

    structure = _read_structure(path)
    calculator = XRDCalculator(wavelength=wavelength_from_energy(energy_kev))
    pattern = calculator.get_pattern(structure, two_theta_range=two_theta_range)

    peaks: list[PhasePeak] = []
    for two_theta, intensity, hkl_group in zip(pattern.x, pattern.y, pattern.hkls):
        peaks.append(PhasePeak(float(two_theta), float(intensity), _format_hkl_group(hkl_group)))
    return peaks


def _read_structure(path: str | Path):
    try:
        from pymatgen.core import Structure
    except ImportError as exc:
        raise RuntimeError("pymatgen is required to calculate CIF Bragg positions") from exc

    return Structure.from_file(str(path))


def _format_hkl_group(hkl_group: object) -> str:
    if not isinstance(hkl_group, list) or not hkl_group:
        return ""
    labels: list[str] = []
    for item in hkl_group:
        if not isinstance(item, dict) or "hkl" not in item:
            continue
        labels.append("".join(str(index) for index in item["hkl"]))
    return ", ".join(labels)
