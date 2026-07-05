from __future__ import annotations

import math
import warnings
from collections.abc import Iterable

from xrdviz.models import normalize_axis_kind

KEV_ANGSTROM = 12.398419843


def wavelength_from_energy(energy_kev: float) -> float:
    if energy_kev <= 0:
        raise ValueError("Energy must be positive")
    return KEV_ANGSTROM / energy_kev


def convert_x(values: Iterable[float], from_axis: str, to_axis: str, energy_kev: float) -> list[float]:
    from_axis = normalize_axis_kind(from_axis)
    to_axis = normalize_axis_kind(to_axis)
    values = [float(value) for value in values]
    if from_axis == to_axis:
        return list(values)

    wavelength = wavelength_from_energy(energy_kev)
    d_values = [_to_d_spacing(value, from_axis, wavelength) for value in values]
    if to_axis == "d":
        return d_values
    return [_from_d_spacing(value, to_axis, wavelength) for value in d_values]


def _to_d_spacing(value: float, axis_kind: str, wavelength: float) -> float:
    if not math.isfinite(value):
        return math.nan
    if axis_kind == "d":
        return value if value > 0 else _invalid(value)
    if axis_kind == "q":
        return (2.0 * math.pi / value) if value > 0 else _invalid(value)

    theta_rad = math.radians(value / 2.0)
    sin_theta = math.sin(theta_rad)
    if sin_theta <= 0:
        return _invalid(value)
    return wavelength / (2.0 * sin_theta)


def _from_d_spacing(d_spacing: float, axis_kind: str, wavelength: float) -> float:
    if not math.isfinite(d_spacing) or d_spacing <= 0:
        return math.nan
    if axis_kind == "q":
        return 2.0 * math.pi / d_spacing

    ratio = wavelength / (2.0 * d_spacing)
    if ratio < 0.0 or ratio > 1.0:
        return _invalid(d_spacing)
    return math.degrees(2.0 * math.asin(ratio))


def _invalid(value: float) -> float:
    warnings.warn(
        f"Value {value!r} is outside valid diffraction geometry and was converted to NaN",
        RuntimeWarning,
        stacklevel=3,
    )
    return math.nan
