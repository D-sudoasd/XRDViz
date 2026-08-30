"""Small, auditable calculations derived from one-dimensional XRD data.

The functions in this module deliberately do not estimate uncertainties.  A
caller must provide a peak position and width (and, for Scherrer/Williamson--
Hall calculations, the wavelength and shape factor) explicitly.  Measurements
are checked at the boundary so malformed values are rejected instead of being
silently converted into plausible-looking materials parameters.

Angular widths are corrected for a Gaussian instrument contribution in
quadrature, ``beta = sqrt(beta_measured**2 - beta_instrument**2)``.  Widths are
then converted to radians before using the Scherrer or Williamson--Hall
equations.  Rocking-curve crossings and area use piecewise-linear interpolation
and the trapezoidal rule, respectively; no smoothing or model-dependent peak
fit is hidden in the result.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any


_ANGLE_UNITS: dict[str, tuple[str, float]] = {
    "deg": ("deg", math.pi / 180.0),
    "degree": ("deg", math.pi / 180.0),
    "degrees": ("deg", math.pi / 180.0),
    "°": ("deg", math.pi / 180.0),
    "rad": ("rad", 1.0),
    "radian": ("rad", 1.0),
    "radians": ("rad", 1.0),
}

# Internal length unit is nanometres.  The accepted aliases are intentionally
# finite and explicit: silently treating an unknown unit as nm would make the
# size result wrong by orders of magnitude.
_LENGTH_TO_NM: dict[str, tuple[str, float]] = {
    "nm": ("nm", 1.0),
    "nanometer": ("nm", 1.0),
    "nanometers": ("nm", 1.0),
    "nanometre": ("nm", 1.0),
    "nanometres": ("nm", 1.0),
    "å": ("angstrom", 0.1),
    "a": ("angstrom", 0.1),
    "ang": ("angstrom", 0.1),
    "angstrom": ("angstrom", 0.1),
    "angstroms": ("angstrom", 0.1),
    "ångström": ("angstrom", 0.1),
    "ångströms": ("angstrom", 0.1),
    "pm": ("pm", 0.001),
    "picometer": ("pm", 0.001),
    "picometers": ("pm", 0.001),
    "picometre": ("pm", 0.001),
    "picometres": ("pm", 0.001),
    "um": ("um", 1_000.0),
    "µm": ("um", 1_000.0),
    "μm": ("um", 1_000.0),
    "micrometer": ("um", 1_000.0),
    "micrometers": ("um", 1_000.0),
    "micrometre": ("um", 1_000.0),
    "micrometres": ("um", 1_000.0),
    "mm": ("mm", 1_000_000.0),
    "millimeter": ("mm", 1_000_000.0),
    "millimeters": ("mm", 1_000_000.0),
    "millimetre": ("mm", 1_000_000.0),
    "millimetres": ("mm", 1_000_000.0),
    "m": ("m", 1_000_000_000.0),
    "meter": ("m", 1_000_000_000.0),
    "meters": ("m", 1_000_000_000.0),
    "metre": ("m", 1_000_000_000.0),
    "metres": ("m", 1_000_000_000.0),
}


def _number(
    value: Any, name: str, *, positive: bool = False, nonnegative: bool = False
) -> float:
    """Return a finite real number, rejecting strings and booleans."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if positive and result <= 0.0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _unit_key(unit: Any, name: str) -> str:
    if not isinstance(unit, str) or not unit.strip():
        raise ValueError(f"{name} must be an explicit supported unit")
    return unit.strip().lower().replace(" ", "").replace("_", "").replace("-", "")


def _angle_unit(unit: Any, *, name: str = "angle_unit") -> tuple[str, float]:
    key = _unit_key(unit, name)
    try:
        return _ANGLE_UNITS[key]
    except KeyError as exc:
        supported = "deg or rad"
        raise ValueError(f"Unsupported {name} {unit!r}; expected {supported}") from exc


def _length_unit(unit: Any, *, name: str = "wavelength_unit") -> tuple[str, float]:
    key = _unit_key(unit, name)
    try:
        return _LENGTH_TO_NM[key]
    except KeyError as exc:
        raise ValueError(f"Unsupported {name} {unit!r}") from exc


def _length_to_nm(value: float, unit: Any, *, name: str) -> float:
    _canonical, factor = _length_unit(unit, name=name)
    converted = value * factor
    if not math.isfinite(converted) or converted <= 0.0:
        raise ValueError(f"{name} conversion must be finite and positive")
    return converted


def _nm_to_length(value_nm: float, unit: Any, *, name: str) -> float:
    _canonical, factor = _length_unit(unit, name=name)
    converted = value_nm / factor
    if not math.isfinite(converted) or converted <= 0.0:
        raise ValueError(f"{name} conversion must be finite and positive")
    return converted


@dataclass(frozen=True, slots=True, init=False)
class PeakMeasurement:
    """A measured peak centre and FWHM in one explicit angular unit.

    ``two_theta`` and ``fwhm`` are stored in ``angle_unit``.  The keyword
    aliases ``position``, ``two_theta_deg`` and ``fwhm_deg`` are accepted for
    callers that use those common names; aliases cannot be mixed with a
    conflicting canonical value.  ``intensity`` is descriptive only and is not
    used by the derived calculations.
    """

    two_theta: float
    fwhm: float
    angle_unit: str
    intensity: float | None
    hkl: str

    def __init__(
        self,
        two_theta: Real | None = None,
        fwhm: Real | None = None,
        angle_unit: str = "deg",
        intensity: Real | None = None,
        hkl: str = "",
        *,
        position: Real | None = None,
        two_theta_deg: Real | None = None,
        fwhm_deg: Real | None = None,
        position_unit: str | None = None,
        unit: str | None = None,
    ) -> None:
        supplied_centres = [
            value is not None for value in (two_theta, position, two_theta_deg)
        ]
        if sum(supplied_centres) != 1:
            raise ValueError(
                "Provide exactly one of two_theta, position, or two_theta_deg"
            )
        supplied_widths = [value is not None for value in (fwhm, fwhm_deg)]
        if sum(supplied_widths) != 1:
            raise ValueError("Provide exactly one of fwhm or fwhm_deg")

        if position is not None:
            centre = position
        elif two_theta_deg is not None:
            centre = two_theta_deg
        else:
            centre = two_theta
        width = fwhm if fwhm is not None else fwhm_deg

        chosen_unit: Any = angle_unit
        if unit is not None:
            if (
                angle_unit != "deg"
                and _angle_unit(angle_unit)[0] != _angle_unit(unit)[0]
            ):
                raise ValueError("angle_unit and unit disagree")
            chosen_unit = unit
        if position_unit is not None:
            if (
                angle_unit != "deg"
                and _angle_unit(angle_unit)[0] != _angle_unit(position_unit)[0]
            ):
                raise ValueError("angle_unit and position_unit disagree")
            chosen_unit = position_unit
        if two_theta_deg is not None or fwhm_deg is not None:
            if chosen_unit != "deg" and _angle_unit(chosen_unit)[0] != "deg":
                raise ValueError("*_deg aliases require angle_unit='deg'")
            chosen_unit = "deg"

        canonical_unit, factor = _angle_unit(chosen_unit)
        centre_value = _number(centre, "two_theta")
        width_value = _number(width, "fwhm", positive=True)
        # A diffraction angle is conventionally in [0, 180) degrees.  Zero is
        # retained as a mathematically valid boundary; Scherrer still checks
        # the cosine denominator at calculation time.
        if not 0.0 <= centre_value * factor < math.pi:
            raise ValueError(
                "two_theta must be in [0, 180) degrees (or [0, pi) radians)"
            )

        intensity_value = (
            None
            if intensity is None
            else _number(intensity, "intensity", nonnegative=True)
        )
        if not isinstance(hkl, str):
            raise ValueError("hkl must be a string")

        object.__setattr__(self, "two_theta", centre_value)
        object.__setattr__(self, "fwhm", width_value)
        object.__setattr__(self, "angle_unit", canonical_unit)
        object.__setattr__(self, "intensity", intensity_value)
        object.__setattr__(self, "hkl", hkl)

    @property
    def position(self) -> float:
        """Alias for ``two_theta`` in the stored angular unit."""

        return self.two_theta

    @property
    def position_deg(self) -> float:
        return self.two_theta_deg

    @property
    def two_theta_deg(self) -> float:
        return self.two_theta * _angle_unit(self.angle_unit)[1] * 180.0 / math.pi

    @property
    def fwhm_deg(self) -> float:
        return self.fwhm * _angle_unit(self.angle_unit)[1] * 180.0 / math.pi

    @property
    def two_theta_rad(self) -> float:
        return self.two_theta * _angle_unit(self.angle_unit)[1]

    @property
    def fwhm_rad(self) -> float:
        return self.fwhm * _angle_unit(self.angle_unit)[1]


def remove_instrument_broadening(
    measured_fwhm: Real,
    instrument_fwhm: Real = 0.0,
    *,
    unit: str = "deg",
) -> float:
    """Remove a Gaussian instrument width in quadrature.

    The returned width has the same unit as the input.  A non-positive result
    is rejected because a zero/imaginary corrected width cannot yield a finite
    Scherrer or Williamson--Hall size.
    """

    _canonical, _factor = _angle_unit(unit)
    measured = _number(measured_fwhm, "measured_fwhm", positive=True)
    instrument = _number(instrument_fwhm, "instrument_fwhm", nonnegative=True)
    if instrument >= measured:
        raise ValueError("instrument_fwhm must be smaller than measured_fwhm")
    corrected = math.sqrt(measured * measured - instrument * instrument)
    if not math.isfinite(corrected) or corrected <= 0.0:
        raise ValueError("instrument-corrected FWHM must be finite and positive")
    return corrected


def _corrected_width_rad(
    measurement: PeakMeasurement,
    instrument_fwhm: Real = 0.0,
    *,
    instrument_unit: str | None = None,
) -> float:
    measured_rad = measurement.fwhm_rad
    if instrument_unit is None:
        instrument_rad = (
            _number(instrument_fwhm, "instrument_fwhm", nonnegative=True)
            * _angle_unit(measurement.angle_unit)[1]
        )
    else:
        instrument_value = _number(instrument_fwhm, "instrument_fwhm", nonnegative=True)
        _canonical, factor = _angle_unit(instrument_unit, name="instrument_fwhm_unit")
        instrument_rad = instrument_value * factor
    if instrument_rad >= measured_rad:
        raise ValueError("instrument_fwhm must be smaller than measured_fwhm")
    corrected = math.sqrt(measured_rad * measured_rad - instrument_rad * instrument_rad)
    if not math.isfinite(corrected) or corrected <= 0.0:
        raise ValueError("instrument-corrected FWHM must be finite and positive")
    return corrected


def _resolve_k(k: Real | None, shape_factor: Real | None) -> float:
    if k is None and shape_factor is None:
        raise ValueError("k (the Scherrer shape factor) must be supplied explicitly")
    if k is not None and shape_factor is not None:
        k_value = _number(k, "k", positive=True)
        shape_value = _number(shape_factor, "shape_factor", positive=True)
        if k_value != shape_value:
            raise ValueError("k and shape_factor disagree")
        return k_value
    return _number(k if k is not None else shape_factor, "k", positive=True)


def _coerce_measurement(
    peak: PeakMeasurement | Real | None,
    fwhm: Real | None,
    *,
    two_theta: Real | None,
    fwhm_corrected: Real | None,
    angle_unit: str,
) -> PeakMeasurement:
    if peak is not None and two_theta is not None:
        raise ValueError("Provide peak or two_theta, not both")
    if fwhm is not None and fwhm_corrected is not None:
        raise ValueError("Provide fwhm or fwhm_corrected, not both")
    width = fwhm if fwhm is not None else fwhm_corrected
    if isinstance(peak, PeakMeasurement):
        if two_theta is not None or width is not None:
            raise ValueError("A PeakMeasurement already contains two_theta and fwhm")
        return peak
    centre = two_theta if two_theta is not None else peak
    if centre is None:
        raise ValueError("A PeakMeasurement or two_theta value is required")
    if width is None:
        raise ValueError("fwhm (or fwhm_corrected) is required for numeric two_theta")
    return PeakMeasurement(centre, width, angle_unit)


def scherrer_crystallite_size(
    peak: PeakMeasurement | Real | None = None,
    fwhm: Real | None = None,
    *,
    k: Real | None = None,
    shape_factor: Real | None = None,
    wavelength: Real | None = None,
    wavelength_unit: str = "nm",
    instrument_fwhm: Real = 0.0,
    instrument_fwhm_deg: Real | None = None,
    instrument_fwhm_unit: str | None = None,
    fwhm_corrected: Real | None = None,
    two_theta: Real | None = None,
    angle_unit: str = "deg",
    output_unit: str | None = None,
) -> float:
    """Calculate Scherrer crystallite size.

    Parameters are explicit: ``k``/``shape_factor``, ``wavelength`` and the
    peak width are required.  ``instrument_fwhm`` is subtracted in quadrature
    from the measured FWHM.  Alternatively, pass ``fwhm_corrected`` (or use a
    measured FWHM with the default zero instrument width) to provide an already
    corrected width.  Numeric peak inputs use ``angle_unit``; a
    :class:`PeakMeasurement` carries its own unit.  The result defaults to the
    wavelength unit and can be requested in another supported length unit.
    """

    shape = _resolve_k(k, shape_factor)
    if wavelength is None:
        raise ValueError("wavelength must be supplied explicitly")
    wavelength_value = _number(wavelength, "wavelength", positive=True)
    canonical_wavelength_unit, _factor = _length_unit(wavelength_unit)
    wavelength_nm = _length_to_nm(wavelength_value, wavelength_unit, name="wavelength")
    result_unit = (
        canonical_wavelength_unit
        if output_unit is None
        else _length_unit(output_unit, name="output_unit")[0]
    )

    if instrument_fwhm_deg is not None:
        if instrument_fwhm != 0.0:
            raise ValueError(
                "instrument_fwhm and instrument_fwhm_deg cannot both be supplied"
            )
        instrument_fwhm = instrument_fwhm_deg
        if instrument_fwhm_unit is None:
            instrument_fwhm_unit = "deg"

    measurement = _coerce_measurement(
        peak,
        fwhm,
        two_theta=two_theta,
        fwhm_corrected=fwhm_corrected,
        angle_unit=angle_unit,
    )
    corrected_beta = _corrected_width_rad(
        measurement, instrument_fwhm, instrument_unit=instrument_fwhm_unit
    )
    theta = measurement.two_theta_rad / 2.0
    denominator = corrected_beta * math.cos(theta)
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("Scherrer denominator must be finite and positive")
    size_nm = shape * wavelength_nm / denominator
    if not math.isfinite(size_nm) or size_nm <= 0.0:
        raise ValueError("Scherrer crystallite size must be finite and positive")
    return _nm_to_length(size_nm, result_unit, name="output_unit")


def scherrer_size(*args: Any, **kwargs: Any) -> float:
    """Compatibility alias for :func:`scherrer_crystallite_size`."""

    return scherrer_crystallite_size(*args, **kwargs)


def calculate_scherrer_size(*args: Any, **kwargs: Any) -> float:
    """Compatibility alias for :func:`scherrer_crystallite_size`."""

    return scherrer_crystallite_size(*args, **kwargs)


@dataclass(frozen=True, slots=True)
class WilliamsonHallResult:
    """Result of an unweighted Williamson--Hall linear regression.

    ``microstrain`` is the dimensionless slope (not multiplied by 10^6), and
    ``crystallite_size`` is expressed in ``size_unit``.  No standard errors or
    confidence intervals are inferred from the supplied points.
    """

    slope: float
    intercept: float
    microstrain: float
    crystallite_size: float
    r_squared: float
    n_points: int
    size_unit: str

    @property
    def size(self) -> float:
        return self.crystallite_size

    @property
    def r2(self) -> float:
        return self.r_squared

    @property
    def point_count(self) -> int:
        return self.n_points


def _instrument_values(
    instrument_fwhm: Real | Sequence[Real], count: int
) -> list[float]:
    if isinstance(instrument_fwhm, bool) or isinstance(instrument_fwhm, Real):
        value = _number(instrument_fwhm, "instrument_fwhm", nonnegative=True)
        return [value] * count
    if isinstance(instrument_fwhm, (str, bytes)):
        raise ValueError("instrument_fwhm must be a number or a sequence of numbers")
    try:
        values = list(instrument_fwhm)
    except TypeError as exc:
        raise ValueError(
            "instrument_fwhm must be a number or a sequence of numbers"
        ) from exc
    if len(values) != count:
        raise ValueError("instrument_fwhm sequence must match the number of peaks")
    return [_number(value, "instrument_fwhm", nonnegative=True) for value in values]


def _instrument_units(unit: str | Sequence[str] | None, count: int) -> list[str | None]:
    if unit is None or isinstance(unit, str):
        return [unit] * count
    if isinstance(unit, bytes):
        raise ValueError(
            "instrument_fwhm_unit must be a unit string or a sequence of unit strings"
        )
    try:
        units = list(unit)
    except TypeError as exc:
        raise ValueError(
            "instrument_fwhm_unit must be a unit string or a sequence of unit strings"
        ) from exc
    if len(units) != count:
        raise ValueError("instrument_fwhm_unit sequence must match the number of peaks")
    return units


def williamson_hall_fit(
    peaks: Iterable[PeakMeasurement],
    *,
    k: Real | None = None,
    shape_factor: Real | None = None,
    wavelength: Real | None = None,
    wavelength_unit: str = "nm",
    instrument_fwhm: Real | Sequence[Real] = 0.0,
    instrument_fwhm_deg: Real | Sequence[Real] | None = None,
    instrument_fwhm_unit: str | Sequence[str] | None = None,
    output_unit: str | None = None,
) -> WilliamsonHallResult:
    """Fit ``beta*cos(theta) = slope*(4*sin(theta)) + intercept``.

    The fit is ordinary unweighted least squares because no measurement
    uncertainties are fabricated.  At least two peaks with distinct positions
    are required.  Corrected widths use the same Gaussian quadrature rule as
    :func:`scherrer_crystallite_size`.
    """

    peak_list = list(peaks)
    if len(peak_list) < 2:
        raise ValueError("Williamson-Hall fit requires at least two peak measurements")
    if any(not isinstance(peak, PeakMeasurement) for peak in peak_list):
        raise ValueError("Williamson-Hall peaks must be PeakMeasurement instances")
    shape = _resolve_k(k, shape_factor)
    if wavelength is None:
        raise ValueError("wavelength must be supplied explicitly")
    wavelength_value = _number(wavelength, "wavelength", positive=True)
    canonical_wavelength_unit, _factor = _length_unit(wavelength_unit)
    wavelength_nm = _length_to_nm(wavelength_value, wavelength_unit, name="wavelength")
    result_unit = (
        canonical_wavelength_unit
        if output_unit is None
        else _length_unit(output_unit, name="output_unit")[0]
    )

    if instrument_fwhm_deg is not None:
        if instrument_fwhm != 0.0:
            raise ValueError(
                "instrument_fwhm and instrument_fwhm_deg cannot both be supplied"
            )
        instrument_fwhm = instrument_fwhm_deg
        if instrument_fwhm_unit is None:
            instrument_fwhm_unit = "deg"

    widths = _instrument_values(instrument_fwhm, len(peak_list))
    units = _instrument_units(instrument_fwhm_unit, len(peak_list))
    x_values: list[float] = []
    y_values: list[float] = []
    for peak, width, unit in zip(peak_list, widths, units):
        theta = peak.two_theta_rad / 2.0
        corrected_beta = _corrected_width_rad(peak, width, instrument_unit=unit)
        x_value = 4.0 * math.sin(theta)
        y_value = corrected_beta * math.cos(theta)
        if not math.isfinite(x_value) or not math.isfinite(y_value) or y_value <= 0.0:
            raise ValueError(
                "Williamson-Hall transformed values must be finite and positive"
            )
        x_values.append(x_value)
        y_values.append(y_value)

    mean_x = sum(x_values) / len(x_values)
    mean_y = sum(y_values) / len(y_values)
    sxx = sum((value - mean_x) ** 2 for value in x_values)
    if not math.isfinite(sxx) or sxx <= 1e-30:
        raise ValueError("Williamson-Hall fit requires distinct peak positions")
    sxy = sum(
        (x_value - mean_x) * (y_value - mean_y)
        for x_value, y_value in zip(x_values, y_values)
    )
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    if not math.isfinite(slope) or not math.isfinite(intercept):
        raise ValueError("Williamson-Hall regression returned non-finite coefficients")
    if slope < 0.0:
        raise ValueError("Williamson-Hall slope implies a negative microstrain")
    if intercept <= 0.0:
        raise ValueError("Williamson-Hall intercept must be positive to calculate size")

    predictions = [slope * value + intercept for value in x_values]
    residual_sum = sum(
        (observed - predicted) ** 2
        for observed, predicted in zip(y_values, predictions)
    )
    total_sum = sum((observed - mean_y) ** 2 for observed in y_values)
    if total_sum <= 1e-30:
        r_squared = 1.0 if residual_sum <= 1e-30 else 0.0
    else:
        r_squared = 1.0 - residual_sum / total_sum
    r_squared = min(1.0, max(0.0, r_squared))
    size_nm = shape * wavelength_nm / intercept
    if not math.isfinite(size_nm) or size_nm <= 0.0:
        raise ValueError("Williamson-Hall crystallite size must be finite and positive")
    size_value = _nm_to_length(size_nm, result_unit, name="output_unit")
    return WilliamsonHallResult(
        slope=slope,
        intercept=intercept,
        microstrain=slope,
        crystallite_size=size_value,
        r_squared=r_squared,
        n_points=len(peak_list),
        size_unit=result_unit,
    )


def williamson_hall(*args: Any, **kwargs: Any) -> WilliamsonHallResult:
    """Compatibility alias for :func:`williamson_hall_fit`."""

    return williamson_hall_fit(*args, **kwargs)


@dataclass(frozen=True, slots=True)
class RockingCurveMetrics:
    """Peak position, FWHM and full-range integrated intensity."""

    peak_position: float
    fwhm: float
    integrated_intensity: float
    x_unit: str

    @property
    def position(self) -> float:
        return self.peak_position

    @property
    def integral(self) -> float:
        return self.integrated_intensity

    @property
    def area(self) -> float:
        return self.integrated_intensity


def _crossing(
    x_left: float, y_left: float, x_right: float, y_right: float, level: float
) -> float:
    if y_right == y_left:
        return (x_left + x_right) / 2.0
    fraction = (level - y_left) / (y_right - y_left)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("half-height crossing lies outside an interpolation segment")
    return x_left + fraction * (x_right - x_left)


def rocking_curve_metrics(
    x_values: Iterable[Real],
    intensity_values: Iterable[Real],
    *,
    x_unit: str = "deg",
    unit: str | None = None,
    interpolation: str = "linear",
) -> RockingCurveMetrics:
    """Return metrics from a rocking curve using transparent linear rules.

    The peak is the highest supplied sample.  FWHM is measured between the
    nearest piecewise-linear crossings of half that absolute peak height.  The
    integrated intensity is the trapezoidal area over the complete supplied
    domain, in intensity times ``x_unit``.  A baseline is not invented or
    subtracted.
    """

    if unit is not None:
        if x_unit != "deg" and _angle_unit(x_unit)[0] != _angle_unit(unit)[0]:
            raise ValueError("x_unit and unit disagree")
        x_unit = unit
    canonical_unit, _factor = _angle_unit(x_unit, name="x_unit")
    if interpolation.strip().lower() not in {
        "linear",
        "piecewise_linear",
        "piecewise-linear",
    }:
        raise ValueError("Only explainable piecewise-linear interpolation is supported")

    try:
        xs = [_number(value, "x") for value in x_values]
        ys = [
            _number(value, "intensity", nonnegative=True) for value in intensity_values
        ]
    except TypeError as exc:
        raise ValueError(
            "x_values and intensity_values must be finite numeric iterables"
        ) from exc
    if len(xs) != len(ys):
        raise ValueError("x_values and intensity_values must have the same length")
    if len(xs) < 3:
        raise ValueError("Rocking-curve metrics require at least three points")

    differences = [right - left for left, right in zip(xs, xs[1:])]
    if all(delta < 0.0 for delta in differences):
        xs.reverse()
        ys.reverse()
        differences = [-delta for delta in differences]
    elif not all(delta > 0.0 for delta in differences):
        raise ValueError("x_values must be strictly monotonic")

    peak_index = max(range(len(ys)), key=ys.__getitem__)
    peak_height = ys[peak_index]
    if peak_height <= 0.0:
        raise ValueError("Rocking curve must contain a positive peak")
    half_height = peak_height / 2.0

    left_crossing: float | None = None
    for index in range(peak_index, 0, -1):
        if ys[index - 1] <= half_height <= ys[index]:
            left_crossing = _crossing(
                xs[index - 1], ys[index - 1], xs[index], ys[index], half_height
            )
            break
    right_crossing: float | None = None
    for index in range(peak_index, len(xs) - 1):
        if ys[index + 1] <= half_height <= ys[index]:
            right_crossing = _crossing(
                xs[index], ys[index], xs[index + 1], ys[index + 1], half_height
            )
            break
    if left_crossing is None or right_crossing is None:
        raise ValueError("Rocking curve does not cross half peak height on both sides")
    width = right_crossing - left_crossing
    if not math.isfinite(width) or width <= 0.0:
        raise ValueError("Rocking-curve FWHM must be finite and positive")

    area = sum(
        (left_y + right_y) * (right_x - left_x) / 2.0
        for left_x, right_x, left_y, right_y in zip(xs, xs[1:], ys, ys[1:])
    )
    if not math.isfinite(area) or area <= 0.0:
        raise ValueError(
            "Rocking-curve integrated intensity must be finite and positive"
        )
    return RockingCurveMetrics(
        peak_position=xs[peak_index],
        fwhm=width,
        integrated_intensity=area,
        x_unit=canonical_unit,
    )


__all__ = [
    "PeakMeasurement",
    "RockingCurveMetrics",
    "WilliamsonHallResult",
    "calculate_scherrer_size",
    "remove_instrument_broadening",
    "rocking_curve_metrics",
    "scherrer_crystallite_size",
    "scherrer_size",
    "williamson_hall",
    "williamson_hall_fit",
]
