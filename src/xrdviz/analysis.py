"""Auditable plot data for common XRD-derived analyses.

This module is intentionally a small boundary layer around :mod:`xrdviz.derived`.
It imports only explicitly named CSV columns, keeps the units supplied by the
caller visible in labels/metrics, and does not manufacture uncertainty values.
``DerivedPlot`` contains plain JSON-compatible data so a caller can persist a
derived plot without serialising a Matplotlib object or re-running a fit.
"""

from __future__ import annotations

import csv
import json
import math
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path
from typing import Any, TextIO

from .derived import (
    PeakMeasurement,
    rocking_curve_metrics,
    scherrer_crystallite_size,
    williamson_hall_fit,
)


_PLOT_KINDS = {"scherrer", "williamson_hall", "rocking_curve"}
_ANGLE_UNITS: dict[str, str] = {
    "deg": "deg",
    "degree": "deg",
    "degrees": "deg",
    "rad": "rad",
    "radian": "rad",
    "radians": "rad",
    "°": "deg",
}
_ANGLE_TO_RAD = {"deg": math.pi / 180.0, "rad": 1.0}
_LENGTH_UNITS: dict[str, str] = {
    "nm": "nm",
    "nanometer": "nm",
    "nanometers": "nm",
    "nanometre": "nm",
    "nanometres": "nm",
    "å": "angstrom",
    "a": "angstrom",
    "ang": "angstrom",
    "angstrom": "angstrom",
    "angstroms": "angstrom",
    "ångström": "angstrom",
    "ångströms": "angstrom",
    "pm": "pm",
    "picometer": "pm",
    "picometers": "pm",
    "picometre": "pm",
    "picometres": "pm",
    "um": "um",
    "µm": "um",
    "μm": "um",
    "micrometer": "um",
    "micrometers": "um",
    "micrometre": "um",
    "micrometres": "um",
    "mm": "mm",
    "millimeter": "mm",
    "millimeters": "mm",
    "millimetre": "mm",
    "millimetres": "mm",
    "m": "m",
    "meter": "m",
    "meters": "m",
    "metre": "m",
    "metres": "m",
}


def _canonical_angle_unit(unit: Any, *, name: str = "angle_unit") -> str:
    if not isinstance(unit, str) or not unit.strip():
        raise ValueError(f"{name} must be an explicit supported unit")
    key = unit.strip().lower().replace(" ", "").replace("_", "").replace("-", "")
    try:
        return _ANGLE_UNITS[key]
    except KeyError as exc:
        raise ValueError(f"Unsupported {name} {unit!r}; expected deg or rad") from exc


def _canonical_length_unit(unit: Any, *, name: str = "length_unit") -> str:
    if not isinstance(unit, str) or not unit.strip():
        raise ValueError(f"{name} must be an explicit supported unit")
    key = unit.strip().lower().replace(" ", "").replace("_", "").replace("-", "")
    try:
        return _LENGTH_UNITS[key]
    except KeyError as exc:
        raise ValueError(f"Unsupported {name} {unit!r}") from exc


def _number(value: Any, name: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if nonnegative and result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _source_value(source: Any) -> str | dict[str, Any]:
    if source is None:
        return ""
    if isinstance(source, os.PathLike):
        return os.fspath(source)
    if isinstance(source, str):
        return source
    if isinstance(source, Mapping):
        value = dict(source)
        try:
            json.dumps(value, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "source metadata must be JSON serializable and finite"
            ) from exc
        return value
    raise ValueError("source must be a path/string or JSON-serializable mapping")


def _copy_json_value(value: Any) -> Any:
    """Return a JSON-compatible copy while rejecting non-finite numbers."""

    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, Real):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("plot metadata numbers must be finite")
        # Keep integers as integers in persisted metrics where possible.
        return int(value) if isinstance(value, int) else result
    if isinstance(value, Mapping):
        return {str(key): _copy_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_copy_json_value(item) for item in value]
    raise ValueError("plot metadata must be JSON-compatible")


def _values(values: Any, name: str) -> list[float]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be an iterable of finite numbers")
    try:
        result = [_number(value, name) for value in values]
    except TypeError as exc:
        raise ValueError(f"{name} must be an iterable of finite numbers") from exc
    return result


def _pairs(values: Any, name: str) -> list[tuple[float, float]]:
    if values is None:
        return []
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must contain (x, y) pairs")
    try:
        items = list(values)
    except TypeError as exc:
        raise ValueError(f"{name} must contain (x, y) pairs") from exc
    result: list[tuple[float, float]] = []
    for index, item in enumerate(items):
        if isinstance(item, (str, bytes)):
            raise ValueError(f"{name}[{index}] must contain exactly two numbers")
        try:
            pair = list(item)
        except TypeError as exc:
            raise ValueError(
                f"{name}[{index}] must contain exactly two numbers"
            ) from exc
        if len(pair) != 2:
            raise ValueError(f"{name}[{index}] must contain exactly two numbers")
        result.append(
            (
                _number(pair[0], f"{name}[{index}][0]"),
                _number(pair[1], f"{name}[{index}][1]"),
            )
        )
    return result


@dataclass(slots=True)
class DerivedPlot:
    """Persistable data contract for one derived XRD plot.

    ``x`` and ``y`` hold the primary line/point arrays.  ``scatter`` and
    ``fit_line`` are optional ``(x, y)`` pairs for consumers that render a
    scatter series and a fitted line separately.  ``labels`` and ``metrics``
    carry explicit units and calculated values; no uncertainty field is
    inserted unless a future caller supplies one as source data.
    """

    kind: str
    x: list[float] = field(default_factory=list)
    y: list[float] = field(default_factory=list)
    scatter: list[tuple[float, float]] = field(default_factory=list)
    fit_line: list[tuple[float, float]] = field(default_factory=list)
    labels: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, int | float | str] = field(default_factory=dict)
    source: str | dict[str, Any] = ""

    def __post_init__(self) -> None:
        if self.kind not in _PLOT_KINDS:
            raise ValueError(f"Unsupported derived plot kind: {self.kind!r}")
        x = _values(self.x, "x")
        y = _values(self.y, "y")
        if not x or not y:
            raise ValueError("DerivedPlot x and y must contain at least one point")
        if len(x) != len(y):
            raise ValueError("DerivedPlot x and y arrays must have the same length")
        scatter = _pairs(self.scatter, "scatter")
        fit_line = _pairs(self.fit_line, "fit_line")

        if not isinstance(self.labels, Mapping):
            raise ValueError("labels must be a mapping of names to strings")
        labels: dict[str, str] = {}
        for key, value in self.labels.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise ValueError("labels must map strings to strings")
            labels[key] = value

        if not isinstance(self.metrics, Mapping):
            raise ValueError("metrics must be a mapping")
        metrics: dict[str, int | float | str] = {}
        for key, value in self.metrics.items():
            if not isinstance(key, str):
                raise ValueError("metric names must be strings")
            if isinstance(value, bool) or not isinstance(value, (str, Real)):
                raise ValueError("metrics must contain only strings or finite numbers")
            metrics[key] = _copy_json_value(value)

        source = _source_value(self.source)
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "scatter", scatter)
        object.__setattr__(self, "fit_line", fit_line)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "metrics", metrics)
        object.__setattr__(self, "source", source)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation of this plot."""

        result = {
            "kind": self.kind,
            "x": list(self.x),
            "y": list(self.y),
            "scatter": [[point[0], point[1]] for point in self.scatter],
            "fit_line": [[point[0], point[1]] for point in self.fit_line],
            "labels": dict(self.labels),
            "metrics": dict(self.metrics),
            "source": _copy_json_value(self.source),
        }
        # Fail closed if a caller somehow mutates a nested metadata value.
        json.dumps(result, allow_nan=False)
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DerivedPlot":
        if not isinstance(data, Mapping):
            raise ValueError("DerivedPlot data must be a mapping")
        allowed = {
            "kind",
            "x",
            "y",
            "scatter",
            "fit_line",
            "labels",
            "metrics",
            "source",
        }
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"Unknown DerivedPlot fields: {sorted(map(str, unknown))}")
        if "kind" not in data:
            raise ValueError("DerivedPlot data is missing kind")
        return cls(
            kind=data["kind"],
            x=data.get("x", []),
            y=data.get("y", []),
            scatter=data.get("scatter", []),
            fit_line=data.get("fit_line", []),
            labels=data.get("labels", {}),
            metrics=data.get("metrics", {}),
            source=data.get("source", ""),
        )

    # Explicit aliases make the persistence boundary convenient without
    # coupling callers to a particular serialisation library.
    as_dict = to_dict

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_json(cls, text: str) -> "DerivedPlot":
        try:
            value = json.loads(text)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("Invalid DerivedPlot JSON") from exc
        return cls.from_dict(value)


_PEAK_HEADERS = {
    "2theta": "two_theta",
    "2θ": "two_theta",
    "twotheta": "two_theta",
    "theta2": "two_theta",
    "two_theta": "two_theta",
    "hkl": "hkl",
    "fwhm": "fwhm",
    "intensity": "intensity",
}
_ROCKING_HEADERS = {
    "omega": "omega",
    "ω": "omega",
    "intensity": "intensity",
}


def _header_key(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("CSV headers must be strings")
    return value.strip().lower().replace(" ", "").replace("_", "").replace("-", "")


def _normalise_headers(
    header: Sequence[str], aliases: Mapping[str, str], required: set[str]
) -> list[str]:
    if not header or any(not str(item).strip() for item in header):
        raise ValueError("CSV must contain non-empty column headers")
    result: list[str] = []
    for item in header:
        key = _header_key(item)
        try:
            semantic = aliases[key]
        except KeyError as exc:
            raise ValueError(
                f"Unknown CSV column {item!r}; units/semantics are not inferred"
            ) from exc
        if semantic in result:
            raise ValueError(f"Duplicate CSV column for {semantic!r}")
        result.append(semantic)
    missing = required - set(result)
    if missing:
        raise ValueError(
            f"CSV is missing required columns: {', '.join(sorted(missing))}"
        )
    return result


def _csv_rows(
    source: str | os.PathLike[str] | TextIO,
    aliases: Mapping[str, str],
    required: set[str],
) -> tuple[list[str], list[list[str]], str]:
    close_after = False
    source_name = ""
    if hasattr(source, "read"):
        handle = source  # type: ignore[assignment]
        source_name = str(getattr(source, "name", ""))
    else:
        try:
            path = Path(source)
        except TypeError as exc:
            raise ValueError("CSV source must be a path or text stream") from exc
        source_name = os.fspath(path)
        try:
            handle = path.open("r", encoding="utf-8-sig", newline="")
        except OSError as exc:
            raise ValueError(f"Unable to read CSV source {path}") from exc
        close_after = True

    try:
        reader = csv.reader(handle)
        try:
            raw_header = next(reader)
        except StopIteration as exc:
            raise ValueError("CSV is empty") from exc
        semantic_header = _normalise_headers(raw_header, aliases, required)
        rows: list[list[str]] = []
        for row_number, row in enumerate(reader, start=2):
            if not row or all(not str(value).strip() for value in row):
                continue
            if len(row) != len(semantic_header):
                raise ValueError(
                    f"CSV row {row_number} has {len(row)} fields; expected {len(semantic_header)}"
                )
            rows.append(row)
        if not rows:
            raise ValueError("CSV contains no data rows")
        return semantic_header, rows, source_name
    finally:
        if close_after:
            handle.close()


def load_peak_measurements_csv(
    source: str | os.PathLike[str] | TextIO,
    *,
    angle_unit: str = "deg",
) -> list[PeakMeasurement]:
    """Load ``2theta,FWHM`` peaks with optional ``hkl`` and ``intensity``.

    The unit is selected by ``angle_unit``; no unit suffix in a header is
    interpreted automatically.  Unknown columns, malformed rows and
    non-finite values are rejected.
    """

    canonical_unit = _canonical_angle_unit(angle_unit)
    header, rows, _source = _csv_rows(source, _PEAK_HEADERS, {"two_theta", "fwhm"})
    positions = {name: index for index, name in enumerate(header)}
    result: list[PeakMeasurement] = []
    for row_number, row in enumerate(rows, start=2):
        try:
            two_theta = float(row[positions["two_theta"]].strip())
            fwhm = float(row[positions["fwhm"]].strip())
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"Invalid peak numeric value on CSV row {row_number}"
            ) from exc
        hkl = row[positions["hkl"]].strip() if "hkl" in positions else ""
        intensity: float | None = None
        if "intensity" in positions:
            raw_intensity = row[positions["intensity"]].strip()
            if raw_intensity:
                try:
                    intensity = float(raw_intensity)
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ValueError(
                        f"Invalid peak intensity on CSV row {row_number}"
                    ) from exc
        try:
            result.append(
                PeakMeasurement(
                    two_theta=two_theta,
                    fwhm=fwhm,
                    angle_unit=canonical_unit,
                    hkl=hkl,
                    intensity=intensity,
                )
            )
        except ValueError as exc:
            raise ValueError(
                f"Invalid peak measurement on CSV row {row_number}: {exc}"
            ) from exc
    return result


parse_peak_measurements_csv = load_peak_measurements_csv
load_peak_csv = load_peak_measurements_csv


def load_rocking_curve_csv(
    source: str | os.PathLike[str] | TextIO,
    *,
    x_unit: str = "deg",
    angle_unit: str | None = None,
) -> tuple[list[float], list[float]]:
    """Load an ``omega,intensity`` rocking curve.

    ``x_unit`` is explicit and defaults to degrees for the conventional XRD
    omega axis.  Only the named columns are accepted; no alternate numeric
    column is guessed to be omega or intensity.
    """

    if angle_unit is not None:
        if x_unit != "deg" and _canonical_angle_unit(x_unit) != _canonical_angle_unit(
            angle_unit
        ):
            raise ValueError("x_unit and angle_unit disagree")
        x_unit = angle_unit
    _canonical_angle_unit(x_unit, name="x_unit")
    header, rows, _source = _csv_rows(source, _ROCKING_HEADERS, {"omega", "intensity"})
    positions = {name: index for index, name in enumerate(header)}
    omega: list[float] = []
    intensity: list[float] = []
    for row_number, row in enumerate(rows, start=2):
        try:
            x_value = float(row[positions["omega"]].strip())
            y_value = float(row[positions["intensity"]].strip())
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"Invalid rocking-curve numeric value on CSV row {row_number}"
            ) from exc
        try:
            omega.append(_number(x_value, "omega"))
            intensity.append(_number(y_value, "intensity", nonnegative=True))
        except ValueError as exc:
            raise ValueError(
                f"Invalid rocking-curve value on CSV row {row_number}: {exc}"
            ) from exc
    if len(omega) > 1:
        deltas = [right - left for left, right in zip(omega, omega[1:])]
        if not all(delta > 0.0 for delta in deltas) and not all(
            delta < 0.0 for delta in deltas
        ):
            raise ValueError("omega values must be strictly monotonic")
    return omega, intensity


parse_rocking_curve_csv = load_rocking_curve_csv
load_rocking_csv = load_rocking_curve_csv


def _coerce_peaks(
    peaks: Iterable[PeakMeasurement], *, angle_unit: str | None = None
) -> tuple[list[PeakMeasurement], str]:
    if isinstance(peaks, (str, bytes, os.PathLike)):
        selected = _canonical_angle_unit(angle_unit or "deg")
        peak_list = load_peak_measurements_csv(peaks, angle_unit=selected)
    else:
        try:
            peak_list = list(peaks)
        except TypeError as exc:
            raise ValueError(
                "peaks must be an iterable of PeakMeasurement values"
            ) from exc
    if not peak_list:
        raise ValueError("At least one peak measurement is required")
    if any(not isinstance(peak, PeakMeasurement) for peak in peak_list):
        raise ValueError("peaks must contain only PeakMeasurement values")
    stored_unit = peak_list[0].angle_unit
    if angle_unit is not None and _canonical_angle_unit(angle_unit) != stored_unit:
        raise ValueError("angle_unit disagrees with PeakMeasurement units")
    if any(peak.angle_unit != stored_unit for peak in peak_list):
        raise ValueError("All peak measurements must use the same explicit angle unit")
    return peak_list, stored_unit


def _expand_parameter(
    value: Any, count: int, name: str, *, default: float | None = None
) -> list[float]:
    if value is None:
        if default is None:
            raise ValueError(f"{name} is required")
        return [default] * count
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number or a sequence of numbers")
    if isinstance(value, Real):
        return [_number(value, name, nonnegative=True)] * count
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a number or a sequence of numbers")
    try:
        values = list(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a number or a sequence of numbers") from exc
    if len(values) != count:
        raise ValueError(f"{name} sequence must match the number of peaks")
    return [_number(item, name, nonnegative=True) for item in values]


def _expand_units(
    value: Any, count: int, name: str, *, default: str | None = None
) -> list[str | None]:
    if value is None:
        return [default] * count
    if isinstance(value, str):
        return [value] * count
    if isinstance(value, bytes):
        raise ValueError(f"{name} must be a unit string or a sequence of unit strings")
    try:
        values = list(value)
    except TypeError as exc:
        raise ValueError(
            f"{name} must be a unit string or a sequence of unit strings"
        ) from exc
    if len(values) != count:
        raise ValueError(f"{name} sequence must match the number of peaks")
    return [item for item in values]


def _prepare_instruments(
    count: int,
    instrument_fwhm: Any,
    instrument_fwhm_deg: Any,
    instrument_fwhm_unit: Any,
) -> tuple[list[float], list[str | None]]:
    base_values = _expand_parameter(
        instrument_fwhm, count, "instrument_fwhm", default=0.0
    )
    if instrument_fwhm_deg is not None:
        if any(value != 0.0 for value in base_values):
            raise ValueError(
                "instrument_fwhm and instrument_fwhm_deg cannot both be supplied"
            )
        values = _expand_parameter(instrument_fwhm_deg, count, "instrument_fwhm_deg")
        units = _expand_units(
            instrument_fwhm_unit, count, "instrument_fwhm_unit", default="deg"
        )
    else:
        values = base_values
        units = _expand_units(instrument_fwhm_unit, count, "instrument_fwhm_unit")
    return values, units


def _shape_metric(k: Any, shape_factor: Any) -> float:
    if k is None and shape_factor is None:
        raise ValueError("k (the Scherrer shape factor) must be supplied explicitly")
    selected = k if k is not None else shape_factor
    result = _number(selected, "k")
    if result <= 0.0:
        raise ValueError("k must be positive")
    if k is not None and shape_factor is not None:
        other = _number(shape_factor, "shape_factor")
        if other <= 0.0 or other != result:
            raise ValueError("k and shape_factor disagree")
    return result


def _wavelength_metric(wavelength: Any, wavelength_unit: str) -> tuple[float, str]:
    value = _number(wavelength, "wavelength")
    if value <= 0.0:
        raise ValueError("wavelength must be positive")
    return value, _canonical_length_unit(wavelength_unit, name="wavelength_unit")


def _source_or_default(source: Any, default: Any = None) -> str | dict[str, Any]:
    return _source_value(default if source is None else source)


def build_scherrer_plot(
    peaks: Iterable[PeakMeasurement] | str | os.PathLike[str],
    *,
    k: Real | None = None,
    shape_factor: Real | None = None,
    wavelength: Real,
    wavelength_unit: str = "nm",
    instrument_fwhm: Real | Sequence[Real] = 0.0,
    instrument_fwhm_deg: Real | Sequence[Real] | None = None,
    instrument_fwhm_unit: str | Sequence[str] | None = None,
    output_unit: str | None = None,
    angle_unit: str | None = None,
    source: Any = None,
) -> DerivedPlot:
    """Build a Scherrer size-vs-peak-position plot data object."""

    peak_list, stored_angle_unit = _coerce_peaks(peaks, angle_unit=angle_unit)
    shape = _shape_metric(k, shape_factor)
    wavelength_value, wavelength_canonical = _wavelength_metric(
        wavelength, wavelength_unit
    )
    size_unit = (
        wavelength_canonical
        if output_unit is None
        else _canonical_length_unit(output_unit, name="output_unit")
    )
    instruments, instrument_units = _prepare_instruments(
        len(peak_list), instrument_fwhm, instrument_fwhm_deg, instrument_fwhm_unit
    )
    sizes: list[float] = []
    for peak, instrument, unit in zip(peak_list, instruments, instrument_units):
        sizes.append(
            scherrer_crystallite_size(
                peak,
                k=shape,
                wavelength=wavelength_value,
                wavelength_unit=wavelength_canonical,
                instrument_fwhm=instrument,
                instrument_fwhm_unit=unit,
                output_unit=size_unit,
            )
        )
    x_values = [peak.two_theta for peak in peak_list]
    scatter = list(zip(x_values, sizes))
    return DerivedPlot(
        kind="scherrer",
        x=x_values,
        y=sizes,
        scatter=scatter,
        labels={
            "x": f"2θ ({stored_angle_unit})",
            "y": f"Crystallite size ({size_unit})",
            "title": "Scherrer crystallite size",
        },
        metrics={
            "k": shape,
            "wavelength": wavelength_value,
            "wavelength_unit": wavelength_canonical,
            "size_unit": size_unit,
            "n_points": len(peak_list),
        },
        source=_source_or_default(source),
    )


def _corrected_beta_rad(
    peak: PeakMeasurement, instrument: float, unit: str | None
) -> float:
    beta = peak.fwhm_rad
    instrument_unit = (
        peak.angle_unit
        if unit is None
        else _canonical_angle_unit(unit, name="instrument_fwhm_unit")
    )
    instrument_rad = instrument * _ANGLE_TO_RAD[instrument_unit]
    if instrument_rad >= beta:
        raise ValueError("instrument_fwhm must be smaller than measured_fwhm")
    corrected = math.sqrt(beta * beta - instrument_rad * instrument_rad)
    if not math.isfinite(corrected) or corrected <= 0.0:
        raise ValueError("instrument-corrected FWHM must be finite and positive")
    return corrected


def build_williamson_hall_plot(
    peaks: Iterable[PeakMeasurement] | str | os.PathLike[str],
    *,
    k: Real | None = None,
    shape_factor: Real | None = None,
    wavelength: Real,
    wavelength_unit: str = "nm",
    instrument_fwhm: Real | Sequence[Real] = 0.0,
    instrument_fwhm_deg: Real | Sequence[Real] | None = None,
    instrument_fwhm_unit: str | Sequence[str] | None = None,
    output_unit: str | None = None,
    angle_unit: str | None = None,
    source: Any = None,
) -> DerivedPlot:
    """Build ``β cos(θ)`` versus ``4 sin(θ)`` Williamson--Hall data."""

    peak_list, _stored_angle_unit = _coerce_peaks(peaks, angle_unit=angle_unit)
    shape = _shape_metric(k, shape_factor)
    wavelength_value, wavelength_canonical = _wavelength_metric(
        wavelength, wavelength_unit
    )
    size_unit = (
        wavelength_canonical
        if output_unit is None
        else _canonical_length_unit(output_unit, name="output_unit")
    )
    instruments, instrument_units = _prepare_instruments(
        len(peak_list), instrument_fwhm, instrument_fwhm_deg, instrument_fwhm_unit
    )
    transformed_x: list[float] = []
    transformed_y: list[float] = []
    for peak, instrument, unit in zip(peak_list, instruments, instrument_units):
        theta = peak.two_theta_rad / 2.0
        beta = _corrected_beta_rad(peak, instrument, unit)
        transformed_x.append(4.0 * math.sin(theta))
        transformed_y.append(beta * math.cos(theta))

    # Pass the same explicit correction parameters through derived.py so the
    # reported fit and the rendered points share one calculation contract.
    fit = williamson_hall_fit(
        peak_list,
        k=shape,
        wavelength=wavelength_value,
        wavelength_unit=wavelength_canonical,
        instrument_fwhm=instruments,
        instrument_fwhm_unit=instrument_units,
        output_unit=size_unit,
    )
    min_x, max_x = min(transformed_x), max(transformed_x)
    fit_line = [
        (min_x, fit.slope * min_x + fit.intercept),
        (max_x, fit.slope * max_x + fit.intercept),
    ]
    return DerivedPlot(
        kind="williamson_hall",
        x=transformed_x,
        y=transformed_y,
        scatter=list(zip(transformed_x, transformed_y)),
        fit_line=fit_line,
        labels={
            "x": "4 sin(theta)",
            "y": "beta cos(theta) (rad)",
            "title": "Williamson-Hall analysis",
        },
        metrics={
            "slope": fit.slope,
            "intercept": fit.intercept,
            "microstrain": fit.microstrain,
            "crystallite_size": fit.crystallite_size,
            "size_unit": fit.size_unit,
            "r_squared": fit.r_squared,
            "n_points": fit.n_points,
            "k": shape,
            "wavelength": wavelength_value,
            "wavelength_unit": wavelength_canonical,
        },
        source=_source_or_default(source),
    )


def _coerce_rocking_input(
    x_values: Any,
    intensity_values: Any,
    *,
    x_unit: str,
) -> tuple[list[float], list[float], str | None]:
    source: str | None = None
    if intensity_values is None:
        if isinstance(x_values, (str, bytes, os.PathLike)):
            source = (
                os.fspath(x_values)
                if isinstance(x_values, os.PathLike)
                else str(x_values)
            )
            xs, ys = load_rocking_curve_csv(x_values, x_unit=x_unit)
            return xs, ys, source
        if isinstance(x_values, Mapping):
            if "omega" not in x_values or "intensity" not in x_values:
                raise ValueError(
                    "rocking input mapping must contain omega and intensity"
                )
            return list(x_values["omega"]), list(x_values["intensity"]), None
        try:
            pair = list(x_values)
        except TypeError as exc:
            raise ValueError(
                "Provide x_values and intensity_values, or a two-item rocking input"
            ) from exc
        if len(pair) != 2:
            raise ValueError(
                "A rocking input without intensity_values must contain (omega, intensity)"
            )
        return list(pair[0]), list(pair[1]), None
    return list(x_values), list(intensity_values), None


def build_rocking_curve_plot(
    x_values: Iterable[Real] | str | os.PathLike[str] | Mapping[str, Any],
    intensity_values: Iterable[Real] | None = None,
    *,
    x_unit: str = "deg",
    angle_unit: str | None = None,
    unit: str | None = None,
    source: Any = None,
) -> DerivedPlot:
    """Build rocking-curve line/scatter data and transparent metrics."""

    if angle_unit is not None:
        if x_unit != "deg" and _canonical_angle_unit(x_unit) != _canonical_angle_unit(
            angle_unit
        ):
            raise ValueError("x_unit and angle_unit disagree")
        x_unit = angle_unit
    if unit is not None:
        if x_unit != "deg" and _canonical_angle_unit(x_unit) != _canonical_angle_unit(
            unit
        ):
            raise ValueError("x_unit and unit disagree")
        x_unit = unit
    canonical_unit = _canonical_angle_unit(x_unit, name="x_unit")
    xs, ys, source_from_input = _coerce_rocking_input(
        x_values, intensity_values, x_unit=canonical_unit
    )
    xs = _values(xs, "omega")
    ys = [_number(value, "intensity", nonnegative=True) for value in ys]
    metrics = rocking_curve_metrics(xs, ys, x_unit=canonical_unit)
    scatter = list(zip(xs, ys))
    return DerivedPlot(
        kind="rocking_curve",
        x=xs,
        y=ys,
        scatter=scatter,
        labels={
            "x": f"omega ({canonical_unit})",
            "y": "Intensity (a.u.)",
            "title": "Rocking curve",
        },
        metrics={
            "peak_position": metrics.peak_position,
            "fwhm": metrics.fwhm,
            "integrated_intensity": metrics.integrated_intensity,
            "area": metrics.integrated_intensity,
            "x_unit": metrics.x_unit,
            "n_points": len(xs),
        },
        source=_source_or_default(source, source_from_input),
    )


# Descriptive aliases for callers that prefer ``*_plot_data`` naming.
scherrer_plot_data = build_scherrer_plot
williamson_hall_plot_data = build_williamson_hall_plot
rocking_curve_plot_data = build_rocking_curve_plot
build_scherrer_plot_data = build_scherrer_plot
build_williamson_hall_plot_data = build_williamson_hall_plot
build_rocking_curve_plot_data = build_rocking_curve_plot


__all__ = [
    "DerivedPlot",
    "build_rocking_curve_plot",
    "build_rocking_curve_plot_data",
    "build_scherrer_plot",
    "build_scherrer_plot_data",
    "build_williamson_hall_plot",
    "build_williamson_hall_plot_data",
    "load_peak_csv",
    "load_peak_measurements_csv",
    "load_rocking_curve_csv",
    "load_rocking_csv",
    "parse_peak_measurements_csv",
    "parse_rocking_curve_csv",
    "rocking_curve_plot_data",
    "scherrer_plot_data",
    "williamson_hall_plot_data",
]
