from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path

from xrdviz.models import normalize_axis_kind


@dataclass(slots=True)
class FitComponent:
    name: str
    y: list[float]
    color: str = ""
    center: float | None = None
    fwhm: float | None = None
    area: float | None = None
    amplitude: float | None = None
    eta: float | None = None
    profile: str = ""

    def __post_init__(self) -> None:
        self.name = str(self.name).strip()
        if not self.name:
            raise ValueError("Fit component name must not be empty")
        self.y = _finite_values(self.y, f"Fit component {self.name!r}")
        self.color = str(self.color or "")
        for field_name in ("center", "fwhm", "area", "amplitude", "eta"):
            value = getattr(self, field_name)
            if value is not None:
                parsed = float(value)
                if not math.isfinite(parsed):
                    raise ValueError(f"Fit component {field_name} must be finite")
                setattr(self, field_name, parsed)
        if self.fwhm is not None and self.fwhm <= 0:
            raise ValueError("Fit component fwhm must be positive")
        if self.area is not None and self.area < 0:
            raise ValueError("Fit component area must be non-negative")
        if self.amplitude is not None and self.amplitude < 0:
            raise ValueError("Fit component amplitude must be non-negative")
        if self.eta is not None and not 0.0 <= self.eta <= 1.0:
            raise ValueError("Fit component eta must be in [0, 1]")
        self.profile = str(self.profile or "")


@dataclass(slots=True)
class PatternFit:
    name: str
    x: list[float]
    observed: list[float]
    calculated: list[float]
    sigma: list[float] = field(default_factory=list)
    background: list[float] = field(default_factory=list)
    components: list[FitComponent] = field(default_factory=list)
    source_path: str = ""
    axis_kind: str = "two_theta"
    fit_kind: str = "profile_fit"
    converged: bool | None = None
    fit_message: str = ""
    # Appended to preserve the legacy positional constructor contract.
    wavelength_angstrom: float | None = None

    def __post_init__(self) -> None:
        self.name = str(self.name).strip() or "Fit"
        self.axis_kind = normalize_axis_kind(self.axis_kind)
        self.x = _finite_values(self.x, "Fit x")
        self.observed = _finite_values(self.observed, "Observed intensity")
        self.calculated = _finite_values(self.calculated, "Calculated intensity")
        if len(self.x) < 2:
            raise ValueError("Pattern fit requires at least two points")
        differences = [right - left for left, right in zip(self.x, self.x[1:])]
        if not (
            all(value > 0 for value in differences)
            or all(value < 0 for value in differences)
        ):
            raise ValueError("Pattern fit x values must be strictly monotonic")
        _require_length(self.observed, len(self.x), "Observed intensity")
        _require_length(self.calculated, len(self.x), "Calculated intensity")
        self.sigma = _finite_values(self.sigma, "Fit sigma") if self.sigma else []
        if self.sigma:
            _require_length(self.sigma, len(self.x), "Fit sigma")
            if any(value <= 0 for value in self.sigma):
                raise ValueError("Fit sigma values must be positive")
        self.background = (
            _finite_values(self.background, "Fit background") if self.background else []
        )
        if self.background:
            _require_length(self.background, len(self.x), "Fit background")
        parsed_components: list[FitComponent] = []
        for component in self.components:
            parsed = (
                component
                if isinstance(component, FitComponent)
                else FitComponent(**component)
            )
            _require_length(parsed.y, len(self.x), f"Fit component {parsed.name!r}")
            parsed_components.append(parsed)
        component_names = [component.name.casefold() for component in parsed_components]
        if len(set(component_names)) != len(component_names):
            raise ValueError("Fit component names must be unique")
        self.components = parsed_components
        self.source_path = str(self.source_path or "")
        self.fit_kind = str(self.fit_kind or "profile_fit").strip()
        self.converged = None if self.converged is None else bool(self.converged)
        self.fit_message = str(self.fit_message or "")
        self.wavelength_angstrom = (
            None
            if self.wavelength_angstrom is None
            else float(self.wavelength_angstrom)
        )
        if self.wavelength_angstrom is not None and (
            not math.isfinite(self.wavelength_angstrom)
            or self.wavelength_angstrom <= 0.0
        ):
            raise ValueError("Fit wavelength must be a positive finite value")

    @property
    def difference(self) -> list[float]:
        return [
            observed - calculated
            for observed, calculated in zip(self.observed, self.calculated)
        ]

    @property
    def rp(self) -> float | None:
        """Return the profile residual in percent, or ``None`` for a zero denominator."""

        denominator = sum(abs(value) for value in self.observed)
        if denominator <= 0:
            return None
        return 100.0 * sum(abs(value) for value in self.difference) / denominator

    @property
    def rwp(self) -> float | None:
        """Return the weighted profile residual in percent.

        ``sigma`` supplies the independent point weights as ``1/sigma**2``.
        Without it, returning an equal-weight value under the conventional
        ``Rwp`` label would be ambiguous, so the result is ``None``. No
        goodness-of-fit is reported because that additionally requires an
        independently declared number of refined parameters.
        """

        if not self.sigma:
            return None
        weights = [1.0 / (value * value) for value in self.sigma]
        denominator = sum(
            weight * observed * observed
            for weight, observed in zip(weights, self.observed)
        )
        if denominator <= 0:
            return None
        numerator = sum(
            weight * difference * difference
            for weight, difference in zip(weights, self.difference)
        )
        return 100.0 * math.sqrt(numerator / denominator)


def parse_pattern_fit_csv(
    text: str,
    *,
    name: str = "Fit",
    source_path: str = "",
    axis_kind: str | None = None,
    wavelength_angstrom: float | None = None,
) -> PatternFit:
    reader = csv.DictReader(StringIO(text))
    if reader.fieldnames is None:
        raise ValueError("Fit CSV must contain a header row")
    headers = _unique_headers(reader.fieldnames)
    x_header, inferred_axis = _find_x_header(headers)
    observed_header = _find_required_header(
        headers, ("observed", "obs", "yobs", "y_observed", "intensity"), "observed"
    )
    calculated_header = _find_required_header(
        headers,
        ("calculated", "calc", "ycalc", "y_calculated", "fitted", "fit"),
        "calculated",
    )
    sigma_header = _find_optional_header(
        headers, ("sigma", "esd", "uncertainty", "error", "std")
    )
    background_header = _find_optional_header(headers, ("background", "bkg", "bg"))
    wavelength_header = _find_optional_header(
        headers,
        ("wavelength_angstrom", "wavelength", "lambda_angstrom"),
        label="wavelength",
    )
    component_headers = [
        (header, _component_name(header))
        for header in reader.fieldnames
        if header is not None and _component_name(header) is not None
    ]
    component_names = [component_name.casefold() for _header, component_name in component_headers]
    if len(set(component_names)) != len(component_names):
        raise ValueError("Fit CSV has duplicate component columns")

    x: list[float] = []
    observed: list[float] = []
    calculated: list[float] = []
    sigma: list[float] = []
    background: list[float] = []
    wavelengths: list[float] = []
    component_values = {
        component_name: [] for _header, component_name in component_headers
    }
    for row_number, row in enumerate(reader, start=2):
        if None in row:
            raise ValueError(f"Fit CSV row {row_number} has more fields than the header")
        if not any(str(value or "").strip() for value in row.values()):
            continue
        x.append(_csv_number(row.get(x_header), row_number, x_header))
        observed.append(
            _csv_number(row.get(observed_header), row_number, observed_header)
        )
        calculated.append(
            _csv_number(row.get(calculated_header), row_number, calculated_header)
        )
        if sigma_header is not None:
            sigma.append(_csv_number(row.get(sigma_header), row_number, sigma_header))
        if background_header is not None:
            background.append(
                _csv_number(row.get(background_header), row_number, background_header)
            )
        if wavelength_header is not None:
            wavelengths.append(
                _csv_number(row.get(wavelength_header), row_number, wavelength_header)
            )
        for header, component_name in component_headers:
            component_values[component_name].append(
                _csv_number(row.get(header), row_number, header)
            )
    if not x:
        raise ValueError("Fit CSV does not contain data rows")
    components = [
        FitComponent(name=component_name, y=values)
        for component_name, values in component_values.items()
    ]
    csv_wavelength = wavelengths[0] if wavelengths else None
    if csv_wavelength is not None and any(
        not math.isclose(value, csv_wavelength, rel_tol=1e-12, abs_tol=0.0)
        for value in wavelengths[1:]
    ):
        raise ValueError("Fit CSV wavelength must be constant across all rows")
    if (
        wavelength_angstrom is not None
        and csv_wavelength is not None
        and not math.isclose(
            float(wavelength_angstrom),
            csv_wavelength,
            rel_tol=1e-12,
            abs_tol=0.0,
        )
    ):
        raise ValueError("Fit CSV wavelength conflicts with the requested wavelength")
    return PatternFit(
        name=name,
        x=x,
        observed=observed,
        calculated=calculated,
        sigma=sigma,
        background=background,
        components=components,
        source_path=source_path,
        axis_kind=axis_kind or inferred_axis,
        wavelength_angstrom=(
            wavelength_angstrom
            if wavelength_angstrom is not None
            else csv_wavelength
        ),
    )


def load_pattern_fit(
    path: str | Path,
    *,
    axis_kind: str | None = None,
    wavelength_angstrom: float | None = None,
) -> PatternFit:
    source = Path(path)
    return parse_pattern_fit_csv(
        source.read_text(encoding="utf-8-sig"),
        name=source.stem,
        source_path=str(source),
        axis_kind=axis_kind,
        wavelength_angstrom=wavelength_angstrom,
    )


def _finite_values(values, label: str) -> list[float]:
    parsed = [float(value) for value in values]
    if any(not math.isfinite(value) for value in parsed):
        raise ValueError(f"{label} values must be finite")
    return parsed


def _require_length(values: list[float], expected: int, label: str) -> None:
    if len(values) != expected:
        raise ValueError(f"{label} must contain {expected} values; got {len(values)}")


def _normalized_header(value: str) -> str:
    normalized = str(value).strip().lower().replace("2θ", "2theta")
    return re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")


def _unique_headers(fieldnames: list[str]) -> dict[str, str]:
    headers: dict[str, str] = {}
    for header in fieldnames:
        if header is None:
            raise ValueError("Fit CSV contains an unnamed column")
        normalized = _normalized_header(header)
        if not normalized:
            raise ValueError("Fit CSV headers must not be empty")
        if normalized in headers:
            raise ValueError(f"Fit CSV has duplicate header {header!r}")
        headers[normalized] = header
    return headers


def _find_x_header(headers: dict[str, str]) -> tuple[str, str]:
    candidates = [
        (headers[alias], axis_kind)
        for alias, axis_kind in (
            ("two_theta", "two_theta"),
            ("2theta", "two_theta"),
            ("d_spacing", "d"),
            ("d", "d"),
            ("q", "q"),
            ("x", "two_theta"),
        )
        if alias in headers
    ]
    if len(candidates) > 1:
        raise ValueError(
            "Fit CSV has multiple x-axis columns: "
            + ", ".join(header for header, _axis in candidates)
        )
    if candidates:
        return candidates[0]
    raise ValueError("Fit CSV must include an x, 2theta, d, or q column")


def _find_required_header(
    headers: dict[str, str], aliases: tuple[str, ...], label: str
) -> str:
    header = _find_optional_header(headers, aliases, label=label)
    if header is None:
        raise ValueError(f"Fit CSV must include a {label} column")
    return header


def _find_optional_header(
    headers: dict[str, str], aliases: tuple[str, ...], *, label: str = "semantic"
) -> str | None:
    candidates = [headers[alias] for alias in aliases if alias in headers]
    if len(candidates) > 1:
        raise ValueError(
            f"Fit CSV has multiple {label} columns: " + ", ".join(candidates)
        )
    return candidates[0] if candidates else None


def _component_name(header: str) -> str | None:
    normalized = _normalized_header(header)
    for prefix in ("component_", "peak_"):
        if normalized.startswith(prefix) and len(normalized) > len(prefix):
            return normalized[len(prefix) :].replace("_", " ")
    return None


def _csv_number(value: object, row_number: int, column: str) -> float:
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Fit CSV row {row_number} column {column!r} must be numeric"
        ) from exc
    if not math.isfinite(parsed):
        raise ValueError(f"Fit CSV row {row_number} column {column!r} must be finite")
    return parsed
