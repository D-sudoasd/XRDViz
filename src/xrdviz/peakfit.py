"""Small, auditable one-dimensional peak-decomposition engine.

The public fitting seam in this module deliberately deals in arrays and
explicit peak seeds.  It is intended for plotting and reporting, rather than
as a replacement for a full Rietveld refinement.  Peak ``width`` and
``fwhm`` are both full-width-at-half-maximum values.  The peak amplitude is
the height above the fitted baseline.

SciPy is used explicitly for the bounded non-linear least-squares step.  The
module does not silently fall back to a different fitter when SciPy is not
available: callers should install the declared numerical dependency.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import find_peaks, peak_widths


_SUPPORTED_PROFILES = {"pseudo_voigt", "gaussian", "lorentzian"}
_GAUSSIAN_AREA_FACTOR = math.sqrt(math.pi) / (2.0 * math.sqrt(math.log(2.0)))
_EPS = np.finfo(float).eps
DEFAULT_PEAK_FIT_MAX_NFEV = 2000


class PeakFitCancelled(RuntimeError):
    """Raised when a cooperative peak-fit cancellation is requested."""


@dataclass(frozen=True, slots=True)
class PeakSeed:
    """Initial parameters for one positive peak.

    Parameters
    ----------
    center:
        Initial peak position in the same coordinate system as ``x``.
    amplitude:
        Initial peak height above the baseline.  It must be non-negative.
    width:
        Initial FWHM.  It must be strictly positive.
    eta:
        Lorentzian fraction for a pseudo-Voigt peak, in ``[0, 1]``.  It is
        ignored for a pure Gaussian or Lorentzian profile.
    name:
        Optional display name.  Blank names receive deterministic names in
        the returned summaries.
    """

    center: float
    amplitude: float
    width: float
    eta: float = 0.5
    name: str = ""

    def __post_init__(self) -> None:
        center = _finite_float(self.center, "Peak center")
        amplitude = _finite_float(self.amplitude, "Peak amplitude")
        width = _finite_float(self.width, "Peak width")
        eta = _finite_float(self.eta, "Peak eta")
        if amplitude < 0.0:
            raise ValueError("Peak amplitude must be non-negative")
        if width <= 0.0:
            raise ValueError("Peak width must be positive")
        if not 0.0 <= eta <= 1.0:
            raise ValueError("Peak eta must be between 0 and 1")
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "amplitude", amplitude)
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "eta", eta)
        object.__setattr__(self, "name", str(self.name or "").strip())


@dataclass(frozen=True, slots=True)
class PeakSummary:
    """Fitted physical summary for one component peak."""

    name: str
    center: float
    fwhm: float
    area: float
    amplitude: float
    eta: float | None = None
    profile: str = ""

    def __post_init__(self) -> None:
        center = _finite_float(self.center, "Fitted peak center")
        fwhm = _finite_float(self.fwhm, "Fitted peak FWHM")
        area = _finite_float(self.area, "Fitted peak area")
        amplitude = _finite_float(self.amplitude, "Fitted peak amplitude")
        if fwhm <= 0.0:
            raise ValueError("Fitted peak FWHM must be positive")
        if amplitude < 0.0 or area < 0.0:
            raise ValueError("Fitted peak amplitude and area must be non-negative")
        eta = None if self.eta is None else _finite_float(self.eta, "Fitted peak eta")
        if eta is not None and not 0.0 <= eta <= 1.0:
            raise ValueError("Fitted peak eta must be between 0 and 1")
        object.__setattr__(self, "name", str(self.name or "").strip())
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "fwhm", fwhm)
        object.__setattr__(self, "area", area)
        object.__setattr__(self, "amplitude", amplitude)
        object.__setattr__(self, "eta", eta)
        object.__setattr__(self, "profile", str(self.profile or ""))

    @property
    def width(self) -> float:
        """Alias for the fitted FWHM, matching :class:`PeakSeed`."""

        return self.fwhm

    @property
    def height(self) -> float:
        """Alias for the fitted peak amplitude."""

        return self.amplitude

    @property
    def position(self) -> float:
        """Alias for the fitted peak center."""

        return self.center


@dataclass(slots=True)
class PeakDecompositionResult:
    """Observable output of :func:`fit_peaks`.

    ``components`` contains the individual peak curves only; ``baseline`` is
    separate and ``fitted`` is their sum.  ``residual`` is always observed
    minus fitted.  A non-converged optimizer still returns its best available
    finite model and sets ``converged`` to ``False`` so a caller can decide
    whether it is suitable for publication.
    """

    x: np.ndarray
    y: np.ndarray
    fitted: np.ndarray
    baseline: np.ndarray
    components: list[np.ndarray]
    residual: np.ndarray
    summaries: list[PeakSummary]
    converged: bool
    profile: str
    baseline_order: int
    message: str = ""
    cost: float | None = None
    nfev: int | None = None
    optimizer_status: int | None = None

    def __post_init__(self) -> None:
        self.x = _result_array(self.x, "Result x")
        self.y = _result_array(self.y, "Result observed intensity")
        self.fitted = _result_array(self.fitted, "Result fitted intensity")
        self.baseline = _result_array(self.baseline, "Result baseline")
        self.residual = _result_array(self.residual, "Result residual")
        if self.x.ndim != 1:
            raise ValueError("Result x must be one-dimensional")
        expected = self.x.size
        for label, values in (
            ("Result observed intensity", self.y),
            ("Result fitted intensity", self.fitted),
            ("Result baseline", self.baseline),
            ("Result residual", self.residual),
        ):
            if values.size != expected:
                raise ValueError(f"{label} must contain {expected} values")
        parsed_components: list[np.ndarray] = []
        for index, component in enumerate(self.components, start=1):
            parsed = _result_array(component, f"Result component {index}")
            if parsed.size != expected:
                raise ValueError(
                    f"Result component {index} must contain {expected} values"
                )
            parsed_components.append(parsed)
        self.components = parsed_components
        self.summaries = list(self.summaries)
        if len(self.summaries) != len(self.components):
            raise ValueError(
                "Result summaries and components must have the same length"
            )
        self.converged = bool(self.converged)
        self.profile = _normalize_profile(self.profile)
        self.baseline_order = _validate_baseline_order(self.baseline_order)
        self.message = str(self.message or "")
        if self.cost is not None:
            self.cost = _finite_float(self.cost, "Result cost")
        if self.nfev is not None:
            self.nfev = int(self.nfev)
        if self.optimizer_status is not None:
            self.optimizer_status = int(self.optimizer_status)

    @property
    def success(self) -> bool:
        """SciPy-style alias for :attr:`converged`."""

        return self.converged

    @property
    def convergence_status(self) -> str:
        """Human-readable convergence status suitable for a report."""

        return "converged" if self.converged else "not_converged"

    @property
    def status(self) -> str:
        """Alias for :attr:`convergence_status`."""

        return self.convergence_status

    @property
    def total(self) -> np.ndarray:
        """Alias for the fitted total curve."""

        return self.fitted

    @property
    def fit(self) -> np.ndarray:
        """Alias for the fitted total curve."""

        return self.fitted

    @property
    def fit_total(self) -> np.ndarray:
        """Alias for the fitted total curve."""

        return self.fitted

    @property
    def calculated(self) -> np.ndarray:
        """Alias useful when exporting an observed/calculated profile."""

        return self.fitted

    @property
    def observed(self) -> np.ndarray:
        """Alias for the input intensity array."""

        return self.y

    @property
    def background(self) -> np.ndarray:
        """Alias for the fitted baseline curve."""

        return self.baseline

    @property
    def residuals(self) -> np.ndarray:
        """Plural alias for the residual curve."""

        return self.residual

    @property
    def peaks(self) -> list[PeakSummary]:
        """Alias for fitted peak summaries."""

        return self.summaries

    @property
    def peak_summaries(self) -> list[PeakSummary]:
        """Explicit alias for fitted peak summaries."""

        return self.summaries


def fit_peaks(
    x: Sequence[float] | np.ndarray,
    y: Sequence[float] | np.ndarray,
    seeds: Sequence[PeakSeed | Mapping[str, Any]],
    *,
    profile: str = "pseudo_voigt",
    baseline_order: int = 0,
    max_nfev: int = DEFAULT_PEAK_FIT_MAX_NFEV,
    cancel_check: Callable[[], bool] | None = None,
) -> PeakDecompositionResult:
    """Fit one or more seeded peaks plus a polynomial baseline.

    The input axis must be finite and strictly increasing.  Peak seed centers
    must be finite, strictly increasing, and inside the observed axis.  A
    seed ``width`` is a positive FWHM; amplitudes are constrained to be
    non-negative.  The returned peak summaries are sorted by fitted center.

    Parameters are fitted with bounded SciPy least squares.  For a
    pseudo-Voigt profile, ``eta`` is refined as the Lorentzian fraction.  Pure
    Gaussian and Lorentzian fits have no free ``eta`` parameter.

    ``max_nfev`` is an explicit evaluation budget.  ``cancel_check`` is a
    cooperative cancellation callback; it is checked before optimization and
    before each residual evaluation and raises :class:`PeakFitCancelled` when
    it returns true.  The default budget is deliberately bounded so GUI
    callers can use the same numerical API without an unbounded optimizer.
    """

    normalized_profile = _normalize_profile(profile)
    normalized_order = _validate_baseline_order(baseline_order)
    resolved_max_nfev = _validate_max_nfev(max_nfev)
    if cancel_check is not None and not callable(cancel_check):
        raise ValueError("cancel_check must be callable")
    _raise_if_cancelled(cancel_check)
    axis = _input_array(x, "x")
    intensity = _input_array(y, "y")
    if axis.size != intensity.size:
        raise ValueError("x and y must contain the same number of values")
    if axis.size < 2:
        raise ValueError("Peak fitting requires at least two data points")
    differences = np.diff(axis)
    if np.any(differences <= 0.0):
        raise ValueError("x values must be strictly increasing")

    parsed_seeds = _parse_seeds(seeds)
    if not parsed_seeds:
        raise ValueError("At least one peak seed is required")
    previous_center = None
    for seed in parsed_seeds:
        if not axis[0] <= seed.center <= axis[-1]:
            raise ValueError("Peak seed centers must lie within the x range")
        if previous_center is not None and seed.center <= previous_center:
            raise ValueError("Peak seeds must be strictly sorted by center")
        previous_center = seed.center

    parameters_per_peak = 4 if normalized_profile == "pseudo_voigt" else 3
    parameter_count = normalized_order + 1 + parameters_per_peak * len(parsed_seeds)
    if axis.size <= parameter_count:
        raise ValueError(
            f"insufficient data for {len(parsed_seeds)} peak(s) and baseline_order={normalized_order}; "
            f"need more than {parameter_count} points"
        )

    x_reference = 0.5 * (float(axis[0]) + float(axis[-1]))
    x_scale = 0.5 * (float(axis[-1]) - float(axis[0]))
    normalized_x = (axis - x_reference) / x_scale
    minimum_step = float(np.min(differences))
    minimum_width = max(_EPS * max(1.0, abs(x_scale)), minimum_step * 1.0e-6)
    maximum_width = max(
        (float(axis[-1]) - float(axis[0])) * 10.0,
        max(seed.width for seed in parsed_seeds) * 10.0,
        minimum_width * 10.0,
    )

    baseline_initial = _initial_baseline(
        normalized_x, intensity, axis, parsed_seeds, normalized_order
    )
    initial: list[float] = list(baseline_initial)
    lower: list[float] = [-np.inf] * (normalized_order + 1)
    upper: list[float] = [np.inf] * (normalized_order + 1)
    for seed in parsed_seeds:
        initial.extend(
            [
                float(np.clip(seed.center, axis[0], axis[-1])),
                seed.amplitude,
                float(
                    np.clip(seed.width, minimum_width * (1.0 + 1.0e-12), maximum_width)
                ),
            ]
        )
        lower.extend([float(axis[0]), 0.0, minimum_width])
        upper.extend([float(axis[-1]), np.inf, maximum_width])
        if normalized_profile == "pseudo_voigt":
            initial.append(seed.eta)
            lower.append(0.0)
            upper.append(1.0)
    initial_array = np.asarray(initial, dtype=float)
    lower_array = np.asarray(lower, dtype=float)
    upper_array = np.asarray(upper, dtype=float)

    def evaluate(
        parameters: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
        coefficients = parameters[: normalized_order + 1]
        fitted_baseline = np.polynomial.polynomial.polyval(normalized_x, coefficients)
        fitted_components: list[np.ndarray] = []
        cursor = normalized_order + 1
        for _index in range(len(parsed_seeds)):
            center = parameters[cursor]
            amplitude = parameters[cursor + 1]
            width = parameters[cursor + 2]
            cursor += 3
            eta = None
            if normalized_profile == "pseudo_voigt":
                eta = parameters[cursor]
                cursor += 1
            fitted_components.append(
                _profile_values(axis, center, amplitude, width, normalized_profile, eta)
            )
        total = fitted_baseline.copy()
        for component in fitted_components:
            total += component
        return total, fitted_baseline, fitted_components

    def residual_function(parameters: np.ndarray) -> np.ndarray:
        _raise_if_cancelled(cancel_check)
        total, _baseline, _components = evaluate(parameters)
        residual = intensity - total
        if np.all(np.isfinite(residual)):
            return residual
        # Keep the optimizer in a finite residual space if an extreme trial
        # polynomial coefficient overflows.  The returned model is still
        # checked below before it is exposed to callers.
        return np.nan_to_num(residual, nan=1.0e100, posinf=1.0e100, neginf=-1.0e100)

    optimization_error: Exception | None = None
    try:
        optimization = least_squares(
            residual_function,
            initial_array,
            bounds=(lower_array, upper_array),
            x_scale="jac",
            max_nfev=resolved_max_nfev,
            ftol=1.0e-12,
            xtol=1.0e-12,
            gtol=1.0e-12,
        )
        fitted_parameters = np.asarray(optimization.x, dtype=float)
        converged = bool(
            optimization.success and np.all(np.isfinite(fitted_parameters))
        )
        message = str(optimization.message)
        cost = float(optimization.cost) if np.isfinite(optimization.cost) else None
        nfev = int(optimization.nfev)
        optimizer_status = int(optimization.status)
    except PeakFitCancelled:
        raise
    except Exception as exc:  # pragma: no cover - defensive SciPy boundary
        optimization_error = exc
        fitted_parameters = initial_array
        converged = False
        message = f"peak optimizer failed: {exc}"
        cost = None
        nfev = None
        optimizer_status = None

    fitted, fitted_baseline, fitted_components = evaluate(fitted_parameters)
    if not all(
        np.all(np.isfinite(values))
        for values in (fitted, fitted_baseline, *fitted_components)
    ):
        # This should only be reachable for a pathological optimizer result;
        # preserve the convergence signal while providing finite outputs.
        converged = False
        message = f"{message}; non-finite model was replaced by the initial model"
        fitted_parameters = initial_array
        fitted, fitted_baseline, fitted_components = evaluate(fitted_parameters)
    residual = intensity - fitted
    if not np.all(np.isfinite(residual)):
        converged = False
        residual = np.nan_to_num(residual, nan=0.0, posinf=0.0, neginf=0.0)

    fitted_peaks: list[tuple[float, float, float, float | None, str, np.ndarray]] = []
    cursor = normalized_order + 1
    for index, (seed, component) in enumerate(
        zip(parsed_seeds, fitted_components), start=1
    ):
        center = float(fitted_parameters[cursor])
        amplitude = float(fitted_parameters[cursor + 1])
        width = float(fitted_parameters[cursor + 2])
        cursor += 3
        eta = None
        if normalized_profile == "pseudo_voigt":
            eta = float(fitted_parameters[cursor])
            cursor += 1
        name = seed.name or f"peak_{index}"
        fitted_peaks.append((center, amplitude, width, eta, name, component))
    fitted_peaks.sort(key=lambda item: item[0])

    summaries: list[PeakSummary] = []
    components: list[np.ndarray] = []
    for center, amplitude, width, eta, name, component in fitted_peaks:
        summaries.append(
            PeakSummary(
                name=name,
                center=center,
                fwhm=width,
                area=_peak_area(amplitude, width, normalized_profile, eta),
                amplitude=amplitude,
                eta=eta,
                profile=normalized_profile,
            )
        )
        components.append(component)

    # ``optimization_error`` is intentionally read only for clarity: the
    # message and convergence flag above are the stable public failure state.
    _ = optimization_error
    return PeakDecompositionResult(
        x=axis,
        y=intensity,
        fitted=fitted,
        baseline=fitted_baseline,
        components=components,
        residual=residual,
        summaries=summaries,
        converged=converged,
        profile=normalized_profile,
        baseline_order=normalized_order,
        message=message,
        cost=cost,
        nfev=nfev,
        optimizer_status=optimizer_status,
    )


def guess_peak_seeds(
    x: Sequence[float] | np.ndarray,
    y: Sequence[float] | np.ndarray,
    *,
    max_peaks: int = 3,
    minimum_prominence_fraction: float = 0.05,
) -> list[PeakSeed]:
    """Suggest deterministic peak seeds from a finite, increasing spectrum.

    The helper is intentionally only an initializer for :func:`fit_peaks`.
    It ranks local maxima by prominence, keeps at most ``max_peaks``, and
    estimates FWHM from the half-prominence crossings.  The caller still sees
    and controls the resulting peak count; these guesses are not phase
    identification or scientific peak assignments.
    """

    axis = _input_array(x, "x")
    intensity = _input_array(y, "y")
    if axis.size != intensity.size:
        raise ValueError("x and y must contain the same number of values")
    if axis.size < 3:
        raise ValueError("Peak seed guessing requires at least three data points")
    if np.any(np.diff(axis) <= 0.0):
        raise ValueError("x values must be strictly increasing")
    if isinstance(max_peaks, bool):
        raise ValueError("max_peaks must be a positive integer")
    try:
        parsed_max_peaks = int(max_peaks)
    except (TypeError, ValueError) as exc:
        raise ValueError("max_peaks must be a positive integer") from exc
    if parsed_max_peaks != max_peaks or parsed_max_peaks < 1:
        raise ValueError("max_peaks must be a positive integer")
    prominence_fraction = _finite_float(
        minimum_prominence_fraction, "minimum_prominence_fraction"
    )
    if not 0.0 < prominence_fraction <= 1.0:
        raise ValueError("minimum_prominence_fraction must be in (0, 1]")

    intensity_span = float(np.ptp(intensity))
    if intensity_span <= 0.0:
        raise ValueError("Peak seed guessing requires a non-constant spectrum")
    indices, properties = find_peaks(
        intensity,
        prominence=intensity_span * prominence_fraction,
    )
    if indices.size == 0:
        raise ValueError("No peaks meet the requested prominence threshold")
    prominences = np.asarray(properties["prominences"], dtype=float)
    ranked = np.argsort(prominences)[::-1][:parsed_max_peaks]
    chosen_indices = indices[ranked]
    chosen_prominences = prominences[ranked]
    widths, _heights, left_ips, right_ips = peak_widths(
        intensity, chosen_indices, rel_height=0.5
    )
    sample_index = np.arange(axis.size, dtype=float)
    left_x = np.interp(left_ips, sample_index, axis)
    right_x = np.interp(right_ips, sample_index, axis)
    minimum_step = float(np.min(np.diff(axis)))
    baseline = float(np.percentile(intensity, 20.0))
    guessed: list[PeakSeed] = []
    for index, prominence, left, right, sample_width in zip(
        chosen_indices,
        chosen_prominences,
        left_x,
        right_x,
        widths,
    ):
        width = max(
            float(right - left), minimum_step, float(sample_width) * minimum_step
        )
        amplitude = max(float(intensity[index] - baseline), float(prominence), _EPS)
        guessed.append(
            PeakSeed(center=float(axis[index]), amplitude=amplitude, width=width)
        )
    guessed.sort(key=lambda seed: seed.center)
    return guessed


def decomposition_to_pattern_fit(
    result: PeakDecompositionResult,
    *,
    name: str = "Peak decomposition",
    source_path: str = "",
    axis_kind: str = "two_theta",
    wavelength_angstrom: float | None = None,
):
    """Convert a decomposition into XRDViz's persisted fit/result model."""

    from xrdviz.fit import FitComponent, PatternFit

    components = []
    for summary, values in zip(result.summaries, result.components):
        components.append(
            FitComponent(
                name=summary.name,
                y=values.tolist(),
                center=summary.center,
                fwhm=summary.fwhm,
                area=summary.area,
                amplitude=summary.amplitude,
                eta=summary.eta,
                profile=summary.profile,
            )
        )
    return PatternFit(
        name=name,
        x=result.x.tolist(),
        observed=result.y.tolist(),
        calculated=result.fitted.tolist(),
        background=result.baseline.tolist(),
        components=components,
        source_path=source_path,
        axis_kind=axis_kind,
        fit_kind="peak_decomposition",
        converged=result.converged,
        fit_message=result.message,
        wavelength_angstrom=wavelength_angstrom,
    )


def _finite_float(value: Any, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be finite") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{label} must be finite")
    return parsed


def _input_array(values: Sequence[float] | np.ndarray, label: str) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain finite numeric values") from exc
    if array.ndim != 1:
        raise ValueError(f"{label} must be one-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} values must be finite")
    return array.copy()


def _result_array(values: Any, label: str) -> np.ndarray:
    array = _input_array(values, label)
    return array


def _normalize_profile(profile: str) -> str:
    if not isinstance(profile, str):
        raise ValueError("profile must be gaussian, lorentzian, or pseudo_voigt")
    normalized = profile.strip().lower().replace("-", "_")
    if normalized not in _SUPPORTED_PROFILES:
        raise ValueError("profile must be gaussian, lorentzian, or pseudo_voigt")
    return normalized


def _validate_baseline_order(order: int) -> int:
    if isinstance(order, bool):
        raise ValueError("baseline_order must be 0, 1, or 2")
    try:
        parsed = int(order)
    except (TypeError, ValueError) as exc:
        raise ValueError("baseline_order must be 0, 1, or 2") from exc
    if parsed != order or parsed not in (0, 1, 2):
        raise ValueError("baseline_order must be 0, 1, or 2")
    return parsed


def _parse_seeds(seeds: Sequence[PeakSeed | Mapping[str, Any]]) -> list[PeakSeed]:
    if isinstance(seeds, (str, bytes)):
        raise ValueError("seeds must be a sequence of PeakSeed values")
    try:
        values = list(seeds)
    except TypeError as exc:
        raise ValueError("seeds must be a sequence of PeakSeed values") from exc
    parsed: list[PeakSeed] = []
    for seed in values:
        if isinstance(seed, PeakSeed):
            parsed.append(seed)
        elif isinstance(seed, Mapping):
            try:
                parsed.append(PeakSeed(**seed))
            except TypeError as exc:
                raise ValueError(
                    "seed mappings must contain center, amplitude, and width"
                ) from exc
        else:
            raise ValueError("seeds must contain PeakSeed values")
    names = [seed.name.casefold() for seed in parsed if seed.name]
    if len(names) != len(set(names)):
        raise ValueError("Peak seed names must be unique")
    return parsed


def _validate_max_nfev(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("max_nfev must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("max_nfev must be a positive integer") from exc
    if parsed != value or parsed < 1:
        raise ValueError("max_nfev must be a positive integer")
    return parsed


def _raise_if_cancelled(cancel_check: Callable[[], bool] | None) -> None:
    if cancel_check is not None and bool(cancel_check()):
        raise PeakFitCancelled("Peak fitting was cancelled")


def _initial_baseline(
    normalized_x: np.ndarray,
    intensity: np.ndarray,
    axis: np.ndarray,
    seeds: Sequence[PeakSeed],
    order: int,
) -> np.ndarray:
    if order == 0:
        # A low quantile is less biased by positive diffraction peaks than a
        # mean while remaining stable for a short observed range.
        return np.asarray([float(np.percentile(intensity, 20.0))], dtype=float)
    mask = np.ones(axis.size, dtype=bool)
    minimum_step = float(np.min(np.diff(axis)))
    for seed in seeds:
        exclusion = max(2.0 * seed.width, 3.0 * minimum_step)
        mask &= np.abs(axis - seed.center) > exclusion
    if int(np.count_nonzero(mask)) < order + 1:
        # A polynomial fit over all points is only a starting point; the
        # peak terms are still refined simultaneously by least squares.
        mask = np.ones(axis.size, dtype=bool)
    try:
        coefficients = np.polynomial.polynomial.polyfit(
            normalized_x[mask], intensity[mask], order
        )
    except (ArithmeticError, np.linalg.LinAlgError, ValueError):
        coefficients = np.zeros(order + 1, dtype=float)
        coefficients[0] = float(np.percentile(intensity, 20.0))
    coefficients = np.asarray(coefficients, dtype=float)
    if coefficients.size != order + 1 or not np.all(np.isfinite(coefficients)):
        coefficients = np.zeros(order + 1, dtype=float)
        coefficients[0] = float(np.percentile(intensity, 20.0))
    return coefficients


def _profile_values(
    x: np.ndarray,
    center: float,
    amplitude: float,
    width: float,
    profile: str,
    eta: float | None,
) -> np.ndarray:
    z = (x - center) / width
    gaussian = np.exp(-4.0 * math.log(2.0) * z**2)
    if profile == "gaussian":
        return amplitude * gaussian
    lorentzian = 1.0 / (1.0 + 4.0 * z**2)
    if profile == "lorentzian":
        return amplitude * lorentzian
    assert eta is not None
    return amplitude * ((1.0 - eta) * gaussian + eta * lorentzian)


def _peak_area(
    amplitude: float, width: float, profile: str, eta: float | None
) -> float:
    if profile == "gaussian":
        factor = _GAUSSIAN_AREA_FACTOR
    elif profile == "lorentzian":
        factor = math.pi / 2.0
    else:
        assert eta is not None
        factor = (1.0 - eta) * _GAUSSIAN_AREA_FACTOR + eta * (math.pi / 2.0)
    return float(amplitude * width * factor)


__all__ = [
    "DEFAULT_PEAK_FIT_MAX_NFEV",
    "PeakFitCancelled",
    "PeakSeed",
    "PeakSummary",
    "PeakDecompositionResult",
    "fit_peaks",
    "guess_peak_seeds",
    "decomposition_to_pattern_fit",
]
