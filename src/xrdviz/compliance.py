"""Publication-oriented checks for Nature-style XRD figures.

The checks in this module intentionally inspect the project configuration and
the visible data layers only.  They do not attempt to infer whether a figure
is scientifically correct, and they do not inspect the internals of a
renderer (in particular, a heatmap is never treated as an all-vector figure).
"""

from __future__ import annotations

import math
from typing import Any

from xrdviz.axes import convert_x
from xrdviz.batch import make_heatmap_matrix, select_spectrum_layers, single_temperature_unit
from xrdviz.models import ProjectState
from xrdviz.transforms import display_y_for_layer


NATURE_TEMPLATE_WIDTHS_MM = {"nature_single": 89.0, "nature_double": 183.0}
MIN_FIGURE_HEIGHT_MM = 0.0
MAX_FIGURE_HEIGHT_MM = 170.0
MIN_FONT_SIZE_PT = 5.0
MAX_FONT_SIZE_PT = 7.0
MIN_LINE_WIDTH_PT = 0.25
MAX_LINE_WIDTH_PT = 1.0
DANGEROUS_CONTINUOUS_COLORMAPS = frozenset({"jet", "rainbow", "turbo", "blue_rose"})
NATURE_TEMPLATES = frozenset(NATURE_TEMPLATE_WIDTHS_MM)


def nature_compliance_issues(state: ProjectState) -> list[str]:
    """Return deterministic, human-readable Nature compliance issues.

    An empty list means that the current project configuration passes the
    checks.  The function is deliberately fail-closed for non-finite numeric
    values: a NaN or infinity is an issue even when ordinary range comparisons
    would otherwise skip it.
    """

    settings = state.settings
    issues: list[str] = []

    template = str(getattr(settings, "template_name", "")).strip().lower()
    expected_width_mm = NATURE_TEMPLATE_WIDTHS_MM.get(template)
    if template not in NATURE_TEMPLATES:
        issues.append(
            "template must be a Nature preset (nature_single or nature_double); "
            f"got {getattr(settings, 'template_name', '')!r}"
        )

    width_mm = _millimetres(getattr(settings, "figure_width_in", float("nan")))
    if expected_width_mm is not None:
        if not math.isfinite(width_mm) or not math.isclose(width_mm, expected_width_mm, abs_tol=0.005, rel_tol=1e-6):
            issues.append(
                f"{template} template requires figure width {expected_width_mm:g} mm; "
                f"got {_format_number(width_mm)} mm"
            )

    height_mm = _millimetres(getattr(settings, "figure_height_in", float("nan")))
    if not math.isfinite(height_mm) or height_mm <= MIN_FIGURE_HEIGHT_MM or height_mm > MAX_FIGURE_HEIGHT_MM:
        issues.append(f"figure height must be >0 and <=170 mm; got {_format_number(height_mm)} mm")

    dpi = _number(getattr(settings, "dpi", float("nan")))
    if not math.isfinite(dpi) or dpi < 300:
        issues.append(f"DPI must be at least 300; got {_format_number(dpi)}")

    show_every_n = _number(getattr(settings, "show_every_n", float("nan")))
    if not math.isfinite(show_every_n) or not show_every_n.is_integer() or show_every_n < 1:
        issues.append(f"show_every_n must be a finite integer >=1; got {_format_number(show_every_n)}")
    heatmap_points = _number(getattr(settings, "heatmap_points", float("nan")))
    if not math.isfinite(heatmap_points) or not heatmap_points.is_integer() or heatmap_points < 2:
        issues.append(f"heatmap_points must be a finite integer >=2; got {_format_number(heatmap_points)}")

    x_min = getattr(settings, "x_min", None)
    x_max = getattr(settings, "x_max", None)
    if x_min is not None and not math.isfinite(_number(x_min)):
        issues.append(f"x_min must be finite; got {_format_number(_number(x_min))}")
    if x_max is not None and not math.isfinite(_number(x_max)):
        issues.append(f"x_max must be finite; got {_format_number(_number(x_max))}")
    if (
        x_min is not None
        and x_max is not None
        and math.isfinite(_number(x_min))
        and math.isfinite(_number(x_max))
        and _number(x_min) >= _number(x_max)
    ):
        issues.append("x_min must be smaller than x_max")

    font_family = str(getattr(settings, "font_family", "")).strip().lower()
    if font_family not in {"arial", "helvetica"}:
        issues.append(
            "font family must be Arial or Helvetica; "
            f"got {getattr(settings, 'font_family', '')!r}"
        )

    for field_name, label in (
        ("font_size", "font size"),
        ("axis_label_size", "axis-label size"),
        ("tick_label_size", "tick-label size"),
    ):
        value = _number(getattr(settings, field_name, float("nan")))
        if not _in_range(value, MIN_FONT_SIZE_PT, MAX_FONT_SIZE_PT):
            issues.append(
                f"{label} must be between 5 and 7 pt; "
                f"got {_format_number(value)} pt"
            )

    settings_line_width = _number(getattr(settings, "line_width", float("nan")))
    if not _in_range(settings_line_width, MIN_LINE_WIDTH_PT, MAX_LINE_WIDTH_PT):
        issues.append(
            "settings line width must be between 0.25 and 1 pt; "
            f"got {_format_number(settings_line_width)} pt"
        )

    for layer in state.spectra:
        if not getattr(layer, "visible", False):
            continue
        line_width = _number(getattr(layer, "linewidth", float("nan")))
        if not _in_range(line_width, MIN_LINE_WIDTH_PT, MAX_LINE_WIDTH_PT):
            issues.append(
                f"visible spectrum line {getattr(layer, 'name', '')!r} width must be "
                f"between 0.25 and 1 pt; got {_format_number(line_width)} pt"
            )

    view_mode = str(getattr(settings, "view_mode", "")).strip().lower()
    colormap = str(getattr(settings, "colormap", "")).strip().lower()
    uses_temperature_metadata = getattr(settings, "sort_by", "") == "temperature" or (
        view_mode == "gradient_stack" and getattr(settings, "color_by", "") == "temperature"
    )
    mapped_temperatures = [
        layer
        for layer in state.spectra
        if getattr(layer, "visible", False)
        and getattr(layer, "temperature", None) is not None
    ]
    finite_temperatures = [
        layer
        for layer in mapped_temperatures
        if math.isfinite(_number(getattr(layer, "temperature", None)))
    ]
    if uses_temperature_metadata:
        temperature_unit = single_temperature_unit(finite_temperatures)
        if not finite_temperatures:
            issues.append("the selected temperature mapping requires finite temperature metadata")
        elif len(finite_temperatures) != len(mapped_temperatures):
            issues.append("all mapped temperature values must be finite")
        elif temperature_unit not in {"°C", "K"}:
            issues.append("all mapped temperature values must declare compatible °C or K units")
    if view_mode in {"heatmap", "gradient_stack", "map"} and colormap in DANGEROUS_CONTINUOUS_COLORMAPS:
        issues.append(
            f"colormap {getattr(settings, 'colormap', '')!r} is not allowed for "
            "heatmap/gradient views"
        )
    if view_mode == "map" and not getattr(settings, "show_colorbar", False):
        issues.append("a quantitative 2D map requires a visible intensity colorbar")
    if view_mode == "map" and state.map_data is not None and state.map_data.kind == "cake":
        if state.map_data.metadata.get("processing") == "flat_detector_cake_preview":
            issues.append(
                "the cake uses the flat-detector preview without distortion, polarization, solid-angle, "
                "or instrument-calibration corrections"
            )
    if view_mode == "refinement" and state.fit is not None and state.fit.converged is False:
        issues.append("the displayed fit is marked as not converged")
    if (
        view_mode == "gradient_stack"
        and getattr(settings, "show_colorbar", False)
        and getattr(settings, "legend_location", "") == "outside right"
        and (
            (
                getattr(settings, "show_legend", False)
                and any(getattr(layer, "visible", False) for layer in state.spectra)
            )
            or (
                getattr(settings, "show_phase_legend", False)
                and any(
                    getattr(phase, "visible", False) and bool(getattr(phase, "peaks", ()))
                    for phase in state.phases
                )
            )
        )
    ):
        issues.append("outside-right legend cannot be combined with a gradient colorbar")

    has_visible_data = (
        any(_has_visible_spectrum_data(layer) for layer in state.spectra)
        or _has_visible_fit_data(state)
        or _has_visible_map_data(state)
        or _has_visible_derived_data(state)
    )
    if not has_visible_data:
        if view_mode == "map" and state.map_data is not None:
            issues.append(
                "map data must contain at least one finite populated intensity value"
            )
        else:
            issues.append("at least one visible spectrum line with data is required")
    elif not _has_finite_display_data(state):
        issues.append("the current display range must contain finite visible spectrum data")

    return issues


def _millimetres(value: Any) -> float:
    try:
        return float(value) * 25.4
    except (TypeError, ValueError, OverflowError):
        return float("nan")


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return float("nan")


def _in_range(value: float, minimum: float, maximum: float) -> bool:
    return math.isfinite(value) and minimum <= value <= maximum


def _format_number(value: float) -> str:
    if not math.isfinite(value):
        return str(value)
    return f"{value:.4g}"


def _has_visible_spectrum_data(layer: Any) -> bool:
    if not getattr(layer, "visible", False):
        return False
    x_values = getattr(layer, "x", ())
    y_values = getattr(layer, "y", ())
    try:
        return any(
            math.isfinite(_number(x_value)) and math.isfinite(_number(y_value))
            for x_value, y_value in zip(x_values, y_values)
        )
    except TypeError:
        return False


def _has_visible_fit_data(state: ProjectState) -> bool:
    if state.settings.view_mode != "refinement" or state.fit is None:
        return False
    try:
        return sum(
            1
            for values in zip(state.fit.x, state.fit.observed, state.fit.calculated)
            if all(math.isfinite(_number(value)) for value in values)
        ) >= 2
    except TypeError:
        return False


def _has_visible_map_data(state: ProjectState) -> bool:
    if state.settings.view_mode != "map" or state.map_data is None:
        return False
    try:
        intensity_values = state.map_data.intensity.flat
        if state.map_data.counts is None:
            return bool(state.map_data.intensity.size) and any(
                math.isfinite(_number(value)) for value in intensity_values
            )
        return any(
            math.isfinite(_number(value)) and _number(count) > 0.0
            for value, count in zip(intensity_values, state.map_data.counts.flat)
        )
    except (AttributeError, TypeError):
        return False


def _has_visible_derived_data(state: ProjectState) -> bool:
    if state.settings.view_mode != "derived" or state.derived_plot is None:
        return False
    try:
        return sum(
            1
            for x_value, y_value in zip(state.derived_plot.x, state.derived_plot.y)
            if math.isfinite(_number(x_value)) and math.isfinite(_number(y_value))
        ) >= 2
    except TypeError:
        return False


def _has_finite_display_data(state: ProjectState) -> bool:
    settings = state.settings
    if settings.view_mode == "map":
        return _has_visible_map_data(state)
    if settings.view_mode == "derived":
        if not _has_visible_derived_data(state):
            return False
        pairs = [
            (float(x_value), float(y_value))
            for x_value, y_value in zip(state.derived_plot.x, state.derived_plot.y)
            if math.isfinite(_number(x_value)) and math.isfinite(_number(y_value))
        ]
        x_min = min(value[0] for value in pairs)
        x_max = max(value[0] for value in pairs)
        if settings.x_min is not None and x_max < settings.x_min:
            return False
        if settings.x_max is not None and x_min > settings.x_max:
            return False
        return True
    if settings.view_mode == "refinement":
        if state.fit is None:
            return False
        try:
            x_values = convert_x(state.fit.x, state.fit.axis_kind, settings.x_axis, settings.energy_kev)
            pairs = [
                (float(x_value), float(observed), float(calculated))
                for x_value, observed, calculated in zip(x_values, state.fit.observed, state.fit.calculated)
                if all(math.isfinite(_number(value)) for value in (x_value, observed, calculated))
            ]
        except (TypeError, ValueError, OverflowError):
            return False
        if len(pairs) < 2:
            return False
        fit_x_min = min(value[0] for value in pairs)
        fit_x_max = max(value[0] for value in pairs)
        if settings.x_min is not None and fit_x_max < settings.x_min:
            return False
        if settings.x_max is not None and fit_x_min > settings.x_max:
            return False
        return True
    if settings.view_mode == "heatmap":
        try:
            _x_grid, _row_values, matrix = make_heatmap_matrix(state)
        except (TypeError, ValueError, OverflowError):
            return False
        return bool(matrix.size) and any(math.isfinite(_number(value)) for value in matrix.flat)

    try:
        layers = select_spectrum_layers(state.spectra, show_every_n=settings.show_every_n)
    except (TypeError, ValueError, OverflowError):
        return False
    for index, layer in enumerate(layers):
        try:
            x_values = convert_x(layer.x, layer.axis_kind, settings.x_axis, settings.energy_kev)
            y_values = display_y_for_layer(layer, settings, index)
        except (TypeError, ValueError, OverflowError):
            continue
        pairs = [
            (float(x_value), float(y_value))
            for x_value, y_value in zip(x_values, y_values)
            if math.isfinite(_number(x_value)) and math.isfinite(_number(y_value))
        ]
        if len(pairs) < 2:
            continue
        x_min = min(x_value for x_value, _y_value in pairs)
        x_max = max(x_value for x_value, _y_value in pairs)
        if settings.x_min is not None and x_max < settings.x_min:
            continue
        if settings.x_max is not None and x_min > settings.x_max:
            continue
        return True
    return False
