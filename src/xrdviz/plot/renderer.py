from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from xrdviz.axes import convert_x
from xrdviz.batch import (
    color_for_value,
    colorbar_label,
    layer_batch_value,
    make_heatmap_matrix,
    matplotlib_colormap,
    select_spectrum_layers,
)
from xrdviz.cif import phase_peak_position_for_axis
from xrdviz.models import PLOT_AXIS_COLOR, PLOT_MUTED_COLOR, PLOT_TEXT_COLOR, ProjectState, default_axis_label
from xrdviz.plot.style import apply_matplotlib_style
from xrdviz.transforms import display_y_for_layer


def render_project(state: ProjectState, figure: Any | None = None) -> tuple[Any, dict[str, Any]]:
    Figure = _figure_class()
    settings = state.settings
    apply_matplotlib_style(settings)

    fig = figure or Figure(figsize=(settings.figure_width_in, settings.figure_height_in), dpi=settings.dpi)
    fig.clear()

    main_ax = fig.add_subplot(111)
    main_ax.set_facecolor("white")
    axes: dict[str, Any] = {"main": main_ax, "bragg": main_ax}
    if settings.view_mode == "heatmap":
        heatmap_artist, colorbar_ax = _draw_heatmap(main_ax, state, fig)
        spectrum_handles = []
        phase_handles = []
        if heatmap_artist is not None:
            axes["heatmap"] = heatmap_artist
        if colorbar_ax is not None:
            axes["colorbar"] = colorbar_ax
    else:
        spectrum_handles, colorbar_ax = _draw_spectra(main_ax, state, fig)
        if colorbar_ax is not None:
            axes["colorbar"] = colorbar_ax
        x_range, exact_x_range = _display_x_range(state)
        _apply_x_range(main_ax, x_range, exact=exact_x_range)
        phase_handles = _draw_bragg_band(main_ax, state)
        _apply_x_range(main_ax, x_range, exact=exact_x_range)

    x_range, exact_x_range = _display_x_range(state)
    _apply_x_range(main_ax, x_range, exact=exact_x_range)

    main_ax.set_xlabel(settings.x_label or default_axis_label(settings.x_axis))
    main_ax.set_ylabel(colorbar_label(settings.sort_by) if settings.view_mode == "heatmap" else settings.y_label)
    if settings.panel_title:
        main_ax.text(
            0.03,
            0.95,
            settings.panel_title,
            transform=main_ax.transAxes,
            ha="left",
            va="top",
            fontsize=max(settings.axis_label_size, settings.font_size),
            fontweight="bold",
            color=PLOT_TEXT_COLOR,
        )
    _polish_main_axis(main_ax, state, spectrum_handles, phase_handles)
    fig.subplots_adjust(
        left=settings.margin_left,
        right=settings.margin_right,
        top=settings.margin_top,
        bottom=settings.margin_bottom,
    )
    return fig, axes


def export_project(state: ProjectState, path: str | Path) -> None:
    output = Path(path)
    fig, _axes = render_project(state)
    fig.savefig(output, dpi=state.settings.dpi, bbox_inches="tight")


def _draw_spectra(ax: Any, state: ProjectState, fig: Any) -> tuple[list[Any], Any | None]:
    settings = state.settings
    visible_layers = select_spectrum_layers(state.spectra, show_every_n=settings.show_every_n)
    gradient_values = [
        layer_batch_value(layer, settings.color_by) if layer_batch_value(layer, settings.color_by) is not None else float(layer.order)
        for layer in visible_layers
    ]
    finite_values = [value for value in gradient_values if value is not None and math.isfinite(value)]
    value_min = min(finite_values) if finite_values else 0.0
    value_max = max(finite_values) if finite_values else 1.0
    handles = []
    for index, layer in enumerate(visible_layers):
        x_values = convert_x(layer.x, layer.axis_kind, settings.x_axis, settings.energy_kev)
        y_values = display_y_for_layer(layer, settings, index)
        pairs = [
            (x_value, y_value)
            for x_value, y_value in zip(x_values, y_values)
            if math.isfinite(x_value) and math.isfinite(y_value)
        ]
        if not pairs:
            continue
        x_clean, y_clean = zip(*pairs)
        color = layer.color
        if settings.view_mode == "gradient_stack":
            value = gradient_values[index] if index < len(gradient_values) else float(index)
            color = color_for_value(float(value), value_min, value_max, colormap=settings.colormap)
        (handle,) = ax.plot(
            x_clean,
            y_clean,
            color=color,
            linewidth=layer.linewidth,
            alpha=0.96,
            solid_capstyle="round",
            solid_joinstyle="round",
            label=layer.name,
        )
        handles.append(handle)
        if settings.direct_labels:
            ax.text(x_clean[-1], y_clean[-1], f"  {layer.name}", va="center", ha="left", fontsize=settings.tick_label_size, color=color)
    colorbar_ax = None
    if settings.show_colorbar and settings.view_mode == "gradient_stack" and finite_values:
        colorbar_ax = _add_scalar_colorbar(fig, ax, settings.colormap, value_min, value_max, colorbar_label(settings.color_by))
    return handles, colorbar_ax


def _draw_heatmap(ax: Any, state: ProjectState, fig: Any) -> tuple[Any | None, Any | None]:
    settings = state.settings
    x_grid, row_values, matrix = make_heatmap_matrix(state)
    if matrix.size == 0:
        return None, None
    y_min, y_max = -0.5, matrix.shape[0] - 0.5
    image = ax.imshow(
        matrix,
        extent=(float(x_grid[0]), float(x_grid[-1]), y_min, y_max),
        origin="lower",
        aspect="auto",
        cmap=matplotlib_colormap(settings.colormap),
        interpolation="nearest",
    )
    ax.set_yticks(list(range(len(row_values))))
    ax.set_yticklabels([_format_metadata_tick(value) for value in row_values])
    colorbar_ax = None
    if settings.show_colorbar:
        colorbar_ax = fig.colorbar(image, ax=ax, pad=0.02, fraction=0.045).ax
        colorbar_ax.set_ylabel("Intensity", color=PLOT_TEXT_COLOR)
        colorbar_ax.tick_params(colors=PLOT_TEXT_COLOR, width=0.55, length=2.5)
    return image, colorbar_ax


def _format_metadata_tick(value: float) -> str:
    if math.isfinite(value) and abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.4g}"


def _draw_bragg_band(ax: Any, state: ProjectState) -> list[Any]:
    visible_phases = [phase for phase in state.phases if phase.visible and phase.peaks]
    if not visible_phases:
        return []

    phase_handles = []
    x_min, x_max = ax.get_xlim()
    data_bottom, data_top = ax.get_ylim()
    data_span = data_top - data_bottom
    if not math.isfinite(data_span) or data_span <= 0:
        data_span = 1.0
        data_bottom = 0.0
        data_top = 1.0

    band_span = data_span * max(settings_bragg_fraction(state), 0.06)
    row_height = band_span / max(len(visible_phases), 1)
    band_bottom = data_bottom - band_span
    tick_linewidth = max(0.55, state.settings.line_width * 0.75)

    for index, phase in enumerate(visible_phases):
        baseline = band_bottom + row_height * index + row_height * 0.18
        tick_top = baseline + row_height * 0.62 * max(min(phase.tick_height, 1.0), 0.15)
        characteristic = _characteristic_peaks(phase.peaks)
        ax.text(
            -0.015,
            baseline + row_height * 0.31,
            phase.phase or phase.name,
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=state.settings.tick_label_size,
            color=phase.color,
            fontweight="normal",
            clip_on=False,
        )
        for peak in phase.peaks:
            x_value = phase_peak_position_for_axis(phase, peak, state.settings.x_axis, state.settings.energy_kev)
            if not math.isfinite(x_value) or x_value < x_min or x_value > x_max:
                continue
            ax.vlines(x_value, baseline, tick_top, color=phase.color, linewidth=tick_linewidth, alpha=0.95, zorder=2)
            if peak in characteristic and phase.label_policy != "none":
                label = peak.label or peak.hkl or phase.phase or phase.name
                ax.text(
                    x_value,
                    tick_top + row_height * 0.08,
                    label,
                    rotation=90,
                    ha="center",
                    va="bottom",
                    fontsize=state.settings.tick_label_size,
                    color=phase.color,
                    clip_on=True,
                )
            if phase.show_guides and peak in characteristic:
                ax.axvline(x_value, color=phase.color, linestyle="--", linewidth=0.55, alpha=0.36, zorder=1)
        if state.settings.show_phase_legend:
            phase_handles.append(_phase_legend_handle(phase.color, phase.phase or phase.name))

    ax.axhline(data_bottom, color=PLOT_MUTED_COLOR, linewidth=0.45, alpha=0.55, zorder=1)
    ax.set_ylim(band_bottom - row_height * 0.08, data_top + data_span * 0.04)
    return phase_handles


def _polish_main_axis(ax: Any, state: ProjectState, spectrum_handles: list[Any], phase_handles: list[Any]) -> None:
    settings = state.settings
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(min(max(settings.line_width, 0.6), 0.8))
        spine.set_color(PLOT_AXIS_COLOR)
    ax.grid(False)
    ax.tick_params(
        axis="both",
        colors=PLOT_TEXT_COLOR,
        width=0.65,
        length=3.0,
        direction="out",
        top=False,
        right=False,
    )
    ax.tick_params(
        axis="y",
        left=settings.show_y_tick_labels,
        right=False,
        labelleft=settings.show_y_tick_labels,
    )
    handles = []
    if settings.show_legend:
        handles.extend(spectrum_handles)
    if settings.show_phase_legend:
        handles.extend(phase_handles)
    if handles and settings.legend_location != "none":
        legend_kwargs = {
            "handles": handles,
            "frameon": False,
            "handlelength": 1.6,
            "handletextpad": 0.5,
            "borderaxespad": 0.35,
        }
        if settings.legend_location == "outside right":
            legend_kwargs.update({"loc": "center left", "bbox_to_anchor": (1.02, 0.5)})
        else:
            legend_kwargs["loc"] = settings.legend_location
        ax.legend(**legend_kwargs)


def _add_scalar_colorbar(fig: Any, ax: Any, colormap: str, vmin: float, vmax: float, label: str) -> Any:
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    mappable = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=matplotlib_colormap(colormap))
    mappable.set_array([])
    colorbar = fig.colorbar(mappable, ax=ax, pad=0.02, fraction=0.045)
    colorbar.ax.set_ylabel(label, color=PLOT_TEXT_COLOR)
    colorbar.ax.tick_params(colors=PLOT_TEXT_COLOR, width=0.55, length=2.5)
    return colorbar.ax


def settings_bragg_fraction(state: ProjectState) -> float:
    return min(max(state.settings.bragg_band_height, 0.05), 0.45)


def _visible_spectrum_x_range(state: ProjectState) -> tuple[float, float] | None:
    x_values: list[float] = []
    for layer in state.spectra:
        if not layer.visible:
            continue
        converted = convert_x(layer.x, layer.axis_kind, state.settings.x_axis, state.settings.energy_kev)
        x_values.extend(value for value in converted if math.isfinite(value))
    if not x_values:
        return None
    return min(x_values), max(x_values)


def _display_x_range(state: ProjectState) -> tuple[tuple[float, float] | None, bool]:
    settings = state.settings
    if settings.x_min is None and settings.x_max is None:
        return _visible_spectrum_x_range(state), False
    auto = _visible_spectrum_x_range(state)
    if settings.x_min is not None and settings.x_max is not None:
        return (settings.x_min, settings.x_max), True
    if auto is None:
        return None, False
    x_min, x_max = auto
    return (settings.x_min if settings.x_min is not None else x_min, settings.x_max if settings.x_max is not None else x_max), True


def _apply_x_range(ax: Any, x_range: tuple[float, float] | None, *, exact: bool) -> None:
    if x_range is None:
        return
    x_min, x_max = x_range
    if not math.isfinite(x_min) or not math.isfinite(x_max) or x_min >= x_max:
        return
    ax.set_xlim((x_min, x_max) if exact else _range_with_padding(x_min, x_max))


def _range_with_padding(x_min: float, x_max: float) -> tuple[float, float]:
    if x_min == x_max:
        pad = max(abs(x_min) * 0.05, 0.5)
    else:
        pad = abs(x_max - x_min) * 0.03
    return x_min - pad, x_max + pad


def _phase_legend_handle(color: str, label: str):
    from matplotlib.lines import Line2D

    return Line2D([0], [0], color=color, marker="|", linestyle="None", markersize=7, markeredgewidth=1.0, label=label)


def _marker_for_shape(shape: str) -> str:
    return {
        "circle": "o",
        "triangle": "^",
        "square": "s",
        "diamond": "D",
        "star": "*",
        "line": "|",
    }.get(shape, "|")


def _characteristic_peaks(peaks):
    sorted_peaks = sorted(peaks, key=lambda peak: peak.intensity, reverse=True)
    return sorted_peaks[: min(3, len(sorted_peaks))]


def _figure_class():
    try:
        from matplotlib.figure import Figure
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for rendering and export") from exc
    return Figure
