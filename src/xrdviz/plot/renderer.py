from __future__ import annotations

import io
import math
from pathlib import Path
from typing import Any

import numpy as np

from xrdviz.axes import convert_x
from xrdviz.batch import (
    color_for_value,
    colorbar_label,
    layer_batch_value,
    make_heatmap_matrix,
    matplotlib_colormap,
    select_spectrum_layers,
    single_temperature_unit,
)
from xrdviz.cif import phase_peak_position_for_axis
from xrdviz.models import PLOT_AXIS_COLOR, PLOT_MUTED_COLOR, PLOT_TEXT_COLOR, ProjectState, default_axis_label
from xrdviz.plot.style import apply_matplotlib_style
from xrdviz.transforms import display_y_for_layer


def render_project(state: ProjectState, figure: Any | None = None) -> tuple[Any, dict[str, Any]]:
    Figure = _figure_class()
    settings = state.settings
    apply_matplotlib_style(settings)
    _validate_outside_legend_layout(state)

    fig = (
        figure
        if figure is not None
        else Figure(figsize=(settings.figure_width_in, settings.figure_height_in), dpi=settings.dpi)
    )
    # A caller may reuse a Figure for previews.  Keep its physical canvas and
    # raster resolution in sync with the current project settings before
    # clearing and rebuilding the axes.
    fig.set_size_inches(settings.figure_width_in, settings.figure_height_in, forward=True)
    fig.set_dpi(settings.dpi)
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
    metadata_layers = select_spectrum_layers(state.spectra, show_every_n=settings.show_every_n)
    main_ax.set_ylabel(
        _metadata_label(settings.sort_by, metadata_layers)
        if settings.view_mode == "heatmap"
        else settings.y_label
    )
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
        right=_layout_right_margin(settings, legend=main_ax.get_legend()),
        top=settings.margin_top,
        bottom=settings.margin_bottom,
    )
    return fig, axes


def export_project(state: ProjectState, path: str | Path) -> None:
    output = Path(path)
    fig, _axes = render_project(state)
    # Keep the requested physical canvas.  ``bbox_inches="tight"`` crops the
    # page and makes the exported dimensions depend on the drawn artists.
    fig.savefig(output, dpi=state.settings.dpi, facecolor="white", edgecolor="white")
    if output.suffix.lower() in {".png", ".tif", ".tiff"}:
        _ensure_opaque_raster(output, state.settings.dpi)


def _draw_spectra(ax: Any, state: ProjectState, fig: Any) -> tuple[list[Any], Any | None]:
    settings = state.settings
    visible_layers = select_spectrum_layers(state.spectra, show_every_n=settings.show_every_n)
    temperature_unit = single_temperature_unit(visible_layers) if settings.color_by == "temperature" else None
    gradient_values = [
        layer_batch_value(layer, settings.color_by, temperature_unit=temperature_unit)
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
            value = gradient_values[index] if index < len(gradient_values) else None
            color = color_for_value(value, value_min, value_max, colormap=settings.colormap)
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
            ax.text(
                x_clean[-1],
                y_clean[-1],
                f"  {layer.name}",
                va="center",
                ha="left",
                fontsize=settings.tick_label_size,
                color=PLOT_TEXT_COLOR,
            )
    colorbar_ax = None
    if settings.show_colorbar and settings.view_mode == "gradient_stack" and finite_values:
        colorbar_ax = _add_scalar_colorbar(
            fig,
            ax,
            settings.colormap,
            value_min,
            value_max,
            _metadata_label(settings.color_by, visible_layers),
        )
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
    tick_positions = _heatmap_tick_positions(len(row_values))
    ax.set_yticks(tick_positions)
    ax.set_yticklabels([_format_metadata_tick(row_values[position]) for position in tick_positions])
    colorbar_ax = None
    if settings.show_colorbar:
        colorbar_ax = fig.colorbar(image, ax=ax, pad=0.02, fraction=0.045).ax
        colorbar_ax.set_ylabel(_intensity_colorbar_label(settings), color=PLOT_TEXT_COLOR)
        colorbar_ax.tick_params(colors=PLOT_TEXT_COLOR, width=0.55, length=2.5)
    return image, colorbar_ax


def _format_metadata_tick(value: float) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.4g}"


def _heatmap_tick_positions(row_count: int, *, max_ticks: int = 7) -> list[int]:
    if row_count <= 0:
        return []
    if row_count <= max_ticks:
        return list(range(row_count))
    # Keep the first and last metadata values visible while distributing the
    # remaining labels over the full ordinal row range.
    positions = np.linspace(0, row_count - 1, num=max_ticks, dtype=int).tolist()
    return list(dict.fromkeys(positions))


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
            color=PLOT_TEXT_COLOR,
            fontweight="normal",
            clip_on=False,
        )
        for peak in phase.peaks:
            x_value = phase_peak_position_for_axis(phase, peak, state.settings.x_axis, state.settings.energy_kev)
            if not math.isfinite(x_value) or x_value < x_min or x_value > x_max:
                continue
            ax.vlines(x_value, baseline, tick_top, color=phase.color, linewidth=tick_linewidth, alpha=0.95, zorder=2)
            marker_shape = str(phase.marker_shape or "line").strip().lower()
            if marker_shape != "line":
                ax.plot(
                    [x_value],
                    [tick_top],
                    linestyle="None",
                    marker=_marker_for_shape(marker_shape),
                    color=phase.color,
                    markersize=max(4.0, state.settings.tick_label_size * 0.9),
                    markeredgewidth=max(0.55, tick_linewidth),
                    clip_on=True,
                    zorder=3,
                )
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
                    color=PLOT_TEXT_COLOR,
                    clip_on=True,
                )
            if phase.show_guides and peak in characteristic:
                ax.axvline(x_value, color=phase.color, linestyle="--", linewidth=0.55, alpha=0.36, zorder=1)
        if state.settings.show_phase_legend:
            phase_handles.append(_phase_legend_handle(phase.color, phase.phase or phase.name, phase.marker_shape))

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
    show_y_ticks = settings.show_y_tick_labels or settings.view_mode == "heatmap"
    ax.tick_params(
        axis="y",
        left=show_y_ticks,
        right=False,
        labelleft=show_y_ticks,
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
            legend_kwargs.update(
                {
                    "labels": [_wrap_legend_label(handle.get_label(), settings) for handle in handles],
                    "loc": "center left",
                    "bbox_to_anchor": (1.02, 0.5),
                }
            )
        else:
            legend_kwargs["loc"] = settings.legend_location
        legend = ax.legend(**legend_kwargs)
        for text in legend.get_texts():
            text.set_color(PLOT_TEXT_COLOR)


def _add_scalar_colorbar(fig: Any, ax: Any, colormap: str, vmin: float, vmax: float, label: str) -> Any:
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    mappable = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=matplotlib_colormap(colormap))
    mappable.set_array([])
    colorbar = fig.colorbar(mappable, ax=ax, pad=0.02, fraction=0.045)
    colorbar.ax.set_ylabel(label, color=PLOT_TEXT_COLOR)
    colorbar.ax.tick_params(colors=PLOT_TEXT_COLOR, width=0.55, length=2.5)
    return colorbar.ax


def _metadata_label(field: str, layers) -> str:
    unit = single_temperature_unit(layers) if field == "temperature" else None
    return colorbar_label(field, temperature_unit=unit)


def _intensity_colorbar_label(settings) -> str:
    if settings.normalize and settings.log_scale:
        return "log10(normalized intensity)"
    if settings.normalize:
        return "Normalized intensity"
    if settings.log_scale:
        return "log10(intensity)"
    return "Intensity"


def _layout_right_margin(settings, *, legend=None) -> float:
    preferred = settings.margin_right
    if settings.legend_location == "outside right" and legend is not None:
        figure_width_pt = max(float(settings.figure_width_in) * 72.0, 1.0)
        text_width_pt = _legend_text_width_points(legend, settings)
        decoration_width_pt = settings.tick_label_size * (1.6 + 0.5 + 1.0) + 6.0
        required_fraction = (text_width_pt + decoration_width_pt) / figure_width_pt + 0.015
        preferred = min(preferred, settings.margin_right - required_fraction)
    if settings.direct_labels:
        preferred = min(preferred, 0.84)
    if settings.show_colorbar and settings.view_mode in {"gradient_stack", "heatmap"}:
        # Matplotlib creates the colorbar before ``subplots_adjust``.  Reserve
        # enough right-side canvas for its tick labels and vertical title;
        # otherwise they are silently clipped from narrow publication figures.
        preferred = min(preferred, 0.84)
    return max(settings.margin_left + 0.12, preferred)


def _validate_outside_legend_layout(state: ProjectState) -> None:
    settings = state.settings
    if settings.legend_location != "outside right" or settings.view_mode == "heatmap":
        return
    labels: list[str] = []
    if settings.show_legend:
        labels.extend(
            layer.name
            for layer in select_spectrum_layers(state.spectra, show_every_n=settings.show_every_n)
        )
    if settings.show_phase_legend:
        labels.extend(phase.phase or phase.name for phase in state.phases if phase.visible and phase.peaks)
    if not labels:
        return
    if settings.show_colorbar and settings.view_mode == "gradient_stack":
        raise ValueError(
            "Outside-right legend cannot share the publication gutter with a colorbar; "
            "choose an inside legend position or disable the spectrum legend."
        )

    wrapped_labels = [_wrap_legend_label(label, settings) for label in labels]
    font_size = float(settings.tick_label_size)
    line_height_pt = font_size * 1.25
    label_spacing_pt = font_size * 0.5
    legend_height_pt = sum(
        max(len(label.splitlines()), 1) * line_height_pt
        for label in wrapped_labels
    )
    legend_height_pt += max(len(wrapped_labels) - 1, 0) * label_spacing_pt + font_size * 0.8
    center = (float(settings.margin_bottom) + float(settings.margin_top)) / 2.0
    available_fraction = max(2.0 * min(center, 1.0 - center) - 0.02, 0.0)
    available_height_pt = float(settings.figure_height_in) * 72.0 * available_fraction
    if legend_height_pt > available_height_pt:
        raise ValueError(
            "Outside-right legend does not fit vertically at the selected figure size; "
            "use a double-column/taller canvas, shorten or sample labels, or choose an inside legend position."
        )


def _wrap_legend_label(label: str, settings) -> str:
    text = str(label)
    max_width_pt = max(36.0, float(settings.figure_width_in) * 72.0 * 0.24)
    lines: list[str] = []
    for paragraph in text.splitlines() or [""]:
        current = ""
        for character in paragraph:
            candidate = current + character
            if current and _text_width_points(candidate, settings) > max_width_pt:
                split_at = current.rfind(" ")
                if split_at > 0:
                    lines.append(current[:split_at].rstrip())
                    current = current[split_at + 1 :] + character
                else:
                    lines.append(current.rstrip())
                    current = character.lstrip()
            else:
                current = candidate
        lines.append(current.rstrip())
    return "\n".join(line for line in lines if line) or text


def _legend_text_width_points(legend, settings) -> float:
    if legend is None:
        return 0.0
    return max(
        (
            _text_width_points(line, settings)
            for text in legend.get_texts()
            for line in text.get_text().splitlines()
        ),
        default=0.0,
    )


def _text_width_points(text: str, settings) -> float:
    if not text:
        return 0.0
    try:
        from matplotlib.font_manager import FontProperties
        from matplotlib.textpath import TextPath

        properties = FontProperties(family=settings.font_family, size=settings.tick_label_size)
        return float(TextPath((0.0, 0.0), text, prop=properties).get_extents().width)
    except (RuntimeError, TypeError, ValueError):
        return len(text) * float(settings.tick_label_size) * 0.75


def _ensure_opaque_raster(path: Path, dpi: int) -> None:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise RuntimeError("Pillow is required for opaque PNG/TIFF export") from exc

    # Open a byte copy rather than the target path itself.  Pillow's TIFF
    # reader can retain a Windows file handle even after ``Image.close()``,
    # which would prevent replacing the exported file in place.
    stream = io.BytesIO(path.read_bytes())
    try:
        with Image.open(stream) as image:
            image.load()
            image_format = image.format
            rgb = image.convert("RGB")
    finally:
        stream.close()
    try:
        rgb.save(path, format=image_format, dpi=(dpi, dpi))
    finally:
        rgb.close()


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


def _phase_legend_handle(color: str, label: str, marker_shape: str = "line"):
    from matplotlib.lines import Line2D

    marker = _marker_for_shape(marker_shape)
    return Line2D(
        [0],
        [0],
        color=color,
        marker=marker,
        linestyle="None",
        markersize=7,
        markeredgewidth=1.0,
        label=label,
    )


def _marker_for_shape(shape: str) -> str:
    shape = str(shape or "line").strip().lower()
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
