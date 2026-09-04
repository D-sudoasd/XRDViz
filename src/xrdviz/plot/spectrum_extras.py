from __future__ import annotations

import math
from typing import Any

from xrdviz.axes import KEV_ANGSTROM, convert_x
from xrdviz.batch import select_spectrum_layers
from xrdviz.models import (
    PLOT_AXIS_COLOR,
    PLOT_MUTED_COLOR,
    PLOT_TEXT_COLOR,
    ProjectState,
    default_axis_label,
)
from xrdviz.plot.layout import (
    prepare_panel_title,
    reserve_axes_top,
    safe_subplot_margins,
    stagger_rotated_label_tops,
    wrap_text,
)
from xrdviz.transforms import display_y_for_layer


def layer_energy_kev(layer: Any, fallback_energy_kev: float) -> float:
    """Return the layer-specific photon energy when wavelength is available.

    Detector-derived spectra can carry their measured wavelength.  Converting
    such a layer with the project-wide fallback energy silently moves peaks;
    keep the fallback for legacy/imported layers that have no wavelength.
    """

    wavelength = getattr(layer, "wavelength_angstrom", None)
    if wavelength is None:
        return float(fallback_energy_kev)
    try:
        wavelength = float(wavelength)
    except (TypeError, ValueError):
        return float(fallback_energy_kev)
    if not math.isfinite(wavelength) or wavelength <= 0.0:
        return float(fallback_energy_kev)
    return KEV_ANGSTROM / wavelength


def draw_annotations(ax: Any, state: ProjectState) -> list[Any]:
    x_min, x_max = sorted(float(value) for value in ax.get_xlim())
    visible = [
        annotation
        for annotation in state.annotations
        if x_min <= float(annotation.x) <= x_max
    ]
    label_x_values, tops, header_fraction = stagger_rotated_label_tops(
        ax,
        x_values=[annotation.x for annotation in visible],
        labels=[annotation.text for annotation in visible],
        desired_tops=[annotation.y_fraction for annotation in visible],
        settings=state.settings,
    )
    reserve_axes_top(ax, header_fraction)
    artists: list[Any] = []
    for annotation, label_x, top in zip(visible, label_x_values, tops):
        line = None
        if annotation.show_line:
            line = ax.axvline(
                annotation.x,
                color=annotation.color,
                linewidth=max(0.45, state.settings.line_width * 0.7),
                linestyle="--",
                alpha=0.75,
                zorder=1,
            )
        shifted = not math.isclose(float(label_x), float(annotation.x), rel_tol=0.0, abs_tol=1e-12)
        text_artist = ax.annotate(
            annotation.text,
            xy=(annotation.x, top - 0.012),
            xycoords=ax.get_xaxis_transform(),
            xytext=(label_x, top),
            textcoords=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            rotation=90,
            fontsize=state.settings.tick_label_size,
            color=annotation.color,
            arrowprops=(
                {
                    "arrowstyle": "-",
                    "color": annotation.color,
                    "linewidth": max(0.4, state.settings.line_width * 0.55),
                    "alpha": 0.75,
                    "shrinkA": 1.5,
                    "shrinkB": 1.5,
                }
                if shifted
                else None
            ),
            clip_on=True,
            zorder=5,
        )
        artists.append((line, text_artist))
    return artists


def draw_inset(main_ax: Any, state: ProjectState) -> Any:
    settings = state.settings
    inset_ax = main_ax.inset_axes([0.57, 0.54, 0.39, 0.39])
    layers = select_spectrum_layers(state.spectra, show_every_n=settings.show_every_n)
    for index, layer in enumerate(layers):
        x_values = convert_x(
            layer.x,
            layer.axis_kind,
            settings.x_axis,
            layer_energy_kev(layer, settings.energy_kev),
        )
        y_values = display_y_for_layer(layer, settings, index)
        finite_points = [
            (x, y)
            for x, y in zip(x_values, y_values)
            if math.isfinite(float(x))
            and math.isfinite(float(y))
            and settings.inset_x_min <= float(x) <= settings.inset_x_max
        ]
        if finite_points:
            x_clean, y_clean = zip(*finite_points)
            inset_ax.plot(
                x_clean,
                y_clean,
                color=layer.color,
                linewidth=max(0.45, layer.linewidth * 0.8),
            )
    inset_ax.set_xlim(settings.inset_x_min, settings.inset_x_max)
    inset_ax.tick_params(
        axis="both",
        direction="out",
        labelsize=max(4.0, settings.tick_label_size - 1.0),
        width=0.5,
        length=2.0,
    )
    inset_ax.grid(False)
    for spine in inset_ax.spines.values():
        spine.set_linewidth(0.55)
        spine.set_color(PLOT_AXIS_COLOR)
    main_ax.indicate_inset_zoom(inset_ax, edgecolor=PLOT_MUTED_COLOR, alpha=0.55)
    return inset_ax


def render_small_multiples(state: ProjectState, fig: Any) -> tuple[Any, dict[str, Any]]:
    settings = state.settings
    layers = select_spectrum_layers(state.spectra, show_every_n=settings.show_every_n)
    if not layers:
        raise ValueError("Small-multiples view requires at least one visible spectrum")
    columns = min(settings.small_multiples_columns, len(layers))
    rows = int(math.ceil(len(layers) / columns))
    panel_grid = fig.subplots(rows, columns, squeeze=False, sharex=True)
    panels = [panel_grid.flat[index] for index in range(len(layers))]
    panel_headers: list[list[Any]] = []
    x_range, exact = _display_x_range(state)
    margins = safe_subplot_margins(
        settings,
        title=prepare_panel_title(settings.panel_title, settings),
    )
    panel_width_points = (
        float(settings.figure_width_in)
        * 72.0
        * max(margins["right"] - margins["left"], 0.1)
        / max(columns, 1)
    )
    for index, (ax, layer) in enumerate(zip(panels, layers)):
        x_values = convert_x(
            layer.x,
            layer.axis_kind,
            settings.x_axis,
            layer_energy_kev(layer, settings.energy_kev),
        )
        y_values = display_y_for_layer(layer, settings, 0)
        finite_points = [
            (x, y)
            for x, y in zip(x_values, y_values)
            if math.isfinite(float(x)) and math.isfinite(float(y))
        ]
        if finite_points:
            x_clean, y_clean = zip(*finite_points)
            ax.plot(x_clean, y_clean, color=layer.color, linewidth=layer.linewidth)
        wrapped_name = wrap_text(
            layer.name,
            settings,
            max_width_points=max(32.0, panel_width_points * 0.64),
            font_size=settings.tick_label_size,
        )
        reserve_axes_top(
            ax,
            min(0.14 + max(len(wrapped_name.splitlines()) - 1, 0) * 0.08, 0.38),
        )
        name_artist = ax.text(
            0.98,
            0.94,
            wrapped_name,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=settings.tick_label_size,
            color=PLOT_TEXT_COLOR,
        )
        header_artists = [name_artist]
        if settings.show_panel_labels:
            header_artists.append(
                ax.text(
                    0.02,
                    0.96,
                    chr(97 + index) if index < 26 else str(index + 1),
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=8.0,
                    fontweight="bold",
                    color=PLOT_TEXT_COLOR,
                )
            )
        panel_headers.append(header_artists)
        _apply_x_range(ax, x_range, exact=exact)
        _polish_panel(ax, settings)
        ax.tick_params(axis="y", left=True, labelleft=index % columns == 0)
        if index // columns == rows - 1:
            ax.set_xlabel(settings.x_label or default_axis_label(settings.x_axis))
        if index % columns == 0:
            ax.set_ylabel(settings.y_label)
    for index in range(len(layers), rows * columns):
        panel_grid.flat[index].set_visible(False)
    fig.subplots_adjust(
        **margins,
        hspace=0.12,
        wspace=0.12,
    )
    title_artist = None
    wrapped_title = prepare_panel_title(settings.panel_title, settings)
    if wrapped_title:
        title_artist = fig.suptitle(
            wrapped_title,
            x=margins["left"],
            y=0.99,
            ha="left",
            va="top",
            fontsize=max(settings.axis_label_size, settings.font_size),
            fontweight="bold",
            color=PLOT_TEXT_COLOR,
        )
    annotation_artists: list[Any] = []
    for ax in panels:
        annotation_artists.extend(draw_annotations(ax, state))
    axes = {
        "main": panels[0],
        "panels": panels,
        "panel_headers": panel_headers,
    }
    if title_artist is not None:
        axes["panel_title"] = title_artist
    if annotation_artists:
        axes["annotations"] = annotation_artists
    return fig, axes


def _polish_panel(ax: Any, settings: Any) -> None:
    for spine in ax.spines.values():
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
        labelsize=settings.tick_label_size,
    )


def _display_x_range(state: ProjectState) -> tuple[tuple[float, float] | None, bool]:
    values: list[float] = []
    for layer in state.spectra:
        if layer.visible:
            converted = convert_x(
                layer.x,
                layer.axis_kind,
                state.settings.x_axis,
                layer_energy_kev(layer, state.settings.energy_kev),
            )
            values.extend(value for value in converted if math.isfinite(value))
    auto = (min(values), max(values)) if values else None
    settings = state.settings
    if settings.x_min is None and settings.x_max is None:
        return auto, False
    if settings.x_min is not None and settings.x_max is not None:
        return (settings.x_min, settings.x_max), True
    if auto is None:
        return None, False
    return (
        settings.x_min if settings.x_min is not None else auto[0],
        settings.x_max if settings.x_max is not None else auto[1],
    ), True


def _apply_x_range(
    ax: Any, x_range: tuple[float, float] | None, *, exact: bool
) -> None:
    if x_range is None:
        return
    x_min, x_max = x_range
    if not math.isfinite(x_min) or not math.isfinite(x_max) or x_min >= x_max:
        return
    if not exact:
        pad = (
            abs(x_max - x_min) * 0.03 if x_max != x_min else max(abs(x_min) * 0.05, 0.5)
        )
        x_min, x_max = x_min - pad, x_max + pad
    ax.set_xlim(x_min, x_max)
