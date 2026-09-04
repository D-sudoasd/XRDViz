from __future__ import annotations

import io
import math
from datetime import datetime, timezone
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
from xrdviz.plot.derived_renderer import render_derived
from xrdviz.plot.finalize import (
    complete_rendered_project as _complete_rendered_project,
    configure_figure as _configure_figure,
)
from xrdviz.plot.layout import (
    add_direct_labels,
    direct_label_decoration_points,
    phase_label_offset_points,
    prepare_panel_title,
    prepare_side_labels,
    reserve_axes_top,
    safe_subplot_margins,
    set_panel_title,
    side_label_height_points,
)
from xrdviz.plot.map_renderer import render_map
from xrdviz.plot.spectrum_extras import (
    draw_annotations,
    draw_inset,
    layer_energy_kev,
    render_small_multiples,
)
from xrdviz.plot.style import apply_matplotlib_style
from xrdviz.transforms import display_uncertainty_for_layer, display_y_for_layer


def render_project(
    state: ProjectState,
    figure: Any | None = None,
    *,
    preview_size: tuple[int, int] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Render a project into a Matplotlib figure.

    ``preview_size`` is reserved for interactive canvases.  It keeps the
    figure raster close to the widget's logical pixel size so a live preview
    does not allocate the publication raster (for example, 600 dpi) and then
    get clipped by Qt.  The default path remains publication-facing and uses
    the exact physical size and DPI from ``PlotSettings``.
    """
    Figure = _figure_class()
    if figure is None:
        fig = Figure()
        return _render_project_inplace(
            state,
            fig,
            preview_size=preview_size,
            validate_layout=True,
        )

    # Render into an isolated figure first.  A malformed import or an
    # unavailable view must not clear the live canvas; only a complete render
    # is committed to the caller's Figure below.  Re-rendering into the live
    # Figure keeps every Axes, transform, and child artist owned by that Figure;
    # Matplotlib does not support safely transplanting a populated Axes.
    staged = Figure()
    _render_project_inplace(
        state,
        staged,
        preview_size=preview_size,
        validate_layout=True,
    )
    return _render_project_inplace(
        state,
        figure,
        preview_size=preview_size,
        validate_layout=False,
    )


def _render_project_inplace(
    state: ProjectState,
    fig: Any,
    *,
    preview_size: tuple[int, int] | None,
    validate_layout: bool,
) -> tuple[Any, dict[str, Any]]:
    settings = state.settings
    apply_matplotlib_style(settings)
    _validate_outside_legend_layout(state)

    _configure_figure(fig, settings, preview_size=preview_size)
    fig.clear()

    if settings.view_mode == "map":
        rendered_fig, axes = render_map(state, fig)
        return _complete_rendered_project(
            state, rendered_fig, axes, validate_layout=validate_layout
        )
    if settings.view_mode == "derived":
        rendered_fig, axes = render_derived(state, fig)
        return _complete_rendered_project(
            state, rendered_fig, axes, validate_layout=validate_layout
        )
    if settings.view_mode == "small_multiples":
        rendered_fig, axes = render_small_multiples(state, fig)
        return _complete_rendered_project(
            state, rendered_fig, axes, validate_layout=validate_layout
        )
    if settings.view_mode == "refinement":
        rendered_fig, axes = _render_refinement(state, fig)
        return _complete_rendered_project(
            state, rendered_fig, axes, validate_layout=validate_layout
        )

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
        spectrum_handles, colorbar_ax, uncertainty_artists = _draw_spectra(main_ax, state, fig)
        if uncertainty_artists:
            axes["uncertainty"] = uncertainty_artists
        if colorbar_ax is not None:
            axes["colorbar"] = colorbar_ax
        x_range, exact_x_range = _display_x_range(state)
        _apply_x_range(main_ax, x_range, exact=exact_x_range)
        (
            phase_handles,
            phase_row_labels,
            phase_peak_labels,
            bragg_band_guard,
        ) = _draw_bragg_band(main_ax, state)
        if phase_row_labels:
            axes["phase_row_labels"] = phase_row_labels
        if phase_peak_labels:
            axes["phase_peak_labels"] = phase_peak_labels
        if bragg_band_guard is not None:
            axes["bragg_band_guard"] = bragg_band_guard
        _apply_x_range(main_ax, x_range, exact=exact_x_range)

    x_range, exact_x_range = _display_x_range(state)
    _apply_x_range(main_ax, x_range, exact=exact_x_range)

    if settings.inset_enabled and settings.inset_x_min is not None and settings.inset_x_max is not None:
        axes["inset"] = draw_inset(main_ax, state)

    main_ax.set_xlabel(settings.x_label or default_axis_label(settings.x_axis))
    metadata_layers = select_spectrum_layers(state.spectra, show_every_n=settings.show_every_n)
    main_ax.set_ylabel(
        _metadata_label(settings.sort_by, metadata_layers)
        if settings.view_mode == "heatmap"
        else settings.y_label
    )
    title_artist = set_panel_title(main_ax, settings.panel_title, settings)
    if title_artist is not None:
        axes["panel_title"] = title_artist
    _polish_main_axis(main_ax, state, spectrum_handles, phase_handles)
    right_labels: list[str] = []
    if settings.legend_location == "outside right" and main_ax.get_legend() is not None:
        right_labels.extend(text.get_text() for text in main_ax.get_legend().get_texts())
    prepared_direct_labels: list[str] = []
    if settings.direct_labels:
        prepared_direct_labels = prepare_side_labels(
            [handle.get_label() for handle in spectrum_handles], settings
        )
        right_labels.extend(prepared_direct_labels)
    left_labels = [
        text.get_text() for text in axes.get("phase_row_labels", [])
    ]
    fig.subplots_adjust(
        **safe_subplot_margins(
            settings,
            title=prepare_panel_title(settings.panel_title, settings),
            left_labels=left_labels,
            left_decoration_points=phase_label_offset_points(settings),
            right_labels=right_labels,
            right_decoration_points=max(
                32.0 if settings.legend_location == "outside right" else 8.0,
                direct_label_decoration_points(settings)
                if prepared_direct_labels
                else 0.0,
            ),
            colorbar=colorbar_ax is not None,
        ),
    )
    annotation_artists = draw_annotations(main_ax, state)
    if annotation_artists:
        axes["annotations"] = annotation_artists
    if prepared_direct_labels:
        direct_labels, direct_label_leaders = add_direct_labels(
            main_ax,
            spectrum_handles,
            prepared_direct_labels,
            settings,
        )
        axes["direct_labels"] = direct_labels
        axes["direct_label_leaders"] = direct_label_leaders
    return _complete_rendered_project(
        state, fig, axes, validate_layout=validate_layout
    )


def _render_refinement(state: ProjectState, fig: Any) -> tuple[Any, dict[str, Any]]:
    fit = state.fit
    if fit is None:
        raise ValueError("Refinement view requires an imported observed/calculated fit result")

    settings = state.settings
    grid = fig.add_gridspec(2, 1, height_ratios=(3.2, 1.0), hspace=0.05)
    main_ax = fig.add_subplot(grid[0])
    residual_ax = fig.add_subplot(grid[1], sharex=main_ax)
    main_ax.set_facecolor("white")
    residual_ax.set_facecolor("white")
    axes: dict[str, Any] = {"main": main_ax, "bragg": main_ax, "residual": residual_ax}

    x_values = convert_x(
        fit.x,
        fit.axis_kind,
        settings.x_axis,
        layer_energy_kev(fit, settings.energy_kev),
    )
    point_indices = [
        index
        for index, values in enumerate(zip(x_values, fit.observed, fit.calculated))
        if all(math.isfinite(float(value)) for value in values)
    ]
    if not point_indices:
        raise ValueError("Refinement result does not contain finite observed/calculated points")

    x_clean = [x_values[index] for index in point_indices]
    observed_raw = [fit.observed[index] for index in point_indices]
    calculated_raw = [fit.calculated[index] for index in point_indices]
    scale = max((value for value in observed_raw if value > 0), default=1.0) if settings.normalize else 1.0

    def display_main(values: list[float]) -> list[float]:
        displayed = [float(value) / scale for value in values]
        if settings.log_scale:
            displayed = [math.log10(max(value, settings.log_epsilon)) for value in displayed]
        return displayed

    observed = display_main(observed_raw)
    calculated = display_main(calculated_raw)
    observed_marker_size = max(2.0, settings.line_width * 2.8)
    observed_handle = main_ax.plot(
        x_clean,
        observed,
        linestyle="None",
        marker="o",
        markersize=observed_marker_size,
        markerfacecolor="none",
        markeredgecolor=PLOT_TEXT_COLOR,
        markeredgewidth=0.55,
        label="Observed",
        zorder=3,
    )[0]
    calculated_handle = main_ax.plot(
        x_clean,
        calculated,
        color="#D62F53",
        linewidth=max(settings.line_width, 0.75),
        label="Calculated",
        zorder=4,
    )[0]
    handles = [observed_handle, calculated_handle]

    if settings.show_fit_background and fit.background:
        background = display_main([fit.background[index] for index in point_indices])
        handles.append(
            main_ax.plot(
                x_clean,
                background,
                color=PLOT_MUTED_COLOR,
                linewidth=max(0.5, settings.line_width * 0.8),
                linestyle="--",
                label="Background",
                zorder=2,
            )[0]
        )

    if settings.show_fit_components:
        for component_index, component in enumerate(fit.components):
            component_values = display_main([component.y[index] for index in point_indices])
            handles.append(
                main_ax.plot(
                    x_clean,
                    component_values,
                    color=component.color or _component_color(component_index),
                    linewidth=max(0.5, settings.line_width * 0.8),
                    linestyle="--",
                    alpha=0.9,
                    label=component.name,
                    zorder=2,
                )[0]
            )

    uncertainty_artists: list[Any] = []
    if fit.sigma and settings.uncertainty_mode != "none":
        sigma = [fit.sigma[index] / scale for index in point_indices]
        if settings.log_scale:
            lower = [
                math.log10(max((value - error) / scale, settings.log_epsilon))
                for value, error in zip(observed_raw, [fit.sigma[index] for index in point_indices])
            ]
            upper = [
                math.log10(max((value + error) / scale, settings.log_epsilon))
                for value, error in zip(observed_raw, [fit.sigma[index] for index in point_indices])
            ]
        else:
            lower = [value - error for value, error in zip(observed, sigma)]
            upper = [value + error for value, error in zip(observed, sigma)]
        if settings.uncertainty_mode == "band":
            uncertainty_artists.append(
                main_ax.fill_between(
                    x_clean,
                    lower,
                    upper,
                    color=PLOT_MUTED_COLOR,
                    alpha=settings.uncertainty_alpha,
                    linewidth=0.0,
                    label="_nolegend_",
                    zorder=1,
                )
            )
        else:
            stride = max(1, int(settings.errorbar_stride))
            sampled_indices = range(0, len(x_clean), stride)
            bar_x = [x_clean[index] for index in sampled_indices]
            bar_y = [observed[index] for index in range(0, len(observed), stride)]
            bar_lower = [lower[index] for index in range(0, len(lower), stride)]
            bar_upper = [upper[index] for index in range(0, len(upper), stride)]
            uncertainty_artists.append(
                main_ax.errorbar(
                    bar_x,
                    bar_y,
                    yerr=[
                        [value - bound for value, bound in zip(bar_y, bar_lower)],
                        [bound - value for value, bound in zip(bar_y, bar_upper)],
                    ],
                    fmt="none",
                    ecolor=PLOT_MUTED_COLOR,
                    elinewidth=0.5,
                    capsize=1.5,
                    label="_nolegend_",
                    zorder=1,
                )
            )
    if uncertainty_artists:
        axes["uncertainty"] = uncertainty_artists

    difference = [(fit.observed[index] - fit.calculated[index]) / scale for index in point_indices]
    residual_ax.axhline(0.0, color=PLOT_MUTED_COLOR, linewidth=0.5, zorder=1)
    residual_ax.plot(x_clean, difference, color="#286FB7", linewidth=max(settings.line_width, 0.65), label="Difference")
    residual_ax.set_ylabel("Obs. − calc.")
    residual_ax.set_xlabel(settings.x_label or default_axis_label(settings.x_axis))
    main_ax.set_ylabel(settings.y_label)
    main_ax.tick_params(axis="x", labelbottom=False)

    x_range, exact_x_range = _fit_x_range(x_clean, settings)
    _apply_x_range(main_ax, x_range, exact=exact_x_range)

    _apply_x_range(residual_ax, x_range, exact=exact_x_range)
    (
        phase_handles,
        phase_row_labels,
        phase_peak_labels,
        bragg_band_guard,
    ) = _draw_bragg_band(main_ax, state)
    if phase_row_labels:
        axes["phase_row_labels"] = phase_row_labels
    if phase_peak_labels:
        axes["phase_peak_labels"] = phase_peak_labels
    if bragg_band_guard is not None:
        axes["bragg_band_guard"] = bragg_band_guard
    _apply_x_range(main_ax, x_range, exact=exact_x_range)
    _polish_main_axis(main_ax, state, handles, phase_handles)
    _polish_residual_axis(residual_ax, settings)

    if settings.show_fit_metrics:
        metric_lines = []
        if fit.rp is not None and math.isfinite(fit.rp):
            metric_lines.append(f"$R_p$ = {fit.rp:.3g}%")
        if fit.rwp is not None and math.isfinite(fit.rwp):
            metric_lines.append(f"$R_{{wp}}$ = {fit.rwp:.3g}%")
        if metric_lines:
            reserve_axes_top(
                main_ax,
                min(0.12 + max(len(metric_lines) - 1, 0) * 0.06, 0.3),
            )
            axes["metrics"] = main_ax.text(
                0.03,
                0.96,
                "\n".join(metric_lines),
                transform=main_ax.transAxes,
                ha="left",
                va="top",
                fontsize=settings.tick_label_size,
                color=PLOT_TEXT_COLOR,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.5},
            )

    title_artist = set_panel_title(main_ax, settings.panel_title, settings)
    if title_artist is not None:
        axes["panel_title"] = title_artist
    right_labels: list[str] = []
    if settings.legend_location == "outside right" and main_ax.get_legend() is not None:
        right_labels.extend(text.get_text() for text in main_ax.get_legend().get_texts())
    prepared_direct_labels: list[str] = []
    if settings.direct_labels:
        prepared_direct_labels = prepare_side_labels(
            [handle.get_label() for handle in handles], settings
        )
        right_labels.extend(prepared_direct_labels)
    fig.subplots_adjust(
        **safe_subplot_margins(
            settings,
            title=prepare_panel_title(settings.panel_title, settings),
            left_labels=[text.get_text() for text in phase_row_labels],
            left_decoration_points=phase_label_offset_points(settings),
            right_labels=right_labels,
            right_decoration_points=max(
                32.0 if settings.legend_location == "outside right" else 8.0,
                direct_label_decoration_points(settings)
                if prepared_direct_labels
                else 0.0,
            ),
        ),
        hspace=0.05,
    )
    observed_marker_indices = _refinement_marker_indices(
        x_clean,
        main_ax,
        marker_size_points=observed_marker_size,
    )
    observed_handle.set_data(
        [x_clean[index] for index in observed_marker_indices],
        [observed[index] for index in observed_marker_indices],
    )
    axes["observed_marker_indices"] = observed_marker_indices
    annotation_artists = draw_annotations(main_ax, state)
    if annotation_artists:
        axes["annotations"] = annotation_artists
    if prepared_direct_labels:
        direct_labels, direct_label_leaders = add_direct_labels(
            main_ax,
            handles,
            prepared_direct_labels,
            settings,
        )
        axes["direct_labels"] = direct_labels
        axes["direct_label_leaders"] = direct_label_leaders
    return fig, axes


def _component_color(index: int) -> str:
    colors = ("#2B9C8F", "#7A5CC7", "#E2A23A", "#45A7E6", "#B05A7A")
    return colors[index % len(colors)]


def _fit_x_range(x_values: list[float], settings: Any) -> tuple[tuple[float, float], bool]:
    if settings.x_min is not None and settings.x_max is not None:
        return (float(settings.x_min), float(settings.x_max)), True
    minimum = float(settings.x_min) if settings.x_min is not None else min(x_values)
    maximum = float(settings.x_max) if settings.x_max is not None else max(x_values)
    return (minimum, maximum), settings.x_min is not None or settings.x_max is not None


def _refinement_marker_indices(
    x_values: list[float],
    ax: Any,
    *,
    marker_size_points: float,
) -> list[int]:
    """Thin display-only fit symbols to their final physical pixel density.

    Calculated and residual curves, metrics, and exported source data continue
    to use every point.  Only overlapping observed markers are reduced.
    """

    point_count = len(x_values)
    if point_count < 1:
        return []
    if point_count == 1:
        return [0]
    # Leave a visible white gap between adjacent open circles at final size.
    # A tighter pitch still produces a dark bead-like baseline even though the
    # markers no longer overlap mathematically.
    marker_pitch_points = max(float(marker_size_points) * 1.8, 4.0)
    marker_pitch_pixels = (
        marker_pitch_points * max(float(ax.figure.dpi), 1.0) / 72.0
    )
    display_x = ax.transData.transform(
        np.column_stack((np.asarray(x_values, dtype=float), np.zeros(point_count)))
    )[:, 0]
    axes_box = ax.get_window_extent()
    visible_indices = [
        index
        for index, x_pixel in enumerate(display_x)
        if axes_box.x0 - 1.0e-9 <= x_pixel <= axes_box.x1 + 1.0e-9
    ]
    if not visible_indices:
        return [0, point_count - 1]

    sampled = [visible_indices[0]]
    for index in visible_indices[1:-1]:
        if (
            abs(float(display_x[index] - display_x[sampled[-1]]))
            >= marker_pitch_pixels
        ):
            sampled.append(index)

    last_visible = visible_indices[-1]
    if sampled[-1] != last_visible:
        if (
            len(sampled) > 1
            and abs(float(display_x[last_visible] - display_x[sampled[-1]]))
            < marker_pitch_pixels
        ):
            sampled[-1] = last_visible
        else:
            sampled.append(last_visible)

    return sorted({0, point_count - 1, *sampled})


def _polish_residual_axis(ax: Any, settings: Any) -> None:
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
        labelsize=settings.tick_label_size,
    )
    ax.yaxis.label.set_size(settings.axis_label_size)
    ax.xaxis.label.set_size(settings.axis_label_size)


def export_project(state: ProjectState, path: str | Path) -> None:
    output = Path(path)
    import matplotlib as mpl

    # Vector backends add the current time and, for SVG, hash-derived marker
    # identifiers by default.  Keep those values stable for publication
    # artifacts while confining the rcParam override to this export call.
    export_context = {"svg.hashsalt": "xrdviz-publication"}
    save_kwargs: dict[str, Any] = {}
    suffix = output.suffix.lower()
    if suffix == ".pdf":
        export_date = datetime(2000, 1, 1, tzinfo=timezone.utc)
        save_kwargs["metadata"] = {"CreationDate": export_date, "ModDate": export_date}
    elif suffix == ".svg":
        save_kwargs["metadata"] = {"Date": "2000-01-01T00:00:00+00:00"}

    with mpl.rc_context(export_context):
        fig, _axes = render_project(state)
        # Keep the requested physical canvas.  ``bbox_inches="tight"`` crops
        # the page and makes the exported dimensions depend on drawn artists.
        fig.savefig(
            output,
            dpi=state.settings.dpi,
            facecolor="white",
            edgecolor="white",
            **save_kwargs,
        )
    if output.suffix.lower() in {".png", ".tif", ".tiff"}:
        _ensure_opaque_raster(output, state.settings.dpi)


def _draw_spectra(ax: Any, state: ProjectState, fig: Any) -> tuple[list[Any], Any | None, list[Any]]:
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
    uncertainty_artists: list[Any] = []
    for index, layer in enumerate(visible_layers):
        x_values = convert_x(
            layer.x,
            layer.axis_kind,
            settings.x_axis,
            layer_energy_kev(layer, settings.energy_kev),
        )
        y_values = display_y_for_layer(layer, settings, index)
        finite_indices = [
            point_index
            for point_index, (x_value, y_value) in enumerate(zip(x_values, y_values))
            if math.isfinite(x_value) and math.isfinite(y_value)
        ]
        if not finite_indices:
            continue
        x_clean = [x_values[point_index] for point_index in finite_indices]
        y_clean = [y_values[point_index] for point_index in finite_indices]
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
        if layer.y_error and settings.uncertainty_mode != "none":
            lower, upper = display_uncertainty_for_layer(layer, settings, index)
            uncertainty_indices = [
                point_index
                for point_index in finite_indices
                if point_index < len(lower)
                and math.isfinite(lower[point_index])
                and math.isfinite(upper[point_index])
            ]
            if uncertainty_indices and settings.uncertainty_mode == "band":
                uncertainty_x = [x_values[point_index] for point_index in uncertainty_indices]
                uncertainty_lower = [lower[point_index] for point_index in uncertainty_indices]
                uncertainty_upper = [upper[point_index] for point_index in uncertainty_indices]
                uncertainty_artists.append(
                    ax.fill_between(
                        uncertainty_x,
                        uncertainty_lower,
                        uncertainty_upper,
                        color=color,
                        alpha=settings.uncertainty_alpha,
                        linewidth=0.0,
                        label="_nolegend_",
                    )
                )
            elif uncertainty_indices and settings.uncertainty_mode == "bars":
                stride = max(1, int(settings.errorbar_stride))
                sampled_indices = uncertainty_indices[::stride]
                uncertainty_x = [x_values[point_index] for point_index in sampled_indices]
                uncertainty_y = [y_values[point_index] for point_index in sampled_indices]
                uncertainty_lower = [lower[point_index] for point_index in sampled_indices]
                uncertainty_upper = [upper[point_index] for point_index in sampled_indices]
                lower_error = [value - bound for value, bound in zip(uncertainty_y, uncertainty_lower)]
                upper_error = [bound - value for value, bound in zip(uncertainty_y, uncertainty_upper)]
                uncertainty_artists.append(
                    ax.errorbar(
                        uncertainty_x,
                        uncertainty_y,
                        yerr=[lower_error, upper_error],
                        fmt="none",
                        ecolor=color,
                        elinewidth=max(0.4, layer.linewidth * 0.7),
                        capsize=1.5,
                        alpha=max(settings.uncertainty_alpha, 0.45),
                        label="_nolegend_",
                    )
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
    return handles, colorbar_ax, uncertainty_artists


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


def _draw_bragg_band(
    ax: Any, state: ProjectState
) -> tuple[list[Any], list[Any], list[Any], Any | None]:
    visible_phases = [phase for phase in state.phases if phase.visible and phase.peaks]
    if not visible_phases:
        return [], [], [], None

    phase_handles = []
    row_labels = []
    peak_labels = []
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
    from matplotlib.transforms import ScaledTranslation

    row_label_transform = ax.get_yaxis_transform() + ScaledTranslation(
        -phase_label_offset_points(state.settings) / 72.0,
        0.0,
        ax.figure.dpi_scale_trans,
    )

    for index, phase in enumerate(visible_phases):
        baseline = band_bottom + row_height * index + row_height * 0.18
        tick_top = baseline + row_height * 0.62 * max(min(phase.tick_height, 1.0), 0.15)
        characteristic = _characteristic_peaks(phase.peaks)
        row_labels.append(
            ax.text(
                0.0,
                baseline + row_height * 0.31,
                phase.phase or phase.name,
                transform=row_label_transform,
                ha="right",
                va="center",
                fontsize=state.settings.tick_label_size,
                color=PLOT_TEXT_COLOR,
                fontweight="normal",
                clip_on=False,
            )
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
                peak_labels.append(
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
                )
            if phase.show_guides and peak in characteristic:
                ax.axvline(x_value, color=phase.color, linestyle="--", linewidth=0.55, alpha=0.36, zorder=1)
        if state.settings.show_phase_legend:
            phase_handles.append(_phase_legend_handle(phase.color, phase.phase or phase.name, phase.marker_shape))

    ax.axhline(data_bottom, color=PLOT_MUTED_COLOR, linewidth=0.45, alpha=0.55, zorder=1)
    band_floor = band_bottom - row_height * 0.08
    ax.set_ylim(band_floor, data_top + data_span * 0.04)
    if state.settings.show_y_tick_labels:
        # The Bragg lane is categorical, not part of the intensity scale.
        # Removing intensity ticks below the data baseline prevents semantic
        # ambiguity and lets phase names sit compactly beside their rows.
        data_ticks = [
            float(value)
            for value in ax.get_yticks()
            if data_bottom - 1.0e-12 <= float(value) <= ax.get_ylim()[1] + 1.0e-12
        ]
        if data_ticks:
            ax.set_yticks(data_ticks)

    # Invisible geometry guard used by the final layout validator.  It spans
    # the complete Bragg phase lane so legends/insets cannot silently obscure
    # ticks whose individual extents would otherwise be difficult to measure.
    from matplotlib.patches import Rectangle

    bragg_band_guard = Rectangle(
        (0.0, band_floor),
        1.0,
        data_bottom - band_floor,
        transform=ax.get_yaxis_transform(),
        facecolor="none",
        edgecolor="none",
        linewidth=0.0,
    )
    ax.add_patch(bragg_band_guard)
    return phase_handles, row_labels, peak_labels, bragg_band_guard


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
    show_y_ticks = settings.show_y_tick_labels or settings.view_mode in {
        "heatmap",
        "map",
        "derived",
        "small_multiples",
    }
    ax.tick_params(
        axis="y",
        left=show_y_ticks,
        right=False,
        labelleft=show_y_ticks,
    )
    handles = []
    if settings.show_legend and not settings.direct_labels:
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
            wrapped_labels = prepare_side_labels(
                [handle.get_label() for handle in handles], settings
            )
            legend_kwargs.update(
                {
                    "labels": wrapped_labels,
                    "loc": "center left",
                    "bbox_to_anchor": (1.02, 0.5),
                }
            )
        else:
            legend_kwargs["loc"] = (
                "lower left"
                if settings.inset_enabled and settings.legend_location == "best"
                else settings.legend_location
            )
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


def _validate_outside_legend_layout(state: ProjectState) -> None:
    settings = state.settings
    if (
        settings.direct_labels
        and settings.show_colorbar
        and settings.view_mode == "gradient_stack"
    ):
        raise ValueError(
            "Direct curve labels cannot share the publication gutter with a colorbar; "
            "disable direct labels or the colorbar."
        )
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
    if settings.direct_labels:
        raise ValueError(
            "Outside-right legend cannot share the publication gutter with direct "
            "curve labels; choose an inside phase legend or disable direct labels."
        )
    if settings.show_colorbar and settings.view_mode == "gradient_stack":
        raise ValueError(
            "Outside-right legend cannot share the publication gutter with a colorbar; "
            "choose an inside legend position or disable the spectrum legend."
        )

    wrapped_labels = prepare_side_labels(labels, settings)
    legend_height_pt = side_label_height_points(wrapped_labels, settings)
    center = (float(settings.margin_bottom) + float(settings.margin_top)) / 2.0
    available_fraction = max(2.0 * min(center, 1.0 - center) - 0.02, 0.0)
    available_height_pt = float(settings.figure_height_in) * 72.0 * available_fraction
    if legend_height_pt > available_height_pt:
        raise ValueError(
            "Outside-right legend does not fit vertically at the selected figure size; "
            "use a double-column/taller canvas, shorten or sample labels, or choose an inside legend position."
        )
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
    settings = state.settings
    phase_count = sum(1 for phase in state.phases if phase.visible and phase.peaks)
    configured = min(max(settings.bragg_band_height, 0.05), 0.45)
    if phase_count == 0:
        return configured

    margins = safe_subplot_margins(
        settings,
        title=prepare_panel_title(settings.panel_title, settings),
    )
    available_height_points = (
        float(settings.figure_height_in)
        * 72.0
        * max(float(margins["top"]) - float(margins["bottom"]), 0.32)
    )
    row_height_points = max(float(settings.tick_label_size) * 1.5, 8.0)
    y_tick_clearance = 0.0
    if settings.show_y_tick_labels:
        y_tick_clearance = float(settings.tick_label_size) * 1.2
    required_band_share = (
        phase_count * row_height_points + y_tick_clearance
    ) / max(
        available_height_points, 1.0
    )
    if required_band_share >= 1.0:
        required = math.inf
    else:
        required = required_band_share * 1.08 / max(
            1.0 - required_band_share, 1.0e-9
        )
    if required > 0.45:
        raise ValueError(
            "Phase rows do not fit without overlap at the selected figure height; "
            "show fewer phases or use a taller canvas."
        )
    if settings.show_y_tick_labels and phase_count == 1:
        configured = max(configured, 0.20)
    return max(configured, required)


def _visible_spectrum_x_range(state: ProjectState) -> tuple[float, float] | None:
    x_values: list[float] = []
    for layer in state.spectra:
        if not layer.visible:
            continue
        converted = convert_x(
            layer.x,
            layer.axis_kind,
            state.settings.x_axis,
            layer_energy_kev(layer, state.settings.energy_kev),
        )
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
