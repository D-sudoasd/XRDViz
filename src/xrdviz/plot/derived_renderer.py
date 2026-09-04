from __future__ import annotations

import math
from typing import Any

from matplotlib.ticker import MaxNLocator

from xrdviz.models import PLOT_AXIS_COLOR, PLOT_TEXT_COLOR, ProjectState
from xrdviz.plot.layout import (
    prepare_metric_lines,
    prepare_panel_title,
    reserve_axes_top,
    safe_subplot_margins,
    set_panel_title,
)
from xrdviz.plot.spectrum_extras import draw_annotations


def render_derived(state: ProjectState, fig: Any) -> tuple[Any, dict[str, Any]]:
    data = state.derived_plot
    if data is None:
        raise ValueError(
            "Derived view requires Scherrer, Williamson-Hall, or rocking-curve data"
        )
    settings = state.settings
    ax = fig.add_subplot(111)
    ax.set_facecolor("white")
    axes: dict[str, Any] = {"main": ax}

    scatter_points = list(data.scatter) if data.scatter else list(zip(data.x, data.y))
    scatter_x = [point[0] for point in scatter_points]
    scatter_y = [point[1] for point in scatter_points]
    if data.kind == "rocking_curve":
        axes["curve"] = ax.plot(
            data.x,
            data.y,
            color="#286FB7",
            linewidth=max(settings.line_width, 0.75),
            marker="o",
            markersize=2.8,
            label="Rocking curve",
        )[0]
        scatter = ax.scatter(scatter_x, scatter_y, s=10.0, color="#286FB7", zorder=3)
    else:
        scatter = ax.scatter(
            scatter_x,
            scatter_y,
            s=18.0,
            facecolors="white",
            edgecolors="#286FB7",
            linewidths=0.75,
            zorder=3,
        )
    axes["scatter"] = scatter
    if data.fit_line:
        axes["fit_line"] = ax.plot(
            [point[0] for point in data.fit_line],
            [point[1] for point in data.fit_line],
            color="#D62F53",
            linewidth=max(settings.line_width, 0.75),
            label="Linear fit",
            zorder=2,
        )[0]
    ax.set_xlabel(str(data.labels.get("x", "x")))
    ax.set_ylabel(str(data.labels.get("y", "y")))
    if settings.x_min is not None or settings.x_max is not None:
        current = ax.get_xlim()
        ax.set_xlim(
            settings.x_min if settings.x_min is not None else current[0],
            settings.x_max if settings.x_max is not None else current[1],
        )
    metric_artist = None
    if data.metrics:
        metric_text = prepare_metric_lines(
            _metric_lines(data.metrics),
            settings,
        )
        reserve_axes_top(
            ax,
            min(0.12 + max(len(metric_text.splitlines()) - 1, 0) * 0.055, 0.45),
        )
        metric_artist = ax.text(
            0.03,
            0.96,
            metric_text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=settings.tick_label_size,
            color=PLOT_TEXT_COLOR,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.5},
            zorder=5,
        )
        axes["metrics"] = metric_artist
    title_artist = set_panel_title(ax, settings.panel_title, settings)
    if title_artist is not None:
        axes["panel_title"] = title_artist
    _polish_axis(ax, settings)
    fig.subplots_adjust(
        **safe_subplot_margins(
            settings,
            title=prepare_panel_title(settings.panel_title, settings),
        ),
    )
    annotation_artists = draw_annotations(ax, state)
    if annotation_artists:
        axes["annotations"] = annotation_artists
    return fig, axes


def _format_metric(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return f"{float(value):.5g}"
    return str(value)


def _metric_lines(metrics: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for key, value in metrics.items():
        if key == "size_unit" and "crystallite_size" in metrics:
            continue
        if key == "wavelength_unit" and "wavelength" in metrics:
            continue
        formatted = _format_metric(value)
        if key == "crystallite_size" and metrics.get("size_unit"):
            formatted = f"{formatted} {metrics['size_unit']}"
        elif key == "wavelength" and metrics.get("wavelength_unit"):
            formatted = f"{formatted} {metrics['wavelength_unit']}"
        lines.append(f"{_metric_label(key)}: {formatted}")
    return lines


def _metric_label(key: Any) -> str:
    """Render persisted metric keys as readable prose without changing them."""

    labels = {
        "fwhm": "FWHM",
        "k": r"$K$",
        "microstrain": r"$\varepsilon$",
        "n_points": r"$n$",
        "r_squared": r"$R^2$",
        "wavelength": r"$\lambda$",
    }
    normalized = str(key)
    return labels.get(normalized, normalized.replace("_", " ").capitalize())


def _polish_axis(ax: Any, settings: Any) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(min(max(settings.line_width, 0.6), 0.8))
        spine.set_color(PLOT_AXIS_COLOR)
    ax.grid(False)
    # Keep decimal-heavy derived coordinates to a small, legible set on the
    # default 89 mm publication canvas.  The locator does not alter data or
    # units; it only controls major tick density.
    ax.xaxis.set_major_locator(_compact_locator())
    ax.yaxis.set_major_locator(_compact_locator())
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


def _compact_locator() -> MaxNLocator:
    return MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10], min_n_ticks=3)
