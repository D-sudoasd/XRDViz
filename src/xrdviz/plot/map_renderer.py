from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.ticker import MaxNLocator

from xrdviz.batch import matplotlib_colormap
from xrdviz.models import PLOT_AXIS_COLOR, PLOT_TEXT_COLOR, ProjectState
from xrdviz.plot.layout import (
    prepare_panel_title,
    safe_subplot_margins,
    set_panel_title,
)


def render_map(state: ProjectState, fig: Any) -> tuple[Any, dict[str, Any]]:
    data = state.map_data
    if data is None:
        raise ValueError(
            "Map view requires imported detector, cake, RSM, or pole-figure data"
        )
    settings = state.settings
    polar = data.kind == "pole_figure"
    ax = fig.add_subplot(111, projection="polar" if polar else None)
    ax.set_facecolor("white")

    displayed = np.asarray(data.intensity, dtype=float).copy()
    finite = np.isfinite(displayed)
    if data.counts is not None:
        finite &= np.asarray(data.counts, dtype=float) > 0.0
    if not np.any(finite):
        raise ValueError("Map data does not contain finite populated intensity values")
    if settings.normalize:
        positive = displayed[finite & (displayed > 0.0)]
        scale = float(np.max(positive)) if positive.size else 1.0
        displayed[finite] /= scale
    if settings.log_scale:
        displayed[finite] = np.log10(
            np.maximum(displayed[finite], settings.log_epsilon)
        )
    displayed = np.ma.masked_where(~finite, displayed)

    if polar:
        mesh = ax.pcolormesh(
            np.radians(np.asarray(data.x, dtype=float)),
            np.asarray(data.y, dtype=float),
            displayed,
            shading="auto",
            cmap=matplotlib_colormap(settings.colormap),
            rasterized=True,
        )
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_xlabel("")
        ax.set_ylabel(_axis_label(data.y_label, data.y_unit))
    else:
        mesh = ax.pcolormesh(
            np.asarray(data.x, dtype=float),
            np.asarray(data.y, dtype=float),
            displayed,
            shading="auto",
            cmap=matplotlib_colormap(settings.colormap),
            rasterized=True,
        )
        ax.set_xlabel(_axis_label(data.x_label, data.x_unit))
        ax.set_ylabel(_axis_label(data.y_label, data.y_unit))
        if data.kind == "detector":
            ax.invert_yaxis()
            ax.set_aspect("equal")

    colorbar_ax = None
    if settings.show_colorbar:
        colorbar = fig.colorbar(
            mesh, ax=ax, pad=0.06 if polar else 0.02, fraction=0.055 if polar else 0.045
        )
        colorbar_ax = colorbar.ax
        intensity_label = _axis_label(data.intensity_label, data.intensity_unit)
        if settings.normalize:
            intensity_label = f"Normalized {intensity_label}"
        if settings.log_scale:
            intensity_label = f"log10({intensity_label})"
        colorbar_ax.set_ylabel(intensity_label, color=PLOT_TEXT_COLOR)
        colorbar_ax.tick_params(colors=PLOT_TEXT_COLOR, width=0.55, length=2.5)

    title_artist = set_panel_title(ax, settings.panel_title, settings)
    _polish_axis(ax, settings, polar=polar)
    fig.subplots_adjust(
        **safe_subplot_margins(
            settings,
            title=prepare_panel_title(settings.panel_title, settings),
            colorbar=colorbar_ax is not None,
        ),
    )
    axes: dict[str, Any] = {"main": ax, "map": mesh}
    if colorbar_ax is not None:
        axes["colorbar"] = colorbar_ax
    if title_artist is not None:
        axes["panel_title"] = title_artist
    return fig, axes


def _axis_label(label: str, unit: str) -> str:
    text = str(label or "").strip()
    unit_text = str(unit or "").strip()
    return f"{text} ({unit_text})" if unit_text else text


def _polish_axis(ax: Any, settings: Any, *, polar: bool) -> None:
    ax.tick_params(colors=PLOT_TEXT_COLOR, labelsize=settings.tick_label_size)
    ax.xaxis.label.set_size(settings.axis_label_size)
    ax.yaxis.label.set_size(settings.axis_label_size)
    # A narrow publication canvas cannot reliably accommodate the default
    # AutoLocator output for q/d/angle coordinates.  Keep the map readable
    # while retaining the full data range and explicit axis units.
    if not polar:
        ax.xaxis.set_major_locator(_compact_locator())
    ax.yaxis.set_major_locator(_compact_locator())
    ax.grid(False)
    if not polar:
        for spine in ax.spines.values():
            spine.set_linewidth(min(max(settings.line_width, 0.6), 0.8))
            spine.set_color(PLOT_AXIS_COLOR)


def _compact_locator() -> MaxNLocator:
    """Return a stable major locator for narrow quantitative map panels."""

    return MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10], min_n_ticks=3)
