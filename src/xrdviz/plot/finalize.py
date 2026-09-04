from __future__ import annotations

import math
from typing import Any

import numpy as np

from xrdviz.models import ProjectState
from xrdviz.plot.layout import validate_figure_layout


def configure_figure(
    fig: Any,
    settings: Any,
    *,
    preview_size: tuple[int, int] | None,
) -> None:
    if preview_size is None:
        fig.set_size_inches(
            settings.figure_width_in,
            settings.figure_height_in,
            forward=True,
        )
        fig.set_dpi(settings.dpi)
        return
    preview_width, preview_height = (int(value) for value in preview_size)
    if preview_width < 1 or preview_height < 1:
        raise ValueError("preview_size must contain positive pixel dimensions")
    aspect = float(settings.figure_width_in) / float(settings.figure_height_in)
    if not math.isfinite(aspect) or aspect <= 0.0:
        raise ValueError("Figure dimensions must be finite and positive")
    if preview_width / preview_height > aspect:
        fitted_width = max(1, int(round(preview_height * aspect)))
        fitted_height = preview_height
    else:
        fitted_width = preview_width
        fitted_height = max(1, int(round(preview_width / aspect)))
    physical_width = float(settings.figure_width_in)
    physical_height = float(settings.figure_height_in)
    preview_dpi = min(
        fitted_width / physical_width,
        fitted_height / physical_height,
    )
    if not math.isfinite(preview_dpi) or preview_dpi <= 0.0:
        raise ValueError("Preview DPI must be finite and positive")
    fig.set_dpi(preview_dpi)
    fig.set_size_inches(
        physical_width,
        physical_height,
        forward=True,
    )


def complete_rendered_project(
    state: ProjectState,
    fig: Any,
    axes: dict[str, Any],
    *,
    validate_layout: bool = True,
) -> tuple[Any, dict[str, Any]]:
    collision_groups: dict[str, list[Any]] = {}
    contained_artists: list[tuple[Any, Any, str]] = []
    leader_groups: dict[str, tuple[list[Any], list[Any]]] = {}
    main_ax = axes.get("main")

    annotation_texts = [
        item[1]
        for item in axes.get("annotations", [])
        if isinstance(item, tuple) and len(item) == 2 and item[1] is not None
    ]
    if annotation_texts:
        collision_groups["peak annotations"] = annotation_texts
        contained_artists.extend(
            (artist, artist.axes, "Peak annotation") for artist in annotation_texts
        )

    for key, label in (
        ("direct_labels", "direct curve labels"),
        ("phase_row_labels", "phase row labels"),
        ("phase_peak_labels", "phase peak labels"),
    ):
        artists = list(axes.get(key, []))
        if artists:
            collision_groups[label] = artists
    direct_labels = list(axes.get("direct_labels", []))
    direct_label_leaders = list(axes.get("direct_label_leaders", []))
    if direct_labels or direct_label_leaders:
        if len(direct_labels) != len(direct_label_leaders):
            raise ValueError(
                "Direct-label text and leader counts must match before layout validation"
            )
        leader_groups["direct curve-label leaders"] = (
            direct_label_leaders,
            direct_labels,
        )
    phase_row_labels = list(axes.get("phase_row_labels", []))
    phase_peak_labels = list(axes.get("phase_peak_labels", []))
    if main_ax is not None:
        contained_artists.extend(
            (artist, main_ax, "Phase peak label")
            for artist in phase_peak_labels
        )
        y_axis_text = [
            artist
            for artist in (
                *main_ax.get_yticklabels(),
                main_ax.yaxis.get_offset_text(),
                main_ax.yaxis.label,
            )
            if artist.get_visible() and artist.get_text().strip()
        ]
        if phase_row_labels and y_axis_text:
            collision_groups["phase rows and y-axis decorations"] = [
                *phase_row_labels,
                *y_axis_text,
            ]

    metric_artist = axes.get("metrics")
    legend = main_ax.get_legend() if main_ax is not None else None
    inset = axes.get("inset")
    main_decorations = [
        artist
        for artist in (
            legend,
            inset,
            metric_artist,
            *annotation_texts,
            *phase_peak_labels,
        )
        if artist is not None
        and (
            artist is inset
            or not hasattr(artist, "axes")
            or artist.axes is main_ax
        )
    ]
    if len(main_decorations) > 1:
        collision_groups["main-panel decorations"] = main_decorations
    if direct_label_leaders and main_decorations:
        leader_groups["direct-label leaders and main-panel decorations"] = (
            direct_label_leaders,
            main_decorations,
        )
    bragg_band_guard = axes.get("bragg_band_guard")
    if legend is not None and bragg_band_guard is not None:
        collision_groups["legend and Bragg phase band"] = [
            legend,
            bragg_band_guard,
        ]

    panel_headers = axes.get("panel_headers", [])
    panels = axes.get("panels", [])
    for index, artists in enumerate(panel_headers):
        panel_artists = list(artists)
        if index < len(panels):
            panel_artists.extend(
                artist for artist in annotation_texts if artist.axes is panels[index]
            )
            contained_artists.extend(
                (artist, panels[index], "Small-multiple panel header")
                for artist in artists
            )
        collision_groups[f"small-multiple panel {index + 1} decorations"] = (
            panel_artists
        )

    if metric_artist is not None and main_ax is not None:
        contained_artists.append((metric_artist, main_ax, "Metric summary"))

    if validate_layout:
        validate_figure_layout(
            fig,
            collision_groups=collision_groups,
            contained_artists=contained_artists,
            leader_groups=leader_groups,
        )
    axes["text_alternative"] = rendered_view_text_alternative(state, axes=axes)
    return fig, axes


def rendered_view_text_alternative(
    state: ProjectState,
    *,
    axes: dict[str, Any] | None = None,
) -> str:
    """Return a concise, attachable text alternative for the active view."""

    settings = state.settings
    mode = settings.view_mode
    if mode == "map" and state.map_data is not None:
        data = state.map_data
        intensity = np.asarray(data.intensity, dtype=float)
        populated = np.isfinite(intensity)
        if data.counts is not None:
            populated &= np.asarray(data.counts, dtype=float) > 0.0
        intensity_range = _accessible_range(intensity, populated)
        return (
            f"Map view ({data.kind}) with {len(data.y)} rows and {len(data.x)} columns; "
            f"horizontal axis {data.x_label}, vertical axis {data.y_label}; "
            f"populated {data.intensity_label} range {intensity_range}."
        )
    if mode == "derived" and state.derived_plot is not None:
        data = state.derived_plot
        points = len(data.scatter) if data.scatter else len(data.x)
        metrics = ", ".join(f"{key}={value}" for key, value in data.metrics.items())
        suffix = f" Metrics: {metrics}." if metrics else ""
        return f"Derived {data.kind} view with {points} data points.{suffix}"
    if mode == "refinement" and state.fit is not None:
        observed = np.asarray(state.fit.observed, dtype=float)
        calculated = np.asarray(state.fit.calculated, dtype=float)
        residual = observed - calculated
        metrics = []
        if state.fit.rp is not None:
            metrics.append(f"Rp={state.fit.rp:.4g} percent")
        if state.fit.rwp is not None:
            metrics.append(f"Rwp={state.fit.rwp:.4g} percent")
        metric_text = f" Metrics: {', '.join(metrics)}." if metrics else ""
        marker_count = len((axes or {}).get("observed_marker_indices", ()))
        marker_text = ""
        if marker_count and marker_count < len(state.fit.x):
            marker_text = (
                f" Observed display markers: {marker_count} of {len(state.fit.x)}; "
                "calculated/residual curves and exported data retain all points."
            )
        return (
            f"Refinement view with {len(state.fit.x)} observed and calculated points; "
            f"observed range {_accessible_range(observed)}, residual range "
            f"{_accessible_range(residual)}.{metric_text}{marker_text}"
        )
    if mode == "small_multiples":
        visible = sum(1 for layer in state.spectra if layer.visible)
        return f"Small-multiples view with {visible} visible spectrum panels."
    visible = sum(1 for layer in state.spectra if layer.visible)
    finite_x = [
        float(value)
        for layer in state.spectra
        if layer.visible
        for value in layer.x
        if math.isfinite(float(value))
    ]
    x_range = _accessible_range(np.asarray(finite_x, dtype=float))
    return (
        f"Spectrum view ({mode}) with {visible} visible spectrum lines; "
        f"source-coordinate range {x_range}."
    )


def _accessible_range(values: np.ndarray, mask: np.ndarray | None = None) -> str:
    array = np.asarray(values, dtype=float)
    finite = np.isfinite(array)
    if mask is not None:
        finite &= np.asarray(mask, dtype=bool)
    if not bool(np.any(finite)):
        return "unavailable"
    minimum = float(np.min(array, where=finite, initial=np.inf))
    maximum = float(np.max(array, where=finite, initial=-np.inf))
    return f"{minimum:.6g} to {maximum:.6g}"
