from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from xrdviz.axes import convert_x
from xrdviz.models import PUBLICATION_PALETTE, PlotSettings, ProjectState, SpectrumLayer
from xrdviz.transforms import transform_intensity

TIME_RE = re.compile(r"(?P<value>[+-]?\d+(?:\.\d+)?)\s*(?P<unit>ms|sec|s|min|m|hr|h)(?![A-Za-z])", re.IGNORECASE)
TEMP_RE = re.compile(r"(?P<value>[+-]?\d+(?:\.\d+)?)\s*(?P<unit>degc|°c|c|k)(?![A-Za-z])", re.IGNORECASE)
NAMED_FRAME_RE = re.compile(
    r"(?:az[_ -]?full|frame|frm|scan|image|img|full)[_ -]*(?P<frame>\d+)",
    re.IGNORECASE,
)
INTEGER_CHUNK_RE = re.compile(r"(?<![\d.])(\d+)(?![\d.])")


@dataclass(slots=True)
class SpectrumMetadata:
    frame_index: int | None = None
    time_s: float | None = None
    temperature: float | None = None
    temperature_unit: str = ""
    group: str = ""


def parse_spectrum_metadata(path: str | Path) -> SpectrumMetadata:
    file_path = Path(path)
    stem = file_path.stem
    temperature, temperature_unit = _parse_temperature(stem)
    return SpectrumMetadata(
        frame_index=_parse_frame_index(stem),
        time_s=_parse_time_s(stem),
        temperature=temperature,
        temperature_unit=temperature_unit,
        group=file_path.parent.name if str(file_path.parent) not in {"", "."} else "",
    )


def apply_batch_metadata(
    layers: list[SpectrumLayer],
    *,
    sort_by: str = "frame",
    color_by: str = "frame",
    colormap: str = "blue_rose",
) -> None:
    for fallback_index, layer in enumerate(layers):
        meta_source = layer.source_path or layer.name
        meta = parse_spectrum_metadata(meta_source)
        if layer.frame_index is None:
            layer.frame_index = meta.frame_index if meta.frame_index is not None else fallback_index
        if layer.time_s is None:
            layer.time_s = meta.time_s
        if layer.temperature is None:
            layer.temperature = meta.temperature
        if not layer.temperature_unit:
            layer.temperature_unit = meta.temperature_unit
        if not layer.group:
            layer.group = meta.group

    layers.sort(key=lambda layer: _sort_key(layer, sort_by))
    for order, layer in enumerate(layers):
        layer.order = order
    for layer in layers:
        value = layer_batch_value(layer, color_by)
        layer.color_value = float(value if value is not None else layer.order)


def select_spectrum_layers(layers: Iterable[SpectrumLayer], *, show_every_n: int = 1) -> list[SpectrumLayer]:
    step = max(int(show_every_n), 1)
    visible = sorted([layer for layer in layers if layer.visible], key=lambda layer: (layer.order, layer.name))
    return [layer for index, layer in enumerate(visible) if index % step == 0]


def layer_batch_value(layer: SpectrumLayer, field: str) -> float | None:
    if field == "frame":
        return None if layer.frame_index is None else float(layer.frame_index)
    if field == "time":
        return layer.time_s
    if field == "temperature":
        return layer.temperature
    if field == "color_value":
        return layer.color_value
    return float(layer.order)


def assign_gradient_colors(layers: Iterable[SpectrumLayer], *, colormap: str = "blue_rose") -> None:
    layers = list(layers)
    values = [layer.color_value for layer in layers if layer.color_value is not None and math.isfinite(layer.color_value)]
    if not values:
        return
    vmin = min(values)
    vmax = max(values)
    cmap = matplotlib_colormap(colormap)
    for layer in layers:
        value = layer.color_value if layer.color_value is not None else float(layer.order)
        layer.color = color_for_value(value, vmin, vmax, colormap=colormap, cmap=cmap)


def color_for_value(value: float, vmin: float, vmax: float, *, colormap: str = "blue_rose", cmap=None) -> str:
    cmap = cmap or matplotlib_colormap(colormap)
    if not math.isfinite(value):
        value = vmin
    fraction = 0.5 if vmin == vmax else (value - vmin) / (vmax - vmin)
    fraction = min(max(fraction, 0.0), 1.0)
    from matplotlib.colors import to_hex

    return to_hex(cmap(fraction), keep_alpha=False)


def make_heatmap_matrix(state: ProjectState) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    settings = state.settings
    layers = select_spectrum_layers(state.spectra, show_every_n=settings.show_every_n)
    if not layers:
        return np.array([]), np.array([]), np.empty((0, 0))

    x_min, x_max = _heatmap_x_range(layers, settings)
    x_grid = np.linspace(x_min, x_max, settings.heatmap_points)
    rows: list[np.ndarray] = []
    row_values: list[float] = []
    for index, layer in enumerate(layers):
        converted_x = convert_x(layer.x, layer.axis_kind, settings.x_axis, settings.energy_kev)
        transformed_y = transform_intensity(
            layer.y,
            normalize=settings.normalize,
            log_scale=settings.log_scale,
            epsilon=settings.log_epsilon,
        )
        pairs = sorted(
            (x_value, y_value)
            for x_value, y_value in zip(converted_x, transformed_y)
            if math.isfinite(x_value) and math.isfinite(y_value)
        )
        if len(pairs) < 2:
            rows.append(np.full_like(x_grid, np.nan, dtype=float))
        else:
            x_values, y_values = zip(*pairs)
            rows.append(np.interp(x_grid, np.asarray(x_values), np.asarray(y_values), left=np.nan, right=np.nan))
        row_value = layer_batch_value(layer, settings.sort_by)
        row_values.append(float(row_value if row_value is not None else index))

    return x_grid, np.asarray(row_values, dtype=float), np.vstack(rows)


def colorbar_label(field: str) -> str:
    return {
        "frame": "Frame",
        "time": "Time (s)",
        "temperature": "Temperature",
        "color_value": "Value",
        "order": "Order",
    }.get(field, field)


def matplotlib_colormap(name: str):
    from matplotlib import colormaps
    from matplotlib.colors import LinearSegmentedColormap

    if name == "blue_rose":
        return LinearSegmentedColormap.from_list("xrdviz_blue_rose", PUBLICATION_PALETTE[:5])
    try:
        return colormaps.get_cmap(name)
    except ValueError:
        return colormaps.get_cmap("viridis")


def _parse_frame_index(stem: str) -> int | None:
    named = NAMED_FRAME_RE.search(stem)
    if named:
        return int(named.group("frame"))
    chunks = INTEGER_CHUNK_RE.findall(stem)
    return int(chunks[-1]) if chunks else None


def _parse_time_s(stem: str) -> float | None:
    match = TIME_RE.search(stem)
    if not match:
        return None
    value = float(match.group("value"))
    unit = match.group("unit").lower()
    if unit == "ms":
        return value / 1000.0
    if unit in {"min", "m"}:
        return value * 60.0
    if unit in {"hr", "h"}:
        return value * 3600.0
    return value


def _parse_temperature(stem: str) -> tuple[float | None, str]:
    match = TEMP_RE.search(stem)
    if not match:
        return None, ""
    unit = match.group("unit").upper().replace("DEGC", "C").replace("°C", "C")
    return float(match.group("value")), unit


def _sort_key(layer: SpectrumLayer, sort_by: str) -> tuple[int, float, str]:
    value = layer_batch_value(layer, sort_by)
    if value is None or not math.isfinite(value):
        return (1, float(layer.order), layer.name)
    return (0, float(value), layer.name)


def _heatmap_x_range(layers: list[SpectrumLayer], settings: PlotSettings) -> tuple[float, float]:
    if settings.x_min is not None and settings.x_max is not None:
        return settings.x_min, settings.x_max

    values: list[float] = []
    for layer in layers:
        converted = convert_x(layer.x, layer.axis_kind, settings.x_axis, settings.energy_kev)
        values.extend(value for value in converted if math.isfinite(value))
    if not values:
        raise ValueError("No finite x values are available for heatmap rendering")
    x_min = settings.x_min if settings.x_min is not None else min(values)
    x_max = settings.x_max if settings.x_max is not None else max(values)
    if x_min >= x_max:
        raise ValueError("Heatmap x range is invalid")
    return x_min, x_max
