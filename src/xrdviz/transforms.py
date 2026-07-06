from __future__ import annotations

import math
from collections.abc import Iterable

from xrdviz.models import PlotSettings, SpectrumLayer


def transform_intensity(
    values: Iterable[float],
    *,
    normalize: bool,
    log_scale: bool,
    epsilon: float = 1e-9,
    vertical_offset: float = 0.0,
) -> list[float]:
    raw = [float(value) for value in values]
    scale = _positive_max(raw) if normalize else 1.0
    transformed: list[float] = []
    for value in raw:
        scaled = value / scale
        if log_scale:
            scaled = math.log10(max(scaled, epsilon))
        transformed.append(scaled + vertical_offset)
    return transformed


def display_y_for_layer(layer: SpectrumLayer, settings: PlotSettings, layer_index: int) -> list[float]:
    stack_mode = settings.stack_enabled or settings.view_mode in {"stack", "gradient_stack"}
    stack_offset = settings.stack_spacing * layer_index if stack_mode else 0.0
    return transform_intensity(
        layer.y,
        normalize=settings.normalize,
        log_scale=settings.log_scale,
        epsilon=settings.log_epsilon,
        vertical_offset=layer.offset + stack_offset,
    )


def _positive_max(values: list[float]) -> float:
    positives = [value for value in values if math.isfinite(value) and value > 0.0]
    if not positives:
        return 1.0
    return max(positives)
