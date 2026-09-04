from __future__ import annotations

import math
from itertools import combinations
from typing import Any, Iterable, Mapping, Sequence


def text_width_points(
    text: str,
    *,
    font_family: str,
    font_size: float,
) -> float:
    """Estimate rendered text width without depending on an active canvas."""

    if not text:
        return 0.0
    try:
        from matplotlib.font_manager import FontProperties
        from matplotlib.textpath import TextPath

        properties = FontProperties(family=font_family, size=font_size)
        return float(TextPath((0.0, 0.0), text, prop=properties).get_extents().width)
    except (RuntimeError, TypeError, ValueError):
        return len(text) * float(font_size) * 0.75


def wrap_text(
    text: str,
    settings: Any,
    *,
    max_width_points: float,
    font_size: float | None = None,
) -> str:
    """Wrap prose and unbroken identifiers to a physical width in points."""

    value = str(text)
    size = float(settings.tick_label_size if font_size is None else font_size)
    lines: list[str] = []
    for paragraph in value.splitlines() or [""]:
        current = ""
        for character in paragraph:
            candidate = current + character
            if (
                current
                and text_width_points(
                    candidate,
                    font_family=settings.font_family,
                    font_size=size,
                )
                > max_width_points
            ):
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
    return "\n".join(line for line in lines if line) or value


def prepare_side_labels(labels: Iterable[str], settings: Any) -> list[str]:
    figure_width_points = max(float(settings.figure_width_in) * 72.0, 1.0)
    max_width_points = max(36.0, figure_width_points * 0.24)
    return [
        wrap_text(
            label,
            settings,
            max_width_points=max_width_points,
            font_size=settings.tick_label_size,
        )
        for label in labels
    ]


def prepare_panel_title(text: str, settings: Any) -> str:
    if not str(text).strip():
        return ""
    title_size = max(float(settings.axis_label_size), float(settings.font_size))
    figure_width_points = max(float(settings.figure_width_in) * 72.0, 1.0)
    return wrap_text(
        str(text).strip(),
        settings,
        max_width_points=max(72.0, figure_width_points * 0.68),
        font_size=title_size,
    )


def prepare_metric_lines(lines: Iterable[str], settings: Any) -> str:
    figure_width_points = max(float(settings.figure_width_in) * 72.0, 1.0)
    max_width_points = max(64.0, figure_width_points * 0.46)
    return "\n".join(
        wrap_text(
            line,
            settings,
            max_width_points=max_width_points,
            font_size=settings.tick_label_size,
        )
        for line in lines
    )


def side_label_width_points(labels: Iterable[str], settings: Any) -> float:
    return max(
        (
            text_width_points(
                line,
                font_family=settings.font_family,
                font_size=settings.tick_label_size,
            )
            for label in labels
            for line in str(label).splitlines()
        ),
        default=0.0,
    )


def side_label_height_points(labels: Iterable[str], settings: Any) -> float:
    values = list(labels)
    font_size = float(settings.tick_label_size)
    line_height = font_size * 1.25
    label_spacing = font_size * 0.5
    height = sum(max(len(str(label).splitlines()), 1) * line_height for label in values)
    return height + max(len(values) - 1, 0) * label_spacing + font_size * 0.8


def phase_label_offset_points(settings: Any) -> float:
    """Keep Bragg row names close to the frame in their data-free band."""

    return 3.0


def safe_subplot_margins(
    settings: Any,
    *,
    title: str = "",
    left_labels: Sequence[str] = (),
    left_decoration_points: float = 6.0,
    right_labels: Sequence[str] = (),
    right_decoration_points: float = 8.0,
    colorbar: bool = False,
) -> dict[str, float]:
    """Return fixed-canvas margins that keep publication decorations on-page."""

    figure_width_points = max(float(settings.figure_width_in) * 72.0, 1.0)
    figure_height_points = max(float(settings.figure_height_in) * 72.0, 1.0)
    tick_size = float(settings.tick_label_size)
    axis_size = float(settings.axis_label_size)

    left = max(
        float(settings.margin_left),
        (axis_size * 2.0 + tick_size * 1.6 + 5.0) / figure_width_points,
    )
    right = min(
        float(settings.margin_right),
        1.0 - max(8.0, tick_size * 1.75) / figure_width_points,
    )
    bottom = max(
        float(settings.margin_bottom),
        (axis_size * 2.0 + tick_size * 1.35 + 4.0) / figure_height_points,
    )
    top = min(
        float(settings.margin_top),
        1.0 - max(5.0, tick_size * 0.8) / figure_height_points,
    )

    if title:
        title_size = max(float(settings.axis_label_size), float(settings.font_size))
        title_lines = max(len(str(title).splitlines()), 1)
        title_height = title_lines * title_size * 1.25 + 4.0
        top = min(top, 1.0 - title_height / figure_height_points)
    if left_labels:
        required = (
            side_label_width_points(left_labels, settings)
            + float(left_decoration_points)
            + 6.0
        )
        left = max(left, required / figure_width_points)
    if right_labels:
        required = side_label_width_points(right_labels, settings) + float(
            right_decoration_points
        )
        right = min(right, 1.0 - required / figure_width_points)
    if colorbar:
        right = min(right, 0.84)

    if right - left < 0.34:
        raise ValueError(
            "Publication side labels do not leave enough plot width at the selected "
            "figure size; use shorter labels or a wider canvas."
        )
    if top - bottom < 0.32:
        raise ValueError(
            "Publication title and axis decorations do not leave enough plot height at "
            "the selected figure size; shorten the title or use a taller canvas."
        )
    return {"left": left, "right": right, "top": top, "bottom": bottom}


def set_panel_title(ax: Any, text: str, settings: Any) -> Any | None:
    wrapped = prepare_panel_title(text, settings)
    if not wrapped:
        return None
    return ax.set_title(
        wrapped,
        loc="left",
        pad=3.0,
        fontsize=max(settings.axis_label_size, settings.font_size),
        fontweight="bold",
        color="#262626",
    )


def add_direct_labels(
    ax: Any,
    handles: Sequence[Any],
    labels: Sequence[str],
    settings: Any,
) -> list[Any]:
    """Place curve labels in a measured side gutter with collision-free leaders."""

    if not handles:
        return []
    if len(handles) != len(labels):
        raise ValueError("Direct-label handles and labels must have matching lengths")

    axes_box = ax.get_window_extent()
    axes_height_points = max(
        axes_box.height * 72.0 / max(float(ax.figure.dpi), 1.0),
        1.0,
    )
    font_size = float(settings.tick_label_size)
    heights = [
        max(len(str(label).splitlines()), 1) * font_size * 1.25 / axes_height_points
        for label in labels
    ]
    gap = font_size * 0.45 / axes_height_points

    anchor_points = [_rightmost_visible_point(ax, handle) for handle in handles]
    targets = [_display_y_fraction(ax, point[0], point[1]) for point in anchor_points]
    centers = _distribute_centers(
        targets,
        heights,
        lower=0.035,
        upper=0.965,
        gap=gap,
        error_message=(
            "Direct labels do not fit vertically at the selected figure size; "
            "use shorter labels, fewer displayed curves, or a taller canvas."
        ),
    )

    artists: list[Any] = []
    for handle, label, point, center in zip(handles, labels, anchor_points, centers):
        artists.append(
            ax.annotate(
                label,
                xy=point,
                xycoords="data",
                xytext=(1.015, center),
                textcoords=ax.transAxes,
                ha="left",
                va="center",
                fontsize=settings.tick_label_size,
                color="#262626",
                annotation_clip=False,
                clip_on=False,
                arrowprops={
                    "arrowstyle": "-",
                    "color": handle.get_color(),
                    "linewidth": max(0.45, float(settings.line_width) * 0.65),
                    "shrinkA": 1.5,
                    "shrinkB": 1.5,
                },
            )
        )
    return artists


def stagger_rotated_label_tops(
    ax: Any,
    *,
    x_values: Sequence[float],
    labels: Sequence[str],
    desired_tops: Sequence[float],
    settings: Any,
) -> tuple[list[float], list[float], float]:
    """Assign nearby 90-degree labels to measured horizontal/vertical lanes."""

    if not (len(x_values) == len(labels) == len(desired_tops)):
        raise ValueError("Annotation layout inputs must have matching lengths")
    axes_box = ax.get_window_extent()
    dpi = max(float(ax.figure.dpi), 1.0)
    width_points = max(axes_box.width * 72.0 / dpi, 1.0)
    height_points = max(axes_box.height * 72.0 / dpi, 1.0)
    font_size = float(settings.tick_label_size)
    label_width = font_size * 1.25 / width_points
    gap_x = font_size * 0.35 / width_points
    gap_y = font_size * 0.45 / height_points

    rectangles: list[tuple[float, float, float, float]] = []
    label_x_values: list[float] = []
    tops: list[float] = []
    for x_value, label, desired in zip(x_values, labels, desired_tops):
        x_fraction = _display_x_fraction(ax, x_value)
        if not 0.0 <= x_fraction <= 1.0:
            label_x_values.append(float(x_value))
            tops.append(float(desired))
            continue
        label_height = max(
            text_width_points(
                str(label),
                font_family=settings.font_family,
                font_size=font_size,
            )
            / height_points,
            font_size * 1.25 / height_points,
        )
        base_top = min(max(float(desired), label_height + 0.025), 0.975)
        lane_step = label_width + gap_x
        max_shift = min(0.14, max(x_fraction, 1.0 - x_fraction))
        lane_offsets = [0.0]
        lane = 1
        while lane * lane_step <= max_shift + 1e-12:
            lane_offsets.extend((lane * lane_step, -lane * lane_step))
            lane += 1
        placement: tuple[float, float, tuple[float, float, float, float]] | None = None
        for lane_index, offset in enumerate(lane_offsets):
            candidate_x = x_fraction + offset
            if not label_width / 2.0 <= candidate_x <= 1.0 - label_width / 2.0:
                continue
            candidate_top = base_top - min(lane_index, 2) * gap_y * 0.35
            rectangle = (
                candidate_x - label_width / 2.0,
                candidate_x + label_width / 2.0,
                candidate_top - label_height,
                candidate_top,
            )
            if not any(
                _rectangles_overlap(rectangle, other, gap_x=gap_x, gap_y=gap_y)
                for other in rectangles
            ):
                placement = (candidate_x, candidate_top, rectangle)
                break
        if placement is None:
            raise ValueError(
                "Peak annotations do not fit without overlap at the selected figure "
                "size; shorten labels, separate their positions, or use a wider canvas."
            )
        candidate_x, candidate_top, rectangle = placement
        rectangles.append(rectangle)
        display_x = ax.transAxes.transform((candidate_x, 0.5))[0]
        label_x_values.append(
            float(ax.transData.inverted().transform((display_x, 0.0))[0])
        )
        tops.append(candidate_top)
    header_fraction = 0.0
    if rectangles:
        header_fraction = 1.0 - min(rectangle[2] for rectangle in rectangles) + 0.035
        if header_fraction > 0.48:
            raise ValueError(
                "Peak annotations require too much vertical space at the selected figure "
                "size; shorten labels or use a taller canvas."
            )
    return label_x_values, tops, header_fraction


def reserve_axes_top(ax: Any, fraction: float) -> None:
    """Reserve a data-free header band without changing the fixed canvas."""

    value = float(fraction)
    if value <= 0.0:
        return
    if not math.isfinite(value) or value >= 0.6:
        raise ValueError("Reserved plot header fraction must be finite and below 0.6")
    lower, upper = (float(item) for item in ax.get_ylim())
    span = upper - lower
    if not math.isfinite(span) or span == 0.0:
        return
    ax.set_ylim(lower, lower + span / (1.0 - value))


def validate_figure_layout(
    fig: Any,
    *,
    collision_groups: Mapping[str, Sequence[Any]] | None = None,
    contained_artists: Sequence[tuple[Any, Any, str]] = (),
) -> None:
    """Fail closed when a fixed-size export would clip or collide decorations."""

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.legend import Legend
    from matplotlib.text import Text

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    figure_box = fig.bbox
    tolerance = max(1.0, float(fig.dpi) / 144.0)

    tick_labels = {
        artist
        for ax in fig.axes
        for artist in (*ax.get_xticklabels(), *ax.get_yticklabels())
    }
    hidden_edge_ticks = False
    for artist in tick_labels:
        if not artist.get_visible():
            continue
        box = _artist_box(artist, renderer)
        if box is not None and (
            box.x0 < figure_box.x0 - tolerance
            or box.y0 < figure_box.y0 - tolerance
            or box.x1 > figure_box.x1 + tolerance
            or box.y1 > figure_box.y1 + tolerance
        ):
            artist.set_visible(False)
            hidden_edge_ticks = True
    if hidden_edge_ticks:
        canvas.draw()
        renderer = canvas.get_renderer()

    bounded_artists: list[tuple[Any, str]] = []
    for artist in fig.findobj(match=lambda item: isinstance(item, (Text, Legend))):
        if not artist.get_visible():
            continue
        if isinstance(artist, Text) and not artist.get_text().strip():
            continue
        label = _artist_label(artist)
        bounded_artists.append((artist, label))

    overflows: list[str] = []
    for artist, label in bounded_artists:
        box = _artist_box(artist, renderer)
        if box is None:
            continue
        if (
            box.x0 < figure_box.x0 - tolerance
            or box.y0 < figure_box.y0 - tolerance
            or box.x1 > figure_box.x1 + tolerance
            or box.y1 > figure_box.y1 + tolerance
        ):
            overflows.append(label)
    if overflows:
        preview = ", ".join(dict.fromkeys(overflows[:4]))
        raise ValueError(
            "Publication layout extends beyond the fixed figure canvas: "
            f"{preview}. Shorten the affected text or use a wider/taller canvas."
        )

    for artist, ax, label in contained_artists:
        box = _artist_box(artist, renderer)
        if box is None:
            continue
        axes_box = ax.get_window_extent(renderer)
        if (
            box.x0 < axes_box.x0 - tolerance
            or box.y0 < axes_box.y0 - tolerance
            or box.x1 > axes_box.x1 + tolerance
            or box.y1 > axes_box.y1 + tolerance
        ):
            raise ValueError(
                f"{label} does not fit inside its plotting panel at the selected figure "
                "size; shorten the text or use a larger canvas."
            )

    for group_name, artists in (collision_groups or {}).items():
        visible = [
            artist for artist in artists if artist is not None and artist.get_visible()
        ]
        for left, right in combinations(visible, 2):
            left_box = _artist_box(left, renderer)
            right_box = _artist_box(right, renderer)
            if left_box is None or right_box is None:
                continue
            if _boxes_overlap(left_box, right_box, tolerance=tolerance):
                raise ValueError(
                    f"Publication layout collision in {group_name}: "
                    f"{_artist_label(left)} overlaps {_artist_label(right)}. "
                    "Shorten labels, reduce displayed items, or use a larger canvas."
                )


def _rightmost_visible_point(ax: Any, handle: Any) -> tuple[float, float]:
    x_values = list(handle.get_xdata())
    y_values = list(handle.get_ydata())
    x_min, x_max = sorted(float(value) for value in ax.get_xlim())
    points = [
        (float(x_value), float(y_value))
        for x_value, y_value in zip(x_values, y_values)
        if math.isfinite(float(x_value))
        and math.isfinite(float(y_value))
        and x_min <= float(x_value) <= x_max
    ]
    if not points:
        raise ValueError(
            "Direct label curve has no finite point in the displayed x range"
        )
    return max(points, key=lambda point: ax.transData.transform(point)[0])


def _display_x_fraction(ax: Any, x_value: float) -> float:
    axes_box = ax.get_window_extent()
    display_x = ax.transData.transform((float(x_value), 0.0))[0]
    return (display_x - axes_box.x0) / max(axes_box.width, 1.0)


def _display_y_fraction(ax: Any, x_value: float, y_value: float) -> float:
    axes_box = ax.get_window_extent()
    display_y = ax.transData.transform((float(x_value), float(y_value)))[1]
    return min(max((display_y - axes_box.y0) / max(axes_box.height, 1.0), 0.035), 0.965)


def _distribute_centers(
    targets: Sequence[float],
    heights: Sequence[float],
    *,
    lower: float,
    upper: float,
    gap: float,
    error_message: str,
) -> list[float]:
    if sum(heights) + max(len(heights) - 1, 0) * gap > upper - lower:
        raise ValueError(error_message)
    order = sorted(range(len(targets)), key=lambda index: targets[index])
    centers: dict[int, float] = {}
    previous_center: float | None = None
    previous_height = 0.0
    for index in order:
        height = heights[index]
        center = min(
            max(float(targets[index]), lower + height / 2.0), upper - height / 2.0
        )
        if previous_center is not None:
            center = max(
                center, previous_center + previous_height / 2.0 + height / 2.0 + gap
            )
        centers[index] = center
        previous_center = center
        previous_height = height

    last = order[-1]
    overflow = centers[last] + heights[last] / 2.0 - upper
    if overflow > 0.0:
        for index in order:
            centers[index] -= overflow
    for position in range(len(order) - 2, -1, -1):
        index = order[position]
        next_index = order[position + 1]
        centers[index] = min(
            centers[index],
            centers[next_index]
            - heights[next_index] / 2.0
            - heights[index] / 2.0
            - gap,
        )
    first = order[0]
    if centers[first] - heights[first] / 2.0 < lower - 1.0e-9:
        raise ValueError(error_message)
    return [centers[index] for index in range(len(targets))]


def _rectangles_overlap(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
    *,
    gap_x: float,
    gap_y: float,
) -> bool:
    return not (
        left[1] + gap_x <= right[0]
        or right[1] + gap_x <= left[0]
        or left[3] + gap_y <= right[2]
        or right[3] + gap_y <= left[2]
    )


def _artist_box(artist: Any, renderer: Any) -> Any | None:
    try:
        from matplotlib.text import Annotation, Text

        box = (
            Text.get_window_extent(artist, renderer)
            if isinstance(artist, Annotation)
            else artist.get_window_extent(renderer)
        )
    except (AttributeError, RuntimeError, ValueError):
        return None
    if not all(math.isfinite(float(value)) for value in box.extents):
        return None
    return box


def _artist_label(artist: Any) -> str:
    text = getattr(artist, "get_text", lambda: "")()
    if str(text).strip():
        return repr(str(text).replace("\n", " / ")[:80])
    label = getattr(artist, "get_label", lambda: "")()
    return repr(str(label)[:80] or artist.__class__.__name__)


def _boxes_overlap(left: Any, right: Any, *, tolerance: float) -> bool:
    overlap_x = min(left.x1, right.x1) - max(left.x0, right.x0)
    overlap_y = min(left.y1, right.y1) - max(left.y0, right.y0)
    return overlap_x > tolerance and overlap_y > tolerance
