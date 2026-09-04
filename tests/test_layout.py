from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.text import Annotation, Text
from matplotlib.transforms import Bbox

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xrdviz.models import (
    PhaseLayer,
    PhasePeak,
    PlotAnnotation,
    PlotSettings,
    ProjectState,
    SpectrumLayer,
)
from xrdviz.plot.renderer import render_project


def _box(artist, renderer):
    if isinstance(artist, Annotation):
        return Text.get_window_extent(artist, renderer)
    return artist.get_window_extent(renderer)


def _assert_inside(outer, inner, *, tolerance=1.0):
    assert inner.x0 >= outer.x0 - tolerance
    assert inner.y0 >= outer.y0 - tolerance
    assert inner.x1 <= outer.x1 + tolerance
    assert inner.y1 <= outer.y1 + tolerance


def _assert_no_overlap(artists, renderer):
    for left, right in combinations(artists, 2):
        left_box = _box(left, renderer)
        right_box = _box(right, renderer)
        overlap_x = min(left_box.x1, right_box.x1) - max(left_box.x0, right_box.x0)
        overlap_y = min(left_box.y1, right_box.y1) - max(left_box.y0, right_box.y0)
        assert overlap_x <= 1.0 or overlap_y <= 1.0


def test_default_publication_legend_uses_least_obstructive_location():
    assert PlotSettings().legend_location == "best"


def test_direct_curve_labels_are_wrapped_distributed_and_inside_canvas():
    state = ProjectState(
        spectra=[
            SpectrumLayer(
                name=f"Anneal condition {index}",
                x=[20.0, 30.0, 40.0],
                y=[1.0, 3.0, 1.0 + index * 0.01],
                order=index,
            )
            for index in range(4)
        ],
        settings=PlotSettings(
            direct_labels=True,
            show_legend=True,
            show_phase_legend=False,
        ),
    )

    figure, axes = render_project(state)
    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    renderer = canvas.get_renderer()
    labels = axes["direct_labels"]

    assert axes["main"].get_legend() is None
    assert len(labels) == 4
    _assert_no_overlap(labels, renderer)
    for label in labels:
        _assert_inside(figure.bbox, _box(label, renderer))


def test_direct_curve_label_leaders_do_not_cross_other_label_text():
    state = ProjectState(
        spectra=[
            SpectrumLayer(
                name=name,
                x=[20.0, 30.0, 40.0],
                y=[1.0, 3.0 - index * 0.1, 1.0 + index * 0.002],
                order=index,
            )
            for index, name in enumerate(
                (
                    "Hydrogen-free reference",
                    "Hydrogen-charged condition",
                    "Recovered after unloading",
                )
            )
        ],
        settings=PlotSettings(
            direct_labels=True,
            show_legend=False,
            show_phase_legend=False,
        ),
    )

    figure, axes = render_project(state)
    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    renderer = canvas.get_renderer()
    labels = axes["direct_labels"]
    leaders = axes["direct_label_leaders"]

    assert len(leaders) == len(labels)
    for leader_index, leader in enumerate(leaders):
        display_path = leader.get_path().transformed(leader.get_transform())
        for label_index, label in enumerate(labels):
            if leader_index == label_index:
                continue
            label_box = _box(label, renderer)
            expanded_box = Bbox.from_extents(
                label_box.x0 - 0.5,
                label_box.y0 - 0.5,
                label_box.x1 + 0.5,
                label_box.y1 + 0.5,
            )
            assert not display_path.intersects_bbox(expanded_box, filled=False)


def test_direct_curve_labels_fit_double_column_gutter():
    state = ProjectState(
        spectra=[
            SpectrumLayer(
                name=name,
                x=[20.0, 30.0, 40.0],
                y=[1.0, 3.0 - index * 0.1, 1.0 + index * 0.002],
                order=index,
            )
            for index, name in enumerate(
                (
                    "Hydrogen-free diffraction reference",
                    "Hydrogen-charged diffraction condition",
                    "Recovered diffraction state after unloading",
                )
            )
        ],
        settings=PlotSettings(
            figure_width_in=183.0 / 25.4,
            direct_labels=True,
            show_legend=False,
            show_phase_legend=False,
        ),
    )

    figure, axes = render_project(state)
    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    renderer = canvas.get_renderer()

    for label in axes["direct_labels"]:
        _assert_inside(figure.bbox, _box(label, renderer))


def test_direct_curve_label_leader_cannot_be_hidden_by_inset():
    state = ProjectState(
        spectra=[
            SpectrumLayer(
                name="Measured profile",
                x=[20.0, 30.0, 40.0],
                y=[0.2, 1.0, 0.8],
            )
        ],
        settings=PlotSettings(
            direct_labels=True,
            show_legend=False,
            show_phase_legend=False,
            x_min=15.0,
            x_max=50.0,
            inset_enabled=True,
            inset_x_min=25.0,
            inset_x_max=35.0,
        ),
    )

    with pytest.raises(ValueError, match="direct-label leaders and main-panel"):
        render_project(state)


def test_close_peak_annotations_are_automatically_staggered():
    state = ProjectState(
        spectra=[
            SpectrumLayer(
                name="sample",
                x=[20.0, 30.0, 40.0],
                y=[1.0, 3.0, 1.0],
            )
        ],
        annotations=[
            PlotAnnotation(x=30.0, text="alpha peak", y_fraction=0.93),
            PlotAnnotation(x=30.12, text="beta peak", y_fraction=0.93),
        ],
        settings=PlotSettings(show_legend=False, show_phase_legend=False),
    )

    figure, axes = render_project(state)
    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    renderer = canvas.get_renderer()
    labels = [item[1] for item in axes["annotations"]]

    assert labels[0].get_position()[1] != labels[1].get_position()[1]
    _assert_no_overlap(labels, renderer)


@pytest.mark.parametrize("preview_size", [None, (900, 600)])
def test_impossible_peak_annotation_density_fails_before_export(preview_size):
    state = ProjectState(
        spectra=[
            SpectrumLayer(
                name="sample",
                x=[20.0, 30.0, 40.0],
                y=[1.0, 3.0, 1.0],
            )
        ],
        annotations=[
            PlotAnnotation(
                x=30.0 + index * 0.02,
                text=f"very long neighboring annotation {index}",
                y_fraction=0.93,
            )
            for index in range(6)
        ],
        settings=PlotSettings(show_legend=False, show_phase_legend=False),
    )

    with pytest.raises(ValueError, match="do not fit without overlap"):
        render_project(state, preview_size=preview_size)


def test_long_title_wraps_above_data_and_remains_inside_fixed_canvas():
    state = ProjectState(
        spectra=[
            SpectrumLayer(
                name="sample",
                x=[20.0, 30.0, 40.0],
                y=[1.0, 3.0, 1.0],
            )
        ],
        settings=PlotSettings(
            panel_title=(
                "Thermal evolution and reversible phase transformation under "
                "high-temperature cycling"
            )
        ),
    )

    figure, axes = render_project(state)
    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    title = axes["panel_title"]

    assert "\n" in title.get_text()
    _assert_inside(figure.bbox, title.get_window_extent(canvas.get_renderer()))
    assert title.get_position()[1] >= 1.0


def test_small_multiples_suptitle_survives_staged_gui_render():
    state = ProjectState(
        spectra=[
            SpectrumLayer(
                name=f"condition {index}",
                x=[20.0, 30.0, 40.0],
                y=[0.1, 1.0, 0.2],
            )
            for index in range(2)
        ],
        settings=PlotSettings(
            view_mode="small_multiples",
            panel_title="Condition-specific diffraction profiles",
        ),
    )
    live_figure = Figure()

    figure, axes = render_project(state, figure=live_figure, preview_size=(900, 600))
    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    title = axes["panel_title"]

    assert figure is live_figure
    assert title.figure is live_figure
    assert live_figure._suptitle is title
    assert title in live_figure.texts
    assert title.get_transform() is live_figure.transSubfigure
    _assert_inside(figure.bbox, title.get_window_extent(canvas.get_renderer()))


def test_staged_gui_render_preserves_limits_and_artist_ownership():
    state = ProjectState(
        spectra=[
            SpectrumLayer(
                name="sample",
                x=[20.0, 30.0, 40.0],
                y=[0.1, 1.0, 0.2],
            )
        ],
        settings=PlotSettings(show_legend=False, show_phase_legend=False),
    )
    live_figure = Figure()

    figure, axes = render_project(
        state,
        figure=live_figure,
        preview_size=(900, 600),
    )
    main_ax = axes["main"]

    assert figure is live_figure
    assert main_ax.figure is live_figure
    assert main_ax.get_xlim()[0] < 20.0 < 40.0 < main_ax.get_xlim()[1]
    assert all(
        getattr(artist, "figure", live_figure) is live_figure
        for artist in main_ax.findobj()
    )


def test_small_multiples_annotation_cannot_cover_panel_header():
    state = ProjectState(
        spectra=[
            SpectrumLayer(
                name="Long condition label at the panel edge",
                x=[20.0, 30.0, 40.0],
                y=[0.1, 1.0, 0.2],
            )
        ],
        annotations=[
            PlotAnnotation(
                x=39.5,
                text="edge peak",
                y_fraction=0.94,
            )
        ],
        settings=PlotSettings(view_mode="small_multiples"),
    )

    with pytest.raises(ValueError, match="small-multiple panel 1 decorations"):
        render_project(state)


def test_phase_rows_expand_band_and_do_not_overlap():
    state = ProjectState(
        spectra=[
            SpectrumLayer(
                name="sample",
                x=[20.0, 30.0, 40.0],
                y=[1.0, 3.0, 1.0],
            )
        ],
        phases=[
            PhaseLayer(
                name=f"Phase {index}",
                phase=f"Phase {index}",
                source_path=f"phase-{index}.cif",
                peaks=[PhasePeak(22.0 + index * 3.0, 100.0, "111")],
            )
            for index in range(4)
        ],
        settings=PlotSettings(show_legend=False, show_phase_legend=False),
    )

    figure, axes = render_project(state)
    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    labels = axes["phase_row_labels"]

    assert len(labels) == 4
    _assert_no_overlap(labels, canvas.get_renderer())


def test_phase_rows_use_a_separate_lane_from_visible_y_ticks():
    state = ProjectState(
        spectra=[
            SpectrumLayer(
                name="sample",
                x=[20.0, 30.0, 40.0],
                y=[0.1, 1.0, 0.2],
            )
        ],
        phases=[
            PhaseLayer(
                name=("Minor phase" if index == 2 else f"P{index}"),
                phase=("Minor phase" if index == 2 else f"P{index}"),
                source_path=f"phase-{index}.cif",
                peaks=[PhasePeak(22.0 + index * 4.0, 100.0, "111")],
            )
            for index in range(3)
        ],
        settings=PlotSettings(
            show_legend=False,
            show_phase_legend=False,
            show_y_tick_labels=True,
        ),
    )

    figure, axes = render_project(state)
    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    renderer = canvas.get_renderer()
    labels = axes["phase_row_labels"]
    y_ticks = [
        label
        for label in axes["main"].get_yticklabels()
        if label.get_visible() and label.get_text()
    ]

    assert labels and y_ticks
    _assert_no_overlap([*labels, *y_ticks], renderer)
    bragg_guard = axes["bragg_band_guard"]
    data_baseline = bragg_guard.get_y() + bragg_guard.get_height()
    assert all(float(value) >= data_baseline - 1.0e-12 for value in axes["main"].get_yticks())
    for label in labels:
        _assert_inside(figure.bbox, _box(label, renderer))
        gap_points = (
            axes["main"].bbox.x0 - _box(label, renderer).x1
        ) * 72.0 / figure.dpi
        assert gap_points == pytest.approx(3.0, abs=0.25)


def test_titled_direct_label_overlay_fits_three_named_phase_rows_and_y_ticks():
    state = ProjectState(
        spectra=[
            SpectrumLayer(
                name=name,
                x=[20.0, 31.7, 44.6, 65.1, 82.0],
                y=[0.04, 1.0 - index * 0.04, 0.68, 0.46, 0.04 + index * 0.002],
                order=index,
            )
            for index, name in enumerate(
                (
                    "Hydrogen-free reference",
                    "Hydrogen-charged condition",
                    "Recovered after unloading",
                )
            )
        ],
        phases=[
            PhaseLayer(
                name=name,
                phase=name,
                source_path=f"phase-{index}.cif",
                peaks=[PhasePeak(31.7 + index * 5.0, 100.0, "111")],
            )
            for index, name in enumerate(
                ("B2 matrix", "FCC secondary", "Laves minor")
            )
        ],
        settings=PlotSettings(
            panel_title=(
                "Hydrogen-dependent diffraction response under tensile loading"
            ),
            direct_labels=True,
            show_legend=False,
            show_phase_legend=False,
            show_y_tick_labels=True,
            x_min=20.0,
            x_max=82.0,
        ),
    )

    figure, axes = render_project(state)
    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    renderer = canvas.get_renderer()
    phase_labels = axes["phase_row_labels"]
    y_ticks = [
        label
        for label in axes["main"].get_yticklabels()
        if label.get_visible() and label.get_text()
    ]

    assert len(phase_labels) == 3
    _assert_no_overlap(phase_labels, renderer)
    _assert_no_overlap([*phase_labels, *y_ticks], renderer)
    for label in phase_labels:
        _assert_inside(figure.bbox, _box(label, renderer))


def test_lower_legend_cannot_cover_bragg_phase_band():
    state = ProjectState(
        spectra=[
            SpectrumLayer(
                name="sample",
                x=[20.0, 30.0, 40.0],
                y=[0.1, 1.0, 0.2],
            )
        ],
        phases=[
            PhaseLayer(
                name="phase",
                phase="phase",
                source_path="phase.cif",
                peaks=[PhasePeak(30.0, 100.0, "111")],
            )
        ],
        settings=PlotSettings(
            legend_location="lower left",
            show_phase_legend=False,
        ),
    )

    with pytest.raises(ValueError, match="legend and Bragg phase band"):
        render_project(state)


def test_visible_tick_labels_never_extend_beyond_exact_canvas():
    state = ProjectState(
        spectra=[
            SpectrumLayer(
                name="sample",
                x=[20.0, 30.0],
                y=[1.0, 2.0],
            )
        ],
        settings=PlotSettings(
            figure_width_in=1.8,
            figure_height_in=1.2,
            dpi=150,
            show_legend=False,
            show_phase_legend=False,
        ),
    )

    figure, axes = render_project(state)
    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    renderer = canvas.get_renderer()
    labels = [
        label
        for label in (*axes["main"].get_xticklabels(), *axes["main"].get_yticklabels())
        if label.get_visible() and label.get_text()
    ]

    assert labels
    for label in labels:
        _assert_inside(figure.bbox, label.get_window_extent(renderer))


def test_impossible_axis_label_length_fails_closed():
    from xrdviz.analysis import DerivedPlot

    state = ProjectState(
        derived_plot=DerivedPlot(
            kind="williamson_hall",
            x=[0.1, 0.2, 0.3],
            y=[0.01, 0.02, 0.03],
            labels={
                "x": "4 sin(theta)",
                "y": "Extremely long vertical description " * 8,
            },
        ),
        settings=PlotSettings(view_mode="derived"),
    )

    with pytest.raises(ValueError, match="fixed figure canvas"):
        render_project(state)
