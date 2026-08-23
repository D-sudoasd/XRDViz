import importlib.util
import re
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xrdviz.models import PhaseLayer, PhasePeak, PlotSettings, ProjectState, SpectrumLayer


@unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib is not installed")
class RendererTests(unittest.TestCase):
    def test_renderer_exports_non_empty_png(self):
        from xrdviz.plot.renderer import export_project

        state = ProjectState(
            spectra=[SpectrumLayer(name="s1", x=[20.0, 30.0, 40.0], y=[10.0, 100.0, 20.0], color="#0072B2")],
            phases=[PhaseLayer(name="p1", source_path="p.cif", color="#D55E00", peaks=[PhasePeak(30.0, 100.0, "111")])],
            settings=PlotSettings(x_label="2theta", y_label="Log intensity (a.u.)", log_scale=True),
        )

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "figure.png"
            export_project(state, output)
            size = output.stat().st_size

        self.assertGreater(size, 1000)

    def test_export_preserves_exact_pdf_media_box(self):
        from xrdviz.plot.renderer import export_project

        width_in, height_in = 2.5, 1.25
        state = ProjectState(
            spectra=[SpectrumLayer(name="s1", x=[20.0, 30.0], y=[1.0, 2.0])],
            settings=PlotSettings(
                figure_width_in=width_in,
                figure_height_in=height_in,
                show_legend=False,
                show_phase_legend=False,
                legend_location="none",
            ),
        )

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "figure.pdf"
            export_project(state, output)
            match = re.search(
                rb"/MediaBox\s*\[\s*0\s+0\s+([0-9.]+)\s+([0-9.]+)\s*\]",
                output.read_bytes(),
            )

        self.assertIsNotNone(match)
        assert match is not None
        self.assertAlmostEqual(float(match.group(1)) / 72.0, width_in, places=6)
        self.assertAlmostEqual(float(match.group(2)) / 72.0, height_in, places=6)

    def test_export_png_and_tiff_are_rgb_with_exact_pixel_size_and_dpi(self):
        from xrdviz.plot.renderer import export_project

        if importlib.util.find_spec("PIL") is None:
            self.skipTest("Pillow is not installed")
        from PIL import Image

        width_in, height_in, dpi = 1.8, 1.2, 150
        state = ProjectState(
            spectra=[SpectrumLayer(name="s1", x=[20.0, 30.0], y=[1.0, 2.0])],
            settings=PlotSettings(
                figure_width_in=width_in,
                figure_height_in=height_in,
                dpi=dpi,
                show_legend=False,
                show_phase_legend=False,
                legend_location="none",
            ),
        )

        with tempfile.TemporaryDirectory() as tmp:
            for suffix in (".png", ".tiff"):
                with self.subTest(suffix=suffix):
                    output = Path(tmp) / f"figure{suffix}"
                    export_project(state, output)
                    with Image.open(output) as image:
                        self.assertEqual(image.mode, "RGB")
                        self.assertEqual(image.size, (round(width_in * dpi), round(height_in * dpi)))
                        self.assertAlmostEqual(image.info["dpi"][0], dpi, delta=1.0)

    def test_reused_figure_syncs_size_and_dpi(self):
        from matplotlib.figure import Figure
        from xrdviz.plot.renderer import render_project

        state = ProjectState(
            spectra=[SpectrumLayer(name="s1", x=[20.0, 30.0], y=[1.0, 2.0])],
            settings=PlotSettings(figure_width_in=2.2, figure_height_in=1.4, dpi=123),
        )
        figure = Figure(figsize=(1.0, 1.0), dpi=72)

        rendered, _axes = render_project(state, figure)

        self.assertIs(rendered, figure)
        self.assertEqual(tuple(round(value, 6) for value in figure.get_size_inches()), (2.2, 1.4))
        self.assertEqual(figure.dpi, 123)

    def test_renderer_draws_reference_symbols_guides_direct_labels_and_phase_legend(self):
        from xrdviz.plot.renderer import render_project

        state = ProjectState(
            spectra=[
                SpectrumLayer(
                    name="Annealed",
                    x=[20.0, 30.0, 40.0],
                    y=[10.0, 100.0, 20.0],
                    color="#0072B2",
                    order=1,
                )
            ],
            phases=[
                PhaseLayer(
                    name="Calcite",
                    phase="Calcite",
                    source_path="reference_peaks.csv",
                    source_type="reference_csv",
                    color="#D55E00",
                    marker_shape="triangle",
                    show_guides=True,
                    label_policy="characteristic",
                    peaks=[
                        PhasePeak(30.0, 100.0, "104", label="Calcite main"),
                        PhasePeak(35.0, 40.0, "110", label="Calcite weak"),
                    ],
                )
            ],
            settings=PlotSettings(show_phase_legend=True, direct_labels=True),
        )

        _fig, axes = render_project(state)
        main_ax = axes["main"]

        main_text = " ".join(text.get_text() for text in main_ax.texts)
        legend_text = " ".join(text.get_text() for text in main_ax.get_legend().texts)

        self.assertIn("Annealed", main_text)
        self.assertIn("Calcite main", main_text)
        self.assertIn("Calcite", legend_text)
        self.assertGreaterEqual(len(main_ax.collections), 1)
        self.assertTrue(any(line.get_linestyle() == "--" for line in main_ax.lines))

    def test_renderer_uses_single_closed_panel_and_spectrum_x_range(self):
        from xrdviz.plot.renderer import render_project

        state = ProjectState(
            spectra=[
                SpectrumLayer(
                    name="Sample",
                    x=[20.0, 30.0, 40.0],
                    y=[10.0, 100.0, 20.0],
                    color="#0072B2",
                )
            ],
            phases=[
                PhaseLayer(
                    name="Phase A",
                    source_path="a.cif",
                    color="#D55E00",
                    peaks=[
                        PhasePeak(30.0, 100.0, "111"),
                        PhasePeak(115.0, 20.0, "333"),
                    ],
                )
            ],
            settings=PlotSettings(),
        )

        fig, axes = render_project(state)
        main_ax = axes["main"]

        self.assertEqual(fig.axes, [main_ax])
        self.assertIs(axes["bragg"], main_ax)
        self.assertTrue(all(spine.get_visible() for spine in main_ax.spines.values()))
        self.assertLess(main_ax.get_xlim()[1], 50.0)

    def test_manual_x_limits_override_auto_range(self):
        from xrdviz.plot.renderer import render_project

        state = ProjectState(
            spectra=[SpectrumLayer(name="Sample", x=[0.6, 1.0, 3.2], y=[10.0, 100.0, 20.0], axis_kind="d")],
            settings=PlotSettings(x_axis="d", x_min=0.9, x_max=3.0),
        )

        _fig, axes = render_project(state)

        self.assertEqual(tuple(round(value, 3) for value in axes["main"].get_xlim()), (0.9, 3.0))

    def test_y_tick_values_are_hidden_by_default_but_can_be_shown(self):
        from xrdviz.plot.renderer import render_project

        state = ProjectState(
            spectra=[SpectrumLayer(name="Sample", x=[20.0, 30.0, 40.0], y=[10.0, 100.0, 20.0])],
            settings=PlotSettings(),
        )

        _fig, axes = render_project(state)
        self.assertFalse(_has_visible_tick_text(axes["main"].get_yticklabels()))
        self.assertFalse(_has_visible_y_tick_marks(axes["main"]))

        state.settings.show_y_tick_labels = True
        _fig, axes = render_project(state)
        self.assertTrue(_has_visible_tick_text(axes["main"].get_yticklabels()))
        self.assertTrue(_has_visible_y_tick_marks(axes["main"]))

    def test_panel_title_is_drawn_inside_axes(self):
        from xrdviz.plot.renderer import render_project

        state = ProjectState(
            spectra=[SpectrumLayer(name="H-free", x=[0.9, 1.6, 3.0], y=[10.0, 100.0, 20.0], axis_kind="d")],
            settings=PlotSettings(x_axis="d", panel_title="Nb-free"),
        )

        _fig, axes = render_project(state)
        text = " ".join(item.get_text() for item in axes["main"].texts)

        self.assertIn("Nb-free", text)

    def test_phase_row_labels_are_outside_data_area(self):
        from xrdviz.plot.renderer import render_project

        state = ProjectState(
            spectra=[SpectrumLayer(name="Sample", x=[0.9, 1.6, 3.0], y=[10.0, 100.0, 20.0], axis_kind="d")],
            phases=[
                PhaseLayer(name="B2", phase="B2", source_path="b2.cif", color="#3154d4", peaks=[PhasePeak(1.2, 100.0, source_axis="d")]),
                PhaseLayer(name="FCC", phase="FCC", source_path="fcc.cif", color="#009E73", peaks=[PhasePeak(2.0, 100.0, source_axis="d")]),
            ],
            settings=PlotSettings(x_axis="d", x_min=0.9, x_max=3.0),
        )

        _fig, axes = render_project(state)
        phase_texts = [text for text in axes["main"].texts if text.get_text() in {"B2", "FCC"}]

        self.assertEqual(len(phase_texts), 2)
        self.assertTrue(all(text.get_position()[0] < 0.0 for text in phase_texts))
        self.assertTrue(all(not text.get_clip_on() for text in phase_texts))

    def test_gradient_stack_applies_show_every_n_and_colorbar(self):
        from xrdviz.plot.renderer import render_project

        state = ProjectState(
            spectra=[
                SpectrumLayer(
                    name=f"f{index}",
                    x=[10.0, 20.0, 30.0],
                    y=[1.0, 10.0 + index, 2.0],
                    frame_index=index,
                    color_value=float(index),
                    order=index,
                )
                for index in range(5)
            ],
            settings=PlotSettings(
                view_mode="gradient_stack",
                color_by="frame",
                colormap="viridis",
                show_colorbar=True,
                show_every_n=2,
                stack_spacing=0.2,
                show_legend=False,
            ),
        )

        fig, axes = render_project(state)

        self.assertIn("colorbar", axes)
        self.assertGreater(len(fig.axes), 1)
        self.assertEqual(len(axes["main"].lines), 3)

    def test_heatmap_view_renders_image_with_colorbar(self):
        from xrdviz.plot.renderer import render_project

        state = ProjectState(
            spectra=[
                SpectrumLayer(name="f1", x=[10.0, 20.0, 30.0], y=[1.0, 3.0, 2.0], frame_index=1, order=0),
                SpectrumLayer(name="f2", x=[10.0, 20.0, 30.0], y=[2.0, 4.0, 1.0], frame_index=2, order=1),
            ],
            settings=PlotSettings(view_mode="heatmap", heatmap_points=16, show_colorbar=True, x_min=10.0, x_max=30.0),
        )

        _fig, axes = render_project(state)

        self.assertIn("heatmap", axes)
        self.assertIn("colorbar", axes)
        self.assertEqual(axes["heatmap"].get_array().shape, (2, 16))

    def test_heatmap_shows_metadata_ticks_by_default_and_sparsifies_rows(self):
        from xrdviz.plot.renderer import render_project

        def make_state(row_count):
            return ProjectState(
                spectra=[
                    SpectrumLayer(
                        name=f"f{index}",
                        x=[10.0, 20.0, 30.0],
                        y=[1.0, 3.0, 2.0],
                        frame_index=100 + index,
                        order=index,
                    )
                    for index in range(row_count)
                ],
                settings=PlotSettings(
                    view_mode="heatmap",
                    heatmap_points=8,
                    show_y_tick_labels=False,
                    show_colorbar=True,
                    normalize=True,
                    log_scale=True,
                    x_min=10.0,
                    x_max=30.0,
                ),
            )

        _fig, axes = render_project(make_state(2))
        labels = axes["main"].get_yticklabels()
        self.assertEqual([label.get_text() for label in labels], ["100", "101"])
        self.assertTrue(all(label.get_visible() for label in labels))
        self.assertEqual(axes["colorbar"].get_ylabel(), "log10(normalized intensity)")

        _fig, axes = render_project(make_state(10))
        sparse_labels = [label.get_text() for label in axes["main"].get_yticklabels()]
        self.assertLessEqual(len(sparse_labels), 7)
        self.assertEqual(sparse_labels[0], "100")
        self.assertEqual(sparse_labels[-1], "109")

    def test_heatmap_colorbar_annotations_fit_inside_canvas(self):
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from xrdviz.plot.renderer import render_project

        state = ProjectState(
            spectra=[
                SpectrumLayer(
                    name=f"f{index}",
                    x=[20.0, 30.0, 40.0],
                    y=[1.0, 3.0, 2.0],
                    order=index,
                    temperature=25.0 + index * 50.0,
                    temperature_unit="C",
                )
                for index in range(8)
            ],
            settings=PlotSettings(
                view_mode="heatmap",
                sort_by="temperature",
                show_colorbar=True,
                figure_width_in=89.0 / 25.4,
                figure_height_in=3.0,
                x_min=20.0,
                x_max=40.0,
            ),
        )

        figure, axes = render_project(state)
        canvas = FigureCanvasAgg(figure)
        canvas.draw()
        renderer = canvas.get_renderer()
        colorbar = axes["colorbar"]
        annotations = [colorbar.yaxis.label, *colorbar.get_yticklabels()]

        self.assertTrue(all(item.get_window_extent(renderer).x1 <= figure.bbox.x1 for item in annotations))

    def test_outside_legend_with_long_labels_fits_inside_exact_canvas(self):
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from xrdviz.plot.renderer import render_project

        state = ProjectState(
            spectra=[
                SpectrumLayer(
                    name="Long unbroken sample identifier MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM",
                    x=[20.0, 30.0, 40.0],
                    y=[1.0, 3.0, 2.0],
                )
            ],
            settings=PlotSettings(legend_location="outside right"),
        )

        figure, axes = render_project(state)
        canvas = FigureCanvasAgg(figure)
        canvas.draw()
        legend = axes["main"].get_legend()

        self.assertIsNotNone(legend)
        assert legend is not None
        self.assertLessEqual(legend.get_window_extent(canvas.get_renderer()).x1, figure.bbox.x1)

    def test_outside_legend_fails_closed_before_vertical_clipping(self):
        from xrdviz.plot.renderer import render_project

        state = ProjectState(
            spectra=[
                SpectrumLayer(
                    name=f"Sample {index} with a very long publication label MMMMMMMMMMMMMMMMMMMMMMMMM",
                    x=[20.0, 30.0, 40.0],
                    y=[1.0, 3.0, 2.0],
                    order=index,
                )
                for index in range(10)
            ],
            settings=PlotSettings(legend_location="outside right"),
        )

        with self.assertRaisesRegex(ValueError, "does not fit vertically"):
            render_project(state)

    def test_gradient_colorbar_and_outside_legend_fail_closed(self):
        from xrdviz.plot.renderer import render_project

        state = ProjectState(
            spectra=[
                SpectrumLayer(name="Frame 1", x=[20.0, 30.0], y=[1.0, 2.0], frame_index=1),
                SpectrumLayer(name="Frame 2", x=[20.0, 30.0], y=[2.0, 1.0], frame_index=2),
            ],
            settings=PlotSettings(
                view_mode="gradient_stack",
                color_by="frame",
                show_colorbar=True,
                legend_location="outside right",
            ),
        )

        with self.assertRaisesRegex(ValueError, "cannot share"):
            render_project(state)

    def test_missing_temperature_is_gray_in_gradient_stack(self):
        from xrdviz.models import PLOT_MUTED_COLOR
        from xrdviz.plot.renderer import render_project

        state = ProjectState(
            spectra=[
                SpectrumLayer(name="known", x=[10.0, 20.0], y=[1.0, 2.0], temperature=300.0, order=0),
                SpectrumLayer(name="missing", x=[10.0, 20.0], y=[2.0, 1.0], temperature=None, order=1),
            ],
            settings=PlotSettings(
                view_mode="gradient_stack",
                color_by="temperature",
                show_legend=False,
                show_colorbar=True,
            ),
        )

        _fig, axes = render_project(state)

        self.assertEqual(axes["main"].lines[1].get_color(), PLOT_MUTED_COLOR)
        self.assertIn("colorbar", axes)

    def test_heatmap_uses_ordinal_rows_for_irregular_metadata_values(self):
        from xrdviz.plot.renderer import render_project

        state = ProjectState(
            spectra=[
                SpectrumLayer(name="t0", x=[10.0, 20.0, 30.0], y=[1.0, 3.0, 2.0], time_s=0.0, order=0),
                SpectrumLayer(name="t10", x=[10.0, 20.0, 30.0], y=[2.0, 4.0, 1.0], time_s=10.0, order=1),
                SpectrumLayer(name="t60", x=[10.0, 20.0, 30.0], y=[1.0, 2.0, 5.0], time_s=60.0, order=2),
            ],
            settings=PlotSettings(
                view_mode="heatmap",
                sort_by="time",
                color_by="time",
                heatmap_points=16,
                show_y_tick_labels=True,
                x_min=10.0,
                x_max=30.0,
            ),
        )

        _fig, axes = render_project(state)

        self.assertEqual(tuple(round(value, 3) for value in axes["main"].get_ylim()), (-0.5, 2.5))
        self.assertEqual([tick.get_text() for tick in axes["main"].get_yticklabels()], ["0", "10", "60"])

    def test_phase_marker_shape_is_drawn_and_synced_to_legend(self):
        from xrdviz.plot.renderer import render_project

        state = ProjectState(
            spectra=[SpectrumLayer(name="Sample", x=[20.0, 30.0], y=[1.0, 2.0])],
            phases=[
                PhaseLayer(
                    name="Phase A",
                    phase="Phase A",
                    source_path="a.cif",
                    marker_shape="diamond",
                    peaks=[PhasePeak(30.0, 100.0, "111")],
                )
            ],
            settings=PlotSettings(show_phase_legend=True),
        )

        _fig, axes = render_project(state)
        main_ax = axes["main"]
        marker_lines = [line for line in main_ax.lines if line.get_marker() == "D"]
        self.assertTrue(marker_lines)
        legend = main_ax.get_legend()
        self.assertIsNotNone(legend)
        assert legend is not None
        handles = legend.legend_handles if hasattr(legend, "legend_handles") else legend.legendHandles
        phase_handle = next(handle for handle in handles if handle.get_label() == "Phase A")
        self.assertEqual(phase_handle.get_marker(), "D")


def _has_visible_tick_text(labels):
    return any(label.get_visible() and label.get_text() for label in labels)


def _has_visible_y_tick_marks(ax):
    ticks = ax.yaxis.get_major_ticks()
    return any(tick.tick1line.get_visible() or tick.tick2line.get_visible() for tick in ticks)


if __name__ == "__main__":
    unittest.main()
