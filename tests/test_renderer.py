import importlib.util
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


def _has_visible_tick_text(labels):
    return any(label.get_visible() and label.get_text() for label in labels)


def _has_visible_y_tick_marks(ax):
    ticks = ax.yaxis.get_major_ticks()
    return any(tick.tick1line.get_visible() or tick.tick2line.get_visible() for tick in ticks)


if __name__ == "__main__":
    unittest.main()
