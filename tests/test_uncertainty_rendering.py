import importlib.util
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xrdviz.fit import PatternFit
from xrdviz.models import PlotSettings, ProjectState, SpectrumLayer
from xrdviz.plot.renderer import render_project


@unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib is not installed")
class UncertaintyRenderingTests(unittest.TestCase):
    @staticmethod
    def _bar_segments(container):
        return container.lines[2][0].get_segments()

    def test_spectrum_bars_sample_artist_inputs_before_rendering(self):
        state = ProjectState(
            spectra=[
                SpectrumLayer(
                    name="sample",
                    x=list(range(10)),
                    y=[1.0 + index for index in range(10)],
                    y_error=[0.1] * 10,
                )
            ],
            settings=PlotSettings(
                uncertainty_mode="bars",
                errorbar_stride=3,
                show_legend=False,
            ),
        )

        figure, axes = render_project(state)
        self.addCleanup(figure.clear)

        self.assertEqual(len(self._bar_segments(axes["uncertainty"][0])), 4)

    def test_refinement_bars_sample_artist_inputs_before_rendering(self):
        fit = PatternFit(
            name="fit",
            x=list(range(10)),
            observed=[1.0 + index for index in range(10)],
            calculated=[1.2 + index for index in range(10)],
            sigma=[0.1] * 10,
        )
        state = ProjectState(
            fit=fit,
            settings=PlotSettings(
                view_mode="refinement",
                uncertainty_mode="bars",
                errorbar_stride=4,
                show_legend=False,
            ),
        )

        figure, axes = render_project(state)
        self.addCleanup(figure.clear)

        self.assertEqual(len(self._bar_segments(axes["uncertainty"][0])), 3)

    def test_band_keeps_full_resolution(self):
        point_count = 10
        state = ProjectState(
            spectra=[
                SpectrumLayer(
                    name="sample",
                    x=list(range(point_count)),
                    y=[1.0 + index for index in range(point_count)],
                    y_error=[0.1] * point_count,
                )
            ],
            settings=PlotSettings(
                uncertainty_mode="band",
                errorbar_stride=3,
                show_legend=False,
            ),
        )

        figure, axes = render_project(state)
        self.addCleanup(figure.clear)
        path = axes["uncertainty"][0].get_paths()[0]

        # fill_between stores both bounds plus the closing vertices, so the
        # path has 2*N+3 vertices when all N input points are retained.
        self.assertEqual(len(path.vertices), 2 * point_count + 3)

    def test_bars_keep_normalized_log_uncertainty_bounds(self):
        state = ProjectState(
            spectra=[
                SpectrumLayer(
                    name="sample",
                    x=[20.0, 21.0, 22.0],
                    y=[2.0, 4.0, 8.0],
                    y_error=[0.5, 1.0, 2.0],
                )
            ],
            settings=PlotSettings(
                normalize=True,
                log_scale=True,
                log_epsilon=1e-9,
                uncertainty_mode="bars",
                errorbar_stride=2,
                show_legend=False,
            ),
        )

        figure, axes = render_project(state)
        self.addCleanup(figure.clear)
        first_segment = self._bar_segments(axes["uncertainty"][0])[0]
        observed = math.log10(2.0 / 8.0)
        lower = math.log10(1.5 / 8.0)
        upper = math.log10(2.5 / 8.0)

        self.assertAlmostEqual(float(first_segment[0, 1]), lower)
        self.assertAlmostEqual(float(first_segment[1, 1]), upper)
        self.assertAlmostEqual(float(axes["main"].lines[0].get_ydata()[0]), observed)


if __name__ == "__main__":
    unittest.main()
