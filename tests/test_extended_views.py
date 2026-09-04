from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xrdviz.analysis import DerivedPlot
from xrdviz.compliance import nature_compliance_issues
from xrdviz.fit import PatternFit
from xrdviz.maps import MapData
from xrdviz.models import PlotAnnotation, PlotSettings, ProjectState, SpectrumLayer
from xrdviz.plot.renderer import render_project
from xrdviz.project import load_project, save_project
from xrdviz.publication import export_publication_bundle


class ExtendedProjectPersistenceTests(unittest.TestCase):
    def test_map_derived_plot_and_annotation_round_trip(self) -> None:
        state = ProjectState(
            map_data=MapData(
                kind="rsm",
                x=[0.0, 1.0],
                y=[2.0, 3.0],
                intensity=[[1.0, 2.0], [3.0, 4.0]],
                labels={"x": "q_parallel", "y": "q_perp", "intensity": "I"},
                units={"x": "A^-1", "y": "A^-1", "intensity": "a.u."},
                source_path="rsm.csv",
            ),
            derived_plot=DerivedPlot(
                kind="williamson_hall",
                x=[0.1, 0.2],
                y=[0.01, 0.02],
                scatter=[(0.1, 0.01), (0.2, 0.02)],
                fit_line=[(0.1, 0.01), (0.2, 0.02)],
                labels={"x": "4 sin(theta)", "y": "beta cos(theta)"},
                metrics={"r_squared": 1.0},
            ),
            annotations=[PlotAnnotation(x=0.15, text="guide")],
            settings=PlotSettings(view_mode="derived"),
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "extended.xrdviz.json"
            save_project(state, path)
            loaded = load_project(path)

        self.assertIsNotNone(loaded.map_data)
        self.assertEqual(loaded.map_data.kind, "rsm")
        self.assertEqual(loaded.map_data.intensity.tolist(), [[1.0, 2.0], [3.0, 4.0]])
        self.assertIsNotNone(loaded.derived_plot)
        self.assertEqual(loaded.derived_plot.kind, "williamson_hall")
        self.assertEqual(loaded.annotations[0].text, "guide")


class ExtendedRendererTests(unittest.TestCase):
    def test_preview_keeps_project_aspect_ratio_and_fits_requested_box(self) -> None:
        state = ProjectState(
            spectra=[SpectrumLayer(name="A", x=[20.0, 21.0], y=[1.0, 2.0])],
            settings=PlotSettings(
                figure_width_in=4.0,
                figure_height_in=2.0,
                dpi=600,
            ),
        )

        figure, _axes = render_project(state, preview_size=(1000, 400))
        self.addCleanup(figure.clear)

        width, height = figure.get_size_inches()
        self.assertAlmostEqual(width, 4.0, places=6)
        self.assertAlmostEqual(height, 2.0, places=6)
        self.assertAlmostEqual(width / height, 2.0, places=6)
        self.assertLessEqual(round(width * figure.dpi), 1000)
        self.assertLessEqual(round(height * figure.dpi), 400)

    def test_renderer_keeps_existing_figure_when_view_validation_fails(self) -> None:
        state = ProjectState(
            spectra=[SpectrumLayer(name="A", x=[20.0, 21.0], y=[1.0, 2.0])],
            settings=PlotSettings(view_mode="overlay"),
        )
        figure, axes = render_project(state)
        old_axis = axes["main"]
        old_size = tuple(figure.get_size_inches())
        state.settings = PlotSettings(view_mode="map")

        with self.assertRaisesRegex(ValueError, "Map view requires"):
            render_project(state, figure)

        self.assertIs(figure.axes[0], old_axis)
        self.assertEqual(tuple(figure.get_size_inches()), old_size)

    def test_refinement_uncertainty_band_and_bars_are_rendered(self) -> None:
        fit = PatternFit(
            name="fit",
            x=[20.0, 21.0, 22.0],
            observed=[10.0, 12.0, 11.0],
            calculated=[9.0, 11.0, 12.0],
            sigma=[0.5, 0.75, 0.6],
        )
        for mode in ("band", "bars"):
            with self.subTest(mode=mode):
                state = ProjectState(
                    fit=fit,
                    settings=PlotSettings(
                        view_mode="refinement",
                        uncertainty_mode=mode,
                    ),
                )
                figure, axes = render_project(state)
                self.addCleanup(figure.clear)
                self.assertTrue(axes["uncertainty"])
                if mode == "band":
                    self.assertEqual(len(axes["uncertainty"][0].get_paths()), 1)
                else:
                    self.assertEqual(len(axes["uncertainty"][0].lines), 3)

    def test_renderer_refinement_error_does_not_clear_existing_figure(self) -> None:
        state = ProjectState(
            spectra=[SpectrumLayer(name="A", x=[20.0, 21.0], y=[1.0, 2.0])],
            settings=PlotSettings(view_mode="overlay"),
        )
        figure, axes = render_project(state)
        old_axis = axes["main"]
        state.settings = PlotSettings(view_mode="refinement")

        with self.assertRaisesRegex(ValueError, "Refinement view requires"):
            render_project(state, figure)

        self.assertIs(figure.axes[0], old_axis)

    def test_layer_wavelength_controls_axis_conversion(self) -> None:
        state = ProjectState(
            spectra=[
                SpectrumLayer(
                    name="A",
                    x=[1.0, 2.0],
                    y=[1.0, 2.0],
                    axis_kind="d",
                    wavelength_angstrom=1.0,
                )
            ],
            settings=PlotSettings(x_axis="two_theta", energy_kev=8.0478),
        )

        figure, axes = render_project(state)
        self.addCleanup(figure.clear)

        # A 1-A wavelength corresponds to 12.3984 keV; using the project
        # default energy would produce a visibly different 2-theta position.
        expected = 2.0 * math.degrees(math.asin(1.0 / (2.0 * 2.0)))
        self.assertAlmostEqual(float(axes["main"].lines[0].get_xdata()[0]), 60.0, places=5)
        self.assertAlmostEqual(float(axes["main"].lines[0].get_xdata()[1]), expected, places=5)

    def test_refinement_wavelength_controls_axis_conversion(self) -> None:
        state = ProjectState(
            fit=PatternFit(
                name="detector fit",
                x=[1.0, 2.0, 3.0],
                observed=[3.0, 2.0, 1.0],
                calculated=[3.0, 2.0, 1.0],
                axis_kind="d",
                wavelength_angstrom=1.0,
            ),
            settings=PlotSettings(
                view_mode="refinement",
                x_axis="two_theta",
                energy_kev=8.0478,
            ),
        )

        figure, axes = render_project(state)
        self.addCleanup(figure.clear)

        expected = [
            2.0 * math.degrees(math.asin(1.0 / (2.0 * d_spacing)))
            for d_spacing in (1.0, 2.0, 3.0)
        ]
        actual = [float(value) for value in axes["main"].lines[0].get_xdata()]
        for value, expected_value in zip(actual, expected):
            self.assertAlmostEqual(value, expected_value, places=5)

    def test_rsm_map_renders_quantitative_mesh_and_colorbar(self) -> None:
        state = ProjectState(
            map_data=MapData(
                kind="rsm",
                x=[0.0, 1.0],
                y=[2.0, 3.0],
                intensity=[[1.0, 2.0], [3.0, 4.0]],
                labels={"x": "q_parallel", "y": "q_perp", "intensity": "Intensity"},
                units={"x": "A^-1", "y": "A^-1", "intensity": "a.u."},
            ),
            settings=PlotSettings(view_mode="map", show_colorbar=True),
        )

        figure, axes = render_project(state)
        self.addCleanup(figure.clear)

        self.assertIn("map", axes)
        self.assertIn("colorbar", axes)
        self.assertIn("q_parallel", axes["main"].get_xlabel())
        self.assertIn("q_perp", axes["main"].get_ylabel())
        self.assertIn("range 1 to 4", axes["text_alternative"])
        self.assertNotIn(
            "at least one visible spectrum line with data is required",
            nature_compliance_issues(state),
        )

    def test_map_renderer_masks_empty_count_bins_and_rejects_an_empty_map(self) -> None:
        populated = ProjectState(
            map_data=MapData(
                kind="cake",
                x=[10.0, 20.0],
                y=[-5.0, 5.0],
                intensity=[[0.0, 2.0], [3.0, 0.0]],
                counts=[[0.0, 2.0], [1.0, 0.0]],
            ),
            settings=PlotSettings(view_mode="map"),
        )
        figure, axes = render_project(populated)
        self.addCleanup(figure.clear)
        mask = np.ma.getmaskarray(axes["map"].get_array())
        self.assertEqual(int(mask.sum()), 2)

        empty = ProjectState(
            map_data=MapData(
                kind="cake",
                x=[10.0, 20.0],
                y=[-5.0, 5.0],
                intensity=[[0.0, 0.0], [0.0, 0.0]],
                counts=[[0.0, 0.0], [0.0, 0.0]],
            ),
            settings=PlotSettings(view_mode="map"),
        )
        with self.assertRaisesRegex(ValueError, "finite populated"):
            render_project(empty)

    def test_narrow_map_keeps_colorbar_decorations_inside_canvas_and_limits_decimal_ticks(
        self,
    ) -> None:
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        state = ProjectState(
            map_data=MapData(
                kind="rsm",
                x=[0.000, 0.002, 0.004, 0.006, 0.008, 0.010],
                y=[0.000, 0.002, 0.004, 0.006, 0.008, 0.010],
                intensity=[[1.0, 2.0, 3.0, 4.0, 3.0, 2.0]] * 6,
                labels={"x": "q_parallel", "y": "q_perp", "intensity": "Intensity"},
                units={"x": "A^-1", "y": "A^-1", "intensity": "a.u."},
            ),
            settings=PlotSettings(
                view_mode="map",
                show_colorbar=True,
                figure_width_in=89.0 / 25.4,
                figure_height_in=2.35,
                dpi=600,
            ),
        )

        figure, axes = render_project(state)
        self.addCleanup(figure.clear)
        canvas = FigureCanvasAgg(figure)
        canvas.draw()
        renderer = canvas.get_renderer()
        colorbar = axes["colorbar"]
        decorations = [colorbar.yaxis.label, *colorbar.get_yticklabels()]

        self.assertTrue(
            all(
                item.get_window_extent(renderer).x1 <= figure.bbox.x1
                for item in decorations
            )
        )
        self.assertLessEqual(len(axes["main"].get_xticks()), 6)
        self.assertLessEqual(len(axes["main"].get_yticks()), 6)

    def test_pole_figure_uses_polar_axis(self) -> None:
        state = ProjectState(
            map_data=MapData(
                kind="pole_figure",
                x=[0.0, 90.0, 180.0],
                y=[0.0, 45.0],
                intensity=[[1.0, 2.0, 1.0], [2.0, 4.0, 2.0]],
                labels={"x": "phi", "y": "chi", "intensity": "m.r.d."},
                units={"x": "deg", "y": "deg", "intensity": "m.r.d."},
            ),
            settings=PlotSettings(view_mode="map", show_colorbar=True),
        )

        figure, axes = render_project(state)
        self.addCleanup(figure.clear)

        self.assertEqual(axes["main"].name, "polar")
        self.assertIn("map", axes)

    def test_derived_plot_renders_scatter_fit_and_metrics(self) -> None:
        state = ProjectState(
            derived_plot=DerivedPlot(
                kind="williamson_hall",
                x=[0.1, 0.2, 0.3],
                y=[0.01, 0.02, 0.03],
                scatter=[(0.1, 0.01), (0.2, 0.02), (0.3, 0.03)],
                fit_line=[(0.1, 0.01), (0.3, 0.03)],
                labels={"x": "4 sin(theta)", "y": "beta cos(theta) (rad)"},
                metrics={"r_squared": 0.999, "microstrain": 0.0012},
            ),
            settings=PlotSettings(view_mode="derived"),
        )

        figure, axes = render_project(state)
        self.addCleanup(figure.clear)

        self.assertIn("scatter", axes)
        self.assertIn("fit_line", axes)
        self.assertTrue(
            any("r squared" in text.get_text() for text in axes["main"].texts)
        )
        self.assertNotIn(
            "at least one visible spectrum line with data is required",
            nature_compliance_issues(state),
        )

    def test_derived_plot_uses_readable_metric_box_and_compact_decimal_ticks(
        self,
    ) -> None:
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        state = ProjectState(
            derived_plot=DerivedPlot(
                kind="williamson_hall",
                x=[0.001, 0.002, 0.003, 0.004, 0.005, 0.006],
                y=[0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006],
                scatter=[
                    (0.001, 0.0001),
                    (0.002, 0.0002),
                    (0.003, 0.0003),
                    (0.004, 0.0004),
                    (0.005, 0.0005),
                    (0.006, 0.0006),
                ],
                fit_line=[(0.001, 0.0001), (0.006, 0.0006)],
                labels={"x": "4 sin(theta)", "y": "beta cos(theta) (rad)"},
                metrics={"r_squared": 0.999, "microstrain": 0.0012},
            ),
            settings=PlotSettings(
                view_mode="derived",
                figure_width_in=89.0 / 25.4,
                figure_height_in=2.35,
                dpi=600,
            ),
        )

        figure, axes = render_project(state)
        self.addCleanup(figure.clear)
        canvas = FigureCanvasAgg(figure)
        canvas.draw()
        metric_text = next(
            text for text in axes["main"].texts if "r squared" in text.get_text()
        )

        self.assertEqual(metric_text.get_position(), (0.03, 0.96))
        self.assertNotIn("_", metric_text.get_text())
        self.assertEqual(metric_text.get_ha(), "left")
        self.assertAlmostEqual(metric_text.get_bbox_patch().get_facecolor()[0], 1.0)
        self.assertLessEqual(len(axes["main"].get_xticks()), 6)
        self.assertLessEqual(len(axes["main"].get_yticks()), 6)

    def test_small_multiples_inset_and_annotations_are_explicit_axes(self) -> None:
        spectra = [
            SpectrumLayer(name="A", x=[20.0, 21.0, 22.0], y=[1.0, 3.0, 1.0]),
            SpectrumLayer(name="B", x=[20.0, 21.0, 22.0], y=[2.0, 4.0, 2.0]),
        ]
        small_state = ProjectState(
            spectra=spectra,
            annotations=[PlotAnnotation(x=21.0, text="peak")],
            settings=PlotSettings(
                view_mode="small_multiples", small_multiples_columns=2
            ),
        )
        figure, axes = render_project(small_state)
        self.addCleanup(figure.clear)
        self.assertEqual(len(axes["panels"]), 2)
        self.assertEqual(len(axes["annotations"]), 2)

        inset_state = ProjectState(
            spectra=spectra,
            annotations=[PlotAnnotation(x=21.0, text="peak")],
            settings=PlotSettings(
                view_mode="overlay",
                inset_enabled=True,
                inset_x_min=20.5,
                inset_x_max=21.5,
            ),
        )
        inset_figure, inset_axes = render_project(inset_state)
        self.addCleanup(inset_figure.clear)
        self.assertIn("inset", inset_axes)
        self.assertIn("annotations", inset_axes)


class ExtendedPublicationTests(unittest.TestCase):
    def test_map_and_derived_views_export_source_data_and_manifest_hashes(self) -> None:
        cases = (
            ProjectState(
                map_data=MapData(
                    kind="rsm",
                    x=[0.0, 1.0],
                    y=[0.0, 1.0],
                    intensity=[[1.0, 2.0], [3.0, 4.0]],
                ),
                settings=PlotSettings(view_mode="map", show_colorbar=True),
            ),
            ProjectState(
                derived_plot=DerivedPlot(
                    kind="rocking_curve",
                    x=[-1.0, 0.0, 1.0],
                    y=[0.0, 1.0, 0.0],
                    scatter=[(-1.0, 0.0), (0.0, 1.0), (1.0, 0.0)],
                    labels={"x": "omega (deg)", "y": "Intensity (a.u.)"},
                    metrics={"fwhm": 1.0},
                ),
                settings=PlotSettings(view_mode="derived"),
            ),
        )

        for state in cases:
            with (
                self.subTest(view_mode=state.settings.view_mode),
                tempfile.TemporaryDirectory() as tmp,
            ):
                outputs = export_publication_bundle(state, tmp)
                manifest = json.loads(outputs.manifest.read_text(encoding="utf-8"))
                kinds = {entry["kind"] for entry in manifest["artifacts"]}
                if state.settings.view_mode == "map":
                    self.assertIsNotNone(outputs.map_data)
                    self.assertIn("map_csv", kinds)
                else:
                    self.assertIsNotNone(outputs.derived_data)
                    self.assertIn("derived_csv", kinds)


if __name__ == "__main__":
    unittest.main()
