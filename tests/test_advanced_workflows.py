from __future__ import annotations

import sys
import tempfile
import unittest
import json
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xrdviz.fit import FitComponent, PatternFit
from xrdviz.compliance import nature_compliance_issues
from xrdviz.models import PlotSettings, ProjectState, SpectrumLayer
from xrdviz.plot.renderer import render_project
from xrdviz.project import load_project, save_project
from xrdviz.publication import export_publication_bundle


class AdvancedProjectWorkflowTests(unittest.TestCase):
    def test_uncertainty_and_pattern_fit_round_trip_through_project_json(self):
        fit = PatternFit(
            name="refined sample",
            x=[20.0, 21.0, 22.0],
            observed=[10.0, 16.0, 11.0],
            calculated=[9.5, 15.5, 11.5],
            sigma=[1.0, 1.0, 2.0],
            background=[2.0, 2.0, 2.0],
            components=[
                FitComponent(name="peak 1", y=[7.5, 13.5, 9.5], color="#2B9C8F")
            ],
            source_path="fit.csv",
        )
        state = ProjectState(
            spectra=[
                SpectrumLayer(
                    name="measured",
                    x=[20.0, 21.0, 22.0],
                    y=[10.0, 16.0, 11.0],
                    y_error=[1.0, 1.0, 2.0],
                )
            ],
            fit=fit,
            settings=PlotSettings(
                view_mode="refinement", normalize=False, uncertainty_mode="band"
            ),
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "advanced.xrdviz.json"
            save_project(state, path)
            loaded = load_project(path)

        self.assertEqual(loaded.spectra[0].y_error, [1.0, 1.0, 2.0])
        self.assertEqual(loaded.settings.uncertainty_mode, "band")
        self.assertIsNotNone(loaded.fit)
        assert loaded.fit is not None
        self.assertEqual(loaded.fit.name, "refined sample")
        self.assertEqual(loaded.fit.components[0].name, "peak 1")
        self.assertEqual(loaded.fit.difference, [0.5, 0.5, -0.5])

    def test_refinement_view_has_observed_calculated_and_residual_panels(self):
        state = ProjectState(
            fit=PatternFit(
                name="fit",
                x=[20.0, 21.0, 22.0, 23.0],
                observed=[5.0, 12.0, 8.0, 4.0],
                calculated=[4.5, 11.0, 8.5, 4.0],
                background=[1.0, 1.0, 1.0, 1.0],
                components=[FitComponent(name="component A", y=[3.5, 10.0, 7.5, 3.0])],
            ),
            settings=PlotSettings(
                view_mode="refinement", normalize=False, show_fit_components=True
            ),
        )

        figure, axes = render_project(state)
        self.addCleanup(figure.clear)

        self.assertIn("main", axes)
        self.assertIn("residual", axes)
        main_labels = {line.get_label() for line in axes["main"].lines}
        residual_labels = {line.get_label() for line in axes["residual"].lines}
        self.assertIn("Observed", main_labels)
        self.assertIn("Calculated", main_labels)
        self.assertIn("Background", main_labels)
        self.assertIn("component A", main_labels)
        self.assertIn("Difference", residual_labels)
        self.assertEqual(axes["main"].get_xlabel(), "")
        self.assertTrue(axes["residual"].get_xlabel())
        metric_text = next(
            text for text in axes["main"].texts if "$R_p$" in text.get_text()
        )
        self.assertLess(metric_text.get_position()[0], 0.5)
        self.assertEqual(metric_text.get_horizontalalignment(), "left")

    def test_dense_refinement_thins_only_display_markers_and_reports_the_rule(self):
        point_count = 640
        x = [25.0 + index * 0.08 for index in range(point_count)]
        calculated = [
            50.0 + 900.0 * math.exp(-0.5 * ((value - 44.0) / 0.7) ** 2)
            for value in x
        ]
        observed = [
            value + 4.0 * math.sin(index * 0.31)
            for index, value in enumerate(calculated)
        ]
        state = ProjectState(
            fit=PatternFit(
                name="dense fit",
                x=x,
                observed=observed,
                calculated=calculated,
            ),
            settings=PlotSettings(
                view_mode="refinement",
                normalize=False,
                show_fit_components=False,
                show_fit_background=False,
                show_fit_metrics=False,
            ),
        )

        _figure, axes = render_project(state)
        observed_line = next(
            line for line in axes["main"].lines if line.get_label() == "Observed"
        )
        calculated_line = next(
            line for line in axes["main"].lines if line.get_label() == "Calculated"
        )
        difference_line = next(
            line for line in axes["residual"].lines if line.get_label() == "Difference"
        )

        self.assertLess(len(observed_line.get_xdata()), point_count)
        self.assertLessEqual(len(observed_line.get_xdata()), 50)
        self.assertGreaterEqual(len(observed_line.get_xdata()), 2)
        self.assertEqual(len(calculated_line.get_xdata()), point_count)
        self.assertEqual(len(difference_line.get_xdata()), point_count)
        self.assertEqual(axes["observed_marker_indices"][0], 0)
        self.assertEqual(axes["observed_marker_indices"][-1], point_count - 1)
        self.assertIn("display markers", axes["text_alternative"])
        self.assertIn(f"of {point_count}", axes["text_alternative"])

    def test_nonuniform_refinement_markers_are_spaced_in_display_coordinates(self):
        clustered_x = [25.0 + index * 0.001 for index in range(500)]
        spread_x = [25.5 + index * (50.5 / 139.0) for index in range(140)]
        x = [*clustered_x, *spread_x]
        calculated = [
            50.0 + 900.0 * math.exp(-0.5 * ((value - 44.0) / 0.7) ** 2)
            for value in x
        ]
        state = ProjectState(
            fit=PatternFit(
                name="nonuniform fit",
                x=x,
                observed=[
                    value + 4.0 * math.sin(index * 0.31)
                    for index, value in enumerate(calculated)
                ],
                calculated=calculated,
            ),
            settings=PlotSettings(
                view_mode="refinement",
                normalize=False,
                show_fit_components=False,
                show_fit_background=False,
                show_fit_metrics=False,
            ),
        )

        figure, axes = render_project(state)
        observed_line = next(
            line for line in axes["main"].lines if line.get_label() == "Observed"
        )
        display_x = axes["main"].transData.transform(
            [(float(value), 0.0) for value in observed_line.get_xdata()]
        )[:, 0]
        gaps_points = [
            abs(float(right - left)) * 72.0 / figure.dpi
            for left, right in zip(display_x, display_x[1:])
        ]

        self.assertGreaterEqual(min(gaps_points), 3.9)
        self.assertEqual(axes["observed_marker_indices"][0], 0)
        self.assertEqual(axes["observed_marker_indices"][-1], len(x) - 1)

    def test_overlay_view_can_draw_uncertainty_band(self):
        state = ProjectState(
            spectra=[
                SpectrumLayer(
                    name="sample",
                    x=[20.0, 21.0, 22.0],
                    y=[10.0, 12.0, 11.0],
                    y_error=[0.5, 1.0, 0.5],
                )
            ],
            settings=PlotSettings(normalize=False, uncertainty_mode="band"),
        )

        figure, axes = render_project(state)
        self.addCleanup(figure.clear)

        self.assertIn("uncertainty", axes)
        self.assertEqual(len(axes["uncertainty"]), 1)

    def test_refinement_data_satisfies_visible_data_preflight_without_spectrum_layers(
        self,
    ):
        state = ProjectState(
            fit=PatternFit(
                name="fit",
                x=[20.0, 21.0, 22.0],
                observed=[10.0, 12.0, 11.0],
                calculated=[9.5, 12.5, 11.0],
            ),
            settings=PlotSettings(view_mode="refinement", normalize=False),
        )

        issues = nature_compliance_issues(state)

        self.assertNotIn(
            "at least one visible spectrum line with data is required", issues
        )

    def test_publication_bundle_includes_fit_data_metrics_and_manifest_hash(self):
        state = ProjectState(
            fit=PatternFit(
                name="fit",
                x=[20.0, 21.0, 22.0],
                observed=[10.0, 12.0, 11.0],
                calculated=[9.5, 12.5, 11.0],
                sigma=[1.0, 1.0, 1.0],
                background=[2.0, 2.0, 2.0],
                components=[FitComponent(name="alpha", y=[7.5, 10.5, 9.0])],
                source_path="refinement.csv",
            ),
            settings=PlotSettings(view_mode="refinement", normalize=False),
        )

        with tempfile.TemporaryDirectory() as tmp:
            outputs = export_publication_bundle(state, tmp)
            self.assertIsNotNone(outputs.fit_data)
            assert outputs.fit_data is not None
            fit_csv = outputs.fit_data.read_text(encoding="utf-8")
            report = outputs.report.read_text(encoding="utf-8")
            manifest = json.loads(outputs.manifest.read_text(encoding="utf-8"))

        self.assertIn("difference", fit_csv)
        self.assertIn("component_alpha", fit_csv)
        self.assertIn("## Pattern fit", report)
        self.assertIn("Rwp", report)
        self.assertTrue(
            any(
                item["kind"] == "fit_csv" and item["sha256"]
                for item in manifest["artifacts"]
            )
        )


if __name__ == "__main__":
    unittest.main()
