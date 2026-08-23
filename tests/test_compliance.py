import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xrdviz.compliance import nature_compliance_issues
from xrdviz.models import PhaseLayer, PhasePeak, PlotSettings, ProjectState, SpectrumLayer


class NatureComplianceTests(unittest.TestCase):
    def test_nature_single_with_visible_spectrum_passes(self):
        state = ProjectState(
            spectra=[SpectrumLayer(name="sample", x=[1.0, 2.0], y=[2.0, 3.0])],
            settings=PlotSettings(
                figure_width_in=89.0 / 25.4,
                figure_height_in=60.0 / 25.4,
                template_name="nature_single",
                font_family="Helvetica",
                font_size=6.0,
                axis_label_size=7.0,
                tick_label_size=5.0,
                line_width=0.75,
            ),
        )

        self.assertEqual(nature_compliance_issues(state), [])

    def test_checks_report_configuration_and_visible_line_issues(self):
        state = ProjectState(
            spectra=[
                SpectrumLayer(
                    name="sample",
                    x=[1.0, 2.0],
                    y=[2.0, 3.0],
                    linewidth=1.5,
                )
            ],
            settings=PlotSettings(
                figure_width_in=100.0 / 25.4,
                figure_height_in=171.0 / 25.4,
                template_name="nature_single",
                font_family="Times New Roman",
                font_size=8.0,
                axis_label_size=4.0,
                tick_label_size=8.0,
                line_width=1.5,
                view_mode="heatmap",
                colormap="jet",
            ),
        )

        issues = nature_compliance_issues(state)
        text = "\n".join(issues)
        for expected in ("width", "height", "font family", "font size", "axis-label", "tick-label", "line", "colormap"):
            self.assertIn(expected.lower(), text.lower())

    def test_template_width_and_dpi_are_strict(self):
        state = ProjectState(
            spectra=[SpectrumLayer(name="sample", x=[1.0], y=[2.0])],
            settings=PlotSettings(
                figure_width_in=183.0 / 25.4,
                figure_height_in=60.0 / 25.4,
                template_name="nature_single",
                dpi=299,
            ),
        )

        text = "\n".join(nature_compliance_issues(state))
        self.assertIn("nature_single template requires figure width 89 mm", text)
        self.assertIn("DPI must be at least 300", text)

        state.settings.template_name = "nature_custom"
        self.assertTrue(any("template must be a Nature preset" in issue for issue in nature_compliance_issues(state)))

    def test_visible_line_requires_a_finite_x_y_pair(self):
        state = ProjectState(
            spectra=[SpectrumLayer(name="sample", x=[math.nan], y=[1.0])],
            settings=PlotSettings(),
        )

        self.assertTrue(any("visible spectrum line with data" in issue for issue in nature_compliance_issues(state)))

    def test_invisible_or_empty_layers_do_not_count_as_visible_line(self):
        state = ProjectState(
            spectra=[SpectrumLayer(name="hidden", x=[1.0], y=[2.0], visible=False)],
            settings=PlotSettings(),
        )

        self.assertTrue(any("visible spectrum line" in issue for issue in nature_compliance_issues(state)))

    def test_display_range_outside_data_fails_for_line_and_heatmap(self):
        state = ProjectState(
            spectra=[SpectrumLayer(name="sample", x=[20.0, 30.0], y=[1.0, 2.0])],
            settings=PlotSettings(x_min=100.0, x_max=110.0),
        )

        self.assertTrue(any("display range" in issue for issue in nature_compliance_issues(state)))
        state.settings.view_mode = "heatmap"
        self.assertTrue(any("display range" in issue for issue in nature_compliance_issues(state)))

    def test_non_finite_x_range_is_rejected_and_reported(self):
        with self.assertRaises(ValueError):
            PlotSettings(x_min=math.nan)

        state = ProjectState(
            spectra=[SpectrumLayer(name="sample", x=[20.0, 30.0], y=[1.0, 2.0])],
            settings=PlotSettings(),
        )
        state.settings.x_min = math.inf
        self.assertTrue(any("x_min must be finite" in issue for issue in nature_compliance_issues(state)))

    def test_gradient_colorbar_and_outside_legend_are_incompatible(self):
        state = ProjectState(
            spectra=[SpectrumLayer(name="sample", x=[20.0, 30.0], y=[1.0, 2.0])],
            settings=PlotSettings(
                view_mode="gradient_stack",
                show_colorbar=True,
                legend_location="outside right",
            ),
        )

        self.assertTrue(any("cannot be combined" in issue for issue in nature_compliance_issues(state)))

        state.settings.show_legend = False
        state.settings.show_phase_legend = True
        state.phases = [
            PhaseLayer(
                name="Phase A",
                source_path="phase.csv",
                peaks=[PhasePeak(25.0, 100.0)],
            )
        ]
        self.assertTrue(any("cannot be combined" in issue for issue in nature_compliance_issues(state)))

    def test_selected_temperature_mapping_requires_declared_compatible_units(self):
        state = ProjectState(
            spectra=[
                SpectrumLayer(
                    name="declared",
                    x=[20.0, 30.0],
                    y=[1.0, 2.0],
                    temperature=25.0,
                    temperature_unit="C",
                ),
                SpectrumLayer(
                    name="unitless",
                    x=[20.0, 30.0],
                    y=[2.0, 1.0],
                    temperature=300.0,
                ),
            ],
            settings=PlotSettings(view_mode="gradient_stack", color_by="temperature"),
        )

        self.assertTrue(any("compatible °C or K" in issue for issue in nature_compliance_issues(state)))

    def test_sampling_fields_reject_or_report_non_finite_values(self):
        for field_name, value in (
            ("show_every_n", math.nan),
            ("show_every_n", math.inf),
            ("heatmap_points", math.nan),
            ("heatmap_points", math.inf),
            ("stack_spacing", math.nan),
            ("bragg_band_height", math.inf),
        ):
            with self.subTest(field_name=field_name, value=value):
                with self.assertRaises(ValueError):
                    PlotSettings(**{field_name: value})

        state = ProjectState(
            spectra=[SpectrumLayer(name="sample", x=[20.0, 30.0], y=[1.0, 2.0])],
            settings=PlotSettings(),
        )
        state.settings.show_every_n = math.nan
        issues = nature_compliance_issues(state)
        self.assertTrue(any("show_every_n" in issue for issue in issues))


if __name__ == "__main__":
    unittest.main()
