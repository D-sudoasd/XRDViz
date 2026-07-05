import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xrdviz.models import PhaseLayer, PhasePeak, PlotSettings, ProjectState, SpectrumLayer
from xrdviz.publication import (
    export_cleaned_data,
    export_publication_bundle,
    make_peak_table_rows,
    write_publication_report,
)


class PublicationWorkflowTests(unittest.TestCase):
    def test_peak_table_reports_two_theta_d_and_q(self):
        state = ProjectState(
            phases=[
                PhaseLayer(
                    name="Calcite",
                    source_path="reference_peaks.csv",
                    source_type="reference_csv",
                    peaks=[PhasePeak(two_theta=30.0, intensity=100.0, hkl="104", label="Main", source_axis="two_theta")],
                )
            ],
            settings=PlotSettings(energy_kev=12.398419843),
        )

        rows = make_peak_table_rows(state)

        self.assertEqual(rows[0]["phase"], "Calcite")
        self.assertEqual(rows[0]["hkl"], "104")
        self.assertAlmostEqual(rows[0]["two_theta"], 30.0)
        self.assertGreater(rows[0]["d"], 0.0)
        self.assertGreater(rows[0]["q"], 0.0)

    def test_cleaned_data_and_report_are_written(self):
        state = ProjectState(
            spectra=[
                SpectrumLayer(
                    name="Annealed",
                    source_path="sample.xy",
                    x=[20.0, 30.0],
                    y=[0.5, 1.0],
                    raw_x=[30.0, 20.0, 20.0],
                    raw_y=[1.0, 0.5, 0.6],
                    removed_rows=1,
                    warnings=["1 duplicate x row removed"],
                )
            ],
            phases=[
                PhaseLayer(
                    name="Calcite",
                    source_path="reference_peaks.csv",
                    source_type="reference_csv",
                    card_id="PDF 05-0586",
                    reference_lattice_a=3.5,
                    lattice_a=3.58,
                    auto_calibrated=True,
                    calibration_confidence="high",
                    calibration_error=0.007,
                    calibration_notes=["matched 2 peaks"],
                    peaks=[PhasePeak(two_theta=30.0, intensity=100.0, hkl="104", label="Main")],
                )
            ],
            settings=PlotSettings(x_axis="two_theta", energy_kev=8.0478, log_scale=True, stack_enabled=True),
        )

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            cleaned_path = export_cleaned_data(state, out_dir)
            report_path = write_publication_report(
                state,
                out_dir,
                exported_figure=out_dir / "figure.pdf",
                cleaned_data=cleaned_path,
            )

            with cleaned_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            report = report_path.read_text(encoding="utf-8")

        self.assertEqual(rows[0]["sample"], "Annealed")
        self.assertIn("sample.xy", report)
        self.assertIn("duplicate x", report)
        self.assertIn("reference_peaks.csv", report)
        self.assertIn("log scale: enabled", report)
        self.assertIn("auto-calibrated a=3.58", report)
        self.assertIn("confidence=high", report)

    def test_publication_bundle_exports_figure_data_peak_table_and_report(self):
        state = ProjectState(
            spectra=[SpectrumLayer(name="S1", x=[20.0, 30.0], y=[1.0, 2.0], source_path="s1.xy")],
            phases=[
                PhaseLayer(
                    name="Calcite",
                    source_path="reference_peaks.csv",
                    source_type="reference_csv",
                    peaks=[PhasePeak(30.0, 100.0, "104", label="Main")],
                )
            ],
        )

        with tempfile.TemporaryDirectory() as tmp:
            outputs = export_publication_bundle(state, Path(tmp), figure_name="figure.svg")

            self.assertTrue(outputs.figure.exists())
            self.assertTrue(outputs.cleaned_data.exists())
            self.assertTrue(outputs.peak_table.exists())
            self.assertTrue(outputs.report.exists())
            self.assertIn("figure.svg", outputs.report.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
