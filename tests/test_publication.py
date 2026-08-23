import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xrdviz.models import PhaseLayer, PhasePeak, PlotSettings, ProjectState, SpectrumLayer
from xrdviz.project import load_project
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

    def test_publication_report_records_batch_and_template_settings(self):
        state = ProjectState(
            spectra=[
                SpectrumLayer(
                    name="frame 1",
                    x=[10.0, 20.0],
                    y=[1.0, 2.0],
                    source_path="scan_0001_300C.xy",
                    frame_index=1,
                    time_s=30.0,
                    temperature=300.0,
                    color_value=1.0,
                )
            ],
            settings=PlotSettings(
                view_mode="heatmap",
                color_by="temperature",
                colormap="magma",
                show_colorbar=True,
                show_every_n=3,
                heatmap_points=128,
                template_name="science_single",
                legend_location="outside right",
            ),
        )

        with tempfile.TemporaryDirectory() as tmp:
            report = write_publication_report(state, tmp, exported_figure="figure.svg")
            text = report.read_text(encoding="utf-8")

        self.assertIn("- view mode: heatmap", text)
        self.assertIn("- color by: temperature", text)
        self.assertIn("- colormap: magma", text)
        self.assertIn("- show every N spectra: 3", text)
        self.assertIn("- heatmap points: 128", text)
        self.assertIn("- template: science_single", text)
        self.assertIn("- legend location: outside right", text)
        self.assertIn("frame=1", text)
        self.assertIn("temperature=300", text)

    def test_bundle_manifest_records_all_hashes_and_project_snapshot_round_trips(self):
        state = ProjectState(
            spectra=[SpectrumLayer(name="S1", x=[20.0, 30.0], y=[1.0, 2.0], source_path="missing.xy")],
            settings=PlotSettings(),
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            outputs = export_publication_bundle(state, root, figure_name="main.svg")
            manifest = json.loads(outputs.manifest.read_text(encoding="utf-8"))
            expected = {
                "main.pdf",
                "main.svg",
                "main.tiff",
                "main.png",
                "cleaned_xrd_data.csv",
                "reference_peak_table.csv",
                "project.xrdviz.json",
                "publication_manifest.json",
                "xrd_plot_report.md",
            }
            self.assertEqual({path.name for path in root.iterdir()}, expected)
            self.assertEqual(
                set(manifest),
                {"application", "target", "nature", "figures", "artifacts", "sources", "manifest"},
            )
            for record in [*manifest["figures"], *manifest["artifacts"]]:
                path = root / record["path"]
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                self.assertEqual(record["sha256"], digest)
            restored = load_project(outputs.project)
            self.assertEqual(restored.spectra[0].name, state.spectra[0].name)
            self.assertEqual(manifest["nature"]["issues"], [])
            self.assertEqual(manifest["manifest"]["path"], "publication_manifest.json")

    def test_bundle_rejects_absolute_and_parent_escaping_figure_names(self):
        state = ProjectState(
            spectra=[SpectrumLayer(name="S1", x=[1.0], y=[2.0])],
            settings=PlotSettings(),
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(ValueError):
                export_publication_bundle(state, root, figure_name=str(root / "outside.svg"))
            with self.assertRaises(ValueError):
                export_publication_bundle(state, root, figure_name="..\\outside.svg")
            with self.assertRaises(ValueError):
                export_publication_bundle(state, root, figure_name="\\outside.svg")
            with self.assertRaises(ValueError):
                export_publication_bundle(state, root, figure_name="C:outside.svg")

    def test_bundle_rejects_symlinked_output_directory_escape(self):
        state = ProjectState(
            spectra=[SpectrumLayer(name="S1", x=[1.0, 2.0], y=[2.0, 3.0])],
            settings=PlotSettings(),
        )

        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as outside_tmp:
            root = Path(tmp)
            link = root / "linked"
            try:
                link.symlink_to(Path(outside_tmp), target_is_directory=True)
            except OSError as exc:
                self.skipTest(f"directory symlinks are unavailable: {exc}")
            with self.assertRaises(ValueError):
                export_publication_bundle(state, root, figure_name="linked/figure.svg")

    def test_bundle_manifest_fails_unknown_mixed_temperature_units(self):
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

        with tempfile.TemporaryDirectory() as tmp:
            outputs = export_publication_bundle(state, Path(tmp), figure_name="mixed.svg")
            manifest = json.loads(outputs.manifest.read_text(encoding="utf-8"))

        self.assertEqual(manifest["nature"]["status"], "FAIL")
        self.assertTrue(any("compatible °C or K" in issue for issue in manifest["nature"]["issues"]))


if __name__ == "__main__":
    unittest.main()
