import csv
import hashlib
import json
import sys
import tempfile
import unittest
from unittest.mock import PropertyMock, patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xrdviz.fit import FitComponent, PatternFit, load_pattern_fit
from xrdviz.maps import MapData, load_map_csv
from xrdviz.models import PhaseLayer, PhasePeak, PlotSettings, ProjectState, SpectrumLayer
from xrdviz.project import load_project
from xrdviz.publication import (
    _source_record,
    export_cleaned_data,
    export_fit_data,
    export_fit_summary,
    export_map_data,
    export_publication_bundle,
    export_peak_table,
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

    def test_cleaned_data_omits_empty_uncertainty_and_keeps_sample_axis_schema(self):
        state = ProjectState(
            spectra=[
                SpectrumLayer(
                    name="Q sample",
                    source_path="sample_q.csv",
                    axis_kind="q",
                    x=[1.0, 2.0],
                    y=[3.0, 4.0],
                )
            ]
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = export_cleaned_data(state, tmp)
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)

        self.assertNotIn("uncertainty", reader.fieldnames)
        self.assertEqual(
            {
                "sample",
                "source_file",
                "axis_kind",
                "x",
                "intensity",
                "frame_index",
                "time_s",
                "temperature",
                "temperature_unit",
                "group",
                "color_value",
            },
            set(reader.fieldnames or []),
        )
        self.assertEqual(rows[0]["sample"], "Q sample")
        self.assertEqual(rows[0]["axis_kind"], "q")

    def test_cleaned_data_keeps_uncertainty_column_for_mixed_layers(self):
        state = ProjectState(
            spectra=[
                SpectrumLayer(
                    name="with sigma",
                    x=[1.0, 2.0],
                    y=[3.0, 4.0],
                    y_error=[0.1, 0.2],
                ),
                SpectrumLayer(name="without sigma", x=[1.0, 2.0], y=[4.0, 5.0]),
            ]
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = export_cleaned_data(state, tmp)
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

        self.assertIn("uncertainty", rows[0])
        self.assertEqual(rows[0]["uncertainty"], "0.1")
        self.assertEqual(rows[2]["uncertainty"], "")

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

    def test_publication_bundle_is_byte_identical_across_output_directories(self):
        state = ProjectState(
            spectra=[SpectrumLayer(name="S1", x=[20.0, 30.0], y=[1.0, 2.0], source_path="s1.xy")],
            settings=PlotSettings(),
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            left = root / "left"
            right = root / "right"
            export_publication_bundle(state, left, figure_name="figure.pdf")
            export_publication_bundle(state, right, figure_name="figure.pdf")

            relative_paths = sorted(path.relative_to(left) for path in left.rglob("*"))
            self.assertEqual(relative_paths, sorted(path.relative_to(right) for path in right.rglob("*")))
            for relative_path in relative_paths:
                self.assertEqual(
                    (left / relative_path).read_bytes(),
                    (right / relative_path).read_bytes(),
                    str(relative_path),
                )

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

    def test_bundle_does_not_follow_symlinked_fixed_artifact(self):
        state = ProjectState(
            spectra=[SpectrumLayer(name="S1", x=[1.0, 2.0], y=[2.0, 3.0])]
        )

        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as outside_tmp:
            root = Path(tmp) / "bundle"
            root.mkdir()
            outside = Path(outside_tmp) / "outside.csv"
            outside.write_text("keep me", encoding="utf-8")
            link = root / "cleaned_xrd_data.csv"
            try:
                link.symlink_to(outside)
            except OSError as exc:
                self.skipTest(f"file symlinks are unavailable: {exc}")

            with self.assertRaises((ValueError, FileExistsError)):
                export_publication_bundle(state, root)

            self.assertEqual(outside.read_text(encoding="utf-8"), "keep me")

    def test_reexport_removes_only_previous_bundle_artifacts(self):
        fit_state = ProjectState(
            spectra=[SpectrumLayer(name="S1", x=[1.0, 2.0], y=[2.0, 3.0])],
            fit=PatternFit(
                name="fit",
                x=[1.0, 2.0],
                observed=[2.0, 3.0],
                calculated=[2.0, 3.0],
                components=[FitComponent(name="peak", y=[1.0, 1.0])],
            ),
        )
        plain_state = ProjectState(
            spectra=[SpectrumLayer(name="S1", x=[1.0, 2.0], y=[2.0, 3.0])]
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            export_publication_bundle(fit_state, root)
            user_file = root / "user_notes.txt"
            user_file.write_text("do not delete", encoding="utf-8")

            export_publication_bundle(plain_state, root)

            self.assertFalse((root / "pattern_fit_data.csv").exists())
            self.assertFalse((root / "peak_fit_summary.csv").exists())
            self.assertTrue(user_file.exists())

    def test_reexport_ignores_manifest_paths_that_do_not_match_owned_names(self):
        state = ProjectState(
            spectra=[SpectrumLayer(name="S1", x=[1.0, 2.0], y=[2.0, 3.0])]
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "bundle"
            root.mkdir()
            user_figure = root / "user_notes.pdf"
            user_figure.write_bytes(b"keep this figure")
            user_artifact = root / "user_notes.txt"
            user_artifact.write_text("keep this note", encoding="utf-8")
            (root / "publication_manifest.json").write_text(
                json.dumps(
                    {
                        "figures": [
                            {"path": "user_notes.pdf", "format": "pdf", "primary": True}
                        ],
                        "artifacts": [
                            {"path": "user_notes.txt", "kind": "report"}
                        ],
                        "manifest": {"path": "publication_manifest.json"},
                    }
                ),
                encoding="utf-8",
            )

            export_publication_bundle(state, root)

            self.assertEqual(user_figure.read_bytes(), b"keep this figure")
            self.assertEqual(user_artifact.read_text(encoding="utf-8"), "keep this note")

    def test_reexport_does_not_trust_a_complete_figure_set_without_bundle_identity(self):
        state = ProjectState(
            spectra=[SpectrumLayer(name="S1", x=[1.0, 2.0], y=[2.0, 3.0])]
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "bundle"
            root.mkdir()
            user_figures = [root / f"user_notes.{suffix}" for suffix in ("pdf", "svg", "tiff", "png")]
            for path in user_figures:
                path.write_bytes(f"keep {path.suffix}".encode())
            (root / "publication_manifest.json").write_text(
                json.dumps(
                    {
                        "figures": [
                            {
                                "path": path.name,
                                "format": "tiff" if path.suffix == ".tiff" else path.suffix[1:],
                                "primary": path.suffix == ".pdf",
                            }
                            for path in user_figures
                        ],
                        "manifest": {"path": "publication_manifest.json"},
                    }
                ),
                encoding="utf-8",
            )

            export_publication_bundle(state, root)

            for path in user_figures:
                self.assertTrue(path.read_bytes().startswith(b"keep"))

    def test_workspace_source_is_hashed_without_exposing_absolute_path(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as workspace_tmp, tempfile.TemporaryDirectory() as output_tmp:
            workspace_path = Path(workspace_tmp)
            source = workspace_path / "sample.xy"
            source.write_bytes(b"20 1\n30 2\n")
            source_name = source.relative_to(Path.cwd()).as_posix()
            record = _source_record(source_name, Path(output_tmp))

        self.assertEqual(record["status"], "ok")
        self.assertTrue(record["readable"])
        self.assertEqual(record["sha256"], hashlib.sha256(b"20 1\n30 2\n").hexdigest())
        self.assertNotIn("resolved_path", record)
        self.assertNotIn(str(Path.cwd()), json.dumps(record))

    def test_external_source_is_unverified_without_filesystem_probe(self):
        external = str(Path.cwd().parent / "outside" / "sample.xy")
        with patch("xrdviz.publication.Path.is_file", side_effect=AssertionError("probed external source")):
            record = _source_record(external, Path.cwd())

        self.assertEqual(record["status"], "unverified_external")
        self.assertIsNone(record["exists"])
        self.assertIsNone(record["readable"])
        self.assertNotIn("sha256", record)

    def test_parent_escaping_source_is_unverified_without_filesystem_probe(self):
        with patch("xrdviz.publication.Path.is_file", side_effect=AssertionError("probed parent escape")):
            record = _source_record("..\\outside\\sample.xy", Path.cwd())

        self.assertEqual(record["status"], "unverified_external")
        self.assertIsNone(record["exists"])
        self.assertIsNone(record["readable"])
        self.assertNotIn("sha256", record)

    def test_source_columns_use_privacy_preserving_labels(self):
        external_root = Path.cwd().parent / "private_lab_data"
        external_spectrum = external_root / "sample.xy"
        external_reference = external_root / "reference.csv"
        external_map = external_root / "map.csv"
        state = ProjectState(
            spectra=[
                SpectrumLayer(
                    name="S1",
                    source_path=str(external_spectrum),
                    x=[1.0],
                    y=[2.0],
                )
            ],
            phases=[
                PhaseLayer(
                    name="phase",
                    source_path=str(external_reference),
                    peaks=[PhasePeak(two_theta=30.0, intensity=1.0, hkl="100")],
                )
            ],
            map_data=MapData(
                kind="rsm",
                source_path=str(external_map),
                x=[0.0],
                y=[0.0],
                intensity=[[1.0]],
            ),
        )

        with tempfile.TemporaryDirectory() as tmp:
            cleaned = export_cleaned_data(state, tmp).read_text(encoding="utf-8")
            peak_table = export_peak_table(state, tmp).read_text(encoding="utf-8")
            map_csv = export_map_data(state, tmp).read_text(encoding="utf-8")

        self.assertNotIn(str(external_root), cleaned)
        self.assertIn("sample.xy", cleaned)
        self.assertNotIn(str(external_root), peak_table)
        self.assertIn("reference.csv", peak_table)
        self.assertNotIn(str(external_root), map_csv)
        self.assertIn("map.csv", map_csv)

    def test_failed_bundle_export_does_not_leave_partial_output(self):
        state = ProjectState(
            spectra=[SpectrumLayer(name="S1", x=[1.0, 2.0], y=[2.0, 3.0])]
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "bundle"
            with patch("xrdviz.publication.export_project", side_effect=RuntimeError("boom")):
                with self.assertRaisesRegex(RuntimeError, "boom"):
                    export_publication_bundle(state, root)

            self.assertFalse(root.exists())

    def test_source_manifest_does_not_expose_absolute_paths_or_external_source_hashes(self):
        state = ProjectState(
            spectra=[SpectrumLayer(name="S1", x=[1.0, 2.0], y=[2.0, 3.0], source_path="/secret/lab/sample.xy")]
        )

        with tempfile.TemporaryDirectory() as tmp:
            outputs = export_publication_bundle(state, Path(tmp) / "bundle")
            manifest = json.loads(outputs.manifest.read_text(encoding="utf-8"))
            report_text = outputs.report.read_text(encoding="utf-8")
            cleaned_text = outputs.cleaned_data.read_text(encoding="utf-8")

        record = manifest["sources"][0]
        self.assertNotIn("resolved_path", record)
        self.assertNotIn("sha256", record)
        self.assertEqual(record["path"], "sample.xy")
        self.assertEqual(record["status"], "unverified_external")
        self.assertNotIn("/secret/lab/sample.xy", json.dumps(manifest))
        self.assertNotIn("/secret/lab/sample.xy", report_text)
        self.assertNotIn("/secret/lab/sample.xy", cleaned_text)

    def test_fit_export_uses_axis_specific_column_and_summary_metadata(self):
        state = ProjectState(
            fit=PatternFit(
                name="q fit",
                axis_kind="q",
                wavelength_angstrom=1.0,
                x=[1.0, 2.0],
                observed=[2.0, 3.0],
                calculated=[2.0, 2.5],
                components=[FitComponent(name="peak", y=[1.0, 1.5])],
            )
        )

        with tempfile.TemporaryDirectory() as tmp:
            data_path = export_fit_data(state, tmp)
            summary_path = export_fit_summary(state, tmp)
            with data_path.open(newline="", encoding="utf-8") as handle:
                data_fields = next(csv.reader(handle))
            with summary_path.open(newline="", encoding="utf-8") as handle:
                summary_fields = next(csv.reader(handle))
            restored = load_pattern_fit(data_path)

        self.assertIn("q", data_fields)
        self.assertNotIn("x", data_fields)
        self.assertEqual(restored.axis_kind, "q")
        self.assertEqual(restored.wavelength_angstrom, 1.0)
        self.assertIn("axis_kind", summary_fields)
        self.assertIn("wavelength_angstrom", summary_fields)

    def test_fit_export_computes_difference_once(self):
        state = ProjectState(
            fit=PatternFit(
                name="fit",
                x=[1.0, 2.0, 3.0],
                observed=[2.0, 3.0, 4.0],
                calculated=[1.0, 2.0, 3.0],
            )
        )

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(PatternFit, "difference", new_callable=PropertyMock) as difference:
                difference.return_value = [1.0, 1.0, 1.0]
                export_fit_data(state, tmp)

        self.assertEqual(difference.call_count, 1)

    def test_bundle_fit_summary_values_and_manifest_hash_match(self):
        component = FitComponent(
            name="alpha",
            y=[2.0, 3.0],
            profile="pseudo_voigt",
            center=20.5,
            fwhm=0.4,
            area=1.25,
            amplitude=3.5,
            eta=0.3,
        )
        state = ProjectState(
            fit=PatternFit(
                name="fit",
                wavelength_angstrom=1.5406,
                x=[20.0, 21.0],
                observed=[2.0, 3.0],
                calculated=[2.0, 3.0],
                components=[component],
            ),
            settings=PlotSettings(view_mode="refinement"),
        )

        with tempfile.TemporaryDirectory() as tmp:
            outputs = export_publication_bundle(state, tmp)
            assert outputs.fit_summary is not None
            with outputs.fit_summary.open(newline="", encoding="utf-8") as handle:
                row = next(csv.DictReader(handle))
            manifest = json.loads(outputs.manifest.read_text(encoding="utf-8"))
            record = next(
                item for item in manifest["artifacts"] if item["kind"] == "fit_summary_csv"
            )
            digest = hashlib.sha256(outputs.fit_summary.read_bytes()).hexdigest()

        self.assertEqual(row["name"], "alpha")
        self.assertEqual(row["profile"], "pseudo_voigt")
        self.assertEqual(float(row["wavelength_angstrom"]), 1.5406)
        self.assertEqual(float(row["center"]), component.center)
        self.assertEqual(float(row["fwhm"]), component.fwhm)
        self.assertEqual(float(row["area"]), component.area)
        self.assertEqual(float(row["amplitude"]), component.amplitude)
        self.assertEqual(float(row["eta"]), component.eta)
        self.assertEqual(record["sha256"], digest)

    def test_map_export_passes_rows_as_stream(self):
        state = ProjectState(
            map_data=MapData(
                kind="rsm",
                x=[0.0, 1.0],
                y=[2.0, 3.0],
                intensity=[[1.0, 2.0], [3.0, 4.0]],
            )
        )

        with tempfile.TemporaryDirectory() as tmp:
            with patch("xrdviz.publication._write_rows") as write_rows:
                export_map_data(state, tmp)
                rows = write_rows.call_args.args[2]

        self.assertNotIsInstance(rows, list)

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

    def test_gradient_colorbar_is_declared_as_combination_raster_content(self):
        state = ProjectState(
            spectra=[
                SpectrumLayer(
                    name=f"{temperature:.0f} °C",
                    x=[20.0, 30.0, 40.0],
                    y=[1.0, 2.0 + index, 1.0],
                    temperature=temperature,
                    temperature_unit="°C",
                    order=index,
                )
                for index, temperature in enumerate((25.0, 200.0, 400.0))
            ],
            settings=PlotSettings(
                view_mode="gradient_stack",
                color_by="temperature",
                show_colorbar=True,
                show_legend=False,
                show_phase_legend=False,
            ),
        )

        with tempfile.TemporaryDirectory() as tmp:
            outputs = export_publication_bundle(
                state,
                Path(tmp),
                figure_name="gradient_stack.pdf",
            )
            manifest = json.loads(outputs.manifest.read_text(encoding="utf-8"))
            report = outputs.report.read_text(encoding="utf-8")

        figures = {item["format"]: item for item in manifest["figures"]}
        for format_name in ("pdf", "svg"):
            self.assertEqual(figures[format_name]["content"], "combination/raster")
            self.assertFalse(figures[format_name]["vector_claim"])
        for format_name in ("png", "tiff"):
            self.assertEqual(figures[format_name]["content"], "raster")
            self.assertFalse(figures[format_name]["vector_claim"])
        self.assertIn("gradient-stack", report)
        self.assertIn("continuous colorbar", report)

    def test_exported_empty_cake_bins_round_trip_through_map_csv(self):
        state = ProjectState(
            map_data=MapData(
                kind="cake",
                x=[10.0, 20.0],
                y=[-5.0, 5.0],
                intensity=[[0.0, 2.0], [3.0, 0.0]],
                counts=[[0.0, 2.0], [1.0, 0.0]],
            )
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = export_map_data(state, Path(tmp))
            restored = load_map_csv(path, kind="cake")

        self.assertEqual(restored.kind, "cake")
        self.assertEqual(restored.counts.tolist(), [[0.0, 2.0], [1.0, 0.0]])
        self.assertEqual(restored.intensity.tolist(), [[0.0, 2.0], [3.0, 0.0]])

    def test_exported_map_without_counts_round_trips_without_inventing_counts(self):
        state = ProjectState(
            map_data=MapData(
                kind="rsm",
                x=[0.0, 1.0],
                y=[2.0, 3.0],
                intensity=[[1.0, 2.0], [3.0, 4.0]],
            )
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = export_map_data(state, Path(tmp))
            restored = load_map_csv(path, kind="rsm")

        self.assertIsNone(restored.counts)
        self.assertEqual(restored.intensity.tolist(), [[1.0, 2.0], [3.0, 4.0]])

    def test_exported_nonuniform_map_round_trips_with_default_loader(self):
        state = ProjectState(
            map_data=MapData(
                kind="rsm",
                x=[0.0, 0.5, 2.0],
                y=[1.0, 3.0],
                intensity=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            )
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = export_map_data(state, Path(tmp))
            restored = load_map_csv(path)

        self.assertEqual(restored.x.tolist(), [0.0, 0.5, 2.0])
        self.assertEqual(restored.y.tolist(), [1.0, 3.0])
        self.assertEqual(restored.intensity.tolist(), state.map_data.intensity.tolist())

    def test_fit_export_disambiguates_cleaned_component_headers_and_round_trips(self):
        state = ProjectState(
            fit=PatternFit(
                name="fit",
                x=[20.0, 21.0],
                observed=[10.0, 11.0],
                calculated=[10.0, 11.0],
                components=[
                    FitComponent(name="Peak 1", y=[1.0, 2.0]),
                    FitComponent(name="peak-1", y=[3.0, 4.0]),
                ],
            )
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = export_fit_data(state, Path(tmp))
            with path.open(newline="", encoding="utf-8") as handle:
                fieldnames = next(csv.reader(handle))
            restored = load_pattern_fit(path)

        component_fields = [name for name in fieldnames if name.startswith("component_")]
        self.assertEqual(len(component_fields), 2)
        self.assertEqual(len(set(fieldnames)), len(fieldnames))
        self.assertEqual(len(restored.components), 2)
        self.assertEqual([component.y for component in restored.components], [[1.0, 2.0], [3.0, 4.0]])


if __name__ == "__main__":
    unittest.main()
