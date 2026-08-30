import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xrdviz.io import (
    apply_sample_metadata,
    clean_spectrum_rows,
    load_reference_peaks_csv,
    load_rigaku_peaks_csv,
    load_sample_labels_csv,
    load_spectrum,
    parse_spectrum_text,
)


class SpectrumParserTests(unittest.TestCase):
    def test_parse_text_skips_headers_and_bad_rows(self):
        text = """angle intensity extra
        20.0 100.0 ignored
        bad row
        21.0, 150.5, 7
        # comment
        22.0\t200.0
        """

        x, y = parse_spectrum_text(text)

        self.assertEqual(x, [20.0, 21.0, 22.0])
        self.assertEqual(y, [100.0, 150.5, 200.0])

    def test_load_spectrum_uses_file_stem_and_declared_axis(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.xy"
            path.write_text("2 20\n1 10\n1 12\nbad row\n3 -5\n", encoding="utf-8")

            layer = load_spectrum(path, axis_kind="q", color="#0072B2")

        self.assertEqual(layer.name, "sample")
        self.assertEqual(layer.axis_kind, "q")
        self.assertEqual(layer.x, [1.0, 2.0])
        self.assertEqual(layer.y, [10.0, 20.0])
        self.assertEqual(layer.raw_x, [2.0, 1.0, 1.0, 3.0])
        self.assertEqual(layer.raw_y, [20.0, 10.0, 12.0, -5.0])
        self.assertGreaterEqual(layer.removed_rows, 2)
        self.assertTrue(any("negative intensity" in warning for warning in layer.warnings))
        self.assertEqual(layer.color, "#0072B2")

    def test_load_spectrum_auto_detects_d_spacing_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profile.txt"
            path.write_text(
                "# Metadata:\n# Axis label: d-spacing\n# Axis unit: A\n# axis\tvalue\n0.9\t10\n1.0\t20\n",
                encoding="utf-8",
            )

            layer = load_spectrum(path, axis_kind="auto")

        self.assertEqual(layer.axis_kind, "d")

    def test_load_spectrum_csv_preserves_declared_uncertainty_column(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "measured.csv"
            path.write_text(
                "two_theta,intensity,sigma\n20,100,2\n21,120,3\n22,110,2.5\n",
                encoding="utf-8",
            )

            layer = load_spectrum(path, axis_kind="auto")

        self.assertEqual(layer.axis_kind, "two_theta")
        self.assertEqual(layer.y_error, [2.0, 3.0, 2.5])

    def test_load_spectrum_reads_single_sample_cleaned_export_without_uncertainty(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cleaned_xrd_data.csv"
            path.write_text(
                "sample,source_file,axis_kind,x,intensity,uncertainty,frame_index,time_s,temperature,temperature_unit,group,color_value\n"
                "annealed,raw_q.xy,q,1.0,10,,3,12.5,450,K,A,0.2\n"
                "annealed,raw_q.xy,q,2.0,20,,3,12.5,450,K,A,0.2\n",
                encoding="utf-8",
            )

            layer = load_spectrum(path, axis_kind="auto")

        self.assertEqual(layer.name, "annealed")
        self.assertEqual(layer.source_path, "raw_q.xy")
        self.assertEqual(layer.axis_kind, "q")
        self.assertEqual(layer.x, [1.0, 2.0])
        self.assertEqual(layer.y, [10.0, 20.0])
        self.assertEqual(layer.y_error, [])
        self.assertEqual(layer.frame_index, 3)
        self.assertEqual(layer.time_s, 12.5)
        self.assertEqual(layer.temperature, 450.0)
        self.assertEqual(layer.temperature_unit, "K")
        self.assertEqual(layer.group, "A")
        self.assertEqual(layer.color_value, 0.2)

    def test_load_spectrum_default_preserves_cleaned_q_axis(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cleaned_q.csv"
            path.write_text(
                "sample,source_file,axis_kind,x,intensity,uncertainty\n"
                "annealed,raw_q.xy,q,1.0,10,0.5\n"
                "annealed,raw_q.xy,q,2.0,20,1.0\n",
                encoding="utf-8",
            )

            layer = load_spectrum(path)
            override = load_spectrum(path, axis_kind="d")

        self.assertEqual(layer.axis_kind, "q")
        self.assertEqual(layer.x, [1.0, 2.0])
        self.assertEqual(layer.y_error, [0.5, 1.0])
        self.assertEqual(override.axis_kind, "d")

    def test_load_spectrum_rejects_multi_sample_cleaned_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cleaned_xrd_data.csv"
            path.write_text(
                "sample,source_file,axis_kind,x,intensity,uncertainty\n"
                "one,a.xy,two_theta,1,10,\n"
                "two,b.xy,two_theta,1,20,\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "multiple samples"):
                load_spectrum(path, axis_kind="auto")

    def test_parse_text_rejects_files_without_two_numeric_columns(self):
        with self.assertRaises(ValueError):
            parse_spectrum_text("angle only\n20\n21\n")

    def test_clean_spectrum_rows_sorts_and_warns(self):
        cleaned = clean_spectrum_rows([30.0, 20.0, 20.0, 40.0], [5.0, 10.0, 12.0, -1.0])

        self.assertEqual(cleaned.x, [20.0, 30.0])
        self.assertEqual(cleaned.y, [10.0, 5.0])
        self.assertEqual(cleaned.raw_x, [30.0, 20.0, 20.0, 40.0])
        self.assertEqual(cleaned.raw_y, [5.0, 10.0, 12.0, -1.0])
        self.assertEqual(cleaned.removed_rows, 2)
        self.assertTrue(any("duplicate x" in warning for warning in cleaned.warnings))
        self.assertTrue(any("negative intensity" in warning for warning in cleaned.warnings))

    def test_sample_labels_csv_maps_layer_metadata(self):
        csv_text = """filename,label,order,color,visible,offset
        sample_a.xy,Annealed,2,#D55E00,true,0.4
        sample_b.xy,As cast,1,#0072B2,false,-0.2
        """

        metadata = load_sample_labels_csv(csv_text)
        layer = load_spectrum_from_text("1 10\n2 20\n", name="sample_a", source_path="sample_a.xy")
        apply_sample_metadata([layer], metadata)

        self.assertEqual(layer.name, "Annealed")
        self.assertEqual(layer.order, 2)
        self.assertEqual(layer.color, "#D55E00")
        self.assertTrue(layer.visible)
        self.assertEqual(layer.offset, 0.4)

    def test_reference_peak_csv_accepts_axis_phase_and_shape(self):
        csv_text = """position,label,phase,intensity,hkl,source_axis,color,shape
        30.0,Main peak,Calcite,100,104,two_theta,#D55E00,triangle
        2.5,Second,Calcite,40,110,d,#D55E00,square
        """

        phase = load_reference_peaks_csv(csv_text, source_path="reference_peaks.csv")

        self.assertEqual(phase.name, "Calcite")
        self.assertEqual(phase.source_type, "reference_csv")
        self.assertEqual(phase.marker_shape, "triangle")
        self.assertEqual(len(phase.peaks), 2)
        self.assertEqual(phase.peaks[0].label, "Main peak")
        self.assertEqual(phase.peaks[0].source_axis, "two_theta")
        self.assertEqual(phase.peaks[1].source_axis, "d")

    def test_rigaku_peaks_csv_uses_first_numeric_peak_column(self):
        csv_text = """No.,Angle,Height
        1,22.5,100
        2,45.0,20
        """

        phase = load_rigaku_peaks_csv(csv_text, source_path="sample peak.csv", phase_name="sample")

        self.assertEqual(phase.source_type, "rigaku_peaks_csv")
        self.assertEqual([peak.two_theta for peak in phase.peaks], [22.5, 45.0])


def load_spectrum_from_text(text: str, name: str, source_path: str):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / source_path
        path.write_text(text, encoding="utf-8")
        return load_spectrum(path, axis_kind="two_theta")


if __name__ == "__main__":
    unittest.main()
