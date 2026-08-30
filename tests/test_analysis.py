from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xrdviz.analysis import (  # noqa: E402
    DerivedPlot,
    build_rocking_curve_plot,
    build_scherrer_plot,
    build_williamson_hall_plot,
    load_peak_measurements_csv,
    load_rocking_curve_csv,
)
from xrdviz.derived import PeakMeasurement  # noqa: E402


class DerivedPlotPersistenceTests(unittest.TestCase):
    def test_round_trip_is_json_serializable_and_keeps_plot_contract(self) -> None:
        plot = DerivedPlot(
            kind="williamson_hall",
            x=[0.1, 0.2],
            y=[0.01, 0.02],
            scatter=[(0.1, 0.01), (0.2, 0.02)],
            fit_line=[(0.1, 0.01), (0.2, 0.02)],
            labels={"x": "4 sin(theta)", "y": "beta cos(theta) (rad)"},
            metrics={"r_squared": 1.0, "n_points": 2},
            source="peaks.csv",
        )

        payload = plot.to_dict()
        json.dumps(payload, allow_nan=False)
        restored = DerivedPlot.from_dict(payload)

        self.assertEqual(restored.kind, "williamson_hall")
        self.assertEqual(restored.x, [0.1, 0.2])
        self.assertEqual(restored.scatter, [(0.1, 0.01), (0.2, 0.02)])
        self.assertEqual(restored.fit_line, [(0.1, 0.01), (0.2, 0.02)])
        self.assertEqual(restored.labels["x"], "4 sin(theta)")
        self.assertEqual(restored.metrics["n_points"], 2)

    def test_rejects_unknown_kind_nonfinite_values_and_mismatched_xy(self) -> None:
        with self.assertRaises(ValueError):
            DerivedPlot(kind="unknown", x=[1.0], y=[1.0])
        with self.assertRaises(ValueError):
            DerivedPlot(kind="scherrer", x=[1.0], y=[math.nan])
        with self.assertRaises(ValueError):
            DerivedPlot(kind="rocking_curve", x=[1.0], y=[])


class AnalysisCsvImportTests(unittest.TestCase):
    def test_peak_measurements_csv_requires_explicit_columns_and_preserves_optional_fields(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "peaks.csv"
            path.write_text(
                "2theta,FWHM,hkl,intensity\n20,0.20,111,100\n30,0.30,200,80\n",
                encoding="utf-8",
            )

            peaks = load_peak_measurements_csv(path, angle_unit="deg")

        self.assertEqual(len(peaks), 2)
        self.assertIsInstance(peaks[0], PeakMeasurement)
        self.assertEqual(peaks[0].two_theta, 20.0)
        self.assertEqual(peaks[0].fwhm, 0.20)
        self.assertEqual(peaks[0].hkl, "111")
        self.assertEqual(peaks[0].intensity, 100.0)

    def test_peak_measurements_csv_rejects_missing_unknown_duplicate_and_nonfinite_fields(
        self,
    ) -> None:
        invalid_csvs = (
            "2theta,hkl\n20,111\n",
            "2theta,FWHM,other\n20,0.2,x\n",
            "2theta,FWHM\n20,nan\n",
            "2theta,FWHM,2theta\n20,0.2,20\n",
        )
        for text in invalid_csvs:
            with self.subTest(text=text):
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / "invalid.csv"
                    path.write_text(text, encoding="utf-8")
                    with self.assertRaises(ValueError):
                        load_peak_measurements_csv(path, angle_unit="deg")

    def test_rocking_curve_csv_requires_omega_and_intensity_without_guessing_extra_columns(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rocking.csv"
            path.write_text("omega,intensity\n-1,0\n0,1\n1,0\n", encoding="utf-8")
            omega, intensity = load_rocking_curve_csv(path, x_unit="deg")

        self.assertEqual(omega, [-1.0, 0.0, 1.0])
        self.assertEqual(intensity, [0.0, 1.0, 0.0])

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.csv"
            path.write_text("angle,intensity\n0,1\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_rocking_curve_csv(path, x_unit="deg")


class DerivedPlotBuilderTests(unittest.TestCase):
    def test_build_scherrer_plot_exposes_sizes_and_units_without_uncertainties(
        self,
    ) -> None:
        peaks = [
            PeakMeasurement(two_theta=20.0, fwhm=0.20, angle_unit="deg", hkl="111"),
            PeakMeasurement(two_theta=30.0, fwhm=0.30, angle_unit="deg", hkl="200"),
        ]
        plot = build_scherrer_plot(
            peaks,
            k=0.9,
            wavelength=0.15406,
            wavelength_unit="nm",
            output_unit="nm",
            source="peaks.csv",
        )

        self.assertEqual(plot.kind, "scherrer")
        self.assertEqual(plot.x, [20.0, 30.0])
        self.assertEqual(len(plot.scatter), 2)
        self.assertEqual(plot.fit_line, [])
        self.assertEqual(plot.metrics["n_points"], 2)
        self.assertEqual(plot.metrics["size_unit"], "nm")
        self.assertNotIn("uncertainty", plot.metrics)
        self.assertEqual(plot.source, "peaks.csv")
        self.assertIn("deg", plot.labels["x"])

    def test_build_williamson_hall_plot_returns_transformed_scatter_and_fit_line(
        self,
    ) -> None:
        k = 0.9
        wavelength_nm = 0.15406
        strain = 1.25e-3
        size_nm = 82.0
        peaks: list[PeakMeasurement] = []
        for two_theta in (20.0, 30.0, 40.0, 50.0):
            theta = math.radians(two_theta / 2.0)
            beta = (
                4.0 * strain * math.sin(theta) + k * wavelength_nm / size_nm
            ) / math.cos(theta)
            peaks.append(PeakMeasurement(two_theta=two_theta, fwhm=math.degrees(beta)))

        plot = build_williamson_hall_plot(
            peaks,
            k=k,
            wavelength=wavelength_nm,
            wavelength_unit="nm",
            output_unit="nm",
        )

        self.assertEqual(plot.kind, "williamson_hall")
        self.assertEqual(len(plot.scatter), 4)
        self.assertEqual(len(plot.fit_line), 2)
        self.assertAlmostEqual(plot.metrics["microstrain"], strain, places=12)
        self.assertAlmostEqual(plot.metrics["crystallite_size"], size_nm, places=8)
        self.assertAlmostEqual(plot.metrics["r_squared"], 1.0, places=12)
        self.assertEqual(plot.labels["x"], "4 sin(theta)")
        self.assertNotIn("uncertainty", plot.metrics)

    def test_build_rocking_curve_plot_reports_only_transparent_metrics(self) -> None:
        plot = build_rocking_curve_plot(
            [0.0, 1.0, 2.0, 4.0, 5.0],
            [0.0, 0.25, 1.0, 0.25, 0.0],
            x_unit="deg",
            source="rocking.csv",
        )

        self.assertEqual(plot.kind, "rocking_curve")
        self.assertEqual(plot.x, [0.0, 1.0, 2.0, 4.0, 5.0])
        self.assertEqual(len(plot.scatter), 5)
        self.assertAlmostEqual(plot.metrics["peak_position"], 2.0)
        self.assertAlmostEqual(plot.metrics["fwhm"], 2.0)
        self.assertAlmostEqual(plot.metrics["integrated_intensity"], 2.125)
        self.assertEqual(plot.labels["x"], "omega (deg)")
        self.assertNotIn("uncertainty", plot.metrics)


if __name__ == "__main__":
    unittest.main()
