from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xrdviz.derived import (  # noqa: E402
    PeakMeasurement,
    rocking_curve_metrics,
    scherrer_crystallite_size,
    williamson_hall_fit,
)


class PeakMeasurementTests(unittest.TestCase):
    def test_peak_measurement_validates_finite_positive_values(self) -> None:
        peak = PeakMeasurement(
            two_theta=30.0, fwhm=0.20, angle_unit="deg", intensity=100.0
        )

        self.assertEqual(peak.two_theta, 30.0)
        self.assertEqual(peak.fwhm, 0.20)
        self.assertEqual(peak.angle_unit, "deg")
        self.assertAlmostEqual(peak.two_theta_deg, 30.0)
        self.assertAlmostEqual(peak.fwhm_deg, 0.20)

        for kwargs in (
            {"two_theta": math.nan, "fwhm": 0.20},
            {"two_theta": 30.0, "fwhm": 0.0},
            {"two_theta": 30.0, "fwhm": -0.20},
            {"two_theta": 30.0, "fwhm": 0.20, "angle_unit": "arcmin"},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    PeakMeasurement(**kwargs)

    def test_scherrer_uses_explicit_k_and_quadrature_instrument_correction(
        self,
    ) -> None:
        peak = PeakMeasurement(two_theta=30.0, fwhm=0.30, angle_unit="deg")
        corrected_rad = math.radians(math.sqrt(0.30**2 - 0.10**2))
        expected_nm = 0.90 * 0.15406 / (corrected_rad * math.cos(math.radians(15.0)))

        value = scherrer_crystallite_size(
            peak,
            k=0.90,
            wavelength=0.15406,
            wavelength_unit="nm",
            instrument_fwhm=0.10,
            output_unit="nm",
        )

        self.assertAlmostEqual(value, expected_nm, places=12)

    def test_scherrer_rejects_invalid_physical_inputs(self) -> None:
        peak = PeakMeasurement(two_theta=30.0, fwhm=0.10, angle_unit="deg")

        invalid_calls = (
            {"k": 0.0, "wavelength": 0.15406},
            {"k": 0.90, "wavelength": 0.0},
            {"k": math.inf, "wavelength": 0.15406},
            {"k": 0.90, "wavelength": 0.15406, "wavelength_unit": "foo"},
            {"k": 0.90, "wavelength": 0.15406, "instrument_fwhm": 0.10},
        )
        for kwargs in invalid_calls:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    scherrer_crystallite_size(peak, **kwargs)


class WilliamsonHallTests(unittest.TestCase):
    def test_linear_fit_returns_strain_size_and_goodness_of_fit(self) -> None:
        k = 0.90
        wavelength_nm = 0.15406
        expected_strain = 1.25e-3
        expected_size_nm = 82.0
        peaks: list[PeakMeasurement] = []
        for two_theta in (20.0, 30.0, 40.0, 50.0):
            theta = math.radians(two_theta / 2.0)
            beta_rad = (
                4.0 * expected_strain * math.sin(theta)
                + k * wavelength_nm / expected_size_nm
            ) / math.cos(theta)
            peaks.append(
                PeakMeasurement(two_theta=two_theta, fwhm=math.degrees(beta_rad))
            )

        result = williamson_hall_fit(
            peaks,
            k=k,
            wavelength=wavelength_nm,
            wavelength_unit="nm",
            output_unit="nm",
        )

        self.assertAlmostEqual(result.slope, expected_strain, places=12)
        self.assertAlmostEqual(result.microstrain, expected_strain, places=12)
        self.assertAlmostEqual(result.crystallite_size, expected_size_nm, places=9)
        self.assertAlmostEqual(
            result.intercept, k * wavelength_nm / expected_size_nm, places=12
        )
        self.assertAlmostEqual(result.r_squared, 1.0, places=12)
        self.assertEqual(result.n_points, 4)
        self.assertEqual(result.point_count, 4)

    def test_williamson_hall_fails_closed_for_insufficient_or_degenerate_input(
        self,
    ) -> None:
        peak = PeakMeasurement(two_theta=30.0, fwhm=0.20)
        with self.assertRaises(ValueError):
            williamson_hall_fit([peak], k=0.9, wavelength=0.15406)
        with self.assertRaises(ValueError):
            williamson_hall_fit(
                [peak, PeakMeasurement(two_theta=30.0, fwhm=0.30)],
                k=0.9,
                wavelength=0.15406,
            )
        with self.assertRaises(ValueError):
            williamson_hall_fit(
                [peak, PeakMeasurement(two_theta=40.0, fwhm=0.30)],
                k=0.9,
                wavelength=0.15406,
                instrument_fwhm=[0.0],
            )


class RockingCurveTests(unittest.TestCase):
    def test_metrics_use_piecewise_linear_crossings_and_trapezoidal_area(self) -> None:
        metrics = rocking_curve_metrics(
            [0.0, 1.0, 2.0, 4.0, 5.0],
            [0.0, 0.25, 1.0, 0.25, 0.0],
        )

        self.assertAlmostEqual(metrics.peak_position, 2.0)
        self.assertAlmostEqual(metrics.fwhm, 2.0, places=12)
        self.assertAlmostEqual(metrics.integrated_intensity, 2.125, places=12)

    def test_metrics_accept_descending_axis_and_reject_missing_half_height_crossings(
        self,
    ) -> None:
        descending = rocking_curve_metrics(
            [5.0, 4.0, 2.0, 1.0, 0.0],
            [0.0, 0.25, 1.0, 0.25, 0.0],
        )
        self.assertAlmostEqual(descending.peak_position, 2.0)
        self.assertAlmostEqual(descending.fwhm, 2.0)

        with self.assertRaises(ValueError):
            rocking_curve_metrics([0.0, 1.0, 2.0], [1.0, 2.0, 3.0])
        with self.assertRaises(ValueError):
            rocking_curve_metrics([0.0, 1.0], [0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
