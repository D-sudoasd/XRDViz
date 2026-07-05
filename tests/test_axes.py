import math
import sys
import unittest
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xrdviz.axes import convert_x, wavelength_from_energy


class AxisConversionTests(unittest.TestCase):
    def test_energy_to_wavelength_uses_keV_angstrom_constant(self):
        self.assertAlmostEqual(wavelength_from_energy(12.398419843), 1.0, places=12)

    def test_two_theta_round_trips_through_d_spacing(self):
        original = [20.0, 40.0, 80.0]
        d_values = convert_x(original, "two_theta", "d", energy_kev=8.0)
        converted = convert_x(d_values, "d", "two_theta", energy_kev=8.0)

        for expected, actual in zip(original, converted):
            self.assertAlmostEqual(actual, expected, places=9)

    def test_q_round_trips_through_d_spacing(self):
        original = [1.0, 2.5, 4.0]
        d_values = convert_x(original, "q", "d", energy_kev=12.0)
        converted = convert_x(d_values, "d", "q", energy_kev=12.0)

        for expected, actual in zip(original, converted):
            self.assertAlmostEqual(actual, expected, places=9)

    def test_invalid_geometry_is_nan_and_warns(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            converted = convert_x([0.1], "d", "two_theta", energy_kev=8.0)

        self.assertTrue(math.isnan(converted[0]))
        self.assertTrue(any("outside valid diffraction geometry" in str(w.message) for w in caught))


if __name__ == "__main__":
    unittest.main()
