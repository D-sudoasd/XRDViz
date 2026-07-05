import sys
import tempfile
import unittest
import importlib.util
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xrdviz.cif import calculate_cif_peaks, peak_position_for_axis, phase_peak_position_for_axis, phase_peaks_for_axis
from xrdviz.models import PhaseLayer, PhasePeak, PlotSettings


class CifPeakDisplayTests(unittest.TestCase):
    def test_phase_peaks_convert_to_current_axis(self):
        phase = PhaseLayer(
            name="phase",
            source_path="phase.cif",
            color="#009E73",
            peaks=[PhasePeak(two_theta=30.0, intensity=100.0, hkl="111")],
        )
        settings = PlotSettings(x_axis="q", energy_kev=12.398419843)

        peaks = phase_peaks_for_axis(phase, settings)

        self.assertEqual(len(peaks), 1)
        self.assertEqual(peaks[0].hkl, "111")
        self.assertGreater(peaks[0].x, 0.0)
        self.assertEqual(peaks[0].intensity, 100.0)

    def test_phase_lattice_a_scales_theoretical_d_spacing(self):
        phase = PhaseLayer(
            name="B2",
            source_path="b2.cif",
            color="#3154d4",
            reference_lattice_a=2.8,
            lattice_a=2.94,
            peaks=[PhasePeak(two_theta=42.0, intensity=100.0, hkl="110")],
        )
        peak = phase.peaks[0]
        base_d = peak_position_for_axis(peak, "d", 8.0478)
        scaled_d = phase_peak_position_for_axis(phase, peak, "d", 8.0478)

        self.assertAlmostEqual(scaled_d, base_d * 1.05, places=8)

    @unittest.skipUnless(importlib.util.find_spec("pymatgen"), "pymatgen is not installed")
    def test_calculate_cif_peaks_reads_structure_file(self):
        from pymatgen.core import Lattice, Structure

        structure = Structure.from_spacegroup(
            "Fm-3m",
            Lattice.cubic(5.64),
            ["Na", "Cl"],
            [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nacl.cif"
            structure.to(filename=str(path))
            peaks = calculate_cif_peaks(path, energy_kev=8.0478, two_theta_range=(0.0, 90.0))

        self.assertGreater(len(peaks), 0)
        self.assertTrue(all(0.0 <= peak.two_theta <= 90.0 for peak in peaks))
        self.assertTrue(any(peak.hkl for peak in peaks))


if __name__ == "__main__":
    unittest.main()
