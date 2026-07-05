import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xrdviz.models import PhaseLayer, PhasePeak, PlotSettings, ProjectState, SpectrumLayer


class CalibrationTests(unittest.TestCase):
    def test_joint_assignment_keeps_b2_and_fcc_on_different_central_peaks(self):
        from xrdviz.calibration import auto_calibrate_phases

        state = ProjectState(
            spectra=[
                SpectrumLayer(
                    name="sample",
                    axis_kind="d",
                    x=[1.75, 1.795, 2.02, 2.035, 2.075, 2.09],
                    y=[20, 90, 15, 100, 95, 20],
                )
            ],
            phases=[
                PhaseLayer(
                    name="B2",
                    source_path="b2.cif",
                    reference_lattice_a=2.88,
                    lattice_a=2.88,
                    peaks=[
                        PhasePeak(two_theta=2.04, intensity=100, hkl="110", source_axis="d"),
                        PhasePeak(two_theta=1.18, intensity=40, hkl="211", source_axis="d"),
                    ],
                ),
                PhaseLayer(
                    name="FCC",
                    source_path="fcc.cif",
                    reference_lattice_a=3.50,
                    lattice_a=3.50,
                    peaks=[
                        PhasePeak(two_theta=2.02, intensity=100, hkl="111", source_axis="d"),
                        PhasePeak(two_theta=1.75, intensity=70, hkl="200", source_axis="d"),
                    ],
                ),
            ],
            settings=PlotSettings(x_axis="d", x_min=1.7, x_max=2.15),
        )

        results = auto_calibrate_phases(state)

        b2_match_positions = [round(match.observed_d, 3) for match in results["B2"].matched_peaks]
        fcc_match_positions = [round(match.observed_d, 3) for match in results["FCC"].matched_peaks]
        self.assertIn(2.035, b2_match_positions)
        self.assertIn(2.075, fcc_match_positions)
        self.assertGreater(state.phases[1].lattice_a, state.phases[1].reference_lattice_a)

    def test_low_confidence_result_does_not_overwrite_existing_manual_lattice(self):
        from xrdviz.calibration import auto_calibrate_phases

        phase = PhaseLayer(
            name="FCC",
            source_path="fcc.cif",
            reference_lattice_a=3.5,
            lattice_a=3.6,
            peaks=[PhasePeak(two_theta=2.02, intensity=100, hkl="111", source_axis="d")],
        )
        state = ProjectState(
            spectra=[SpectrumLayer(name="sample", axis_kind="d", x=[1.0, 1.5, 2.08], y=[3, 5, 100])],
            phases=[phase],
            settings=PlotSettings(x_axis="d", x_min=0.9, x_max=2.2),
        )

        result = auto_calibrate_phases(state)["FCC"]

        self.assertEqual(result.confidence, "low")
        self.assertAlmostEqual(phase.lattice_a, 3.6)
        self.assertFalse(phase.auto_calibrated)

    def test_nb_free_project_helper_uses_b2_and_fcc_only(self):
        from xrdviz.calibration import build_publication_state

        state = build_publication_state(
            title="Nb-free",
            spectra=[("H-free", "h_free.xy"), ("H-charged", "h_charged.xy")],
            phase_paths=[("B2", "b2.cif"), ("FCC", "fcc.cif")],
        )

        self.assertEqual([phase.phase for phase in state.phases], ["B2", "FCC"])
        self.assertNotIn("Laves", [phase.phase for phase in state.phases])
        self.assertEqual(state.settings.panel_title, "Nb-free")
