import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xrdviz.models import PhaseLayer, PhasePeak, PlotSettings, ProjectState, SpectrumLayer
from xrdviz.project import load_project, save_project


class ProjectSerializationTests(unittest.TestCase):
    def test_project_round_trips_to_json(self):
        state = ProjectState(
            spectra=[SpectrumLayer(name="s", x=[1.0], y=[2.0], axis_kind="q", color="#000000")],
            phases=[
                PhaseLayer(
                    name="p",
                    source_path="phase.cif",
                    color="#D55E00",
                    peaks=[PhasePeak(30.0, 100.0, "111")],
                    reference_lattice_a=3.5,
                    lattice_a=3.57,
                    auto_calibrated=True,
                    calibration_confidence="high",
                    calibration_error=0.006,
                    calibration_notes=["2 matched peaks"],
                )
            ],
            settings=PlotSettings(x_axis="d", energy_kev=18.0, log_scale=True, show_y_tick_labels=True),
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "project.xrdviz.json"
            save_project(state, path)
            loaded = load_project(path)

        self.assertEqual(loaded.settings.x_axis, "d")
        self.assertTrue(loaded.settings.log_scale)
        self.assertTrue(loaded.settings.show_y_tick_labels)
        self.assertEqual(loaded.spectra[0].axis_kind, "q")
        self.assertEqual(loaded.phases[0].peaks[0].hkl, "111")
        self.assertTrue(loaded.phases[0].auto_calibrated)
        self.assertEqual(loaded.phases[0].calibration_confidence, "high")
        self.assertAlmostEqual(loaded.phases[0].calibration_error, 0.006)
        self.assertEqual(loaded.phases[0].calibration_notes, ["2 matched peaks"])


if __name__ == "__main__":
    unittest.main()
