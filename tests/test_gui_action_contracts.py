from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


_HAS_QT = all(
    importlib.util.find_spec(name) is not None
    for name in ("PySide6", "matplotlib", "scipy")
)


@unittest.skipUnless(
    _HAS_QT,
    "PySide6, matplotlib, and scipy are required for GUI action tests",
)
class GuiActionContractTests(unittest.TestCase):
    """Exercise the user-facing import actions without opening native dialogs."""

    @classmethod
    def setUpClass(cls) -> None:
        from PySide6.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])
        from xrdviz.ui.main_window import MainWindow

        cls.MainWindow = MainWindow

    def setUp(self) -> None:
        self.window = self.MainWindow()

    def tearDown(self) -> None:
        self.window.close()
        self.app.processEvents()

    def test_detector_action_accepted_dialog_updates_map_view_and_parameters(self):
        from PySide6.QtWidgets import QDialog

        from xrdviz.ui import advanced_workflows
        from xrdviz.ui.analysis_dialogs import DetectorImportDialog

        class AcceptedDetectorDialog(DetectorImportDialog):
            def exec(self):  # noqa: D401 - deterministic test dialog boundary
                self.mode.setCurrentIndex(self.mode.findData("raw"))
                self.accept()
                return QDialog.Accepted

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "detector.npy"
            # The action must load the selected file before constructing the
            # parameter dialog, so this is a real, minimal detector input.
            import numpy as np

            np.save(path, np.arange(9, dtype=float).reshape(3, 3))

            with (
                patch.object(
                    advanced_workflows.QFileDialog,
                    "getOpenFileName",
                    return_value=(str(path), ""),
                ),
                patch.object(
                    advanced_workflows,
                    "DetectorImportDialog",
                    AcceptedDetectorDialog,
                ),
            ):
                self.window.detector_action.trigger()
                self.app.processEvents()

        self.assertIsNotNone(self.window.state.map_data)
        assert self.window.state.map_data is not None
        self.assertEqual(self.window.state.map_data.kind, "detector")
        self.assertEqual(self.window.view_mode_combo.currentData(), "map")
        self.assertIn("map", self.window.last_render_axes)

    def test_detector_action_rejected_dialog_preserves_state_and_view(self):
        from PySide6.QtWidgets import QDialog

        from xrdviz.ui import advanced_workflows
        from xrdviz.ui.analysis_dialogs import DetectorImportDialog

        class RejectedDetectorDialog(DetectorImportDialog):
            def exec(self):  # noqa: D401 - deterministic test dialog boundary
                self.reject()
                return QDialog.Rejected

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "detector.npy"
            import numpy as np

            np.save(path, np.ones((2, 2), dtype=float))
            before_map = self.window.state.map_data
            before_mode = self.window.view_mode_combo.currentData()

            with (
                patch.object(
                    advanced_workflows.QFileDialog,
                    "getOpenFileName",
                    return_value=(str(path), ""),
                ),
                patch.object(
                    advanced_workflows,
                    "DetectorImportDialog",
                    RejectedDetectorDialog,
                ),
            ):
                self.window.detector_action.trigger()
                self.app.processEvents()

        self.assertIs(self.window.state.map_data, before_map)
        self.assertEqual(self.window.view_mode_combo.currentData(), before_mode)

    def test_detector_import_dialog_parameters_are_typed_and_complete(self):
        from xrdviz.ui.analysis_dialogs import DetectorImportDialog

        dialog = DetectorImportDialog((8, 10), self.window)
        dialog.mode.setCurrentIndex(dialog.mode.findData("radial"))
        dialog.center_x.setValue(2.25)
        dialog.center_y.setValue(3.5)
        dialog.pixel_x.setValue(0.12)
        dialog.pixel_y.setValue(0.18)
        dialog.distance.setValue(125.0)
        dialog.wavelength.setValue(1.7889)
        dialog.radial_unit.setCurrentIndex(dialog.radial_unit.findData("q"))
        dialog.radial_bins.setValue(2048)

        self.assertEqual(
            dialog.parameters(),
            {
                "mode": "radial",
                "center": (2.25, 3.5),
                "pixel_size": (0.12, 0.18),
                "distance": 125.0,
                "wavelength": 1.7889,
                "radial_unit": "q",
                "radial_bins": 2048,
                "chi_bins": 360,
            },
        )
        dialog.close()
        self.app.processEvents()

    def test_map_action_accepted_dialog_updates_map_view_and_metadata(self):
        from PySide6.QtWidgets import QDialog

        from xrdviz.ui import advanced_workflows
        from xrdviz.ui.analysis_dialogs import MapImportDialog

        class AcceptedMapDialog(MapImportDialog):
            def exec(self):  # noqa: D401 - deterministic test dialog boundary
                self.x_label.setText("q parallel")
                self.y_label.setText("q perpendicular")
                self.x_unit.setText("A^-1")
                self.y_unit.setText("A^-1")
                self.intensity_label.setText("Counts")
                self.intensity_unit.setText("counts")
                self.accept()
                return QDialog.Accepted

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rsm.csv"
            path.write_text(
                "qx,qz,intensity\n0,0,1\n1,0,2\n0,1,3\n1,1,4\n",
                encoding="utf-8",
            )

            with (
                patch.object(
                    advanced_workflows.QFileDialog,
                    "getOpenFileName",
                    return_value=(str(path), ""),
                ),
                patch.object(advanced_workflows, "MapImportDialog", AcceptedMapDialog),
            ):
                self.window.rsm_action.trigger()
                self.app.processEvents()

        self.assertIsNotNone(self.window.state.map_data)
        assert self.window.state.map_data is not None
        self.assertEqual(self.window.state.map_data.kind, "rsm")
        self.assertEqual(self.window.state.map_data.labels["x"], "q parallel")
        self.assertEqual(self.window.state.map_data.units["intensity"], "counts")
        self.assertEqual(self.window.view_mode_combo.currentData(), "map")
        self.assertEqual(self.window.last_render_axes["main"].name, "rectilinear")

    def test_map_action_rejected_dialog_preserves_existing_map(self):
        from PySide6.QtWidgets import QDialog

        from xrdviz import maps
        from xrdviz.ui import advanced_workflows
        from xrdviz.ui.analysis_dialogs import MapImportDialog

        class RejectedMapDialog(MapImportDialog):
            def exec(self):  # noqa: D401 - deterministic test dialog boundary
                self.reject()
                return QDialog.Rejected

        existing = maps.MapData(
            "rsm",
            [0.0, 1.0],
            [0.0, 1.0],
            [[1.0, 2.0], [3.0, 4.0]],
        )
        self.window.state.map_data = existing
        self.window.view_mode_combo.setCurrentIndex(
            self.window.view_mode_combo.findData("map")
        )
        before_mode = self.window.view_mode_combo.currentData()

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "new_rsm.csv"
            path.write_text(
                "qx,qz,intensity\n0,0,1\n1,0,2\n0,1,3\n1,1,4\n",
                encoding="utf-8",
            )

            with (
                patch.object(
                    advanced_workflows.QFileDialog,
                    "getOpenFileName",
                    return_value=(str(path), ""),
                ),
                patch.object(advanced_workflows, "MapImportDialog", RejectedMapDialog),
            ):
                self.window.rsm_action.trigger()
                self.app.processEvents()

        self.assertIs(self.window.state.map_data, existing)
        self.assertEqual(self.window.view_mode_combo.currentData(), before_mode)

    def test_fit_action_accepts_selected_csv_and_cancelled_picker_is_noop(self):
        from xrdviz.ui import advanced_workflows

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fit.csv"
            path.write_text(
                "x,observed,calculated,sigma,background,component_alpha\n"
                "20,10,9,1,2,7\n"
                "21,20,19,1,2,17\n"
                "22,11,12,2,2,10\n",
                encoding="utf-8",
            )

            with patch.object(
                advanced_workflows.QFileDialog,
                "getOpenFileName",
                return_value=(str(path), ""),
            ):
                self.window.open_fit_action.trigger()
                self.app.processEvents()

            self.assertIsNotNone(self.window.state.fit)
            assert self.window.state.fit is not None
            existing_fit = self.window.state.fit
            self.assertEqual(self.window.state.fit.components[0].name, "alpha")
            self.assertEqual(
                self.window.view_mode_combo.currentData(), "refinement"
            )
            self.assertIn("residual", self.window.last_render_axes)

            with patch.object(
                advanced_workflows.QFileDialog,
                "getOpenFileName",
                return_value=("", ""),
            ):
                self.window.open_fit_action.trigger()
                self.app.processEvents()

            self.assertIs(self.window.state.fit, existing_fit)
            self.assertEqual(
                self.window.view_mode_combo.currentData(), "refinement"
            )


if __name__ == "__main__":
    unittest.main()
