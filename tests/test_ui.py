import importlib.util
import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))


@unittest.skipUnless(
    importlib.util.find_spec("PySide6") and importlib.util.find_spec("matplotlib"),
    "PySide6 and matplotlib are not installed",
)
class UiSmokeTests(unittest.TestCase):
    def test_main_window_constructs(self):
        script = """
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication
from xrdviz.ui.main_window import MainWindow
app = QApplication([])
window = MainWindow()
assert window.windowTitle() == "XRDViz"
assert hasattr(window, "layer_table")
assert hasattr(window, "peak_table")
assert hasattr(window, "auto_fit_button")
assert hasattr(window, "y_tick_labels_check")
assert hasattr(window, "panel_title_edit")
assert hasattr(window, "x_min_edit")
assert hasattr(window, "x_max_edit")
assert window.y_tick_labels_check.isChecked() is False
window.log_check.setChecked(True)
window.stack_check.setChecked(True)
window.y_tick_labels_check.setChecked(True)
window.panel_title_edit.setText("Nb-free")
window.x_min_edit.setText("0.9")
window.x_max_edit.setText("3.0")
window.stack_spacing_spin.setValue(0.5)
window.x_axis_combo.setCurrentIndex(1)
from pathlib import Path
import tempfile
with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    spectrum = root / "a.xy"
    spectrum.write_text("20 10\\n30 20\\n", encoding="utf-8")
    labels = root / "sample_labels.csv"
    labels.write_text("filename,label,order,color,visible,offset\\na.xy,Sample A,1,#D55E00,true,0.3\\n", encoding="utf-8")
    refs = root / "reference_peaks.csv"
    refs.write_text("position,label,phase,intensity,hkl,source_axis,color,shape\\n30,Main,Calcite,100,104,two_theta,#009E73,triangle\\n", encoding="utf-8")
    window.add_files([spectrum])
    window.import_sample_metadata(labels)
    window.import_reference_peaks(refs)
    assert window.layer_table.rowCount() == 2
    assert window.layer_table.columnCount() == 7
    assert window.peak_table.rowCount() == 1
    window.layer_table.item(0, 2).setText("H-free")
    window.layer_table.item(1, 2).setText("B2")
    assert window.state.spectra[0].name == "H-free"
    assert window.state.phases[0].phase == "B2"
    window.state.phases[0].reference_lattice_a = 2.8
    window.state.phases[0].lattice_a = 2.8
    window.refresh_layers()
    window.layer_table.item(1, 6).setText("2.9")
    assert abs(window.state.phases[0].lattice_a - 2.9) < 1e-9
    window.auto_fit_phase_peaks()
    assert hasattr(window.state.phases[0], "calibration_confidence")
    window.layer_table.selectRow(0)
    window.toggle_selected_visibility()
    assert window.state.spectra[0].visible is False
    window.toggle_selected_visibility()
    assert window.state.spectra[0].visible is True
    outputs = window.export_publication_bundle_to(root / "export")
    assert outputs.report.exists()
app.processEvents()
window.close()
app.processEvents()
"""
        env = dict(os.environ)
        env["PYTHONPATH"] = str(SRC)

        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )

        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)


if __name__ == "__main__":
    unittest.main()
