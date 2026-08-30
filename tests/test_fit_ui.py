from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


@unittest.skipUnless(
    importlib.util.find_spec("PySide6") and importlib.util.find_spec("matplotlib"),
    "PySide6 and matplotlib are not installed",
)
class FitUiWorkflowTests(unittest.TestCase):
    def test_user_can_import_and_render_observed_calculated_fit_csv(self):
        script = r"""
import os
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication
from xrdviz.ui.main_window import MainWindow

app = QApplication([])
window = MainWindow()
assert hasattr(window, "open_fit_action")
assert hasattr(window, "uncertainty_mode_combo")
assert hasattr(window, "errorbar_stride_spin")
assert hasattr(window, "fit_components_check")
assert hasattr(window, "fit_background_check")
assert hasattr(window, "fit_metrics_check")

with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "fit.csv"
    path.write_text(
        "x,observed,calculated,sigma,background,component_alpha\n"
        "20,10,9,1,2,7\n"
        "21,20,19,1,2,17\n"
        "22,11,12,2,2,10\n",
        encoding="utf-8",
    )
    window.import_pattern_fit(path)
    assert window.state.fit is not None
    assert window.state.fit.components[0].name == "alpha"
    assert window.view_mode_combo.currentData() == "refinement"
    assert len(window.figure.axes) == 2
    assert "residual" in window.last_render_axes
    assert "Fit imported" in window.statusBar().currentMessage()

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
