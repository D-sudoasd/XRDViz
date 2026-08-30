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
    importlib.util.find_spec("PySide6")
    and importlib.util.find_spec("matplotlib")
    and importlib.util.find_spec("scipy"),
    "PySide6, matplotlib, and scipy are required",
)
class AnalysisUiWorkflowTests(unittest.TestCase):
    def test_advanced_workflows_are_user_operable_and_rendered(self) -> None:
        script = r"""
import math
import os
import tempfile
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication
from xrdviz.ui.main_window import MainWindow

app = QApplication([])
window = MainWindow()
for attribute in (
    "peak_decomposition_action",
    "detector_action",
    "rsm_action",
    "pole_figure_action",
    "peak_width_action",
    "rocking_curve_action",
    "annotation_action",
    "analysis_summary_label",
    "small_multiples_columns_spin",
    "inset_check",
):
    assert hasattr(window, attribute), attribute

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    x = np.linspace(20.0, 30.0, 301)
    y = 2.0 + 7.0 * np.exp(-4.0 * math.log(2.0) * ((x - 24.0) / 0.7) ** 2)
    spectrum = root / "sample.xy"
    spectrum.write_text("\n".join(f"{xx:.8g} {yy:.8g}" for xx, yy in zip(x, y)), encoding="utf-8")
    window.add_files([spectrum])
    result = window.decompose_spectrum(centers=[24.0], profile="gaussian", baseline_order=0)
    assert result.converged
    assert window.state.fit.fit_kind == "peak_decomposition"
    assert window.view_mode_combo.currentData() == "refinement"
    assert "residual" in window.last_render_axes

    image = np.arange(81, dtype=float).reshape(9, 9) + 1.0
    detector = root / "detector.npy"
    np.save(detector, image)
    raw = window.import_detector_image(detector, mode="raw")
    assert raw.kind == "detector"
    assert window.view_mode_combo.currentData() == "map"
    assert "map" in window.last_render_axes

    radial = window.import_detector_image(
        detector,
        mode="radial",
        center=(4.0, 4.0),
        pixel_size=(0.1, 0.1),
        distance=100.0,
        wavelength=1.5406,
        radial_unit="two_theta",
        radial_bins=16,
    )
    assert radial.axis_kind == "two_theta"
    assert len(radial.x) >= 2
    assert radial.wavelength_angstrom == 1.5406

    cake = window.import_detector_image(
        detector,
        mode="cake",
        center=(4.0, 4.0),
        pixel_size=(0.1, 0.1),
        distance=100.0,
        wavelength=1.5406,
        radial_bins=12,
        chi_bins=12,
    )
    assert cake.kind == "cake"
    assert cake.counts is not None

    rsm_path = root / "rsm.csv"
    rsm_path.write_text(
        "qx,qz,intensity\n0,0,1\n1,0,2\n0,1,3\n1,1,4\n",
        encoding="utf-8",
    )
    rsm = window.import_map_csv(rsm_path, kind="rsm")
    assert rsm.kind == "rsm"
    assert window.state.map_data.kind == "rsm"

    pole_path = root / "pole.csv"
    pole_path.write_text(
        "phi,chi,intensity\n0,0,1\n90,0,2\n0,45,3\n90,45,4\n",
        encoding="utf-8",
    )
    pole = window.import_map_csv(pole_path, kind="pole_figure")
    assert pole.kind == "pole_figure"
    assert window.last_render_axes["main"].name == "polar"

    peaks_path = root / "peaks.csv"
    peaks_path.write_text(
        "2theta,FWHM,hkl\n20,0.20,111\n30,0.25,200\n40,0.30,220\n",
        encoding="utf-8",
    )
    scherrer = window.import_peak_measurements(
        peaks_path,
        kind="scherrer",
        wavelength=0.15406,
        wavelength_unit="nm",
        output_unit="nm",
    )
    assert scherrer.kind == "scherrer"
    assert window.view_mode_combo.currentData() == "derived"
    assert "scatter" in window.last_render_axes

    rocking_path = root / "rocking.csv"
    rocking_path.write_text(
        "omega,intensity\n-2,0\n-1,0.25\n0,1\n1,0.25\n2,0\n",
        encoding="utf-8",
    )
    rocking = window.import_rocking_curve(rocking_path)
    assert rocking.kind == "rocking_curve"
    assert "curve" in window.last_render_axes

    annotation = window.add_annotation(0.0, "omega peak")
    assert annotation.text == "omega peak"
    assert "annotations" in window.last_render_axes
    assert "Derived:" in window.analysis_summary_label.text()

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
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)

    def test_peak_decomposition_action_uses_worker_and_exposes_cancel(self) -> None:
        script = r"""
import os
import tempfile
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication, QDialog
from xrdviz.ui import advanced_workflows
from xrdviz.ui.main_window import MainWindow

app = QApplication([])
window = MainWindow()

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    x = np.linspace(20.0, 30.0, 101)
    y = 2.0 + 7.0 * np.exp(-4.0 * np.log(2.0) * ((x - 24.0) / 0.7) ** 2)
    spectrum = root / "sample.xy"
    spectrum.write_text(
        "\n".join(f"{xx:.8g} {yy:.8g}" for xx, yy in zip(x, y)),
        encoding="utf-8",
    )
    window.add_files([spectrum])

    class AcceptedDialog:
        def __init__(self, parent=None):
            self.parent = parent

        def exec(self):
            return QDialog.Accepted

        def parameters(self):
            return {
                "centers": [24.0],
                "profile": "gaussian",
                "baseline_order": 0,
                "max_peaks": 1,
            }

    advanced_workflows.PeakDecompositionDialog = AcceptedDialog
    window.peak_decomposition_action.trigger()
    deadline = time.monotonic() + 10.0
    while getattr(window, "_peak_fit_thread", None) is not None:
        app.processEvents()
        assert time.monotonic() < deadline, "peak-fit worker did not finish"
        time.sleep(0.01)
    assert window.state.fit is not None
    assert window.state.fit.fit_kind == "peak_decomposition"

    class SlowDialog(AcceptedDialog):
        pass

    advanced_workflows.PeakDecompositionDialog = SlowDialog
    window.decompose_spectrum_async(
        centers=[24.0], profile="gaussian", baseline_order=0, max_nfev=100
    )
    assert window.cancel_peak_decomposition()
    deadline = time.monotonic() + 10.0
    while getattr(window, "_peak_fit_thread", None) is not None:
        app.processEvents()
        assert time.monotonic() < deadline, "cancelled peak-fit worker did not finish"
        time.sleep(0.01)
    assert "cancel" in window.statusBar().currentMessage().lower()

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
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)

    def test_running_peak_worker_can_be_cancelled_and_waited_on_shutdown(self) -> None:
        script = r"""
import os
import tempfile
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication
from xrdviz import peakfit
from xrdviz.ui.main_window import MainWindow

app = QApplication([])
window = MainWindow()

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    x = np.linspace(20.0, 30.0, 101)
    y = 2.0 + 7.0 * np.exp(-4.0 * np.log(2.0) * ((x - 24.0) / 0.7) ** 2)
    spectrum = root / "sample.xy"
    spectrum.write_text(
        "\n".join(f"{xx:.8g} {yy:.8g}" for xx, yy in zip(x, y)),
        encoding="utf-8",
    )
    window.add_files([spectrum])

    original_fit_peaks = peakfit.fit_peaks

    def wait_for_cancel(*args, cancel_check=None, **kwargs):
        while cancel_check is None or not cancel_check():
            time.sleep(0.005)
        raise peakfit.PeakFitCancelled("cancelled by shutdown")

    peakfit.fit_peaks = wait_for_cancel
    try:
        window.decompose_spectrum_async(
            centers=[24.0], profile="gaussian", baseline_order=0
        )
        deadline = time.monotonic() + 3.0
        while (
            getattr(window, "_peak_fit_thread", None) is None
            or not window._peak_fit_thread.isRunning()
        ):
            app.processEvents()
            assert time.monotonic() < deadline, "peak-fit worker did not start"
            time.sleep(0.01)
        assert window.close()
        assert getattr(window, "_peak_fit_thread", None) is None
    finally:
        peakfit.fit_peaks = original_fit_peaks

if window.isVisible():
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
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)


if __name__ == "__main__":
    unittest.main()
