import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xrdviz.models import PlotSettings, ProjectState, SpectrumLayer


class BatchXrdTests(unittest.TestCase):
    def test_filename_metadata_detects_frame_time_and_temperature(self):
        from xrdviz.batch import parse_spectrum_metadata

        meta = parse_spectrum_metadata(Path("run_A/scan_0007_12.5min_650C.xy"))

        self.assertEqual(meta.frame_index, 7)
        self.assertAlmostEqual(meta.time_s, 750.0)
        self.assertEqual(meta.temperature, 650.0)
        self.assertEqual(meta.temperature_unit, "C")

        az_meta = parse_spectrum_metadata(Path("Az_Full_000123.txt"))
        self.assertEqual(az_meta.frame_index, 123)

    def test_batch_metadata_sorts_layers_and_assigns_gradient_values(self):
        from xrdviz.batch import apply_batch_metadata

        layers = [
            SpectrumLayer(name="scan_002", x=[1.0, 2.0], y=[10.0, 20.0], source_path="scan_0002_350C.xy"),
            SpectrumLayer(name="scan_001", x=[1.0, 2.0], y=[5.0, 15.0], source_path="scan_0001_250C.xy"),
        ]

        apply_batch_metadata(layers, sort_by="temperature", color_by="frame", colormap="viridis")

        self.assertEqual([layer.name for layer in layers], ["scan_001", "scan_002"])
        self.assertEqual([layer.order for layer in layers], [0, 1])
        self.assertEqual([layer.frame_index for layer in layers], [1, 2])
        self.assertEqual([layer.temperature for layer in layers], [250.0, 350.0])
        self.assertEqual([layer.color_value for layer in layers], [1.0, 2.0])
        self.assertTrue(all(layer.color.startswith("#") for layer in layers))

    def test_batch_metadata_preserves_user_assigned_colors(self):
        from xrdviz.batch import apply_batch_metadata

        layers = [
            SpectrumLayer(name="scan_002", x=[1.0, 2.0], y=[10.0, 20.0], source_path="scan_0002.xy", color="#111111"),
            SpectrumLayer(name="scan_001", x=[1.0, 2.0], y=[5.0, 15.0], source_path="scan_0001.xy", color="#222222"),
        ]

        apply_batch_metadata(layers, sort_by="frame", color_by="frame", colormap="viridis")

        self.assertEqual([layer.name for layer in layers], ["scan_001", "scan_002"])
        self.assertEqual([layer.color for layer in layers], ["#222222", "#111111"])
        self.assertEqual([layer.color_value for layer in layers], [1.0, 2.0])

    def test_heatmap_matrix_uses_common_grid_and_transformed_intensity(self):
        from xrdviz.batch import make_heatmap_matrix

        state = ProjectState(
            spectra=[
                SpectrumLayer(name="f2", x=[10.0, 20.0, 30.0], y=[0.0, 10.0, 20.0], frame_index=2, order=1),
                SpectrumLayer(name="f1", x=[10.0, 20.0, 30.0], y=[5.0, 10.0, 15.0], frame_index=1, order=0),
            ],
            settings=PlotSettings(view_mode="heatmap", heatmap_points=5, x_min=10.0, x_max=30.0, normalize=True),
        )

        x_grid, row_values, matrix = make_heatmap_matrix(state)

        self.assertEqual(len(x_grid), 5)
        self.assertEqual(list(row_values), [1.0, 2.0])
        self.assertEqual(matrix.shape, (2, 5))
        self.assertLessEqual(float(matrix.max()), 1.0)
        self.assertAlmostEqual(float(matrix[0, -1]), 1.0)


if __name__ == "__main__":
    unittest.main()
