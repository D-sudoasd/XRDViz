import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xrdviz.models import PlotSettings, SpectrumLayer
from xrdviz.transforms import display_y_for_layer, transform_intensity


class DisplayTransformTests(unittest.TestCase):
    def test_normalize_log_and_offset_do_not_mutate_raw_values(self):
        raw = [10.0, 100.0, 1000.0]

        transformed = transform_intensity(raw, normalize=True, log_scale=True, epsilon=1e-6, vertical_offset=2.0)

        self.assertEqual(raw, [10.0, 100.0, 1000.0])
        self.assertAlmostEqual(transformed[0], -2.0 + 2.0, places=5)
        self.assertAlmostEqual(transformed[-1], 0.0 + 2.0, places=5)

    def test_display_layer_adds_automatic_stack_offset(self):
        layer = SpectrumLayer(name="a", x=[1.0, 2.0], y=[5.0, 10.0])
        settings = PlotSettings(normalize=True, stack_enabled=True, stack_spacing=0.75)

        y = display_y_for_layer(layer, settings, layer_index=2)

        self.assertEqual(y, [2.0, 2.5])

    def test_non_positive_log_values_are_clamped_to_epsilon(self):
        y = transform_intensity([-1.0, 0.0, 10.0], normalize=False, log_scale=True, epsilon=0.01)

        self.assertEqual(y[0], -2.0)
        self.assertEqual(y[1], -2.0)
        self.assertEqual(y[2], 1.0)


if __name__ == "__main__":
    unittest.main()
