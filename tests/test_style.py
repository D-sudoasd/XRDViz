import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xrdviz.models import PlotSettings
from xrdviz.plot.style import NATURE_SINGLE_WIDTH_IN, nature_single_column


class NatureStyleTests(unittest.TestCase):
    def test_nature_single_column_uses_print_size_typography(self):
        settings = nature_single_column(PlotSettings())

        self.assertAlmostEqual(settings.figure_width_in, 89.0 / 25.4)
        self.assertEqual(settings.figure_width_in, NATURE_SINGLE_WIDTH_IN)
        self.assertLessEqual(settings.font_size, 7.0)
        self.assertGreaterEqual(settings.font_size, 5.0)
        self.assertLessEqual(settings.axis_label_size, 7.0)
        self.assertGreaterEqual(settings.axis_label_size, 5.0)
        self.assertLessEqual(settings.tick_label_size, 7.0)
        self.assertGreaterEqual(settings.tick_label_size, 5.0)
        self.assertLessEqual(settings.line_width, 1.0)
        self.assertGreaterEqual(settings.line_width, 0.25)
        self.assertEqual(settings.font_family, "Arial")


if __name__ == "__main__":
    unittest.main()
