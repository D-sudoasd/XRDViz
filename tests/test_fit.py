from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xrdviz.fit import FitComponent, PatternFit, load_pattern_fit, parse_pattern_fit_csv


class PatternFitImportTests(unittest.TestCase):
    def test_csv_import_exposes_components_difference_and_profile_residuals(self):
        fit = parse_pattern_fit_csv(
            "x,observed,calculated,sigma,background,component_alpha\n"
            "20,10,9,1,2,7\n"
            "21,20,21,2,2,19\n",
            name="sample",
        )

        self.assertEqual(fit.name, "sample")
        self.assertEqual(fit.axis_kind, "two_theta")
        self.assertEqual(fit.difference, [1.0, -1.0])
        self.assertEqual(fit.components[0].name, "alpha")
        self.assertAlmostEqual(fit.rp, 100.0 * 2.0 / 30.0)
        self.assertAlmostEqual(fit.rwp, 100.0 * math.sqrt(1.25 / 200.0))

    def test_x_header_infers_q_axis_and_path_loader_records_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "refined.csv"
            path.write_text("q,obs,calc\n1,4,3.5\n2,5,5.5\n", encoding="utf-8")
            fit = load_pattern_fit(path)

        self.assertEqual(fit.name, "refined")
        self.assertEqual(fit.axis_kind, "q")
        self.assertEqual(fit.source_path, str(path))
        self.assertIsNone(fit.rwp)

    def test_import_rejects_missing_columns_nonfinite_values_and_nonpositive_sigma(
        self,
    ):
        with self.assertRaisesRegex(ValueError, "calculated"):
            parse_pattern_fit_csv("x,observed\n1,2\n2,3\n")
        with self.assertRaisesRegex(ValueError, "finite"):
            parse_pattern_fit_csv("x,observed,calculated\n1,2,nan\n2,3,3\n")
        with self.assertRaisesRegex(ValueError, "positive"):
            parse_pattern_fit_csv("x,observed,calculated,sigma\n1,2,2,0\n2,3,3,1\n")

    def test_import_rejects_duplicate_normalized_and_semantic_columns(self):
        with self.assertRaisesRegex(ValueError, "duplicate header"):
            parse_pattern_fit_csv(
                "x,observed,OBSERVED,calculated\n1,2,2,2\n2,3,3,3\n"
            )
        with self.assertRaisesRegex(ValueError, "multiple observed"):
            parse_pattern_fit_csv(
                "x,observed,obs,calculated\n1,2,2,2\n2,3,3,3\n"
            )
        with self.assertRaisesRegex(ValueError, "multiple x-axis"):
            parse_pattern_fit_csv(
                "x,q,observed,calculated\n1,1,2,2\n2,2,3,3\n"
            )
        with self.assertRaisesRegex(ValueError, "component"):
            parse_pattern_fit_csv(
                "x,observed,calculated,component_alpha,peak_alpha\n"
                "1,2,2,1,1\n2,3,3,2,2\n"
            )

    def test_import_rejects_rows_with_extra_fields(self):
        with self.assertRaisesRegex(ValueError, "more fields"):
            parse_pattern_fit_csv(
                "x,observed,calculated\n1,2,2,unexpected\n2,3,3\n"
            )

    def test_import_validates_wavelength_metadata(self):
        fit = parse_pattern_fit_csv(
            "q,observed,calculated,wavelength_angstrom\n"
            "1,2,2,1.0\n2,3,3,1.0\n"
        )
        self.assertEqual(fit.wavelength_angstrom, 1.0)

        with self.assertRaisesRegex(ValueError, "constant"):
            parse_pattern_fit_csv(
                "q,observed,calculated,wavelength_angstrom\n"
                "1,2,2,1.0\n2,3,3,1.1\n"
            )
        with self.assertRaisesRegex(ValueError, "conflicts"):
            parse_pattern_fit_csv(
                "q,observed,calculated,wavelength_angstrom\n"
                "1,2,2,1.0\n2,3,3,1.0\n",
                wavelength_angstrom=1.5406,
            )

    def test_model_rejects_component_length_mismatch(self):
        with self.assertRaisesRegex(ValueError, "component"):
            PatternFit(
                name="fit",
                x=[1, 2],
                observed=[2, 3],
                calculated=[2, 3],
                components=[FitComponent(name="short", y=[1])],
            )

    def test_model_requires_monotonic_x_and_unique_component_names(self):
        with self.assertRaisesRegex(ValueError, "monotonic"):
            PatternFit(
                name="fit",
                x=[1, 2, 1.5],
                observed=[1, 2, 3],
                calculated=[1, 2, 3],
            )
        with self.assertRaisesRegex(ValueError, "unique"):
            PatternFit(
                name="fit",
                x=[1, 2],
                observed=[1, 2],
                calculated=[1, 2],
                components=[
                    FitComponent("Alpha", [1, 1]),
                    FitComponent("alpha", [0, 1]),
                ],
            )


if __name__ == "__main__":
    unittest.main()
