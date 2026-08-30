from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xrdviz.peakfit import (
    PeakDecompositionResult,
    PeakFitCancelled,
    PeakSeed,
    PeakSummary,
    decomposition_to_pattern_fit,
    fit_peaks,
    guess_peak_seeds,
)


class PeakDecompositionTests(unittest.TestCase):
    def test_duplicate_seed_names_are_rejected_before_fitting(self):
        x = np.linspace(0.0, 4.0, 81)
        y = np.ones_like(x)

        with self.assertRaisesRegex(ValueError, "names must be unique"):
            fit_peaks(
                x,
                y,
                [
                    PeakSeed(1.0, 1.0, 0.3, name="Peak A"),
                    PeakSeed(2.0, 1.0, 0.3, name="peak a"),
                ],
                profile="gaussian",
            )

    def test_fit_peaks_supports_cooperative_cancellation(self):
        x = np.linspace(0.0, 4.0, 81)
        y = np.ones_like(x)
        checks = 0

        def cancel_after_first_check():
            nonlocal checks
            checks += 1
            return checks >= 2

        with self.assertRaises(PeakFitCancelled):
            fit_peaks(
                x,
                y,
                [PeakSeed(2.0, 1.0, 0.3)],
                profile="gaussian",
                cancel_check=cancel_after_first_check,
            )
        self.assertGreaterEqual(checks, 2)

    def test_gaussian_recovers_known_peak_and_linear_background(self):
        x = np.linspace(-3.0, 4.0, 351)
        true_center = 0.8
        true_amplitude = 7.0
        true_fwhm = 0.6
        baseline = 1.8 + 0.2 * x
        y = baseline + true_amplitude * np.exp(
            -4.0 * math.log(2.0) * ((x - true_center) / true_fwhm) ** 2
        )

        result = fit_peaks(
            x,
            y,
            [PeakSeed(center=0.72, amplitude=6.2, width=0.72)],
            profile="gaussian",
            baseline_order=1,
        )

        self.assertIsInstance(result, PeakDecompositionResult)
        self.assertTrue(result.converged)
        self.assertEqual(result.profile, "gaussian")
        self.assertEqual(result.baseline_order, 1)
        self.assertEqual(len(result.components), 1)
        self.assertEqual(result.fitted.shape, x.shape)
        self.assertTrue(
            np.allclose(result.fitted, result.baseline + result.components[0])
        )
        self.assertTrue(np.allclose(result.residual, y - result.fitted))

        summary = result.summaries[0]
        self.assertIsInstance(summary, PeakSummary)
        self.assertAlmostEqual(summary.center, true_center, delta=1.0e-3)
        self.assertAlmostEqual(summary.amplitude, true_amplitude, delta=1.0e-3)
        self.assertAlmostEqual(summary.fwhm, true_fwhm, delta=1.0e-3)
        expected_area = (
            true_amplitude
            * true_fwhm
            * math.sqrt(math.pi)
            / (2.0 * math.sqrt(math.log(2.0)))
        )
        self.assertAlmostEqual(summary.area, expected_area, delta=1.0e-3)
        self.assertLess(float(np.max(np.abs(result.residual))), 1.0e-8)

    def test_pseudo_voigt_recovers_two_known_peaks_in_sorted_order(self):
        x = np.linspace(0.0, 10.0, 501)
        baseline = np.full_like(x, 1.25)

        def pseudo_voigt(
            center: float, amplitude: float, fwhm: float, eta: float
        ) -> np.ndarray:
            z = (x - center) / fwhm
            gaussian = np.exp(-4.0 * math.log(2.0) * z**2)
            lorentzian = 1.0 / (1.0 + 4.0 * z**2)
            return amplitude * ((1.0 - eta) * gaussian + eta * lorentzian)

        y = (
            baseline
            + pseudo_voigt(2.7, 5.5, 0.55, 0.25)
            + pseudo_voigt(7.1, 3.2, 0.8, 0.75)
        )
        result = fit_peaks(
            x,
            y,
            [
                PeakSeed(center=2.62, amplitude=5.0, width=0.62, eta=0.4),
                PeakSeed(center=7.18, amplitude=3.0, width=0.72, eta=0.6),
            ],
            profile="pseudo_voigt",
            baseline_order=0,
        )

        self.assertTrue(result.converged)
        self.assertEqual(
            [summary.name for summary in result.summaries], ["peak_1", "peak_2"]
        )
        self.assertEqual(len(result.components), 2)
        self.assertLess(result.summaries[0].center, result.summaries[1].center)
        self.assertAlmostEqual(result.summaries[0].center, 2.7, delta=2.0e-3)
        self.assertAlmostEqual(result.summaries[1].center, 7.1, delta=2.0e-3)
        self.assertAlmostEqual(result.summaries[0].fwhm, 0.55, delta=2.0e-3)
        self.assertAlmostEqual(result.summaries[1].fwhm, 0.8, delta=2.0e-3)
        self.assertAlmostEqual(result.summaries[0].eta, 0.25, delta=2.0e-3)
        self.assertAlmostEqual(result.summaries[1].eta, 0.75, delta=2.0e-3)
        self.assertLess(float(np.max(np.abs(result.residual))), 1.0e-8)

    def test_lorentzian_area_uses_fitted_fwhm(self):
        x = np.linspace(-5.0, 5.0, 401)
        center = -1.1
        amplitude = 4.2
        fwhm = 0.9
        y = 0.7 + amplitude / (1.0 + 4.0 * ((x - center) / fwhm) ** 2)

        result = fit_peaks(
            x,
            y,
            [PeakSeed(center=-1.0, amplitude=4.0, width=1.0)],
            profile="lorentzian",
            baseline_order=0,
        )

        summary = result.summaries[0]
        self.assertTrue(result.success)
        self.assertAlmostEqual(summary.center, center, delta=1.0e-3)
        self.assertAlmostEqual(summary.fwhm, fwhm, delta=1.0e-3)
        self.assertAlmostEqual(
            summary.area, amplitude * math.pi * fwhm / 2.0, delta=1.0e-3
        )

    def test_validation_rejects_nonfinite_unsorted_and_invalid_width_inputs(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            fit_peaks([0.0, 1.0, 2.0], [1.0, math.nan, 1.0], [PeakSeed(1.0, 1.0, 0.5)])
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            fit_peaks([0.0, 1.0, 1.0], [1.0, 2.0, 1.0], [PeakSeed(0.8, 1.0, 0.5)])
        with self.assertRaisesRegex(ValueError, "positive"):
            PeakSeed(center=1.0, amplitude=1.0, width=0.0)
        with self.assertRaisesRegex(ValueError, "sorted"):
            fit_peaks(
                np.linspace(0.0, 4.0, 25),
                np.ones(25),
                [PeakSeed(3.0, 1.0, 0.5), PeakSeed(1.0, 1.0, 0.5)],
            )

    def test_validation_rejects_unsupported_profile_baseline_and_insufficient_data(
        self,
    ):
        x = np.linspace(0.0, 2.0, 4)
        with self.assertRaisesRegex(ValueError, "profile"):
            fit_peaks(x, np.ones_like(x), [PeakSeed(1.0, 1.0, 0.5)], profile="emg")
        with self.assertRaisesRegex(ValueError, "baseline_order"):
            fit_peaks(x, np.ones_like(x), [PeakSeed(1.0, 1.0, 0.5)], baseline_order=3)
        with self.assertRaisesRegex(ValueError, "insufficient"):
            fit_peaks(x, np.ones_like(x), [PeakSeed(1.0, 1.0, 0.5)], profile="gaussian")

    def test_peak_seed_guessing_and_pattern_fit_conversion_are_auditable(self):
        x = np.linspace(10.0, 30.0, 401)
        y = 2.0 + 8.0 * np.exp(-4.0 * math.log(2.0) * ((x - 15.0) / 0.8) ** 2)
        y += 5.0 * np.exp(-4.0 * math.log(2.0) * ((x - 24.0) / 1.2) ** 2)

        seeds = guess_peak_seeds(x, y, max_peaks=2)
        self.assertEqual(len(seeds), 2)
        self.assertAlmostEqual(seeds[0].center, 15.0, delta=0.1)
        self.assertAlmostEqual(seeds[1].center, 24.0, delta=0.1)

        result = fit_peaks(x, y, seeds, profile="gaussian", baseline_order=0)
        fit = decomposition_to_pattern_fit(
            result,
            name="sample peaks",
            source_path="sample.xy",
            wavelength_angstrom=1.0,
        )
        self.assertEqual(fit.fit_kind, "peak_decomposition")
        self.assertTrue(fit.converged)
        self.assertEqual(len(fit.components), 2)
        self.assertIsNotNone(fit.components[0].fwhm)
        self.assertEqual(fit.source_path, "sample.xy")
        self.assertEqual(fit.wavelength_angstrom, 1.0)


if __name__ == "__main__":
    unittest.main()
