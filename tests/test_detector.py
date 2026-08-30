from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys
import importlib.util

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xrdviz.detector import (
    CakeResult,
    DetectorGeometry,
    MAX_CAKE_CELLS,
    MAX_DETECTOR_PIXELS,
    RadialIntegrationResult,
    generate_cake,
    integrate_radial,
    load_detector_image,
)


class DetectorGeometryTests(unittest.TestCase):
    def test_geometry_accepts_pair_form_and_exposes_scalar_values(self):
        geometry = DetectorGeometry(
            center=(2.0, 3.0),
            pixel_size=(0.1, 0.2),
            distance=100.0,
            wavelength=0.15406,
        )

        self.assertEqual((geometry.center_x, geometry.center_y), (2.0, 3.0))
        self.assertEqual((geometry.pixel_size_x, geometry.pixel_size_y), (0.1, 0.2))
        self.assertEqual(geometry.center, (2.0, 3.0))
        self.assertEqual(geometry.pixel_size, (0.1, 0.2))

    def test_geometry_rejects_nonfinite_or_nonpositive_parameters(self):
        valid = dict(
            center=(2.0, 3.0), pixel_size=(0.1, 0.2), distance=100.0, wavelength=0.15406
        )
        for field, value in (
            ("center", (np.nan, 3.0)),
            ("pixel_size", (0.1, 0.0)),
            ("distance", np.inf),
            ("wavelength", -1.0),
        ):
            arguments = dict(valid)
            arguments[field] = value
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    DetectorGeometry(**arguments)


class DetectorIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.geometry = DetectorGeometry(
            center=(2.0, 2.0),
            pixel_size=(1.0, 1.0),
            distance=10.0,
            wavelength=1.0,
        )

    def test_radial_integration_returns_centres_means_and_counts(self):
        image = np.ones((5, 5), dtype=float)

        result = integrate_radial(
            image,
            self.geometry,
            unit="2theta",
            bin_edges=[0.0, 4.0, 10.0, 20.0],
        )

        self.assertIsInstance(result, RadialIntegrationResult)
        np.testing.assert_array_equal(result.counts, [1, 8, 16])
        np.testing.assert_allclose(result.intensity, [1.0, 1.0, 1.0])
        self.assertEqual(result.unit, "two_theta")
        self.assertEqual(len(result.bin_centers), 3)

    def test_radial_integration_applies_mask_and_supports_q_and_d_units(self):
        image = np.arange(9, dtype=float).reshape(3, 3) + 1.0
        mask = np.zeros_like(image, dtype=bool)
        mask[0, 0] = True

        q_result = integrate_radial(image, self.geometry, unit="q", n_bins=4, mask=mask)
        d_result = integrate_radial(image, self.geometry, unit="d", n_bins=4, mask=mask)

        self.assertEqual(int(q_result.counts.sum()), 8)
        # d-spacing is undefined at the direct-beam centre (theta=0), so that
        # unmasked centre pixel is intentionally omitted from d integration.
        self.assertEqual(int(d_result.counts.sum()), 7)
        self.assertTrue(np.all(np.isfinite(q_result.bin_centers)))
        self.assertTrue(np.all(np.isfinite(d_result.bin_centers)))
        self.assertTrue(np.all(np.isfinite(d_result.intensity[d_result.counts > 0])))

    def test_radial_integration_rejects_invalid_image_mask_and_geometry_shape(self):
        cases = (
            (np.ones(9), None),
            (np.array([[1.0, np.nan], [2.0, 3.0]]), None),
            (np.ones((2, 2)), np.zeros((3, 3), dtype=bool)),
        )
        for image, mask in cases:
            with self.subTest(image_shape=image.shape):
                with self.assertRaises(ValueError):
                    integrate_radial(image, self.geometry, n_bins=2, mask=mask)

        outside = DetectorGeometry(
            center=(20.0, 2.0),
            pixel_size=(1.0, 1.0),
            distance=10.0,
            wavelength=1.0,
        )
        with self.assertRaises(ValueError):
            integrate_radial(np.ones((5, 5)), outside, n_bins=2)

    def test_radial_integration_rejects_edges_and_range_before_binning(self):
        with self.assertRaises(ValueError):
            integrate_radial(
                np.ones((5, 5)),
                self.geometry,
                bin_edges=[0.0, 1.0],
                radial_range=(0.0, 1.0),
            )


class CakeAndImageLoadingTests(unittest.TestCase):
    def test_generate_cake_returns_chi_by_two_theta_matrix(self):
        image = np.arange(25, dtype=float).reshape(5, 5) + 1.0
        geometry = DetectorGeometry(
            center=(2.0, 2.0),
            pixel_size=(1.0, 1.0),
            distance=10.0,
            wavelength=1.0,
        )

        result = generate_cake(image, geometry, n_two_theta=6, n_chi=8)

        self.assertIsInstance(result, CakeResult)
        self.assertEqual(result.intensity.shape, (8, 6))
        self.assertEqual(result.counts.shape, (8, 6))
        self.assertEqual(result.two_theta.shape, (6,))
        self.assertEqual(result.chi.shape, (8,))
        self.assertEqual(int(result.counts.sum()), 25)
        self.assertTrue(np.all(np.isfinite(result.intensity[result.counts > 0])))

    def test_load_detector_image_supports_npy_without_extra_dependencies(self):
        image = np.arange(12, dtype=np.float32).reshape(3, 4)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "detector.npy"
            np.save(path, image)
            loaded = load_detector_image(path)

        np.testing.assert_array_equal(loaded, image)
        self.assertEqual(loaded.dtype, image.dtype)

    def test_load_detector_image_supports_single_array_npz_and_rejects_multiple(self):
        image = np.arange(12, dtype=np.float32).reshape(3, 4)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            single_path = root / "detector.npz"
            np.savez(single_path, detector=image)
            loaded = load_detector_image(single_path)
            np.testing.assert_array_equal(loaded, image)
            self.assertEqual(loaded.dtype, image.dtype)

            multiple_path = root / "multiple.npz"
            np.savez(multiple_path, first=image, second=image)
            with self.assertRaisesRegex(ValueError, "exactly one array"):
                load_detector_image(multiple_path)

    def test_load_detector_image_supports_grayscale_rgb_and_rgba(self):
        from PIL import Image

        grayscale = np.array([[0, 64], [128, 255]], dtype=np.uint8)
        rgb = np.array(
            [
                [[255, 0, 0], [0, 255, 0]],
                [[0, 0, 255], [255, 255, 255]],
            ],
            dtype=np.uint8,
        )
        rgba = np.concatenate(
            [rgb, np.array([[[1], [64]], [[128], [255]]], dtype=np.uint8)],
            axis=-1,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cases = (
                ("gray.png", Image.fromarray(grayscale, mode="L"), grayscale),
                (
                    "rgb.png",
                    Image.fromarray(rgb, mode="RGB"),
                    0.2126 * rgb[..., 0]
                    + 0.7152 * rgb[..., 1]
                    + 0.0722 * rgb[..., 2],
                ),
                (
                    "rgba.png",
                    Image.fromarray(rgba, mode="RGBA"),
                    0.2126 * rgb[..., 0]
                    + 0.7152 * rgb[..., 1]
                    + 0.0722 * rgb[..., 2],
                ),
            )
            for name, image, expected in cases:
                with self.subTest(name=name):
                    path = root / name
                    image.save(path)
                    loaded = load_detector_image(path)
                    np.testing.assert_allclose(loaded, expected)
                    self.assertEqual(loaded.shape, expected.shape)

    def test_generate_cake_wraps_full_chi_intervals_relative_to_lower_bound(self):
        image = np.ones((21, 21), dtype=float)
        geometry = DetectorGeometry(
            center=(10.0, 10.0),
            pixel_size=(1.0, 1.0),
            distance=100.0,
            wavelength=1.0,
        )

        for chi_range in ((-180.0, 180.0), (0.0, 360.0), (-90.0, 270.0), (30.0, 390.0)):
            with self.subTest(chi_range=chi_range):
                result = generate_cake(
                    image,
                    geometry,
                    n_two_theta=8,
                    n_chi=16,
                    chi_range=chi_range,
                )
                self.assertEqual(int(result.counts.sum()), image.size)

    def test_generate_cake_rejects_edges_and_ranges_together(self):
        image = np.ones((5, 5), dtype=float)
        geometry = DetectorGeometry(
            center=(2.0, 2.0),
            pixel_size=(1.0, 1.0),
            distance=10.0,
            wavelength=1.0,
        )

        with self.assertRaises(ValueError):
            generate_cake(
                image,
                geometry,
                two_theta_edges=[0.0, 1.0],
                two_theta_range=(0.0, 1.0),
            )
        with self.assertRaises(ValueError):
            generate_cake(
                image,
                geometry,
                chi_edges=[-180.0, 180.0],
                chi_range=(-180.0, 180.0),
            )

    def test_generate_cake_rejects_excessive_output_budget(self):
        image = np.ones((2, 2), dtype=float)
        geometry = DetectorGeometry(
            center=(0.5, 0.5),
            pixel_size=(1.0, 1.0),
            distance=10.0,
            wavelength=1.0,
        )

        with self.assertRaises(ValueError):
            generate_cake(
                image,
                geometry,
                n_two_theta=MAX_CAKE_CELLS // 8 + 1,
                n_chi=8,
            )

    def test_non_asymmetric_geometry_maps_x_y_to_all_public_coordinates(self):
        image = np.zeros((6, 7), dtype=float)
        image[3, 2] = 1.0
        geometry = DetectorGeometry(
            center=(1.0, 2.0),
            pixel_size=(2.0, 3.0),
            distance=10.0,
            wavelength=1.0,
        )
        radius = float(np.hypot(2.0, 3.0))
        two_theta = float(np.degrees(np.arctan2(radius, 10.0)))
        q_value = float(4.0 * np.pi * np.sin(np.radians(two_theta * 0.5)))
        d_value = float(1.0 / (2.0 * np.sin(np.radians(two_theta * 0.5))))
        mask = np.ones_like(image, dtype=bool)
        mask[3, 2] = False

        for unit, expected in (
            ("two_theta", two_theta),
            ("q", q_value),
            ("d", d_value),
        ):
            with self.subTest(unit=unit):
                result = integrate_radial(
                    image,
                    geometry,
                    unit=unit,
                    n_bins=1,
                    radial_range=(expected - 1.0e-8, expected + 1.0e-8),
                    mask=mask,
                )
                self.assertEqual(int(result.counts.sum()), 1)

        chi = float(np.degrees(np.arctan2(3.0, 2.0)))
        cake = generate_cake(
            image,
            geometry,
            two_theta_edges=[two_theta - 1.0e-8, two_theta + 1.0e-8],
            chi_edges=[chi - 1.0e-8, chi + 1.0e-8],
            mask=mask,
        )
        self.assertEqual(int(cake.counts.sum()), 1)

    def test_load_detector_image_rejects_oversized_npy_from_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "too_large.npy"
            dtype = np.dtype(np.uint8)
            shape = (MAX_DETECTOR_PIXELS + 1, 1)
            with path.open("wb") as handle:
                handle.write(b"\x93NUMPY\x02\x00")
                np.lib.format.write_array_header_2_0(
                    handle,
                    {
                        "descr": np.lib.format.dtype_to_descr(dtype),
                        "fortran_order": False,
                        "shape": shape,
                    },
                )
            with self.assertRaises(ValueError):
                load_detector_image(path)


@unittest.skipUnless(
    importlib.util.find_spec("PySide6"),
    "PySide6 is required for detector dialog tests",
)
class DetectorImportDialogTests(unittest.TestCase):
    def test_dialog_preserves_high_resolution_radial_mode_and_couples_cake_axes(self):
        from PySide6.QtWidgets import QApplication

        from xrdviz.ui.analysis_dialogs import DetectorImportDialog

        app = QApplication.instance() or QApplication([])
        dialog = DetectorImportDialog((100, 100))
        self.assertEqual(dialog.mode.currentData(), "raw")
        dialog.mode.setCurrentIndex(dialog.mode.findData("radial"))
        self.assertGreaterEqual(dialog.radial_bins.maximum(), 100_000)
        dialog.radial_bins.setValue(100_000)
        self.assertEqual(dialog.parameters()["radial_bins"], 100_000)

        dialog.mode.setCurrentIndex(dialog.mode.findData("cake"))
        self.assertLessEqual(
            dialog.radial_bins.maximum() * dialog.chi_bins.value(), MAX_CAKE_CELLS
        )
        self.assertLessEqual(
            dialog.chi_bins.maximum() * dialog.radial_bins.value(), MAX_CAKE_CELLS
        )
        self.assertLessEqual(
            dialog.radial_bins.value() * dialog.chi_bins.value(), MAX_CAKE_CELLS
        )
        dialog.chi_bins.setValue(64)
        dialog.radial_bins.setValue(50_000)
        parameters = dialog.parameters()
        self.assertEqual(parameters["mode"], "cake")
        self.assertLessEqual(
            parameters["radial_bins"] * parameters["chi_bins"], MAX_CAKE_CELLS
        )
        dialog.close()
        app.processEvents()


if __name__ == "__main__":
    unittest.main()
