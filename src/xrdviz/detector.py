"""Small, dependency-free helpers for converting detector images to XRD views.

The routines in this module intentionally implement the geometry and binning
needed for a flat-area-detector preview.  They do not attempt detector
distortion correction, polarization correction, or an instrument refinement.
Coordinates use the usual image convention: ``x`` is the column coordinate
and ``y`` is the row coordinate. Pixel size and detector distance must share
one length unit because their ratio defines the scattering angle. Wavelength
may use a separate declared convention: returned ``q`` uses its reciprocal
unit and returned ``d`` uses its unit (for example, millimetres for geometry
and angstrom for wavelength is valid).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np


# Resource limits are deliberately conservative for an interactive desktop
# workflow.  A cake result contains one float64 intensity and one int64 count
# per cell; the implementation also needs temporary histogram arrays while it
# computes the means.  Callers can still request smaller output explicitly.
MAX_CAKE_CELLS = 4_000_000
MAX_CAKE_RESULT_BYTES = 128 * 1024 * 1024
MAX_CAKE_WORKING_BYTES = 256 * 1024 * 1024

# A detector image is materialized at least once by the public API and usually
# converted to float64 for binning.  Rejecting oversized metadata before that
# conversion prevents a malformed file or accidental UI selection from
# allocating an unbounded array.
MAX_DETECTOR_PIXELS = 16_000_000
MAX_DETECTOR_BYTES = 256 * 1024 * 1024


_UNIT_ALIASES = {
    "2theta": "two_theta",
    "2-theta": "two_theta",
    "2 theta": "two_theta",
    "2_theta": "two_theta",
    "2θ": "two_theta",
    "angle": "two_theta",
    "degrees": "two_theta",
    "degree": "two_theta",
    "deg": "two_theta",
    "two-theta": "two_theta",
    "two theta": "two_theta",
    "two_theta": "two_theta",
    "q": "q",
    "q-space": "q",
    "q space": "q",
    "q_space": "q",
    "d": "d",
    "d-spacing": "d",
    "d spacing": "d",
    "d_spacing": "d",
}


def normalize_detector_unit(unit: str) -> str:
    """Return the canonical radial coordinate name.

    ``two_theta`` is measured in degrees.  ``q`` and ``d`` inherit the
    reciprocal/length units implied by the geometry's wavelength.
    """

    if not isinstance(unit, str):
        raise ValueError("radial unit must be a string")
    key = unit.strip().lower().replace("°", "θ")
    normalized = _UNIT_ALIASES.get(key)
    if normalized is None:
        raise ValueError(f"Unsupported detector radial unit: {unit!r}")
    return normalized


def _finite_float(value: object, name: str, *, positive: bool = False) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    if positive and converted <= 0.0:
        raise ValueError(f"{name} must be positive")
    return converted


def _finite_pair(value: object, name: str) -> tuple[float, float]:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must contain exactly two finite numbers")
    try:
        values = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError(f"{name} must contain exactly two finite numbers") from exc
    if len(values) != 2:
        raise ValueError(f"{name} must contain exactly two finite numbers")
    return (
        _finite_float(values[0], f"{name}[0]"),
        _finite_float(values[1], f"{name}[1]"),
    )


@dataclass(frozen=True, slots=True, init=False)
class DetectorGeometry:
    """Geometry for a planar detector image.

    The preferred constructor is::

        DetectorGeometry(
            center=(cx, cy), pixel_size=(sx, sy),
            distance=distance, wavelength=wavelength,
        )

    ``center`` is in pixel coordinates, with ``(0, 0)`` at the top-left
    pixel.  For convenience, six scalar positional arguments and the six
    scalar keyword arguments ``center_x``, ``center_y``, ``pixel_size_x`` and
    ``pixel_size_y`` are also accepted.
    """

    center_x: float
    center_y: float
    pixel_size_x: float
    pixel_size_y: float
    distance: float
    wavelength: float

    def __init__(
        self,
        *args: object,
        center: object | None = None,
        pixel_size: object | None = None,
        distance: object | None = None,
        wavelength: object | None = None,
        center_x: object | None = None,
        center_y: object | None = None,
        pixel_size_x: object | None = None,
        pixel_size_y: object | None = None,
    ) -> None:
        """Create and validate a detector geometry.

        Accepted positional forms are ``(center, pixel_size, distance,
        wavelength)`` and ``(center_x, center_y, pixel_size_x, pixel_size_y,
        distance, wavelength)``.
        """

        if len(args) == 6:
            if any(
                value is not None
                for value in (
                    center,
                    pixel_size,
                    distance,
                    wavelength,
                    center_x,
                    center_y,
                    pixel_size_x,
                    pixel_size_y,
                )
            ):
                raise TypeError(
                    "Do not mix six positional geometry values with keyword values"
                )
            center_x, center_y, pixel_size_x, pixel_size_y, distance, wavelength = args
        elif len(args) == 4:
            # Four scalar values plus distance/wavelength keywords are a
            # useful spelling of the scalar form.  Otherwise interpret the
            # four values as (center, pixel_size, distance, wavelength).
            if distance is not None or wavelength is not None:
                if any(
                    value is not None
                    for value in (
                        center,
                        pixel_size,
                        center_x,
                        center_y,
                        pixel_size_x,
                        pixel_size_y,
                    )
                ):
                    raise TypeError("Ambiguous positional and keyword geometry values")
                center_x, center_y, pixel_size_x, pixel_size_y = args
            else:
                if any(
                    value is not None
                    for value in (
                        center,
                        pixel_size,
                        distance,
                        wavelength,
                        center_x,
                        center_y,
                        pixel_size_x,
                        pixel_size_y,
                    )
                ):
                    raise TypeError(
                        "Do not mix pair-form positional geometry values with keyword values"
                    )
                center, pixel_size, distance, wavelength = args
        elif args:
            raise TypeError("DetectorGeometry expects four or six positional values")

        if center is not None:
            if center_x is not None or center_y is not None:
                raise TypeError("Specify either center or center_x/center_y")
            resolved_center_x, resolved_center_y = _finite_pair(center, "center")
        else:
            if center_x is None or center_y is None:
                raise TypeError("center or both center_x and center_y are required")
            resolved_center_x = _finite_float(center_x, "center_x")
            resolved_center_y = _finite_float(center_y, "center_y")

        if pixel_size is not None:
            if pixel_size_x is not None or pixel_size_y is not None:
                raise TypeError(
                    "Specify either pixel_size or pixel_size_x/pixel_size_y"
                )
            resolved_pixel_x, resolved_pixel_y = _finite_pair(pixel_size, "pixel_size")
        else:
            if pixel_size_x is None or pixel_size_y is None:
                raise TypeError(
                    "pixel_size or both pixel_size_x and pixel_size_y are required"
                )
            resolved_pixel_x = _finite_float(pixel_size_x, "pixel_size_x")
            resolved_pixel_y = _finite_float(pixel_size_y, "pixel_size_y")

        if resolved_pixel_x <= 0.0 or resolved_pixel_y <= 0.0:
            raise ValueError("pixel sizes must be positive")
        if distance is None:
            raise TypeError("distance is required")
        if wavelength is None:
            raise TypeError("wavelength is required")

        resolved_distance = _finite_float(distance, "distance", positive=True)
        resolved_wavelength = _finite_float(wavelength, "wavelength", positive=True)

        object.__setattr__(self, "center_x", resolved_center_x)
        object.__setattr__(self, "center_y", resolved_center_y)
        object.__setattr__(self, "pixel_size_x", resolved_pixel_x)
        object.__setattr__(self, "pixel_size_y", resolved_pixel_y)
        object.__setattr__(self, "distance", resolved_distance)
        object.__setattr__(self, "wavelength", resolved_wavelength)

    @property
    def center(self) -> tuple[float, float]:
        """Beam centre as ``(x, y)`` pixel coordinates."""

        return (self.center_x, self.center_y)

    @property
    def pixel_size(self) -> tuple[float, float]:
        """Pixel size as ``(x, y)`` lengths."""

        return (self.pixel_size_x, self.pixel_size_y)

    def validate_for_shape(self, shape: Sequence[int]) -> None:
        """Validate that the beam centre lies inside a 2-D image shape."""

        try:
            dimensions = tuple(shape)
        except TypeError as exc:
            raise ValueError("detector image shape must have two dimensions") from exc
        if len(dimensions) != 2:
            raise ValueError("detector image must be two-dimensional")
        try:
            height, width = (int(dimensions[0]), int(dimensions[1]))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "detector image shape must contain positive integers"
            ) from exc
        if height <= 0 or width <= 0:
            raise ValueError("detector image shape must contain positive integers")
        if not (0.0 <= self.center_x < width and 0.0 <= self.center_y < height):
            raise ValueError(
                "detector centre must lie inside the image: "
                f"got ({self.center_x}, {self.center_y}) for shape {(height, width)}"
            )

    # Method spellings keep the class convenient at the interactive prompt,
    # while the module-level functions remain the stable public seam.
    def radial_integrate(self, image: Any, **kwargs: Any) -> "RadialIntegrationResult":
        return integrate_radial(image, self, **kwargs)

    def integrate_radial(self, image: Any, **kwargs: Any) -> "RadialIntegrationResult":
        return integrate_radial(image, self, **kwargs)

    def generate_cake(self, image: Any, **kwargs: Any) -> "CakeResult":
        return generate_cake(image, self, **kwargs)

    def cake(self, image: Any, **kwargs: Any) -> "CakeResult":
        return generate_cake(image, self, **kwargs)


@dataclass(frozen=True, slots=True)
class RadialIntegrationResult:
    """Binned radial intensity and population counts."""

    bin_centers: np.ndarray
    intensity: np.ndarray
    counts: np.ndarray
    unit: str
    bin_edges: np.ndarray

    @property
    def bins(self) -> np.ndarray:
        return self.bin_centers

    @property
    def intensities(self) -> np.ndarray:
        return self.intensity

    @property
    def x(self) -> np.ndarray:
        return self.bin_centers

    @property
    def y(self) -> np.ndarray:
        return self.intensity

    def __iter__(self) -> Iterator[np.ndarray]:
        """Allow ``centres, intensity, counts = result`` for small scripts."""

        yield self.bin_centers
        yield self.intensity
        yield self.counts


@dataclass(frozen=True, slots=True)
class CakeResult:
    """Azimuthally binned detector image.

    ``intensity`` and ``counts`` have shape ``(n_chi, n_two_theta)``.  The
    first axis is therefore the vertical/row (chi) axis and the second is the
    horizontal/column (2theta) axis, matching ``imshow``'s usual convention.
    """

    two_theta: np.ndarray
    chi: np.ndarray
    intensity: np.ndarray
    counts: np.ndarray
    two_theta_edges: np.ndarray
    chi_edges: np.ndarray

    @property
    def two_theta_centers(self) -> np.ndarray:
        return self.two_theta

    @property
    def chi_centers(self) -> np.ndarray:
        return self.chi

    @property
    def matrix(self) -> np.ndarray:
        return self.intensity

    @property
    def data(self) -> np.ndarray:
        return self.intensity

    def __iter__(self) -> Iterator[np.ndarray]:
        yield self.two_theta
        yield self.chi
        yield self.intensity


def _validate_detector_metadata(
    shape: Sequence[int],
    dtype: object | None = None,
    *,
    preserve_dtype: bool = False,
) -> tuple[int, int]:
    """Validate detector shape/dtype before an array is copied or decoded."""

    try:
        dimensions = tuple(int(value) for value in shape)
    except (TypeError, ValueError) as exc:
        raise ValueError("detector image shape must contain positive integers") from exc
    if len(dimensions) != 2:
        raise ValueError("detector image must be a two-dimensional array")
    if any(value <= 0 for value in dimensions):
        raise ValueError("detector image shape must contain positive integers")
    pixels = dimensions[0] * dimensions[1]
    if pixels > MAX_DETECTOR_PIXELS:
        raise ValueError(
            "detector image exceeds the pixel limit: "
            f"{pixels:,} > {MAX_DETECTOR_PIXELS:,}"
        )
    if dtype is not None:
        try:
            dtype_value = np.dtype(dtype)
        except TypeError as exc:
            raise ValueError("detector image must contain real numeric values") from exc
        if dtype_value.kind == "b" or not np.issubdtype(dtype_value, np.number):
            raise ValueError("detector image must contain real numeric values")
        if np.issubdtype(dtype_value, np.complexfloating):
            raise ValueError("detector image must contain real numeric values")
        itemsize = dtype_value.itemsize if preserve_dtype else np.dtype(float).itemsize
        estimated_bytes = pixels * itemsize
        if estimated_bytes > MAX_DETECTOR_BYTES:
            raise ValueError(
                "detector image exceeds the memory limit: "
                f"{estimated_bytes:,} > {MAX_DETECTOR_BYTES:,} bytes"
            )
    return dimensions


def _all_finite(array: np.ndarray) -> bool:
    """Check finiteness in row chunks to avoid a full-size boolean copy."""

    rows, width = array.shape
    rows_per_chunk = max(1, 1_000_000 // max(width, 1))
    for start in range(0, rows, rows_per_chunk):
        if not bool(np.isfinite(array[start : start + rows_per_chunk]).all()):
            return False
    return True


def _validate_detector_array(image: Any, *, preserve_dtype: bool = False) -> np.ndarray:
    if np.ma.isMaskedArray(image):
        raise ValueError("masked arrays are not accepted; pass the mask explicitly")
    shape_hint = getattr(image, "shape", None)
    if shape_hint is not None:
        _validate_detector_metadata(
            shape_hint,
            getattr(image, "dtype", None),
            preserve_dtype=preserve_dtype,
        )
    try:
        array = np.asarray(image)
    except Exception as exc:  # pragma: no cover - NumPy controls the concrete exception
        raise ValueError(
            "detector image must be a numeric two-dimensional array"
        ) from exc
    _validate_detector_metadata(
        array.shape,
        array.dtype,
        preserve_dtype=preserve_dtype,
    )
    if np.iscomplexobj(array) or not np.issubdtype(array.dtype, np.number):
        raise ValueError("detector image must contain real numeric values")
    if not _all_finite(array):
        raise ValueError("detector image must contain only finite values")
    if preserve_dtype:
        return np.array(array, copy=True)
    return np.asarray(array, dtype=float)


def _validate_mask(mask: Any, shape: tuple[int, int]) -> np.ndarray:
    try:
        array = np.asarray(mask)
    except Exception as exc:  # pragma: no cover - NumPy controls the concrete exception
        raise ValueError(
            "mask must be a boolean array matching the detector image"
        ) from exc
    if array.shape != shape:
        raise ValueError(
            f"mask shape {array.shape} does not match detector image shape {shape}"
        )
    if np.iscomplexobj(array):
        raise ValueError("mask must contain finite boolean/numeric values")
    if array.dtype != np.bool_:
        if not np.issubdtype(array.dtype, np.number):
            raise ValueError("mask must contain finite boolean/numeric values")
        if not bool(np.all(np.isfinite(array))):
            raise ValueError("mask must contain finite boolean/numeric values")
    return np.asarray(array, dtype=bool)


def _positive_integer(value: object, name: str) -> int:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if not math.isfinite(number) or number < 1 or not number.is_integer():
        raise ValueError(f"{name} must be a positive integer")
    return int(number)


def _validate_cake_budget(n_two_theta: int, n_chi: int) -> int:
    """Validate the dense cake result before allocating output arrays."""

    cells = n_two_theta * n_chi
    if cells > MAX_CAKE_CELLS:
        raise ValueError(
            "cake output exceeds the cell limit: "
            f"{cells:,} > {MAX_CAKE_CELLS:,}"
        )
    result_bytes = cells * (np.dtype(float).itemsize + np.dtype(np.int64).itemsize)
    if result_bytes > MAX_CAKE_RESULT_BYTES:
        raise ValueError(
            "cake output exceeds the memory limit: "
            f"{result_bytes:,} > {MAX_CAKE_RESULT_BYTES:,} bytes"
        )
    # ``bincount`` holds counts, totals, and means concurrently.  Keep a
    # separate working-set guard so future changes cannot accidentally make a
    # legal result allocate an unbounded number of dense temporary arrays.
    working_bytes = cells * (
        np.dtype(np.int64).itemsize + 2 * np.dtype(float).itemsize
    )
    if working_bytes > MAX_CAKE_WORKING_BYTES:
        raise ValueError(
            "cake working set exceeds the memory limit: "
            f"{working_bytes:,} > {MAX_CAKE_WORKING_BYTES:,} bytes"
        )
    return cells


def _validate_range(
    value: object, name: str, *, max_width: float | None = None
) -> tuple[float, float]:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a finite (low, high) pair")
    try:
        values = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError(f"{name} must be a finite (low, high) pair") from exc
    if len(values) != 2:
        raise ValueError(f"{name} must be a finite (low, high) pair")
    low = _finite_float(values[0], f"{name}[0]")
    high = _finite_float(values[1], f"{name}[1]")
    if not high > low:
        raise ValueError(f"{name} must have high > low")
    if max_width is not None and high - low > max_width + 1e-12:
        raise ValueError(f"{name} width must not exceed {max_width:g}")
    return low, high


def _validate_edges(edges: object, name: str) -> np.ndarray:
    try:
        array = np.asarray(edges, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a one-dimensional finite edge array") from exc
    if array.ndim != 1 or array.size < 2:
        raise ValueError(
            f"{name} must be a one-dimensional edge array with at least two values"
        )
    if not bool(np.all(np.isfinite(array))):
        raise ValueError(f"{name} must contain only finite values")
    if not bool(np.all(np.diff(array) > 0.0)):
        raise ValueError(f"{name} must be strictly increasing")
    return array


def _make_edges(
    values: np.ndarray,
    *,
    n_bins: int,
    requested_range: tuple[float, float] | None,
    default_range: tuple[float, float] | None = None,
) -> np.ndarray:
    finite_values = values[np.isfinite(values)]
    if requested_range is not None:
        low, high = requested_range
    elif finite_values.size:
        low = float(np.min(finite_values))
        high = float(np.max(finite_values))
    elif default_range is not None:
        low, high = default_range
    else:
        low, high = (0.0, 1.0)

    if not math.isfinite(low) or not math.isfinite(high) or high < low:
        raise ValueError("bin range must be finite and have high >= low")
    if high == low:
        # A constant image/radius is still a valid input.  Widen the interval
        # enough to make a useful single populated bin without introducing a
        # machine-scale zero-width edge.
        half_width = max(abs(low) * 1e-6, 1e-6)
        low -= half_width
        high += half_width
    return np.linspace(low, high, n_bins + 1, dtype=float)


def _pixel_coordinates(
    image_shape: tuple[int, int], geometry: DetectorGeometry
) -> tuple[np.ndarray, np.ndarray]:
    geometry.validate_for_shape(image_shape)
    height, width = image_shape
    # Keep the two coordinate axes one-dimensional/broadcastable.  ``np.indices``
    # used to allocate two full image-sized grids before any geometry was
    # calculated; broadcasting preserves the same x/y convention with much
    # less temporary memory.
    x = (np.arange(width, dtype=float) - geometry.center_x) * geometry.pixel_size_x
    y = (np.arange(height, dtype=float) - geometry.center_y) * geometry.pixel_size_y
    x = x[np.newaxis, :]
    y = y[:, np.newaxis]
    return x, y


def _two_theta_from_radius(
    radius: np.ndarray, geometry: DetectorGeometry
) -> np.ndarray:
    return np.degrees(np.arctan2(radius, geometry.distance))


def _radial_coordinate(
    two_theta: np.ndarray, geometry: DetectorGeometry, unit: str
) -> np.ndarray:
    if unit == "two_theta":
        return two_theta
    theta = np.radians(two_theta * 0.5)
    with np.errstate(divide="ignore", invalid="ignore"):
        if unit == "q":
            return 4.0 * np.pi * np.sin(theta) / geometry.wavelength
        return geometry.wavelength / (2.0 * np.sin(theta))


def _bin_means(
    coordinate: np.ndarray,
    image: np.ndarray,
    edges: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n_bins = edges.size - 1
    flat_coordinate = coordinate.ravel()
    flat_image = image.ravel()
    flat_valid = valid.ravel() & np.isfinite(flat_coordinate)
    if not bool(np.any(flat_valid)):
        return np.full(n_bins, np.nan, dtype=float), np.zeros(n_bins, dtype=np.int64)

    coordinate_values = flat_coordinate[flat_valid]
    indices = np.searchsorted(edges, coordinate_values, side="right") - 1
    # ``searchsorted(..., side='right')`` puts a value exactly equal to the
    # final edge one past the last bin.  The final edge is inclusive; values
    # above it remain excluded.
    at_upper_edge = (indices == n_bins) & (coordinate_values <= edges[-1])
    indices[at_upper_edge] = n_bins - 1
    inside = (indices >= 0) & (indices < n_bins) & (coordinate_values <= edges[-1])
    if not bool(np.any(inside)):
        return np.full(n_bins, np.nan, dtype=float), np.zeros(n_bins, dtype=np.int64)
    indices = indices[inside]
    values = flat_image[flat_valid][inside]
    counts = np.bincount(indices, minlength=n_bins).astype(np.int64, copy=False)
    totals = np.bincount(indices, weights=values, minlength=n_bins)
    means = np.full(n_bins, np.nan, dtype=float)
    occupied = counts > 0
    means[occupied] = totals[occupied] / counts[occupied]
    return means, counts


def integrate_radial(
    image: Any,
    geometry: DetectorGeometry,
    *,
    unit: str = "two_theta",
    n_bins: int | None = 360,
    bins: int | Sequence[float] | None = None,
    bin_edges: Sequence[float] | None = None,
    radial_range: Sequence[float] | None = None,
    mask: Any | None = None,
) -> RadialIntegrationResult:
    """Integrate a detector image into radial 1-D bins.

    The returned intensity is the arithmetic mean of unmasked pixels in each
    bin and empty bins are represented by ``NaN``.  A ``True`` mask value
    excludes that pixel, following NumPy/pyFAI mask conventions.  The ``d``
    coordinate is undefined at the direct-beam centre; that pixel is simply
    omitted from ``d`` integration rather than producing an infinite output.
    """

    if not isinstance(geometry, DetectorGeometry):
        raise TypeError("geometry must be a DetectorGeometry instance")
    array = _validate_detector_array(image)
    shape = (int(array.shape[0]), int(array.shape[1]))
    if mask is None:
        excluded = np.zeros(shape, dtype=bool)
    else:
        excluded = _validate_mask(mask, shape)
    unit_name = normalize_detector_unit(unit)
    geometry.validate_for_shape(shape)

    requested_edges: object | None = bin_edges
    if bins is not None:
        if requested_edges is not None:
            raise ValueError("Specify only one of bins and bin_edges")
        if isinstance(bins, (int, np.integer)):
            n_bins = int(bins)
        elif np.ndim(bins) == 0:
            n_bins = _positive_integer(bins, "bins")
        else:
            requested_edges = bins
    if requested_edges is not None:
        edges = _validate_edges(requested_edges, "bin_edges")
    else:
        if n_bins is None:
            n_bins = 360
        n_bins = _positive_integer(n_bins, "n_bins")

    if requested_edges is not None and radial_range is not None:
        raise ValueError("Specify only one of radial_range and bin_edges")
    requested = (
        None if radial_range is None else _validate_range(radial_range, "radial_range")
    )
    x, y = _pixel_coordinates(shape, geometry)
    radius = np.hypot(x, y)
    two_theta = _two_theta_from_radius(radius, geometry)
    coordinate = _radial_coordinate(two_theta, geometry, unit_name)
    valid = ~excluded
    # d-spacing at theta=0 is +infinity and is not a valid finite binned
    # coordinate.  The shared binning helper excludes it.
    valid &= np.isfinite(coordinate)

    if requested_edges is None:
        edges = _make_edges(coordinate[valid], n_bins=n_bins, requested_range=requested)
    means, counts = _bin_means(coordinate, array, edges, valid)
    centers = (edges[:-1] + edges[1:]) * 0.5
    return RadialIntegrationResult(
        bin_centers=centers,
        intensity=means,
        counts=counts,
        unit=unit_name,
        bin_edges=edges,
    )


def _chi_for_range(
    x: np.ndarray, y: np.ndarray, chi_range: tuple[float, float]
) -> np.ndarray:
    raw = np.degrees(np.arctan2(y, x))
    low, _ = chi_range
    # Map every raw angle into the requested interval's coordinate convention.
    # This is relative to ``low`` rather than a special-case [0, 360) wrap, so
    # shifted full circles such as (-90, 270) and (30, 390) remain complete.
    return np.mod(raw - low, 360.0) + low


def generate_cake(
    image: Any,
    geometry: DetectorGeometry,
    *,
    n_two_theta: int = 360,
    n_chi: int = 360,
    two_theta_range: Sequence[float] | None = None,
    chi_range: Sequence[float] | None = None,
    two_theta_edges: Sequence[float] | None = None,
    chi_edges: Sequence[float] | None = None,
    mask: Any | None = None,
) -> CakeResult:
    """Create a mean-intensity 2theta--chi (``cake``) matrix.

    The matrix is shaped ``(n_chi, n_two_theta)``.  Angles are in degrees;
    ``chi`` is measured from the positive detector-x direction.  The default
    chi interval is ``[-180, 180]``.  A non-negative interval such as
    ``[0, 360]`` requests the equivalent wrapped azimuth convention.
    """

    if not isinstance(geometry, DetectorGeometry):
        raise TypeError("geometry must be a DetectorGeometry instance")
    if two_theta_edges is not None and two_theta_range is not None:
        raise ValueError("Specify only one of two_theta_range and two_theta_edges")
    if chi_edges is not None and chi_range is not None:
        raise ValueError("Specify only one of chi_range and chi_edges")
    array = _validate_detector_array(image)
    shape = (int(array.shape[0]), int(array.shape[1]))
    if mask is None:
        excluded = np.zeros(shape, dtype=bool)
    else:
        excluded = _validate_mask(mask, shape)
    geometry.validate_for_shape(shape)

    if two_theta_edges is not None:
        theta_edges = _validate_edges(two_theta_edges, "two_theta_edges")
        theta_count = theta_edges.size - 1
    else:
        theta_count = _positive_integer(n_two_theta, "n_two_theta")
        theta_requested = (
            None
            if two_theta_range is None
            else _validate_range(two_theta_range, "two_theta_range")
        )

    if chi_edges is not None:
        resolved_chi_edges = _validate_edges(chi_edges, "chi_edges")
        resolved_chi_range = (
            float(resolved_chi_edges[0]),
            float(resolved_chi_edges[-1]),
        )
        chi_count = resolved_chi_edges.size - 1
    else:
        resolved_chi_range = _validate_range(
            (-180.0, 180.0) if chi_range is None else chi_range,
            "chi_range",
            max_width=360.0,
        )
        chi_count = _positive_integer(n_chi, "n_chi")

    _validate_cake_budget(theta_count, chi_count)

    x, y = _pixel_coordinates(shape, geometry)
    radius = np.hypot(x, y)
    theta_values = _two_theta_from_radius(radius, geometry)
    chi_values = _chi_for_range(x, y, resolved_chi_range)
    valid = ~excluded & np.isfinite(theta_values) & np.isfinite(chi_values)

    if two_theta_edges is None:
        theta_edges = _make_edges(
            theta_values[valid],
            n_bins=theta_count,
            requested_range=theta_requested,
            default_range=(0.0, 1.0),
        )
    if chi_edges is None:
        chi_edges = np.linspace(
            resolved_chi_range[0], resolved_chi_range[1], chi_count + 1, dtype=float
        )
    else:
        chi_edges = resolved_chi_edges

    flat_theta = theta_values.ravel()
    flat_chi = chi_values.ravel()
    theta_indices = np.searchsorted(theta_edges, flat_theta, side="right") - 1
    chi_indices = np.searchsorted(chi_edges, flat_chi, side="right") - 1
    n_theta = theta_edges.size - 1
    n_chi_actual = chi_edges.size - 1
    theta_at_upper = (theta_indices == n_theta) & (flat_theta <= theta_edges[-1])
    chi_at_upper = (chi_indices == n_chi_actual) & (flat_chi <= chi_edges[-1])
    theta_indices[theta_at_upper] = n_theta - 1
    chi_indices[chi_at_upper] = n_chi_actual - 1
    flat_valid = valid.ravel()
    inside = (
        flat_valid
        & (theta_indices >= 0)
        & (theta_indices < n_theta)
        & (chi_indices >= 0)
        & (chi_indices < n_chi_actual)
        & (flat_theta <= theta_edges[-1])
        & (flat_chi <= chi_edges[-1])
    )
    if bool(np.any(inside)):
        flat_indices = chi_indices[inside] * n_theta + theta_indices[inside]
        values = array.ravel()[inside]
        flat_counts = np.bincount(
            flat_indices, minlength=n_chi_actual * n_theta
        ).astype(np.int64, copy=False)
        flat_totals = np.bincount(
            flat_indices, weights=values, minlength=n_chi_actual * n_theta
        )
        flat_means = np.full(n_chi_actual * n_theta, np.nan, dtype=float)
        occupied = flat_counts > 0
        flat_means[occupied] = flat_totals[occupied] / flat_counts[occupied]
    else:
        flat_counts = np.zeros(n_chi_actual * n_theta, dtype=np.int64)
        flat_means = np.full(n_chi_actual * n_theta, np.nan, dtype=float)

    return CakeResult(
        two_theta=(theta_edges[:-1] + theta_edges[1:]) * 0.5,
        chi=(chi_edges[:-1] + chi_edges[1:]) * 0.5,
        intensity=flat_means.reshape(n_chi_actual, n_theta),
        counts=flat_counts.reshape(n_chi_actual, n_theta),
        two_theta_edges=theta_edges,
        chi_edges=chi_edges,
    )


def _read_npy_header(stream: Any) -> tuple[tuple[int, ...], np.dtype[Any]]:
    """Read only an NPY header from a seekable or ZIP member stream."""

    try:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, _fortran_order, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version in ((2, 0), (3, 0)):
            # NumPy uses the v2 header reader for the v3 header layout; v3
            # differs only in the header's text encoding.
            shape, _fortran_order, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError(f"unsupported NPY format version {version!r}")
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError("invalid detector NPY header") from exc
    return tuple(int(value) for value in shape), np.dtype(dtype)


def _check_detector_file_size(file_path: Path) -> None:
    try:
        file_size = file_path.stat().st_size
    except OSError as exc:  # pragma: no cover - path existence was checked above
        raise ValueError(f"Could not inspect detector file: {file_path}") from exc
    if file_size > MAX_DETECTOR_BYTES:
        raise ValueError(
            "detector file exceeds the memory limit: "
            f"{file_size:,} > {MAX_DETECTOR_BYTES:,} bytes"
        )


def load_detector_image(path: str | Path) -> np.ndarray:
    """Load a 2-D ``.npy`` array or a common image without new dependencies.

    ``.npy`` uses NumPy directly.  Other image formats are decoded lazily via
    Pillow, which is already an application dependency; importing this module
    therefore does not require Pillow when only ``.npy`` files are used.
    RGB/RGBA images are converted to a luminance plane.
    """

    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(str(file_path))
    _check_detector_file_size(file_path)
    suffix = file_path.suffix.lower()
    if suffix == ".npy":
        # mmap mode reads the shape/dtype header without materializing the
        # payload.  The metadata guard therefore runs before a potentially
        # huge file is copied into an ordinary ndarray.
        loaded = np.load(file_path, mmap_mode="r", allow_pickle=False)
        _validate_detector_metadata(
            loaded.shape,
            loaded.dtype,
            preserve_dtype=True,
        )
        return _validate_detector_array(loaded, preserve_dtype=True)
    if suffix == ".npz":
        with np.load(file_path, allow_pickle=False) as archive:
            names = list(archive.files)
            if len(names) != 1:
                raise ValueError("detector .npz must contain exactly one array")
            member_name = f"{names[0]}.npy"
            try:
                member_info = archive.zip.getinfo(member_name)
            except KeyError as exc:
                raise ValueError("detector .npz member is not a valid NPY array") from exc
            if member_info.file_size > MAX_DETECTOR_BYTES:
                raise ValueError(
                    "detector .npz member exceeds the memory limit: "
                    f"{member_info.file_size:,} > {MAX_DETECTOR_BYTES:,} bytes"
                )
            with archive.zip.open(member_info) as member:
                member_shape, member_dtype = _read_npy_header(member)
            _validate_detector_metadata(
                member_shape,
                member_dtype,
                preserve_dtype=True,
            )
            return _validate_detector_array(archive[names[0]], preserve_dtype=True)

    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - Pillow is a project dependency
        raise RuntimeError(
            "Pillow is required to load non-NPY detector images"
        ) from exc
    try:
        with Image.open(file_path) as image:
            width, height = image.size
            _validate_detector_metadata((height, width))
            loaded = np.asarray(image)
    except Exception as exc:
        raise ValueError(f"Could not read detector image: {file_path}") from exc
    if loaded.ndim == 3:
        if loaded.shape[-1] == 1:
            loaded = loaded[..., 0]
        elif loaded.shape[-1] >= 3:
            rgb = np.asarray(loaded[..., :3], dtype=float)
            loaded = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
        else:
            raise ValueError("detector image must be grayscale or RGB/RGBA")
    return _validate_detector_array(loaded, preserve_dtype=True)


# Short aliases make the public module discoverable without committing users
# to one verb for the same scientific operation.
radial_integrate = integrate_radial
radial_integration = integrate_radial
cake = generate_cake
cake_2theta_chi = generate_cake
load_image = load_detector_image


__all__ = [
    "CakeResult",
    "DetectorGeometry",
    "RadialIntegrationResult",
    "cake",
    "cake_2theta_chi",
    "generate_cake",
    "integrate_radial",
    "load_detector_image",
    "load_image",
    "normalize_detector_unit",
    "radial_integrate",
    "radial_integration",
]
