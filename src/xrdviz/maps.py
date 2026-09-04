"""Strict two-dimensional XRD map data and import helpers.

The plotting application primarily works with one-dimensional spectra, but a
number of common XRD views are naturally two-dimensional: a detector image,
an azimuthal cake, a reciprocal-space map (RSM), or a pole figure.  This
module keeps those data separate from plotting and instrument calibration.

``MapData`` stores an explicit Cartesian grid.  ``intensity`` is indexed as
``[y, x]`` and the coordinate vectors are the corresponding cell centres.
The model is immutable at the array level (the arrays are copied and marked
read-only), which makes JSON/dict persistence predictable and prevents a
caller from changing a map after it has entered an analysis workflow.

The CSV importer accepts *long-form* rows (one ``x, y, z`` tuple per row) and
requires every Cartesian grid point exactly once.  Regular axis spacing can
be required explicitly when a downstream method needs it.  The importer does
not infer a wavelength, detector distance,
pixel size, or any other missing physical parameter.  A raw detector image is
therefore represented in pixel-index coordinates until the caller supplies a
calibrated result (for example, a cake from :mod:`xrdviz.detector`).
"""

from __future__ import annotations

import csv
import json
import math
import re
from array import array
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from io import StringIO
from itertools import chain
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Iterator

import numpy as np


MAP_KINDS = frozenset({"detector", "cake", "rsm", "pole_figure"})

# CSV is a convenient interchange format, but it must not be allowed to turn
# an accidental coordinate typo into an unbounded allocation in the desktop
# process.  Callers handling a genuinely larger map can opt into a larger
# explicit budget after validating the source themselves.
DEFAULT_MAX_MAP_ROWS = 5_000_000
DEFAULT_MAX_MAP_CELLS = 4_000_000


_KIND_ALIASES = {
    "detector": "detector",
    "raw_detector": "detector",
    "raw_detector_image": "detector",
    "cake": "cake",
    "cake_2theta_chi": "cake",
    "cake_2_theta_chi": "cake",
    "rsm": "rsm",
    "reciprocal_space_map": "rsm",
    "reciprocal_space_mapping": "rsm",
    "pole_figure": "pole_figure",
    "polefigure": "pole_figure",
    "pole figure": "pole_figure",
}


_META_ALIASES = {
    "x": "x",
    "x_label": "x",
    "horizontal": "x",
    "horizontal_axis": "x",
    "y": "y",
    "y_label": "y",
    "vertical": "y",
    "vertical_axis": "y",
    "z": "intensity",
    "z_label": "intensity",
    "intensity": "intensity",
    "intensity_label": "intensity",
    "signal": "intensity",
}


_X_COLUMN_ALIASES = (
    "qx",
    "q_x",
    "q_parallel",
    "qparallel",
    "q_parallel_component",
    "qpar",
    "q_par",
    "phi",
    "azimuth",
    "azimuthal_angle",
    "two_theta",
    "2theta",
    "two_theta_deg",
    "d_spacing",
    "dspacing",
    "d",
    "x",
)

_Y_COLUMN_ALIASES = (
    "qz",
    "q_z",
    "q_perp",
    "qperp",
    "q_perpendicular",
    "qperpendicular",
    "q_y",
    "chi",
    "tilt",
    "tilt_angle",
    "elevation",
    "y",
)

_INTENSITY_COLUMN_ALIASES = (
    "intensity",
    "i",
    "signal",
    "z",
    "value",
)

_COUNTS_COLUMN_ALIASES = (
    "counts",
    "count",
    "n",
    "population",
    "pixel_count",
    "pixels",
    "weight",
)


def _normalise_column(value: object) -> str:
    """Normalise a CSV header without discarding common Greek symbols."""

    text = str(value).strip().lower()
    replacements = {
        "θ": "theta",
        "φ": "phi",
        "χ": "chi",
        "∥": "parallel",
        "⊥": "perp",
        "²": "2",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def _normalise_kind(kind: object) -> str:
    if not isinstance(kind, str):
        raise ValueError(f"Map kind must be one of {sorted(MAP_KINDS)}")
    key = kind.strip().lower().replace("-", "_")
    try:
        return _KIND_ALIASES[key]
    except KeyError as exc:
        raise ValueError(
            f"Map kind must be one of {sorted(MAP_KINDS)}; got {kind!r}"
        ) from exc


def _numeric_array(value: object, name: str, *, ndim: int) -> np.ndarray:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a finite numeric array")
    try:
        raw = np.asarray(value)
    except Exception as exc:  # pragma: no cover - NumPy controls concrete errors
        raise ValueError(f"{name} must be a finite numeric array") from exc
    if raw.ndim != ndim:
        raise ValueError(f"{name} must be a {ndim}-dimensional array")
    if raw.size == 0:
        raise ValueError(f"{name} must not be empty")
    if np.iscomplexobj(raw) or not np.issubdtype(raw.dtype, np.number):
        raise ValueError(f"{name} must contain real numeric values")
    try:
        array = np.asarray(raw, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain real numeric values") from exc
    if not bool(np.all(np.isfinite(array))):
        raise ValueError(f"{name} must contain only finite values")
    return np.array(array, dtype=float, copy=True)


def _axis_array(value: object, name: str) -> np.ndarray:
    axis = _numeric_array(value, name, ndim=1)
    if axis.size > 1 and not bool(np.all(np.diff(axis) > 0.0)):
        raise ValueError(f"{name} must be strictly increasing")
    axis.setflags(write=False)
    return axis


def _canonical_metadata_key(value: object) -> str:
    key = _normalise_column(value)
    return _META_ALIASES.get(key, key)


def _metadata_items(value: object, name: str) -> dict[str, str]:
    """Convert a mapping or 2/3-item sequence into canonical metadata."""

    if value is None:
        return {}
    if isinstance(value, Mapping):
        items = value.items()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = list(value)
        if len(values) not in (2, 3):
            raise ValueError(f"{name} sequence must contain two or three values")
        keys = ("x", "y") if len(values) == 2 else ("x", "y", "intensity")
        items = zip(keys, values)
    else:
        raise ValueError(f"{name} must be a mapping or a two/three-item sequence")

    result: dict[str, str] = {}
    for key, raw_value in items:
        canonical = _canonical_metadata_key(key)
        if canonical not in {"x", "y", "intensity"}:
            # Unknown metadata is retained so persistence does not silently
            # discard caller-provided descriptive labels.
            canonical = str(key).strip()
        if not canonical:
            raise ValueError(f"{name} keys must not be empty")
        if raw_value is None:
            result[canonical] = ""
        elif isinstance(raw_value, (str, Path)):
            result[canonical] = str(raw_value)
        else:
            raise ValueError(f"{name} values must be strings")
    return result


def _processing_metadata(value: object) -> MappingProxyType:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise ValueError("metadata must be a JSON-serializable mapping")
    copied = dict(value)
    try:
        encoded = json.dumps(copied, ensure_ascii=False, allow_nan=False)
        decoded = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "metadata must be JSON serializable and contain only finite values"
        ) from exc
    return MappingProxyType(decoded)


def _resolve_metadata(
    value: object,
    name: str,
    defaults: Mapping[str, str],
    *,
    overrides: Mapping[str, object | None] | None = None,
) -> MappingProxyType:
    result = dict(defaults)
    result.update(_metadata_items(value, name))
    for key, raw_value in (overrides or {}).items():
        if raw_value is not None:
            if not isinstance(raw_value, (str, Path)):
                raise ValueError(f"{name} values must be strings")
            result[key] = str(raw_value)
    return MappingProxyType(result)


def _array_attr(value: object, names: Sequence[str], name: str) -> object:
    if isinstance(value, Mapping):
        for candidate in names:
            if candidate in value and value[candidate] is not None:
                return value[candidate]
    else:
        for candidate in names:
            if hasattr(value, candidate):
                resolved = getattr(value, candidate)
                if resolved is not None:
                    return resolved
    raise ValueError(f"{name} is required")


@dataclass(frozen=True, slots=True, init=False)
class MapData:
    """Validated, immutable two-dimensional XRD map data.

    ``intensity`` and ``counts`` use matrix order ``(len(y), len(x))``.  The
    constructor accepts either canonical ``labels``/``units`` mappings or
    convenient ``x_label``/``y_label``/``intensity_label`` keyword aliases.
    Unknown metadata keys are retained as strings for persistence, while the
    canonical keys ``x``, ``y`` and ``intensity`` are always present.
    """

    kind: str
    x: np.ndarray
    y: np.ndarray
    intensity: np.ndarray
    labels: Mapping[str, str]
    units: Mapping[str, str]
    source_path: str
    counts: np.ndarray | None
    metadata: Mapping[str, Any]

    _DEFAULT_LABELS: ClassVar[Mapping[str, str]] = {
        "x": "x",
        "y": "y",
        "intensity": "Intensity",
    }
    _DEFAULT_UNITS: ClassVar[Mapping[str, str]] = {
        "x": "",
        "y": "",
        "intensity": "",
    }

    def __init__(
        self,
        kind: object,
        x: object,
        y: object,
        intensity: object,
        labels: object | None = None,
        units: object | None = None,
        source_path: object = "",
        counts: object | None = None,
        metadata: object | None = None,
        *,
        x_label: object | None = None,
        y_label: object | None = None,
        intensity_label: object | None = None,
        z_label: object | None = None,
        x_unit: object | None = None,
        y_unit: object | None = None,
        intensity_unit: object | None = None,
        z_unit: object | None = None,
    ) -> None:
        resolved_kind = _normalise_kind(kind)
        resolved_x = _axis_array(x, "x")
        resolved_y = _axis_array(y, "y")
        resolved_intensity = _numeric_array(intensity, "intensity", ndim=2)
        expected_shape = (resolved_y.size, resolved_x.size)
        if resolved_intensity.shape != expected_shape:
            raise ValueError(
                "intensity shape must be (len(y), len(x)); "
                f"got {resolved_intensity.shape}, expected {expected_shape}"
            )
        resolved_intensity.setflags(write=False)

        resolved_counts: np.ndarray | None
        if counts is None:
            resolved_counts = None
        else:
            resolved_counts = _numeric_array(counts, "counts", ndim=2)
            if resolved_counts.shape != expected_shape:
                raise ValueError(
                    "counts shape must be (len(y), len(x)); "
                    f"got {resolved_counts.shape}, expected {expected_shape}"
                )
            if bool(np.any(resolved_counts < 0.0)):
                raise ValueError("counts must be non-negative")
            resolved_counts.setflags(write=False)

        resolved_labels = _resolve_metadata(
            labels,
            "labels",
            self._DEFAULT_LABELS,
            overrides={
                "x": x_label,
                "y": y_label,
                "intensity": intensity_label
                if intensity_label is not None
                else z_label,
            },
        )
        resolved_units = _resolve_metadata(
            units,
            "units",
            self._DEFAULT_UNITS,
            overrides={
                "x": x_unit,
                "y": y_unit,
                "intensity": intensity_unit if intensity_unit is not None else z_unit,
            },
        )

        if source_path is None:
            resolved_source = ""
        elif isinstance(source_path, (str, Path)):
            resolved_source = str(source_path)
        else:
            raise ValueError("source_path must be a string or path-like value")

        object.__setattr__(self, "kind", resolved_kind)
        object.__setattr__(self, "x", resolved_x)
        object.__setattr__(self, "y", resolved_y)
        object.__setattr__(self, "intensity", resolved_intensity)
        object.__setattr__(self, "labels", resolved_labels)
        object.__setattr__(self, "units", resolved_units)
        object.__setattr__(self, "source_path", resolved_source)
        object.__setattr__(self, "counts", resolved_counts)
        object.__setattr__(self, "metadata", _processing_metadata(metadata))

    @property
    def z(self) -> np.ndarray:
        """Alias for the intensity matrix used by long-form CSV conventions."""

        return self.intensity

    @property
    def matrix(self) -> np.ndarray:
        """Alias useful for image/map plotting APIs."""

        return self.intensity

    @property
    def data(self) -> np.ndarray:
        return self.intensity

    @property
    def x_label(self) -> str:
        return self.labels["x"]

    @property
    def y_label(self) -> str:
        return self.labels["y"]

    @property
    def intensity_label(self) -> str:
        return self.labels["intensity"]

    @property
    def z_label(self) -> str:
        return self.intensity_label

    @property
    def x_unit(self) -> str:
        return self.units["x"]

    @property
    def y_unit(self) -> str:
        return self.units["y"]

    @property
    def intensity_unit(self) -> str:
        return self.units["intensity"]

    @property
    def z_unit(self) -> str:
        return self.intensity_unit

    @property
    def shape(self) -> tuple[int, int]:
        return self.intensity.shape

    def __iter__(self) -> Iterator[np.ndarray]:
        """Yield ``x, y, intensity`` for small interactive scripts."""

        yield self.x
        yield self.y
        yield self.intensity

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of the map."""

        return {
            "kind": self.kind,
            "x": self.x.tolist(),
            "y": self.y.tolist(),
            "intensity": self.intensity.tolist(),
            "counts": None if self.counts is None else self.counts.tolist(),
            "labels": dict(self.labels),
            "units": dict(self.units),
            "source_path": self.source_path,
            "metadata": dict(self.metadata),
        }

    as_dict = to_dict

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MapData":
        if not isinstance(payload, Mapping):
            raise ValueError("MapData payload must be a mapping")
        missing = [key for key in ("kind", "x", "y") if key not in payload]
        if "intensity" in payload and "z" in payload:
            raise ValueError("MapData payload must not contain both intensity and z")
        if "intensity" not in payload and "z" not in payload:
            missing.append("intensity")
        if missing:
            raise ValueError(
                f"MapData payload is missing required fields: {', '.join(missing)}"
            )
        return cls(
            kind=payload["kind"],
            x=payload["x"],
            y=payload["y"],
            intensity=payload.get("intensity", payload.get("z")),
            counts=payload.get("counts"),
            labels=payload.get("labels"),
            units=payload.get("units"),
            source_path=payload.get("source_path", ""),
            metadata=payload.get("metadata"),
        )

    @classmethod
    def from_json(cls, payload: str | bytes) -> "MapData":
        try:
            decoded = json.loads(payload)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("MapData JSON must contain a valid object") from exc
        return cls.from_dict(decoded)

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
            indent=indent,
        )

    def save(self, path: str | Path, *, indent: int | None = 2) -> Path:
        file_path = Path(path)
        file_path.write_text(self.to_json(indent=indent), encoding="utf-8")
        return file_path

    @classmethod
    def load(cls, path: str | Path) -> "MapData":
        file_path = Path(path)
        return cls.from_json(file_path.read_text(encoding="utf-8"))

    @classmethod
    def from_detector_raw(
        cls,
        image: object,
        *,
        counts: object | None = None,
        source_path: object = "",
        labels: object | None = None,
        units: object | None = None,
        x_label: object | None = None,
        y_label: object | None = None,
        intensity_label: object | None = None,
        x_unit: object | None = None,
        y_unit: object | None = None,
        intensity_unit: object | None = None,
        metadata: object | None = None,
    ) -> "MapData":
        """Wrap an uncalibrated detector image in pixel-index coordinates.

        No detector centre, pixel size, distance, or wavelength is invented.
        The caller can later construct a calibrated cake/RSM map from an
        explicit geometry calculation.
        """

        matrix = _numeric_array(image, "detector image", ndim=2)
        height, width = matrix.shape
        resolved_labels = {
            "x": "Detector x",
            "y": "Detector y",
            "intensity": "Intensity",
        }
        resolved_labels.update(_metadata_items(labels, "labels"))
        resolved_units = {"x": "pixel", "y": "pixel", "intensity": ""}
        resolved_units.update(_metadata_items(units, "units"))
        return cls(
            kind="detector",
            x=np.arange(width, dtype=float),
            y=np.arange(height, dtype=float),
            intensity=matrix,
            counts=counts,
            source_path=source_path,
            labels=resolved_labels,
            units=resolved_units,
            x_label=x_label,
            y_label=y_label,
            intensity_label=intensity_label,
            x_unit=x_unit,
            y_unit=y_unit,
            intensity_unit=intensity_unit,
            metadata=metadata,
        )

    @classmethod
    def from_cake(
        cls,
        cake_result: object,
        *,
        counts: object | None = None,
        source_path: object = "",
        labels: object | None = None,
        units: object | None = None,
        x_label: object | None = None,
        y_label: object | None = None,
        intensity_label: object | None = None,
        x_unit: object | None = None,
        y_unit: object | None = None,
        intensity_unit: object | None = None,
        metadata: object | None = None,
    ) -> "MapData":
        """Convert a detector cake result into a ``kind='cake'`` map.

        The helper uses the result's explicit ``two_theta``/``chi`` centres
        and intensity matrix.  It accepts both the native detector result and
        a mapping/duck-typed object with equivalent fields.
        """

        x = _array_attr(
            cake_result, ("two_theta", "two_theta_centers", "x"), "cake two_theta"
        )
        y = _array_attr(cake_result, ("chi", "chi_centers", "y"), "cake chi")
        intensity = _array_attr(
            cake_result, ("intensity", "matrix", "data", "z"), "cake intensity"
        )
        if counts is None:
            if isinstance(cake_result, Mapping):
                counts = cake_result.get("counts")
            elif hasattr(cake_result, "counts"):
                counts = getattr(cake_result, "counts")

        if counts is not None:
            try:
                intensity_array = np.asarray(intensity, dtype=float)
                counts_array = np.asarray(counts, dtype=float)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "cake intensity and counts must be numeric arrays"
                ) from exc
            if intensity_array.shape != counts_array.shape:
                raise ValueError("cake intensity and counts must have the same shape")
            populated = counts_array > 0
            if bool(np.any(~np.isfinite(intensity_array[populated]))):
                raise ValueError(
                    "populated cake bins must contain finite intensity values"
                )
            # Empty bins remain explicit through counts==0. A finite storage
            # placeholder keeps JSON standards-compliant; renderers and CSV
            # exporters mask those cells rather than treating zero as data.
            intensity = np.where(populated, intensity_array, 0.0)

        resolved_labels = {
            "x": r"$2\theta$",
            "y": r"$\chi$",
            "intensity": "Intensity",
        }
        resolved_labels.update(_metadata_items(labels, "labels"))
        resolved_units = {"x": "deg", "y": "deg", "intensity": ""}
        resolved_units.update(_metadata_items(units, "units"))
        return cls(
            kind="cake",
            x=x,
            y=y,
            intensity=intensity,
            counts=counts,
            source_path=source_path,
            labels=resolved_labels,
            units=resolved_units,
            x_label=x_label,
            y_label=y_label,
            intensity_label=intensity_label,
            x_unit=x_unit,
            y_unit=y_unit,
            intensity_unit=intensity_unit,
            metadata=metadata,
        )


def _clean_csv_text(text: object):
    """Yield non-empty, non-comment CSV lines without materialising rows."""

    if isinstance(text, bytes):
        try:
            text = text.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError("Map CSV must be UTF-8 text") from exc
    if isinstance(text, str):
        lines: Iterable[object] = StringIO(text)
    else:
        try:
            lines = iter(text)  # type: ignore[arg-type]
        except TypeError as exc:
            raise TypeError(
                "Map CSV input must be a string, bytes, or iterable of text lines"
            ) from exc
    found = False
    for raw_line in lines:
        if not isinstance(raw_line, str):
            raise TypeError("Map CSV line iterator must yield strings")
        line = raw_line.rstrip("\r\n")
        if line.strip() and not line.lstrip().startswith("#"):
            found = True
            yield line
    if not found:
        raise ValueError("Map CSV must contain a header and data rows")


def _csv_delimiter(header: str, delimiter: str | None) -> str:
    if delimiter is not None:
        if delimiter not in {",", ";", "\t"}:
            raise ValueError("Map CSV delimiter must be ',', ';', or tab")
        return delimiter
    counts = {candidate: header.count(candidate) for candidate in (",", ";", "\t")}
    selected = max(counts, key=counts.get)
    return selected if counts[selected] else ","


def _resolve_column(
    headers: Mapping[str, str], aliases: Sequence[str], role: str
) -> str:
    candidates = [headers[alias] for alias in aliases if alias in headers]
    if not candidates:
        raise ValueError(f"Map CSV must include a {role} column")
    if len(candidates) > 1:
        raise ValueError(
            f"Map CSV has multiple {role} columns: {', '.join(candidates)}"
        )
    return candidates[0]


def _resolve_metadata_header(
    headers: Mapping[str, str], aliases: Sequence[str], role: str
) -> str | None:
    candidates = [headers[alias] for alias in aliases if alias in headers]
    if len(candidates) > 1:
        raise ValueError(
            f"Map CSV has multiple {role} metadata columns: {', '.join(candidates)}"
        )
    return candidates[0] if candidates else None


def _capture_metadata_value(
    values: dict[str, str],
    row: Mapping[object, object],
    header: str,
    key: str,
    row_number: int,
    *,
    required: bool = False,
) -> None:
    raw_value = row.get(header)
    value = "" if raw_value is None else str(raw_value).strip()
    if required and not value:
        raise ValueError(f"Map CSV row {row_number} has an empty {key} metadata value")
    if key not in values:
        values[key] = value
    elif values[key] != value:
        raise ValueError(f"Map CSV has conflicting {key} metadata values")


def _validate_budget(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"Map CSV {name} must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Map CSV {name} must be a positive integer") from exc
    if parsed <= 0 or parsed != value:
        raise ValueError(f"Map CSV {name} must be a positive integer")
    return parsed


def _csv_number(value: object, row_number: int, column: str) -> float:
    if value is None or isinstance(value, (list, tuple)) or not str(value).strip():
        raise ValueError(f"Map CSV row {row_number} column {column!r} must be numeric")
    try:
        result = float(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Map CSV row {row_number} column {column!r} must be numeric"
        ) from exc
    if not math.isfinite(result):
        raise ValueError(f"Map CSV row {row_number} column {column!r} must be finite")
    return result


def _check_regular_axis(values: np.ndarray, name: str) -> None:
    if values.size <= 2:
        return
    differences = np.diff(values)
    if not bool(np.all(differences > 0.0)):
        raise ValueError(f"Map {name} coordinates must be strictly increasing")
    if not bool(np.allclose(differences, differences[0], rtol=1e-8, atol=1e-12)):
        raise ValueError(f"Map {name} coordinates must form a regular grid")


def parse_map_csv(
    text: str | bytes | Iterable[str],
    *,
    kind: str | None = None,
    labels: object | None = None,
    units: object | None = None,
    source_path: object = "",
    delimiter: str | None = None,
    require_regular: bool = False,
    x_label: object | None = None,
    y_label: object | None = None,
    intensity_label: object | None = None,
    x_unit: object | None = None,
    y_unit: object | None = None,
    intensity_unit: object | None = None,
    max_rows: int = DEFAULT_MAX_MAP_ROWS,
    max_cells: int = DEFAULT_MAX_MAP_CELLS,
) -> MapData:
    """Parse a complete Cartesian long-form map CSV.

    Supported x aliases include ``qx``, ``q_parallel`` and ``phi``; y
    aliases include ``qz``, ``q_perp`` and ``chi``.  The z column can be
    named ``intensity``, ``I``, ``z`` or (when no separate intensity column
    exists) ``counts``.  If both intensity and counts are present, counts are
    retained as the optional population matrix.
    """

    row_limit = _validate_budget(max_rows, "row limit")
    cell_limit = _validate_budget(max_cells, "grid limit")
    line_iterator = iter(_clean_csv_text(text))
    try:
        header_line = next(line_iterator)
    except StopIteration as exc:  # pragma: no cover - _clean_csv_text reports this
        raise ValueError("Map CSV must contain a header and data rows") from exc
    csv_reader = csv.DictReader(
        chain((header_line,), line_iterator),
        delimiter=_csv_delimiter(header_line, delimiter),
    )
    if csv_reader.fieldnames is None:
        raise ValueError("Map CSV must include a header row")

    headers: dict[str, str] = {}
    for raw_header in csv_reader.fieldnames:
        if raw_header is None:
            raise ValueError("Map CSV contains an unnamed column")
        normalised = _normalise_column(raw_header)
        if not normalised:
            raise ValueError("Map CSV headers must not be empty")
        if normalised in headers:
            raise ValueError(f"Map CSV has duplicate header {raw_header!r}")
        headers[normalised] = raw_header

    kind_header = _resolve_metadata_header(headers, ("kind",), "kind")
    source_header = _resolve_metadata_header(
        headers, ("source_file", "source_path", "source"), "source"
    )
    label_headers = {
        key: _resolve_metadata_header(headers, (f"{key}_label",), key)
        for key in ("x", "y", "intensity")
    }
    unit_headers = {
        key: _resolve_metadata_header(headers, (f"{key}_unit",), key)
        for key in ("x", "y", "intensity")
    }

    x_header = _resolve_column(
        headers, tuple(_normalise_column(alias) for alias in _X_COLUMN_ALIASES), "x"
    )
    y_header = _resolve_column(
        headers, tuple(_normalise_column(alias) for alias in _Y_COLUMN_ALIASES), "y"
    )

    intensity_candidates = [
        headers[alias]
        for alias in tuple(
            _normalise_column(alias) for alias in _INTENSITY_COLUMN_ALIASES
        )
        if alias in headers
    ]
    if len(intensity_candidates) > 1:
        raise ValueError(
            f"Map CSV has multiple intensity columns: {', '.join(intensity_candidates)}"
        )
    if intensity_candidates:
        z_header = intensity_candidates[0]
        counts_candidates = [
            headers[alias]
            for alias in tuple(
                _normalise_column(alias) for alias in _COUNTS_COLUMN_ALIASES
            )
            if alias in headers and headers[alias] != z_header
        ]
        if len(counts_candidates) > 1:
            raise ValueError(
                f"Map CSV has multiple counts columns: {', '.join(counts_candidates)}"
            )
        counts_header = counts_candidates[0] if counts_candidates else None
    else:
        # ``counts`` is a valid z name when it is the only signal column, but
        # it is not copied into the optional counts field a second time.
        z_header = _resolve_column(
            headers,
            tuple(_normalise_column(alias) for alias in _COUNTS_COLUMN_ALIASES),
            "intensity or counts",
        )
        counts_header = None

    # Store numeric rows in compact typed arrays.  A Python tuple/list per
    # cell is needlessly expensive for publication-sized maps and used to
    # make malformed input particularly easy to turn into a memory spike.
    row_x = array("d")
    row_y = array("d")
    row_z = array("d")
    row_counts = array("d") if counts_header is not None else None
    metadata_values: dict[str, str] = {}
    row_count = 0
    for row_number, row in enumerate(csv_reader, start=2):
        if row_count >= row_limit:
            raise ValueError(
                f"Map CSV row limit exceeded ({row_limit} rows); choose a smaller map or raise max_rows explicitly"
            )
        # DictReader uses a None key for excess fields; accepting those would
        # silently shift a z value into the wrong column.
        if None in row:
            raise ValueError(
                f"Map CSV row {row_number} has more fields than the header"
            )
        if not row or not any(
            str(value or "").strip()
            for value in row.values()
            if not isinstance(value, list)
        ):
            continue

        if kind_header is not None:
            _capture_metadata_value(
                metadata_values, row, kind_header, "kind", row_number, required=True
            )
        if source_header is not None:
            _capture_metadata_value(metadata_values, row, source_header, "source", row_number)
        for key, header in label_headers.items():
            if header is not None:
                _capture_metadata_value(
                    metadata_values, row, header, f"{key}_label", row_number
                )
        for key, header in unit_headers.items():
            if header is not None:
                _capture_metadata_value(
                    metadata_values, row, header, f"{key}_unit", row_number
                )

        x_value = _csv_number(row.get(x_header), row_number, x_header)
        y_value = _csv_number(row.get(y_header), row_number, y_header)
        raw_z_value = row.get(z_header)
        z_value = (
            math.nan
            if raw_z_value is None or not str(raw_z_value).strip()
            else _csv_number(raw_z_value, row_number, z_header)
        )
        row_x.append(x_value)
        row_y.append(y_value)
        row_z.append(z_value)
        if row_counts is not None:
            raw_count = row.get(counts_header)
            row_counts.append(
                math.nan
                if raw_count is None or not str(raw_count).strip()
                else _csv_number(raw_count, row_number, counts_header)
            )
        row_count += 1

    if row_count == 0:
        raise ValueError("Map CSV does not contain data rows")

    # An all-blank optional counts column is how the publication exporter
    # represents a map without population data.  Mixed blank/non-blank counts
    # are malformed and are rejected rather than silently inventing support.
    if row_counts is not None:
        count_missing = any(math.isnan(value) for value in row_counts)
        count_present = any(not math.isnan(value) for value in row_counts)
        if count_missing and count_present:
            raise ValueError(
                "Map CSV counts must be numeric for every row or blank for every row"
            )
        if count_missing:
            row_counts = None

    if row_counts is None:
        if any(math.isnan(value) for value in row_z):
            raise ValueError(f"Map CSV column {z_header!r} must be numeric")
        z_values = row_z
    else:
        z_values = array("d")
        for z_value, count_value in zip(row_z, row_counts):
            if math.isnan(z_value):
                if count_value != 0.0:
                    raise ValueError(f"Map CSV column {z_header!r} must be numeric")
                # Exported empty bins intentionally leave intensity blank.
                # Keep a finite placeholder while counts preserves emptiness.
                z_values.append(0.0)
            else:
                z_values.append(z_value)

    # ``set(array('d'))`` creates a Python object for every unique coordinate
    # and can turn a sparse/malformed file into a large memory spike.  NumPy's
    # typed unique/sort path keeps the temporary representation compact.
    x_values = np.unique(np.frombuffer(row_x, dtype=np.float64))
    y_values = np.unique(np.frombuffer(row_y, dtype=np.float64))
    if require_regular:
        _check_regular_axis(x_values, "x")
        _check_regular_axis(y_values, "y")

    expected_cells = int(x_values.size) * int(y_values.size)
    if expected_cells > cell_limit:
        raise ValueError(
            f"Map CSV grid limit exceeded ({expected_cells} cells; max {cell_limit})"
        )
    # One byte per cell gives a bounded, compact occupancy map.  It replaces
    # the previous full Cartesian tuple set and sorted missing set.
    occupancy = bytearray(expected_cells)
    x_index = {float(value): index for index, value in enumerate(x_values)}
    y_index = {float(value): index for index, value in enumerate(y_values)}
    for index, (x_value, y_value) in enumerate(zip(row_x, row_y)):
        ix = x_index[float(x_value)]
        iy = y_index[float(y_value)]
        cell_index = iy * int(x_values.size) + ix
        if occupancy[cell_index]:
            raise ValueError("Map CSV contains duplicate (x, y) points")
        occupancy[cell_index] = 1

    if row_count != expected_cells:
        try:
            first_missing_index = occupancy.index(0)
        except ValueError:  # pragma: no cover - row_count guards complete maps
            first_missing_index = None
        if first_missing_index is None:
            first_missing = None
        else:
            x_size = int(x_values.size)
            first_missing = (
                float(x_values[first_missing_index % x_size]),
                float(y_values[first_missing_index // x_size]),
            )
        raise ValueError(f"Map CSV is missing grid point {first_missing!r}")

    intensity_matrix = np.empty((y_values.size, x_values.size), dtype=float)
    counts_matrix = np.empty_like(intensity_matrix) if row_counts is not None else None
    for index, (x_value, y_value, z_value) in enumerate(zip(row_x, row_y, z_values)):
        ix = x_index[x_value]
        iy = y_index[y_value]
        intensity_matrix[iy, ix] = z_value
        if counts_matrix is not None:
            # ``counts`` validation (including non-negative values) remains in
            # MapData, so this path has the same contract as direct creation.
            counts_matrix[iy, ix] = float(row_counts[index])  # type: ignore[index]

    declared_kind = metadata_values.get("kind", "").strip()
    if declared_kind:
        resolved_declared_kind = _normalise_kind(declared_kind)
        if kind is not None and _normalise_kind(kind) != resolved_declared_kind:
            raise ValueError("Map CSV kind metadata conflicts with requested kind")
        resolved_kind = resolved_declared_kind
    else:
        resolved_kind = _normalise_kind(kind if kind is not None else "rsm")

    inferred_labels: dict[str, str] = {
        "x": str(x_header).strip(),
        "y": str(y_header).strip(),
        "intensity": str(z_header).strip(),
    }
    declared_labels = {
        key: metadata_values[f"{key}_label"]
        for key in ("x", "y", "intensity")
        if metadata_values.get(f"{key}_label", "")
    }
    provided_labels = _metadata_items(labels, "labels")
    for key, value in declared_labels.items():
        if key in provided_labels and provided_labels[key] != value:
            raise ValueError(f"Map CSV {key}_label metadata conflicts with requested labels")
    inferred_labels.update(declared_labels)
    inferred_labels.update(provided_labels)

    inferred_units = {
        key: metadata_values[f"{key}_unit"]
        for key in ("x", "y", "intensity")
        if metadata_values.get(f"{key}_unit", "")
    }
    provided_units = _metadata_items(units, "units")
    for key, value in inferred_units.items():
        if key in provided_units and provided_units[key] != value:
            raise ValueError(f"Map CSV {key}_unit metadata conflicts with requested units")
    inferred_units.update(provided_units)

    declared_source = metadata_values.get("source", "").strip()
    resolved_source = declared_source or source_path
    return MapData(
        kind=resolved_kind,
        x=x_values,
        y=y_values,
        intensity=intensity_matrix,
        counts=counts_matrix,
        labels=inferred_labels,
        units=inferred_units,
        source_path=resolved_source,
        x_label=x_label,
        y_label=y_label,
        intensity_label=intensity_label,
        x_unit=x_unit,
        y_unit=y_unit,
        intensity_unit=intensity_unit,
    )


def load_map_csv(
    path: str | Path,
    *,
    kind: str | None = None,
    labels: object | None = None,
    units: object | None = None,
    source_path: object | None = None,
    delimiter: str | None = None,
    require_regular: bool = False,
    x_label: object | None = None,
    y_label: object | None = None,
    intensity_label: object | None = None,
    x_unit: object | None = None,
    y_unit: object | None = None,
    intensity_unit: object | None = None,
    max_rows: int = DEFAULT_MAX_MAP_ROWS,
    max_cells: int = DEFAULT_MAX_MAP_CELLS,
) -> MapData:
    """Load and validate a long-form map CSV from disk."""

    file_path = Path(path)
    with file_path.open("r", encoding="utf-8-sig", newline="") as handle:
        return parse_map_csv(
            handle,
            kind=kind,
            labels=labels,
            units=units,
            source_path=file_path if source_path is None else source_path,
            delimiter=delimiter,
            require_regular=require_regular,
            x_label=x_label,
            y_label=y_label,
            intensity_label=intensity_label,
            x_unit=x_unit,
            y_unit=y_unit,
            intensity_unit=intensity_unit,
            max_rows=max_rows,
            max_cells=max_cells,
        )


def from_detector_raw(image: object, **kwargs: Any) -> MapData:
    """Module-level alias for :meth:`MapData.from_detector_raw`."""

    return MapData.from_detector_raw(image, **kwargs)


def from_cake(cake_result: object, **kwargs: Any) -> MapData:
    """Module-level alias for :meth:`MapData.from_cake`."""

    return MapData.from_cake(cake_result, **kwargs)


parse_long_form_csv = parse_map_csv
load_long_form_csv = load_map_csv


__all__ = [
    "MAP_KINDS",
    "MapData",
    "from_cake",
    "from_detector_raw",
    "load_long_form_csv",
    "load_map_csv",
    "parse_long_form_csv",
    "parse_map_csv",
]
