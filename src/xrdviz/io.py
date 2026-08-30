from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path

from xrdviz.models import OKABE_ITO, PhaseLayer, PhasePeak, SpectrumLayer, normalize_axis_kind

SPLIT_RE = re.compile(r"[\s,;]+")


@dataclass(slots=True)
class CleanedSpectrum:
    x: list[float]
    y: list[float]
    raw_x: list[float]
    raw_y: list[float]
    warnings: list[str]
    removed_rows: int
    y_error: list[float] = field(default_factory=list)


@dataclass(slots=True)
class SampleMetadata:
    filename: str
    label: str
    order: int = 0
    color: str = ""
    visible: bool = True
    offset: float = 0.0


@dataclass(slots=True)
class _CleanedSpectrumTable:
    """The single-layer subset of the publication cleaned-data contract."""

    x: list[float]
    y: list[float]
    y_error: list[float]
    axis_kind: str
    sample: str
    source_file: str
    frame_index: int | None = None
    time_s: float | None = None
    temperature: float | None = None
    temperature_unit: str = ""
    group: str = ""
    color_value: float | None = None


def parse_spectrum_text(text: str) -> tuple[list[float], list[float]]:
    x_values: list[float] = []
    y_values: list[float] = []

    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        numeric = _numeric_tokens(line)
        if len(numeric) < 2:
            continue
        x_values.append(numeric[0])
        y_values.append(numeric[1])

    if not x_values:
        raise ValueError("No rows with at least two numeric columns were found")
    return x_values, y_values


def load_spectrum(path: str | Path, axis_kind: str = "auto", color: str | None = None) -> SpectrumLayer:
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8-sig")
    cleaned_table = _parse_cleaned_spectrum_table(text)
    if cleaned_table is not None:
        x_values = cleaned_table.x
        y_values = cleaned_table.y
        y_error = cleaned_table.y_error
        inferred_axis = cleaned_table.axis_kind
        name = cleaned_table.sample
        source_path = cleaned_table.source_file or str(file_path)
    else:
        structured = _parse_uncertainty_table(text)
        if structured is None:
            x_values, y_values = parse_spectrum_text(text)
            y_error = []
            inferred_axis = detect_spectrum_axis(text)
        else:
            x_values, y_values, y_error, inferred_axis = structured
        name = file_path.stem
        source_path = str(file_path)

    cleaned = clean_spectrum_rows(x_values, y_values, y_error=y_error or None)
    resolved_axis = inferred_axis if axis_kind == "auto" else axis_kind
    kwargs: dict[str, object] = {}
    if cleaned_table is not None:
        kwargs.update(
            {
                "frame_index": cleaned_table.frame_index,
                "time_s": cleaned_table.time_s,
                "temperature": cleaned_table.temperature,
                "temperature_unit": cleaned_table.temperature_unit,
                "group": cleaned_table.group,
                "color_value": cleaned_table.color_value,
            }
        )
    return SpectrumLayer(
        name=name,
        x=cleaned.x,
        y=cleaned.y,
        axis_kind=resolved_axis,
        color=color or OKABE_ITO[0],
        source_path=source_path,
        raw_x=cleaned.raw_x,
        raw_y=cleaned.raw_y,
        y_error=cleaned.y_error,
        warnings=cleaned.warnings,
        removed_rows=cleaned.removed_rows,
        **kwargs,
    )


def _parse_cleaned_spectrum_table(text: str) -> _CleanedSpectrumTable | None:
    """Parse one layer from ``cleaned_xrd_data.csv``.

    The publication CSV is a long-form table and can contain several samples.
    ``load_spectrum`` deliberately accepts only a single sample so it cannot
    silently merge unrelated scans.  Callers with multiple samples should use
    the project JSON (which preserves the layer boundaries).
    """

    meaningful = [
        line for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")
    ]
    if not meaningful or ("," not in meaningful[0] and ";" not in meaningful[0]):
        return None
    delimiter = "," if meaningful[0].count(",") >= meaningful[0].count(";") else ";"
    reader = csv.DictReader(StringIO("\n".join(meaningful)), delimiter=delimiter)
    if reader.fieldnames is None:
        return None
    headers = _unique_columns(reader.fieldnames, "Spectrum CSV")
    if "sample" not in headers and "axis_kind" not in headers:
        return None
    required = {"sample", "axis_kind", "x", "intensity"}
    missing = sorted(required - set(headers))
    if missing:
        raise ValueError(
            "Cleaned spectrum CSV must include " + ", ".join(missing) + " columns"
        )

    uncertainty_header = _single_header(
        headers, ("uncertainty", "sigma", "error", "esd", "std"), "uncertainty"
    )
    metadata_headers = {
        key: headers[key]
        for key in (
            "sample",
            "source_file",
            "axis_kind",
            "frame_index",
            "time_s",
            "temperature",
            "temperature_unit",
            "group",
            "color_value",
        )
        if key in headers
    }
    if "source_path" in headers:
        if "source_file" in metadata_headers:
            raise ValueError("Cleaned spectrum CSV has duplicate source metadata columns")
        metadata_headers["source_file"] = headers["source_path"]

    x_values: list[float] = []
    y_values: list[float] = []
    raw_errors: list[float | None] = []
    samples: list[str] = []
    axes: list[str] = []
    source_files: list[str] = []
    optional_text: dict[str, list[str]] = {
        key: []
        for key in ("temperature_unit", "group")
        if key in metadata_headers
    }
    optional_numbers: dict[str, list[float | None]] = {
        key: []
        for key in ("frame_index", "time_s", "temperature", "color_value")
        if key in metadata_headers
    }

    for row_number, row in enumerate(reader, start=2):
        if None in row:
            raise ValueError(
                f"Cleaned spectrum CSV row {row_number} has more fields than the header"
            )
        if not any(str(value or "").strip() for value in row.values()):
            continue
        sample = str(row.get(metadata_headers["sample"], "") or "").strip()
        if not sample:
            raise ValueError(f"Cleaned spectrum CSV row {row_number} has an empty sample")
        axis_text = str(row.get(metadata_headers["axis_kind"], "") or "").strip()
        if not axis_text:
            raise ValueError(
                f"Cleaned spectrum CSV row {row_number} has an empty axis_kind"
            )
        try:
            parsed_axis = normalize_axis_kind(axis_text)
        except ValueError as exc:
            raise ValueError(
                f"Cleaned spectrum CSV row {row_number} has an invalid axis_kind"
            ) from exc
        samples.append(sample)
        axes.append(parsed_axis)
        source_files.append(
            str(row.get(metadata_headers.get("source_file", ""), "") or "").strip()
        )
        x_values.append(_table_number(row.get(headers["x"]), row_number, headers["x"]))
        y_values.append(
            _table_number(row.get(headers["intensity"]), row_number, headers["intensity"])
        )
        if uncertainty_header is None:
            raw_errors.append(None)
        else:
            raw_error = row.get(uncertainty_header)
            raw_errors.append(
                None
                if raw_error is None or not str(raw_error).strip()
                else _table_number(raw_error, row_number, uncertainty_header)
            )
        for key, header in metadata_headers.items():
            if key in optional_text:
                optional_text[key].append(str(row.get(header, "") or "").strip())
            elif key in optional_numbers:
                raw_value = row.get(header)
                optional_numbers[key].append(
                    None
                    if raw_value is None or not str(raw_value).strip()
                    else _table_number(raw_value, row_number, header)
                )

    if not x_values:
        raise ValueError("Cleaned spectrum CSV does not contain data rows")
    if len(set(samples)) != 1:
        raise ValueError(
            "Cleaned spectrum CSV contains multiple samples; load the project JSON instead"
        )
    if len(set(axes)) != 1:
        raise ValueError("Cleaned spectrum CSV contains multiple axis_kind values")
    if len(set(source_files)) > 1:
        raise ValueError("Cleaned spectrum CSV contains conflicting source_file values")
    if any(value is None for value in raw_errors) and any(
        value is not None for value in raw_errors
    ):
        raise ValueError(
            "Cleaned spectrum CSV uncertainty must be numeric for every row or blank for every row"
        )
    for key, values in optional_text.items():
        if len(set(values)) > 1:
            raise ValueError(f"Cleaned spectrum CSV contains conflicting {key} values")
    for key, values in optional_numbers.items():
        if len(set(values)) > 1:
            raise ValueError(f"Cleaned spectrum CSV contains conflicting {key} values")

    frame_index_value = _consistent_optional(optional_numbers.get("frame_index", []))
    if frame_index_value is not None and not frame_index_value.is_integer():
        raise ValueError("Cleaned spectrum CSV frame_index values must be integers")
    return _CleanedSpectrumTable(
        x=x_values,
        y=y_values,
        y_error=[value for value in raw_errors if value is not None],
        axis_kind=axes[0],
        sample=samples[0],
        source_file=source_files[0],
        frame_index=None if frame_index_value is None else int(frame_index_value),
        time_s=_consistent_optional(optional_numbers.get("time_s", [])),
        temperature=_consistent_optional(optional_numbers.get("temperature", [])),
        temperature_unit=_consistent_text(optional_text.get("temperature_unit", [])),
        group=_consistent_text(optional_text.get("group", [])),
        color_value=_consistent_optional(optional_numbers.get("color_value", [])),
    )


def _unique_columns(fieldnames: list[str], context: str) -> dict[str, str]:
    headers: dict[str, str] = {}
    for raw_header in fieldnames:
        if raw_header is None:
            raise ValueError(f"{context} contains an unnamed column")
        normalized = _normalized_column(raw_header)
        if not normalized:
            raise ValueError(f"{context} headers must not be empty")
        if normalized in headers:
            raise ValueError(f"{context} has duplicate header {raw_header!r}")
        headers[normalized] = raw_header
    return headers


def _single_header(
    headers: dict[str, str], aliases: tuple[str, ...], role: str
) -> str | None:
    candidates = [headers[alias] for alias in aliases if alias in headers]
    if len(candidates) > 1:
        raise ValueError(
            f"Spectrum CSV has multiple {role} columns: {', '.join(candidates)}"
        )
    return candidates[0] if candidates else None


def _consistent_text(values: list[str]) -> str:
    if not values:
        return ""
    return values[0]


def _consistent_optional(values: list[float | None]) -> float | None:
    if not values:
        return None
    return values[0]


def detect_spectrum_axis(text: str) -> str:
    lowered = text.lower()
    if "d-spacing" in lowered or "d spacing" in lowered or "axis unit: a" in lowered:
        return "d"
    if "q" in lowered and ("a^-1" in lowered or "1/a" in lowered or "angstrom^-1" in lowered):
        return "q"
    return "two_theta"


def clean_spectrum_rows(
    x_values: list[float],
    y_values: list[float],
    *,
    y_error: list[float] | None = None,
) -> CleanedSpectrum:
    if len(x_values) != len(y_values):
        raise ValueError("x and y must have the same number of values")
    if y_error and len(y_error) != len(y_values):
        raise ValueError("Uncertainty values must match the spectrum length")

    raw_x = [float(value) for value in x_values]
    raw_y = [float(value) for value in y_values]
    warnings: list[str] = []
    removed = 0
    raw_error = [float(value) for value in y_error] if y_error is not None else []
    rows: list[tuple[float, float, float | None]] = []
    seen_x: set[float] = set()

    errors = raw_error if raw_error else [None] * len(raw_y)
    for x_value, y_value, error_value in zip(raw_x, raw_y, errors):
        if y_value < 0:
            removed += 1
            warnings.append("1 negative intensity row removed")
            continue
        if x_value in seen_x:
            removed += 1
            warnings.append("1 duplicate x row removed")
            continue
        seen_x.add(x_value)
        rows.append((x_value, y_value, error_value))

    rows.sort(key=lambda pair: pair[0])
    if len(rows) < 10:
        warnings.append("Very few valid data points were found")
    if not rows:
        raise ValueError("No valid spectrum rows remain after cleaning")

    x_clean = [x for x, _y, _error in rows]
    y_clean = [y for _x, y, _error in rows]
    error_clean = [float(error) for _x, _y, error in rows if error is not None]
    return CleanedSpectrum(x_clean, y_clean, raw_x, raw_y, warnings, removed, error_clean)


def _parse_uncertainty_table(text: str) -> tuple[list[float], list[float], list[float], str] | None:
    meaningful = [line for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    if not meaningful or ("," not in meaningful[0] and ";" not in meaningful[0]):
        return None
    delimiter = "," if meaningful[0].count(",") >= meaningful[0].count(";") else ";"
    reader = csv.DictReader(StringIO("\n".join(meaningful)), delimiter=delimiter)
    if reader.fieldnames is None:
        return None
    headers = _unique_columns(reader.fieldnames, "Spectrum CSV")
    uncertainty_header = _first_column(headers, ("sigma", "uncertainty", "error", "esd", "std"))
    if uncertainty_header is None:
        return None
    x_header, inferred_axis = _spectrum_x_column(headers)
    y_header = _first_column(headers, ("intensity", "y", "counts", "count", "i"))
    if y_header is None:
        raise ValueError("Spectrum CSV with uncertainty must include an intensity column")
    x_values: list[float] = []
    y_values: list[float] = []
    raw_errors: list[float | None] = []
    for row_number, row in enumerate(reader, start=2):
        if None in row:
            raise ValueError(
                f"Spectrum CSV row {row_number} has more fields than the header"
            )
        if not any(str(value or "").strip() for value in row.values()):
            continue
        x_values.append(_table_number(row.get(x_header), row_number, x_header))
        y_values.append(_table_number(row.get(y_header), row_number, y_header))
        raw_error = row.get(uncertainty_header)
        raw_errors.append(
            None
            if raw_error is None or not str(raw_error).strip()
            else _table_number(raw_error, row_number, uncertainty_header)
        )
    if not x_values:
        raise ValueError("Spectrum CSV with uncertainty does not contain data rows")
    if any(value is None for value in raw_errors) and any(
        value is not None for value in raw_errors
    ):
        raise ValueError(
            "Spectrum CSV uncertainty must be numeric for every row or blank for every row"
        )
    y_error = [value for value in raw_errors if value is not None]
    return x_values, y_values, y_error, inferred_axis


def _normalized_column(value: str) -> str:
    normalized = str(value).strip().lower().replace("2θ", "2theta")
    return re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")


def _first_column(headers: dict[str, str], aliases: tuple[str, ...]) -> str | None:
    return next((headers[alias] for alias in aliases if alias in headers), None)


def _spectrum_x_column(headers: dict[str, str]) -> tuple[str, str]:
    candidates = [
        (headers[alias], axis_kind)
        for alias, axis_kind in (
        ("two_theta", "two_theta"),
        ("2theta", "two_theta"),
        ("d_spacing", "d"),
        ("d", "d"),
        ("q", "q"),
        ("x", "two_theta"),
        )
        if alias in headers
    ]
    if len(candidates) > 1:
        raise ValueError(
            "Spectrum CSV has multiple x-axis columns: "
            + ", ".join(header for header, _axis in candidates)
        )
    if candidates:
        return candidates[0]
    raise ValueError("Spectrum CSV with uncertainty must include an x, 2theta, d, or q column")


def _table_number(value: object, row_number: int, column: str) -> float:
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Spectrum CSV row {row_number} column {column!r} must be numeric") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"Spectrum CSV row {row_number} column {column!r} must be finite")
    return parsed


def load_sample_labels_csv(text: str) -> dict[str, SampleMetadata]:
    rows = csv.DictReader(StringIO(text))
    required = {"filename", "label"}
    if rows.fieldnames is None or not required.issubset({name.strip() for name in rows.fieldnames}):
        raise ValueError("sample_labels.csv must include filename and label columns")

    mapping: dict[str, SampleMetadata] = {}
    for index, row in enumerate(rows, start=1):
        filename = str(row.get("filename", "")).strip()
        if not filename:
            continue
        mapping[filename] = SampleMetadata(
            filename=filename,
            label=str(row.get("label", "")).strip() or Path(filename).stem,
            order=_int_or_default(row.get("order"), index),
            color=str(row.get("color", "")).strip(),
            visible=_bool_or_default(row.get("visible"), True),
            offset=_float_or_default(row.get("offset"), 0.0),
        )
    return mapping


def apply_sample_metadata(layers: list[SpectrumLayer], metadata: dict[str, SampleMetadata]) -> None:
    for layer in layers:
        key = Path(layer.source_path).name if layer.source_path else f"{layer.name}.xy"
        item = metadata.get(key)
        if item is None:
            continue
        layer.name = item.label
        layer.order = item.order
        layer.visible = item.visible
        layer.offset = item.offset
        if item.color:
            layer.color = item.color
    layers.sort(key=lambda layer: (layer.order or 0, layer.name))


def load_reference_peaks_csv(text: str, source_path: str = "reference_peaks.csv") -> PhaseLayer:
    phases = load_reference_peaks_csv_many(text, source_path=source_path)
    if not phases:
        raise ValueError("No valid reference peaks found")
    return phases[0]


def load_reference_peaks_csv_many(text: str, source_path: str = "reference_peaks.csv") -> list[PhaseLayer]:
    rows = csv.DictReader(StringIO(text))
    if rows.fieldnames is None or "position" not in {name.strip() for name in rows.fieldnames}:
        raise ValueError("reference_peaks.csv must include a position column")

    grouped: dict[str, PhaseLayer] = {}
    for row in rows:
        try:
            position = float(str(row.get("position", "")).strip())
        except ValueError:
            continue
        phase_name = str(row.get("phase", "")).strip() or "Reference"
        source_axis = normalize_axis_kind(str(row.get("source_axis", "") or "two_theta"))
        color = str(row.get("color", "")).strip() or OKABE_ITO[len(grouped) % len(OKABE_ITO)]
        shape = str(row.get("shape", "")).strip() or "line"
        layer = grouped.get(phase_name)
        if layer is None:
            layer = PhaseLayer(
                name=phase_name,
                phase=phase_name,
                source_path=source_path,
                source_type="reference_csv",
                source_axis=source_axis,
                color=color,
                marker_shape=shape,
                show_guides=True,
                label_policy="characteristic",
            )
            grouped[phase_name] = layer
        layer.peaks.append(
            PhasePeak(
                two_theta=position,
                intensity=_float_or_default(row.get("intensity"), 100.0),
                hkl=str(row.get("hkl", "")).strip(),
                label=str(row.get("label", "")).strip(),
                source_axis=source_axis,
                relative_intensity=_float_or_none(row.get("intensity")),
            )
        )
    return list(grouped.values())


def load_rigaku_peaks_csv(text: str, source_path: str = "peaks.csv", phase_name: str | None = None) -> PhaseLayer:
    rows = list(csv.DictReader(StringIO(text)))
    if not rows:
        raise ValueError("Rigaku peaks CSV did not contain rows")
    headers = rows[0].keys()
    angle_header = _find_rigaku_angle_column(headers)
    peaks: list[PhasePeak] = []
    for row in rows:
        try:
            position = float(str(row.get(angle_header, "")).strip())
        except ValueError:
            continue
        peaks.append(PhasePeak(two_theta=position, intensity=_first_numeric(row, exclude={angle_header}) or 100.0))
    if not peaks:
        raise ValueError("No numeric peak positions found in Rigaku peaks CSV")
    name = phase_name or Path(source_path).stem
    return PhaseLayer(name=name, phase=name, source_path=source_path, source_type="rigaku_peaks_csv", peaks=peaks)


def _numeric_tokens(line: str) -> list[float]:
    values: list[float] = []
    for token in SPLIT_RE.split(line):
        if not token:
            continue
        try:
            values.append(float(token))
        except ValueError:
            continue
    return values


def _int_or_default(value: object, default: int) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def _float_or_default(value: object, default: float) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def _float_or_none(value: object) -> float | None:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _bool_or_default(value: object, default: bool) -> bool:
    if value is None:
        return default
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _find_rigaku_angle_column(headers) -> str:
    for header in headers:
        normalized = header.lower()
        if "angle" in normalized or "theta" in normalized or "position" in normalized:
            return header
    return next(iter(headers))


def _first_numeric(row: dict[str, str], exclude: set[str]) -> float | None:
    for key, value in row.items():
        if key in exclude:
            continue
        parsed = _float_or_none(value)
        if parsed is not None:
            return parsed
    return None
