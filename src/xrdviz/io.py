from __future__ import annotations

import csv
import re
from dataclasses import dataclass
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


@dataclass(slots=True)
class SampleMetadata:
    filename: str
    label: str
    order: int = 0
    color: str = ""
    visible: bool = True
    offset: float = 0.0


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


def load_spectrum(path: str | Path, axis_kind: str = "two_theta", color: str | None = None) -> SpectrumLayer:
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8-sig")
    x_values, y_values = parse_spectrum_text(text)
    cleaned = clean_spectrum_rows(x_values, y_values)
    resolved_axis = detect_spectrum_axis(text) if axis_kind == "auto" else axis_kind
    return SpectrumLayer(
        name=file_path.stem,
        x=cleaned.x,
        y=cleaned.y,
        axis_kind=resolved_axis,
        color=color or OKABE_ITO[0],
        source_path=str(file_path),
        raw_x=cleaned.raw_x,
        raw_y=cleaned.raw_y,
        warnings=cleaned.warnings,
        removed_rows=cleaned.removed_rows,
    )


def detect_spectrum_axis(text: str) -> str:
    lowered = text.lower()
    if "d-spacing" in lowered or "d spacing" in lowered or "axis unit: a" in lowered:
        return "d"
    if "q" in lowered and ("a^-1" in lowered or "1/a" in lowered or "angstrom^-1" in lowered):
        return "q"
    return "two_theta"


def clean_spectrum_rows(x_values: list[float], y_values: list[float]) -> CleanedSpectrum:
    if len(x_values) != len(y_values):
        raise ValueError("x and y must have the same number of values")

    raw_x = [float(value) for value in x_values]
    raw_y = [float(value) for value in y_values]
    warnings: list[str] = []
    removed = 0
    rows: list[tuple[float, float]] = []
    seen_x: set[float] = set()

    for x_value, y_value in zip(raw_x, raw_y):
        if y_value < 0:
            removed += 1
            warnings.append("1 negative intensity row removed")
            continue
        if x_value in seen_x:
            removed += 1
            warnings.append("1 duplicate x row removed")
            continue
        seen_x.add(x_value)
        rows.append((x_value, y_value))

    rows.sort(key=lambda pair: pair[0])
    if len(rows) < 10:
        warnings.append("Very few valid data points were found")
    if not rows:
        raise ValueError("No valid spectrum rows remain after cleaning")

    x_clean = [x for x, _y in rows]
    y_clean = [y for _x, y in rows]
    return CleanedSpectrum(x_clean, y_clean, raw_x, raw_y, warnings, removed)


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
