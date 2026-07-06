from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from xrdviz.cif import phase_peak_position_for_axis
from xrdviz.models import ProjectState
from xrdviz.plot.renderer import export_project


@dataclass(slots=True)
class PublicationOutputs:
    figure: Path
    cleaned_data: Path
    peak_table: Path
    report: Path


def make_peak_table_rows(state: ProjectState) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for phase in state.phases:
        if not phase.visible:
            continue
        for peak in phase.peaks:
            rows.append(
                {
                    "phase": phase.phase or phase.name,
                    "source": phase.source_path,
                    "source_type": phase.source_type,
                    "card_id": phase.card_id,
                    "label": peak.label,
                    "hkl": peak.hkl,
                    "two_theta": phase_peak_position_for_axis(phase, peak, "two_theta", state.settings.energy_kev),
                    "d": phase_peak_position_for_axis(phase, peak, "d", state.settings.energy_kev),
                    "q": phase_peak_position_for_axis(phase, peak, "q", state.settings.energy_kev),
                    "intensity": peak.intensity,
                    "lattice_a": phase.lattice_a or "",
                    "auto_calibrated": phase.auto_calibrated,
                    "calibration_confidence": phase.calibration_confidence,
                    "calibration_error": phase.calibration_error if phase.calibration_error is not None else "",
                }
            )
    return rows


def export_peak_table(state: ProjectState, output_dir: str | Path) -> Path:
    output = Path(output_dir) / "reference_peak_table.csv"
    rows = make_peak_table_rows(state)
    fieldnames = [
        "phase",
        "source",
        "source_type",
        "card_id",
        "label",
        "hkl",
        "two_theta",
        "d",
        "q",
        "intensity",
        "lattice_a",
        "auto_calibrated",
        "calibration_confidence",
        "calibration_error",
    ]
    _write_rows(output, fieldnames, rows)
    return output


def export_cleaned_data(state: ProjectState, output_dir: str | Path) -> Path:
    output = Path(output_dir) / "cleaned_xrd_data.csv"
    rows: list[dict[str, object]] = []
    for layer in state.spectra:
        for x_value, y_value in zip(layer.x, layer.y):
            rows.append(
                {
                    "sample": layer.name,
                    "source_file": layer.source_path,
                    "axis_kind": layer.axis_kind,
                    "x": x_value,
                    "intensity": y_value,
                    "frame_index": layer.frame_index if layer.frame_index is not None else "",
                    "time_s": layer.time_s if layer.time_s is not None else "",
                    "temperature": layer.temperature if layer.temperature is not None else "",
                    "temperature_unit": layer.temperature_unit,
                    "group": layer.group,
                    "color_value": layer.color_value if layer.color_value is not None else "",
                }
            )
    _write_rows(
        output,
        [
            "sample",
            "source_file",
            "axis_kind",
            "x",
            "intensity",
            "frame_index",
            "time_s",
            "temperature",
            "temperature_unit",
            "group",
            "color_value",
        ],
        rows,
    )
    return output


def write_publication_report(
    state: ProjectState,
    output_dir: str | Path,
    *,
    exported_figure: str | Path,
    cleaned_data: str | Path | None = None,
    peak_table: str | Path | None = None,
) -> Path:
    output = Path(output_dir) / "xrd_plot_report.md"
    settings = state.settings
    stacking_enabled = settings.stack_enabled or settings.view_mode in {"stack", "gradient_stack"}
    lines = [
        "# XRD Plot Report",
        "",
        "## Figure",
        "",
        f"- exported figure: `{exported_figure}`",
        f"- figure size: {settings.figure_width_in:.3f} in x {settings.figure_height_in:.3f} in",
        f"- dpi: {settings.dpi}",
        f"- x axis: {settings.x_axis}",
        f"- energy: {settings.energy_kev:g} keV",
        f"- template: {settings.template_name}",
        f"- legend location: {settings.legend_location}",
        f"- view mode: {settings.view_mode}",
        f"- sort by: {settings.sort_by}",
        f"- color by: {settings.color_by}",
        f"- colormap: {settings.colormap}",
        f"- colorbar: {'enabled' if settings.show_colorbar else 'disabled'}",
        f"- show every N spectra: {settings.show_every_n}",
        f"- heatmap points: {settings.heatmap_points}",
        f"- normalization: {'enabled' if settings.normalize else 'disabled'}",
        f"- log scale: {'enabled' if settings.log_scale else 'disabled'}",
        f"- stacking: {'enabled' if stacking_enabled else 'disabled'}",
        "",
        "## Spectra",
        "",
    ]
    if not state.spectra:
        lines.append("- No spectra loaded.")
    for layer in state.spectra:
        metadata = []
        if layer.frame_index is not None:
            metadata.append(f"frame={layer.frame_index}")
        if layer.time_s is not None:
            metadata.append(f"time_s={layer.time_s:g}")
        if layer.temperature is not None:
            suffix = layer.temperature_unit or ""
            metadata.append(f"temperature={layer.temperature:g}{suffix}")
        if layer.group:
            metadata.append(f"group={layer.group}")
        if layer.color_value is not None:
            metadata.append(f"color_value={layer.color_value:g}")
        metadata_text = f" ({', '.join(metadata)})" if metadata else ""
        lines.append(f"- `{layer.source_path}` -> `{layer.name}`: {len(layer.x)} cleaned points{metadata_text}")
        if layer.removed_rows:
            lines.append(f"  - removed rows: {layer.removed_rows}")
        for warning in layer.warnings:
            lines.append(f"  - warning: {warning}")

    lines.extend(["", "## Reference Peaks", ""])
    if not state.phases:
        lines.append("- No reference peaks loaded.")
    for phase in state.phases:
        source = f"{phase.card_id} ({phase.source_path})" if phase.card_id else phase.source_path
        line = f"- {phase.phase or phase.name}: {len(phase.peaks)} peaks from `{source}`"
        if phase.lattice_a is not None:
            line += f", a={phase.lattice_a:g} A"
        if phase.auto_calibrated:
            error = "" if phase.calibration_error is None else f", rms={phase.calibration_error:.4g} A"
            line += f", auto-calibrated a={phase.lattice_a:g}, confidence={phase.calibration_confidence}{error}"
        elif phase.calibration_confidence:
            line += f", calibration confidence={phase.calibration_confidence}"
        lines.append(line)
        for note in phase.calibration_notes:
            lines.append(f"  - calibration note: {note}")

    lines.extend(["", "## Outputs", ""])
    if cleaned_data is not None:
        lines.append(f"- cleaned data: `{cleaned_data}`")
    if peak_table is not None:
        lines.append(f"- peak table: `{peak_table}`")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def export_publication_bundle(state: ProjectState, output_dir: str | Path, *, figure_name: str = "xrd_figure.pdf") -> PublicationOutputs:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    figure = root / figure_name
    export_project(state, figure)
    cleaned_data = export_cleaned_data(state, root)
    peak_table = export_peak_table(state, root)
    report = write_publication_report(state, root, exported_figure=figure, cleaned_data=cleaned_data, peak_table=peak_table)
    return PublicationOutputs(figure=figure, cleaned_data=cleaned_data, peak_table=peak_table, report=report)


def _write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
