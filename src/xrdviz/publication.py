from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping

from xrdviz import __version__
from xrdviz.cif import phase_peak_position_for_axis
from xrdviz.compliance import nature_compliance_issues
from xrdviz.models import ProjectState, project_to_dict
from xrdviz.plot.renderer import export_project


@dataclass(slots=True)
class PublicationOutputs:
    figure: Path
    cleaned_data: Path
    peak_table: Path
    report: Path
    figures: tuple[Path, ...] = ()
    project: Path | None = None
    manifest: Path | None = None
    fit_data: Path | None = None
    fit_summary: Path | None = None
    map_data: Path | None = None
    derived_data: Path | None = None


def make_peak_table_rows(state: ProjectState) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for phase in state.phases:
        if not phase.visible:
            continue
        for peak in phase.peaks:
            rows.append(
                {
                    "phase": phase.phase or phase.name,
                    "source": _source_label(phase.source_path),
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
    include_uncertainty = any(layer.y_error for layer in state.spectra)
    fieldnames = [
        "sample",
        "source_file",
        "axis_kind",
        "x",
        "intensity",
    ]
    if include_uncertainty:
        fieldnames.append("uncertainty")
    fieldnames.extend(
        [
            "frame_index",
            "time_s",
            "temperature",
            "temperature_unit",
            "group",
            "color_value",
        ]
    )

    def rows() -> Iterable[dict[str, object]]:
        for layer in state.spectra:
            for index, (x_value, y_value) in enumerate(zip(layer.x, layer.y)):
                row: dict[str, object] = {
                    "sample": layer.name,
                    "source_file": _source_label(layer.source_path),
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
                if include_uncertainty:
                    row["uncertainty"] = layer.y_error[index] if layer.y_error else ""
                yield row

    _write_rows(output, fieldnames, rows())
    return output


def write_publication_report(
    state: ProjectState,
    output_dir: str | Path,
    *,
    exported_figure: str | Path,
    cleaned_data: str | Path | None = None,
    peak_table: str | Path | None = None,
    fit_data: str | Path | None = None,
    fit_summary: str | Path | None = None,
    map_data: str | Path | None = None,
    derived_data: str | Path | None = None,
    figures: Iterable[str | Path] | None = None,
    project: str | Path | None = None,
    manifest: str | Path | None = None,
    compliance_issues: Iterable[str] | None = None,
) -> Path:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    output = root / "xrd_plot_report.md"
    settings = state.settings
    stacking_enabled = settings.stack_enabled or settings.view_mode in {"stack", "gradient_stack"}
    issues = list(nature_compliance_issues(state) if compliance_issues is None else compliance_issues)
    figure_paths = list(figures) if figures is not None else [exported_figure]
    target_width_mm = float(settings.figure_width_in) * 25.4
    target_height_mm = float(settings.figure_height_in) * 25.4
    nature_status = "PASS" if not issues else "FAIL"
    lines = [
        "# XRD Plot Report",
        "",
        "## Figure",
        "",
        f"- exported figure: `{_display_path(exported_figure, root)}`",
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
        "## Nature compliance",
        "",
        f"- status: {nature_status}",
        f"- export canvas: {target_width_mm:.3f} mm x {target_height_mm:.3f} mm at {settings.dpi} dpi",
    ]
    if issues:
        lines.append("- issues:")
        lines.extend(f"  - {issue}" for issue in issues)
    else:
        lines.append("- issues: none")

    lines.extend(
        [
            "",
            "## Figures",
            "",
            "- content note: heatmap and 2D-map PDF/SVG exports contain raster image content and are not claimed to be all-vector.",
        ]
    )
    for figure_path in figure_paths:
        display_path = _display_path(figure_path, root)
        suffix = Path(figure_path).suffix.lower().lstrip(".") or "unknown"
        content = _figure_content(settings.view_mode, suffix)
        lines.append(
            f"- `{display_path}` ({suffix.upper()}; {content}; "
            f"target {target_width_mm:.3f} mm x {target_height_mm:.3f} mm)"
        )

    lines.extend(
        [
            "",
            "## Spectra",
        ]
    )
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
        lines.append(f"- `{_source_label(layer.source_path)}` -> `{layer.name}`: {len(layer.x)} cleaned points{metadata_text}")
        if layer.removed_rows:
            lines.append(f"  - removed rows: {layer.removed_rows}")
        for warning in layer.warnings:
            lines.append(f"  - warning: {warning}")

    lines.extend(["", "## Pattern fit", ""])
    if state.fit is None:
        lines.append("- No observed/calculated fit result loaded.")
    else:
        lines.append(f"- name: `{state.fit.name}`")
        lines.append(f"- kind: {state.fit.fit_kind}")
        lines.append(f"- source: `{_source_label(state.fit.source_path)}`")
        lines.append(f"- axis: {state.fit.axis_kind}")
        if state.fit.wavelength_angstrom is not None:
            lines.append(
                f"- wavelength: {state.fit.wavelength_angstrom:.8g} angstrom"
            )
        lines.append(f"- points: {len(state.fit.x)}")
        lines.append(f"- components: {len(state.fit.components)}")
        lines.append(f"- uncertainty: {'sigma provided' if state.fit.sigma else 'not provided'}")
        if state.fit.converged is not None:
            lines.append(f"- convergence: {'converged' if state.fit.converged else 'not converged'}")
        if state.fit.fit_message:
            lines.append(f"- optimizer message: {state.fit.fit_message}")
        if state.fit.rp is not None:
            lines.append(f"- Rp: {state.fit.rp:.6g}%")
        if state.fit.rwp is not None:
            lines.append(f"- Rwp: {state.fit.rwp:.6g}%")
        lines.append("- GoF: not reported (the number of independently refined parameters was not provided)")
        for component in state.fit.components:
            details = []
            for field_name in ("center", "fwhm", "area", "amplitude", "eta"):
                value = getattr(component, field_name, None)
                if value is not None:
                    details.append(f"{field_name}={value:.6g}")
            if getattr(component, "profile", ""):
                details.append(f"profile={component.profile}")
            if details:
                lines.append(f"  - {component.name}: {', '.join(details)}")

    lines.extend(["", "## Two-dimensional map", ""])
    if state.map_data is None:
        lines.append("- No detector, cake, RSM, or pole-figure map loaded.")
    else:
        lines.append(f"- kind: {state.map_data.kind}")
        lines.append(f"- source: `{_source_label(state.map_data.source_path)}`")
        lines.append(f"- grid: {len(state.map_data.y)} rows x {len(state.map_data.x)} columns")
        lines.append(
            f"- axes: {state.map_data.x_label} ({state.map_data.x_unit}); "
            f"{state.map_data.y_label} ({state.map_data.y_unit})"
        )
        for key, value in state.map_data.metadata.items():
            lines.append(f"- {key}: {json.dumps(value, ensure_ascii=False, sort_keys=True)}")
        lines.append("- interpretation boundary: map rendering does not perform phase identification or structure refinement")

    lines.extend(["", "## Derived analysis", ""])
    if state.derived_plot is None:
        lines.append("- No Scherrer, Williamson-Hall, or rocking-curve result loaded.")
    else:
        lines.append(f"- kind: {state.derived_plot.kind}")
        lines.append(f"- source: `{_source_label(state.derived_plot.source)}`")
        lines.append(f"- points: {len(state.derived_plot.x)}")
        for key, value in state.derived_plot.metrics.items():
            lines.append(f"- {key}: {value}")
        lines.append("- uncertainty: not reported unless explicitly present in the imported source")

    lines.extend(["", "## Reference Peaks", ""])
    if not state.phases:
        lines.append("- No reference peaks loaded.")
    for phase in state.phases:
        source_label = _source_label(phase.source_path)
        source = f"{phase.card_id} ({source_label})" if phase.card_id else source_label
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
        lines.append(f"- cleaned data: `{_display_path(cleaned_data, root)}`")
    if peak_table is not None:
        lines.append(f"- peak table: `{_display_path(peak_table, root)}`")
    if fit_data is not None:
        lines.append(f"- pattern fit data: `{_display_path(fit_data, root)}`")
    if fit_summary is not None:
        lines.append(f"- peak fit summary: `{_display_path(fit_summary, root)}`")
    if map_data is not None:
        lines.append(f"- map data: `{_display_path(map_data, root)}`")
    if derived_data is not None:
        lines.append(f"- derived analysis data: `{_display_path(derived_data, root)}`")
    if project is not None:
        lines.append(f"- project snapshot: `{_display_path(project, root)}`")
    if manifest is not None:
        lines.append(f"- manifest: `{_display_path(manifest, root)}`")
    report_text = "\n".join(lines) + "\n"
    _atomic_write(output, lambda handle: handle.write(report_text))
    return output


def export_fit_data(state: ProjectState, output_dir: str | Path) -> Path:
    if state.fit is None:
        raise ValueError("No observed/calculated fit result is loaded")
    output = Path(output_dir) / "pattern_fit_data.csv"
    component_fields = _component_field_names(state.fit.components)
    axis_field = {
        "two_theta": "2theta",
        "q": "q",
        "d": "d_spacing",
    }[state.fit.axis_kind]
    fieldnames = [axis_field, "observed", "calculated", "difference"]
    if state.fit.wavelength_angstrom is not None:
        fieldnames.append("wavelength_angstrom")
    if state.fit.sigma:
        fieldnames.append("sigma")
    if state.fit.background:
        fieldnames.append("background")
    fieldnames.extend(component_fields)
    difference = state.fit.difference

    def rows() -> Iterable[dict[str, object]]:
        for index, x_value in enumerate(state.fit.x):
            row: dict[str, object] = {
                axis_field: x_value,
                "observed": state.fit.observed[index],
                "calculated": state.fit.calculated[index],
                "difference": difference[index],
            }
            if state.fit.wavelength_angstrom is not None:
                row["wavelength_angstrom"] = state.fit.wavelength_angstrom
            if state.fit.sigma:
                row["sigma"] = state.fit.sigma[index]
            if state.fit.background:
                row["background"] = state.fit.background[index]
            for field_name, component in zip(component_fields, state.fit.components):
                row[field_name] = component.y[index]
            yield row

    _write_rows(output, fieldnames, rows())
    return output


def export_fit_summary(state: ProjectState, output_dir: str | Path) -> Path:
    if state.fit is None:
        raise ValueError("No observed/calculated fit result is loaded")
    output = Path(output_dir) / "peak_fit_summary.csv"
    fieldnames = [
        "axis_kind",
        "wavelength_angstrom",
        "name",
        "profile",
        "center",
        "fwhm",
        "area",
        "amplitude",
        "eta",
    ]
    rows = [
        {
            "axis_kind": state.fit.axis_kind,
            "wavelength_angstrom": (
                ""
                if state.fit.wavelength_angstrom is None
                else state.fit.wavelength_angstrom
            ),
            "name": component.name,
            "profile": component.profile,
            "center": "" if component.center is None else component.center,
            "fwhm": "" if component.fwhm is None else component.fwhm,
            "area": "" if component.area is None else component.area,
            "amplitude": "" if component.amplitude is None else component.amplitude,
            "eta": "" if component.eta is None else component.eta,
        }
        for component in state.fit.components
    ]
    _write_rows(output, fieldnames, rows)
    return output


def export_map_data(state: ProjectState, output_dir: str | Path) -> Path:
    if state.map_data is None:
        raise ValueError("No two-dimensional map is loaded")
    output = Path(output_dir) / "map_data.csv"
    fieldnames = [
        "kind",
        "x",
        "y",
        "intensity",
        "counts",
        "x_label",
        "y_label",
        "intensity_label",
        "x_unit",
        "y_unit",
        "intensity_unit",
        "source_file",
    ]

    def rows() -> Iterable[dict[str, object]]:
        map_data = state.map_data
        counts = map_data.counts
        for row_index, y_value in enumerate(map_data.y):
            for column_index, x_value in enumerate(map_data.x):
                count = None if counts is None else float(counts[row_index, column_index])
                populated = count is None or count > 0.0
                yield {
                    "kind": map_data.kind,
                    "x": float(x_value),
                    "y": float(y_value),
                    "intensity": (
                        float(map_data.intensity[row_index, column_index]) if populated else ""
                    ),
                    "counts": "" if count is None else count,
                    "x_label": map_data.x_label,
                    "y_label": map_data.y_label,
                    "intensity_label": map_data.intensity_label,
                    "x_unit": map_data.x_unit,
                    "y_unit": map_data.y_unit,
                    "intensity_unit": map_data.intensity_unit,
                    "source_file": _source_label(map_data.source_path),
                }

    _write_rows(output, fieldnames, rows())
    return output


def export_derived_data(state: ProjectState, output_dir: str | Path) -> Path:
    if state.derived_plot is None:
        raise ValueError("No derived analysis result is loaded")
    output = Path(output_dir) / "derived_analysis_data.csv"
    rows: list[dict[str, object]] = []
    for x_value, y_value in zip(state.derived_plot.x, state.derived_plot.y):
        rows.append({"series": "data", "x": x_value, "y": y_value, "metric": "", "value": ""})
    for x_value, y_value in state.derived_plot.fit_line:
        rows.append({"series": "fit", "x": x_value, "y": y_value, "metric": "", "value": ""})
    for key, value in state.derived_plot.metrics.items():
        rows.append({"series": "metric", "x": "", "y": "", "metric": key, "value": value})
    _write_rows(output, ["series", "x", "y", "metric", "value"], rows)
    return output


def export_publication_bundle(
    state: ProjectState,
    output_dir: str | Path,
    *,
    figure_name: str = "xrd_figure.pdf",
) -> PublicationOutputs:
    """Export a deterministic, self-describing publication bundle.

    ``figure_name`` remains the primary figure path for backwards
    compatibility.  The other three formats use the same stem and all formats
    are emitted in the fixed PDF, SVG, TIFF, PNG order.
    """

    root = Path(output_dir)
    _validate_bundle_root(root)
    previous_owned = _previous_bundle_paths(root)
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{root.name or 'xrdviz'}-staging-", dir=str(root.parent))
    )
    try:
        figure_map = _figure_paths(staging, figure_name)
        primary_format = _primary_format(figure_name)
        figures = tuple(figure_map[format_name] for format_name in _ordered_figure_formats())

        for figure_path in figures:
            figure_path.parent.mkdir(parents=True, exist_ok=True)
            export_project(state, figure_path)

        cleaned_data = export_cleaned_data(state, staging)
        peak_table = export_peak_table(state, staging)
        fit_data = export_fit_data(state, staging) if state.fit is not None else None
        fit_summary = (
            export_fit_summary(state, staging)
            if state.fit is not None and state.fit.components
            else None
        )
        map_data = export_map_data(state, staging) if state.map_data is not None else None
        derived_data = export_derived_data(state, staging) if state.derived_plot is not None else None
        project = _write_project_snapshot(state, staging / "project.xrdviz.json")
        issues = nature_compliance_issues(state)
        manifest = staging / "publication_manifest.json"
        report = write_publication_report(
            state,
            staging,
            exported_figure=figure_map[primary_format],
            cleaned_data=cleaned_data,
            peak_table=peak_table,
            fit_data=fit_data,
            fit_summary=fit_summary,
            map_data=map_data,
            derived_data=derived_data,
            figures=figures,
            project=project,
            manifest=manifest,
            compliance_issues=issues,
        )
        manifest_data = _build_manifest(
            state,
            staging,
            figure_map=figure_map,
            cleaned_data=cleaned_data,
            peak_table=peak_table,
            fit_data=fit_data,
            fit_summary=fit_summary,
            map_data=map_data,
            derived_data=derived_data,
            project=project,
            report=report,
            manifest=manifest,
            issues=issues,
            primary_format=primary_format,
            source_root=Path.cwd(),
        )
        _write_stable_json(manifest, manifest_data)

        staged_paths = _staged_file_paths(staging)
        _publish_staged_files(staging, root, staged_paths, previous_owned)
        final_paths = {path: root / path.relative_to(staging) for path in staged_paths}
        final_figure_map = {
            format_name: final_paths[figure_map[format_name]]
            for format_name in _ordered_figure_formats()
        }
        return PublicationOutputs(
            figure=final_figure_map[primary_format],
            cleaned_data=final_paths[cleaned_data],
            peak_table=final_paths[peak_table],
            report=final_paths[report],
            figures=tuple(final_figure_map[format_name] for format_name in _ordered_figure_formats()),
            project=final_paths[project],
            manifest=final_paths[manifest],
            fit_data=None if fit_data is None else final_paths[fit_data],
            fit_summary=None if fit_summary is None else final_paths[fit_summary],
            map_data=None if map_data is None else final_paths[map_data],
            derived_data=None if derived_data is None else final_paths[derived_data],
        )
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _validate_bundle_root(root: Path) -> None:
    """Reject output roots that could redirect bundle writes through a reparse point."""

    _assert_no_reparse_ancestors(root.parent)
    if os.path.lexists(str(root)):
        if _is_reparse_point(root):
            raise ValueError("output_dir must not be a symlink or reparse point")
        if not root.is_dir():
            raise ValueError("output_dir must be a directory")


def _previous_bundle_paths(root: Path) -> set[Path]:
    """Return generated paths that a previous manifest can safely own.

    A manifest is data produced by a previous run, not an authority to remove
    arbitrary files.  Keep the fixed artifact names tied to their artifact
    kinds, and accept figure paths only when the complete, four-format figure
    set is internally consistent.
    """

    manifest = root / "publication_manifest.json"
    if not os.path.lexists(str(manifest)):
        return set()
    if _is_reparse_point(manifest) or not manifest.is_file():
        return set()
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return set()
    manifest_record = data.get("manifest") if isinstance(data, Mapping) else None
    manifest_owned: set[Path] = set()
    if (
        isinstance(manifest_record, Mapping)
        and manifest_record.get("path") == "publication_manifest.json"
    ):
        manifest_owned.add(Path("publication_manifest.json"))
    if not _looks_like_xrdviz_bundle_manifest(data):
        return manifest_owned
    owned: set[Path] = set(manifest_owned)

    figures = _validated_previous_figure_paths(data.get("figures"))
    owned.update(figures)

    artifacts = data.get("artifacts", [])
    if isinstance(artifacts, list):
        for record in artifacts:
            if not isinstance(record, Mapping):
                continue
            kind = record.get("kind")
            if not isinstance(kind, str):
                continue
            expected_name = _PREVIOUS_ARTIFACT_NAMES.get(kind)
            relative = _safe_relative_path(record.get("path"))
            if expected_name is not None and relative == Path(expected_name):
                owned.add(relative)
    return owned


def _looks_like_xrdviz_bundle_manifest(data: object) -> bool:
    """Require the generated application marker and fixed core artifacts."""

    if not isinstance(data, Mapping):
        return False
    application = data.get("application")
    if not isinstance(application, Mapping) or application.get("app") != "xrdviz":
        return False
    manifest_record = data.get("manifest")
    if not isinstance(manifest_record, Mapping) or manifest_record.get("path") != "publication_manifest.json":
        return False
    artifacts = data.get("artifacts")
    if not isinstance(artifacts, list):
        return False
    declared = {
        (record.get("kind"), record.get("path"))
        for record in artifacts
        if isinstance(record, Mapping)
    }
    required = {
        ("cleaned_csv", _PREVIOUS_ARTIFACT_NAMES["cleaned_csv"]),
        ("peak_csv", _PREVIOUS_ARTIFACT_NAMES["peak_csv"]),
        ("project", _PREVIOUS_ARTIFACT_NAMES["project"]),
        ("report", _PREVIOUS_ARTIFACT_NAMES["report"]),
    }
    return required <= declared


_PREVIOUS_ARTIFACT_NAMES = {
    "cleaned_csv": "cleaned_xrd_data.csv",
    "peak_csv": "reference_peak_table.csv",
    "project": "project.xrdviz.json",
    "report": "xrd_plot_report.md",
    "fit_csv": "pattern_fit_data.csv",
    "fit_summary_csv": "peak_fit_summary.csv",
    "map_csv": "map_data.csv",
    "derived_csv": "derived_analysis_data.csv",
}

_PREVIOUS_FIGURE_SUFFIXES = {
    "pdf": {".pdf"},
    "svg": {".svg"},
    "tiff": {".tif", ".tiff"},
    "png": {".png"},
}


def _validated_previous_figure_paths(value: object) -> set[Path]:
    """Validate a complete figure set before treating its paths as owned."""

    if not isinstance(value, list) or len(value) != len(_PREVIOUS_FIGURE_SUFFIXES):
        return set()
    by_format: dict[str, Path] = {}
    for record in value:
        if not isinstance(record, Mapping):
            return set()
        format_name = record.get("format")
        relative = _safe_relative_path(record.get("path"))
        if not isinstance(format_name, str) or format_name not in _PREVIOUS_FIGURE_SUFFIXES or relative is None:
            return set()
        if format_name in by_format or relative.suffix.lower() not in _PREVIOUS_FIGURE_SUFFIXES[format_name]:
            return set()
        by_format[format_name] = relative
    if set(by_format) != set(_PREVIOUS_FIGURE_SUFFIXES):
        return set()
    if sum(record.get("primary") is True for record in value) != 1:
        return set()
    figure_paths = list(by_format.values())
    if len({path.parent for path in figure_paths}) != 1:
        return set()
    if len({path.stem.casefold() for path in figure_paths}) != 1:
        return set()
    return set(figure_paths)


def _staged_file_paths(staging: Path) -> list[Path]:
    paths: list[Path] = []
    for path in staging.rglob("*"):
        if path.is_dir():
            if _is_reparse_point(path):
                raise ValueError("staging directory contains a reparse point")
            continue
        if _is_reparse_point(path):
            raise ValueError("staging bundle contains a symlink or reparse point")
        if not path.is_file():
            raise ValueError("staging bundle contains a non-file artifact")
        paths.append(path)
    return sorted(paths, key=lambda path: path.relative_to(staging).as_posix())


def _publish_staged_files(
    staging: Path,
    root: Path,
    staged_paths: Iterable[Path],
    previous_owned: set[Path],
) -> None:
    """Publish a complete staged bundle while preserving unrelated user files."""

    staged_paths = list(staged_paths)
    new_relative = {path.relative_to(staging) for path in staged_paths}
    stale_relative = previous_owned - new_relative
    all_relative = sorted(new_relative | stale_relative, key=lambda path: path.as_posix())

    root_was_missing = not os.path.lexists(str(root))
    root.mkdir(parents=True, exist_ok=True)
    try:
        for relative in all_relative:
            target = root / relative
            _validate_bundle_target(target, root, relative)
            if os.path.lexists(str(target)) and relative not in previous_owned:
                raise FileExistsError(
                    f"refusing to overwrite an unowned existing bundle path: {relative.as_posix()}"
                )

        backup = Path(tempfile.mkdtemp(prefix=f".{root.name or 'xrdviz'}-backup-", dir=str(root.parent)))
        moved: dict[Path, Path] = {}
        installed: list[Path] = []
        try:
            for relative in all_relative:
                target = root / relative
                if not os.path.lexists(str(target)):
                    continue
                backup_target = backup / relative
                backup_target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(target, backup_target)
                moved[relative] = backup_target

            for staged in staged_paths:
                relative = staged.relative_to(staging)
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(staged, target)
                installed.append(relative)
        except BaseException:
            for relative in reversed(installed):
                _unlink_no_follow(root / relative)
            for relative, backup_target in moved.items():
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(backup_target, target)
            raise
        finally:
            shutil.rmtree(backup, ignore_errors=True)
    except BaseException:
        if root_was_missing:
            _remove_empty_tree(root)
        raise


def _validate_bundle_target(target: Path, root: Path, relative: Path) -> None:
    if relative.is_absolute() or relative.drive or relative.root or ".." in relative.parts:
        raise ValueError("bundle artifact path must remain inside output_dir")
    if not relative.parts:
        raise ValueError("bundle artifact path must not be empty")
    parent = root
    for part in relative.parts[:-1]:
        parent = parent / part
        if os.path.lexists(str(parent)):
            if _is_reparse_point(parent):
                raise ValueError(f"bundle artifact parent is a symlink or reparse point: {parent}")
            if not parent.is_dir():
                raise ValueError(f"bundle artifact parent is not a directory: {parent}")
    if os.path.lexists(str(target)) and target.is_dir() and not _is_reparse_point(target):
        raise FileExistsError(f"bundle artifact target is a directory: {relative.as_posix()}")


def _safe_relative_path(value: object) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    relative = Path(value)
    if relative.is_absolute() or relative.drive or relative.root or ".." in relative.parts:
        return None
    return relative


def _is_reparse_point(path: Path) -> bool:
    try:
        if path.is_symlink():
            return True
        info = path.stat(follow_symlinks=False)
    except OSError:
        return False
    attributes = getattr(info, "st_file_attributes", 0)
    return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def _assert_no_reparse_ancestors(path: Path) -> None:
    current = Path(os.path.abspath(str(path)))
    while True:
        if os.path.lexists(str(current)) and _is_reparse_point(current):
            raise ValueError(f"path contains a symlink or reparse point: {current}")
        if current.parent == current:
            return
        current = current.parent


def _unlink_no_follow(path: Path) -> None:
    if not os.path.lexists(str(path)):
        return
    if path.is_dir() and not _is_reparse_point(path):
        shutil.rmtree(path)
    else:
        path.unlink()


def _remove_empty_tree(root: Path) -> None:
    if not os.path.lexists(str(root)) or _is_reparse_point(root):
        return
    try:
        for child in root.iterdir():
            if child.is_dir() and not _is_reparse_point(child):
                _remove_empty_tree(child)
        root.rmdir()
    except OSError:
        return


def _figure_paths(root: Path, figure_name: str | Path) -> dict[str, Path]:
    requested = Path(figure_name)
    if requested.is_absolute() or requested.drive or requested.root or ".." in requested.parts:
        raise ValueError("figure_name must be a relative path inside output_dir and must not contain '..'")
    suffix = requested.suffix.lower()
    suffix_to_format = {".pdf": "pdf", ".svg": "svg", ".tif": "tiff", ".tiff": "tiff", ".png": "png"}
    if suffix not in suffix_to_format:
        raise ValueError("figure_name must use one of .pdf, .svg, .tif, .tiff, or .png")
    stem = requested.stem
    primary = suffix_to_format[suffix]
    requested_output = root / requested
    parent = requested_output.parent
    canonical = {
        "pdf": parent / f"{stem}.pdf",
        "svg": parent / f"{stem}.svg",
        "tiff": parent / (requested_output.name if primary == "tiff" else f"{stem}.tiff"),
        "png": parent / f"{stem}.png",
    }
    if primary == "pdf":
        canonical["pdf"] = requested_output
    elif primary == "svg":
        canonical["svg"] = requested_output
    elif primary == "png":
        canonical["png"] = requested_output
    root_resolved = root.resolve(strict=False)
    for output in canonical.values():
        try:
            output.resolve(strict=False).relative_to(root_resolved)
        except ValueError as exc:
            raise ValueError("figure_name must resolve inside output_dir") from exc
    return canonical


def _primary_format(figure_name: str | Path) -> str:
    suffix = Path(figure_name).suffix.lower()
    return {".pdf": "pdf", ".svg": "svg", ".tif": "tiff", ".tiff": "tiff", ".png": "png"}[suffix]


def _ordered_figure_formats() -> tuple[str, ...]:
    """Return the fixed on-disk figure order used by every bundle."""

    return ("pdf", "svg", "tiff", "png")


def _write_project_snapshot(state: ProjectState, output: Path) -> Path:
    _write_stable_json(output, project_to_dict(state))
    return output


def _build_manifest(
    state: ProjectState,
    root: Path,
    *,
    figure_map: Mapping[str, Path],
    cleaned_data: Path,
    peak_table: Path,
    fit_data: Path | None,
    fit_summary: Path | None,
    map_data: Path | None,
    derived_data: Path | None,
    project: Path,
    report: Path,
    manifest: Path,
    issues: list[str],
    primary_format: str,
    source_root: Path | None = None,
) -> dict[str, object]:
    settings = state.settings
    target_width_mm = float(settings.figure_width_in) * 25.4
    target_height_mm = float(settings.figure_height_in) * 25.4
    nature_status = "PASS" if not issues else "FAIL"
    figures: list[dict[str, object]] = []
    for format_name in ("pdf", "svg", "tiff", "png"):
        path = figure_map[format_name]
        content = _figure_content(settings.view_mode, format_name)
        figures.append(
            {
                "path": _relative_path(path, root),
                "format": format_name,
                "primary": format_name == primary_format,
                "content": content,
                "content_type": "combination/raster content" if content == "combination/raster" else content,
                "vector_claim": content == "vector",
                "sha256": _sha256(path),
            }
        )
    artifacts = [
        _artifact_record(cleaned_data, root, kind="cleaned_csv"),
        _artifact_record(peak_table, root, kind="peak_csv"),
        _artifact_record(project, root, kind="project"),
        _artifact_record(report, root, kind="report"),
    ]
    if fit_data is not None:
        artifacts.append(_artifact_record(fit_data, root, kind="fit_csv"))
    if fit_summary is not None:
        artifacts.append(_artifact_record(fit_summary, root, kind="fit_summary_csv"))
    if map_data is not None:
        artifacts.append(_artifact_record(map_data, root, kind="map_csv"))
    if derived_data is not None:
        artifacts.append(_artifact_record(derived_data, root, kind="derived_csv"))
    sources = _source_records(state, root if source_root is None else source_root)
    return {
        "application": {"app": "xrdviz", "version": __version__},
        "target": {
            "template": settings.template_name,
            "width_mm": target_width_mm,
            "height_mm": target_height_mm,
            "dpi": int(settings.dpi),
        },
        "nature": {
            "status": nature_status,
            "issues": list(issues),
        },
        "figures": figures,
        "artifacts": artifacts,
        "manifest": {"path": _relative_path(manifest, root)},
        "sources": sources,
    }


def _artifact_record(path: Path, root: Path, *, kind: str) -> dict[str, object]:
    return {"path": _relative_path(path, root), "kind": kind, "sha256": _sha256(path)}


def _source_records(state: ProjectState, root: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for kind, layers in (("spectrum", state.spectra), ("phase", state.phases)):
        for layer in layers:
            source_path = str(getattr(layer, "source_path", "") or "")
            key = (kind, source_path)
            if key in seen:
                continue
            seen.add(key)
            record = _source_record(source_path, root)
            record["kind"] = kind
            record["name"] = str(getattr(layer, "name", "") or "")
            records.append(record)
    if state.fit is not None:
        source_path = str(state.fit.source_path or "")
        record = _source_record(source_path, root)
        record["kind"] = "fit"
        record["name"] = str(state.fit.name or "")
        records.append(record)
    if state.map_data is not None:
        source_path = str(state.map_data.source_path or "")
        record = _source_record(source_path, root)
        record["kind"] = state.map_data.kind
        record["name"] = state.map_data.intensity_label
        records.append(record)
    if state.derived_plot is not None and isinstance(state.derived_plot.source, str):
        source_path = state.derived_plot.source
        record = _source_record(source_path, root)
        record["kind"] = state.derived_plot.kind
        record["name"] = state.derived_plot.kind
        records.append(record)
    records.sort(key=lambda item: (str(item.get("path", "")), str(item.get("kind", "")), str(item.get("name", ""))))
    return records


def _source_record(source_path: str, root: Path) -> dict[str, object]:
    """Describe a source without probing outside the current workspace.

    ``root`` is retained for the private-call compatibility of the manifest
    builder, but it is the output root and is never used as an input trust
    boundary.  Source verification is intentionally scoped to ``cwd``.
    """

    path_text = _source_label(source_path)
    if not source_path:
        return {
            "path": path_text,
            "exists": False,
            "readable": False,
            "status": "missing_path",
        }
    raw = Path(source_path)
    workspace = Path.cwd().resolve(strict=False)
    if any(part == ".." for part in raw.parts):
        return {
            "path": path_text,
            "exists": None,
            "readable": None,
            "status": "unverified_external",
        }
    if raw.is_absolute() or raw.drive or raw.root:
        try:
            raw.relative_to(workspace)
        except ValueError:
            return {
                "path": path_text,
                "exists": None,
                "readable": None,
                "status": "unverified_external",
            }
        candidate = raw
    else:
        candidate = workspace / raw
    try:
        resolved = candidate.resolve(strict=False)
        resolved.relative_to(workspace)
    except (OSError, ValueError):
        return {
            "path": path_text,
            "exists": None,
            "readable": None,
            "status": "unverified_external",
        }
    if not resolved.is_file():
        return {
            "path": path_text,
            "exists": False,
            "readable": False,
            "status": "missing",
        }
    try:
        digest = _sha256(resolved)
    except OSError:
        return {
            "path": path_text,
            "exists": True,
            "readable": False,
            "status": "unreadable",
        }
    return {
        "path": path_text,
        "exists": True,
        "readable": True,
        "status": "ok",
        "sha256": digest,
    }


def _source_label(source_path: object) -> str:
    """Return a privacy-preserving source label for reports and manifests."""

    text = str(source_path or "")
    if not text:
        return ""
    raw = Path(text)
    if raw.is_absolute() or raw.drive or raw.root or ".." in raw.parts:
        return raw.name or "<external source>"
    return raw.as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_stable_json(path: Path, data: object) -> None:
    text = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True, separators=(",", ": "))
    _atomic_write(path, lambda handle: handle.write(text + "\n"))


def _relative_path(path: str | Path, root: Path) -> str:
    path_obj = Path(path)
    try:
        return path_obj.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path_obj.as_posix()


def _display_path(path: str | Path, root: Path) -> str:
    return _relative_path(path, root)


def _figure_content(view_mode: str, format_name: str) -> str:
    if str(view_mode).strip().lower() in {"heatmap", "map"}:
        return "combination/raster"
    if format_name in {"pdf", "svg"}:
        return "vector"
    return "raster"


def _field_name(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", str(value).strip()).strip("_").lower()
    return normalized or "component"


def _component_field_names(components: Iterable[object]) -> list[str]:
    """Return unique CSV fields while retaining deterministic component order."""

    fields: list[str] = []
    used: set[str] = set()
    for component in components:
        base = f"component_{_field_name(component.name)}"
        field = base
        suffix = 2
        while field.casefold() in used:
            field = f"{base}_{suffix}"
            suffix += 1
        fields.append(field)
        used.add(field.casefold())
    return fields


def _write_rows(
    path: Path,
    fieldnames: list[str],
    rows: Iterable[Mapping[str, object]],
) -> None:
    def write_csv(handle) -> None:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    _atomic_write(path, write_csv, newline="")


def _atomic_write(
    path: Path,
    writer: Callable,
    *,
    newline: str | None = None,
) -> None:
    """Write one artifact to a sibling temporary file, then replace it atomically."""

    path = Path(path)
    _assert_no_reparse_ancestors(path.parent)
    path.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(str(path)) and _is_reparse_point(path):
        raise ValueError(f"refusing to write through a symlink or reparse point: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("w", newline=newline, encoding="utf-8") as handle:
            writer(handle)
        os.replace(temporary, path)
    finally:
        if os.path.lexists(str(temporary)):
            _unlink_no_follow(temporary)
