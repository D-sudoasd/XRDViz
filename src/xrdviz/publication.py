from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

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
            "- content note: heatmap PDF/SVG exports contain raster image content and are not claimed to be all-vector.",
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
        lines.append(f"- cleaned data: `{_display_path(cleaned_data, root)}`")
    if peak_table is not None:
        lines.append(f"- peak table: `{_display_path(peak_table, root)}`")
    if project is not None:
        lines.append(f"- project snapshot: `{_display_path(project, root)}`")
    if manifest is not None:
        lines.append(f"- manifest: `{_display_path(manifest, root)}`")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
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
    root.mkdir(parents=True, exist_ok=True)
    figure_map = _figure_paths(root, figure_name)
    primary_format = _primary_format(figure_name)
    figures = tuple(figure_map[format_name] for format_name in _ordered_figure_formats())

    for figure_path in figures:
        figure_path.parent.mkdir(parents=True, exist_ok=True)
        export_project(state, figure_path)

    cleaned_data = export_cleaned_data(state, root)
    peak_table = export_peak_table(state, root)
    project = _write_project_snapshot(state, root / "project.xrdviz.json")
    issues = nature_compliance_issues(state)
    manifest = root / "publication_manifest.json"
    report = write_publication_report(
        state,
        root,
        exported_figure=figure_map[_primary_format(figure_name)],
        cleaned_data=cleaned_data,
        peak_table=peak_table,
        figures=figures,
        project=project,
        manifest=manifest,
        compliance_issues=issues,
    )
    manifest_data = _build_manifest(
        state,
        root,
        figure_map=figure_map,
        cleaned_data=cleaned_data,
        peak_table=peak_table,
        project=project,
        report=report,
        manifest=manifest,
        issues=issues,
        primary_format=primary_format,
    )
    _write_stable_json(manifest, manifest_data)
    return PublicationOutputs(
        figure=figure_map[_primary_format(figure_name)],
        cleaned_data=cleaned_data,
        peak_table=peak_table,
        report=report,
        figures=figures,
        project=project,
        manifest=manifest,
    )


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
    project: Path,
    report: Path,
    manifest: Path,
    issues: list[str],
    primary_format: str,
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
    sources = _source_records(state, root)
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
    records.sort(key=lambda item: (str(item.get("path", "")), str(item.get("kind", "")), str(item.get("name", ""))))
    return records


def _source_record(source_path: str, root: Path) -> dict[str, object]:
    path_text = Path(source_path).as_posix() if source_path else ""
    if not source_path:
        return {
            "path": path_text,
            "resolved_path": None,
            "exists": False,
            "readable": False,
            "status": "missing_path",
            "sha256": None,
        }
    raw = Path(source_path)
    candidates = [raw] if raw.is_absolute() else [Path.cwd() / raw, root / raw]
    resolved: Path | None = next((candidate for candidate in candidates if candidate.is_file()), None)
    if resolved is None:
        return {
            "path": path_text,
            "resolved_path": None,
            "exists": False,
            "readable": False,
            "status": "missing",
            "sha256": None,
        }
    try:
        digest = _sha256(resolved)
    except OSError:
        return {
            "path": path_text,
            "resolved_path": str(resolved.resolve()),
            "exists": True,
            "readable": False,
            "status": "unreadable",
            "sha256": None,
        }
    return {
        "path": path_text,
        "resolved_path": str(resolved.resolve()),
        "exists": True,
        "readable": True,
        "status": "ok",
        "sha256": digest,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_stable_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True, separators=(",", ": "))
    path.write_text(text + "\n", encoding="utf-8")


def _relative_path(path: str | Path, root: Path) -> str:
    path_obj = Path(path)
    try:
        return path_obj.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path_obj.as_posix()


def _display_path(path: str | Path, root: Path) -> str:
    return _relative_path(path, root)


def _figure_content(view_mode: str, format_name: str) -> str:
    if str(view_mode).strip().lower() == "heatmap":
        return "combination/raster"
    if format_name in {"pdf", "svg"}:
        return "vector"
    return "raster"


def _write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
