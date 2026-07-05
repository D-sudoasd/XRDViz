from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from xrdviz.axes import convert_x
from xrdviz.cif import load_cif_phase, peak_position_for_axis
from xrdviz.io import load_spectrum
from xrdviz.models import OKABE_ITO, PhaseLayer, PhasePeak, PlotSettings, ProjectState, SpectrumLayer, default_axis_label
from xrdviz.plot.style import nature_single_column


@dataclass(slots=True)
class CalibrationMatch:
    hkl: str
    theoretical_d: float
    observed_d: float
    error: float
    intensity: float


@dataclass(slots=True)
class CalibrationResult:
    phase: str
    fitted_lattice_a: float | None
    scale: float
    rms_error: float | None
    confidence: str
    matched_peaks: list[CalibrationMatch] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class _Candidate:
    phase_index: int
    result: CalibrationResult
    observed_indices: tuple[int, ...]
    score: float


def auto_calibrate_phases(
    state: ProjectState,
    *,
    apply: bool = True,
    min_confidence: str = "high",
    max_scale_delta: float = 0.10,
) -> dict[str, CalibrationResult]:
    observed = detect_experimental_peaks(state)
    phase_candidates = [
        _candidate_solutions(index, phase, observed, state.settings, max_scale_delta=max_scale_delta)
        for index, phase in enumerate(state.phases)
        if phase.visible and phase.peaks
    ]
    selected = _select_joint_candidates([candidates for candidates in phase_candidates if candidates])

    results: dict[str, CalibrationResult] = {}
    for candidate in selected:
        phase = state.phases[candidate.phase_index]
        result = candidate.result
        results[phase.phase or phase.name] = result
        if apply:
            _apply_calibration_result(phase, result, min_confidence=min_confidence)

    for phase in state.phases:
        name = phase.phase or phase.name
        if phase.visible and phase.peaks and name not in results:
            result = CalibrationResult(
                phase=name,
                fitted_lattice_a=None,
                scale=1.0,
                rms_error=None,
                confidence="low",
                warnings=["No reliable experimental peak matches found"],
            )
            results[name] = result
            if apply:
                _apply_calibration_result(phase, result, min_confidence=min_confidence)
    return results


def detect_experimental_peaks(state: ProjectState, *, top_n: int = 36, min_separation: float = 0.025) -> list[float]:
    candidates: list[tuple[float, float]] = []
    d_range = _settings_d_range(state.settings)
    for layer in state.spectra:
        if not layer.visible:
            continue
        x_values = convert_x(layer.x, layer.axis_kind, "d", state.settings.energy_kev)
        pairs = sorted(
            (x, y)
            for x, y in zip(x_values, layer.y)
            if math.isfinite(x) and math.isfinite(y) and _within_range(x, d_range)
        )
        if len(pairs) < 3:
            continue
        xs = [pair[0] for pair in pairs]
        ys = _normalize([pair[1] for pair in pairs])
        smooth = _moving_average(ys, window=7 if len(ys) >= 7 else 3)
        local = []
        for index in range(1, len(xs) - 1):
            if smooth[index] >= smooth[index - 1] and smooth[index] >= smooth[index + 1]:
                local.append((xs[index], smooth[index]))
        ranked_points = sorted(zip(xs, smooth), key=lambda pair: pair[1], reverse=True)[: top_n * 2]
        candidates.extend(local)
        candidates.extend(ranked_points)
    return _deduplicate_peaks(candidates, top_n=top_n, min_separation=min_separation)


def build_publication_state(
    *,
    title: str,
    spectra: Sequence[tuple[str, str | Path]],
    phase_paths: Sequence[tuple[str, str | Path]],
    energy_kev: float = 8.0478,
    x_min: float = 0.9,
    x_max: float = 3.0,
) -> ProjectState:
    spectrum_layers: list[SpectrumLayer] = []
    for index, (label, path) in enumerate(spectra):
        file_path = Path(path)
        if file_path.exists():
            layer = load_spectrum(file_path, axis_kind="auto", color=OKABE_ITO[index % len(OKABE_ITO)])
        else:
            layer = SpectrumLayer(name=label, x=[x_min, x_max], y=[0.0, 1.0], axis_kind="d", color=OKABE_ITO[index % len(OKABE_ITO)], source_path=str(file_path))
        layer.name = label
        layer.order = index
        layer.linewidth = 0.75
        spectrum_layers.append(layer)

    phase_layers: list[PhaseLayer] = []
    for index, (label, path) in enumerate(phase_paths):
        file_path = Path(path)
        color = OKABE_ITO[(len(spectrum_layers) + index) % len(OKABE_ITO)]
        if file_path.exists():
            phase = load_cif_phase(file_path, energy_kev=energy_kev, color=color)
        else:
            phase = PhaseLayer(name=label, phase=label, source_path=str(file_path), color=color)
        phase.name = label
        phase.phase = label
        phase.tick_height = 0.95
        phase.label_policy = "none"
        phase.show_guides = False
        phase_layers.append(phase)

    settings = nature_single_column(
        PlotSettings(
            x_axis="d",
            x_label=default_axis_label("d"),
            y_label="Log (intensity) (a.u.)",
            panel_title=title,
            x_min=x_min,
            x_max=x_max,
            normalize=True,
            log_scale=True,
            stack_enabled=True,
            stack_spacing=0.20,
            show_legend=True,
            direct_labels=False,
            show_phase_legend=False,
            show_y_tick_labels=False,
        )
    )
    return ProjectState(spectra=spectrum_layers, phases=phase_layers, settings=settings)


def _candidate_solutions(
    phase_index: int,
    phase: PhaseLayer,
    observed: list[float],
    settings: PlotSettings,
    *,
    max_scale_delta: float,
) -> list[_Candidate]:
    if phase.reference_lattice_a is None or phase.reference_lattice_a <= 0 or not observed:
        return []
    theoretical = _theoretical_d_peaks(phase, settings)
    if not theoretical:
        return []

    scale_values = {1.0}
    for peak_index, (theory_d, _intensity, _hkl) in enumerate(theoretical):
        del peak_index
        for observed_d in observed:
            scale = observed_d / theory_d
            if 1.0 - max_scale_delta <= scale <= 1.0 + max_scale_delta:
                scale_values.add(scale)

    candidates: list[_Candidate] = []
    for scale in scale_values:
        matches, observed_indices = _match_theoretical_to_observed(theoretical, observed, scale)
        if not matches:
            continue
        rms = _rms([match.error for match in matches])
        fitted_lattice = phase.reference_lattice_a * scale
        confidence = _confidence_for(matches, rms)
        result = CalibrationResult(
            phase=phase.phase or phase.name,
            fitted_lattice_a=fitted_lattice,
            scale=scale,
            rms_error=rms,
            confidence=confidence,
            matched_peaks=matches,
            warnings=[] if confidence == "high" else ["Low confidence: fewer than 2 clean matches or RMS error above threshold"],
        )
        score = len(matches) * 100.0 - rms * 1000.0 + sum(match.intensity for match in matches) * 0.01
        candidates.append(_Candidate(phase_index=phase_index, result=result, observed_indices=tuple(observed_indices), score=score))
    candidates.sort(key=lambda candidate: candidate.score, reverse=True)
    return candidates[:8]


def _select_joint_candidates(candidate_groups: list[list[_Candidate]]) -> list[_Candidate]:
    if not candidate_groups:
        return []
    selected: list[_Candidate] = []
    best_score = -1e18

    def walk(group_index: int, current: list[_Candidate]) -> None:
        nonlocal selected, best_score
        if group_index == len(candidate_groups):
            score = _joint_score(current)
            if score > best_score:
                best_score = score
                selected = list(current)
            return
        for candidate in candidate_groups[group_index]:
            current.append(candidate)
            walk(group_index + 1, current)
            current.pop()

    walk(0, [])
    return selected


def _joint_score(candidates: list[_Candidate]) -> float:
    observed_indices = [index for candidate in candidates for index in candidate.observed_indices]
    unique = len(set(observed_indices))
    conflicts = len(observed_indices) - unique
    return sum(candidate.score for candidate in candidates) + unique * 25.0 - conflicts * 250.0


def _match_theoretical_to_observed(
    theoretical: list[tuple[float, float, str]],
    observed: list[float],
    scale: float,
) -> tuple[list[CalibrationMatch], list[int]]:
    used: set[int] = set()
    matches: list[CalibrationMatch] = []
    ordered = sorted(theoretical, key=lambda item: item[1], reverse=True)
    for theory_d, intensity, hkl in ordered:
        scaled = theory_d * scale
        best_index = None
        best_error = None
        for index, observed_d in enumerate(observed):
            if index in used:
                continue
            error = observed_d - scaled
            tolerance = max(0.018, scaled * 0.018)
            if abs(error) <= tolerance and (best_error is None or abs(error) < abs(best_error)):
                best_index = index
                best_error = error
        if best_index is None or best_error is None:
            continue
        used.add(best_index)
        matches.append(
            CalibrationMatch(
                hkl=hkl,
                theoretical_d=theory_d,
                observed_d=observed[best_index],
                error=best_error,
                intensity=intensity,
            )
        )
    return matches, sorted(used)


def _theoretical_d_peaks(phase: PhaseLayer, settings: PlotSettings, *, max_peaks: int = 10) -> list[tuple[float, float, str]]:
    d_range = _settings_d_range(settings)
    rows = []
    for peak in phase.peaks:
        d_value = peak_position_for_axis(peak, "d", settings.energy_kev)
        if math.isfinite(d_value) and _within_range(d_value, d_range):
            rows.append((d_value, peak.intensity, peak.hkl))
    rows.sort(key=lambda item: item[1], reverse=True)
    return rows[:max_peaks]


def _apply_calibration_result(phase: PhaseLayer, result: CalibrationResult, *, min_confidence: str) -> None:
    phase.calibration_confidence = result.confidence
    phase.calibration_error = result.rms_error
    phase.calibration_notes = list(result.warnings)
    if result.matched_peaks:
        phase.calibration_notes.append(f"matched {len(result.matched_peaks)} peaks")
        for match in result.matched_peaks:
            phase.calibration_notes.append(
                f"{match.hkl or 'peak'}: {match.theoretical_d:.4f} A -> {match.observed_d:.4f} A, error={match.error:.4f} A"
            )
    should_apply = result.fitted_lattice_a is not None and _confidence_rank(result.confidence) >= _confidence_rank(min_confidence)
    if should_apply:
        phase.lattice_a = result.fitted_lattice_a
        phase.auto_calibrated = True
    else:
        phase.auto_calibrated = False


def _confidence_for(matches: list[CalibrationMatch], rms: float) -> str:
    if len(matches) >= 2 and rms <= 0.015:
        return "high"
    if len(matches) >= 2 and rms <= 0.025:
        return "medium"
    return "low"


def _confidence_rank(value: str) -> int:
    return {"low": 0, "medium": 1, "high": 2}.get(value, 0)


def _settings_d_range(settings: PlotSettings) -> tuple[float, float] | None:
    if settings.x_min is None or settings.x_max is None:
        return None
    values = convert_x([settings.x_min, settings.x_max], settings.x_axis, "d", settings.energy_kev)
    values = [value for value in values if math.isfinite(value)]
    if len(values) != 2:
        return None
    return min(values), max(values)


def _within_range(value: float, value_range: tuple[float, float] | None) -> bool:
    if value_range is None:
        return True
    return value_range[0] <= value <= value_range[1]


def _normalize(values: list[float]) -> list[float]:
    positive = [value for value in values if math.isfinite(value) and value > 0.0]
    scale = max(positive) if positive else 1.0
    return [value / scale for value in values]


def _moving_average(values: list[float], *, window: int) -> list[float]:
    if window <= 1 or len(values) < window:
        return list(values)
    half = window // 2
    smoothed = []
    for index in range(len(values)):
        left = max(0, index - half)
        right = min(len(values), index + half + 1)
        smoothed.append(sum(values[left:right]) / (right - left))
    return smoothed


def _deduplicate_peaks(candidates: list[tuple[float, float]], *, top_n: int, min_separation: float) -> list[float]:
    selected: list[tuple[float, float]] = []
    for x_value, score in sorted(candidates, key=lambda item: item[1], reverse=True):
        if any(abs(x_value - existing_x) < min_separation for existing_x, _score in selected):
            continue
        selected.append((x_value, score))
        if len(selected) >= top_n:
            break
    return sorted(x for x, _score in selected)


def _rms(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values) / len(values)) if values else math.inf
