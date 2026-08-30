from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xrdviz.models import (  # noqa: E402
    PLOT_MUTED_COLOR,
    PLOT_TEXT_COLOR,
    PlotAnnotation,
    PlotSettings,
    ProjectState,
    SpectrumLayer,
    project_from_dict,
    project_to_dict,
)
from xrdviz.fit import PatternFit  # noqa: E402


def _contrast_with_white(hex_color: str) -> float:
    channels = [int(hex_color[index : index + 2], 16) / 255.0 for index in (1, 3, 5)]
    linear = [
        channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4
        for channel in channels
    ]
    return (1.0 + 0.05) / (0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2] + 0.05)


def test_spectrum_new_fields_are_appended_after_the_legacy_positional_api() -> None:
    layer = SpectrumLayer(
        "legacy",
        [1.0],
        [2.0],
        "two_theta",
        "#123456",
        False,
        0.4,
        1.2,
        "legacy.xy",
        [1.0],
        [2.0],
        ["old warning"],
        3,
        4,
        5,
        6.0,
        300.0,
        "K",
        "group",
        7.0,
    )

    assert layer.warnings == ["old warning"]
    assert layer.removed_rows == 3
    assert layer.order == 4
    assert layer.frame_index == 5
    assert layer.time_s == 6.0
    assert layer.temperature == 300.0
    assert layer.temperature_unit == "K"
    assert layer.group == "group"
    assert layer.color_value == 7.0
    assert layer.y_error == []


def test_spectrum_uncertainty_and_wavelength_round_trip_and_validate() -> None:
    layer = SpectrumLayer(
        name="measured",
        x=[1.0, 2.0],
        y=[10.0, 20.0],
        y_error=[0.5, 1.0],
        wavelength_angstrom=1.5406,
    )
    encoded = project_to_dict(ProjectState(spectra=[layer]))
    restored = project_from_dict(encoded)

    assert restored.spectra[0].y_error == [0.5, 1.0]
    assert restored.spectra[0].wavelength_angstrom == pytest.approx(1.5406)

    with pytest.raises(ValueError, match="wavelength"):
        SpectrumLayer(name="bad", x=[1.0], y=[2.0], wavelength_angstrom=0.0)
    with pytest.raises(ValueError, match="wavelength"):
        SpectrumLayer(name="bad", x=[1.0], y=[2.0], wavelength_angstrom=float("nan"))


def test_fit_wavelength_round_trip_and_validate() -> None:
    fit = PatternFit(
        name="detector fit",
        x=[1.0, 2.0],
        observed=[10.0, 20.0],
        calculated=[9.0, 19.0],
        axis_kind="d",
        wavelength_angstrom=1.0,
    )
    restored = project_from_dict(project_to_dict(ProjectState(fit=fit)))

    assert restored.fit is not None
    assert restored.fit.wavelength_angstrom == pytest.approx(1.0)

    with pytest.raises(ValueError, match="wavelength"):
        PatternFit(
            name="bad",
            x=[1.0, 2.0],
            observed=[1.0, 2.0],
            calculated=[1.0, 2.0],
            wavelength_angstrom=0.0,
        )


def test_plot_settings_new_fields_are_appended_after_the_legacy_positional_api() -> None:
    settings = PlotSettings(
        "q",
        12.0,
        "Q",
        "Intensity",
        "legacy",
        0.1,
        2.0,
        False,
        True,
        1e-6,
        True,
        0.4,
        2.0,
        3.0,
        300,
        "DejaVu Sans",
        8.0,
        9.0,
        10.0,
        1.0,
        0.2,
        False,
        True,
        False,
        True,
        "heatmap",
        "temperature",
        "color_value",
        "blue_rose",
        True,
        3,
        9,
        "lower left",
        "science",
        0.1,
        0.9,
        0.8,
        0.2,
    )

    assert settings.x_axis == "q"
    assert settings.view_mode == "heatmap"
    assert settings.sort_by == "temperature"
    assert settings.color_by == "color_value"
    assert settings.colormap == "blue_rose"
    assert settings.show_colorbar is True
    assert settings.show_every_n == 3
    assert settings.heatmap_points == 9
    assert settings.legend_location == "lower left"
    assert settings.template_name == "science"
    assert settings.margin_left == 0.1
    assert settings.margin_bottom == 0.2
    assert settings.uncertainty_mode == "none"


def test_project_state_new_fields_are_appended_after_settings() -> None:
    settings = PlotSettings(x_axis="d")
    state = ProjectState([], [], settings)

    assert state.settings is settings
    assert state.fit is None
    assert state.map_data is None
    assert state.derived_plot is None
    assert state.annotations == []


def test_default_annotation_color_is_readable_on_white() -> None:
    annotation = PlotAnnotation(x=1.0, text="guide")

    assert PLOT_MUTED_COLOR == "#9A9A9A"
    assert annotation.color == PLOT_TEXT_COLOR
    assert _contrast_with_white(annotation.color) >= 4.5
    assert annotation.color.upper() not in {"#9A9A9A", "#777777"}


def test_plot_settings_normalizes_integral_dpi_and_rejects_fractional_or_bool() -> None:
    assert PlotSettings(dpi=600.0).dpi == 600

    with pytest.raises(ValueError, match="positive integer"):
        PlotSettings(dpi=600.5)
    with pytest.raises(ValueError, match="positive integer"):
        PlotSettings(dpi=True)
