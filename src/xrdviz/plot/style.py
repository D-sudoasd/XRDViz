from __future__ import annotations

from dataclasses import replace

from xrdviz.models import PLOT_AXIS_COLOR, PLOT_TEXT_COLOR, PUBLICATION_PALETTE, PlotSettings

NATURE_SINGLE_WIDTH_IN = 89.0 / 25.4
NATURE_DOUBLE_WIDTH_IN = 183.0 / 25.4


def nature_single_column(settings: PlotSettings | None = None) -> PlotSettings:
    base = settings or PlotSettings()
    return replace(
        base,
        figure_width_in=NATURE_SINGLE_WIDTH_IN,
        figure_height_in=2.35,
        dpi=600,
        font_family="Arial",
        font_size=7.0,
        axis_label_size=7.0,
        tick_label_size=6.0,
        line_width=0.75,
        bragg_band_height=0.16,
    )


def nature_double_column(settings: PlotSettings | None = None) -> PlotSettings:
    base = settings or PlotSettings()
    return replace(
        base,
        figure_width_in=NATURE_DOUBLE_WIDTH_IN,
        figure_height_in=3.15,
        dpi=600,
        font_family="Arial",
        font_size=7.0,
        axis_label_size=7.0,
        tick_label_size=6.0,
        line_width=0.75,
        bragg_band_height=0.16,
    )


def apply_matplotlib_style(settings: PlotSettings) -> None:
    mpl = _matplotlib()
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [settings.font_family, "Arial", "Helvetica", "DejaVu Sans"],
            "font.size": settings.font_size,
            "axes.labelsize": settings.axis_label_size,
            "axes.edgecolor": PLOT_AXIS_COLOR,
            "axes.labelcolor": PLOT_TEXT_COLOR,
            "xtick.labelsize": settings.tick_label_size,
            "ytick.labelsize": settings.tick_label_size,
            "xtick.color": PLOT_TEXT_COLOR,
            "ytick.color": PLOT_TEXT_COLOR,
            "text.color": PLOT_TEXT_COLOR,
            "axes.linewidth": min(settings.line_width, 0.8),
            "lines.linewidth": settings.line_width,
            "legend.frameon": False,
            "legend.fontsize": settings.tick_label_size,
            "legend.labelcolor": PLOT_TEXT_COLOR,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "figure.dpi": settings.dpi,
            "savefig.dpi": settings.dpi,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.prop_cycle": mpl.cycler(color=PUBLICATION_PALETTE),
        }
    )


def _matplotlib():
    try:
        import matplotlib as mpl
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for rendering and export") from exc
    return mpl
