# XRDViz

> **已归档 / Archived — 2026-09-27**：本项目已停止使用和维护，保留源码与历史记录供查阅。详见 [归档说明 / Archive notice](ARCHIVED.md)。
>
> This project is retired and no longer maintained. Source code and Git history are retained for reference.

**Desktop XRD visualization and evidence-aware publication figures** (v0.2.0)

Python / Qt application for turning one-dimensional spectra, fit results, detector maps, and selected XRD analyses into clear, traceable publication figures. Source install only — no Windows EXE is claimed.

Requires **Python 3.10+**. Entry points: `xrdviz` and `python -m xrdviz`.

## What it does

- Drag-and-drop `.txt`, `.csv`, `.xy`, and `.dat` spectra; declare X as `2theta`, `d`, or `q` and convert layers through a global energy setting.
- Display linear, normalized, log, stacked, or log-stacked spectra **without mutating raw data**.
- Batch-import spectrum folders; infer frame / time / temperature from filenames; overlay, stack, gradient stack, heatmap, or small-multiple views.
- Uncertainty bands or error bars from structured intensity uncertainty.
- Zoom insets, vertical annotations, panel labels, shared-axis small multiples.
- CIF Bragg ticks; `sample_labels.csv` for labels / order / colors; `reference_peaks.csv` or Rigaku-style `peaks.csv` markers.
- Observed / calculated fit CSV panels (observed, calculated, background, components, Bragg, difference) with Rp / Rwp when present.
- Seeded or prominence-suggested Gaussian / Lorentzian / pseudo-Voigt peak fits with polynomial backgrounds; retain centre, FWHM, area, height, convergence, residuals.
- Flat-detector radial integration or 2θ–χ cake from raw arrays; import complete RSM and pole-figure grids.
- Scherrer, Williamson–Hall, and rocking-curve plots from explicit CSV contracts.
- Nature 89 mm / 183 mm and Science presets (or custom templates); live Nature preflight while editing.
- Export PDF / SVG (vector where applicable) or PNG / TIFF; optional publication bundle with figures, source CSVs, restorable `project.xrdviz.json`, report, and SHA-256 manifest.

## Install and run

```powershell
git clone https://github.com/D-sudoasd/XRDViz.git
cd XRDViz
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
xrdviz
# equivalent: python -m xrdviz
```

Linux / macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
xrdviz
```

Dependencies (from `pyproject.toml`): NumPy, pandas, SciPy, Matplotlib, Pillow, PySide6, pymatgen. Dev tests: `python -m pip install -e ".[dev]"` then `pytest -q`.

XRDViz is distributed for **source execution**. There is no packaged Windows executable in this repository.

## Nature-oriented export

Nature presets use Arial, 5–7 pt typography, restrained line weights, exact **89 mm** (single-column) or **183 mm** (double-column) widths, and **600 dpi** raster output. Quantitative heatmaps default to Cividis. Mixed Celsius / Kelvin series are compared in Kelvin; mixed declared + missing units fail closed. Missing time / temperature stays missing (`n/a` or muted gray), never replaced by acquisition order.

For ordinary line plots, PDF and SVG keep vector paths and editable text. Heatmaps and 2D maps embed raster content inside PDF / SVG and are reported as combination / raster figures — not labelled all-vector. In-app preflight checks configuration and visible data; it is **not** a guarantee of editorial acceptance. See Nature’s [figure construction guide](https://research-figure-guide.nature.com/figures/building-and-exporting-figure-panels/) and [initial submission guidance](https://www.nature.com/nature/for-authors/initial-submission).

Publication bundle contents:

- `<name>.pdf`, `.svg`, `.tiff`, `.png`
- `cleaned_xrd_data.csv`, `reference_peak_table.csv`
- when present: `pattern_fit_data.csv`, `peak_fit_summary.csv`, `map_data.csv`, `derived_analysis_data.csv`
- `project.xrdviz.json` (reopenable), `xrd_plot_report.md`, `publication_manifest.json` (hashes and source status)

## Batch / in-situ workflow

**File → Import spectra folder…** for folders of `.txt` / `.csv` / `.xy` / `.dat`. Filename forms such as:

- `Az_Full_000123.txt` → frame `123`
- `scan_0007_12.5min_650C.xy` → frame `7`, time `750 s`, temperature `650 °C`

The **Batch** tab controls view mode (overlay, stack, gradient stack, heatmap, small multiples, fit/residual, 2D map, derived analysis), sort/color fields, colormap, and sampling. The publication report records batch settings and inferred metadata. Heatmap row labels stay readable (sparsified; first and last frames always included).

## CSV helpers

`sample_labels.csv`:

```csv
filename,label,order,color,visible,offset
sample_a.xy,Annealed,1,#D62F53,true,0.3
sample_b.xy,As cast,2,#45A7E6,true,0.0
```

`reference_peaks.csv`:

```csv
position,label,phase,intensity,hkl,source_axis,color,shape
30.0,Main peak,Calcite,100,104,two_theta,#2B9C8F,triangle
2.5,d peak,Calcite,40,110,d,#2B9C8F,square
```

`source_axis` accepts `two_theta`, `d`, or `q`; peaks convert to the current plot axis via the global energy setting.

Observed / calculated fit CSVs need a header; x may be `x`, `2theta`, `d`, or `q`; `observed` and `calculated` are required. Optional: `sigma`, `background`, `component_<name>`, `peak_<name>`.

Peak-width analyses need explicit degree-based positions and widths (`2theta`, `FWHM`, …). Rocking curves: `omega,intensity`. RSM / pole figures: complete regular long-form grids (`qx,qz,intensity` or `phi,chi,intensity`); every coordinate pair exactly once.

## Scientific boundary

| In scope | Out of scope |
| --- | --- |
| Traceable plotting and publication export | Database Search/Match |
| Bounded peak decomposition for summaries | Rietveld / Pawley / Le Bail solvers |
| Flat untilted detector radial / cake preview | Distortion, polarization, solid-angle, or full instrument calibration |
| Declared RSM / pole-figure grids | Raw goniometer → reciprocal transforms, ODF, texture mechanism claims |
| Scherrer / W–H with user wavelength, shape factor, instrument broadening | Invented uncertainty when none is supplied |
| Fit importer for **external** obs/calc results | Quantitative phase fractions / automated QPA |

Detector preview assumptions are stored in the project / report and flagged by publication preflight. Suggested peak seeds are not phase identification.

## Package names

| Shown name | Machine name |
| --- | --- |
| XRDViz | PyPI / project: `xrdviz` |
| | Import: `xrdviz` |
| | CLI: `xrdviz` |

## Development

```bash
python -m pip install -e ".[dev]"
pytest -q
```

Repository: [github.com/D-sudoasd/XRDViz](https://github.com/D-sudoasd/XRDViz). Issues welcome for reproducible bugs and focused feature proposals.

## License

No `LICENSE` file is present in this checkout. Do not assume MIT or any other terms until a license is added to the repository.
