# XRDViz

XRDViz is a Python/Qt desktop application for turning one-dimensional spectra, fit results, detector maps, and selected XRD analyses into clear, traceable publication figures.

## Features

- Drag and drop `.txt`, `.csv`, `.xy`, and `.dat` spectrum files.
- Declare input X axis as `2theta`, `d`, or `q`, then convert all layers through a global energy setting.
- Display linear, normalized, log, stacked, or log-stacked spectra without mutating raw data.
- Batch import spectrum folders, infer frame/time/temperature metadata from filenames, sort by metadata, and draw overlay, stack, gradient stack, heatmap, or small-multiple views.
- Import structured intensity uncertainty and display it as a shaded band or sampled error bars.
- Add zoom insets, explicit vertical annotations, automatic panel labels, and shared-axis small multiples.
- Load CIF files and draw Bragg tick marks below the main plot.
- Import `sample_labels.csv` to control sample labels, order, colors, visibility, and offsets.
- Import `reference_peaks.csv` or simple Rigaku-style `peaks.csv` files as reference phase markers.
- Show phase-specific marker symbols, guide lines, direct curve labels, compact phase legends, and a reference peak table.
- Import observed/calculated fit CSV files and draw observed, calculated, background, component, Bragg, and difference panels with Rp/Rwp values.
- Fit seeded or prominence-suggested Gaussian, Lorentzian, or pseudo-Voigt components with polynomial backgrounds; retain peak centre, FWHM, area, height, convergence, and residual data.
- Load raw detector arrays/images, perform explicit flat-detector radial integration, or build a 2theta-chi cake; import complete RSM and pole-figure grids.
- Build Scherrer size, Williamson-Hall, and rocking-curve plots from explicit CSV contracts.
- Apply exact 89 mm / 183 mm Nature presets, Science presets, or custom templates with adjustable legend placement, fonts, dimensions, and margins.
- Pan, zoom, reset, and inspect the live plot with the built-in navigation toolbar.
- See a permanent Nature preflight status while editing; invalid numeric input leaves the last valid plot visible and reports the field that needs attention.
- Export line plots as editable PDF/SVG or opaque RGB PNG/TIFF at the configured resolution (600 dpi in the Nature presets).
- Export a traceable publication bundle with four figure formats, source data for the active advanced analysis, a restorable project snapshot, a report, and a SHA-256 manifest.

## Nature-oriented export

The Nature presets use Arial, 5--7 pt typography, restrained line weights, exact 89 mm (single-column) or 183 mm (double-column) widths, and 600 dpi raster output. Quantitative heatmaps and gradients default to the perceptually uniform, color-vision-friendly Cividis map. Mixed Celsius/Kelvin series are compared in Kelvin; a mixed series that combines declared and missing or unknown units fails closed instead of receiving a misleading scale. Missing time or temperature metadata remains missing and is shown as `n/a` or muted gray rather than being replaced by acquisition order.

For ordinary line plots, PDF and SVG retain vector paths and editable text. Heatmaps and 2D maps necessarily embed raster image content inside PDF/SVG and are therefore reported as combination/raster figures; XRDViz does not label them as all-vector. The in-app preflight checks configuration and visible data, but it is not a guarantee of editorial acceptance. See Nature's current [figure construction and export guide](https://research-figure-guide.nature.com/figures/building-and-exporting-figure-panels/) and the journal's [initial submission guidance](https://www.nature.com/nature/for-authors/initial-submission).

The publication bundle contains:

- `<name>.pdf`, `<name>.svg`, `<name>.tiff`, and `<name>.png`
- `cleaned_xrd_data.csv` and `reference_peak_table.csv`
- when present: `pattern_fit_data.csv`, `peak_fit_summary.csv`, `map_data.csv`, and/or `derived_analysis_data.csv`
- `project.xrdviz.json`, which can be reopened in XRDViz
- `xrd_plot_report.md` and `publication_manifest.json`, including output hashes and source-file status

## Batch and In-situ Workflow

Use **File -> Import spectra folder...** for folders of `.txt`, `.csv`, `.xy`, or `.dat` patterns. XRDViz infers metadata from common filename forms such as:

- `Az_Full_000123.txt` -> frame `123`
- `scan_0007_12.5min_650C.xy` -> frame `7`, time `750 s`, temperature `650 C`

The **Batch** tab controls:

- view mode: overlay, stack, gradient stack, heatmap, small multiples, fit/residual, 2D map, or derived analysis
- sort/color fields: frame, time, temperature, or current order
- colormap, colorbar, show every N spectra, and heatmap sampling points

The publication bundle report records these batch settings and each spectrum's inferred metadata. Heatmap row labels are always shown, sparsified to a readable set for long series, and include the first and last frame.

## CSV Helpers

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

`source_axis` accepts `two_theta`, `d`, or `q`; peaks are converted to the current plot axis using the global energy setting.

Observed/calculated fit data use a header row. The x column may be `x`, `2theta`, `d`, or `q`; `observed` and `calculated` are required. `sigma`, `background`, and any `component_<name>` or `peak_<name>` columns are optional:

```csv
2theta,observed,calculated,sigma,background,component_alpha
20,100,98,3,12,86
21,180,176,4,13,163
22,110,113,3,12,101
```

Peak-width analyses require explicit degree-based positions and widths:

```csv
2theta,FWHM,hkl,intensity
35.1,0.20,111,100
50.3,0.24,200,72
63.0,0.29,220,48
```

Rocking curves require `omega,intensity`. RSM and pole-figure imports use complete, regular long-form grids; every coordinate pair must appear exactly once:

```csv
qx,qz,intensity
0.0,1.0,120
0.1,1.0,135
0.0,1.1,98
0.1,1.1,111
```

For a pole figure, use `phi,chi,intensity` instead.

## Advanced Analysis Boundaries

- The fit importer presents external observed/calculated results. It is not a Rietveld, Pawley, or Le Bail solver.
- Peak decomposition is a bounded profile fit for plotting and peak summaries; suggested seeds are not phase identification.
- Detector radial/cake processing is an explicit flat, untilted preview. It does not invent distortion, polarization, solid-angle, or instrument-calibration corrections; the preview assumptions are stored in the project/report and flagged by publication preflight.
- RSM and pole-figure workflows render already-declared reciprocal/angle coordinates. They do not transform raw goniometer scans, calculate an ODF, or claim a texture mechanism.
- Scherrer and Williamson-Hall results require explicit wavelength, shape factor, and instrument broadening. They do not report uncertainty unless independently supplied.
- Database Search/Match, quantitative phase analysis, and automated phase fractions are outside the current scope.

## Run

```powershell
py -3.12 -m pip install -e .
py -3.12 -m xrdviz
```

XRDViz is currently distributed for source execution rather than as a Windows executable.
