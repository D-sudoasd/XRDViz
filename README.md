# XRDViz

XRDViz is a Python/Qt desktop application for turning one-dimensional XRD spectra into clear, traceable publication figures.

## Features

- Drag and drop `.txt`, `.csv`, `.xy`, and `.dat` spectrum files.
- Declare input X axis as `2theta`, `d`, or `q`, then convert all layers through a global energy setting.
- Display linear, normalized, log, stacked, or log-stacked spectra without mutating raw data.
- Batch import spectrum folders, infer frame/time/temperature metadata from filenames, sort by metadata, and draw overlay, stack, gradient stack, or heatmap views.
- Load CIF files and draw Bragg tick marks below the main plot.
- Import `sample_labels.csv` to control sample labels, order, colors, visibility, and offsets.
- Import `reference_peaks.csv` or simple Rigaku-style `peaks.csv` files as reference phase markers.
- Show phase-specific marker symbols, guide lines, direct curve labels, compact phase legends, and a reference peak table.
- Apply exact 89 mm / 183 mm Nature presets, Science presets, or custom templates with adjustable legend placement, fonts, dimensions, and margins.
- Pan, zoom, reset, and inspect the live plot with the built-in navigation toolbar.
- See a permanent Nature preflight status while editing; invalid numeric input leaves the last valid plot visible and reports the field that needs attention.
- Export line plots as editable PDF/SVG or opaque RGB PNG/TIFF at the configured resolution (600 dpi in the Nature presets).
- Export a traceable publication bundle with four figure formats, cleaned data, a reference peak table, a restorable project snapshot, a report, and a SHA-256 manifest.

## Nature-oriented export

The Nature presets use Arial, 5--7 pt typography, restrained line weights, exact 89 mm (single-column) or 183 mm (double-column) widths, and 600 dpi raster output. Quantitative heatmaps and gradients default to the perceptually uniform, color-vision-friendly Cividis map. Mixed Celsius/Kelvin series are compared in Kelvin; a mixed series that combines declared and missing or unknown units fails closed instead of receiving a misleading scale. Missing time or temperature metadata remains missing and is shown as `n/a` or muted gray rather than being replaced by acquisition order.

For ordinary line plots, PDF and SVG retain vector paths and editable text. Heatmaps necessarily embed raster image content inside PDF/SVG and are therefore reported as combination/raster figures; XRDViz does not label them as all-vector. The in-app preflight checks configuration and visible data, but it is not a guarantee of editorial acceptance. See Nature's current [figure construction and export guide](https://research-figure-guide.nature.com/figures/building-and-exporting-figure-panels/) and the journal's [initial submission guidance](https://www.nature.com/nature/for-authors/initial-submission).

The publication bundle contains:

- `<name>.pdf`, `<name>.svg`, `<name>.tiff`, and `<name>.png`
- `cleaned_xrd_data.csv` and `reference_peak_table.csv`
- `project.xrdviz.json`, which can be reopened in XRDViz
- `xrd_plot_report.md` and `publication_manifest.json`, including output hashes and source-file status

## Batch and In-situ Workflow

Use **File -> Import spectra folder...** for folders of `.txt`, `.csv`, `.xy`, or `.dat` patterns. XRDViz infers metadata from common filename forms such as:

- `Az_Full_000123.txt` -> frame `123`
- `scan_0007_12.5min_650C.xy` -> frame `7`, time `750 s`, temperature `650 C`

The **Batch** tab controls:

- view mode: overlay, stack, gradient stack, or heatmap
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

## Run

```powershell
py -3.12 -m pip install -e .
py -3.12 -m xrdviz
```

XRDViz is currently distributed for source execution rather than as a Windows executable.
