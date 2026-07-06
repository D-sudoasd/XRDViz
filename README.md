# XRDViz

XRDViz is a source-run Python/Qt desktop application for plotting one-dimensional XRD spectra.

## Features

- Drag and drop `.txt`, `.csv`, `.xy`, and `.dat` spectrum files.
- Declare input X axis as `2theta`, `d`, or `q`, then convert all layers through a global energy setting.
- Display linear, normalized, log, stacked, or log-stacked spectra without mutating raw data.
- Batch import spectrum folders, infer frame/time/temperature metadata from filenames, sort by metadata, and draw overlay, stack, gradient stack, or heatmap views.
- Load CIF files and draw Bragg tick marks below the main plot.
- Import `sample_labels.csv` to control sample labels, order, colors, visibility, and offsets.
- Import `reference_peaks.csv` or simple Rigaku-style `peaks.csv` files as reference phase markers.
- Show phase-specific marker symbols, guide lines, direct curve labels, compact phase legends, and a reference peak table.
- Apply Nature single/double, Science single/double, or custom publication templates with adjustable legend placement, fonts, dimensions, and margins.
- Export publication-oriented figures as PDF, SVG, PNG, or TIFF, including batch heatmaps with colorbars.
- Export a publication bundle with the figure, cleaned data, reference peak table, and reproducibility report.

## Batch and In-situ Workflow

Use **File -> Import spectra folder...** for folders of `.txt`, `.csv`, `.xy`, or `.dat` patterns. XRDViz infers metadata from common filename forms such as:

- `Az_Full_000123.txt` -> frame `123`
- `scan_0007_12.5min_650C.xy` -> frame `7`, time `750 s`, temperature `650 C`

The **Batch** tab controls:

- view mode: overlay, stack, gradient stack, or heatmap
- sort/color fields: frame, time, temperature, or current order
- colormap, colorbar, show every N spectra, and heatmap sampling points

The publication bundle report records these batch settings and each spectrum's inferred metadata.

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
python -m pip install -e .
python -m xrdviz
```

The first version is designed for source execution rather than Windows executable packaging.
