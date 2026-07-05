# XRDViz

XRDViz is a source-run Python/Qt desktop application for plotting one-dimensional XRD spectra.

## Features

- Drag and drop `.txt`, `.csv`, `.xy`, and `.dat` spectrum files.
- Declare input X axis as `2theta`, `d`, or `q`, then convert all layers through a global energy setting.
- Display linear, normalized, log, stacked, or log-stacked spectra without mutating raw data.
- Load CIF files and draw Bragg tick marks below the main plot.
- Import `sample_labels.csv` to control sample labels, order, colors, visibility, and offsets.
- Import `reference_peaks.csv` or simple Rigaku-style `peaks.csv` files as reference phase markers.
- Show phase-specific marker symbols, guide lines, direct curve labels, compact phase legends, and a reference peak table.
- Export publication-oriented figures as PDF, SVG, PNG, or TIFF.
- Export a publication bundle with the figure, cleaned data, reference peak table, and reproducibility report.

## CSV Helpers

`sample_labels.csv`:

```csv
filename,label,order,color,visible,offset
sample_a.xy,Annealed,1,#D55E00,true,0.3
sample_b.xy,As cast,2,#0072B2,true,0.0
```

`reference_peaks.csv`:

```csv
position,label,phase,intensity,hkl,source_axis,color,shape
30.0,Main peak,Calcite,100,104,two_theta,#009E73,triangle
2.5,d peak,Calcite,40,110,d,#009E73,square
```

`source_axis` accepts `two_theta`, `d`, or `q`; peaks are converted to the current plot axis using the global energy setting.

## Run

```powershell
python -m pip install -e .
python -m xrdviz
```

The first version is designed for source execution rather than Windows executable packaging.
