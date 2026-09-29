# Input specification

The pipeline reproduces the UVSM study design and expects the inputs below.

## Morphology

Provide one worksheet named `phenotype` in `phenotype.xlsx`, or a UTF-8 CSV with
the same headers. Preserve the original row order. Required columns:

```
Strain, Chemical, UV exposure (mJ/cm2), Area size, Perimeter, Length,
Width, LWR, Circularity, IS and CG
```

The design includes four isolates (`P24-192`, `CGH17`, `CGH5`, `CGH49`), three
conditions (`Control`, `PhSOAM`, `PhSOFA`) and UV levels 0, 12, 35 and 70.
Exactly nine finite measurement rows per cell are expected. Noninteger dose
labels, missing values, exact duplicate measurement records and nonpositive
area stop processing; they are not automatically removed or corrected.

`source_excel_row` and `row_identifier` locate source records. Run and batch
identifiers are not available in the source phenotype table.

## Fluorescence

Provide `fluorescence/A1_F_pixelSpectra.csv` through A9; likewise B1–B9,
C1–C9, D1–D9, E1–E4 and F1–F4. Each file begins with `row,col` followed by
60 numeric wavelength headings on a common increasing grid. Exported coordinates
and intensities are used as supplied.

Circular colony ROIs were selected to approximate the colony outlines; irregular
marginal extensions could be excluded. Central and peripheral zones are calculated
from the pixel coordinates, rather than read from separately averaged region spectra.
Instrument calibration and physical pixel dimensions are not encoded in the
wavelength headings.

Alternatively, supply an HSI ZIP. Only matching fluorescence filenames are read;
arbitrary paths are not extracted. Byte-identical duplicates are counted once.
Different byte versions for the same export identifier stop the analysis.
Do not supply both a folder and a ZIP in the same input directory.

An optional `input_manifest.json` lists `files` as objects with `file` (basename)
and `sha256`. When supplied, it must match all selected inputs exactly.

## Optional archived data

`archived_jmp_variable_clustering.csv` can be supplied in the input directory.
It is included as an archived summary without recalculation. Without it, the
corresponding worksheet contains headings but no data rows.

Historical comparisons use an archived baseline CSV directory supplied through
`--baseline-dir`. These reference files are not required to calculate results
from the experimental inputs.
