# Input specification

This is a study-specific reproduction pipeline, not a generic plate-analysis package.

## Morphology

One worksheet named `phenotype` in `phenotype.xlsx`, or a UTF-8 CSV with the same
headers. Preserve original row order. Required columns:

```
Strain, Chemical, UV exposure (mJ/cm2), Area size, Perimeter, Length,
Width, LWR, Circularity, IS and CG
```

The study design is four isolates (`P24-192`, `CGH17`, `CGH5`, `CGH49`), three
conditions (`Control`, `PhSOAM`, `PhSOFA`) and UV levels 0, 12, 35, 70.
Exactly nine finite measurement rows per cell are expected. Noninteger dose
labels, missing values, exact duplicate measurement records and nonpositive
area stop processing rather than triggering automatic deletion or correction.

`source_excel_row` and `row_identifier` are location labels, not evidence of
biological independence or experimental dates. No run/batch values are invented.

## Fluorescence

`fluorescence/A1_F_pixelSpectra.csv` through A9; likewise B1–B9, C1–C9,
D1–D9, E1–E4 and F1–F4. Each file begins with `row,col` followed by the 60
numeric wavelength headings on a common increasing grid. All exported intensities
are used as supplied. Instrument calibration and physical pixel dimensions are
not inferred from the CSV headings.

Alternatively supply an HSI ZIP. Only matching fluorescence filenames are read;
arbitrary paths are not extracted. Byte-identical duplicates are counted once,
and different byte versions for the same export identifier stop the analysis.
Do not supply both a folder and a ZIP in the same input directory.

An optional `input_manifest.json` lists `files` as objects with `file` (basename)
and `sha256`. When supplied, it must match all selected inputs exactly.

## Optional archived data

`archived_jmp_variable_clustering.csv` can be supplied in the input directory.
It is copied as an archived summary, never recomputed or replaced by approximate
variable clustering. Without it, the workbook retains the corresponding heading
with no invented results.

Historical reproduction checks need the separately archived baseline CSV directory
passed as `--baseline-dir`. Those files are not required to calculate new results.
