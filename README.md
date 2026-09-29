# UVSM analysis

This repository provides code for analysing colony morphology and spatial fluorescence
in UV-C-treated fungal bioassays. Version 1.2.0 generates analysis tables and the
Supplementary Data 1 workbook. Figure preparation is maintained separately.

## Run

Python 3.10 or later is required. Use an existing scientific environment or
install the dependencies in a separate environment with `pip install -r requirements.txt`.
The scripts do not install or upgrade packages themselves.

Supply exactly one `input/phenotype.xlsx` or `input/phenotype.csv`, and either
`input/fluorescence/` containing the 44 fluorescence pixel CSVs or `input/HSI.zip`.
See the [input specification](docs/INPUTS.md) for file formats.

```bash
python scripts/run_analysis.py --input-dir input --outdir out/run_r1
```

The destination must be new or empty; input files are read-only.
Outputs include CSV tables, model diagnostics, input and output checksums,
`Supplementary_Data_1_R1.xlsx`, and a run summary. No figures are generated.
The workbook is generated from the current run. Its formulas calculate display
quantities; model estimates and inferential statistics are calculated in Python.

```bash
python -m unittest discover -s tests -v
```

The tests check numerical routines, input validation and workbook export using
small synthetic datasets.

## Analyses

- Full isolate × compound × categorical UV model with HC3 covariance, treatment
  contrasts, and ordinary OLS and log-area sensitivity analyses.
- Cell means, bootstrap confidence intervals, and inhibition relative to the
  same-isolate, same-UV vehicle. Shared controls are accounted for in bootstrap comparisons.
- Standardised morphometric PCA and correlations. An archived JMP clustering
  summary can be supplied separately and is not recalculated.
- Pixelwise normalised fluorescence, radial ROI ratios, observed UV contrasts,
  historical Monte Carlo comparisons, exact maximum-absolute-difference tests,
  and studentized maximum-|t| sensitivity tests.
- Radial-threshold sensitivity with pointwise intervals using the same observations.

The [methods](docs/METHODS.md) describe estimands, multiplicity corrections and
random seeds. Both maximum-statistic tests are reported.

## Interpretation and historical comparisons

The inputs contain 432 morphology measurements and 44 unique fluorescence exports.
The morphology table does not contain experimental-run identifiers; its row labels
identify source records. These data do not establish between-run reproducibility
or distinguish substrate changes from compound photochemistry.

Circular colony ROIs were selected to approximate colony outlines, so irregular
marginal extensions could lie outside the sampled regions. Central and peripheral
zones are defined from the exported coordinates. The peripheral zone is not an
exact tracing of the outermost growth boundary. Physical pixel dimensions and
biochemical assignments of spectral features require separate acquisition evidence.

An optional `--baseline-dir` compares archived reference tables. Without it,
historical comparisons are omitted. Some archived dose-table confidence-interval
limits have not been reproduced exactly; the comparison records this discrepancy
as a warning when those reference tables are supplied.

The former `run_all_nofig_v4.py` workflow is retired; its source remains in Git
history. Reflectance and SWIR analyses are not part of this fluorescence workflow.

## Data availability

This repository contains analysis code only. Experimental inputs and the generated
Supplementary Data 1 workbook are not included. The required phenotype table and
fluorescence ROI pixel spectra are described in [docs/INPUTS.md](docs/INPUTS.md).

## Repository contents

`src/uvsm/`: analyses, input validation and workbook export.  
`scripts/`: command-line entry points.  
`tests/`: numerical and file-safety checks.  
`docs/`: input specification, methods and migration notes.  
`input/` and `out/`: placeholders; data and generated results are ignored by Git.

## Citation and license

Use `CITATION.cff` to cite this software. Cite the associated article separately
when available. The code is distributed under the MIT License; see `LICENSE`.
