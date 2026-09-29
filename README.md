# UVSM analysis

Reproducible morphology and fluorescence analyses for UV-C assay-history effects
in fungal bioassays. Release **1.2.0** separates numerical analysis from private
recovery tools and figure preparation. It retains the numerical definitions
validated in R1 analysis 1.0.0.

## Run

Python 3.10 or later is required. Use an existing scientific environment or
install the dependencies in a separate environment with `pip install -r requirements.txt`.
The scripts do not install or upgrade packages themselves.

Supply exactly one `input/phenotype.xlsx` or `input/phenotype.csv`, and either
`input/fluorescence/` containing the 44 fluorescence pixel CSVs or `input/HSI.zip`.
Input data are not distributed in this repository. See [input format](docs/INPUTS.md).

```bash
python scripts/run_analysis.py --input-dir input --outdir out/run_r1
```

The destination must be new or empty. No source files are edited.
Results include CSV tables, model diagnostics, a source-hash manifest,
`Supplementary_Data_1_R1.xlsx`, and the run summary. No figures are generated.
The workbook is built from the current run, not copied from a frozen result file.
Its formulas perform display arithmetic only; inferential statistics come from Python.

```bash
python -m unittest discover -s tests -v
```

These tests use small artificial arrays to check algorithms and I/O guards;
they are not biological validation experiments.

## Analyses

- Full isolate × compound × categorical UV model; HC3 tests and explicit treatment
  contrasts, with ordinary OLS and log-area sensitivity results retained.
- Cell means, bootstrap confidence intervals, and inhibition relative to the
  same-isolate, same-UV vehicle. Shared control uncertainty is reused in comparisons.
- Standardised morphometric PCA and correlations. Archived JMP clustering may be
  supplied separately; no Python approximation is substituted for the JMP result.
- Pixelwise normalised fluorescence, radial ROI ratios, observed UV contrasts,
  historical Monte Carlo comparisons, exact maximum-absolute-difference tests,
  and studentized maximum-|t| sensitivity tests.
- Radial-threshold sensitivity with pointwise intervals. The same exports are
  reused; this is not independent validation.

All inferential definitions, multiplicity families, and seeds are documented in
[methods](docs/METHODS.md). Both maximum-statistic analyses are always retained.
No test is selected because it gives a smaller p value.

## Reproduction and scope

The inputs comprise 432 measurement rows and 44 unique fluorescence exports.
Row locators are not recovered experimental-run identifiers. The code does not
establish independent batch replication, distinguish substrate change from
compound photochemistry, or identify a fluorophore. The inner/outer regions are
defined geometrically in the supplied ROI coordinates. Their correspondence to
an anatomical growth front requires acquisition provenance.

An optional `--baseline-dir` compares preserved historical tables. Without it,
the run explicitly reports that historical comparisons were not performed.
The historical dose-table confidence-interval discrepancy is retained as a
warning when those baselines are supplied; it is not repaired by widening tolerances.

The former `run_all_nofig_v4.py` workflow is retired from this release. Its
source remains in Git history; use the command above. Reflectance and SWIR
branches are not part of the current R1 fluorescence-only analysis.

## Repository contents

`src/uvsm/`: analyses, input validation and workbook export.  
`scripts/`: command-line entry points.  
`tests/`: mathematical and file-safety checks.  
`docs/`: input specification, numerical methods and migration notes.  
`input/` and `out/`: placeholders only; real files are ignored by Git.

No acquisition data, private computer paths, review reports, draft manuscripts,
figure-production code or generated figures are included.

## Citation and license

Use `CITATION.cff` to cite this software. A journal article should be cited
separately when available.
The repository retains its existing MIT designation; see `LICENSE`.
