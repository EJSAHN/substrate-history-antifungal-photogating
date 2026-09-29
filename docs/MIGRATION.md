# Migration to version 1.2.0

Use `scripts/run_analysis.py` in place of `scripts/run_all_nofig_v4.py`.
The previous runner and overlapping HSI wrappers remain available in Git history.

The updated workflow adds factorial models, matched-control inhibition,
exact spectral permutation tests and sensitivity analyses. Workbook export uses
an explicit worksheet schema. Approximate variable clustering has been removed;
archived JMP results are supplied separately and are not recalculated.

Inputs are specified by `--input-dir` rather than machine-specific recovery paths.
Experimental data, historical reference tables and figure-generation scripts are
maintained separately. The public and local workflows use the same numerical code.

The repository retains its MIT license designation. The full license text replaces
the abbreviated text in the initial version, with the copyright line unchanged.

The documentation update clarifies circular ROI selection, data availability and
workbook descriptions. Numerical definitions and random seeds are unchanged.


## 1.2.1

Added focal-band component contrasts, eight-test Holm-adjusted p values,
sampled-ROI geometry tables and a descriptive paired log-ratio decomposition.
The existing numerical analyses and their random streams are unchanged.
Supplementary Data 1 includes four additional sheets. Archive-processing
labels are not interpreted as acquisition sessions. Local figure production
includes component-specific panels; no figure code is added to this repository.
