# UVSM — substrate-history carryover analysis

This repository provides a **no-figure**, reproducible pipeline to generate the analysis tables and the
**Supplementary Data 1 Excel workbook** for the UV substrate-history (UVSM) experiments described in the
Photogating / substrate environmental history manuscript.

**What it does**
- Reads phenotype/morphometrics (plate images → SmartGrain outputs summarized in an Excel workbook)
- Reads fluorescence HSI data from a ZIP bundle
- Produces analysis tables under `out/analysis_tables/`
- Exports `out/supplementary_data_1_generated.xlsx` (CSV tables packaged into a single workbook)

This repo intentionally **does not generate figures**.

---

## Quickstart (Windows / Anaconda)

From the repository root:

```bat
conda create -n uvsm python=3.11 -y
conda activate uvsm
pip install -r requirements.txt
set PYTHONPATH=%CD%\src
python scripts/run_all_nofig_v4.py --input-dir input --outdir out --nboot 5000
```

Outputs:
- `out/analysis_tables/*.csv`
- `out/supplementary_data_1_generated.xlsx`

---

## Inputs

Place inputs under `input/`:

- `input/phenotype.xlsx`
  - The phenotype workbook (see `scripts/phenotype_pipeline.py` for expected sheet/column names).
- `input/HSI.zip`
  - Zipped HSI bundle consumed by `scripts/hsi_excel_only_tables_v4.py`.

Notes:
- `input/README.md` explains the expected input layout.
- Large/raw data should **not** be committed to GitHub.

---

## Reproducibility parameters (paper alignment)

The pipeline uses bootstrap-based uncertainty summaries.

Defaults (as executed by `run_all_nofig_v4.py`):
- **Morphometrics / phenotype**: `--nboot 5000` (bootstrap resamples), internal seed = **1337**
- **Fluorescence HSI**: `--boot 10000`, `--seed 42`

If you need to regenerate morphometrics CIs with a different number of resamples, set `--nboot` in the
`run_all_nofig_v4.py` command. (HSI bootstrap parameters are already fixed in the runner call.)

---

## Validate against a reference workbook (optional)

If you have a reference workbook (e.g., the submission-time “final” workbook), you can compare:

```bat
python scripts/validate_workbook.py out/supplementary_data_1_generated.xlsx input/reference.xlsx
```

`validate_workbook.py` expects two positional arguments: `generated` then `reference`.

---

## Repository layout

- `src/uvsm/` : lightweight utilities for Excel formatting/export
- `scripts/`  : executable pipelines (no-figure)
- `input/`    : inputs (ignored by default; keep only `input/README.md`)
- `out/`      : outputs (ignored by default; keep only `out/README.md`)

---

## License & citation

See `LICENSE` and `CITATION.cff`.
