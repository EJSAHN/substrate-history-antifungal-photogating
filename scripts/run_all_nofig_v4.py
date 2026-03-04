#!/usr/bin/env python
"""
run_all_nofig_v4.py

One command -> Supplementary Data 1 workbook (NO figures).

Expected input layout (default):
  input/
    phenotype.xlsx
    HSI.zip

Outputs:
  out/
    analysis_tables/*.csv
    supplementary_data_1_generated.xlsx

What runs:
1) scripts/phenotype_pipeline.py          (phenotype -> CSV tables)
2) scripts/hsi_excel_only_tables_v4.py   (HSI -> CSV tables; excludes R_0075 / e4_0.0075)
3) scripts/export_supplementary_data.py  (pack all CSVs -> Excel)

This intentionally skips any plotting scripts and any add-on figure makers.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def run(cmd: list[str]) -> None:
    print("[RUN]", " ".join(cmd))
    r = subprocess.run(cmd, check=False)
    if r.returncode != 0:
        raise SystemExit(r.returncode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default="input", help="Base input dir (default: input)")
    ap.add_argument("--outdir", default="out", help="Base output dir (default: out)")
    ap.add_argument("--sheet", default="phenotype", help="Phenotype Excel sheet name")
    ap.add_argument("--nboot", type=int, default=5000, help="Bootstrap resamples for phenotype tables (default 5000)")

    # HSI params
    ap.add_argument("--hsi-wl", type=float, default=518.8, help="HSI RF target wavelength (nm)")
    ap.add_argument("--swir-wl", type=float, default=1450.0, help="HSI SWIR target wavelength (nm)")
    ap.add_argument("--core-frac", type=float, default=0.30, help="Core radius fraction")
    ap.add_argument("--edge-frac", type=float, default=0.80, help="Edge inner radius fraction")
    ap.add_argument("--hsi-boot", type=int, default=10000, help="Bootstrap resamples for HSI CIs (default 10000)")
    ap.add_argument("--hsi-seed", type=int, default=42, help="Seed for HSI bootstrap (default 42)")

    args = ap.parse_args()

    base = Path(args.input_dir)
    out = Path(args.outdir)
    tables = out / "analysis_tables"
    tables.mkdir(parents=True, exist_ok=True)

    phen = base / "phenotype.xlsx"
    hsi_zip = base / "HSI.zip"

    if not phen.exists():
        raise SystemExit(f"[ERROR] phenotype.xlsx not found: {phen}")
    if not hsi_zip.exists():
        raise SystemExit(f"[ERROR] HSI.zip not found: {hsi_zip}")

    # 1) phenotype tables
    run([sys.executable, "scripts/phenotype_pipeline.py",
         "--phenotype", str(phen),
         "--sheet", args.sheet,
         "--outdir", str(tables),
         "--nboot", str(args.nboot)])

    # 2) HSI tables (NO figures, excludes R_0075)
    run([sys.executable, "scripts/hsi_excel_only_tables_v4.py",
         "--zip", str(hsi_zip),
         "--outdir", str(tables),
         "--modalities", "F,R,SWIR",
         "--wl", str(args.hsi_wl),
         "--swir-wl", str(args.swir_wl),
         "--core-frac", str(args.core_frac),
         "--edge-frac", str(args.edge_frac),
         "--boot", str(args.hsi_boot),
         "--seed", str(args.hsi_seed)])

    # 3) Pack to Excel
    run([sys.executable, "scripts/export_supplementary_data.py",
         "--tables-dir", str(tables),
         "--out-xlsx", str(out / "supplementary_data_1_generated.xlsx")])

    print("[OK] Done. Outputs under:", out.resolve())


if __name__ == "__main__":
    main()
