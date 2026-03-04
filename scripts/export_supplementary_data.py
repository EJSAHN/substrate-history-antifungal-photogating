#!/usr/bin/env python
"""
Pack analysis tables into a single Excel workbook (Supplementary Data 1).

This script is intentionally "dumb": it does not recompute anything.
It simply reads all CSV files in a tables directory and writes each one to a sheet.

Backwards-compatible args:
  --input-dir /path/to/tables   (alias: --tables-dir)
  --output /path/to.xlsx        (alias: --out-xlsx)
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import pandas as pd

def safe_sheet_name(name: str) -> str:
    # Excel sheet name: <=31 chars, no []:*?/\
    name = re.sub(r'[\[\]\:\*\?\/\\]', "_", name)
    return name[:31]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default=None, help="Directory containing CSV tables (legacy arg).")
    ap.add_argument("--tables-dir", default=None, help="Directory containing CSV tables.")
    ap.add_argument("--output", default=None, help="Output .xlsx path (legacy arg).")
    ap.add_argument("--out-xlsx", default=None, help="Output .xlsx path.")
    args = ap.parse_args()

    tables_dir = Path(args.tables_dir or args.input_dir or ".").resolve()
    out_xlsx = Path(args.out_xlsx or args.output or "supplementary_data_1_generated.xlsx").resolve()

    if not tables_dir.exists():
        raise SystemExit(f"[ERROR] tables dir not found: {tables_dir}")

    csvs = sorted([p for p in tables_dir.glob("*.csv")])
    if not csvs:
        raise SystemExit(f"[ERROR] no CSV files found in: {tables_dir}")

    out_xlsx.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as xw:
        # index sheet
        index_rows = []
        for p in csvs:
            df = pd.read_csv(p)
            sheet = safe_sheet_name(p.stem)
            index_rows.append({"sheet": sheet, "file": p.name, "rows": len(df), "cols": len(df.columns)})
            df.to_excel(xw, sheet_name=sheet, index=False)
        pd.DataFrame(index_rows).to_excel(xw, sheet_name="INDEX", index=False)

    print(f"[OK] Wrote: {out_xlsx}")

if __name__ == "__main__":
    main()
