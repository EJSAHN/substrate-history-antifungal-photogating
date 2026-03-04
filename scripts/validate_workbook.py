#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.rename(columns=lambda c: str(c).strip())
    for c in out.columns:
        # Normalize NaNs and whitespace in strings
        if out[c].dtype == object:
            out[c] = out[c].astype(str).str.strip()
            out[c] = out[c].replace({"nan": ""})
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Validate generated workbook against a reference.")
    p.add_argument("generated", type=str, help="Generated workbook.")
    p.add_argument("reference", type=str, help="Reference workbook.")
    args = p.parse_args()

    gen = Path(args.generated)
    ref = Path(args.reference)

    gen_xl = pd.ExcelFile(gen)
    ref_xl = pd.ExcelFile(ref)

    gen_sheets = set(gen_xl.sheet_names)
    ref_sheets = set(ref_xl.sheet_names)

    missing = sorted(ref_sheets - gen_sheets)
    extra = sorted(gen_sheets - ref_sheets)

    if missing:
        print(f"Missing sheets: {', '.join(missing)}")
    if extra:
        print(f"Extra sheets: {', '.join(extra)}")

    common = sorted(gen_sheets & ref_sheets)
    any_diff = False

    for name in common:
        g = _normalize(pd.read_excel(gen, sheet_name=name))
        r = _normalize(pd.read_excel(ref, sheet_name=name))

        if g.shape != r.shape or list(g.columns) != list(r.columns):
            print(f"Sheet '{name}': shape/columns differ (gen={g.shape}, ref={r.shape}).")
            any_diff = True
            continue

        # Compare values (exact match after normalization)
        neq = (g.values != r.values)
        if hasattr(neq, "any") and neq.any():
            diffs = int(neq.sum())
            print(f"Sheet '{name}': {diffs} cell(s) differ.")
            any_diff = True

    if not any_diff and not missing and not extra:
        print("Validation passed.")
        return 0

    print("Validation failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
