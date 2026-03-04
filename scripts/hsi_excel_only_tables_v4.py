#!/usr/bin/env python3
"""
hsi_excel_only_tables_v4.py

Goal (GitHub-friendly, no figures)
- Compute UVSM HSI summaries (plate metrics, group mean+bootstrap CI, contrasts) and write CSVs
- NO plotting, NO matplotlib import
- Explicitly EXCLUDE the reflectance 0.0075 setting (R_0075 / e4_0.0075) from discovery/processing.

Inputs
- Either:
  --zip <HSI.zip>              (preferred for repo users)
  OR
  --input-dir <HSI root dir>   (folder containing RF/ and swir/)

Outputs (written into --outdir)
- hsi_ec_metrics_per_plate.csv
- hsi_ec_ratio_group_summary.csv
- hsi_contrasts.csv

Notes
- Group letters in filenames (A–F) are preserved for traceability, but outputs also include
  human-readable labels (EtOH_UV0, etc.).
- Seed controls bootstrap resampling only (reproducible CIs).
"""

from __future__ import annotations

import argparse
import os
import re
import zipfile
from dataclasses import dataclass
from io import TextIOWrapper
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ----------------------------
# Group mapping (paper labels)
# ----------------------------
GROUP_LABEL: Dict[str, str] = {
    "A": "EtOH_UV0",
    "B": "EtOH_UV70",
    "C": "PhSOFA_UV0",
    "D": "PhSOFA_UV70",
    "E": "Blank_UV0",
    "F": "Blank_UV70",
}


# ----------------------------
# Parsing helpers
# ----------------------------
def parse_group_rep_from_filename(filename: str) -> Tuple[Optional[str], Optional[int]]:
    """
    Basenames like:
      A1_F_pixelSpectra.csv
      C7_R_pixelSpectra.csv
      B9_pixelSpectra.csv  (SWIR)
    """
    base = os.path.basename(filename)
    base = re.sub(r'(_F_pixelSpectra|_R_pixelSpectra|_pixelSpectra)\.csv$', '', base)
    m = re.match(r'^([A-F])(\d+)$', base)
    if not m:
        return None, None
    return m.group(1), int(m.group(2))


def is_wavelength_col(c: str) -> bool:
    return bool(re.fullmatch(r"\d+\.\d+", str(c)))


def nearest_wavelength_column(columns: List[str], target_nm: float) -> str:
    wls = [float(c) for c in columns if is_wavelength_col(str(c))]
    if not wls:
        raise ValueError("No wavelength columns found.")
    closest = min(wls, key=lambda x: abs(x - target_nm))
    return f"{closest:.1f}"


# ----------------------------
# ROI definitions
# ----------------------------
@dataclass(frozen=True)
class ROIParamsFractional:
    core_frac: float = 0.30
    edge_frac: float = 0.80


def core_edge_masks(rowcol: np.ndarray, roi: ROIParamsFractional) -> Tuple[np.ndarray, np.ndarray]:
    centroid = rowcol.mean(axis=0)
    d = np.sqrt(((rowcol - centroid) ** 2).sum(axis=1))
    maxd = float(d.max()) if d.size else 0.0
    if maxd == 0.0:
        core = np.ones_like(d, dtype=bool)
        edge = np.ones_like(d, dtype=bool)
        return core, edge
    core = d <= roi.core_frac * maxd
    edge = d >= roi.edge_frac * maxd
    return core, edge


# ----------------------------
# Reader abstraction (ZIP or dir)
# ----------------------------
class Reader:
    def list_paths(self) -> List[str]:
        raise NotImplementedError

    def open_text(self, path: str):
        raise NotImplementedError


class ZipReader(Reader):
    def __init__(self, zip_path: str):
        self.zip_path = zip_path
        self.zf = zipfile.ZipFile(zip_path)

    def list_paths(self) -> List[str]:
        return self.zf.namelist()

    def open_text(self, path: str):
        b = self.zf.open(path)
        return TextIOWrapper(b, encoding="utf-8", newline="")

    def close(self) -> None:
        self.zf.close()


class DirReader(Reader):
    def __init__(self, input_dir: str):
        self.root = Path(input_dir).resolve()

    def list_paths(self) -> List[str]:
        out: List[str] = []
        for p in self.root.rglob("*"):
            if p.is_file():
                out.append(p.relative_to(self.root).as_posix())
        return out

    def open_text(self, rel_path: str):
        p = self.root / Path(rel_path)
        return open(p, "r", encoding="utf-8", newline="")

    def to_abs(self, rel_path: str) -> str:
        return str((self.root / Path(rel_path)).resolve())


# ----------------------------
# Metric computation
# ----------------------------
def compute_plate_ec(
    df: pd.DataFrame,
    roi: ROIParamsFractional,
    *,
    mode: str,
    target_wl: float,
) -> Dict[str, Any]:
    """
    mode:
      - "F_NFI": RF fluorescence: NFI at target_wl = I(wl) / sum(I)
      - "RAW_WL": reflectance/SWIR: raw value at target_wl
    """
    if not {"row", "col"}.issubset(df.columns):
        raise ValueError("Expected columns 'row' and 'col'.")

    rowcol = df[["row", "col"]].to_numpy(dtype=float)
    core_mask, edge_mask = core_edge_masks(rowcol, roi)

    n_core = int(core_mask.sum())
    n_edge = int(edge_mask.sum())

    if n_core == 0 or n_edge == 0:
        return dict(core_mean=np.nan, edge_mean=np.nan, ec_ratio=np.nan, n_core=n_core, n_edge=n_edge, wl_col=None)

    wl_col = nearest_wavelength_column(list(df.columns), target_wl)
    spec_cols = [c for c in df.columns if c not in ("row", "col")]

    X = df[spec_cols].to_numpy(dtype=float)
    j = spec_cols.index(wl_col)

    if mode == "F_NFI":
        denom = X.sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            vals = np.where(denom > 0, X[:, j] / denom, np.nan)
    elif mode == "RAW_WL":
        vals = X[:, j]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    core_vals = vals[core_mask]
    edge_vals = vals[edge_mask]

    core_mean = float(np.nanmean(core_vals)) if core_vals.size else np.nan
    edge_mean = float(np.nanmean(edge_vals)) if edge_vals.size else np.nan
    ec = float(edge_mean / core_mean) if (not np.isnan(core_mean) and core_mean != 0.0) else np.nan

    return dict(core_mean=core_mean, edge_mean=edge_mean, ec_ratio=ec, n_core=n_core, n_edge=n_edge, wl_col=wl_col)


def bootstrap_mean_ci(x: np.ndarray, nboot: int, seed: int) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.nan, np.nan, np.nan
    means = np.empty(nboot, dtype=float)
    for i in range(nboot):
        means[i] = rng.choice(x, size=x.size, replace=True).mean()
    return float(np.mean(x)), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def bootstrap_mean_diff_ci(x: np.ndarray, y: np.ndarray, nboot: int, seed: int) -> Tuple[float, float, float]:
    """Return mean(y)-mean(x) and bootstrap 95% CI."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    if x.size == 0 or y.size == 0:
        return np.nan, np.nan, np.nan
    diffs = np.empty(nboot, dtype=float)
    for i in range(nboot):
        xb = rng.choice(x, size=x.size, replace=True).mean()
        yb = rng.choice(y, size=y.size, replace=True).mean()
        diffs[i] = yb - xb
    return float(np.mean(diffs)), float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))


# ----------------------------
# File discovery (EXCLUDES R_0075)
# ----------------------------
MODALITY_SPECS = {
    "F": {
        "regex": r"/RF/F_roi_spectra/.+_F_pixelSpectra\.csv$",
        "mode": "F_NFI",
        "target_wl_arg": "wl",
        "name": "RF_F",
    },
    "R": {
        "regex": r"/RF/R_roi_spectra/.+_R_pixelSpectra\.csv$",
        "mode": "RAW_WL",
        "target_wl_arg": "wl",
        "name": "RF_R",
    },
    "SWIR": {
        "regex": r"/swir/swir_roi_spectra/.+_pixelSpectra\.csv$",
        "mode": "RAW_WL",
        "target_wl_arg": "swir_wl",
        "name": "SWIR",
    },
}


def discover_files(reader: Reader, modality_key: str, all_paths: List[str]) -> List[str]:
    rx = re.compile(MODALITY_SPECS[modality_key]["regex"], flags=re.IGNORECASE)
    # hard exclude any path containing R_0075 (0.0075 setting)
    found = [p for p in all_paths if ("r_0075_roi_spectra" not in p.lower()) and (rx.search("/" + p) or rx.search(p))]
    found.sort()
    return found


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--zip", dest="zip_path", help="Path to HSI ZIP (HSI.zip).")
    src.add_argument("--input-dir", dest="input_dir", help="Root folder containing RF/ and swir/.")

    ap.add_argument("--outdir", required=True, help="Output directory for CSV tables.")
    ap.add_argument("--modalities", default="F,R,SWIR", help="Comma-separated: F,R,SWIR (default: F,R,SWIR).")
    ap.add_argument("--wl", type=float, default=518.8, help="Target wavelength (nm) for RF metrics (default: 518.8).")
    ap.add_argument("--swir-wl", type=float, default=1450.0, help="Target wavelength (nm) for SWIR metrics (default: 1450).")
    ap.add_argument("--core-frac", type=float, default=0.30, help="Core radius as fraction of max distance.")
    ap.add_argument("--edge-frac", type=float, default=0.80, help="Edge inner radius as fraction of max distance.")
    ap.add_argument("--boot", type=int, default=10000, help="Bootstrap resamples for CIs (default: 10000).")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for bootstrap (default: 42).")
    args = ap.parse_args()

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    roi = ROIParamsFractional(core_frac=float(args.core_frac), edge_frac=float(args.edge_frac))

    reader: Reader
    zip_reader: Optional[ZipReader] = None
    dir_reader: Optional[DirReader] = None
    if args.zip_path:
        zip_reader = ZipReader(args.zip_path)
        reader = zip_reader
    else:
        dir_reader = DirReader(args.input_dir)
        reader = dir_reader

    try:
        all_paths = reader.list_paths()

        modalities = [m.strip() for m in args.modalities.split(",") if m.strip()]
        for m in modalities:
            if m not in MODALITY_SPECS:
                raise ValueError(f"Unknown modality '{m}'. Use one of: {', '.join(MODALITY_SPECS.keys())}")

        # 1) per-plate metrics
        plate_rows: List[Dict[str, Any]] = []

        for m in modalities:
            spec = MODALITY_SPECS[m]
            files = discover_files(reader, m, all_paths)
            if not files:
                print(f"[WARN] No files found for modality={m} (regex={spec['regex']})")
                continue

            target_wl = args.wl if spec["target_wl_arg"] == "wl" else args.swir_wl
            mode = spec["mode"]

            for rel_path in files:
                with reader.open_text(rel_path) as f:
                    df = pd.read_csv(f)

                group, rep = parse_group_rep_from_filename(rel_path)
                if group is None or rep is None:
                    continue

                metrics = compute_plate_ec(df, roi, mode=mode, target_wl=float(target_wl))

                abs_path = rel_path
                if dir_reader is not None:
                    abs_path = dir_reader.to_abs(rel_path)

                plate_rows.append({
                    "modality": spec["name"],
                    "group_code": group,                 # A-F
                    "group": GROUP_LABEL.get(group, group),
                    "rep": rep,
                    "plate": f"{group}{rep}",
                    "target_wl_nm": float(target_wl),
                    "wl_col_used": metrics["wl_col"],
                    "core_mean": metrics["core_mean"],
                    "edge_mean": metrics["edge_mean"],
                    "ec_ratio": metrics["ec_ratio"],
                    "n_core": metrics["n_core"],
                    "n_edge": metrics["n_edge"],
                    "path": abs_path,
                })

        plate_df = pd.DataFrame(plate_rows).sort_values(["modality", "group_code", "rep"]).reset_index(drop=True)
        (outdir / "hsi_ec_metrics_per_plate.csv").write_text("", encoding="utf-8")  # ensure file is creatable
        plate_df.to_csv(outdir / "hsi_ec_metrics_per_plate.csv", index=False)
        print(f"[OK] Wrote: {outdir / 'hsi_ec_metrics_per_plate.csv'}")

        # 2) group summary
        grp_rows: List[Dict[str, Any]] = []
        for (modality, group_code, group_label), sub in plate_df.groupby(["modality", "group_code", "group"], dropna=False):
            x = sub["ec_ratio"].to_numpy(dtype=float)
            mean, lo, hi = bootstrap_mean_ci(x, nboot=int(args.boot), seed=int(args.seed))
            grp_rows.append({
                "modality": modality,
                "group_code": group_code,
                "group": group_label,
                "n": int(np.sum(~np.isnan(x))),
                "mean_ec_ratio": mean,
                "ci_low": lo,
                "ci_high": hi,
                "seed": int(args.seed),
                "n_boot": int(args.boot),
            })
        grp_df = pd.DataFrame(grp_rows).sort_values(["modality", "group_code"]).reset_index(drop=True)
        grp_df.to_csv(outdir / "hsi_ec_ratio_group_summary.csv", index=False)
        print(f"[OK] Wrote: {outdir / 'hsi_ec_ratio_group_summary.csv'}")

        # 3) contrasts (B-A and D-C) per modality (using codes)
        contrast_rows: List[Dict[str, Any]] = []
        for modality, subm in plate_df.groupby("modality"):
            def get_vals(code: str) -> np.ndarray:
                return subm.loc[subm["group_code"] == code, "ec_ratio"].to_numpy(dtype=float)

            xA, xB = get_vals("A"), get_vals("B")
            xC, xD = get_vals("C"), get_vals("D")

            md_ba, lo_ba, hi_ba = bootstrap_mean_diff_ci(xA, xB, nboot=int(args.boot), seed=int(args.seed))
            md_dc, lo_dc, hi_dc = bootstrap_mean_diff_ci(xC, xD, nboot=int(args.boot), seed=int(args.seed))

            contrast_rows += [
                {
                    "modality": modality,
                    "contrast": "B_minus_A",
                    "label": "EtOH (UV70-UV0)",
                    "mean_diff": md_ba,
                    "ci_low": lo_ba,
                    "ci_high": hi_ba,
                    "n_g1": int(np.sum(~np.isnan(xA))),
                    "n_g2": int(np.sum(~np.isnan(xB))),
                    "seed": int(args.seed),
                    "n_boot": int(args.boot),
                },
                {
                    "modality": modality,
                    "contrast": "D_minus_C",
                    "label": "PhSOFA (UV70-UV0)",
                    "mean_diff": md_dc,
                    "ci_low": lo_dc,
                    "ci_high": hi_dc,
                    "n_g1": int(np.sum(~np.isnan(xC))),
                    "n_g2": int(np.sum(~np.isnan(xD))),
                    "seed": int(args.seed),
                    "n_boot": int(args.boot),
                },
            ]

        contrast_df = pd.DataFrame(contrast_rows).sort_values(["modality", "contrast"]).reset_index(drop=True)
        contrast_df.to_csv(outdir / "hsi_contrasts.csv", index=False)
        print(f"[OK] Wrote: {outdir / 'hsi_contrasts.csv'}")

    finally:
        if zip_reader is not None:
            zip_reader.close()


if __name__ == "__main__":
    main()
