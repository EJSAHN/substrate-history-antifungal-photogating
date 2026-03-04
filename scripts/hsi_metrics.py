#!/usr/bin/env python3
"""
photogating_hsi_metrics_v2.py

Purpose
- Read HSI pixelSpectra exports directly from either:
  (a) an input directory (recommended for local work), or
  (b) a ZIP archive
- Compute plate-level Edge/Core metrics (E/C) for selected modalities:
  - RF fluorescence:   RF/F_roi_spectra/*_F_pixelSpectra.csv   (default)
  - RF reflectance:    RF/R_roi_spectra/*_R_pixelSpectra.csv
  - RF reflectance 75: RF/R_0075_roi_spectra/*_R_0075_pixelSpectra.csv
  - SWIR reflectance:  swir/swir_roi_spectra/*_pixelSpectra.csv
- Handle saturated samples (e.g., all 1.0 across wavelengths) by flagging and excluding from contrasts.

Outputs
- per-plate table CSV
- group summary CSV (mean + bootstrap 95% CI)
- contrast summary CSV (B-A, D-C) per modality
- (optional) spectrum-scan CSV for RF fluorescence (wavelength-by-wavelength contrasts)

Notes
- "Small CSV" outputs are expected: they are plate-level summaries, not raw HSI data.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import zipfile
from dataclasses import dataclass
from io import BytesIO, TextIOWrapper
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd

# Human-readable group labels (avoid internal A/B/C/D in outputs)
GROUP_LABEL = {
    'A': 'EtOH_UV0',
    'B': 'EtOH_UV70',
    'C': 'PhSOFA_UV0',
    'D': 'PhSOFA_UV70',
    'E': 'Blank_UV0',
    'F': 'Blank_UV70',
}


# ----------------------------
# Parsing & patterns
# ----------------------------

def parse_group_rep_from_filename(filename: str) -> Tuple[Optional[str], Optional[int]]:
    """
    Extract group letter and replicate number from basenames like:
      A1_F_pixelSpectra.csv
      C7_R_pixelSpectra.csv
      D3_R_0075_pixelSpectra.csv
      B9_pixelSpectra.csv  (SWIR)
    """
    base = os.path.basename(filename)
    base = re.sub(r'(_F_pixelSpectra|_R_0075_pixelSpectra|_R_pixelSpectra|_pixelSpectra)\.csv$', '', base)
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
    """
    Compute boolean masks for core and edge pixels using centroid distance.
    rowcol: (N,2) array of [row, col]
    """
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
# FS abstraction: ZIP or dir
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
        # return a text wrapper around the zip member
        b = self.zf.open(path)
        return TextIOWrapper(b, encoding="utf-8", newline="")

    def close(self) -> None:
        self.zf.close()


class DirReader(Reader):
    def __init__(self, input_dir: str):
        self.root = Path(input_dir).resolve()

    def list_paths(self) -> List[str]:
        # return POSIX-like relative paths (for consistent regex matching)
        out = []
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
# Metrics
# ----------------------------

def detect_saturated_all_ones(df: pd.DataFrame, tol: float = 1e-6) -> bool:
    """Return True if all spectral values are ~1.0 (e.g., a saturated reflectance export)."""
    spec_cols = [c for c in df.columns if c not in ("row", "col")]
    if not spec_cols:
        return False
    X = df[spec_cols].to_numpy(dtype=float)
    return bool(np.allclose(X, 1.0, atol=tol))


def compute_plate_ec(
    df: pd.DataFrame,
    roi: ROIParamsFractional,
    *,
    mode: str,
    target_wl: float,
) -> Dict[str, Any]:
    """
    mode:
      - "F_NFI": RF fluorescence, compute NFI at target_wl = I(wl) / sum(I)
      - "RAW_WL": reflectance or other, use raw value at target_wl
    Returns core_mean, edge_mean, ec_ratio, n_core, n_edge, wl_col
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
    ec = float(edge_mean / core_mean) if (core_mean not in (0.0, np.nan) and not np.isnan(core_mean)) else np.nan

    return dict(core_mean=core_mean, edge_mean=edge_mean, ec_ratio=ec, n_core=n_core, n_edge=n_edge, wl_col=wl_col)


def bootstrap_diff(x: np.ndarray, y: np.ndarray, nboot: int = 10_000, seed: int = 42) -> Tuple[float, float, float]:
    """
    Bootstrap mean difference (y - x).
    Returns: mean_diff, ci_low, ci_high
    """
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

    return float(diffs.mean()), float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))


# ----------------------------
# File discovery
# ----------------------------

MODALITY_SPECS = {
    "F": {
        "regex": r"/RF/F_roi_spectra/.+_F_pixelSpectra\.csv$",
        "mode": "F_NFI",
        "target_wl_arg": "wl",
        "default_wl": 518.8,
        "name": "RF_F",
    },
    "R": {
        "regex": r"/RF/R_roi_spectra/.+_R_pixelSpectra\.csv$",
        "mode": "RAW_WL",
        "target_wl_arg": "wl",
        "default_wl": 518.8,
        "name": "RF_R",
    },
    "R0075": {
        "regex": r"/RF/R_0075_roi_spectra/.+_R_0075_pixelSpectra\.csv$",
        "mode": "RAW_WL",
        "target_wl_arg": "wl",
        "default_wl": 518.8,
        "name": "RF_R0075",
        "saturation_check": True,
    },
    "SWIR": {
        "regex": r"/swir/swir_roi_spectra/.+_pixelSpectra\.csv$",
        "mode": "RAW_WL",
        "target_wl_arg": "swir_wl",
        "default_wl": 1450.0,
        "name": "SWIR",
    },
}


def discover_files(reader: Reader, modality_key: str, all_paths: List[str]) -> List[str]:
    spec = MODALITY_SPECS[modality_key]
    rx = re.compile(spec["regex"], flags=re.IGNORECASE)
    found = [p for p in all_paths if rx.search("/" + p) or rx.search(p)]
    # sort deterministically
    found.sort()
    return found


# ----------------------------
# Spectrum scan (optional)
# ----------------------------

def compute_ec_spectrum(df: pd.DataFrame, roi: ROIParamsFractional, *, mode: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (wavelengths, ec_ratio_per_wl) for a plate.
    mode:
      - "nfi": normalize each pixel spectrum to NFI (I/sum(I))
      - "raw": use raw values
    """
    rowcol = df[["row", "col"]].to_numpy(dtype=float)
    core_mask, edge_mask = core_edge_masks(rowcol, roi)
    if core_mask.sum() == 0 or edge_mask.sum() == 0:
        return np.array([]), np.array([])

    spec_cols = [c for c in df.columns if c not in ("row", "col")]
    X = df[spec_cols].to_numpy(dtype=float)

    if mode == "nfi":
        denom = X.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            X = np.where(denom > 0, X / denom, np.nan)

    core_mean = np.nanmean(X[core_mask, :], axis=0)
    edge_mean = np.nanmean(X[edge_mask, :], axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ec = edge_mean / core_mean
    ec[np.isinf(ec)] = np.nan

    wls = np.array([float(c) for c in spec_cols], dtype=float)
    return wls, ec


def spectrum_contrast_table(plate_ec_spectra: pd.DataFrame, g1: str, g2: str, nboot: int = 2000) -> pd.DataFrame:
    out = []
    for wl, sub in plate_ec_spectra.groupby("wl"):
        x = sub.loc[sub["group"] == g1, "ec_ratio"].to_numpy(dtype=float)
        y = sub.loc[sub["group"] == g2, "ec_ratio"].to_numpy(dtype=float)
        md, lo, hi = bootstrap_diff(x, y, nboot=nboot, seed=42)
        out.append(dict(wl=float(wl), diff=md, ci_low=lo, ci_high=hi, n1=int(np.sum(~np.isnan(x))), n2=int(np.sum(~np.isnan(y)))))
    return pd.DataFrame(out).sort_values("wl")


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--zip", dest="zip_path", help="Path to HSI ZIP (optional).")
    src.add_argument("--input-dir", dest="input_dir", help="Root folder containing RF/ and swir/ (recommended).")

    ap.add_argument("--outdir", default="hsi_out", help="Output directory.")
    ap.add_argument("--modalities", default="F", help="Comma-separated: F,R,R0075,SWIR (default: F).")

    ap.add_argument("--wl", type=float, default=518.8, help="Target wavelength (nm) for RF metrics (default: 518.8).")
    ap.add_argument("--swir-wl", type=float, default=1450.0, help="Target wavelength (nm) for SWIR metrics (default: 1450).")

    ap.add_argument("--core-frac", type=float, default=0.30, help="Core radius as fraction of max distance.")
    ap.add_argument("--edge-frac", type=float, default=0.80, help="Edge inner radius as fraction of max distance.")

    ap.add_argument("--boot", type=int, default=10_000, help="Bootstrap replicates for group CIs/contrasts.")
    ap.add_argument("--scan-spectrum", action="store_true",
                    help="Optional: scan RF fluorescence spectrum (E/C of NFI per wavelength) and write contrast tables.")

    args = ap.parse_args()

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    roi = ROIParamsFractional(core_frac=float(args.core_frac), edge_frac=float(args.edge_frac))

    # Reader setup
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

        # per-plate table
        plate_rows: List[Dict[str, Any]] = []

        for m in modalities:
            spec = MODALITY_SPECS[m]
            files = discover_files(reader, m, all_paths)
            if len(files) == 0:
                print(f"[WARN] No files found for modality={m} using regex={spec['regex']}")
                continue

            target_wl = args.wl if spec["target_wl_arg"] == "wl" else args.swir_wl
            mode = spec["mode"]
            do_sat = bool(spec.get("saturation_check", False))

            for rel_path in files:
                with reader.open_text(rel_path) as f:
                    df = pd.read_csv(f)

                group, rep = parse_group_rep_from_filename(rel_path)
                if group is None:
                    continue

                saturated = False
                if do_sat:
                    saturated = detect_saturated_all_ones(df)

                metrics = dict(core_mean=np.nan, edge_mean=np.nan, ec_ratio=np.nan, n_core=0, n_edge=0, wl_col=None)
                if not saturated:
                    metrics = compute_plate_ec(df, roi, mode=mode, target_wl=float(target_wl))

                abs_path = rel_path
                if dir_reader is not None:
                    abs_path = dir_reader.to_abs(rel_path)

                plate_rows.append({
                    "modality": spec["name"],
                    "group": group,
                    "rep": rep,
                    "plate": f"{group}{rep}",
                    "target_wl_nm": float(target_wl),
                    "wl_col_used": metrics["wl_col"],
                    "core_mean": metrics["core_mean"],
                    "edge_mean": metrics["edge_mean"],
                    "ec_ratio": metrics["ec_ratio"],
                    "n_core": metrics["n_core"],
                    "n_edge": metrics["n_edge"],
                    "saturated": saturated,
                    "path": abs_path,
                })

        plate_df = pd.DataFrame(plate_rows).sort_values(["modality", "group", "rep"]).reset_index(drop=True)
        plate_csv = outdir / "hsi_ec_metrics_per_plate.csv"
        plate_df.to_csv(plate_csv, index=False)
        print(f"[OK] Wrote per-plate table: {plate_csv}")

        # group summary (bootstrap CI of ec_ratio)
        grp_rows = []
        for (modality, group), sub in plate_df.groupby(["modality", "group"], dropna=False):
            vals = sub["ec_ratio"].to_numpy(dtype=float)
            vals = vals[~np.isnan(vals)]
            if vals.size == 0:
                continue
            md, lo, hi = bootstrap_diff(vals, vals, nboot=args.boot, seed=42)  # diff of same -> mean; hack
            # Above returns ~0; not what we want. We'll compute bootstrap CI of mean directly:
            rng = np.random.default_rng(42)
            means = np.empty(args.boot, dtype=float)
            for i in range(args.boot):
                means[i] = rng.choice(vals, size=vals.size, replace=True).mean()
            grp_rows.append({
                "modality": modality,
                "group": group,
                "n": int(vals.size),
                "mean_ec_ratio": float(vals.mean()),
                "ci_low": float(np.quantile(means, 0.025)),
                "ci_high": float(np.quantile(means, 0.975)),
            })

        grp_df = pd.DataFrame(grp_rows).sort_values(["modality", "group"]).reset_index(drop=True)
        grp_csv = outdir / "hsi_ec_ratio_group_summary.csv"
        grp_df.to_csv(grp_csv, index=False)
        print(f"[OK] Wrote group summary: {grp_csv}")

        # contrasts B-A and D-C per modality
        contrast_rows = []
        for modality, subm in plate_df.groupby("modality"):
            for label, g1, g2 in [("EtOH (UV70-UV0)", "EtOH_UV0", "EtOH_UV70"), ("PhSOFA (UV70-UV0)", "PhSOFA_UV0", "PhSOFA_UV70")]:
                x = subm.loc[subm["group"] == g1, "ec_ratio"].to_numpy(dtype=float)
                y = subm.loc[subm["group"] == g2, "ec_ratio"].to_numpy(dtype=float)
                md, lo, hi = bootstrap_diff(x, y, nboot=args.boot, seed=42)
                contrast_rows.append({
                    "modality": modality,
                    "contrast": label,
                    "mean_diff": md,
                    "ci_low": lo,
                    "ci_high": hi,
                    "n_g1": int(np.sum(~np.isnan(x))),
                    "n_g2": int(np.sum(~np.isnan(y))),
                })
                print(f"[Contrast] {modality} {label}: mean_diff={md:.6f}, 95%CI=[{lo:.6f}, {hi:.6f}] (n={int(np.sum(~np.isnan(x)))},{int(np.sum(~np.isnan(y)))})")

        contrast_df = pd.DataFrame(contrast_rows).sort_values(["modality", "contrast"]).reset_index(drop=True)
        contrast_csv = outdir / "hsi_contrasts.csv"
        contrast_df.to_csv(contrast_csv, index=False)
        print(f"[OK] Wrote contrasts: {contrast_csv}")

        # Optional: spectrum scan for RF fluorescence
        if args.scan_spectrum:
            # We scan only fluorescence F_roi_spectra, NFI mode.
            # Find the underlying fluorescence files again.
            files = discover_files(reader, "F", all_paths)
            if len(files) == 0:
                print("[WARN] scan-spectrum: no RF fluorescence files found; skipping.")
            else:
                spec = MODALITY_SPECS["F"]
                spec_rows = []
                for rel_path in files:
                    with reader.open_text(rel_path) as f:
                        df = pd.read_csv(f)
                    group, rep = parse_group_rep_from_filename(rel_path)
                    if group is None:
                        continue
                    wls, ec = compute_ec_spectrum(df, roi, mode="nfi")
                    if wls.size == 0:
                        continue
                    spec_rows.append(pd.DataFrame({
                        "group": group,
                        "rep": rep,
                        "plate": f"{group}{rep}",
                        "wl": wls,
                        "ec_ratio": ec,
                    }))
                spec_df = pd.concat(spec_rows, ignore_index=True)
                out_spec_plate = outdir / "rf_fluorescence_ec_spectrum_per_plate.csv"
                spec_df.to_csv(out_spec_plate, index=False)
                print(f"[OK] Wrote spectrum per-plate table: {out_spec_plate}")

                # contrasts across wavelengths
                c_etoh = spectrum_contrast_table(spec_df, "A", "B", nboot=min(2000, args.boot))
                c_phsofa = spectrum_contrast_table(spec_df, "C", "D", nboot=min(2000, args.boot))
                c_etoh.to_csv(outdir / "rf_fluorescence_ec_spectrum_contrast_B_minus_A.csv", index=False)
                c_phsofa.to_csv(outdir / "rf_fluorescence_ec_spectrum_contrast_D_minus_C.csv", index=False)
                print("[OK] Wrote spectrum contrast tables for fluorescence.")

    finally:
        if zip_reader is not None:
            zip_reader.close()


if __name__ == "__main__":
    main()
