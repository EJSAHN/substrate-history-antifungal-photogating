#!/usr/bin/env python
"""
Phenotype analysis pipeline for UV substrate memory (UVSM).

Reads a long-format phenotype table and emits analysis tables used for Supplementary Data 1.

Design assumptions (kept simple & reproducible):
- Input table contains one row per plate (replicate).
- Required columns (case-insensitive match):
  strain, chemical, uv_dose_mj_cm2, area_mm2, perimeter_mm, length_mm, width_mm,
  lwr, circularity, is_cg_mm
- UV doses may include {0,12,35,70} but code works with any set (needs 0 and 70 for deltas).

Outputs (CSV):
- delta_area_summary.csv
- delta_area_map_long.csv
- dose_response_summary.csv
- circularity_delta_summary.csv
- levene_circularity_pvalues.csv
- morphometrics_pca_scores.csv
- variable_clustering_summary.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd

def _colmap(df: pd.DataFrame) -> dict[str, str]:
    """Map expected semantic names to actual columns (case-insensitive)."""
    want = {
        "strain": ["strain", "isolate"],
        "chemical": ["chemical", "compound", "treatment"],
        "uv": ["uv", "uv_dose", "uv_dose_mj_cm2", "uvdose", "uv_dose_mj/cm2", "uv dose (mj cm-2)", "uv exposure (mj/cm2)", "uv exposure (mj cm-2)"],
        "area": ["area_mm2", "area", "area size(mm2)", "area size(AS)[mm2]", "areasize", "area size"],
        "perimeter": ["perimeter_mm", "perimeter", "perimeter length(PL)[mm]", "perimeter length"],
        "length": ["length_mm", "length", "length(L)[mm]"],
        "width": ["width_mm", "width", "width(W)[mm]"],
        "lwr": ["lwr", "length-to-width ratio(lwr)", "length-to-width ratio(LWR)"],
        "circularity": ["circularity", "circularity(cs)", "circularity(CS)"],
        "is_cg": ["is_cg_mm", "distance between is and cg (ds)[mm]", "is and cg", "ds", "distance_is_cg_mm"],
    }
    lower = {c.lower(): c for c in df.columns}
    out = {}
    for key, aliases in want.items():
        found = None
        for a in aliases:
            if a.lower() in lower:
                found = lower[a.lower()]
                break
        if found is None:
            raise KeyError(f"Missing required column for '{key}'. Tried aliases: {aliases}. "
                           f"Available columns: {list(df.columns)}")
        out[key] = found
    return out

def _bootstrap_ci(x: np.ndarray, n_boot: int = 5000, ci: float = 0.95, seed: int = 1337) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return (np.nan, np.nan)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    means = x[idx].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    lo = np.quantile(means, alpha)
    hi = np.quantile(means, 1.0 - alpha)
    return float(lo), float(hi)

def _group_bootstrap_summary(df: pd.DataFrame, value_col: str, group_cols: list[str], n_boot: int = 5000) -> pd.DataFrame:
    rows = []
    for keys, g in df.groupby(group_cols, dropna=False):
        x = g[value_col].to_numpy(dtype=float)
        mu = float(np.mean(x))
        lo, hi = _bootstrap_ci(x, n_boot=n_boot)
        if not isinstance(keys, tuple):
            keys = (keys,)
        rows.append((*keys, mu, lo, hi, len(x)))
    out = pd.DataFrame(rows, columns=[*group_cols, "mean", "ci_lo", "ci_hi", "n"])
    return out

def _delta_70_0(df: pd.DataFrame, value_col: str, strain_col: str, chem_col: str, uv_col: str, n_boot: int = 5000) -> pd.DataFrame:
    """Compute Δ(70-0) with bootstrap CI within each strain×chemical."""
    rows = []
    for (strain, chem), g in df.groupby([strain_col, chem_col], dropna=False):
        g0 = g.loc[g[uv_col] == 0, value_col].to_numpy(dtype=float)
        g70 = g.loc[g[uv_col] == 70, value_col].to_numpy(dtype=float)
        if g0.size == 0 or g70.size == 0:
            continue
        delta = float(np.mean(g70) - np.mean(g0))
        # bootstrap delta (paired only by resampling within groups, independent)
        rng = np.random.default_rng(1337)
        idx0 = rng.integers(0, g0.size, size=(n_boot, g0.size))
        idx70 = rng.integers(0, g70.size, size=(n_boot, g70.size))
        deltas = g70[idx70].mean(axis=1) - g0[idx0].mean(axis=1)
        lo = float(np.quantile(deltas, 0.025))
        hi = float(np.quantile(deltas, 0.975))
        sig = (lo > 0) or (hi < 0)
        rows.append((strain, chem, delta, lo, hi, sig, int(g0.size), int(g70.size)))
    return pd.DataFrame(rows, columns=[strain_col, chem_col, "delta_70_0", "ci_lo", "ci_hi", "significant", "n0", "n70"])

def _pca_scores(df: pd.DataFrame, cols: list[str], n_components: int = 2) -> tuple[pd.DataFrame, dict]:
    # local import to keep base requirements small
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    X = df[cols].to_numpy(dtype=float)
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=n_components, random_state=1337)
    scores = pca.fit_transform(Xs)
    meta = {
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "components": pca.components_.tolist(),
        "feature_names": cols,
    }
    out = pd.DataFrame(scores, columns=[f"PC{i+1}" for i in range(n_components)], index=df.index)
    return out, meta

def _varclus_summary(df: pd.DataFrame, vars_: list[str], n_clusters: int = 2) -> pd.DataFrame:
    """
    Simple variable clustering (approx JMP/SAS varclus):
    1) cluster variables by |corr| using agglomerative clustering
    2) within each cluster, compute 1st PC of standardized variables (cluster component)
    3) for each variable, compute R^2 with own cluster component and next-closest component
    """
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    X = df[vars_].to_numpy(dtype=float)
    Xs = StandardScaler().fit_transform(X)
    corr = np.corrcoef(Xs, rowvar=False)
    dist = 1.0 - np.abs(corr)
    model = AgglomerativeClustering(n_clusters=n_clusters, metric="precomputed", linkage="average")
    labels = model.fit_predict(dist)

    # cluster components
    comps = []
    for k in range(n_clusters):
        idx = np.where(labels == k)[0]
        pca = PCA(n_components=1, random_state=1337)
        comp = pca.fit_transform(Xs[:, idx]).ravel()
        comps.append(comp)
    comps = np.vstack(comps).T  # n_samples x n_clusters

    rows = []
    for j, v in enumerate(vars_):
        own = labels[j]
        r2_own = float(np.corrcoef(Xs[:, j], comps[:, own])[0, 1] ** 2)
        r2_next = 0.0
        for k in range(n_clusters):
            if k == own:
                continue
            r2 = float(np.corrcoef(Xs[:, j], comps[:, k])[0, 1] ** 2)
            r2_next = max(r2_next, r2)
        ratio = float((1.0 - r2_own) / (1.0 - r2_next + 1e-12))
        rows.append((int(own + 1), v, r2_own, r2_next, ratio))
    out = pd.DataFrame(rows, columns=["cluster", "variable", "r2_own_cluster", "r2_next_closest", "one_minus_r2_ratio"])
    # sort like JMP-ish
    out = out.sort_values(["cluster", "one_minus_r2_ratio", "variable"]).reset_index(drop=True)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phenotype", required=True, help="Path to phenotype.xlsx or phenotype.csv")
    ap.add_argument("--sheet", default="phenotype", help="Excel sheet name (if .xlsx)")
    ap.add_argument("--outdir", required=True, help="Output directory for analysis tables")
    ap.add_argument("--nboot", type=int, default=5000, help="Bootstrap resamples (default 5000)")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    p = Path(args.phenotype)
    if p.suffix.lower() in [".xlsx", ".xlsm", ".xls"]:
        df = pd.read_excel(p, sheet_name=args.sheet)
    else:
        df = pd.read_csv(p)

    cm = _colmap(df)
    # normalize key columns
    df = df.copy()
    df.rename(columns={
        cm["strain"]: "Strain",
        cm["chemical"]: "Chemical",
        cm["uv"]: "UV_Dose_mJ_cm2",
        cm["area"]: "Area_mm2",
        cm["perimeter"]: "Perimeter_mm",
        cm["length"]: "Length_mm",
        cm["width"]: "Width_mm",
        cm["lwr"]: "LWR",
        cm["circularity"]: "Circularity",
        cm["is_cg"]: "IS_CG_mm",
    }, inplace=True)

    # enforce numeric UV
    df["UV_Dose_mJ_cm2"] = pd.to_numeric(df["UV_Dose_mJ_cm2"], errors="coerce")

    # -----------------
    # Dose-response mean+CI for area
    # -----------------
    dose_resp = _group_bootstrap_summary(
        df, value_col="Area_mm2",
        group_cols=["Strain", "Chemical", "UV_Dose_mJ_cm2"],
        n_boot=args.nboot
    ).rename(columns={"mean": "Mean_Area_mm2", "ci_lo": "Lower_95_CI", "ci_hi": "Upper_95_CI", "n": "n"})
    dose_resp.to_csv(outdir / "dose_response_summary.csv", index=False)

    # -----------------
    # ΔArea70-0 summary + long map (for Fig1)
    # -----------------
    delta_area = _delta_70_0(df, "Area_mm2", "Strain", "Chemical", "UV_Dose_mJ_cm2", n_boot=args.nboot)
    delta_area.to_csv(outdir / "delta_area_summary.csv", index=False)

    # long map with labels matching manuscript language
    delta_area_map = delta_area.copy()
    delta_area_map.rename(columns={
        "delta_70_0": "DeltaArea_70_minus_0",
        "ci_lo": "Lower_95_CI",
        "ci_hi": "Upper_95_CI",
        "significant": "Significant"
    }, inplace=True)
    delta_area_map.to_csv(outdir / "delta_area_map_long.csv", index=False)

    # -----------------
    # Circularity Δ(70-0)
    # -----------------
    circ_delta = _delta_70_0(df, "Circularity", "Strain", "Chemical", "UV_Dose_mJ_cm2", n_boot=args.nboot)
    circ_delta.rename(columns={"delta_70_0": "DeltaCircularity_70_minus_0", "ci_lo": "Lower_95_CI", "ci_hi": "Upper_95_CI"}, inplace=True)
    circ_delta.to_csv(outdir / "circularity_delta_summary.csv", index=False)

    # -----------------
    # Levene (circularity variance across doses)
    # -----------------
    from scipy.stats import levene
    rows = []
    for (strain, chem), g in df.groupby(["Strain", "Chemical"], dropna=False):
        groups = []
        for dose, gg in g.groupby("UV_Dose_mJ_cm2"):
            x = gg["Circularity"].dropna().to_numpy(dtype=float)
            if x.size > 0:
                groups.append(x)
        pval = np.nan
        if len(groups) >= 2:
            try:
                stat, pval = levene(*groups, center="median")
            except Exception:
                pval = np.nan
        rows.append((strain, chem, pval))
    lev = pd.DataFrame(rows, columns=["Strain", "Chemical", "Levene_pvalue"])
    lev.to_csv(outdir / "levene_circularity_pvalues.csv", index=False)

    # -----------------
    # PCA scores (7 morphometrics)
    # -----------------
    morph_cols = ["Perimeter_mm", "Area_mm2", "Length_mm", "Width_mm", "Circularity", "LWR", "IS_CG_mm"]
    scores, meta = _pca_scores(df, morph_cols, n_components=2)
    pca_out = pd.concat([df[["Strain", "Chemical", "UV_Dose_mJ_cm2"]].reset_index(drop=True), scores.reset_index(drop=True)], axis=1)
    pca_out.to_csv(outdir / "morphometrics_pca_scores.csv", index=False)

    # -----------------
    # Variable clustering summary (approx)
    # -----------------
    # include UV dose as a variable to show "external driver" behavior
    vdf = df[morph_cols + ["UV_Dose_mJ_cm2"]].copy()
    vdf.rename(columns={"UV_Dose_mJ_cm2": "UV_Dose"}, inplace=True)
    vars_ = ["Perimeter_mm", "Area_mm2", "Length_mm", "Width_mm", "Circularity", "LWR", "IS_CG_mm", "UV_Dose"]
    varclus = _varclus_summary(vdf, vars_, n_clusters=2)
    varclus.to_csv(outdir / "variable_clustering_summary.csv", index=False)

    print(f"[OK] Wrote phenotype tables to: {outdir}")

if __name__ == "__main__":
    main()
