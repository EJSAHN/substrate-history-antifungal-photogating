from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .excel import SheetSpec, build_workbook


@dataclass(frozen=True)
class ExportConfig:
    input_dir: Path
    output_file: Path
    reference_workbook: Optional[Path] = None


DEFAULT_INPUTS: List[Tuple[str, str]] = [
    ("uv_delta_summary.csv", "DeltaArea_70_0"),
    ("photogating_summary_by_strain_chemical_dose.csv", "AreaDoseSummary"),
    ("photogating_pairwise_tests_vs_control.csv", "Pairwise_vs_Control"),
    ("si_circularity_delta_table.xlsx", "Circularity_Delta"),
    ("si_levene_circularity.xlsx", "Levene_Circularity"),
    ("hsi_ec_ratio_group_summary.csv", "HSI_EC_GroupSummary"),
    ("hsi_ec_metrics_per_plate.csv", "HSI_EC_PerPlate"),
    ("hsi_contrasts.csv", "HSI_Contrasts"),
]


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def _sanitize_df(df: pd.DataFrame) -> pd.DataFrame:
    # Remove directory-like columns or path-like values.
    df = df.copy()
    for col in list(df.columns):
        if col.lower() in {"path", "file", "filename", "input_file", "input_path"}:
            df[col] = df[col].astype(str).map(lambda x: Path(x).name)
    # Also strip full paths embedded in any string cells.
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).map(lambda x: Path(x).name if ("/" in x or "\\" in x) else x)
    return df


def _build_manifest(rows: List[Dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["sheet", "source", "n_rows", "n_cols"])


def export_supplementary_workbook(
    input_dir: str | Path,
    output_file: str | Path,
    reference_workbook: str | Path | None = None,
    extra_inputs: Optional[List[Tuple[str, str]]] = None,
) -> Path:
    """Compile inputs into a single Excel workbook.

    Parameters
    ----------
    input_dir:
        Directory containing analysis outputs.
    output_file:
        Path of the workbook to write.
    reference_workbook:
        Optional workbook to compare against (not written into the output).
    extra_inputs:
        Optional additional (filename, sheet_name) pairs.
    """
    cfg = ExportConfig(
        input_dir=Path(input_dir),
        output_file=Path(output_file),
        reference_workbook=Path(reference_workbook) if reference_workbook else None,
    )

    inputs = list(DEFAULT_INPUTS)
    if extra_inputs:
        inputs.extend(extra_inputs)

    sheet_specs: List[SheetSpec] = []
    manifest_rows: List[Dict[str, object]] = []

    for filename, sheet_name in inputs:
        fpath = cfg.input_dir / filename
        if not fpath.exists():
            continue
        df = _read_table(fpath)
        df = _sanitize_df(df)
        sheet_specs.append(SheetSpec(name=sheet_name, df=df, source_label=filename))
        manifest_rows.append(
            {"sheet": sheet_name, "source": filename, "n_rows": int(df.shape[0]), "n_cols": int(df.shape[1])}
        )

    manifest = _build_manifest(manifest_rows)
    sheet_specs.insert(0, SheetSpec(name="Manifest", df=manifest, source_label="generated"))

    wb = build_workbook(sheet_specs)
    cfg.output_file.parent.mkdir(parents=True, exist_ok=True)
    wb.save(cfg.output_file)

    return cfg.output_file
