#!/usr/bin/env python3
"""Export the results of a current R1 analysis, not a directory of arbitrary CSVs."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from uvsm.workbook import build_workbook
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--results-dir',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
build_workbook(a.results_dir,a.output)
print('[OK] '+str(a.output))
