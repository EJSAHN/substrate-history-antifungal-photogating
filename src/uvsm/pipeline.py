"""Analysis-only command line; all public results originate in supplied inputs."""
from __future__ import annotations
import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time
import traceback
from . import analysis as a
from .inputs import InputFiles


def run(input_dir: Path, outdir: Path, baseline_dir: Path | None = None,
        workbook: bool = True) -> dict:
    out=Path(outdir).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Output must be new or empty: '+str(out))
    source_root=Path(input_dir).resolve()
    if out==source_root or source_root.is_relative_to(out) or out.is_relative_to(source_root):
        raise ValueError('Keep input and output directories separate')
    out.mkdir(parents=True,exist_ok=True)
    a.OUT=out;a.INPUT_DIR=source_root
    a.REFERENCE_DIR=Path(baseline_dir).resolve() if baseline_dir else None
    a.AUDIT.clear();a.ISSUES.clear()
    started=time.monotonic()
    summary={'package_version':a.VERSION,'numerical_baseline':a.NUMERICAL_BASELINE,
             'status':'STARTED','source_files_changed':False,'new_experiments':False,
             'historical_comparisons_requested':bool(baseline_dir),'figures_generated':False}
    src=None
    try:
        packages={}
        for name in ['numpy','pandas','scipy','statsmodels','scikit-learn','openpyxl']:
            try:packages[name]=importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:packages[name]=None
        a.dump_json(out/'environment.json',{'python':platform.python_version(),'platform':platform.system(),'packages':packages})
        src=InputFiles(source_root,out)
        a.dump_json(out/'input_sha256.json',src.manifest)
        a.log('Read phenotype and validate the 48-cell study design')
        d=a.load_pheno(src);arr,circ=a.reconcile_morph(d)
        a.log('Factorial HC3, matched vehicle inhibition and specified output contrasts')
        main,delt,intr,res=a.morphology(d,arr,circ)
        a.log('PCA and archived JMP summary (not recomputed)')
        a.pca_analysis(d);a.log_model_diagnostics(d)
        a.log('Supplied fluorescence pixels -> NFI -> radial ROI ratios')
        pix,waves,per,spec=a.hsi_inputs(src)
        a.log('Observed differences; historical and exact maximum-statistic tests')
        a.hsi_contrasts(pix,waves)
        a.log('Radial-threshold sensitivity with pointwise intervals')
        a.hsi_sensitivity(pix,waves,src)
        a.table('baseline_checks.csv',a.AUDIT,'audit')
        a.table('unresolved_author_checks.csv',a.ISSUES,'audit')
        failed=[r for r in a.AUDIT if not r['pass'] and r['critical']]
        if failed:raise ArithmeticError('Critical baseline comparison failed; see audit/baseline_checks.csv')
        # Expose the pixel-map data underlying illustrative C1/D1 maps.
        maps=[]
        for pid in ['C1','D1']:
            p=pix[pid]; j0=int(a.np.argmin(abs(waves-518.8)));j1=int(a.np.argmin(abs(waves-717.9)))
            for i,(row,col) in enumerate(p['xy']):
                maps.append({'Export ID':pid,'Row':int(row),'Column':int(col),
                  'Normalised radial distance':p['radial'][i],
                  'Radial zone':'Central' if p['core_mask'][i] else ('Peripheral' if p['edge_mask'][i] else 'Middle'),
                  'NFI 518.8 nm':p['nfi'][i,j0],'NFI 717.9 nm':p['nfi'][i,j1]})
        a.table('NFI_Map_Source.csv',maps)
        if workbook:
            from .workbook import build_workbook
            a.log('Export current analysis tables to Supplementary Data 1')
            build_workbook(out,out/'Supplementary_Data_1_R1.xlsx')
        src.verify_unchanged()
        summary.update(status=('COMPLETED_WITH_BASELINE_WARNINGS' if any(not r['pass'] for r in a.AUDIT)
                                else 'COMPLETED_WITH_PROVENANCE_NOTES'),
            baseline_checks=len(a.AUDIT),baseline_checks_passed=sum(r['pass'] for r in a.AUDIT),
            input_hash_manifest_verified=src.manifest_verified,source_files_changed=False)
    except Exception as exc:
        summary.update(status='FAILED',error=type(exc).__name__+': '+str(exc))
        (out/'error.txt').write_text(traceback.format_exc(),encoding='utf8')
        raise
    finally:
        a.table('baseline_checks.csv',a.pd.DataFrame(a.AUDIT,columns=['check','pass','critical','max_absolute_difference','tolerance']),'audit')
        a.table('unresolved_author_checks.csv',a.ISSUES,'audit')
        summary['elapsed_seconds']=round(time.monotonic()-started,3)
        a.dump_json(out/'summary.json',summary)
        # Finalize the log before hashing it; do not modify outputs after the manifest.
        if summary['status'].startswith('COMPLETED'):
            a.log('Completed: '+summary['status'])
        a.dump_json(out/'output_sha256.json',{p.relative_to(out).as_posix():a.digest(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='output_sha256.json'})
    return summary


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input-dir',type=Path,default=Path('input'))
    ap.add_argument('--outdir',type=Path,default=Path('out/run_r1'))
    ap.add_argument('--baseline-dir',type=Path,default=None,help='Optional private historical tables; not distributed in the public repository')
    ap.add_argument('--no-workbook',action='store_true')
    args=ap.parse_args(argv)
    try:run(args.input_dir,args.outdir,args.baseline_dir,not args.no_workbook)
    except Exception as exc:
        print('[ERROR] '+str(exc),file=sys.stderr);return 2
    return 0

if __name__=='__main__':sys.exit(main())
