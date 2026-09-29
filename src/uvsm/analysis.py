#!/usr/bin/env python3
"""Numerical routines for UVSM morphology and fluorescence analyses."""
from __future__ import annotations
import argparse, collections, csv, datetime as dt, hashlib, itertools, json
import math, os, platform, re, sys, time, traceback, zipfile
from pathlib import Path
import xml.etree.ElementTree as ET

VERSION = '1.2.1'
NUMERICAL_BASELINE = '1.0.0'
REFERENCE_DIR = INPUT_DIR = None
ISOLATES = ['P24-192', 'CGH17', 'CGH5', 'CGH49']
CHEMICALS = ['Control', 'PhSOAM', 'PhSOFA']
UVS = [0, 12, 35, 70]
MORPH = ['Perimeter_mm', 'Area_mm2', 'Length_mm', 'Width_mm', 'Circularity', 'LWR', 'IS_CG_mm']
PHENO_SHA = 'fa1cc1b1f18e1f0d153ecfb840bee1b3279ba045045804d5074239c2aa20a00a'
NB_MORPH, SEED_MORPH, NB_HSI, SEED_HSI = 5000, 1337, 10000, 42
GROUP_LABEL = {'A':'EtOH UV0','B':'EtOH UV70','C':'PhSOFA UV0',
               'D':'PhSOFA UV70','E':'Blank E (legacy UV0 label)',
               'F':'Blank F (legacy UV70 label)'}
NS = {'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
OUT = None
START = time.monotonic()
AUDIT, ISSUES = [], []
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

def log(text):
    msg = f'[{time.strftime("%H:%M:%S")}] {text}'
    print(msg, flush=True)
    if OUT:
        with (OUT/'run.log').open('a',encoding='utf-8') as f: f.write(msg+'\n')

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def dump_json(path, obj):
    def conv(o):
        if isinstance(o,Path): return str(o)
        if hasattr(o,'item'): return o.item()
        if hasattr(o,'tolist'): return o.tolist()
        raise TypeError(type(o).__name__)
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=conv),encoding='utf-8')

def table(name, data, directory='tables'):
    d=OUT/directory; d.mkdir(parents=True,exist_ok=True)
    df=data if isinstance(data,pd.DataFrame) else pd.DataFrame(data)
    df.to_csv(d/name,index=False,encoding='utf-8-sig',float_format='%.15g')
    return df

def issue(code, detail):
    if not any(r['code']==code for r in ISSUES): ISSUES.append({'code':code,'detail':detail})

def check(label, actual, expected, tol=1e-9, critical=True):
    a=np.asarray(actual,dtype=float); e=np.asarray(expected,dtype=float)
    same=a.shape==e.shape
    err=float(np.max(np.abs(a-e))) if same and a.size else (0. if same else float('inf'))
    ok=bool(same and np.all(np.isfinite(a)) and np.all(np.isfinite(e)) and err<=tol)
    AUDIT.append({'check':label,'pass':ok,'critical':critical,'max_absolute_difference':err,'tolerance':tol})
    return ok

def read_xlsx_values(path, sheet_name='phenotype'):
    """Value-only OOXML reader for this source, with explicit cached-formula guard.
    Does not modify a workbook or infer formats/dates. No spreadsheet package needed.
    """
    with zipfile.ZipFile(path) as z:
        strings=[]
        if 'xl/sharedStrings.xml' in z.namelist():
            root=ET.fromstring(z.read('xl/sharedStrings.xml'))
            strings=[''.join(n.itertext()) for n in root.findall('s:si',NS)]
        w=ET.fromstring(z.read('xl/workbook.xml'))
        rel=ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))
        rels={r.attrib['Id']:r.attrib['Target'] for r in rel}
        node=next((r for r in w.findall('s:sheets/s:sheet',NS) if r.attrib['name']==sheet_name),None)
        if node is None: raise ValueError(f'Missing source worksheet: {sheet_name}')
        rid=node.attrib['{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id']
        target=rels[rid]
        member=target.lstrip('/') if target.startswith('/') else 'xl/'+target
        root=ET.fromstring(z.read(member))
        rows=[]
        for row in root.findall('s:sheetData/s:row',NS):
            cells={}
            for c in row.findall('s:c',NS):
                addr=c.attrib['r']; letters=re.match(r'[A-Z]+',addr).group(); ci=0
                for ch in letters: ci=ci*26+ord(ch)-64
                typ=c.attrib.get('t','n'); v=c.find('s:v',NS)
                if c.find('s:f',NS) is not None and v is None:
                    raise ValueError(f'Formula without cached value: {addr}')
                if typ=='s': val=strings[int(v.text)] if v is not None else ''
                elif typ=='inlineStr':
                    elem=c.find('s:is',NS); val=''.join(elem.itertext()) if elem is not None else ''
                elif typ=='b': val=bool(int(v.text)) if v is not None else None
                elif typ in ('str','e'): val=v.text if v is not None else None
                else:
                    val=float(v.text) if v is not None and v.text is not None else None
                cells[ci-1]=val
            if cells: rows.append([cells.get(i) for i in range(max(cells)+1)])
        width=max(map(len,rows))
        return [r+[None]*(width-len(r)) for r in rows]

def load_reference(name):
    if REFERENCE_DIR is None:
        raise FileNotFoundError("No baseline directory was supplied")
    return pd.read_csv(REFERENCE_DIR / name)

def load_pheno(src):
    raw=read_xlsx_values(src.phenotype) if src.phenotype.suffix.lower()=='.xlsx' else list(csv.reader(src.phenotype.open(encoding='utf-8-sig',newline='')))
    d=pd.DataFrame(raw[1:],columns=raw[0])
    rename={'Strain':'Isolate','Chemical':'Chemical','UV exposure (mJ/cm2)':'UV',
            'Area size':'Area_mm2','Perimeter':'Perimeter_mm','Length':'Length_mm',
            'Width':'Width_mm','LWR':'LWR','Circularity':'Circularity','IS and CG':'IS_CG_mm'}
    d=d.rename(columns=rename)
    d.insert(0,'source_excel_row',np.arange(2,len(d)+2))
    d.insert(1,'row_identifier',[f'phenotype_row_{i:04d}' for i in d.source_excel_row])
    for col in ['UV']+MORPH:
        d[col]=pd.to_numeric(d[col],errors='raise')
        if not np.isfinite(d[col].to_numpy()).all(): raise ValueError(f'Nonfinite phenotype: {col}; no automatic dropping')
    if not (d.Area_mm2>0).all(): raise ValueError('Nonpositive colony area')
    if not np.equal(d.UV, np.floor(d.UV)).all(): raise ValueError('UV levels must be integer dose labels; no truncation is allowed')
    d.UV=d.UV.astype(int)
    if set(d.Isolate)!=set(ISOLATES) or set(d.Chemical)!=set(CHEMICALS) or set(d.UV)!=set(UVS):
        raise ValueError('Unexpected factor levels')
    n=d.groupby(['Isolate','Chemical','UV']).size()
    if len(d)!=432 or len(n)!=48 or not (n==9).all(): raise ValueError('Expected study design: 432 rows, 48 cells, nine observations per cell')
    if d[MORPH+['Isolate','Chemical','UV']].duplicated().any():
        raise ValueError('Exact duplicate measurement rows; inspect before analysis')
    table('phenotype_value_copy.csv',d,'inputs_snapshot')
    table('cell_counts.csv',n.rename('n').reset_index(),'audit')
    issue('RUN_METADATA_NOT_ENCODED','The 432 morphology rows do not include experiment/run, batch, lot, biological plate ID, inoculation date or operator fields. row_identifier locates a source record. Models assume row-level independence; between-run replication cannot be assessed from these fields.')
    return d

def cell_arrays(d,col='Area_mm2'):
    return {(s,c,u):g[col].to_numpy(float) for (s,c,u),g in d.groupby(['Isolate','Chemical','UV'],sort=True)}

def boot_mean(x,b=NB_MORPH,seed=SEED_MORPH):
    rng=np.random.default_rng(seed); idx=rng.integers(0,len(x),size=(b,len(x)))
    return x[idx].mean(1)

def boot_delta_legacy(x,y,b=NB_MORPH,seed=SEED_MORPH):
    rng=np.random.default_rng(seed)
    ix=rng.integers(0,len(x),size=(b,len(x))); iy=rng.integers(0,len(y),size=(b,len(y)))
    dist=y[iy].mean(1)-x[ix].mean(1)
    return float(y.mean()-x.mean()),np.quantile(dist,[.025,.975]),dist

def boot_interleaved(x,y,b=NB_HSI,seed=SEED_HSI,rng=None,high_first=False):
    """Same draw order as original HSI helper / original figure notebook.
    Accepts n-by-features arrays; identical draws shared across all spectral bands.
    """
    rng=np.random.default_rng(seed) if rng is None else rng
    xx=np.asarray(x); yy=np.asarray(y)
    if len(xx)==len(yy):
        idx=rng.integers(0,len(xx),size=(b,2,len(xx)))
        ix,iy=(idx[:,1],idx[:,0]) if high_first else (idx[:,0],idx[:,1])
    else:
        ix=np.empty((b,len(xx)),int); iy=np.empty((b,len(yy)),int)
        for i in range(b):
            if high_first:
                iy[i]=rng.integers(0,len(yy),len(yy)); ix[i]=rng.integers(0,len(xx),len(xx))
            else:
                ix[i]=rng.integers(0,len(xx),len(xx)); iy[i]=rng.integers(0,len(yy),len(yy))
    dist=yy[iy].mean(1)-xx[ix].mean(1)
    return yy.mean(0)-xx.mean(0),np.quantile(dist,[.025,.975],axis=0),dist

def reconcile_morph(d):
    a=cell_arrays(d); circ=cell_arrays(d,'Circularity')
    if REFERENCE_DIR is None:
        issue('BASELINE_NOT_SUPPLIED','Historical reference tables were not supplied, so numerical comparisons with them were not performed.')
        return a,circ
    sub=load_reference('submitted_Morph_DoseResponse.csv')
    mean_actual=np.array([a[(r.Strain,r.Chemical,int(r.UV_Dose_mJ_cm2))].mean() for r in sub.itertuples()])
    check('48 cell means vs submitted workbook',mean_actual,sub.Mean_Area_mm2)
    ci=np.array([np.quantile(boot_mean(a[(r.Strain,r.Chemical,int(r.UV_Dose_mJ_cm2))]),[.025,.975]) for r in sub.itertuples()])
    check('5000/1337 cell CIs vs submitted workbook',ci,sub[['Lower_95_CI','Upper_95_CI']])
    delta=load_reference('submitted_Morph_DeltaArea_Summary.csv')
    vals=[]
    for r in delta.itertuples():
        v,c,_=boot_delta_legacy(a[(r.Strain,r.Chemical,0)],a[(r.Strain,r.Chemical,70)])
        vals.append([v,*c])
    check('5000/1337 delta-area vs submitted workbook',vals,delta[['delta_70_0','ci_lo','ci_hi']])
    old=load_reference('historical_figure_delta_area.csv')
    replay={}; rng=np.random.default_rng(42)
    for s in ISOLATES:
        for c in CHEMICALS:
            v,ci,_=boot_interleaved(a[(s,c,0)],a[(s,c,70)],b=10000,rng=rng,high_first=True)
            replay[(s,c)]=[float(v),*ci]
    check('10000/42 sequential-draw figure CIs vs historical figure table',
          [replay[(r.Strain,r.Chemical)] for r in old.itertuples()],old[['ΔArea(70-0)','CI_low','CI_high']])
    histdose=load_reference('historical_dose.csv')
    hc=np.array([np.quantile(boot_mean(a[(r.Strain,r.Chemical,int(r.UV))],10000,42),[.025,.975]) for r in histdose.itertuples()])
    ok_old_dose=check('10000/42 dose CIs vs historical dose table',hc,histdose[['lo','hi']],critical=False)
    auditdose=histdose.copy()
    auditdose['replay_10000_seed42_ci_low']=hc[:,0]
    auditdose['replay_10000_seed42_ci_high']=hc[:,1]
    auditdose['low_difference']=hc[:,0]-histdose.lo
    auditdose['high_difference']=hc[:,1]-histdose.hi
    table('historical_dose_CI_unresolved_comparison.csv',auditdose,'audit')
    check('48 historical Word/figure dose means vs source', [a[(r.Strain,r.Chemical,int(r.UV))].mean() for r in histdose.itertuples()],histdose['mean'])
    if not ok_old_dose:
        issue('HISTORICAL_DOSE_CI_NOT_EXACTLY_REPRODUCED','All 48 historical dose means agree with the source measurements. Some archived dose-table confidence limits differ from the 10000-resample, seed-42 replay; their original settings or ordering remain unresolved. Archived limits are retained in the comparison output. The current summaries use the reproduced 5000-resample, seed-1337 convention.')
    cr=load_reference('submitted_Morph_CircDelta_Summary.csv')
    cv=[]
    for r in cr.itertuples():
        v,ci,_=boot_delta_legacy(circ[(r.Strain,r.Chemical,0)],circ[(r.Strain,r.Chemical,70)])
        cv.append([v,*ci])
    check('5000/1337 circularity vs submitted workbook',cv,cr[['DeltaCircularity_70_minus_0','Lower_95_CI','Upper_95_CI']])
    issue('LEGACY_MORPH_CI_SETTINGS','The historical delta-area figure uses 10000 resamples, seed 42 and sequential draws; the submitted workbook uses 5000 resamples and seed 1337. The historical dose-table CI discrepancy is recorded separately. Current mean/delta summaries use 5000/1337, while matched-control quantities use the documented per-cell streams.')
    table('legacy_delta_area_replay.csv',[{'Isolate':s,'Chemical':c,'mean_difference':v[0],'ci_low':v[1],'ci_high':v[2],'n_boot':10000,'seed':42,'draw_order':'one RNG over P24-192, CGH17, CGH5, CGH49 and Control, PhSOAM, PhSOFA; high then low'} for (s,c),v in replay.items()],'audit')
    return a,circ

def stream_seed(name):
    return int.from_bytes(hashlib.sha256(('UVSM-R1-20260928|'+name).encode()).digest()[:4],'little')

def morphology(d,a,circ):
    from statsmodels.stats.multitest import multipletests
    model=smf.ols('Area_mm2 ~ C(Isolate, Sum)*C(Chemical, Sum)*C(UV, Sum)',data=d,eval_env=-1).fit()
    hc=model.get_robustcov_results(cov_type='HC3',use_t=True)
    for response,dd in [('Area_mm2',d),('log_Area_mm2',d.assign(log_Area_mm2=np.log(d.Area_mm2)))]:
        fit=model if response=='Area_mm2' else smf.ols(response+' ~ C(Isolate, Sum)*C(Chemical, Sum)*C(UV, Sum)',data=dd,eval_env=-1).fit()
        standard=sm.stats.anova_lm(fit,typ=3).reset_index(names='term')
        table(f'factorial_{response}_ordinary_type3.csv',standard)
        robust=sm.stats.anova_lm(fit,typ=3,robust='hc3').reset_index(names='term')
        robust=robust.loc[robust.term!='Residual',['term','df','F','PR(>F)']].rename(columns={'df':'df_num','PR(>F)':'p_value'})
        robust['df_denom']=fit.df_resid; robust['covariance']='HC3'; robust['coding']='sum-to-zero'
        table(f'factorial_{response}_HC3_tests.csv',robust)
    (OUT/'models').mkdir(exist_ok=True)
    (OUT/'models'/'area_model_summary.txt').write_text(model.summary().as_text()+'\n\nHC3\n'+hc.summary().as_text(),encoding='utf-8')
    from patsy import build_design_matrices
    def design(s,c,u):
        return np.asarray(build_design_matrices([model.model.data.design_info],pd.DataFrame({'Isolate':[s],'Chemical':[c],'UV':[u]}))[0]).ravel()
    def ttest(vec):
        t=hc.t_test(vec); inter=np.asarray(t.conf_int()).ravel()
        return {'model_effect':float(np.asarray(t.effect).ravel()[0]),'model_se':float(np.asarray(t.sd).ravel()[0]),
                'model_ci_low':float(inter[0]),'model_ci_high':float(inter[1]),'t_value':float(np.asarray(t.tvalue).ravel()[0]),
                'p_value':float(np.asarray(t.pvalue).ravel()[0])}
    # Shared, independent bootstrap streams per biological cell; control draws are
    # reused within derived contrasts so their shared uncertainty is preserved.
    draws={k:boot_mean(x,NB_MORPH,stream_seed('|'.join(map(str,k)))) for k,x in a.items()}
    rows=[]
    for s in ISOLATES:
        for c in CHEMICALS:
            for u in UVS:
                x=a[(s,c,u)]; ctrl=a[(s,'Control',u)]; ci=np.quantile(boot_mean(x),[.025,.975])
                if c=='Control': est=0.; ic=[0.,0.]; note='reference (0 by definition; not an uncertainty estimate)'
                else:
                    est=100*(1-x.mean()/ctrl.mean()); bs=100*(1-draws[(s,c,u)]/draws[(s,'Control',u)])
                    ic=np.quantile(bs,[.025,.975]); note='ratio of group means; independent cells; unadjusted percentile CI'
                rows.append({'Isolate':s,'Chemical':c,'UV_mJ_cm2':u,'n':len(x),'mean_area_mm2':x.mean(),
                             'area_ci_low':ci[0],'area_ci_high':ci[1],'inhibition_percent':est,
                             'inhibition_ci_low':ic[0],'inhibition_ci_high':ic[1],
                             'n_boot':NB_MORPH,'inhibition_note':note})
    main=table('Table1_48cells_matched_vehicle_inhibition.csv',rows)
    deltarows=[]; cirrows=[]; levene=[]; interaction=[]
    for s in ISOLATES:
        for c in CHEMICALS:
            x,y=a[(s,c,0)],a[(s,c,70)]; v,ci,_=boot_delta_legacy(x,y)
            deltarows.append({'Isolate':s,'Chemical':c,'delta_70_minus_0':v,'ci_low':ci[0],'ci_high':ci[1],
                              'n0':len(x),'n70':len(y),'n_boot':NB_MORPH,'seed':SEED_MORPH,
                              **ttest(design(s,c,70)-design(s,c,0))})
            v,ci,_=boot_delta_legacy(circ[(s,c,0)],circ[(s,c,70)])
            cirrows.append({'Isolate':s,'Chemical':c,'delta_circularity':v,'ci_low':ci[0],'ci_high':ci[1],'n_boot':NB_MORPH,'seed':SEED_MORPH})
            st,p=stats.levene(*[circ[(s,c,u)] for u in UVS],center='median')
            levene.append({'Isolate':s,'Chemical':c,'statistic':st,'p_value':p,'centre':'median'})
            if c!='Control':
                bctrl=draws[(s,'Control',70)]-draws[(s,'Control',0)]
                bt=draws[(s,c,70)]-draws[(s,c,0)]
                dic=np.quantile(bt-bctrl,[.025,.975])
                bi0=100*(1-draws[(s,c,0)]/draws[(s,'Control',0)])
                bi70=100*(1-draws[(s,c,70)]/draws[(s,'Control',70)])
                ic=np.quantile(bi70-bi0,[.025,.975])
                i0=100*(1-x.mean()/a[(s,'Control',0)].mean())
                i70=100*(1-y.mean()/a[(s,'Control',70)].mean())
                interaction.append({'Isolate':s,'Chemical':c,'inhibition_UV0_pct':i0,'inhibition_UV70_pct':i70,
                                    'change_inhibition_percentage_points':i70-i0,'change_ci_low':ic[0],'change_ci_high':ic[1],
                                    'area_difference_in_differences':(y.mean()-x.mean())-(a[(s,'Control',70)].mean()-a[(s,'Control',0)].mean()),
                                    'did_boot_ci_low':dic[0],'did_boot_ci_high':dic[1],
                                    **ttest(design(s,c,70)-design(s,c,0)-design(s,'Control',70)+design(s,'Control',0))})
    delt=pd.DataFrame(deltarows); delt['p_holm_12']=multipletests(delt.p_value,method='holm')[1]
    intr=pd.DataFrame(interaction); intr['p_holm_8']=multipletests(intr.p_value,method='holm')[1]
    table('Morph_delta_area_12_HC3_and_bootstrap.csv',delt)
    table('Morph_UV_by_compound_matched_control_8.csv',intr)
    table('Morph_circularity_12.csv',cirrows)
    table('Morph_Levene_circularity.csv',levene)
    table('Morph_CGH17_all_cells.csv',main[main.Isolate=='CGH17'])
    table('Morph_CGH17_UV_changes.csv',delt[delt.Isolate=='CGH17'])
    res=d[['row_identifier','Isolate','Chemical','UV']].copy()
    res['fitted_area']=model.fittedvalues; res['residual']=model.resid
    inf=model.get_influence(); res['studentized_residual']=inf.resid_studentized_internal
    res['cooks_distance']=inf.cooks_distance[0]
    table('Morph_model_residuals_all_rows.csv',res)
    issue('MODEL_SCOPE','Factorial HC3 tests, matched-control contrasts and log-area sensitivity analyses were added during revision. UV is categorical, and all source rows are retained regardless of residual size. These analyses do not establish experiment-level reproducibility or identify the photochemical mechanism.')
    return main,delt,intr,res

def pca_analysis(d):
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    x=StandardScaler().fit_transform(d[MORPH].to_numpy(float))
    fit=PCA(n_components=len(MORPH),svd_solver='full').fit(x)
    scores=fit.transform(x)
    # Deterministic sign orientation, explicitly recorded. It changes neither
    # explained variance nor distances. No rotation fitted to improve agreement.
    signs=[]
    for j in range(len(MORPH)):
        si=1 if fit.components_[j,np.argmax(np.abs(fit.components_[j]))]>=0 else -1
        scores[:,j]*=si; signs.append(si)
    sc=d[['row_identifier','Isolate','Chemical','UV']].copy()
    for j in range(len(MORPH)): sc[f'PC{j+1}']=scores[:,j]
    table('Morph_PCA_scores_Python.csv',sc)
    table('Morph_PCA_explained_variance.csv',[{'PC':f'PC{j+1}','variance_fraction':v,'sign_multiplier':signs[j]} for j,v in enumerate(fit.explained_variance_ratio_)])
    loads=[]
    for k,m in enumerate(MORPH):
        loads.append({'feature':m,**{f'PC{j+1}_correlation':float(np.corrcoef(x[:,k],scores[:,j])[0,1]) for j in range(len(MORPH))}})
    ld=table('Morph_PCA_variable_correlations.csv',loads)
    comp=[]
    if REFERENCE_DIR is not None:
        # Submitted scores are compared only if their row-wise condition labels agree.
        prev=load_reference('submitted_Morph_PCA_Scores.csv')
        aligned=prev.shape[0]==len(d)
        if aligned:
            aligned=bool((prev.Strain.to_numpy()==d.Isolate.to_numpy()).all() and (prev.Chemical.to_numpy()==d.Chemical.to_numpy()).all() and (prev.UV_Dose_mJ_cm2.to_numpy()==d.UV.to_numpy()).all())
        comp=[]
        if aligned:
            for j in range(2):
                r=float(np.corrcoef(scores[:,j],prev[f'PC{j+1}'])[0,1]); sign=1 if r>=0 else -1
                err=float(np.max(np.abs(scores[:,j]*sign-prev[f'PC{j+1}'].to_numpy())))
                comp.append({'component':f'PC{j+1}','correlation_up_to_sign':abs(r),'comparison_sign':sign,'max_abs_difference':err})
                check(f'PCA PC{j+1} vs submitted coordinates (sign only)',scores[:,j]*sign,prev[f'PC{j+1}'],tol=1e-7)
        else: issue('PCA_ROW_MATCH','The historical PCA scores could not be aligned to the current rows by condition labels, so coordinate comparison was omitted.')
        table('PCA_legacy_coordinate_comparison.csv',comp,'audit')
    corr=pd.DataFrame(np.corrcoef(d[MORPH].to_numpy(float),rowvar=False),index=MORPH,columns=MORPH).reset_index(names='feature')
    table('Morph_correlation_matrix.csv',corr)
    # This is a copy/visualisation of a historical JMP table, not a rerun of JMP.
    jmp_path=(REFERENCE_DIR/'legacy_JMP_variable_clustering.csv') if REFERENCE_DIR is not None else INPUT_DIR/'archived_jmp_variable_clustering.csv'
    jmp=pd.read_csv(jmp_path) if jmp_path.is_file() else pd.DataFrame(columns=['Cluster','Variable','R-Square with Own Cluster','R-Square with Next Closest','1-R² Ratio'])
    table('Legacy_JMP_variable_clustering_NOT_recomputed.csv',jmp,'legacy_reference')
    issue('JMP_PROVENANCE','PCA is calculated from seven morphometrics in Python; comparisons with preserved scores are made when reference tables are supplied. The archived JMP variable-clustering summary is included without recalculation. Low R-square alone does not establish a causal role.')
    return sc,ld,corr,fit.explained_variance_ratio_,jmp

def hsi_inputs(src):
    pix={}; plate_rows=[]; spectra=[]; checks=[]; ref_waves=None
    for name,path in src.pixels():
        pid=name.split('_')[0]; group=pid[0]
        with path.open(encoding='utf-8-sig') as f: header=next(csv.reader(f))
        if header[:2]!=['row','col']: raise ValueError('Unexpected pixel coordinate columns: '+name)
        waves=np.asarray([float(v) for v in header[2:]])
        raw=np.loadtxt(path,delimiter=',',skiprows=1,ndmin=2)
        if raw.shape[1]!=len(header) or not np.isfinite(raw).all(): raise ValueError('Invalid pixel data '+name)
        xy=raw[:,:2]; inten=raw[:,2:]
        if np.any(inten.sum(1)<=0): raise ValueError('Nonpositive normalisation denominator; inspect '+name)
        if len(np.unique(xy,axis=0))!=len(xy): raise ValueError('Duplicated coordinates '+name)
        if ref_waves is None: ref_waves=waves
        if not np.array_equal(waves,ref_waves): raise ValueError('Different spectral grids')
        dist=np.linalg.norm(xy-xy.mean(0),axis=1)
        if not dist.max()>0: raise ValueError('Degenerate ROI '+name)
        radial=dist/dist.max(); cm=radial<=.30; em=radial>=.80
        if cm.sum()<2 or em.sum()<2: raise ValueError('Insufficient core or edge '+name)
        nfi=inten/inten.sum(1,keepdims=True)
        core=nfi[cm].mean(0); edge=nfi[em].mean(0)
        if np.any(core<=0): raise ValueError('Invalid core denominator '+name)
        ec=edge/core
        pix[pid]={'xy':xy,'intensity':inten,'nfi':nfi,'radial':radial,'core_mask':cm,'edge_mask':em,'ec':ec,'core':core,'edge':edge}
        checks.append({'plate':pid,'group_code':group,'n_roi_pixels':len(raw),'n_core':int(cm.sum()),'n_edge':int(em.sum()),
                       'n_middle':int((~cm & ~em).sum()),'n_bands':len(waves),'wavelength_min_nm':waves.min(),'wavelength_max_nm':waves.max(),
                       'min_raw_export':inten.min(),'max_raw_export':inten.max(),'source_sha256':digest(path),
                       'note':'Intensity values as supplied in pixel CSV; acquisition calibration not inferred.'})
        for j,w in enumerate(waves):
            spectra.append({'plate':pid,'group_code':group,'wavelength_nm':w,'ec_ratio':ec[j],'core_nfi':core[j],'edge_nfi':edge[j],
                            'whole_roi_nfi':nfi[:,j].mean(),'core_mean_intensity_export_units':inten[cm,j].mean(),
                            'edge_mean_intensity_export_units':inten[em,j].mean()})
        for target in [518.8,717.9]:
            j=int(np.argmin(abs(waves-target)))
            plate_rows.append({'plate':pid,'group_code':group,'label_from_legacy_codebook':GROUP_LABEL[group],
                               'wavelength_nm':waves[j],'core_nfi':core[j],'edge_nfi':edge[j],'ec_ratio':ec[j],
                               'n_core':int(cm.sum()),'n_edge':int(em.sum())})
    if ref_waves is None or len(ref_waves)!=60 or not np.all(np.diff(ref_waves)>0): raise ValueError('Expected 60 strictly increasing exported fluorescence wavelengths')
    counts=collections.Counter(p[0] for p in pix)
    if counts!={'A':9,'B':9,'C':9,'D':9,'E':4,'F':4}: raise ValueError('Unexpected HSI group counts')
    table('HSI_pixel_export_QC_44plates.csv',checks,'audit')
    per=table('HSI_target_indices_per_plate.csv',plate_rows)
    spec=table('HSI_all60bands_per_plate.csv',spectra)
    if REFERENCE_DIR is not None:
        old=load_reference('submitted_HSI_EC_PerPlate.csv'); old=old[old.modality=='RF_F'].sort_values('plate')
        now=per[per.wavelength_nm==518.8].sort_values('plate')
        if old.plate.tolist()!=now.plate.tolist(): raise ValueError('HSI plate IDs mismatch')
        check('44 fluorescence plate metrics at 518.8 vs submitted workbook',now[['core_nfi','edge_nfi','ec_ratio']].to_numpy(),old[['core_mean','edge_mean','ec_ratio']].to_numpy())
        oldscan=load_reference('submitted_HSI_Fscan_PerPlate.csv').sort_values(['plate','wl'])
        newscan=spec.sort_values(['plate','wavelength_nm'])
        check('2640 plate-band EC values vs submitted workbook',newscan.ec_ratio,oldscan.ec_ratio)
    issue('HSI_METADATA','A–D labels and the P24-192/36 h attribution follow the manuscript and codebook. Circular colony ROIs were selected to approximate the outlines; irregular marginal extensions could be excluded. Central and peripheral zones are computed from exported coordinates. E/F composition and UV labels, physical pixel dimensions, instrument calibration and acquisition-run metadata require documentation. Date-like archive labels refer to data processing/export organization, not acquisition sessions, as clarified by the study author. No session factor is inferred from those labels.')
    issue('NO_ADDITIONAL_HSI_ISOLATES','The supplied fluorescence inputs contain 44 unique A–F exports. Duplicate copies are counted once. These inputs do not provide independent isolate validation. Physical pixel dimensions and detector spectral resolution are not determined from wavelength-column spacing.')
    return pix,ref_waves,per,spec

def legacy_maxdiff_mc(x,y,seed):
    z=np.vstack([x,y]); lab=np.array([0]*len(x)+[1]*len(y)); obs=y.mean(0)-x.mean(0)
    rng=np.random.default_rng(seed); maxima=[]
    for _ in range(10000):
        p=rng.permutation(lab); dd=z[p==1].mean(0)-z[p==0].mean(0); maxima.append(abs(dd).max())
    maxima=np.asarray(maxima)
    return (np.sum(maxima[:,None]>=abs(obs)[None,:],axis=0)+1)/10001

def exact_spectral_tests(x,y):
    """Enumerate unique label assignments. Whole plate spectrum is permuted.
    Raw max absolute difference is the inherited statistic; studentised max-|t|
    is reported separately, regardless of whether it is more or less significant.
    Both rely on label exchangeability, not independent experimental replication.
    """
    n0,n1=len(x),len(y); z=np.vstack([x,y]); n=len(z); ncomb=math.comb(n,n0)
    if ncomb>100000: raise ValueError('Exact permutation budget exceeded')
    total=z.sum(0); total2=(z*z).sum(0); obs=y.mean(0)-x.mean(0)
    denom=np.sqrt(x.var(0,ddof=1)/n0+y.var(0,ddof=1)/n1)
    if np.any(denom<=0): raise ValueError('Degenerate spectral t statistic')
    tobs=obs/denom
    rawcount=np.zeros(z.shape[1],dtype=np.int64); rawmax=rawcount.copy(); tcount=rawcount.copy(); tmax=rawcount.copy()
    maxima_raw=[]; maxima_t=[]; it=itertools.combinations(range(n),n0)
    done=0
    while True:
        cc=list(itertools.islice(it,512))
        if not cc: break
        ids=np.asarray(cc); aa=z[ids]; sx=aa.sum(1); ssx=(aa*aa).sum(1); sy=total-sx; ssy=total2-ssx
        df=sy/n1-sx/n0
        vx=np.maximum((ssx-sx*sx/n0)/(n0-1),0); vy=np.maximum((ssy-sy*sy/n1)/(n1-1),0)
        se=np.sqrt(vx/n0+vy/n1)
        if np.any(se<=0): raise ValueError('Degenerate permuted spectral t statistic')
        tt=df/se; ar=abs(df); at=abs(tt); mr=ar.max(1); mt=at.max(1)
        tol=1e-12*np.maximum(1,abs(obs)); ttol=1e-12*np.maximum(1,abs(tobs))
        rawcount+=(ar>=abs(obs)[None,:]-tol).sum(0); rawmax+=(mr[:,None]>=abs(obs)[None,:]-tol).sum(0)
        tcount+=(at>=abs(tobs)[None,:]-ttol).sum(0); tmax+=(mt[:,None]>=abs(tobs)[None,:]-ttol).sum(0)
        maxima_raw.extend(mr.tolist()); maxima_t.extend(mt.tolist()); done+=len(cc)
    return {'n_assignments':done,'mean_difference':obs,'t_observed':tobs,
            'p_raw_difference':rawcount/done,'p_max_abs_difference':rawmax/done,
            'p_studentized_t':tcount/done,'p_max_abs_t':tmax/done,
            'null_max_difference':np.asarray(maxima_raw),'null_max_t':np.asarray(maxima_t)}

def hsi_contrasts(pix,waves):
    xs={g:np.stack([v['ec'] for p,v in sorted(pix.items()) if p[0]==g]) for g in 'ABCDEF'}
    rows=[]; allspec=[]; legacyrows=[]
    ref=load_reference('submitted_HSI_Contrasts.csv') if REFERENCE_DIR is not None else None
    for g0,g1,label,seed in [('A','B','EtOH',123),('C','D','PhSOFA',42)]:
        x,y=xs[g0],xs[g1]; obs,ci,dist=boot_interleaved(x,y)
        # Full reproduction of old per-wavelength bootstrap uses B=2000.
        _,ci2,di2=boot_interleaved(x,y,b=2000)
        if REFERENCE_DIR is not None:
            prev=load_reference('submitted_HSI_Fscan_'+('BminusA' if g0=='A' else 'DminusC')+'.csv')
            check(f'{g1}-{g0} old 2000/42 spectral bootstrap vs submitted workbook',np.column_stack([di2.mean(0),ci2.T]),prev[['diff','ci_low','ci_high']].to_numpy(),tol=1e-9)
        pmc=legacy_maxdiff_mc(x,y,seed)
        if REFERENCE_DIR is not None:
            refp=load_reference('submitted_HSI_Fscan_'+('BmA' if g0=='A' else 'DmC')+'_FWER.csv')
            check(f'{g1}-{g0} legacy max-absolute-difference MC FWER',pmc,refp.p_fwer.to_numpy(),tol=1e-12)
        log(f'Exact spectral label assignments: {label}, {math.comb(len(x)+len(y),len(x)):,}')
        ex=exact_spectral_tests(x,y)
        for j,w in enumerate(waves):
            allspec.append({'comparison':g1+'-'+g0,'condition_from_codebook':label,'wavelength_nm':w,
                            'mean_difference_observed':obs[j],'ci_low':ci[0,j],'ci_high':ci[1,j],
                            'legacy_mean_of_boot_differences_2000':di2[:,j].mean(),'legacy_mc_max_abs_diff_p':pmc[j],
                            **{k:v[j] for k,v in ex.items() if isinstance(v,np.ndarray) and len(v)==len(waves)},
                            'n0':len(x),'n70':len(y),'n_boot':NB_HSI,'seed':SEED_HSI,'exact_assignments':ex['n_assignments']})
        table(f'HSI_exact_null_maxima_{g1}_minus_{g0}.csv',{'max_abs_difference':ex['null_max_difference'],'max_abs_t':ex['null_max_t']})
        for target in [518.8,717.9]:
            j=int(np.argmin(abs(waves-target))); xx=x[:,j]; yy=y[:,j]
            sp=math.sqrt(((len(xx)-1)*xx.var(ddof=1)+(len(yy)-1)*yy.var(ddof=1))/(len(xx)+len(yy)-2))
            correction=math.exp(math.lgamma((len(xx)+len(yy)-2)/2)-.5*math.log((len(xx)+len(yy)-2)/2)-math.lgamma((len(xx)+len(yy)-3)/2))
            rows.append({'condition_from_codebook':label,'comparison':g1+'-'+g0,'wavelength_nm':waves[j],
                         'n0':len(xx),'n70':len(yy),'mean_UV0':xx.mean(),'mean_UV70':yy.mean(),
                         'sd_UV0':xx.std(ddof=1),'sd_UV70':yy.std(ddof=1),
                         'mean_difference_observed':obs[j],'ci_low':ci[0,j],'ci_high':ci[1,j],
                         'relative_to_UV0_percent':100*obs[j]/xx.mean(),'hedges_g_descriptive':correction*obs[j]/sp,
                         'legacy_bootstrap_mean_10000':dist[:,j].mean(),'n_boot':NB_HSI,'seed':SEED_HSI,
                         'p_exact_max_abs_difference':ex['p_max_abs_difference'][j],
                         'p_exact_max_abs_t_sensitivity':ex['p_max_abs_t'][j]})
            if target==518.8 and ref is not None:
                rr=ref[(ref.modality=='RF_F')&(ref.contrast==g1+'_minus_'+g0)].iloc[0]
                check(f'{label} legacy reported target contrast', [dist[:,j].mean(),*ci[:,j]], [rr.mean_diff,rr.ci_low,rr.ci_high])
        legacyrows.append({'comparison':g1+'-'+g0,'best_wavelength_by_absolute_raw_mean_diff':waves[np.argmax(abs(obs))],
                           'legacy_MC_min_adjusted_p':pmc.min(),'exact_raw_min_adjusted_p':ex['p_max_abs_difference'].min(),
                           'exact_t_min_adjusted_p':ex['p_max_abs_t'].min()})
    # Difference of contrasts; do not infer interaction from one significant
    # within-chemical contrast and one non-significant within-chemical contrast.
    interactions=[]; rng=np.random.default_rng(20260928)
    bs={g:xx[rng.integers(0,len(xx),size=(NB_HSI,len(xx)))].mean(1) for g,xx in xs.items() if g in 'ABCD'}
    dib=bs['D']-bs['C']-bs['B']+bs['A']; did=xs['D'].mean(0)-xs['C'].mean(0)-xs['B'].mean(0)+xs['A'].mean(0)
    for target in [518.8,717.9]:
        j=int(np.argmin(abs(waves-target))); ci=np.quantile(dib[:,j],[.025,.975])
        interactions.append({'wavelength_nm':waves[j],'comparison':'(D-C)-(B-A)','estimate':did[j],
                             'ci_low_unadjusted':ci[0],'ci_high_unadjusted':ci[1],'n_boot':NB_HSI,'seed':20260928,
                             'scope':'exploratory comparison of supplied groups; acquisition/batch design not independently verified'})
    # Blank codes are retained without claiming calibration of their UV treatment.
    ob,ci,dist=boot_interleaved(xs['E'],xs['F'])
    table('HSI_blank_codes_F_minus_E_DESCRIPTIVE_LABELS_REQUIRE_CONFIRMATION.csv',
          [{'wavelength_nm':w,'mean_difference_F_minus_E':ob[j],'ci_low':ci[0,j],'ci_high':ci[1,j],
            'n_E':len(xs['E']),'n_F':len(xs['F']),'note':'Legacy labels require author confirmation; no proof of background equivalence'} for j,w in enumerate(waves)])
    targetdf=table('HSI_target_effect_sizes.csv',rows)
    spectral=table('HSI_spectrum_legacy_and_exact_tests.csv',allspec)
    table('HSI_target_difference_of_contrasts.csv',interactions)
    table('HSI_permutation_comparison.csv',legacyrows,'audit')
    issue('HSI_POINT_ESTIMATOR','Current point estimates are observed differences between group means. The historical helper instead reported the average of bootstrap mean differences; those estimates remain in comparison columns. Bootstrap samples provide the current confidence intervals.')
    issue('MAX_T_TERMINOLOGY','The historical permutation statistic is the maximum absolute group-mean difference, without studentization. The analysis reports the original Monte Carlo procedure, exact enumeration of 48620 whole-export label assignments and a separate studentized maximum-|t| sensitivity test. Permutation inference assumes label exchangeability.')
    issue('HSI_SELECTED_BAND','717.9 nm was selected from the same 60-band dataset and is its highest exported wavelength. Its pointwise bootstrap CI is not adjusted for selection. No biochemical assignment, independent prediction or advantage over same-time morphometry has been established. 518.8 nm is the target retained from the original analysis.')
    return xs,targetdf,spectral,pd.DataFrame(interactions)

def grid_for_pixel(p,waves):
    cfs=np.round(np.arange(.20,.401,.02),2); efs=np.round(np.arange(.70,.901,.02),2)
    r=p['radial']; order=np.argsort(r); ds=r[order]
    out=[]
    for target in [518.8,717.9]:
        j=int(np.argmin(abs(waves-target))); vv=p['nfi'][order,j]; cs=np.cumsum(vv); total=cs[-1]
        cnt=np.searchsorted(ds,cfs,side='right'); st=np.searchsorted(ds,efs,side='left')
        if (cnt<2).any() or (len(ds)-st<2).any(): raise ValueError('Grid contains insufficient region pixels')
        core=cs[cnt-1]/cnt; before=np.where(st>0,cs[np.maximum(st-1,0)],0)
        edge=(total-before)/(len(ds)-st)
        out.append(edge[None,:]/core[:,None])
    return np.stack(out),cfs,efs

def hsi_sensitivity(pix,waves,src):
    gg={p:grid_for_pixel(v,waves)[0] for p,v in sorted(pix.items()) if p[0] in 'CD'}
    _,cfs,efs=grid_for_pixel(next(iter(pix.values())),waves)
    cc=np.stack([g for p,g in gg.items() if p[0]=='C']); dd=np.stack([g for p,g in gg.items() if p[0]=='D'])
    rows=[]; rng=np.random.default_rng(2026)
    for j,target in enumerate([518.8,717.9]):
        ob,ci,di=boot_interleaved(cc[:,j],dd[:,j],b=2000,rng=rng)
        if REFERENCE_DIR is not None:
            legacy=load_reference(f'sensitivity_D_minus_C_wl_{target:.1f}.csv').sort_values(['core_frac','edge_frac'])
            check(f'{target:.1f} nm legacy grid bootstrap',np.column_stack([ob.ravel(),ci[0].ravel(),ci[1].ravel()]),legacy[['mean_diff','ci_low','ci_high']],tol=1e-8)
        # Higher resample count is an explicitly labelled revision analysis.
        ro,rc,_=boot_interleaved(cc[:,j],dd[:,j],b=NB_HSI,seed=42)
        for a,cf in enumerate(cfs):
            for b,ef in enumerate(efs):
                rows.append({'wavelength_nm':target,'core_fraction':cf,'edge_fraction':ef,
                             'mean_difference':ro[a,b],'ci_low':rc[0,a,b],'ci_high':rc[1,a,b],
                             'pointwise_ci_excludes_zero':bool(rc[0,a,b]>0 or rc[1,a,b]<0),
                             'legacy_2000_ci_low':ci[0,a,b],'legacy_2000_ci_high':ci[1,a,b],
                             'legacy_pointwise_ci_excludes_zero':bool(ci[0,a,b]>0 or ci[1,a,b]<0),
                             'is_original_default':bool(cf==.3 and ef==.8),'n_boot':NB_HSI,'seed':42})
    frame=table('HSI_ROI_sensitivity_grid_pointwise.csv',rows)
    issue('GRID_NOT_INDEPENDENT_VALIDATION','The 121 radial-cutoff combinations reuse the same C/D observations. Their confidence intervals are pointwise and not adjusted across the cutoff grid. This evaluates parameter sensitivity, not independent validation or between-run reproducibility.')
    return frame

def log_model_diagnostics(d):
    """Calculate log-area residuals for diagnostic plots."""
    groups={}
    for r in d.to_dict('records'):
        k=(r['Isolate'],r['Chemical'],str(r['UV']))
        groups.setdefault(k,[]).append(math.log(float(r['Area_mm2'])))
    rows=[]
    for r in d.to_dict('records'):
        fit=float(np.mean(groups[(r['Isolate'],r['Chemical'],str(r['UV']))]))
        rows.append({'row_identifier':r['row_identifier'],'Isolate':r['Isolate'],
                     'Chemical':r['Chemical'],'UV':r['UV'],'fitted_log_area':fit,
                     'residual_log_area':math.log(float(r['Area_mm2']))-fit,
                     'method':'cellwise mean of log(area); equivalent saturated model; display diagnostics only'})
    return table('Morph_log_area_display_diagnostics.csv', rows)
