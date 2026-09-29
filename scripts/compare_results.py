#!/usr/bin/env python3
"""Compare CSV results without assuming byte identity across BLAS/OS builds."""
from __future__ import annotations
import argparse,csv,json,math
from pathlib import Path

def compare(expected:Path,actual:Path):
    current={}
    for p in actual.rglob('*.csv'):
        if p.name in current:raise ValueError('Duplicate output basename: '+p.name)
        current[p.name]=p
    report=[]
    for p in sorted(expected.glob('*.csv')):
        if p.name not in current:
            report.append({'file':p.name,'pass':False,'error':'missing'});continue
        e=list(csv.reader(p.open(encoding='utf-8-sig',newline='')))
        g=list(csv.reader(current[p.name].open(encoding='utf-8-sig',newline='')))
        if len(e)!=len(g) or e[0]!=g[0]:
            report.append({'file':p.name,'pass':False,'error':'shape/header'});continue
        ok=True;maxerr=0.;mismatches=0
        for er,gr in zip(e[1:],g[1:]):
            if len(er)!=len(gr):ok=False;continue
            for col,(x,y) in enumerate(zip(er,gr)):
                if x==y:continue
                try:
                    xx,yy=float(x),float(y)
                    if not(math.isfinite(xx) and math.isfinite(yy)):same=math.isnan(xx) and math.isnan(yy)
                    else:
                        err=abs(xx-yy);maxerr=max(maxerr,err)
                        # Fixed tolerances: covariance algebra can vary at roundoff level.
                        isp=('p_value'==e[0][col] or e[0][col].startswith('p_') or e[0][col]=='PR(>F)')
                        same=err <= ((1e-12+1e-9*abs(xx)) if isp else (1e-9+1e-10*abs(xx)))
                        if isp and ((xx<.05)!=(yy<.05)):same=False
                    if not same:ok=False;mismatches+=1
                except ValueError:ok=False;mismatches+=1
        report.append({'file':p.name,'pass':ok,'rows':len(e)-1,'max_abs_difference':maxerr,'mismatches':mismatches})
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference',type=Path,required=True);p.add_argument('--actual',type=Path,required=True);p.add_argument('--report',type=Path,required=True);a=p.parse_args()
    result=compare(a.reference,a.actual);a.report.parent.mkdir(parents=True,exist_ok=True)
    a.report.write_text(json.dumps({'all_pass':all(r['pass'] for r in result),'files':result},indent=2),encoding='utf8')
    print('[REGRESSION]',sum(r['pass'] for r in result),'/',len(result))
    raise SystemExit(0 if all(r['pass'] for r in result) else 2)
