"""Validated, read-only inputs for the specific UVSM study.

No duplicate export is treated as another observation. Identical archive copies
are deduplicated by basename and SHA-256; conflicting copies stop the run.
"""
from __future__ import annotations
import csv
import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path

PIXEL_RE = re.compile(r'^[A-F][1-9][0-9]*_F_pixelSpectra\.csv$', re.I)

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1024*1024), b''): h.update(part)
    return h.hexdigest()

class InputFiles:
    """Inputs use conventional paths, not private recovery-object identifiers."""
    def __init__(self, root: Path, work: Path):
        self.root=Path(root).resolve()
        if not self.root.is_dir(): raise FileNotFoundError(f'Input directory: {self.root}')
        found=[p for p in (self.root/'phenotype.xlsx', self.root/'phenotype.csv') if p.is_file()]
        if len(found)!=1: raise ValueError('Supply exactly one phenotype.xlsx or phenotype.csv')
        self.phenotype=found[0]
        directory=self.root/'fluorescence'
        archive=self.root/'HSI.zip'
        if directory.is_dir() and archive.is_file():
            raise ValueError('Ambiguous fluorescence source: supply fluorescence/ OR HSI.zip, not both')
        self._pixels=[]
        hashes={}; paths={}; duplicate_counts={}
        def register(name,path):
            if not PIXEL_RE.fullmatch(name): return
            # Canonical identifiers retain original case in report files.
            pid=name.split('_')[0].upper()
            name=pid+'_F_pixelSpectra.csv'
            h=sha256(path)
            if name in hashes and hashes[name]!=h: raise ValueError('Conflicting byte versions for '+name)
            duplicate_counts[name]=duplicate_counts.get(name,0)+1
            hashes[name]=h;paths.setdefault(name,path)
        if directory.is_dir():
            for p in sorted(directory.rglob('*.csv')):
                if p.is_symlink() or not p.resolve().is_relative_to(directory.resolve()):
                    raise ValueError('Symlink/outside path in fluorescence inputs')
                register(p.name,p)
        elif archive.is_file():
            dest=Path(work)/'_extracted_fluorescence';dest.mkdir(parents=True,exist_ok=False)
            with zipfile.ZipFile(archive) as z:
                for i,item in enumerate(z.infolist()):
                    name=item.filename.replace('\\','/').split('/')[-1]
                    if not PIXEL_RE.fullmatch(name): continue
                    if item.file_size>20*1024*1024:raise ValueError('Unexpectedly large pixel CSV: '+name)
                    if item.flag_bits & 1:raise ValueError('Encrypted input member not supported')
                    # Do not extract arbitrary archive paths.
                    path=dest/(str(i)+'_'+name)
                    with z.open(item) as src,path.open('wb') as dst:shutil.copyfileobj(src,dst)
                    register(name,path)
        else:raise FileNotFoundError('Supply input/fluorescence/*.csv or input/HSI.zip')
        expected={f'{g}{i}_F_pixelSpectra.csv' for g in 'ABCDEF' for i in range(1,10 if g in 'ABCD' else 5)}
        if set(paths)!=expected:
            raise ValueError('Fluorescence ID mismatch; missing='+str(sorted(expected-set(paths)))+' extra='+str(sorted(set(paths)-expected)))
        self._pixels=sorted(paths.items())
        self.manifest=[{'file':self.phenotype.name,'sha256':sha256(self.phenotype),'role':'phenotype','copies':1}]
        self.manifest += [{'file':n,'sha256':hashes[n],'role':'fluorescence','copies':duplicate_counts[n]} for n in sorted(paths)]
        jmp=self.root/'archived_jmp_variable_clustering.csv'
        if jmp.is_file(): self.manifest.append({'file':jmp.name,'sha256':sha256(jmp),'role':'archived_summary','copies':1})
        manifest=self.root/'input_manifest.json'
        if manifest.is_file():
            spec=json.loads(manifest.read_text(encoding='utf-8-sig'))
            expected_map={r['file']:r['sha256'] for r in spec['files']}
            actual_map={r['file']:r['sha256'] for r in self.manifest}
            if actual_map!=expected_map:raise ValueError('Input hashes differ from input_manifest.json; originals were not changed by this run')
        self.manifest_verified=manifest.is_file()
    def pixels(self):return list(self._pixels)
    def verify_unchanged(self):
        before={r['file']:r['sha256'] for r in self.manifest}
        now={self.phenotype.name:sha256(self.phenotype)}
        now.update({name:sha256(p) for name,p in self._pixels})
        jmp=self.root/'archived_jmp_variable_clustering.csv'
        if jmp.is_file():now[jmp.name]=sha256(jmp)
        if now!=before:raise RuntimeError('An input file changed during the run')
