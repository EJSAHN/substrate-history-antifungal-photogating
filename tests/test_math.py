"""Development checks of resampling and immutable-source behaviour. No project code imported."""
import unittest, itertools, tempfile
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from uvsm import analysis as a

a.np=np

class MathChecks(unittest.TestCase):
    def test_interleaved_replays_original(self):
        x=np.array([.1,.4,.6,.7]); y=np.array([.2,.3,.5,.9])
        for hf in (False,True):
            obs,ci,bs=a.boot_interleaved(x,y,b=37,seed=42,high_first=hf)
            rng=np.random.default_rng(42); manual=[]
            for _ in range(37):
                if hf:
                    yy=rng.choice(y,len(y),replace=True).mean(); xx=rng.choice(x,len(x),replace=True).mean()
                else:
                    xx=rng.choice(x,len(x),replace=True).mean(); yy=rng.choice(y,len(y),replace=True).mean()
                manual.append(yy-xx)
            np.testing.assert_allclose(bs,manual,rtol=0,atol=0)
            self.assertAlmostEqual(obs,y.mean()-x.mean())
    def test_exact_stats_against_simple_enumeration(self):
        x=np.array([[1,2],[2,4],[3,5]],float); y=np.array([[4,3],[5,6],[6,7]],float)
        got=a.exact_spectral_tests(x,y); z=np.vstack([x,y]); diffs=[]; ts=[]
        for c in itertools.combinations(range(6),3):
            ix=np.array(c); iy=np.array([v for v in range(6) if v not in c])
            xx,yy=z[ix],z[iy]; dd=yy.mean(0)-xx.mean(0)
            tt=dd/np.sqrt(xx.var(0,ddof=1)/3+yy.var(0,ddof=1)/3)
            diffs.append(abs(dd)); ts.append(abs(tt))
        diffs=np.array(diffs); ts=np.array(ts)
        obs=abs(y.mean(0)-x.mean(0)); tobs=obs/np.sqrt(x.var(0,ddof=1)/3+y.var(0,ddof=1)/3)
        raw=(diffs.max(1)[:,None]>=obs[None,:]-1e-12).mean(0)
        t=(ts.max(1)[:,None]>=tobs[None,:]-1e-12).mean(0)
        np.testing.assert_allclose(got['p_max_abs_difference'],raw)
        np.testing.assert_allclose(got['p_max_abs_t'],t)
        swapped=a.exact_spectral_tests(y,x)
        np.testing.assert_allclose(got['p_max_abs_difference'],swapped['p_max_abs_difference'])
        np.testing.assert_allclose(got['p_max_abs_t'],swapped['p_max_abs_t'])
        self.assertEqual(got['n_assignments'],20)
        self.assertTrue((got['p_max_abs_difference']>=got['p_raw_difference']).all())
    def test_cell_bootstrap_reproducibility(self):
        x=np.arange(1,10,dtype=float); y=x*2
        obs,ci,b=a.boot_delta_legacy(x,y,b=5000)
        obs2,ci2,b2=a.boot_delta_legacy(x,y,b=5000)
        np.testing.assert_array_equal(b,b2); self.assertEqual(obs,5)
    def test_hash_seed_is_stable(self):
        self.assertEqual(a.stream_seed('A'),a.stream_seed('A'))
        self.assertNotEqual(a.stream_seed('A'),a.stream_seed('B'))

if __name__=='__main__': unittest.main(verbosity=2)
