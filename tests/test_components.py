"""Independent small-array checks for the added focal-component calculations."""
import itertools
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from uvsm import analysis as a
from uvsm import components as c

class ComponentChecks(unittest.TestCase):
    def test_exact_against_independent_enumeration(self):
        x=np.array([[.002,.005,2.5],[.003,.007,7/3],[.004,.006,1.5]])
        y=np.array([[.004,.007,1.75],[.007,.008,8/7],[.006,.009,1.5]])
        z=np.vstack([x,y]);obs=y.mean(0)-x.mean(0);null=[]
        for ix in itertools.combinations(range(6),3):
            iy=[j for j in range(6) if j not in ix]
            null.append(z[iy].mean(0)-z[list(ix)].mean(0))
        expected=(abs(np.asarray(null))>=abs(obs)-1e-12*np.maximum(1.,abs(obs))).mean(0)
        got=c.exact_component_pvalues(x,y)
        np.testing.assert_array_equal(got['p'],expected)
        self.assertEqual(got['n_assignments'],20)
    def test_exact_p_is_symmetric(self):
        x=np.arange(1,13).reshape(4,3)/71.;y=x*.7+.003
        np.testing.assert_array_equal(c.exact_component_pvalues(x,y)['p'],c.exact_component_pvalues(y,x)['p'])
    def test_constant_columns_have_unit_p(self):
        p=c.exact_component_pvalues(np.ones((3,2)),np.ones((3,2)))
        np.testing.assert_array_equal(p['p'],[1.,1.])
    def test_exact_invalid_input_stops(self):
        for x,y in [(np.array([[1,np.nan],[2,3]]),np.ones((2,2))),
                    (np.ones((1,2)),np.ones((2,2))), (np.ones((2,2)),np.ones((2,3)))]:
            with self.assertRaises(ValueError):c.exact_component_pvalues(x,y)
    def test_joint_bootstrap_preserves_log_identity(self):
        core=np.array([.01,.02,.03,.045]);edge=np.array([.019,.025,.04,.031])
        x=np.log(np.column_stack((core,edge,edge/core)))
        y=x+np.array([.13,.05,-.08])
        obs,ci,d=a.boot_interleaved(x,y,b=213,seed=42)
        np.testing.assert_allclose(d[:,2],d[:,1]-d[:,0],rtol=0,atol=2e-15)
        self.assertAlmostEqual(obs[2],obs[1]-obs[0])
    def test_shared_rows_keep_existing_ratio_bootstrap(self):
        x=np.arange(1,55,dtype=float).reshape(9,6)/100.;y=x**1.1
        obs,ci,bs=a.boot_interleaved(x,y,b=117,seed=42)
        for i in [2,5]:
            oo,cc,bb=a.boot_interleaved(x[:,i],y[:,i],b=117,seed=42)
            np.testing.assert_allclose(bs[:,i],bb,rtol=0,atol=1e-15)
            np.testing.assert_allclose(ci[:,i],cc,rtol=0,atol=1e-15)
    def test_holm_excludes_four_ratio_rows(self):
        rows=[]
        for group in ['EtOH','PhSOFA']:
            for band in c.FOCAL_BANDS:
                for metric in c.COMPONENTS:
                    rows.append({'condition':group,'wavelength_nm':band,'metric':metric,'p_exact_two_sided':.01})
        result=c.apply_component_holm(pd.DataFrame(rows))
        np.testing.assert_allclose(result.loc[result.metric!='ec_ratio','p_holm_8_components'],.08)
        self.assertTrue(result.loc[result.metric=='ec_ratio','p_holm_8_components'].isna().all())
    def test_eight_test_family_must_be_complete(self):
        with self.assertRaises(ValueError):
            c.apply_component_holm(pd.DataFrame({'metric':['core_nfi'],'p_exact_two_sided':[.01]}))
    def test_arithmetic_ratio_not_ratio_of_means(self):
        core=np.array([1.,4.,3.]);edge=np.array([2.,1.,6.])
        self.assertNotAlmostEqual(np.mean(edge/core),np.mean(edge)/np.mean(core))
        self.assertAlmostEqual(np.mean(np.log(edge/core)),np.mean(np.log(edge))-np.mean(np.log(core)))

if __name__=='__main__':unittest.main(verbosity=2)
