"""Focal-band fluorescence components and sampled-ROI geometry.

These are exploratory, plate-level summaries. They do not identify the biological
source of fluorescence or isolate the contribution of the inoculation plug.
"""
from __future__ import annotations

import itertools
import math
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from . import analysis as a

FOCAL_BANDS = (518.8, 717.9)
COMPONENTS = ('core_nfi', 'edge_nfi', 'ec_ratio')
COMPONENT_COLUMNS = [
    'condition', 'comparison', 'wavelength_nm', 'metric', 'n0', 'n70',
    'mean_UV0', 'mean_UV70', 'sd_UV0', 'sd_UV70',
    'mean_difference_observed', 'ci_low', 'ci_high',
    'relative_to_UV0_percent', 'p_exact_two_sided', 'p_holm_8_components',
    'multiplicity_family', 'exact_assignments', 'n_boot', 'seed', 'interpretation'
]
ROI_COLUMNS = [
    'plate', 'group_code', 'treatment_label', 'n_roi_pixels', 'n_core_pixels',
    'n_edge_pixels', 'n_middle_pixels', 'row_span_pixels', 'column_span_pixels',
    'max_radius_grid_units', 'central_fraction', 'peripheral_fraction',
    'mean_log_ratio_identity_max_abs_error'
]
GROUP_COLUMNS = [
    'group_code', 'treatment_label', 'n_exports', 'mean_roi_pixels', 'sd_roi_pixels',
    'min_roi_pixels', 'median_roi_pixels', 'max_roi_pixels', 'mean_core_pixels',
    'mean_edge_pixels', 'mean_middle_pixels', 'mean_row_span_pixels',
    'mean_column_span_pixels', 'interpretation'
]
LOG_COLUMNS = [
    'condition', 'comparison', 'wavelength_nm', 'n0', 'n70',
    'delta_mean_log_core', 'log_core_ci_low', 'log_core_ci_high',
    'delta_mean_log_edge', 'log_edge_ci_low', 'log_edge_ci_high',
    'delta_mean_log_ratio', 'log_ratio_ci_low', 'log_ratio_ci_high',
    'delta_log_edge_minus_delta_log_core', 'identity_abs_error',
    'max_bootstrap_identity_abs_error', 'n_boot', 'seed', 'interpretation'
]


def exact_component_pvalues(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Two-sided exhaustive unstudentized mean-difference tests.

Rows (whole plates) are permuted jointly across columns. The observed assignment
is included; there is no Monte Carlo plus-one adjustment. No max-statistic test
is performed across quantities with different scales. Comparison tolerance is
identical to the inherited raw-difference spectral calculation.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError('Expected two row-by-feature arrays with identical columns')
    if min(len(x), len(y)) < 2 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('At least two finite plate observations per group are required')
    n0, n1 = len(x), len(y)
    z = np.vstack((x, y))
    assignments = math.comb(len(z), n0)
    if assignments > 100_000:
        raise ValueError('Exact component permutation budget exceeded')
    observed = y.mean(0) - x.mean(0)
    threshold = abs(observed) - 1e-12 * np.maximum(1., abs(observed))
    total = z.sum(0)
    count = np.zeros(z.shape[1], dtype=np.int64)
    iterator = itertools.combinations(range(len(z)), n0)
    done = 0
    while True:
        combinations = list(itertools.islice(iterator, 512))
        if not combinations:
            break
        low_sum = z[np.asarray(combinations, dtype=int)].sum(1)
        differences = (total - low_sum) / n1 - low_sum / n0
        count += (abs(differences) >= threshold).sum(0)
        done += len(combinations)
    if done != assignments:
        raise ArithmeticError('Incomplete enumeration')
    return {'observed': observed, 'p': count / done, 'n_assignments': done}


def apply_component_holm(frame: pd.DataFrame) -> pd.DataFrame:
    """Correct the eight newly interpreted component tests, not legacy E/C tests."""
    result = frame.copy()
    result['p_holm_8_components'] = np.nan
    mask = result.metric.isin(['core_nfi', 'edge_nfi'])
    if int(mask.sum()) != 8:
        raise ValueError('Expected eight component tests: 2 treatments x 2 bands x 2 regions')
    result.loc[mask, 'p_holm_8_components'] = multipletests(
        result.loc[mask, 'p_exact_two_sided'].to_numpy(), method='holm')[1]
    result['multiplicity_family'] = np.where(
        mask, '8 exploratory central/peripheral component tests',
        'Existing ratio comparison; outside the new eight-test family')
    return result


def _plate_matrix(pix: dict, group: str, indices: list[int], log: bool = False) -> np.ndarray:
    rows = []
    for pid, p in sorted(pix.items()):
        if pid[0] != group:
            continue
        values = []
        for j in indices:
            values.extend((p['core'][j], p['edge'][j], p['ec'][j]))
        rows.append(values)
    result = np.asarray(rows, dtype=float)
    if not np.isfinite(result).all() or np.any(result <= 0):
        raise ValueError('Focal-band means and ratios must be finite and positive')
    return np.log(result) if log else result


def run(pix: dict, waves: np.ndarray) -> dict[str, pd.DataFrame]:
    """Add four tables; preserve all inherited tables and resampling settings."""
    indices = []
    for band in FOCAL_BANDS:
        hits = np.flatnonzero(np.isclose(waves, band, atol=1e-8, rtol=0))
        if len(hits) != 1:
            raise ValueError(f'Missing or ambiguous focal wavelength: {band}')
        indices.append(int(hits[0]))
    rows, logs = [], []
    for g0, g1, label in [('A', 'B', 'EtOH'), ('C', 'D', 'PhSOFA')]:
        x = _plate_matrix(pix, g0, indices)
        y = _plate_matrix(pix, g1, indices)
        if x.shape != (9, 6) or y.shape != (9, 6):
            raise ValueError('Expected nine exports per group and six focal quantities')
        # A single stream resamples entire plate rows jointly across regions and bands.
        observed, ci, draws = a.boot_interleaved(x, y, b=a.NB_HSI, seed=a.SEED_HSI)
        ex = exact_component_pvalues(x, y)
        np.testing.assert_allclose(observed, ex['observed'], rtol=0, atol=0)
        lx, ly = np.log(x), np.log(y)
        log_observed, log_ci, log_draws = a.boot_interleaved(
            lx, ly, b=a.NB_HSI, seed=a.SEED_HSI)
        for k, band in enumerate(FOCAL_BANDS):
            for m, metric in enumerate(COMPONENTS):
                i = 3*k + m
                rows.append({
                    'condition': label, 'comparison': f'{g1}-{g0}',
                    'wavelength_nm': float(waves[indices[k]]), 'metric': metric,
                    'n0': len(x), 'n70': len(y),
                    'mean_UV0': x[:, i].mean(), 'mean_UV70': y[:, i].mean(),
                    'sd_UV0': x[:, i].std(ddof=1), 'sd_UV70': y[:, i].std(ddof=1),
                    'mean_difference_observed': observed[i],
                    'ci_low': ci[0, i], 'ci_high': ci[1, i],
                    'relative_to_UV0_percent': 100*observed[i]/x[:, i].mean(),
                    'p_exact_two_sided': ex['p'][i],
                    'exact_assignments': ex['n_assignments'],
                    'n_boot': a.NB_HSI, 'seed': a.SEED_HSI,
                    'interpretation': ('Exploratory focal-band component analysis; pointwise CI; '
                                       '717.9 nm selected in the same dataset; no plug exclusion '
                                       'or absolute-intensity inference.')
                })
            c, e, r = 3*k, 3*k+1, 3*k+2
            identity = log_observed[e]-log_observed[c]
            err = float(abs(log_observed[r]-identity))
            boot_err = float(np.max(abs(log_draws[:, r]-(log_draws[:, e]-log_draws[:, c]))))
            if err > 1e-12 or boot_err > 1e-12:
                raise ArithmeticError('Paired log-ratio decomposition identity failed')
            logs.append({
                'condition':label, 'comparison':f'{g1}-{g0}', 'wavelength_nm':band,
                'n0':len(x), 'n70':len(y),
                'delta_mean_log_core':log_observed[c],
                'log_core_ci_low':log_ci[0,c], 'log_core_ci_high':log_ci[1,c],
                'delta_mean_log_edge':log_observed[e],
                'log_edge_ci_low':log_ci[0,e], 'log_edge_ci_high':log_ci[1,e],
                'delta_mean_log_ratio':log_observed[r],
                'log_ratio_ci_low':log_ci[0,r], 'log_ratio_ci_high':log_ci[1,r],
                'delta_log_edge_minus_delta_log_core':identity, 'identity_abs_error':err,
                'max_bootstrap_identity_abs_error':boot_err,
                'n_boot':a.NB_HSI, 'seed':a.SEED_HSI,
                'interpretation':('Descriptive identity for differences of mean natural logs; '
                                  'not a decomposition of arithmetic mean E/C differences, '
                                  'not causal attribution, and not plug-removal validation.')
            })
    contrasts = apply_component_holm(pd.DataFrame(rows))[COMPONENT_COLUMNS]
    geometry = []
    for pid, p in sorted(pix.items()):
        xy = p['xy']
        dist = np.linalg.norm(xy-xy.mean(0), axis=1)
        cm, em = p['core_mask'], p['edge_mask']
        if np.any(cm & em) or np.any(p['core']<=0) or np.any(p['edge']<=0):
            raise ValueError('Invalid region partition or nonpositive component')
        err = float(np.max(abs(np.log(p['ec'])-(np.log(p['edge'])-np.log(p['core'])))))
        geometry.append({
            'plate':pid, 'group_code':pid[0], 'treatment_label':a.GROUP_LABEL[pid[0]],
            'n_roi_pixels':len(xy), 'n_core_pixels':int(cm.sum()),
            'n_edge_pixels':int(em.sum()), 'n_middle_pixels':int((~cm & ~em).sum()),
            'row_span_pixels':float(np.ptp(xy[:,0])+1),
            'column_span_pixels':float(np.ptp(xy[:,1])+1),
            'max_radius_grid_units':float(dist.max()),
            'central_fraction':float(cm.mean()), 'peripheral_fraction':float(em.mean()),
            'mean_log_ratio_identity_max_abs_error':err
        })
    roi = pd.DataFrame(geometry)[ROI_COLUMNS]
    grouped = []
    for group, q in roi.groupby('group_code', sort=True):
        n = q.n_roi_pixels
        grouped.append({
            'group_code':group, 'treatment_label':a.GROUP_LABEL[group], 'n_exports':len(q),
            'mean_roi_pixels':n.mean(), 'sd_roi_pixels':n.std(ddof=1),
            'min_roi_pixels':int(n.min()), 'median_roi_pixels':float(n.median()),
            'max_roi_pixels':int(n.max()), 'mean_core_pixels':q.n_core_pixels.mean(),
            'mean_edge_pixels':q.n_edge_pixels.mean(), 'mean_middle_pixels':q.n_middle_pixels.mean(),
            'mean_row_span_pixels':q.row_span_pixels.mean(),
            'mean_column_span_pixels':q.column_span_pixels.mean(),
            'interpretation':('Exported ROI geometry in pixel-grid units; not measured whole-colony area. '
                              'Export or processing dates are not acquisition-session variables.')
        })
    tables = {
        'HSI_focal_component_contrasts.csv':contrasts,
        'HSI_focal_log_ratio_decomposition.csv':pd.DataFrame(logs)[LOG_COLUMNS],
        'HSI_ROI_geometry_per_plate.csv':roi,
        'HSI_ROI_geometry_by_group.csv':pd.DataFrame(grouped)[GROUP_COLUMNS]
    }
    for name, frame in tables.items():
        a.table(name, frame)
    # The E/C rows must reproduce existing focal results, not replace their inference.
    old = pd.read_csv(a.OUT/'tables/HSI_target_effect_sizes.csv')
    oldscan = pd.read_csv(a.OUT/'tables/HSI_spectrum_legacy_and_exact_tests.csv')
    for row in contrasts[contrasts.metric=='ec_ratio'].itertuples():
        target = old[(old.condition_from_codebook==row.condition) & (old.wavelength_nm==row.wavelength_nm)]
        scan = oldscan[(oldscan.comparison==row.comparison) & (oldscan.wavelength_nm==row.wavelength_nm)]
        if len(target)!=1 or len(scan)!=1:
            raise ValueError('Missing inherited comparison for component cross-check')
        np.testing.assert_allclose(
            [row.mean_difference_observed, row.ci_low, row.ci_high],
            target[['mean_difference_observed','ci_low','ci_high']].iloc[0].to_numpy(float),
            rtol=0, atol=1e-12)
        if abs(row.p_exact_two_sided-float(scan.p_raw_difference.iloc[0]))>1e-12:
            raise ArithmeticError('E/C exact p value differs from inherited result')
    a.dump_json(a.OUT/'HSI_component_checks.json', {
        'status':'PASS', 'ratio_reference_comparisons':4, 'component_tests':8,
        'holm_family_size':8, 'roi_exports':len(roi), 'group_counts':roi.groupby('group_code').size().to_dict(),
        'pointwise_bootstrap_draws':a.NB_HSI, 'seed':a.SEED_HSI,
        'resampling_unit':'export/plate; paired central and peripheral values resampled together',
        'log_identity_max_error':float(max(r['max_bootstrap_identity_abs_error'] for r in logs)),
        'session_factor_from_dates':False, 'plug_excluded':False,
        'note':'Component tests are exploratory; eight-test Holm adjustment does not account for prior band selection.'
    })
    return tables
