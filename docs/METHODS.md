# Numerical definitions

The following definitions specify the calculations used in version 1.2.0.

## Morphology

The raw-area model uses sum-to-zero coding and all isolate × compound × categorical
UV terms. Type III tests use HC3 covariance. Ordinary OLS Type III tables and a
log-area model are retained as sensitivity analyses. Rows are not removed because
of residual magnitude. Independence and exchangeability depend on the acquisition
design and cannot be established by this software.

Inherited mean and UV70−UV0 summaries use 5,000 percentile bootstrap draws and
seed 1337. Matched-control quantities use independent cell streams: first four
SHA-256 bytes of `UVSM-R1-20260928|` plus the `(isolate, condition, UV)` label,
interpreted little endian. Control draws are reused across derived comparisons.

Inhibition is 100 × (1 − treated mean / matched-vehicle mean). Vehicle references
are matched by isolate and UV. Vehicle inhibition is zero by definition; its
inhibition CI is not a precision estimate and is blank in the presentation table.

The eight compound-versus-vehicle UV contrasts are differences in differences on
the **area scale**. They use HC3 t tests with Holm correction across eight tests.
Their individual 95% CIs are not simultaneous Holm-adjusted intervals and are
not p values for percentage-inhibition changes. The twelve within-treatment UV
contrasts have a separate Holm family of twelve. Circularity bootstrap intervals
and median-centred Levene p values are descriptive/unadjusted.

PCA uses seven standardised morphometrics and full SVD. Component signs are
oriented so the largest absolute coefficient is positive. This changes neither
variance explained nor distances. Variable–component correlations are not
presented as eigenvector coefficients. Archived JMP values are not recalculated.

## Spatial fluorescence

Circular ROIs were selected to approximate the colony outlines. Irregular marginal
extensions could fall outside these regions. The central/peripheral partition is
computed within the supplied pixel coordinates; it does not trace the full growth
boundary or use separately exported central/peripheral mean spectra. Distances are
expressed in the exported coordinate grid rather than millimetres.

For pixel i and exported wavelength j, `NFI[i,j] = I[i,j] / sum_k I[i,k]` over
all 60 exported bands. Coordinates are used unchanged. Centroid is the coordinate
mean, r is Euclidean distance in the exported grid, and rmax is the largest r.
Central pixels satisfy r/rmax ≤ 0.30; peripheral pixels satisfy r/rmax ≥ 0.80.
Middle pixels are excluded only from the ratio. E/C is mean peripheral NFI divided
by mean central NFI, not a mean of pixelwise E/C ratios.

Group contrasts report the observed mean difference. Pointwise percentile CIs
use 10,000 draws, seed 42, in the original interleaved sampling order. Historical
means of bootstrap differences and historical 2,000-draw scan CIs are retained
for comparison but are not relabelled as direct observations.

The historical Monte Carlo test uses the maximum absolute **unstandardised mean
difference** over 60 bands (10,000 permutations; A/B seed 123, C/D seed 42;
plus-one correction). It is not a studentized max-t statistic.

Exact tests enumerate all 48,620 assignments of nine out of eighteen exports,
permuting entire spectra. Exact p values include the observed assignment without
an additional Monte Carlo plus-one adjustment. A studentized maximum-|t| test is
reported separately as a sensitivity analysis.
Whole-spectrum label exchangeability is an assumption, not proof of batch balance.

518.8 nm is the original target. 717.9 nm was selected in the same dataset and
is its highest exported band. Its pointwise CI is not corrected for selection.
No biochemical identity is assigned. Direct `(D−C)−(B−A)` contrasts use a separate
10,000-draw stream, seed 20260928. E/F comparisons remain descriptively labelled
until acquisition details are confirmed.

Radial sensitivity evaluates 11 central cutoffs (0.20–0.40) and 11 peripheral
cutoffs (0.70–0.90). Legacy 2,000/2026 and current 10,000/42 intervals remain in
separate columns. This reuses the same exports; intervals are pointwise, not
adjusted across the ROI grid or independent validation.

## Historical discrepancy

All recovered dose means agree, but some historical dose-table CI limits have
not been reproduced exactly (maximum endpoint difference approximately 4.57 mm²).
When historical references are supplied, the archived limits and their differences are retained in the comparison output. The current
R1 inherited morphology convention is the reproduced 5,000/1337 workbook method.
