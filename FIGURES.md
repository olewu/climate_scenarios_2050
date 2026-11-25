# Output figures

What each figure of the pipeline shows. Figures are written to `paths.outputs.figures`
(`/projects/NS9873K/www/users/owul/climate_scenarios_2050/`) as

    <figures>/<step>/<hash>/<name>.png

`<hash>` is the hash of the step's output folder: the configuration behind a figure is in
`<paths.outputs.root>/<step>/<hash>/config.yaml` (same hash), next to the netCDF files the figure is drawn from.
To find the hashes of an experiment: `pixi run outputs --config configs/experiments/<x>.yaml [--step linear-model]`.
To compare experiments side by side (drawn in the browser): `pixi run gallery --config ...`, then the pages in
`<figures>/gallery/` (https://ns9873k.web.sigma2.no/users/owul/climate_scenarios_2050/gallery/): `index_skill.html`
(evaluate-index), `linear_model.html` (linear-model), `target.html` (evaluate-target), `partition.html` (partition).

| Step | Figure folder | Files |
|------|---------------|-------|
| `evaluate-index` | `evaluate_index/<hash>/` | `skill_corr_<scenario>`, `skill_rmsess_<scenario>`, `continuous_all_<scenario>`, `hindcast_<scenario>`, `windows_<scenario>` |
| `linear-model` | `linear_model/<hash>/` | `linear_model_<target variable>_<scenario>` |
| `evaluate-target` | `evaluate_target/<hash>/` | `corr_<target>_<target variable>_<scenario>`, `crpss_...`, `dcrpss_...`, `dcorr_...`, `corr_anom_...` |
| `forecast` (with `--site`) | `forecast/<hash>/` | `site_<lat>N_<lon>E_init<year>[_ref<year>]` |
| `partition` | `partition/<hash>/` | `partition_<index>`, `partition_<target>_<region>` |

Conventions used throughout (details in `DESIGN.md`, section 2):
- **Init year Y**: the forecast is issued after the DJFM season Y. **Lead year k** is calendar year Y+k for annual
  targets and season Y+k for the index.
- **Target window a-b** (e.g. `1-10`): the mean over lead years a ... b. A **change** is that window mean minus the
  mean of the **reference period**, the `target.reference_length` (20) years up to and including Y.
- **Index units**: standardized, i.e. in units of the reference dataset's (ERA5) normal-period standard deviation.
  The hindcasts verify the **internal** index: the forced part (scaled CMIP6 multi-model ensemble mean) is removed
  from the observed series.
- **Scores**: `corr` is the Pearson correlation over the init years. `nrmse` (shown as RMSESS or 1 - RMSE/std) is
  1 - RMSE / std(observed): 0 is as good as the observed mean, 1 is perfect, and it can be negative. CRPSS is
  1 - CRPS / CRPS of a reference forecast (named in each figure below).
- **Confidence bands**: 95 % moving-block bootstrap over the init years (`evaluation.bootstrap`, 1000 samples).
  The block length is at least the averaging length (the running mean or the window length) because neighbouring
  inits overlap.

With several indices (e.g. `jet_lat` + `jet_speed`, NAO + EA), the index evaluation writes one set of figures per
index, each in its own `<hash>` folder (the title names the index). The analogues are matched jointly on all of them.

---

## 1. Index hindcast (`evaluate-index`)

Products compared in these figures (product name, colour in `continuous_all` and `windows`):

| Product | Colour | What it is |
|---------|--------|------------|
| `analogue` | C0 (blue) | Mean of the `analogues.M` (100) CMIP6 members whose filtered index best matches the observed index over the matching window up to Y. Their internal index after Y is the hindcast |
| `cmip6_all` | black | All members of the scenario: the climatological reference without any selection. Its mean is close to zero and the same for every init (lead only shifts the verified years), so its correlation means nothing (noise of the ~400-member mean plus a small forced remainder); its RMSE skill and CRPS are meaningful |
| `ar1_fair` | C1 (orange) | AR(1) fitted to the observed annual internal index over the matching window (around a linear trend that is not extrapolated); 100 realizations |
| `ar1_all` | C3 (red) | The same AR(1), fitted to all observed years up to Y |
| `ar1_rm_fair` | C4 (default) | AR(1) fitted to the 8-year running mean (old method); only in the continuous scores |
| `sine_trend` | C2 (green) | Sine + trend fit of the low-pass filtered observed index plus MA(1) residuals (old FFT forecast), deterministic |
| `persistence` | C5 | The mean of the last 20 observed years, held at all leads, deterministic |

The synthetic null ("synthetic p95" / "random 95th") is the 95th percentile of the same score when the analogue
method is run against `synthetic_null.n` (500) synthetic AR(1) "observations" in place of the real index. Above
that line, a skill is unlikely to come from the method itself, e.g. from the overlap of the running means. It is
only computed when the analogues are matched on a single index.

### `skill_corr_<scenario>.png` / `skill_rmsess_<scenario>.png` (layout of the old `evaluate_full_synth.py`)
- **x**: lead time in years (1-33, `evaluation.index_continuous.lead_times`).
- **y**: correlation, or RMSESS (`nrmse`), over all init years, between the hindcast and the observed internal
  index. Both are smoothed with the same centred 8-year running mean around each lead
  (`evaluation.index_continuous.running_mean`). The observed index is split with coefficients fitted on the full
  record.
- **Lines**: `Analogue` (blue) with its 95 % bootstrap band; `AR(1) fair` (orange) with band; `AR(1) full` (thin
  orange dashed); `random 95th` (black dotted, the synthetic null). Dashed grey line at 0.
- **Black bar with label** (correlation only): the reference skill from `predictor.reference_skill`, e.g. the DCPP
  multi-model NAO correlation for lead years 2-9, at its central lead. It is absent for the jet.
- The y range is fixed: -0.4 to 0.9 for the correlation, -0.5 to 0.5 for RMSESS.

### `continuous_all_<scenario>.png`
The same continuous scores as above, for **all** products in the table. Left: correlation, right: 1 - RMSE/std.
Each line has a light band (95 % bootstrap) and the synthetic p95 is grey dotted. The title says "running means"
because every value compares 8-year centred running means.

### `hindcast_<scenario>.png` (visual verification)
- **(a) top panel**: the ensemble-mean analogue hindcast, smoothed with the centred 8-year running mean.
  - **x**: the valid year. **y**: the init year.
  - **colour**: the index value, red positive and blue negative. The colour scale is symmetric up to the 98th
    percentile of |values|.
  - Each row is one hindcast, starting just right of the diagonal (lead 1).
  - Grey: no hindcast (before init, or beyond the last lead).
  - Dotted contours: lead times 5, 15, 25 and 35 years, drawn only up to the last observed year (the right edge of
    the panel).
  - The arrow shows the length of the matching window (`analogues.window`, 70 years) at an example init.
- **(b) bottom panel**, same x axis:
  - The observed internal index (`obs.matching_index`, e.g. the merged HadSLP/ERA5 or 20CR/ERA5 series) with the
    same running mean.
  - The hindcasts at fixed leads of 5 (solid), 15 (dashed) and 25 years (dotted), each plotted at its valid year.
    With skill, these lines follow the observed curve.

### `windows_<scenario>.png`
Skill for the **change** of the target-window mean against the 20 seasons before init, the quantity the forecasts
use.
- **x**: the target windows (`target.windows`: 1-5, 6-10, ..., 1-10, 11-20, 21-30, 1-20, 11-30).
- **Panels**:
  - correlation;
  - 1 - RMSE/std;
  - CRPSS against a "no change" forecast: a Gaussian with mean 0 and the spread of the observed changes, estimated
    leaving out the overlapping samples.
- **Observed changes** use a leave-out split: the forced-part coefficients are fitted without the verified window
  and its reference period.
- **Hindcast changes**: each product uses its own reference. For the analogue and CMIP6 products this is the
  members' own 20 years before init. For the baselines it is the observed years.
- **Markers per product**:
  - The CRPSS exists only for the ensemble products (analogue, cmip6_all, AR(1)); `sine_trend` and `persistence`
    are deterministic.
  - `ar1_rm_fair` is left out, since it forecasts the smoothed index.
- **Grey dotted**: the synthetic p95 for correlation and RMSE skill.
- The longer windows (21-30, 11-30) have few independent samples; the CRPSS there rests on only a handful of
  non-overlapping windows.

### CRPS on the gallery page (`index_skill.html`)
Not in the PNG figures. Under "Window changes": the CRPS of every product with members, and the CRPSS against
`cmip6_all`, 1 - CRPS(product) / CRPS(cmip6_all) (does the selection beat the unselected ensemble?), without an
uncertainty band. Under "Skill per lead time": the same for the 8-year running means per lead
(`continuous_crps.nc`), with a 95 % paired moving-block bootstrap band (blocks of max(n^1/3, 8) inits).

---

## 2. Linear model (`linear-model`): `linear_model_<target variable>_<scenario>.png`

Maps over the target region. Columns are the target windows 1-5, 1-10, 11-20 and 1-20.

The linear model is fitted per grid point on sliding window means of the observed internal target and of the
ERA5 index (`obs.training_index`). Its prediction is: internal target change = slope × (index window mean - index
reference mean).

There is one hindcast fit per init year and window. Training samples that overlap the target window are excluded,
and the forced signal is removed without the target window and its reference period.
- **Row 1, "corr given obs. index change"**: the correlation between the predicted and the observed **internal**
  target change over the init years, out of sample. Each case's fit is fed the **observed** index change of its
  window.
  - This is the part of the target change that the index explains. It is an upper limit for what a perfect index
    forecast could give through this model.
  - Unlike `perfect_index` in the target evaluation, it has no reversion-to-the-mean term.
  - Colour scale -1 to 1.
- **Next row(s), "change per 1 std index"** (one row per predictor with several indices, e.g.
  "change per 1 std jet_lat"): the slope, averaged over the init years' fits, × the std of the observed window
  means of the internal training index.
  - This is the target change for a typical index anomaly in that window. It is in the target's units (m/s for
    wsp100, K for tas, ...).
  - The std used is given in each panel title ("std of 1-10 index mean: 0.xx"). It shrinks for longer windows.
  - Colour scale centred on 0, robust limits (2nd-98th percentile).
- **For relative targets** (e.g. wsp100, pr), an extra row per predictor gives the same change in % of the
  long-term mean.

---

## 3. Target hindcast (`evaluate-target`)

Maps over the target region, with one row per method and one column per window (1-5, 6-10, 1-10, 11-20, 1-20,
21-30). Each panel title starts with the method and the window.

**Methods** (rows; all predictions are Gaussian with a mean and a spread; changes against the observed 20-year
reference):

| Row | Method |
|-----|--------|
| `M0` | Return to the long-term mean, without index information: change = minus the observed internal reference anomaly (plus the forced part for the full target); spread: RMS of its own errors (non-overlapping samples). The baseline the significance test compares against |
| `M2a` | Change of the target's CMIP6 variable (e.g. sfcWind for wsp100) in all members (not for energy). Not on the `internal` target: there it is its mean minus the MMEM change, i.e. zero |
| `M3a` | The same in the analogue members |
| `M2b` | Linear model on the index of all members |
| `M3b` | Linear model on the index of the analogue members |
| `perfect_index` | Linear model given the **observed** index of the window: an upper limit, not a forecast |
| `M0c` | Observations only, full target: return to the mean of the observed target over the years up to Y (`evaluation.obs_methods`) |
| `M0t` | Observations only, full target: a linear trend fitted to the observed target up to Y, read at the target window, minus the observed reference mean |
| `M1` | Persistence (no change); appears only in the significance maps, because it is the CRPSS reference of the score maps |

**`<target>`** in the file name:
- `full`: the observed change. M2b/M3b add the forced part from the index multi-model ensemble mean (option a).
- `internal`: the observed internal change, with the forced signal removed outside the target and reference period.
  The forced signal is the regression on the index MMEM (wind, energy) or on the target's own CMIP6 MMEM per grid
  point (tas, pr), see `targets.<name>.forced`.
  For M2a/M3a, the change of the CMIP6 multi-model ensemble mean is subtracted.

The `_b` variants (forced part from the CMIP6 variable) are in `scores.nc`, but not in the figures.

### Score maps: `corr_<target>_...png`, `crpss_<target>_...png`
- **corr**: the correlation over the init years between the predicted mean change and the observed change. Colour
  scale -1 to 1.
- **crpss**: the CRPSS against `M1`, a Gaussian with mean 0 (no change) and the RMS of the observed changes
  (leave-out, non-overlapping samples). Positive means better than persistence. Colour scale -0.5 to 0.5.
- **With the significance test** (`evaluation.significance.enabled`), every row except M0 also gets:
  - **Dots** (`....`): grid points where the method beats M0 locally. The corr map uses `dcorr`, the crpss map
    `dcrpss`, after the false discovery rate correction (Benjamini-Hochberg, q = 2 × alpha = 0.10).
  - **A second title line**, e.g. `+0.05 (p=0.012), 64% (p=0.030)`:
    - first, the area-weighted mean of that statistic and its field p-value;
    - second, the share of the area where it is > 0 and its p-value.
- **The null of the test**: no information beyond M0. It uses 1000 phase-randomized surrogates of the observed
  internal target that keep its autocorrelation and spatial covariance. The model-based parts of the predictions
  stay fixed.

**Test against M1** (`evaluation.significance.versus_m1`, `significance_m1.nc`, not drawn in the PNGs; the
comparison pages show it as "dots: method beats M1"): whether a method is better than persistence, for `crpss`
and `msss_m1` = 1 - MSE(m)/MSE(M1). A paired moving-block bootstrap over the init years (block: max(n^(1/3), window
length), the same draws for all points); one-sided local p-values, FDR, field test with the area mean.
The surrogate test cannot answer this question: its null keeps the autocorrelation, which is where M0's
advantage over M1 comes from.

### Significance maps: `dcrpss_<target>_...png`, `dcorr_<target>_...png`, `corr_anom_<target>_...png`
The skill **beyond M0** itself, with the same dots and title lines. Rows: M2a, M3a, M2b, M3b, perfect_index and M1.
- `dcrpss` = CRPSS(method) - CRPSS(M0). This is the primary statistic. Colour scale ±0.3.
- `dcorr` = corr(method) - corr(M0). Colour scale ±0.5.
- `corr_anom` = corr(prediction - M0, observed - M0): does the method predict the departure from the return to the
  mean? For M2b/M3b it equals the correlation of the window levels. Colour scale ±1.

---

## 5. Variance partition (`partition`): `partition_<index>.png`, `partition_<target>_<region>.png`

How much of the spread of the CMIP6 projections comes from internal variability, model uncertainty and scenario
uncertainty (Hawkins & Sutton 2009), for the change of an N-year window mean against the 20 years up to
`partition.init` (2025). One row per window length (`partition.window_lengths`: 5, 10, 20).
- **x**: the centre year of the window. **Dotted line**: 2050.
- **Left**: fractions of the total variance, stacked: internal (blue), model (orange), scenario (green).
- **Right**: the same as standard deviations (index units, % for relative targets, K for tas), total dashed.
- Indices: from the split of each scenario. Targets: the area mean of the target's CMIP6 field over a box of
  `partition.regions` (default: Norway and Sweden, 4-32E, 55-72N); wsp100 uses CMIP6 sfcWind (10 m), energy has none.
- Definitions (model-weighted means, var with ddof 1): internal = mean over scenarios and models of the mean squared
  internal change of the members; model = mean over scenarios of the variance of the models' forced changes;
  scenario = variance over the scenarios of the multi-model mean forced change.
- The scenarios have different model sets (ssp119: 17 models), so the scenario part partly reflects those. With
  `split.method: mmem` each model's forced part is a scaled multi-model mean: model uncertainty is only a
  difference in amplitude, likely an underestimate.

## 4. Forecast at a site (`forecast --init-year Y --site lat,lon`): `site_<lat>N_<lon>E_init<Y>.png`

The predictive distribution of each method for one init year at one grid point (design of the old
`make_vio_plot`). The numbers are in the CSV of the same name in the forecast output folder.
- **Rows**: the scenarios. **x**: the target windows (lead years after Y). The reference period is in the x label.
  **y**: the change in the target's units (% for relative targets).
- **One violin per method and window**, with colours:
  - M0 dark grey, M1 light grey, M0c brown, M0t purple (observations only);
  - M2a orange, M2b red;
  - M3a green, M3b blue.
- **The shape of each violin** is the predictive density, shown between its 0.1 and 99.9 % quantiles:
  - M0, M1, M0c, M0t: a single Gaussian (M0c/M0t: spread from their errors at earlier, complete windows).
  - M2b, M3b: a mixture of one Gaussian per member. Each is centred at the linear-model prediction from that
    member's index window mean and has the residual spread of the linear model. For M2b the members are weighted
    like `cmip6.weighting`.
  - M2a, M3a: a kernel density of the members' CMIP6 changes, scaled so that it keeps the members' spread.
- **Thick horizontal bar**: the median. **Thin vertical line**: 5-95 %.
- **Legend**: the number in brackets is the number of members in the mixture (e.g. `M3b (100)`).
- **Black dashed**: the observed change, when the window lies inside the observed record. This only happens for
  past init years.
- **Grey line at 0**: no change.
- With `--reference-value` (a measured mean at the site), the CSV also gives absolute changes. The figure stays
  relative.
