# Restructuring design: decadal wind prediction from CMIP6 NAO analogues

Status: step 1 (preprocessing) implemented and tested on a subset; steps 2-7 not yet implemented.

## 1. Goal

Predict the change of 100 m wind speed (optionally wind energy) over Europe for the next
30 years, relative to the 20 years before initialization, and verify the methods in hindcast
mode without leaking future information (as far as the short record allows).

Forecast methods (all per CMIP6 scenario):

| ID | Method | Wind from | Needs linear model |
|----|--------|-----------|--------------------|
| M1 | Persistence of the last 20 years (zero change) | observations | no |
| M2a | All CMIP6 members, sfcWind directly | CMIP6 sfcWind | no |
| M2b | All CMIP6 members, NAO -> linear model | CMIP6 NAO | yes |
| M3a | NAO-analogue members, sfcWind directly | CMIP6 sfcWind | no |
| M3b | NAO-analogue members, NAO -> linear model | CMIP6 NAO | yes |

M2b vs M3b isolates the value of the analogue selection; M3a vs M3b isolates the value of the
statistical model vs model wind. Wind energy is only possible for M1, M2b, M3b.

## 2. Decisions (from review, Oct 2026)

### Target and time conventions
- **Target windows** (lead years, fixed for evaluation): 1-5, 6-10, 11-15, 16-20, 21-25, 26-30,
  1-10, 11-20, 21-30, 1-20, 11-30. Users can define their own windows later. No running
  mean on top of the window mean.
- **Init year Y**: the forecast is issued after the DJFM season Y (Dec Y - Mar Y+1), i.e. end
  of March Y+1.
- **Lead year k** = calendar year Y+k for annual quantities (wind). Window 1-5 = Y+1 ... Y+5.
- **Reference period** = calendar years Y-19 ... Y (last 20 complete years at issue time).
- **Season labels**: DJFM (and DJF) seasons are labelled by the year of December (current
  convention, `make_season_group`). Annual means are labelled by calendar year.
- **NAO ↔ wind pairing**: wind in calendar year t is paired with the DJFM season t-1 (the
  winter ending in year t) - this is what the current `year - 1` relabelling does. Hence
  the NAO window used for wind window [a, b] is seasons [Y+a-1, Y+b-1], and the NAO reference
  is seasons [Y-20, Y-1]. Configurable (`nao_wind_lag: 1`).
- **NAO hindcast verification** uses the same windows on seasons [Y+a, Y+b] relative to the
  20 seasons before init, in standardized units, plus the current "continuous" evaluation
  (correlation over the whole series per lead time, as in `evaluate_full_synth.py`).
- **Units**: relative change (%) by default, absolute (m/s or energy units) as option.
- **Sample size**: wind from 1940 + 20-year reference -> first init 1959; with data to 2025
  the last verifiable init for 21-30 is 1995 (~37 overlapping samples), for 1-5 ~62.

### Observations
- **Matching (analogue) NAO**: merged HadSLP -> ERA5 by default (HadSLP until 1979, ERA5 from
  1980; merge year and harmonization overlap period configurable, harmonization fitted once
  on the fixed overlap and frozen). The merge is applied to the full (not yet split) indices, so
  the merged series is split into forced/internal like any other series (old code merged after the split). Alternatives: HadSLP, 20CR, ERA5.
- **Linear model training**: always ERA5 (consistent with the ERA5-based wind data).
- **Common pattern**: all indices (obs and CMIP6) are projections onto **one** pattern,
  default ERA5 EOF1 over the normal period 1950-2015 on the 1.5° grid, standardized with that
  pattern's scores over the normal period. Options: each dataset's own pattern; each model's
  own pattern (pattern-correlation matching of modes to the observed pattern as a safeguard;
  models without full coverage of the normal period, e.g. EC-Earth3, are fitted on their
  available years and that period is recorded in the output).

### Forced / internal split
- Forced signal = multi-model ensemble mean (MMEM), scaled per model (`cov(SMEM, MMEM)/var(MMEM)`)
  and per observational series, as now. Alternative forced estimates: polynomial in time
  (any degree), low-pass filtered MMEM (Butterworth, as in `split_variability.py`).
- **Model-weighted MMEM** (mean of single-model ensemble means) by default, member-weighted
  as option - applied everywhere (NAO MMEM, sfcWind MMEM, ensemble statistics of M2a/M2b).
- **Observations, leakage**: default C = means and regression coefficients fitted on all years
  except the verified window and its reference period; option A = full period (old behaviour).
  For actual forecasts all data up to init is used. CMIP6 splits stay in-sample (1850-2100).
- **Scenarios are kept separate** (own ensemble, MMEM, split, analogue pool, hindcasts).
  Hindcast evaluation: ssp245 and ssp370. Forecasts: all scenarios side by side.

### Forced wind component in M2b/M3b
- Components are always stored separately: internal (linear model), NAO-forced
  (β_wsp · ΔNAO_MMEM, β from ERA5 wind), CMIP6 sfcWind forced (plain MMEM relative change),
  observed reference climatology.
- Default "full" forecast = internal + NAO-forced (option a); alternative = internal + CMIP6
  sfcWind forced (option b, replaces a); internal only (option c).
- Verification on both internal targets (internal predictions vs internal observed wind) and
  full targets.

### Analogues (step 3)
One configurable generator:
- `selection`: subsample (M best) | shift (best lag within ±max_lag per member) | shift+subsample
- `measure`: CORR | RMSE | two-stage (CORR then RMSE among the best)
- `filter`: running mean (length, default 41) | Butterworth low-/band-pass (cutoffs)
- `window`: matching window length (default 70) or "all"
- `M` (default 100), `max_lag`, optional model pre-selection (varsplit_Ny, Smith lists, NAO_AR)
- obs are cut at init; model series are filtered over their full length by default (`full`), option `cut`
- synthetic AR(1) "observations" as null test, for any configuration

### Significance of the wind skill maps (step 5, `evaluation.significance`)
- Null: no information beyond M0. Spatially coherent phase-randomized surrogates of the observed
  annual internal wind (one random phase rotation per frequency for all grid points: keeps the
  autocorrelation and the spatial covariance). Model-based parts of the predictions stay fixed.
- The leave-out split is repeated for every surrogate (exact, as linear operators per init/window):
  keeping the leave-out offsets fixed breaks their dependence on the anomalies and makes
  persistence-type methods look significant everywhere.
- Statistics: dCRPSS = CRPSS_m - CRPSS_M0 (primary), dcorr, corr_anom = corr(pred - M0, obs - M0)
  (= corr(P, W) of window levels for M2b/M3b). One-sided local p, FDR (Benjamini-Hochberg) at
  q = 2 alpha (Wilks 2016) for stippling; field tests with the area-weighted mean and the area
  fraction > 0. `significance.nc` in the evaluation folder.
- Calibrated on synthetic AR(1) fields (rejection rate 0.05 at 0.05). Northern Spain has a residual
  trend in the internal wind (0.43 m/s / 85 y) that the stationary null does not represent.
- Cost: ~1.7 h per scenario and wind variable with 1000 surrogates and 4 workers (0.25 deg grid).

### Predictors
Generic index interface: an index = (source variable, season/aggregation, domain, method).
Methods: EOF mode k projected on a common pattern (implemented first), box difference
(station NAO), jet latitude/speed (needs CMIP6 `ua`, not downloaded yet), AMV (SST; needs obs
SST choice). Single predictor for now; code takes a list so multi-predictor matching/regression
can be added later.

### Preprocessing CMIP6
- From the original files in `data/CMIP6/{historical,sspXXX}` (-> `/datalake/NS9873K/DATA/CMIP6`).
- Historical + scenario joined **per member at monthly resolution** before seasonal averaging
  (fixes the DJFM 2014 gap). Only members with both historical and scenario runs.
- Aggregate on the native grid first (DJFM for indices; annual, DJF, JJA for sfcWind),
  months weighted by their length in each model's own calendar, then calendar dropped; then
  regrid (bilinear for psl/sfcWind/tas, conservative for pr) to a global 1.5° grid, weights
  cached per source grid. Monthly regridded output optional (off by default).
- Members with gaps inside their monthly series are dropped and logged (no filling). Members that
  start later (e.g. EC-Earth3 r101+ from 1970) are kept with NaN before their start; each later step
  requires complete data only for the years it actually uses (e.g. the matching window).
- Historical runs are used until Dec 2014, scenario runs from Jan 2015 (some historical files
  extend beyond 2014, e.g. FGOALS-g3).
- Output: `/datalake/NS9873K/DATA/CMIP6/processed/` (currently empty).
- The archive state is part of the hashes: each model's regridded store is named
  `<model>_<key>.zarr`, with the key a hash of its members and their file names. New members or
  files give a new store (old stores stay), and only the changed models are regridded again.
  Downstream, the member list enters per scenario: the CMIP6 indices have their own output per
  scenario (`indices_cmip6/<hash>/`, separate from the observed indices in `indices/<hash>/`), and the
  split of a scenario depends on that scenario's psl and sfcWind member lists. A download for one
  scenario therefore leaves the hashes of the other scenarios unchanged. Run the preprocessing after
  a download has finished, not while files are still arriving.

### Output, safety, environment
- All other output: `/datalake/NS9873K/users/owul/climate_scenarios_v2/` (currently empty).
  Figures: `/projects/NS9873K/www/users/owul/climate_scenarios_2050/`. Both set in `configs/paths.yaml`.
- Every output path contains a short hash of the configuration that produced it (including
  upstream hashes) and a copy of the resolved config is written next to it. Existing files are
  never overwritten; large archives have an explicit `overwrite` flag (default false).
- Resources: `n_jobs`, dask threads/workers and chunk sizes from the config; defaults mirror the
  current scripts (member-wise processing, `n_jobs=-5`) since jobs run interactively.
- Environment managed by **pixi** (`pixi.toml`), versions pinned to the currently working
  conda env (python 3.10, xarray 2023.7, xeofs 3.0.3 + xarray-datatree, xesmf 0.8.8, ...).
  Old `environment.yml`/`requirements.txt`/`spec-file.txt` removed once pixi works.

## 3. Pipeline and modules

```
configs/
  paths.yaml            # inputs (read-only) and output roots
  default.yaml          # all parameters with defaults (see §4)
  experiments/*.yaml    # small overrides, e.g. analogue_rmse.yaml
src/climate_scenarios_2050/
  core/        config loading + hashing, paths, safe I/O (no-overwrite, config sidecar),
               year/season/window conventions, resource settings, logging
  preprocess/  regrid.py (CMIP6 originals -> aggregated 1.5° zarr)
               obs.py (HadSLP, 20CR, ERA5 SLP loaders, season completeness checks)
               indices/ (base interface; eof.py; box.py; jet.py, amv.py stubs)
               merge.py (HadSLP/ERA5 harmonization)
  split/       forced.py (MMEM / polynomial / low-pass; model + obs scaling; leave-out C)
  analogues/   filters.py, matching.py (measures, subsample, shift, two-stage, pre-selection),
               generate.py (hindcasts per init year), synthetic.py (AR(1) null obs)
  baselines/   nao.py (AR(1), FFT/sine, persistence, climatology), wind.py (persistence)
  linear_model/ fit.py (per window length, leave-out, saved per init/target)
  methods/     wind.py (M1-M3b, component bookkeeping, relative/absolute)
  evaluate/    metrics.py (deterministic, CRPS, Brier), significance.py (block bootstrap,
               phase randomization; from evaluate.py), nao.py, wind.py
  forecast/    forecast.py (any init year allowed by the data, all methods/scenarios)
  update/      update_index.py (new ERA5 winter + saved pattern -> extended merged index)
  plotting/
  legacy/      old scripts, kept for reference, not imported
```

Each step is run as `pixi run <step> --config configs/default.yaml [--config configs/experiments/x.yaml]`:

1. `preprocess` - regrid/aggregate CMIP6; observed fields; patterns; indices (obs + CMIP6).
2. `split` - forced/internal for CMIP6 indices, CMIP6 sfcWind (MMEM only), observed indices,
   observed wind.
3. `hindcast-nao` - analogue hindcasts (+ synthetic null); `evaluate-nao` incl. baselines.
4. `linear-model` - NAO -> wind (and energy) per window length, leave-out.
5. `hindcast-wind` + `evaluate-wind` - M1-M3b on the target windows.
6. `forecast --init-year Y` - all methods/scenarios, components stored.
7. `update-index` - append new winter(s) to the observed index using the frozen pattern and
   frozen harmonization.

## 4. Config schema (defaults)

```yaml
run:
  scenarios_hindcast: [ssp245, ssp370]
  scenarios_forecast: [ssp119, ssp126, ssp245, ssp370, ssp585]
  resources: {n_jobs: -5, dask_threads: 4, member_chunk: 20}
preprocess:
  grid: {spacing: 1.5}
  regrid_method: {psl: bilinear, sfcWind: bilinear, tas: bilinear, pr: conservative}
  sfcWind_aggregations: [annual, DJF, JJA]
  keep_monthly: false
  overwrite: false
predictor:
  name: nao
  variable: psl
  season: [12, 1, 2, 3]
  domain: {lon: [-90, 40], lat: [20, 80]}
  method: eof
  mode: 1
  normal_period: [1950, 2015]
  pattern_source: ERA5          # ERA5 | HadSLP | 20CR | own
  sign_reference_point: {lat: 35.25, lon: -20.25}
obs:
  matching_index: merged         # merged | HadSLP | 20CR | ERA5
  merge: {year: 1980, harmonization_period: [1940, 2018]}   # full ERA5/HadSLP overlap (complete seasons)
  training_index: ERA5
  wind: {variable: wsp100}       # wsp100 | energy
split:
  method: mmem                   # mmem | polynomial | lowpass_mmem
  weighting: model               # model | member
  polynomial_degree: 2
  lowpass: {cutoff_period: 5, order: 4}
  obs_leakage: leave_target_out  # leave_target_out (C) | full_period (A)
analogues:
  selection: subsample           # subsample | shift | shift_subsample
  measure: CORR                  # CORR | RMSE | two_stage
  filter: {type: running_mean, length: 41, min_periods_frac: 0.25}
  window: 70
  M: 100
  max_lag: 25
  preselection: null
  model_filter_at_init: full     # full (default, as before) | cut (same as obs)
  synthetic_null: {enabled: true, n: 500}
target:
  windows: [[1,5],[6,10],[11,15],[16,20],[21,25],[26,30],[1,10],[11,20],[21,30],[1,20],[11,30]]
  reference_length: 20
  nao_wind_lag: 1
  change: relative               # relative | absolute
linear_model:
  leave_out: overlap_target      # exclude training windows overlapping the target window
forced_wind: nao_mmem            # nao_mmem (a) | cmip6_sfcwind (b) | none (c)
evaluation:
  quantiles: [0.5, 0.9]
  bootstrap: {n: 1000, method: moving_block}
  nao_continuous: {running_mean: 8, lead_times: [1, 33]}
```

## 5. Bugs found in the existing code (fixed in the new code)

1. **Analogue ranking (`generate_NAO_hindcast.ens_subsample`, also `generate_synthNAO_hindcast`)**:
   members are "re-centred" by subtracting the single-model ensemble mean **per year**. For the
   14 single-member models this gives zeros -> correlation NaN, and `sortby(..., ascending=False)`
   puts NaN **first**. Verified on `hc_cmip6-ssp370_None_sync_HadSLP_bpy20-50_70yvf_25ymx`:
   all 14 single-member models are in every analogue set and fill 9-10 of the top 10 places.
   For models with few members the per-year subtraction also removes most of each member's
   own variability. Fix: centre each member by its time mean over the matching window; drop
   NaN scores before ranking.
2. Same per-year subtraction in `wind_pred_functionality.get_ranking`; there NaN scores are
   dropped, so single-member models are never selected.
3. Not a bug, but now an explicit option (`analogues.model_filter_at_init`): obs are cut at init and filtered
   with `min_periods` (one-sided at the end) while model series are filtered over their full length.
   No observed future is used, and the runs' future exists at forecast time as well. Default: `full`
   (as before). `cut` (model filtered like the obs) gives no NAO skill, see §8.
4. `evaluate_full_synth.py`: observed NAO is smoothed with a trailing running mean
   (`rolling(year=Nyr)`, not centred) while hindcasts use centred means -> 4-year offset in the
   printed correlations, plots and the obs-based Brier thresholds.
5. DJFM 2014 has no real value: historical runs end Dec 2014 and `cut_dataarray` keeps seasons
   up to `end-1`; the gap is linearly interpolated in `split_variability.py`.
6. Observed split files (`internal/forced-NAO-variability_f-None.nc`) are overwritten for
   every scenario in the loop; the scenario is not in the file name.
7. `cmip6_nao.py`: fallback EOF fit on 1970-2015 saved under the `_1950-2015_` name;
   `recompute_EOFs=True` silently refits every pattern.
8. Partial seasons are averaged as if complete (ERA5 1939 = Jan-Mar 1940 only; an incomplete
   last winter).
9. `split_variability.py`: `xr.cov` (ddof=1) divided by `.var()` (ddof=0).
10. `filters.butter_lowpass_filter`: `isinstance(pad,int) & pad>0` evaluates as
    `(isinstance(pad,int) & pad) > 0`, so even `pad` values are rejected; with `pad` and
    `nancut` the NaN mask is computed on the unpadded series (length mismatch).
11. `nao_ar1_lofreq_forecast.py`: the AR(1) baseline is fitted to the 8-year running mean
    (overlapping, smoothed) series rather than the annual series. New default: fit AR(1) on the
    annual internal NAO up to init, simulate, then apply the same window means; old variant
    kept as option.
12. `wind_pred_functionality.py`: slices integer years with strings, reads the old parameter
    file name, and `compute_change_future_reference` has undefined outputs when the reference
    has no `model_id` dimension.
13. `verify_wind_hindcasts.py`: the climatological spread is in-sample -> leave-out estimate. Brier
    thresholds stay one fixed in-sample value per window: leave-out thresholds reveal the sample's side for
    slowly varying series (a no-skill ensemble reached a Brier skill score of 0.89).

Earlier analogue skill results may change noticeably after fix 1.

## 6. Old -> new mapping

| Old script | New location |
|-----------|--------------|
| `cmip6_nao.py`, `compute_NAO.py`, `interpolate.py`, `regrid_model_fields.py`, `harmonize_single_var_mme.py` | `preprocess/` |
| `merge_NAO.py`, `update_merged_NAO.py` | `preprocess/merge.py`, `update/update_index.py` |
| `split_variability.py`, `split_variability_wrefper.py`, `variability_split.py` | `split/`, `evaluate/forced_share.py` |
| `generate_NAO_hindcast.py`, `generate_synthNAO_hindcast.py`, `subsampling_list.py`, `AR_model.py` (pre-selection list) | `analogues/` |
| `nao_ar1_lofreq_forecast.py`, `nao_fft_forecast.py` | `baselines/nao.py` |
| `evaluate.py` (bootstrap), `evaluate_full_synth.py` | `evaluate/significance.py`, `evaluate/nao.py` |
| `linear_model_wsp_NAO.py` | `linear_model/fit.py` |
| `verify_full_pred_wsp.py`, `verify_wind_hindcasts.py` | `methods/wind.py`, `evaluate/wind.py` |
| `wind_pred_functionality.py` | `forecast/forecast.py` |
| `filters.py`, `spectral_analysis.py`, `utils.py` (parts) | `analogues/filters.py`, `core/` |
| plotting scripts, notebooks, `figure_presentation.py`, field hindcast scripts, `cmip_retrieval/`, `update_era5.py` | `plotting/` or `legacy/` (download scripts unchanged) |

## 7. Open points (not blocking)

- Harmonization overlap period: HadSLP ends Dec 2019 (last complete DJFM 2018), ERA5 starts Jan 1940
  (first complete DJFM 1940) -> default 1940-2018. 20CRv3 ends Dec 2015 (last DJFM 2014).
- Observational SST dataset for AMV and download of CMIP6 `ua` for jet indices (later).
- Whether the NAO season Y (observed at init) should replace the analogue members' values in
  the NAO window for wind window 1-5 (default: no, use model values consistently).

## 8. Results so far (NAO hindcasts, Oct 2026)

Correlation of 8-year running means at lead 5/10/15/20 years, ssp245: analogue with `full` filter 0.48/0.40/0.34/0.19,
with `cut` filter -0.06/0.09/0.12/0.12, persistence of the last 20 winters 0.43/0.29/0.09/-0.12, synthetic null p95
about 0.38. The `full` filter works as a constraint of the members' next ~20 years to the recent observed state.
