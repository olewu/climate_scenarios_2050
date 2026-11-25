# Restructuring design: decadal wind prediction from CMIP6 NAO analogues

Status (Nov 2026): all steps implemented (CMIP6 inventory/download, preprocessing, split, index hindcast and
evaluation, linear model, target hindcast and evaluation, forecast, index update); predictors NAO/EA (EOF),
station NAO (box), jet latitude/speed, SST/tas-based EOF indices (PDO) and area means (AMV from tas, DCENT).

## 1. Goal

Predict the change of 100 m wind speed (optionally wind energy) over Europe for the next
30 years, relative to the 20 years before initialization, and verify the methods in hindcast
mode without leaking future information (as far as the short record allows).

Forecast methods (all per CMIP6 scenario):

| ID | Method | Wind from | Needs linear model |
|----|--------|-----------|--------------------|
| M0 | Return to the long-term mean: the internal anomaly of the last 20 years vanishes (+ forced part) | observations (internal reference) | no |
| M1 | Persistence of the last 20 years (zero change) | observations | no |
| M0c | Observations only: return to the observed mean of the years up to init | observations | no |
| M0t | Observations only: observed linear trend up to init, read at the target window | observations | no |
| M2a | All CMIP6 members, sfcWind directly | CMIP6 sfcWind | no |
| M2b | All CMIP6 members, NAO -> linear model | CMIP6 NAO | yes |
| M3a | NAO-analogue members, sfcWind directly | CMIP6 sfcWind | no |
| M3b | NAO-analogue members, NAO -> linear model | CMIP6 NAO | yes |
| perfect_index | Linear model given the observed index in the target window (upper limit, not a forecast) | observed index | yes |

Data each method needs for its prediction (NAO/sfcWind/wsp100 as the example; other indices and targets
analogously). "Observed target" = the monthly target record (e.g. GWA-scaled ERA5 wsp100, 1940-); "observed
index (training)" = `obs.training_index` (ERA5, consistent with the target); "observed index (matching)" =
`obs.matching_index` (long record, e.g. HadSLP/ERA5 merged); "CMIP6 index" = the index of every member
(historical + scenario), its MMEM giving the forced part of the index; "CMIP6 target variable" = the target's
CMIP6 field (`targets.<name>.cmip6`, e.g. sfcWind) of every member.

| ID | Observations | CMIP6 | Forced part (full target) |
|----|--------------|-------|---------------------------|
| M0 | observed target (internal reference anomaly; spread: RMS of its own errors, i.e. of the internal window means) | CMIP6 index MMEM or CMIP6 target-variable MMEM (for the split, see right) | `a`: index MMEM regression (`forced: index`) or scaled MMEM of the CMIP6 target variable (`forced: cmip6`) |
| M1 | observed target (reference mean; spread: RMS of the observed changes) | - | - (no change) |
| M0c | observed target (mean of the years up to init and reference mean; spread: RMS of its own errors) | - | constant (observed mean) |
| M0t | observed target (linear trend fit up to init and reference mean; spread: RMS of its own errors) | - | observed linear trend |
| M2a | - | CMIP6 target variable, all members | in the members' own change |
| M3a | observed index (matching) | CMIP6 index (selection) + CMIP6 target variable of the selected members | in the members' own change |
| M2b | observed target + observed index (training) for the linear model | CMIP6 index, all members | `a` (default), or `b`: CMIP6 target-variable MMEM change (needs the CMIP6 target variable) |
| M3b | as M2b, plus observed index (matching) | CMIP6 index (selection and predictor) | as M2b |
| perfect_index | observed target + observed index (training), incl. the index in the target window | - | `a` |

Every method is verified against the observed target; the `internal` target and M0 also need the forced
estimate of the split (CMIP6 index MMEM, or the CMIP6 target-variable MMEM for `forced: cmip6`). Wind energy has
no CMIP6 variable: M2a/M3a, option `b` and `forced: cmip6` are not available for it.

M2b vs M3b isolates the value of the analogue selection; M3a vs M3b isolates the value of the
statistical model vs model wind. Wind energy is only possible for M0, M1, M2b, M3b.

M0c and M0t (`evaluation.obs_methods`, full target only) are the cheapest references: M0 with the forced part
estimated from the observed target alone, a constant (M0c) or a linear trend (M0t), fitted on the years up to init
(`fit: up_to_init`, a real forecast; `leave_out`: all years except target window and reference, as M0). They show
whether the CMIP6-based methods are worth their data. Both are linear in the observations, so the surrogate test
(beyond M0) rebuilds them for every surrogate; `significance_m1.nc` tests every method against M1 (paired block
bootstrap of the score differences over the init years).

M0 is the baseline without index information, and the null hypothesis of the significance test. The internal
part of the target has zero mean over the fitting years, so the expected internal level of the target window is
zero: the predicted internal change is minus the observed internal anomaly of the reference period (the 20 years
before init, from the leave-out split without the target window and reference). On the full target the forced
change is added, the same forced part `a` as for M2b/M3b (the forced part of the target's split:
`targets.<name>.forced: index` regression on the index MMEM, `cmip6` the scaled MMEM of the target's CMIP6
variable). M0 thus differs from M1 (persistence) by assuming complete reversion of the recent internal anomaly.
Its spread is the RMS of its own errors over the non-overlapping samples (`evaluation.m0_spread: own_errors`), as
for M1, M0c and M0t, so that all no-information references are calibrated the same way. M0's error is the observed
internal window mean (it predicts internal = 0; the forced part cancels on the full target), so this is the typical
size of the internal window means, estimated with the same leave-out fits as the evaluation. Until 2026-11-24 the
evaluation used the linear model's residual spread σ_LM, which leaves out the index-explained variance (too narrow:
1.15 % instead of 1.76 % for wsp100, window 1-10, and a too low CRPSS of M0); the forecast used
√(σ_LM² + β′ var(W[I]) β). The forecast now uses the RMS of M0's errors at earlier, complete windows.
Skill beyond M0 (dcorr, dcrpss, corr_anom) is the skill a method gets from predicting the departure from this
reversion, i.e. from the index (M2b/M3b) or from the selected members' own change (M3a). In the forecasts the
forced part of M0 follows `forced_target` (index_mmem | cmip6_mmem | none), as for M2b/M3b.

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
- **Forced part of the observed targets** (`targets.<name>.forced`): `index` (default; wind, energy) regresses
  the target on the predictor's MMEM (season t-1 for year t), as designed for wind/energy (no CMIP6 variable
  needed, and the "b" methods would make sfcWind projections unnecessary). `cmip6` (tas, pr) regresses each grid
  point on the MMEM of the target's own CMIP6 variable (year t, regridded to the target grid, regressor centred),
  with the same leave-out cases. With `index`, tas kept almost all of its warming in the "internal" part
  (Europe, 1-10: internal reference means -0.7 K in 1960-85 rising to +0.8 K in 2015), which made persistence
  beat M0 and gave M3a spurious internal skill; with `cmip6` the internal means stay within +-0.15 K.
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
Generic index interface (`predictor`): an index = (variable, aggregation, domain, method), computed
for the observational datasets `predictor.obs_datasets` (read as described in `obs_sources`) and
all CMIP6 members. Methods: EOF mode k projected on a common pattern, box difference (station NAO),
jet latitude/speed (`ua` at 850 hPa, see below), area mean (`method: amv`, see below). NAO-only parts are optional settings of the
predictor or of `obs`: the HadSLP/ERA5 merge (`obs.merge`, null to switch off), the DCPP marker
(`predictor.reference_skill`), the preselection lists.
- SST predictors (e.g. PDO): `tos` from the CMIP6 table Omon (`cmip6.tables`); land points (NaN in
  all samples) are left out of the EOF and the projection; `predictor.remove_global_mean` subtracts
  the area-weighted global mean (stored with the field, `preprocess.fields.<var>.global_mean`).
- Domains may cross the date line (`lon: [west, east]` with west > east, e.g. [110, -100]); their
  fields use longitudes 0 ... 360 (`core/domain.py`).
- Fields are named entries of `preprocess.fields` (`variable` defaults to the name), so one variable
  can be a predictor and a target with different domains (`predictor.field`, e.g. `tas_np`).
- `predictor.mask: ocean` restricts the index field to the ocean, `global_mean: ocean` the global
  mean (mask from ERA5 SST, nearest neighbour on each grid, also on the models' native grids).
- Examples: `configs/experiments/south_america_pdo.yaml` (needs CMIP6 tos and a long SST record);
  `configs/experiments/pdo_from_tas.yaml`, a PDO-like index from tas over the ocean as a test with
  existing data (ERA5: corr. 0.96 with the PDO from ERA5 SST, 1940-2025).

### Several indices
- `predictor.indices: {name: {overrides}}` defines several indices on the base `predictor` block (e.g.
  NAO and EA as modes 1 and 2, `configs/experiments/nao_ea.yaml`). `index_cfg(cfg, name)` gives the
  configuration of one index; patterns, indices, split and index evaluation run per index and hash like a
  single-index configuration, so an index shared by several configurations is computed once. The first
  index is the primary one: its split removes the forced part of the targets.
- Sign of a pattern: positive at `sign_reference_point`, or negative there with `sign: negative` in the
  point (EA in the CPC convention: `{lat: 55, lon: -20, sign: negative}`, EA+ = low west of the British Isles).
- Jet indices (`predictor.method: jet`, `configs/experiments/jet.yaml`): latitude and speed of the maximum of
  the DJFM 850 hPa zonal wind averaged over 0-60W, within 15-75N (Woollings et al. 2010 on seasonal means
  instead of low-pass filtered daily data). Computed on the common 1.5 deg grid like the other indices
  (native and regridded agree to corr 0.99 for the latitude, 1.00 for the speed), with a parabola through
  the maximum for a continuous latitude. Anomaly indices: the own normal-period mean is removed (with CORR
  matching an offset does not change the selection, and the targets are changes against each member's
  own reference); the raw mean is stored as `raw_climatology` for comparing the models' jet with ERA5.
  Below-ground points (850 hPa under Greenland and the Atlas) are missing in about half of the models and
  extrapolated in ERA5/20CR: a fixed mask (`predictor.jet.mask`, `configs/masks/ua850_below_ground.nc`,
  points missing in >= 3 of 49 models; its checksum enters the hashes) removes them from every dataset, and a
  latitude needs 80 % of its unmasked longitudes (`min_valid`). The observed indices are unchanged by the
  mask (the jet is never over Greenland in DJFM means).
- Value checks of the native data (`cmip6.VALUE_CHECKS`, recorded in the store hash; so far `ua`): 0 and
  1e20 count as missing, points with |u| > 150 m/s are corrupt (NaN; a member with a time step of > 1 % such
  points is dropped). The OPeNDAP subsets from esgf.nci.org.au contain all-zero chunks and ~1e37 garbage
  (71 of 51597 files); full files are checked against their checksum.
- Analogues: matched jointly on `analogues.match_weights` (weighted mean of the per-index correlations,
  or root of the weighted mean of the squared errors for RMSE; one lag per member for all indices). The
  pool are the members with all needed indices; the hindcast stores every needed index (`index_<name>`).
  The synthetic null test is only run for matching on a single index.
- Linear model: `linear_model.predictors`, multiple regression on the window means, optional ridge
  (`linear_model.ridge`, penalty relative to the predictors' variance). M2b/M3b: mean
  sum_p slope_p x_p, spread^2 = slope' C slope + sigma^2 with C the covariance of the members' window
  means.
- Settings for several indices only enter the hashes when used (`core.io.HASH_NEUTRAL`).

### Regions and targets
- `region`: domain of the prediction targets, the active targets and the sites. The CMIP6 fields
  of the targets use `domain: region` (resolved when the config is loaded).
- `targets`: registry of prediction targets, each with its observations (`source: file` for prepared
  files such as the GWA-scaled wind, or a dataset of `obs_sources`, e.g. ERA5 tas/pr/10 m wind),
  its CMIP6 variable for M2a/M3a and option (b) (`cmip6`, null for energy) and its `change`
  (relative % or absolute). Observed targets are annual means on their own grid, cut to the region.
- The CMIP6 data of a target enter only in `hindcast-target` (per scenario and target, with the
  MMEM for option b) and the forecast; the split and the index steps depend on the predictor only.

### Preprocessing CMIP6
- From the original files in `data/CMIP6/{historical,sspXXX}` (-> `/datalake/NS9873K/DATA/CMIP6`).
- Historical + scenario joined **per member at monthly resolution** before seasonal averaging
  (fixes the DJFM 2014 gap). Only members with both historical and scenario runs.
- Only the fields the configuration uses are processed (predictor variable and the targets' CMIP6
  variables).
- Aggregate on the native grid first (DJFM for indices; annual, DJF, JJA for sfcWind),
  months weighted by their length in each model's own calendar, then calendar dropped; then
  regrid (bilinear for psl/sfcWind/tas, conservative for pr) to a global 1.5° grid, weights
  cached per source grid. Monthly regridded output optional (off by default).
- Only members whose historical + scenario files cover every month of `cmip6.full_period`
  (1850-2055) by their file names are used; late historical starts (EC-Earth3 r101+ from 1970, GISS
  r30x from 1990) and short scenario runs are left out. The variables of `cmip6.common_members`
  (psl, pr, sfcWind, tas, ua) use only the members complete in all of them, so the predictor and all
  target fields share one member list per scenario (2026-11-22: ssp119 204 members / 17 models, ssp126
  292/43, ssp245 417/44, ssp370 417/43, ssp585 358/48). Members with gaps in the data read are dropped
  and logged as well (no filling).
- Historical runs are used until Dec 2014, scenario runs from Jan 2015 (some historical files
  extend beyond 2014, e.g. FGOALS-g3).
- Output: `/datalake/NS9873K/DATA/CMIP6/processed/` (currently empty).
- The archive state is part of the hashes: each model's regridded store is named
  `<model>_<key>.zarr`, with the key a hash of its members and their file names. New members or
  files give a new store (old stores stay), and only the changed models are regridded again.
  Downstream, the member list enters per scenario: the CMIP6 indices have their own output per
  scenario (`indices_cmip6/<hash>/`, separate from the observed indices in `indices/<hash>/`); the
  split of a scenario depends on that scenario's predictor members, and each target's hindcast inputs
  on the members of its CMIP6 variable. A download for one scenario or one target variable (outside `cmip6.common_members`) therefore
  leaves the other hashes unchanged. Run the preprocessing after
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
               indices/ (compute.py interface; eof.py; box.py; jet.py)
               merge.py (HadSLP/ERA5 harmonization)
  split/       forced.py (MMEM / polynomial / low-pass; model + obs scaling; leave-out C)
  analogues/   filters.py, matching.py (measures, subsample, shift, two-stage, pre-selection),
               generate.py (hindcasts per init year), synthetic.py (AR(1) null obs)
  core/domain.py  domains, incl. crossing the date line
  cmip_retrieval/ esgf.py, inventory.py, download.py (CMIP6 inventory and download)
  baselines/   index.py (AR(1), FFT/sine, persistence)
  linear_model/ fit.py (per window length, leave-out, saved per init/target)
  methods/     target.py (CMIP6 inputs of M2a-M3b per target)
  evaluate/    metrics.py (deterministic, CRPS, Brier), significance.py (block bootstrap,
               phase randomization), surrogates.py, index.py, target.py
  forecast/    forecast.py (any init year allowed by the data, all methods/scenarios)
  update/      update_index.py (new ERA5 winter + saved pattern -> extended merged index)
  plotting/
  legacy/      old scripts, kept for reference, not imported
```

Each step is run as `pixi run <step> --config configs/default.yaml [--config configs/experiments/x.yaml]`:

0. `cmip6-inventory`, `cmip6-download` - CMIP6 archive.
1. `preprocess` - regrid/aggregate CMIP6; observed fields; observed targets; patterns; indices.
2. `split` - forced/internal for CMIP6 indices, observed indices, observed targets.
3. `hindcast-index` - analogue hindcasts (+ synthetic null); `evaluate-index` incl. baselines.
4. `linear-model` - index -> target per window length, leave-out.
5. `hindcast-target` + `evaluate-target` - M0-M3b on the target windows, per target.
6. `forecast --init-year Y` - all methods/scenarios, components stored.
   `partition` (any time after `split`) - variance partition of the projected window changes into internal
   variability, model and scenario uncertainty (Hawkins & Sutton 2009) per lead, for every index and for area
   means of the targets' CMIP6 fields over `partition.regions` (`partition.py`; one folder per index or
   target/region, hash of `partition`, the split hashes or the CMIP6 field ids, `target.reference_length`).
7. `update-index` - append new winter(s) to the observed index using the frozen pattern and
   frozen harmonization.

`pixi run outputs [--config ...] [--step <step>] [--scenario ...] [--target-variable ...]` lists the output
folders (hashes) of a configuration per step and case, and whether they exist (and have figures); it only
computes hashes and writes nothing.

`pixi run gallery [--config ...] [--name NAME]` registers a configuration ("experiment") on the comparison pages
in `<figures>/gallery/` (https://ns9873k.web.sigma2.no/users/owul/climate_scenarios_2050/gallery/). It exports
each evaluate-target folder once into one binary file (int16 maps, significance bit masks) plus a JSON index;
the page fetches only the bytes of the maps shown (HTTP range requests) and draws them in the browser
(experiments side by side, per-column method/window, difference column, values on hover, view in the URL).
Exports are named after the folder hash (shared between experiments) and never rewritten; a folder whose
significance test arrives later gets a second export (`_ss` next to `_s`). Three pages:
- `index_skill.html`: index evaluation per experiment and index (skill per lead time with bootstrap bands and the
  synthetic-null p95, window scores, hindcast overview by init and valid year); side by side or overlaid in one
  chart. Exported as plain JSON per folder (a few 10 kB).
- `linear_model.html`: maps of the linear model (skill given the observed index change; target change per 1 std
  of the window-mean index, per predictor, in target units and %), with a difference column.
- `target.html`: target skill maps (scores and statistics beyond M0 with significance dots).
- `partition.html`: variance partition per experiment and quantity (stacked fractions, standard deviations per
  source, multi-model mean change per scenario with the internal spread), per window length.
The index page also shows the CRPS of the window changes and the CRPSS against `cmip6_all`, and per lead time the
CRPSS against `cmip6_all` with a bootstrap band (`continuous_crps.nc` of evaluate-index: a separate file, so that
existing folders get it on a rerun of the step without recomputing the rest).

## 4. Config schema

`configs/default.yaml` is the reference (commented). Main blocks: `cmip6` (scenarios, weighting,
tables, minimum scenario end), `download`, `region`, `targets`, `obs_sources`, `preprocess.fields`,
`predictor`, `obs` (matching/training index, merge, default forecast target), `split`, `analogues`,
`baselines`, `target` (windows, reference, lag), `linear_model`, `forced_target`, `evaluation`,
`update`. The config is validated when it is loaded (unknown targets, datasets without the predictor
variable, missing fields, ...).

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
| `verify_full_pred_wsp.py`, `verify_wind_hindcasts.py` | `methods/target.py`, `evaluate/target.py` |
| `wind_pred_functionality.py` | `forecast/forecast.py` |
| `filters.py`, `spectral_analysis.py`, `utils.py` (parts) | `analogues/filters.py`, `core/` |
| plotting scripts, notebooks, `figure_presentation.py`, field hindcast scripts, `cmip_retrieval/`, `update_era5.py` | `plotting/` or `legacy/` (download scripts unchanged) |

## 7. Open points (not blocking)

- Harmonization overlap period: HadSLP ends Dec 2019 (last complete DJFM 2018), ERA5 starts Jan 1940
  (first complete DJFM 1940) -> default 1940-2018. 20CRv3 (`prmsl`, `uwnd`) ends Dec 2015 (last DJFM 2014).
- AMV index (`method: amv`, `configs/experiments/amv.yaml`): annual-mean tas over the North Atlantic ocean
  (0-60N, 80W-0, ERA5 land-sea mask), cos-lat weighted mean of the anomalies, standardized with the DCENT
  normal-period std (0.29 K). No detrending or low-pass filter in the index: the forced part is removed in
  the split like for the NAO (each model mean and DCENT regressed on the AMV MMEM, y = c0 + c1 * MMEM, same
  leave-out cases), the analogue matching filters. Variant: `predictor.remove_global_mean: true` (Trenberth
  and Shea 2006; the field stores the global mean over the ocean). Observations: DCENT ensemble mean
  (`obs_sources.DCENT`, blended `temperature`), annual means of a point from >= 6 valid months
  (`min_months`; complete years only would leave ~40 % of the box before 1900), nearest-neighbour regridding
  (`regrid`; bilinear spreads the gaps); a year needs data at half the box area (`predictor.amv.min_coverage`),
  which leaves 1861-1872 missing (coverage 40-47 %). Checked in memory: r = 0.999 with an AMV on the native
  5 deg grid; MPI-ESM1-2-LR members follow DCENT's decadal course (cool 1860s/1970s, warm since 2000).
  DCENT is too sparse in the North Pacific for an EOF before ~1950 (PDO: an infilled product).
- Whether the NAO season Y (observed at init) should replace the analogue members' values in
  the NAO window for wind window 1-5 (default: no, use model values consistently).

## 8. Results so far (NAO hindcasts, Oct 2026)

Correlation of 8-year running means at lead 5/10/15/20 years, ssp245: analogue with `full` filter 0.48/0.40/0.34/0.19,
with `cut` filter -0.06/0.09/0.12/0.12, persistence of the last 20 winters 0.43/0.29/0.09/-0.12, synthetic null p95
about 0.38. The `full` filter works as a constraint of the members' next ~20 years to the recent observed state.
