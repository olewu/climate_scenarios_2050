# To-do

Checked items stay in the list until they are removed after agreement. `(you)` = runs/decisions on your side.

## Running / on your side
- [ ] (you) CMIP6 download, ssp245 + ssp370 first: `pixi run cmip6-download --scenario ssp245 --scenario ssp370` (~385 GB), then ssp119/ssp126/ssp585 (~730 GB); rerun `pixi run cmip6-inventory` afterwards
- [ ] (you) wind evaluation with the significance test (`pixi run evaluate-wind`, ~1.7 h per scenario/variable case)
- [ ] (you) reconstruct the April 2026 hindcast
- [ ] (you) decide which old runs/data to remove before restarting the final experiments
  (`data/CMIP6_processed` 342 GB and `data/CMIP6_hindcast` 59 GB are only used by `legacy/`)

## 1. CMIP6 archive
- [x] new `cmip_retrieval` (inventory over the CEDA/DKRZ/ORNL index nodes, plan, per-file download with checksum and resume, `ua` as 850 hPa / 10-80°N OPeNDAP subset); old scripts moved to `legacy/`
- [x] member list in the hashes: model stores `<model>_<key>.zarr` (key over members + file names), CMIP6 indices
  per scenario (`indices_cmip6/`), split depends on the scenario's psl/sfcWind member lists (all hashes new)
- [ ] (you) after the download: `pixi run preprocess`, then the steps from `split` on (everything recomputes once)
- [ ] preprocessing reads the `ua_plev850_lat10-80` subsets
- [ ] jet speed / jet latitude index from `ua` at 850 hPa
- [ ] export step for the cmip6-api stores in `wind_db` (same 1.5° grid and chunking, new store name, then switch the
  s3 link); **decision**: which variables and scenarios the API serves
- [ ] add `requests` to `pixi.toml` (now only a transitive dependency)

## 2. Temperature and precipitation as targets
- [ ] rename `wind_variable` -> `target_variable` (new hashes accepted)
- [ ] ERA5 annual means of tas and pr as observations
- [ ] change type per target (tas absolute, pr relative, wind as now)
- [ ] CMIP6 tas/pr (annual) in the preprocessing for M2a/M3a; mapping observed -> CMIP6 variable (energy has none)

## 3. Several indices
- [ ] multi-linear model (e.g. NAO + EA), optional ridge shrinkage; NAO-only stays the default
- [ ] joint analogue matching on several indices (weighted mean of the scores, new parameter)
- [ ] index evaluation for every index in the hindcast

## Smaller points
- [ ] commit the current work on `restructure_project` (nothing committed since 4235d45)
- [ ] index files carry the psl attributes (`units: Pa`)
- [ ] `obs_wind` hash uses the input path, not the content (declined for now)
- [ ] plain `pytest` fails on the `--cov` options in `setup.cfg` (`pixi run test` works)
- [ ] regenerate the figures with the new plot code (evaluate-nao, linear-model, forecast)

## Done since the last commit
- [x] AR(1)/sine baselines: `fit_window: matching` follows `analogues.window` (hashes unchanged)
- [x] surrogate significance test of the wind skill maps (local FDR + field significance)
- [x] NAO figures in the old design (skill with DCPP marker, hindcast overview), forecast violins
- [x] linear-model maps as slope x std of the window-mean index
- [x] linear model: missing wind coefficient files computed on demand
