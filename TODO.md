# To-do

Checked items stay in the list until they are removed after agreement. `(you)` = runs/decisions on your side.

## Running / on your side
- [ ] (you) CMIP6 download ssp245 + ssp370 (running since 2026-11-14 01:08 with the plan `cmip6_inventory/20261114-002725`,
  incl. `min_scenario_end: 2054`; order pr, psl, sfcWind, tas, ua; its log appears only at the end - fixed for later runs)
- [ ] (you) when it has finished: `pixi run cmip6-download --scenario ssp245 --scenario ssp370` once more to retry the
  failed files (existing files are skipped)
- [ ] (you) then the remaining scenarios: `pixi run cmip6-download --scenario ssp119 --scenario ssp126 --scenario ssp585` (~730 GB);
  rerun `pixi run cmip6-inventory` afterwards
- [ ] (you) target evaluation with the significance test (`pixi run evaluate-target`, ~1.7 h per scenario/target;
  with tas and pr now 8 cases)
- [ ] (you) reconstruct the April 2026 hindcast
- [ ] (you) decide which old runs/data to remove before restarting the final experiments
  (`data/CMIP6_processed` 342 GB and `data/CMIP6_hindcast` 59 GB are only used by `legacy/`)

## 1. CMIP6 archive
- [x] new `cmip_retrieval` (inventory over the CEDA/DKRZ/ORNL index nodes, plan, per-file download with checksum and resume, `ua` as 850 hPa / 10-80°N OPeNDAP subset); old scripts moved to `legacy/`
- [x] member list in the hashes: model stores `<model>_<key>.zarr` (key over members + file names), CMIP6 indices
  per scenario (`indices_cmip6/`), split depends on the scenario's predictor members, `hindcast-target` on the
  members of the target's CMIP6 variable (all hashes new)
- [ ] (you) after the download: `pixi run preprocess`, then the steps from `split` on (everything recomputes once)
- [ ] preprocessing reads the `ua_plev850_lat10-80` subsets
- [ ] jet speed / jet latitude index from `ua` at 850 hPa
- [ ] export step for the cmip6-api stores in `wind_db` (same 1.5° grid and chunking, new store name, then switch the
  s3 link); **decision**: which variables and scenarios the API serves
- [ ] add `requests` to `pixi.toml` (now only a transitive dependency)

## 2. Temperature and precipitation as targets
- [x] rename `wind_variable` -> `target_variable` (also `paths.inputs.targets`, `obs_target` step, `--target-variable`, `--reference-value`)
- [x] steps/modules renamed: `hindcast-target`, `evaluate-target` (`methods/`, `evaluate/`, `plotting/target.py`), `--part targets`
- [x] ERA5 annual means of tas and pr as observations (checked on the real archive: 1940-2025)
- [x] change type per target (tas absolute, pr relative, wind as now)
- [x] CMIP6 tas/pr (annual) in the preprocessing for M2a/M3a; mapping observed -> CMIP6 variable (energy has none)
- [x] `region` and `targets` blocks (default targets: wsp100, energy, tas, pr); the target CMIP6 data moved out of
  the split into `hindcast-target` (per target), so a tas download no longer changes the index steps

## 3. Several indices
- [ ] multi-linear model (e.g. NAO + EA), optional ridge shrinkage; NAO-only stays the default
- [ ] joint analogue matching on several indices (weighted mean of the scores, new parameter)
- [ ] index evaluation for every index in the hindcast

## 4. Other regions and predictors (code in place, data later)
- [x] rename `nao` -> `index` steps/modules (`hindcast-index`, `evaluate-index`, `perfect_index`, `forced_target`)
- [x] SST predictor: `tos` (Omon), land-masked EOF, `remove_global_mean`, domains across the date line, optional
  `obs.merge`, configurable reference skill; template `configs/experiments/south_america_pdo.yaml`
- [ ] (you) download an SST reconstruction (e.g. DCENT); then add it to `obs_sources`, `paths.inputs` and the PDO config
- [ ] (you) download CMIP6 tos (add to `download.variables`, rerun the inventory)
- [x] fields named independently of their variable (`predictor.field`), ocean mask from ERA5 SST (`predictor.mask`,
  `global_mean: ocean`), test experiment `pdo_from_tas.yaml` (ERA5: corr. 0.96 with the PDO from ERA5 SST)
- [ ] (you) PDO-from-tas test run: `--config south_america_pdo.yaml --config pdo_from_tas.yaml` (preprocess tas_np, then
  the steps); hindcast skill needs a longer record than ERA5 (waits for DCENT or similar)
- [ ] first real PDO run with tos: check the regridding of curvilinear ocean grids (only tested on synthetic data)

## Smaller points
- [x] commit the current work on `restructure_project` (586f930)
- [ ] index files carry the psl attributes (`units: Pa`)
- [ ] `obs_target` hash uses the input path, not the content (declined for now)
- [ ] plain `pytest` fails on the `--cov` options in `setup.cfg` (`pixi run test` works)
- [ ] regenerate the figures with the new plot code (evaluate-index, linear-model, forecast)

## Done since the last commit
- [x] AR(1)/sine baselines: `fit_window: matching` follows `analogues.window` (hashes unchanged)
- [x] surrogate significance test of the wind skill maps (local FDR + field significance)
- [x] NAO figures in the old design (skill with DCPP marker, hindcast overview), forecast violins
- [x] linear-model maps as slope x std of the window-mean index
- [x] linear model: missing wind coefficient files computed on demand
