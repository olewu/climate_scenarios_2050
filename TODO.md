# To-do

Checked items stay in the list until they are removed after agreement. `(you)` = runs/decisions on your side.

## Runs on your side
- [ ] (you) `pixi run preprocess` with the current archive, then the steps from `split` on (everything recomputes once)
- [ ] (you) first NAO + EA run (`--config configs/experiments/nao_ea.yaml`); only the observed side is checked on real
  data so far (NAO 48 %, EA 15 % of the ERA5 DJFM SLP variance)
- [ ] (you) target evaluation with the significance test (`pixi run evaluate-target`, ~1.7 h per scenario/target;
  with tas and pr 8 cases)
- [ ] (you) reconstruct the April 2026 hindcast
- [ ] regenerate the figures with the new plot code (evaluate-index, linear-model, forecast) - comes with the rerun
- [ ] (you) decide which old runs/data to remove before the final experiments (`data/CMIP6_processed` 342 GB and
  `data/CMIP6_hindcast` 59 GB are only used by `legacy/`; old inventories in `cmip6_inventory/` 20261113, 20261114,
  20261114-002725, 20261122 (no plan))

## 1. CMIP6 archive
State after the inventory `20261122-163805`: 18-61 models per variable/scenario, 1-5 members per variable/scenario
still missing except ssp126 (~30).
- [x] **decision**: only members complete 1850-2055 (`cmip6.full_period`, replaces `min_scenario_end`; drops the late
  historical starts) and common to psl, pr, sfcWind, tas, ua (`cmip6.common_members`): ssp119 204 members / 17 models,
  ssp126 292/43, ssp245 417/44, ssp370 417/43, ssp585 358/48. Lost models (single members, one variable missing):
  CIESM, IITM-ESM, MCM-UA-1-0, NESM3, NorESM2-LM, UKESM1-1-LL, MIROC-ES2H, TaiESM1, CESM2-FV2 (5-6 per scenario)
- [x] gaps between files: the preprocessing now checks the month coverage per member by the file names
  (`full_period`); the inventory's `local` still counts members with gaps as usable
- [ ] files that cannot be downloaded (retry or write to the data centres): MIROC6 ssp126 (~21 members, 107 datasets;
  esgf-data02.diasjp.net answers 404), MIROC-ES2H (404), CNRM-ESM2-1 (esg1.umr-cnrm.fr unreachable since 2026-11-16)
- [ ] CEDA index node (esgf.ceda.ac.uk) fails since 2026-11-21 (503 / TLS handshake cut, dap.ceda.ac.uk as well); the
  last inventory is based on DKRZ + ORNL only. Index node searches have no pause like the data servers: each
  dataset loses ~1 min to retries in a download run (option: leave `ceda` out of `download.index_nodes`)
- [ ] a download run with an old plan re-searches every dataset of that plan (a final rerun with the 2026-11-19 plan
  checked existing files for >24 h) - always run the inventory first; option: skip datasets already complete locally
- [ ] preprocessing reads the `ua_plev850_lat10-80` subsets
- [ ] jet speed / jet latitude index from `ua` at 850 hPa
- [ ] export step for the cmip6-api stores in `wind_db` (same 1.5° grid and chunking, new store name, then switch the
  s3 link); **decision**: which variables and scenarios the API serves
- [ ] add `requests` to `pixi.toml` (now only a transitive dependency); possible now that no download is running

## 4. Other regions and predictors (code in place, data later)
- [ ] (you) download an SST reconstruction (e.g. DCENT); then add it to `obs_sources`, `paths.inputs` and the PDO config
- [ ] (you) download CMIP6 tos (add to `download.variables`, rerun the inventory)
- [ ] (you) PDO-from-tas test run: `--config south_america_pdo.yaml --config pdo_from_tas.yaml` (preprocess tas_np, then
  the steps); hindcast skill needs a longer record than ERA5 (waits for DCENT or similar)
- [ ] first real PDO run with tos: check the regridding of curvilinear ocean grids (only tested on synthetic data)

## Smaller points
- [ ] `obs_target` hash uses the input path, not the content (declined for now)

## Done since the last commit (e065c9e)
- [x] index files: `units: 1`, long_name, method, mode, normal period (were the psl attributes)
- [x] `--cov` removed from `addopts` in `setup.cfg` (plain `pytest` works)
- [x] download: `ua` via HTTP + local cut when OPeNDAP fails; progress line every 200 files; no retries of dead
  connections; unreachable servers paused (`skip_host_after`, `skip_host_minutes`) and their files deferred and
  retried after the pause (`deferred_rounds`); `min_request_interval` per server
- [x] download: copies from all index nodes merged (CEDA lists EC-Earth3 only on esg-dn1.nsc.liu.se; ORNL records
  were dropped by the version check) - the EC-Earth3 files came from ORNL and ALCF
