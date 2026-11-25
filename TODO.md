# To-do

Checked items stay in the list until they are removed after agreement. `(you)` = runs/decisions on your side.

## Runs on your side
- [ ] (you) `pixi run preprocess` with the current archive, then the steps from `split` on (everything recomputes once)
- [ ] (you) jet run: the steps from `split` on with `--config configs/experiments/jet.yaml` (preprocessing done
  2026-11-23)
- [ ] (you) first NAO + EA run (`--config configs/experiments/nao_ea.yaml`); only the observed side is checked on real
  data so far (NAO 48 %, EA 15 % of the ERA5 DJFM SLP variance)
- [ ] (you) tas/pr with their own forced part (`targets.<name>.forced: cmip6`, new 2026-11-23): rerun `split`,
  `linear-model`, `evaluate-target` (and `forecast`) - only tas/pr get new hashes, wind/energy outputs are reused
- [ ] (you) observations-only methods M0c/M0t, the test against M1 and M0's spread from its own errors (new
  2026-11-24): rerun `evaluate-target`
  for every experiment (new hashes, nothing upstream changes; includes the surrogate test again), then
  `pixi run gallery --config ...` per experiment. The forecast step has them too (new forecast hash)
- [ ] (you) target evaluation with the significance test (`pixi run evaluate-target`, ~1.7 h per scenario/target;
  with tas and pr 8 cases)
- [ ] (you) reconstruct the April 2026 hindcast
- [ ] regenerate the figures with the new plot code (evaluate-index, linear-model, forecast) - comes with the rerun
- [ ] (you) decide which old runs/data to remove before the final experiments (`data/CMIP6_processed` 342 GB and
  `data/CMIP6_hindcast` 59 GB are only used by `legacy/`; old inventories in `cmip6_inventory/` 20261113, 20261114,
  20261114-002725, 20261122 (no plan))

## CMIP6 archive
State after the inventory `20261122-163805`: 18-61 models per variable/scenario, 1-5 members per variable/scenario
still missing except ssp126 (~30).
- [ ] files that cannot be downloaded (retry or write to the data centres): MIROC6 ssp126 (~21 members, 107 datasets;
  esgf-data02.diasjp.net answers 404), MIROC-ES2H (404), CNRM-ESM2-1 (esg1.umr-cnrm.fr unreachable since 2026-11-16)
- [ ] CEDA index node (esgf.ceda.ac.uk) fails since 2026-11-21 (503 / TLS handshake cut, dap.ceda.ac.uk as well); the
  last inventory is based on DKRZ + ORNL only. Index node searches have no pause like the data servers: each
  dataset loses ~1 min to retries in a download run (option: leave `ceda` out of `download.index_nodes`)
- [ ] a download run with an old plan re-searches every dataset of that plan (a final rerun with the 2026-11-19 plan
  checked existing files for >24 h) - always run the inventory first; option: skip datasets already complete locally
- [ ] export step for the cmip6-api stores in `wind_db` (same 1.5° grid and chunking, new store name, then switch the
  s3 link); **decision**: which variables and scenarios the API serves
- [ ] add `requests` to `pixi.toml` (now only a transitive dependency); possible now that no download is running
- [ ] `analogues/generate.py:init_years` only looks at the observation years: with observations starting before
  the models (20CRv3 from 1806, CMIP6 from 1850) the first inits have no complete matching window in the models
  (`ValueError: init 1875: matching window not covered`). Fix: start at max(first obs year, first model year) +
  window - 1 (also in `evaluate/index.py`). Not done while evaluate runs were active; jet works around it with
  `analogues.first_init: 1919`, which can stay (same inits)

## Predictors
- [ ] option: screen/weight models by their climatological jet latitude/speed vs ERA5 (`raw_climatology`), separate
  from the matching
- [x] add DCENT (`data/DCENT`: ensemble mean, 5 deg, 1850-2025, anomalies) to `obs_sources` / `paths.inputs`
  (reader for the anomaly file, blended `temperature` for tas-based indices); observations of the AMV (North
  Atlantic coverage > 90 % from 1890 except ~1900 and the war years)
- [ ] (you) download an infilled SST product for the PDO (e.g. ERSSTv5, 2 deg from 1854; or COBE-SST2): DCENT is too
  sparse in the North Pacific for an EOF before ~1950; optionally also as a second AMV dataset
- [x] AMV index (`predictor.method: amv`, `configs/experiments/amv.yaml`): North Atlantic mean tas, forced part
  removed in the split (MMEM), DCENT as observations; variant with the global mean removed
- [ ] (you) AMV run: `pixi run preprocess --config configs/experiments/amv.yaml` (new CMIP6 field `tas_na`, all
  models/scenarios), then the steps from `split` on with the same config
- [ ] (you) PDO-from-tas test run: `--config south_america_pdo.yaml --config pdo_from_tas.yaml` (preprocess tas_np, then
  the steps); hindcast skill needs a merge of the infilled SST product with ERA5 (ERA5 alone is too short for a
  70-year matching window)

## Extras
- [x] make a figure comparing scenario and model uncertainty with internal variability as in the legacy code
  (`pixi run partition`, `partition.html`; done for the default config)
- [ ] (you) `pixi run partition --config ...` for the other experiments (indices EA, jet, AMV; the target regions are
  shared between configs with the same targets), then `pixi run gallery --config ...`
- [ ] option: scenario uncertainty from the models common to all scenarios (ssp119 has 17 models)
- [ ] make a conceptual figure that illustrates the analog selection process using a single index (NAO) and a simple matching (e.g. CORR) and how a forecast is derived from this

## Comparison pages (gallery)
- [ ] (you) rerun `pixi run evaluate-index --config ...` for every experiment (adds only `continuous_crps.nc`, ~20 s
  per index and scenario), then `pixi run gallery --config ...`: CRPS per lead time on the index page
- [x] index page (`index_skill.html`) and linear-model page (`linear_model.html`) in the same way as `target.html`
- [x] (you) rerun `pixi run gallery --config ...` after new target evaluations (e.g. NAO+EA energy got its
  significance test after the first export)

## Smaller points
- [ ] `obs_target` hash uses the input path, not the content (declined for now)
