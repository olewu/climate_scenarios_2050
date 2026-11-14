"""
Step 7: extend the observed index with new ERA5 winters, using what has been saved (no refitting):
the pattern, its standardization, the ERA5 normal-period climatology and the HadSLP/ERA5 harmonization.
Existing years are left untouched, so everything fitted or ranked with them stays valid.

The new monthly SLP files are processed like all other fields (complete seasons only, same regridding).
Seasons already in the index are recomputed as a check: the update stops without writing anything if they
differ by more than `update.tolerance` (e.g. a changed pattern, or reprocessed input data).
Previous versions are kept as `obs_<dataset>_until<last season>.nc`.

Only available for EOF indices with a common pattern. The CMIP6 side does not change.
"""

import logging
import shutil
from datetime import datetime

import numpy as np
import xarray as xr

from climate_scenarios_2050.core import domain as dom
from climate_scenarios_2050.core.config import index_cfg, index_names, predictor_field
from climate_scenarios_2050.core.io import write_netcdf
from climate_scenarios_2050.core.timeconv import seasonal_mean
from climate_scenarios_2050.preprocess import masks, obs
from climate_scenarios_2050.preprocess.cmip6 import get_regridder, grid_key, target_grid
from climate_scenarios_2050.preprocess.indices import eof
from climate_scenarios_2050.preprocess.indices.compute import index_output

log = logging.getLogger(__name__)


def new_seasons_index(cfg, files=None) -> xr.DataArray:
    """index of all complete seasons in the given monthly files, from the saved pattern and climatology"""
    pred = cfg["predictor"]
    if pred["method"] != "eof" or pred["pattern_source"] == "own":
        raise NotImplementedError("the update needs an EOF index with a common pattern")
    variable = pred["variable"]
    field_name = predictor_field(cfg)
    fcfg = cfg["preprocess"]["fields"][field_name]
    out = index_output(cfg)
    with xr.open_dataset(out.path(f"pattern_{pred['pattern_source']}.nc")) as ds:
        pattern = ds.load()
    field_out = obs.obs_field_output(cfg, field_name)
    with xr.open_dataset(field_out.path(f"ERA5_{field_name}_{pred['aggregation']}.nc")) as ds:
        history = ds[variable].load()
        if pred.get("remove_global_mean"):
            history = history - ds[f"{variable}_global_mean"].load()
    domain = pred["domain"]
    climatology = eof.normal_period_climatology(dom.select(history, domain), pred["normal_period"])

    months = fcfg["aggregations"][pred["aggregation"]]
    loaded = obs.open_obs_monthly(cfg, "ERA5", variable, fcfg["domain"], files=files, with_global_mean=fcfg.get("global_mean") if pred.get("remove_global_mean") else False)
    monthly, gm = loaded if pred.get("remove_global_mean") else (loaded, None)
    tgt = target_grid(cfg["preprocess"]["grid_spacing"], fcfg["domain"])
    weights_file = field_out.dir / "regrid_weights" / f"ERA5_{grid_key(monthly)}_{fcfg['method']}.nc"
    regridder = get_regridder(monthly, tgt, fcfg["method"], weights_file)
    seasons = regridder(seasonal_mean(monthly, months), keep_attrs=True).astype("float32")
    if gm is not None:
        seasons = seasons - seasonal_mean(gm, months).reindex(year=seasons.year)
    field = masks.apply(cfg, dom.select(seasons, domain), pred.get("mask"))
    return eof.project(field, pattern, pred["mode"], climatology).compute()


def _replace(cfg, path, updated: xr.Dataset, last_old: int):
    backup = path.with_name(f"{path.stem}_until{last_old}{path.suffix}")
    if not backup.exists():
        shutil.copy(path, backup)
    path.unlink()
    write_netcdf(updated, path, cfg)
    log.info(f"updated {path.name} (previous version: {backup.name})")


def run(cfg, files=None):
    """every configured index"""
    added = {}
    for name in index_names(cfg):
        log.info(f"update of index {name}")
        added[name] = run_index(index_cfg(cfg, name), files)
    return added


def run_index(cfg, files=None):
    out = index_output(cfg)
    tolerance = cfg["update"]["tolerance"]
    with xr.open_dataset(out.path("obs_ERA5.nc")) as ds:
        era5_old = ds.load()
    merged_old = None
    if cfg["obs"].get("merge"):
        with xr.open_dataset(out.path("obs_merged.nc")) as ds:
            merged_old = ds.load()
    new = new_seasons_index(cfg, files)
    last_old = int(era5_old.year.max())

    common = np.intersect1d(new.year.values, era5_old.year.values)
    if len(common):
        diff = float(np.abs(new.sel(year=common) - era5_old["index"].sel(year=common)).max())
        log.info(f"check on {len(common)} existing seasons ({common[0]}-{common[-1]}): largest difference {diff:.2e}")
        if diff > tolerance:
            raise RuntimeError(f"existing seasons differ by {diff:.3f} (tolerance {tolerance}); nothing has been written")
    added = new.sel(year=new.year > last_old)
    if added.sizes["year"] == 0:
        log.info(f"no new complete season after {last_old}")
        return []
    expected = np.arange(last_old + 1, last_old + 1 + added.sizes["year"])
    if not np.array_equal(added.year.values, expected):
        raise RuntimeError(f"new seasons {added.year.values} do not continue the index, which ends in {last_old}")

    stamp = f"{datetime.now():%Y-%m-%d %H:%M}: appended seasons {int(added.year.min())}-{int(added.year.max())}"
    era5_new = xr.concat([era5_old["index"], added.drop_vars([c for c in added.coords if c != "year"])], dim="year").to_dataset(name="index")
    era5_new.attrs = {**era5_old.attrs, "update_history": "; ".join(filter(None, [era5_old.attrs.get("update_history"), stamp]))}

    if merged_old is not None:
        attrs = merged_old.attrs
        harmonized = (added - attrs["late_mean"]) * attrs["early_std"] / attrs["late_std"] + attrs["early_mean"]
        source = xr.full_like(harmonized.astype(object), cfg["obs"]["merge"]["datasets"][1]).astype(str)
        merged_new = xr.Dataset({
            "index": xr.concat([merged_old["index"], harmonized.drop_vars([c for c in harmonized.coords if c != "year"])], dim="year"),
            "source": xr.concat([merged_old["source"].astype(str), source.drop_vars([c for c in source.coords if c != "year"])], dim="year"),
        })
        merged_new.attrs = {**attrs, "update_history": "; ".join(filter(None, [attrs.get("update_history"), stamp]))}
    _replace(cfg, out.path("obs_ERA5.nc"), era5_new, last_old)
    if merged_old is not None:
        _replace(cfg, out.path("obs_merged.nc"), merged_new, last_old)
    log.info(stamp)
    return list(added.year.values)
