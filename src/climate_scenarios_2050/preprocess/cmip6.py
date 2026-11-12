"""
CMIP6 preprocessing: original monthly files -> seasonal/annual means on a common grid.

For every model and member with both a historical and a scenario run:
1. historical (until Dec 2014) and scenario (from Jan 2015) monthly data are joined, so seasons
   crossing the boundary (DJFM 2014) are complete
2. members with gaps inside their covered period are dropped (members starting later, e.g.
   EC-Earth3 r101+ from 1970, are kept with NaN before their start)
3. seasonal/annual means are computed on the native grid (month lengths of the model's own
   calendar, only complete seasons), restricted to the target domain plus a margin
4. the means are regridded (xesmf, weights cached per source grid) to the regular global grid,
   cut to the target domain

Output: one zarr per model, `<cmip6_processed>/<variable>_<aggregation>/<hash>/<scenario>/<model>.zarr`
with dims (member, year, lat, lon); member names `<model id><variant>` (ids from configs/model_ids.yaml).
"""

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import xesmf as xe
from joblib import Parallel, delayed

from climate_scenarios_2050.core.config import config_hash, load_model_ids
from climate_scenarios_2050.core.io import ensure_writable, write_yaml
from climate_scenarios_2050.core.timeconv import seasonal_mean

log = logging.getLogger(__name__)

VARIANT_PATTERN = re.compile(r"_(r\d+i\d+p\d+f\d+)_")
HIST_END = 2014
DOMAIN_MARGIN = 5.0  # degrees added around the target domain before regridding
GRID_TOLERANCE = 0.01  # degrees, largest accepted coordinate difference between files of a member


#---------------grids--------------#
def target_grid(spacing: float, domain: dict = None) -> xr.Dataset:
    """regular global grid with cell centres at +-spacing/2 (lon -180 ... 180), optionally cut to a domain"""
    lat = np.arange(-90 + spacing / 2, 90, spacing)
    lon = np.arange(-180 + spacing / 2, 180, spacing)
    grid = xr.Dataset(coords={"lat": ("lat", lat), "lon": ("lon", lon)})
    if domain is not None:
        grid = grid.sel(lat=slice(*domain["lat"]), lon=slice(*domain["lon"]))
    grid["lat"].attrs = {"units": "degrees_north", "standard_name": "latitude"}
    grid["lon"].attrs = {"units": "degrees_east", "standard_name": "longitude"}
    return grid


def add_cell_bounds(grid: xr.Dataset) -> xr.Dataset:
    """lat_b/lon_b for conservative regridding of a regular grid"""
    out = grid.copy()
    for dim in ["lat", "lon"]:
        centres = grid[dim].values
        edges = np.concatenate([[centres[0] - (centres[1] - centres[0]) / 2], (centres[:-1] + centres[1:]) / 2, [centres[-1] + (centres[-1] - centres[-2]) / 2]])
        if dim == "lat":
            edges = np.clip(edges, -90, 90)
        out = out.assign_coords({f"{dim}_b": (f"{dim}_b", edges)})
    return out


def standardize_coords(ds: xr.Dataset) -> xr.Dataset:
    """rename horizontal coordinates to lat/lon, longitudes to -180 ... 180, ascending order (1D grids)"""
    rename = {}
    for name in ds.coords:
        low = name.lower()
        if low in ["latitude", "nav_lat"]:
            rename[name] = "lat"
        elif low in ["longitude", "nav_lon"]:
            rename[name] = "lon"
    ds = ds.rename(rename)
    if ds["lon"].ndim == 1:
        ds = ds.assign_coords(lon=(ds["lon"] + 180) % 360 - 180).sortby("lon")
        ds = ds.sortby("lat")
    else:
        ds = ds.assign_coords(lon=(ds["lon"] + 180) % 360 - 180)
    return ds


def cut_native(ds: xr.Dataset, domain: dict, margin: float = DOMAIN_MARGIN) -> xr.Dataset:
    """restrict a (1D) native grid to the domain plus a margin"""
    lat0, lat1 = domain["lat"][0] - margin, domain["lat"][1] + margin
    lon0, lon1 = domain["lon"][0] - margin, domain["lon"][1] + margin
    if ds["lon"].ndim == 1:
        return ds.sel(lat=slice(lat0, lat1), lon=slice(lon0, lon1))
    mask = (ds["lat"] >= lat0) & (ds["lat"] <= lat1) & (ds["lon"] >= lon0) & (ds["lon"] <= lon1)
    return ds.where(mask.compute(), drop=True)


#---------------file discovery--------------#
def discover_members(cmip6_root, variable: str, scenario: str) -> dict:
    """
    {model: {variant: {"historical": [files], scenario: [files]}}} for all members with
    both historical and scenario files
    """
    cmip6_root = Path(cmip6_root)
    found = {}
    for exp in ["historical", scenario]:
        for fl in sorted(cmip6_root.glob(f"{exp}/*/{variable}/{variable}_Amon_*_{exp}_*.nc")):
            if fl.stat().st_size == 0:
                continue
            model = fl.parent.parent.name
            match = VARIANT_PATTERN.search(fl.name)
            if match is None:
                continue
            found.setdefault(model, {}).setdefault(match.group(1), {}).setdefault(exp, []).append(fl)
    members = {}
    for model, variants in found.items():
        both = {var: files for var, files in variants.items() if "historical" in files and scenario in files}
        if both:
            members[model] = dict(sorted(both.items()))
    return dict(sorted(members.items()))


def _join_time(parts):
    """
    concatenate along time on the grid of the first part: horizontal coordinates of a member's files
    can differ by rounding errors, which an outer join would turn into duplicated rows of NaNs
    """
    for part in parts[1:]:
        for name in ["lat", "lon"]:
            if part[name].shape != parts[0][name].shape or np.abs(part[name].values - parts[0][name].values).max() > GRID_TOLERANCE:
                raise ValueError(f"{name} differs between the files of a member")
    return xr.concat(parts, dim="time", data_vars="minimal", coords="minimal", compat="override", join="override")


def _open_runs(files, variable, years):
    """open and join the files of one experiment, restricted to calendar years `years` (inclusive)"""
    parts = []
    for fl in files:
        ds = xr.open_dataset(fl, use_cftime=True, decode_timedelta=False)
        # FGOALS/EC-Earth3-Veg: some files run beyond 2100 or overlap the other experiment
        year = ds["time"].dt.year
        keep = ((year >= years[0]) & (year <= years[1])).values
        if keep.any():
            parts.append(standardize_coords(ds[[variable]].isel(time=keep)))
    if not parts:
        return None
    return _join_time(parts).sortby("time")


def load_member_monthly(files: dict, variable: str, scenario: str, period, domain: dict):
    """
    joined monthly data (historical until 2014, scenario from 2015) of one member, cut to the
    domain; returns (DataArray, info) or (None, reason)
    """
    hist = _open_runs(files["historical"], variable, (period[0], HIST_END))
    scen = _open_runs(files[scenario], variable, (HIST_END + 1, period[1]))
    if hist is None or scen is None:
        return None, "no data in period"
    ds = _join_time([hist, scen])
    # remove duplicated time steps (keep the first occurrence)
    _, first = np.unique(ds["time"].values, return_index=True)
    ds = ds.isel(time=np.sort(first))

    # check that the monthly series has no gaps between its first and last month
    time = ds["time"]
    months_since_start = (time.dt.year - time.dt.year[0]) * 12 + (time.dt.month - time.dt.month[0])
    if not np.array_equal(months_since_start.values, np.arange(time.size)):
        return None, "gaps in the monthly series"

    da = cut_native(ds, domain)[variable].load()
    if da.isnull().all(dim=[d for d in da.dims if d != "time"]).any():
        return None, "time steps without any data"
    info = {
        "first_month": str(time.values[0])[:7],
        "last_month": str(time.values[-1])[:7],
        "calendar": time.values[0].calendar,
    }
    return da, info


#---------------regridding--------------#
def grid_key(da: xr.DataArray) -> str:
    return config_hash({"lat": np.round(da["lat"].values, 4).tolist(), "lon": np.round(da["lon"].values, 4).tolist()})


def get_regridder(src: xr.DataArray, tgt: xr.Dataset, method: str, weights_file: Path):
    src_grid = src.isel(time=0, drop=True).to_dataset(name="v") if "time" in src.dims else src.to_dataset(name="v")
    if method == "conservative":
        src_grid = add_cell_bounds(src_grid)
        tgt = add_cell_bounds(tgt)
    if weights_file.exists():
        return xe.Regridder(src_grid, tgt, method, weights=str(weights_file), ignore_degenerate=True, unmapped_to_nan=True)
    # ignore_degenerate: cells collapsing at the poles of grids with a pole row
    # unmapped_to_nan: target cells outside the source grid (north of a Gaussian grid's last row) would otherwise be 0
    regridder = xe.Regridder(src_grid, tgt, method, ignore_degenerate=True, unmapped_to_nan=True)
    weights_file.parent.mkdir(parents=True, exist_ok=True)
    regridder.to_netcdf(str(weights_file))
    return regridder


#---------------main routine--------------#
def field_store(cfg, variable: str, aggregation: str) -> Path:
    """folder of the regridded archive of one variable/aggregation (hash over all relevant settings)"""
    fcfg = cfg["preprocess"]["fields"][variable]
    record = {
        "variable": variable,
        "aggregation": aggregation,
        "months": fcfg["aggregations"][aggregation],
        "method": fcfg["method"],
        "domain": fcfg["domain"],
        "grid_spacing": cfg["preprocess"]["grid_spacing"],
        "period": cfg["cmip6"]["period"],
    }
    return Path(cfg["paths"]["outputs"]["cmip6_processed"]) / f"{variable}_{aggregation}" / config_hash(record), record


def process_model(cfg, model: str, variants: dict, variable: str, scenario: str, stores: dict):
    """process all members of one model and write one zarr per aggregation"""
    fcfg = cfg["preprocess"]["fields"][variable]
    model_ids = load_model_ids()
    if model not in model_ids:
        raise KeyError(f"{model} has no id in configs/model_ids.yaml - append it there")
    mid = model_ids[model]
    out_files = {agg: stores[agg] / scenario / f"{model}.zarr" for agg in fcfg["aggregations"]}
    todo = [agg for agg, fl in out_files.items() if cfg["preprocess"]["overwrite"] or not fl.exists()]
    if not todo:
        return {"model": model, "status": "exists"}

    tgt = target_grid(cfg["preprocess"]["grid_spacing"], fcfg["domain"])
    weights_dir = Path(cfg["paths"]["outputs"]["cmip6_processed"]) / "regrid_weights"
    regridders = {}
    results = {agg: [] for agg in todo}
    report = []
    for variant, files in variants.items():
        try:
            da, info = load_member_monthly(files, variable, scenario, cfg["cmip6"]["period"], fcfg["domain"])
        except Exception as err:  # e.g. a corrupt file: drop the member, keep the run going
            da, info = None, f"could not read: {err}"
        if da is None:
            report.append({"model": model, "variant": variant, "kept": False, "reason": info})
            log.info(f"{model} {variant}: dropped ({info})")
            continue
        key = grid_key(da)
        if key not in regridders:
            weights_file = weights_dir / f"{model}_{key}_{fcfg['method']}_{config_hash([fcfg['domain'], cfg['preprocess']['grid_spacing']])}.nc"
            regridders[key] = get_regridder(da, tgt, fcfg["method"], weights_file)
        for agg in todo:
            seas = seasonal_mean(da, fcfg["aggregations"][agg])
            seas = regridders[key](seas, keep_attrs=True).astype("float32")
            info[f"nan_fraction_{agg}"] = round(float(seas.isnull().all("year").mean()), 4)
            if bool((seas == 0).any()):
                raise ValueError(f"{model} {variant}: exact zeros in the regridded {variable} field")
            seas = seas.expand_dims(member=[f"{mid}{variant}"])
            results[agg].append(seas)
        report.append({"model": model, "variant": variant, "kept": True, **info})

    kept = [r for r in report if r["kept"]]
    if not kept:
        return {"model": model, "status": "no valid members", "report": report}

    for agg in todo:
        data = xr.concat(results[agg], dim="member", join="outer")
        data = data.assign_coords(
            model_id=("member", [mid] * data.sizes["member"]),
            model=("member", [model] * data.sizes["member"]),
            variant=("member", [r["variant"] for r in kept]),
            calendar=("member", [r["calendar"] for r in kept]),
            first_month=("member", [r["first_month"] for r in kept]),
            last_month=("member", [r["last_month"] for r in kept]),
        )
        ds = data.to_dataset(name=variable)
        ds.attrs = {"model": model, "scenario": scenario, "aggregation": agg, "months": str(fcfg["aggregations"][agg]), "regrid_method": fcfg["method"]}
        path = ensure_writable(out_files[agg], cfg, overwrite=cfg["preprocess"]["overwrite"])
        ds.chunk({"member": 1, "year": -1, "lat": -1, "lon": -1}).to_zarr(path, mode="w")
    return {"model": model, "status": "written", "report": report}


def _process_model_safe(cfg, model, variants, variable, scenario, stores):
    """a failing model is reported at the end instead of stopping the other models"""
    try:
        return process_model(cfg, model, variants, variable, scenario, stores)
    except Exception as err:
        return {"model": model, "status": f"FAILED: {err}"}


def run(cfg, variables=None, scenarios=None):
    """preprocess all models of the given variables/scenarios (defaults: all configured)"""
    variables = variables or list(cfg["preprocess"]["fields"])
    scenarios = scenarios or cfg["cmip6"]["scenarios"]
    for variable in variables:
        fcfg = cfg["preprocess"]["fields"][variable]
        stores = {}
        for agg in fcfg["aggregations"]:
            store, record = field_store(cfg, variable, agg)
            if not (store / "config.yaml").exists():
                write_yaml(record, store / "config.yaml", cfg)
            stores[agg] = store
        for scenario in scenarios:
            members = discover_members(cfg["paths"]["inputs"]["cmip6"], variable, scenario)
            if not members:
                log.info(f"no {variable} data for {scenario}")
                continue
            log.info(f"{variable} {scenario}: {len(members)} models, {sum(len(v) for v in members.values())} members")
            results = Parallel(n_jobs=cfg["resources"]["n_jobs"])(
                delayed(_process_model_safe)(cfg, model, variants, variable, scenario, stores) for model, variants in members.items()
            )
            rows = [row for res in results for row in res.get("report", [])]
            if rows:
                report_file = stores[list(stores)[0]] / f"report_{scenario}_{pd.Timestamp.now():%Y%m%d%H%M%S}.csv"
                pd.DataFrame(rows).to_csv(ensure_writable(report_file, cfg), index=False)
            for res in results:
                log.info(f"{variable} {scenario} {res['model']}: {res['status']}")


def _open_model(path) -> xr.Dataset:
    """
    one model's archive, with its member coordinates in memory and without the on-disk encoding: a
    string width kept in the encoding (<U10) would cut longer names of other models when the joined
    data is written (m12 -> m1, MPI-ESM1-2-HR -> MPI-ESM1-2).
    Scalar coordinates that only some models have (e.g. `height` of sfcWind) are dropped.
    """
    ds = xr.open_zarr(path)
    ds = ds.drop_vars([name for name in ds.coords if ds[name].ndim == 0])
    ds = ds.assign_coords({name: ds[name].load() for name in ds.coords if name not in ds.dims})
    return ds.reset_encoding()


def open_field(cfg, variable: str, aggregation: str, scenario: str) -> xr.Dataset:
    """all models of one variable/aggregation/scenario as one dataset (member, year, lat, lon)"""
    store, _ = field_store(cfg, variable, aggregation)
    files = sorted((store / scenario).glob("*.zarr"))
    if not files:
        raise FileNotFoundError(f"no preprocessed {variable} {aggregation} {scenario} in {store}")
    return xr.concat([_open_model(fl) for fl in files], dim="member", join="outer")
